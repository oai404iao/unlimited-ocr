"""Low-peak-memory loading: BF16 checkpoint -> one FP32 linear -> packed INT8.

FBGEMM's dynamic-int8 kernels dispatch to AVX2/VNNI; activations/attention use
FP32. Without AVX2 use BF16 storage/FP32 compute instead of FBGEMM. No full FP32 copy.
"""
import json
from pathlib import Path
import warnings
from unlimited_ocr.cpu_isa import POLICY

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from torch import nn
from torch.ao.nn.quantized.dynamic import Linear as Int8Linear


class CPUEmbedding(nn.Embedding):
    """Store embeddings in checkpoint precision; only expand the selected rows."""
    def forward(self, ids):
        return super().forward(ids).float()


class BF16StorageLinear(nn.Linear):
    """Unquantized memory-conscious reference for CPUs without BF16 arithmetic.

    Re-expand the original checkpoint weight for each operation. Deliberately
    no packed-weight cache: that would require a full FP32 model in memory.
    """
    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight.float(),
                                          None if self.bias is None else self.bias.float())


def configure_cpu(threads):
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    if POLICY["selected"] != "generic":
        torch.backends.quantized.engine = "fbgemm"


def pack_linear(weight, bias=None):
    weight = weight.float()
    scales = weight.abs().amax(dim=1).clamp_min(1e-8) / 127
    quantized = torch.quantize_per_channel(
        weight, scales, torch.zeros(weight.shape[0], dtype=torch.int64), 0, torch.qint8
    )
    packed = Int8Linear(weight.shape[1], weight.shape[0], bias is not None)
    packed.set_weight_bias(quantized, None if bias is None else bias.float())
    return packed


def load_model(directory="models/unlimited-ocr-2", quantized=True, bf16_storage=False):
    from upstream.modeling_unlimitedocr import UnlimitedOCRConfig, UnlimitedOCRForCausalLM
    directory = Path(directory)
    if quantized and bf16_storage:
        raise ValueError("Choose INT8 or BF16 reference storage, not both")
    from unlimited_ocr.cpu_checkpoint import is_quantized_checkpoint, load_quantized_model
    if is_quantized_checkpoint(directory):
        if not quantized or bf16_storage:
            raise ValueError("Offline INT8 checkpoints do not contain original reference weights")
        return load_quantized_model(directory)
    if quantized and POLICY["selected"] == "generic":
        warnings.warn("AVX2 unavailable/disabled: using BF16 weight storage and FP32 computation, not FBGEMM INT8")
        quantized, bf16_storage = False, True
    config = UnlimitedOCRConfig.from_pretrained(directory)
    config._attn_implementation = "eager"
    config.sparse_attention_config["dash_use_triton"] = False
    # Parameter allocation on meta; nonpersistent RoPE buffers still initialize on CPU.
    with init_empty_weights(include_buffers=False):
        model = UnlimitedOCRForCausalLM(config)
    mapping = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    from contextlib import ExitStack
    with ExitStack() as stack:
        shards = {name: stack.enter_context(safe_open(directory / name, framework="pt", device="cpu"))
                  for name in set(mapping.values())}
        loaded = set()

        def tensor(key):
            loaded.add(key)
            return shards[mapping[key]].get_tensor(key)

        for name, module in list(model.named_modules()):
            # Keep routing decisions in FP32; small indexer projections also expose
            # their parameter device to the upstream page-position bookkeeping.
            if not isinstance(module, nn.Linear) or not quantized or ".indexer." in name or ".dash_router." in name:
                continue
            prefix = name + "."
            weight = tensor(prefix + "weight")
            bias = tensor(prefix + "bias") if module.bias is not None else None
            parent_name, _, child = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child, pack_linear(weight, bias))
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                module.__class__ = CPUEmbedding
            if bf16_storage and isinstance(module, nn.Linear):
                module.__class__ = BF16StorageLinear
            for attr, param in list(module.named_parameters(recurse=False)):
                key = f"{name}.{attr}" if name else attr
                value = tensor(key)
                if not isinstance(module, (nn.Embedding, BF16StorageLinear)):
                    value = value.float()
                setattr(module, attr, nn.Parameter(value, requires_grad=False))
            for attr, buffer in list(module.named_buffers(recurse=False)):
                key = f"{name}.{attr}" if name else attr
                if key in mapping:
                    setattr(module, attr, tensor(key).to(buffer.dtype))
        unused = set(mapping) - loaded
        if unused:
            raise RuntimeError(f"Unloaded checkpoint tensors: {sorted(unused)}")
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError("Uninitialized meta parameters remain")
    model._cpu_quantized_weights = quantized
    model._cpu_weight_format = "int8" if quantized else "bf16_storage" if bf16_storage else "fp32"
    return model.eval()
