"""Portable dynamic-INT8 state dictionaries, never pickled model/packed objects."""
import gc
import json
import shutil
from collections import OrderedDict
from pathlib import Path
import warnings

from unlimited_ocr.cpu_isa import POLICY
import torch
from accelerate import init_empty_weights
from torch import nn
from torch.ao.nn.quantized.dynamic import Linear as Int8Linear
from torch.ao.nn.quantized.modules.linear import LinearPackedParams

MANIFEST_FILENAME = "quantized.json"
STATE_FILENAME = "quantized.pt"
FORMAT = "unlimited-ocr-dynamic-int8"
VERSION = 1
ASSETS = ("config.json", "generation_config.json", "tokenizer.json",
          "tokenizer_config.json", "special_tokens_map.json")


def is_quantized_checkpoint(directory):
    """Also recognize incomplete exports so they cannot fall through to HF loading."""
    directory = Path(directory)
    return (directory / MANIFEST_FILENAME).exists() or (directory / STATE_FILENAME).exists()


class DequantizingLinear(nn.Module):
    """Keep INT8 storage on generic CPUs; dequantize only the current operation."""
    def __init__(self, weight, bias):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight.dequantize(), self.bias)


def export_quantized_model(source, destination):
    """Export to a new directory. The manifest is the last completion marker.

    An interrupted export is deliberately left incomplete; choose a new destination
    or explicitly remove that directory before retrying. Never overwrite user data.
    """
    from unlimited_ocr.cpu_runtime import load_model

    source, destination = Path(source), Path(destination)
    if POLICY["selected"] == "generic":
        raise ValueError("INT8 export requires AVX2/FMA and the FBGEMM backend")
    if is_quantized_checkpoint(source):
        raise ValueError("Export requires original BF16 model data, not an INT8 checkpoint")
    if destination.exists():
        raise FileExistsError(destination)
    for name in ASSETS:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    destination.mkdir(parents=True, exist_ok=False)
    torch.backends.quantized.engine = "fbgemm"
    model = load_model(source, quantized=True)
    if not model._cpu_quantized_weights:
        raise ValueError("Source model did not produce INT8 weights")
    # PyTorch's state_dict unpacks quantized Linear weights into ordinary quantized
    # tensors plus bias. Keep its _metadata, needed by Linear's versioned loader.
    state = model.state_dict()
    del model
    gc.collect()
    torch.save(state, destination / STATE_FILENAME)
    for name in ASSETS:
        shutil.copyfile(source / name, destination / name)
    manifest = dict(format=FORMAT, version=VERSION, torch_version=torch.__version__.split("+")[0])
    (destination / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


def load_quantized_model(directory):
    from unlimited_ocr.cpu_runtime import CPUEmbedding
    from upstream.modeling_unlimitedocr import UnlimitedOCRConfig, UnlimitedOCRForCausalLM

    directory = Path(directory)
    try:
        manifest = json.loads((directory / MANIFEST_FILENAME).read_text())
    except (OSError, ValueError) as error:
        raise ValueError("Missing or invalid offline INT8 completion manifest") from error
    if (not isinstance(manifest, dict) or manifest.get("format") != FORMAT
            or manifest.get("version") != VERSION
            or manifest.get("torch_version") != torch.__version__.split("+")[0]):
        raise ValueError("Unsupported offline INT8 format/version or PyTorch version")
    state = torch.load(directory / STATE_FILENAME, map_location="cpu", weights_only=True, mmap=True)
    metadata = getattr(state, "_metadata", None)
    if (not isinstance(state, OrderedDict) or not isinstance(metadata, dict)
            or any(not isinstance(value, dict) for value in metadata.values())):
        raise ValueError("Offline INT8 state_dict is missing serialization metadata")
    config = UnlimitedOCRConfig.from_pretrained(directory, local_files_only=True)
    config._attn_implementation = "eager"
    config.sparse_attention_config["dash_use_triton"] = False
    with init_empty_weights(include_buffers=False):
        model = UnlimitedOCRForCausalLM(config)
    generic = POLICY["selected"] == "generic"
    if generic:
        warnings.warn("AVX2 unavailable/disabled: offline INT8 storage with per-operation FP32 dequantization")
    else:
        torch.backends.quantized.engine = "fbgemm"
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Embedding):
            module.__class__ = CPUEmbedding
        if not isinstance(module, nn.Linear) or ".indexer." in name or ".dash_router." in name:
            continue
        prefix = name + "."
        packed = prefix + "_packed_params."
        keys = ("scale", "zero_point", "_packed_params.dtype", "_packed_params._packed_params")
        if any(prefix + key not in state for key in keys):
            raise ValueError(f"Missing INT8 state entries for {name}")
        if (metadata.get(name, {}).get("version") != Int8Linear._version
                or metadata.get(name + "._packed_params", {}).get("version") != LinearPackedParams._version):
            raise ValueError(f"Invalid INT8 serialization metadata for {name}")
        pair = state.get(packed + "_packed_params")
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(f"Missing INT8 weight/bias for {name}")
        weight, bias = pair
        if (not isinstance(weight, torch.Tensor) or weight.dtype != torch.qint8
                or weight.shape != (module.out_features, module.in_features)
                or state.get(packed + "dtype") != torch.qint8
                or (module.bias is None) != (bias is None)
                or bias is not None and (not isinstance(bias, torch.Tensor)
                                        or bias.dtype != torch.float32 or bias.shape != (module.out_features,))):
            raise ValueError(f"Invalid INT8 tensor layout for {name}")
        if generic:
            replacement = DequantizingLinear(weight, bias)
            for key in keys:
                del state[prefix + key]
            state[prefix + "weight"] = weight
            if bias is not None:
                state[prefix + "bias"] = bias
        else:
            # Avoid allocating and prepacking a full dummy matrix for every layer.
            replacement = Int8Linear(1, 1, bias is not None)
            replacement.in_features, replacement.out_features = module.in_features, module.out_features
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, replacement)
    model.load_state_dict(state, strict=True, assign=True)
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError("Uninitialized meta parameters remain")
    model.requires_grad_(False)
    model._cpu_quantized_weights = not generic
    model._cpu_weight_format = "int8_dequantize" if generic else "int8"
    return model.eval()
