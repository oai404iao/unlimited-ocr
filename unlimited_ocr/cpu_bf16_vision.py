"""Optional native BF16 SAM experiment; unsupported CPUs retain the current path."""
from contextlib import ExitStack
from functools import wraps
import json
from pathlib import Path
import warnings

from unlimited_ocr.cpu_isa import POLICY
import torch
from torch import nn
from torch.ao.nn.quantized.dynamic import Linear as Int8Linear
from safetensors import safe_open


def install_bf16_sam(model, directory="models/unlimited-ocr-2"):
    if not POLICY["bf16"]:
        warnings.warn("Native BF16 unavailable/disabled; retaining existing SAM path")
        return False
    sam = model.model.sam_model
    if getattr(sam, "_cpu_bf16", False):
        return True
    root = Path(directory)
    mapping = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    # Restore original checkpoint linears, never expand already-quantized INT8
    # weights and mislabel them as a native-precision BF16 baseline.
    with ExitStack() as stack:
        shards = {}

        def tensor(key):
            name = mapping[key]
            if name not in shards:
                shards[name] = stack.enter_context(safe_open(root / name, framework="pt", device="cpu"))
            return shards[name].get_tensor(key).to(torch.bfloat16)

        for name, module in list(sam.named_modules()):
            if not isinstance(module, (nn.Linear, Int8Linear)):
                continue
            prefix = f"model.sam_model.{name}."
            has_bias = prefix + "bias" in mapping
            linear = nn.Linear(module.in_features, module.out_features, has_bias,
                               device="meta", dtype=torch.bfloat16)
            linear.weight = nn.Parameter(tensor(prefix + "weight"), requires_grad=False)
            if has_bias:
                linear.bias = nn.Parameter(tensor(prefix + "bias"), requires_grad=False)
            parent, _, child = name.rpartition(".")
            setattr(sam.get_submodule(parent) if parent else sam, child, linear.eval())
    original = sam.forward

    @wraps(original)
    def forward(x):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = original(x)
        # CLIP/projector/decoder remain on the established FP32-activation path.
        return output.float()

    sam.forward = forward
    sam._cpu_bf16 = True
    return True
