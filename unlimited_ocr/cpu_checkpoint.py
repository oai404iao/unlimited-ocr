"""Offline INT8 v2: safetensors of ordinary tensors, runtime-only native packing."""
import gc
import json
import shutil
from pathlib import Path
import warnings

from unlimited_ocr.cpu_isa import POLICY
from unlimited_ocr.cpu_linear import CPUInt8Linear, native_available
import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn

MANIFEST_FILENAME = "quantized.json"
STATE_FILENAME = "quantized.safetensors"
FORMAT = "unlimited-ocr-dynamic-int8"
VERSION = 2
ASSETS = ("config.json", "generation_config.json", "tokenizer.json",
          "tokenizer_config.json", "special_tokens_map.json")


def is_quantized_checkpoint(directory):
    directory = Path(directory)
    return (directory / MANIFEST_FILENAME).exists() or (directory / STATE_FILENAME).exists()


def export_quantized_model(source, destination):
    """Write a new directory; publish the completion manifest only after all data."""
    from unlimited_ocr.cpu_runtime import load_model

    source, destination = Path(source), Path(destination)
    if POLICY["selected"] == "generic" or not native_available():
        raise ValueError("INT8 export requires AVX2/FMA and a built native FBGEMM library")
    if is_quantized_checkpoint(source):
        raise ValueError("Export requires original BF16 model data, not an INT8 checkpoint")
    if destination.exists():
        raise FileExistsError(destination)
    for name in ASSETS:
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    destination.mkdir(parents=True, exist_ok=False)
    model = load_model(source, quantized=True)
    state = model.state_dict()
    del model
    gc.collect()
    save_file(state, destination / STATE_FILENAME)
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
        raise ValueError("Unsupported offline INT8 format/version or PyTorch version; re-export from BF16")
    config = UnlimitedOCRConfig.from_pretrained(directory, local_files_only=True)
    config._attn_implementation = "eager"
    config.sparse_attention_config["dash_use_triton"] = False
    with init_empty_weights(include_buffers=False):
        model = UnlimitedOCRForCausalLM(config)
    accelerated = native_available()
    if not accelerated:
        warnings.warn("Native INT8 unavailable/disabled: offline INT8 storage with per-operation FP32 dequantization")
    with safe_open(directory / STATE_FILENAME, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        loaded = set()

        def tensor(key):
            if key not in keys:
                raise ValueError(f"Missing checkpoint tensor: {key}")
            loaded.add(key)
            return handle.get_tensor(key)

        for name, module in list(model.named_modules()):
            if not isinstance(module, nn.Linear) or ".indexer." in name or ".dash_router." in name:
                continue
            weight, scales = tensor(name + ".weight"), tensor(name + ".scales")
            if weight.shape != (module.out_features, module.in_features):
                raise ValueError(f"Invalid INT8 weight shape for {name}")
            bias = tensor(name + ".bias") if module.bias is not None else None
            replacement = CPUInt8Linear(weight, scales, bias)
            parent_name, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent_name) if parent_name else model, child, replacement)
        for name, module in model.named_modules():
            if isinstance(module, CPUInt8Linear):
                continue
            if isinstance(module, nn.Embedding):
                module.__class__ = CPUEmbedding
            for attr, parameter in list(module.named_parameters(recurse=False)):
                key = f"{name}.{attr}" if name else attr
                value = tensor(key)
                if value.shape != parameter.shape:
                    raise ValueError(f"Invalid parameter shape for {key}")
                setattr(module, attr, nn.Parameter(value, requires_grad=False))
            for attr, buffer in list(module.named_buffers(recurse=False)):
                if attr in module._non_persistent_buffers_set:
                    continue
                key = f"{name}.{attr}" if name else attr
                value = tensor(key)
                if value.shape != buffer.shape:
                    raise ValueError(f"Invalid buffer shape for {key}")
                setattr(module, attr, value)
        if keys - loaded:
            raise ValueError(f"Unexpected checkpoint tensors: {sorted(keys - loaded)}")
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError("Uninitialized meta parameters remain")
    model.requires_grad_(False)
    model._cpu_quantized_weights = accelerated
    model._cpu_weight_format = "int8" if accelerated else "int8_dequantize"
    return model.eval()
