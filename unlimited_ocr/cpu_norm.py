"""Runtime-selected RMSNorm, with a PyTorch fallback when native code is unavailable."""
import ctypes
from functools import lru_cache
from pathlib import Path
import warnings

from unlimited_ocr.cpu_isa import POLICY

import torch

@lru_cache(None)
def _kernel(isa):
    if isa == "generic":
        return None
    try:
        library = ctypes.CDLL(str(Path(__file__).with_name("native.so")))
    except OSError as error:
        warnings.warn(f"Native RMSNorm unavailable; using PyTorch: {error}")
        return None
    function = getattr(library, f"rmsnorm_{isa}")
    function.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 2 + [ctypes.c_float]
    function.restype = None
    return function


def rmsnorm(x, weight, eps):
    if x.device.type != "cpu" or weight.device.type != "cpu" or x.dtype != torch.float32 or weight.dtype != torch.float32:
        raise ValueError("CPU RMSNorm requires CPU FP32 tensors")
    if weight.ndim != 1 or x.shape[-1] != weight.numel():
        raise ValueError("RMSNorm weight shape mismatch")
    function = _kernel(POLICY["selected"])
    if function is None:
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * weight
    x, weight = x.contiguous(), weight.contiguous()
    out = torch.empty_like(x)
    function(x.data_ptr(), weight.data_ptr(), out.data_ptr(),
             x.numel() // weight.numel(), weight.numel(), eps)
    return out


def backend_name():
    return POLICY["selected"] if _kernel(POLICY["selected"]) is not None else "torch"


def install_norm(model):
    from upstream._sparse_base_impl import DeepseekV2RMSNorm
    from types import MethodType

    def forward(self, hidden_states):
        return rmsnorm(hidden_states, self.weight, self.variance_epsilon)

    count = 0
    for module in model.modules():
        if isinstance(module, DeepseekV2RMSNorm):
            module.forward = MethodType(forward, module)
            count += 1
    return count
