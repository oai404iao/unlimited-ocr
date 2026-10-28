"""FP32 CPU sparse decode with AVX2/AVX512 dispatch or PyTorch fallback.

Build native_attention.so with python -m tools.build_native, then call
install_native_attention after unlimited_ocr.cpu_attention.install_attention.
"""
import ctypes
import importlib
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
        library = ctypes.CDLL(str(Path(__file__).with_name('native_attention.so')))
    except OSError as error:
        warnings.warn(f"Native attention unavailable; using PyTorch SDPA: {error}")
        return None
    function = getattr(library, f"decode_attention_{isa}")
    function.argtypes = [ctypes.c_void_p] * 9 + [ctypes.c_int64] * 2 + [ctypes.c_float]
    function.restype = ctypes.c_int
    return function


@torch.no_grad()
def decode_native(q, k_all, v_all, topi, bias, valid, q_pos,
                  block_size, local_window, scale):
    tensors = (q, k_all, v_all, topi, bias, valid, q_pos)
    if any(t.device.type != 'cpu' for t in tensors):
        raise ValueError('native decode requires CPU tensors')
    if any(t.dtype != torch.float32 for t in (q, k_all, v_all, bias)):
        raise ValueError('native decode requires FP32 q/k/v/bias')
    if topi.dtype != torch.int64 or q_pos.dtype != torch.int64 or valid.dtype != torch.bool:
        raise ValueError('native decode requires int64 indices/positions and bool validity')
    if any(t.ndim != 4 for t in (q, k_all, v_all)) or topi.ndim != 3:
        raise ValueError('native decode tensor rank mismatch')
    B, H, L, D = q.shape
    BK, HK, S, DK = k_all.shape
    if (B != BK or D != DK or HK <= 0 or H <= 0 or H % HK or
            v_all.shape[:3] != (B, HK, S) or D <= 0 or v_all.shape[3] <= 0 or
            topi.shape[:2] != (B, L) or bias.shape != topi.shape or
            valid.shape != topi.shape or q_pos.shape != (B, L)):
        raise ValueError('native decode tensor shape mismatch')
    limit = torch.iinfo(torch.int64).max
    if (not isinstance(block_size, int) or not 0 < block_size <= limit
            or not isinstance(local_window, int) or not 0 <= local_window <= limit):
        raise ValueError('block size/window must fit positive/nonnegative int64')
    function = _kernel(POLICY["selected"])
    if function is None:
        from unlimited_ocr.cpu_attention import _decode_sdpa
        return _decode_sdpa(q, k_all, v_all, topi, bias, valid, q_pos,
                            block_size, local_window, scale)
    topi, bias, valid, q_pos = (t.contiguous() for t in (topi, bias, valid, q_pos))
    out = torch.empty((B, H, L, v_all.shape[3]), dtype=torch.float32)
    metadata = (ctypes.c_int64 * 20)(B, H, L, D, HK, S, v_all.shape[3], topi.shape[2],
                                    *q.stride(), *k_all.stride(), *v_all.stride())
    status = function(
        *(t.data_ptr() for t in (q, k_all, v_all, topi, bias, valid, q_pos, out)),
        metadata, block_size, local_window, scale)
    if status:
        raise RuntimeError('native decode workspace allocation failed')
    return out


def install_native_attention(modeling_module=None):
    """Opt-in replacement for decode only; call after install_attention."""
    if modeling_module is None:
        modeling_module = importlib.import_module('upstream.modeling_deepseekv2')
    modeling_module._decode_attend = decode_native


def backend_name():
    return POLICY["selected"] if _kernel(POLICY["selected"]) is not None else "torch_sdpa"
