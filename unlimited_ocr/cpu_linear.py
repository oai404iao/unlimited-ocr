"""Plain INT8 tensor storage and torch.library FBGEMM inference.

Native packing is runtime-only and tied to the installed PyTorch/FBGEMM ABI;
rebuild linear.so after upgrading PyTorch. state_dict contains only ordinary
tensors. Weights are inference constants; load_state_dict rebuilds the handle.
"""
from functools import lru_cache
from pathlib import Path
import warnings

from unlimited_ocr.cpu_isa import POLICY
import torch
from torch import nn


@lru_cache(None)
def _load_library():
    try:
        torch.ops.load_library(str(Path(__file__).with_name("linear.so")))
    except (OSError, RuntimeError) as error:
        warnings.warn(f"Native INT8 unavailable; using FP32 dequantization: {error}")
        return False
    return True


def native_available():
    return POLICY["selected"] != "generic" and _load_library()


def backend_name():
    return "fbgemm" if native_available() else "fp32_dequantize"


class CPUInt8Linear(nn.Module):
    def __init__(self, weight, scales, bias=None):
        super().__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("scales", scales)
        self.register_buffer("bias", bias)
        self._packed = None
        self._repack()

    @classmethod
    def from_float(cls, weight, bias=None):
        weight = weight.detach().float()
        scales = weight.abs().amax(dim=1).clamp_min(1e-8) / 127
        scaled = weight * scales.reciprocal()[:, None]
        quantized = scaled.round_().clamp_(-128, 127).to(torch.int8)
        return cls(quantized, scales, None if bias is None else bias.detach().float())

    @property
    def in_features(self):
        return self.weight.shape[1]

    @property
    def out_features(self):
        return self.weight.shape[0]

    def _repack(self):
        self._packed = None
        if native_available():
            self._packed = torch.classes.unlimited_ocr.PackedLinear(self.weight, self.scales, self.bias)
        else:
            if (self.weight.device.type != "cpu" or self.weight.dtype != torch.int8
                    or self.weight.ndim != 2 or min(self.weight.shape) < 1):
                raise ValueError("weight must be a nonempty CPU int8 matrix")
            if (self.scales.device.type != "cpu" or self.scales.dtype != torch.float32
                    or self.scales.shape != (self.out_features,)
                    or not torch.isfinite(self.scales).all() or not (self.scales > 0).all()):
                raise ValueError("scales must be finite positive CPU FP32 values per output")
            if self.bias is not None and (self.bias.device.type != "cpu" or self.bias.dtype != torch.float32
                                         or self.bias.shape != (self.out_features,)):
                raise ValueError("bias must be a CPU FP32 vector per output")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)
        self._repack()

    def _apply(self, fn, recurse=True):
        self._packed = None
        super()._apply(fn, recurse=recurse)
        self._repack()
        return self

    def forward(self, x):
        if self._packed is not None:
            return torch.ops.unlimited_ocr.linear(x, self._packed)
        if x.device.type != "cpu" or x.dtype != torch.float32 or x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError("input must be CPU FP32 with matching last dimension")
        return torch.nn.functional.linear(x, self.weight.float() * self.scales[:, None], self.bias)
