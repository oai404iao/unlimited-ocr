import pytest
import torch

from unlimited_ocr.cpu_isa import POLICY, cpu_flags, select_isa
from unlimited_ocr.cpu_norm import rmsnorm
from unlimited_ocr.cpu_native_attention import decode_native
from tests.test_native_attention import case
from unlimited_ocr.cpu_attention import _decode_sparse

AVX2 = {"avx2", "fma"}
AVX512 = AVX2 | {"avx512f", "avx512bw", "avx512dq", "avx512vl"}


@pytest.mark.parametrize("requested,flags,selected", [
    ("auto", set(), "generic"), ("auto", AVX2, "avx2"),
    ("auto", AVX512, "avx512"), ("avx512", AVX2, "avx2"),
    ("avx512", set(), "generic"), ("avx2", AVX512, "avx2"),
    ("generic", AVX512, "generic"), ("avx2", {"avx2"}, "generic"),
    ("vnni256", AVX2, "avx2"), ("vnni512", AVX512, "avx512"),
])
def test_safe_selection(requested, flags, selected):
    assert select_isa(requested, flags)["selected"] == selected


def test_vnni_bf16_policy():
    flags = AVX512 | {"avx512_vnni", "avx512_bf16"}
    assert select_isa("auto", flags)["fbgemm"] == "AVX512_E1_256"
    assert select_isa("vnni512", flags)["fbgemm"] == "AVX512_E1"
    assert select_isa("vnni512", AVX512)["downgraded"]
    assert select_isa("auto", flags)["onednn"] == "AVX512_CORE_BF16"
    assert not select_isa("avx512", flags)["vnni"]
    with pytest.raises(ValueError):
        select_isa("invalid", flags)


@pytest.mark.parametrize("isa", ["generic", "avx2", "avx512"])
def test_kernel_variants_and_portable_fallback(isa, monkeypatch):
    required = AVX2 if isa == "avx2" else AVX512 if isa == "avx512" else set()
    if not required <= cpu_flags():
        pytest.skip(f"CPU lacks {isa}")
    monkeypatch.setitem(POLICY, "selected", isa)
    x, weight = torch.randn(3, 129), torch.randn(129)
    expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * weight
    torch.testing.assert_close(rmsnorm(x, weight, 1e-6), expected, rtol=3e-6, atol=3e-6)
    args = case(strided=True)
    torch.testing.assert_close(decode_native(*args), _decode_sparse(*args), rtol=3e-5, atol=3e-6)


def test_missing_native_uses_torch(monkeypatch):
    from unlimited_ocr import cpu_norm
    from unlimited_ocr import cpu_native_attention
    monkeypatch.setattr(cpu_norm, "_kernel", lambda isa: None)
    monkeypatch.setattr(cpu_native_attention, "_kernel", lambda isa: None)
    assert cpu_norm.backend_name() == "torch"
    assert cpu_native_attention.backend_name() == "torch_sdpa"
    args = case()
    torch.testing.assert_close(decode_native(*args), _decode_sparse(*args), rtol=3e-5, atol=3e-6)


def test_bf16_unavailable_leaves_model_untouched(monkeypatch):
    from unlimited_ocr.cpu_bf16_vision import install_bf16_sam
    monkeypatch.setitem(POLICY, "bf16", False)
    with pytest.warns(UserWarning, match="BF16 unavailable"):
        assert not install_bf16_sam(None)


def test_library_load_failure_falls_back(monkeypatch):
    import ctypes
    from unlimited_ocr import cpu_norm
    from unlimited_ocr import cpu_native_attention
    def missing(*args, **kwargs):
        raise OSError("test unavailable library")
    monkeypatch.setattr(ctypes, "CDLL", missing)
    monkeypatch.setitem(POLICY, "selected", "avx2")
    cpu_norm._kernel.cache_clear()
    cpu_native_attention._kernel.cache_clear()
    try:
        with pytest.warns(UserWarning, match="Native RMSNorm unavailable"):
            assert cpu_norm.backend_name() == "torch"
        with pytest.warns(UserWarning, match="Native attention unavailable"):
            assert cpu_native_attention.backend_name() == "torch_sdpa"
        x, w = torch.randn(2, 13), torch.randn(13)
        expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * w
        torch.testing.assert_close(rmsnorm(x, w, 1e-6), expected)
    finally:
        cpu_norm._kernel.cache_clear()
        cpu_native_attention._kernel.cache_clear()


def test_optional_probe_cannot_block_results(monkeypatch):
    import ctypes
    from unlimited_ocr.cpu_isa import isa_report
    from pathlib import Path
    monkeypatch.setitem(POLICY, "selected", "avx2")
    monkeypatch.setattr(Path, "exists", lambda path: True)
    def invalid(*args):
        raise OSError("test incompatible probe")
    monkeypatch.setattr(ctypes, "CDLL", invalid)
    assert "probe_error" in isa_report()


def test_bf16_restores_original_checkpoint_linears(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from safetensors.torch import save_file
    from unlimited_ocr.cpu_runtime import pack_linear
    from unlimited_ocr.cpu_bf16_vision import install_bf16_sam
    class Encoder(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.proj = layer
        def forward(self, x):
            return self.proj(x)
    torch.manual_seed(4)
    weight = torch.randn(8, 16).bfloat16()
    bias = torch.randn(8).bfloat16()
    tensors = {"model.sam_model.proj.weight": weight, "model.sam_model.proj.bias": bias}
    save_file(tensors, tmp_path / "model.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {key: "model.safetensors" for key in tensors}}))
    sam = Encoder(pack_linear(weight, bias)).eval()
    model = SimpleNamespace(model=SimpleNamespace(sam_model=sam))
    monkeypatch.setitem(POLICY, "bf16", True)
    assert install_bf16_sam(model, tmp_path)
    assert install_bf16_sam(model, tmp_path)
    torch.testing.assert_close(sam.proj.weight, weight, atol=0, rtol=0)
    torch.testing.assert_close(sam.proj.bias, bias, atol=0, rtol=0)
    x = torch.randn(2, 16)
    with torch.inference_mode():
        expected = torch.nn.functional.linear(x.bfloat16(), weight, bias).float()
        torch.testing.assert_close(sam(x), expected, atol=0, rtol=0)
