import pytest
import torch
from unlimited_ocr.cpu_runtime import pack_linear
from unlimited_ocr.cpu_norm import rmsnorm


@pytest.mark.parametrize("width", [7, 128, 1280, 1291])
def test_norm(width):
    torch.manual_seed(10)
    x = torch.randn(2, 3, width)
    w = torch.randn(width)
    expected = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * w
    torch.testing.assert_close(rmsnorm(x, w, 1e-6), expected, atol=2e-6, rtol=2e-6)


def test_int8():
    torch.manual_seed(20)
    torch.backends.quantized.engine = "fbgemm"
    w = torch.randn(128, 1280) * 0.02
    x = torch.randn(4, 1280)
    b = torch.randn(128)
    packed = pack_linear(w, b)
    expected = torch.nn.functional.linear(x, w, b)
    actual = packed(x)
    assert ((actual - expected).square().mean().sqrt() / expected.square().mean().sqrt()) < 0.025


def test_streamed_loader(tmp_path, monkeypatch):
    import json
    from safetensors.torch import save_file
    import upstream.modeling_unlimitedocr as upstream
    from unlimited_ocr.cpu_runtime import load_model, CPUEmbedding

    class Tiny(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.embed = torch.nn.Embedding(20, 16)
            self.proj = torch.nn.Linear(16, 8)
            self.norm = torch.nn.LayerNorm(8)
            self.register_buffer("positions", torch.arange(8))

    torch.manual_seed(5)
    original = Tiny(None).eval()
    state = {k: v.to(torch.bfloat16) if v.is_floating_point() else v
             for k, v in original.state_dict().items()}
    save_file(state, tmp_path / "weights.safetensors")
    (tmp_path / "config.json").write_text('{"sparse_attention_config": {}}')
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "weights.safetensors" for k in state}}))
    monkeypatch.setattr(upstream, "UnlimitedOCRForCausalLM", Tiny)
    loaded = load_model(tmp_path)
    assert isinstance(loaded.embed, CPUEmbedding)
    assert loaded.embed.weight.dtype == torch.bfloat16
    assert loaded.embed(torch.tensor([1])).dtype == torch.float32
    assert not any(p.is_meta for p in loaded.parameters())
    torch.testing.assert_close(loaded.norm.weight, state["norm.weight"].float())
    torch.testing.assert_close(loaded.positions, state["positions"])
    fp = load_model(tmp_path, quantized=False)
    torch.testing.assert_close(fp.proj.weight, state["proj.weight"].float())
    bf = load_model(tmp_path, quantized=False, bf16_storage=True)
    assert bf.proj.weight.dtype == torch.bfloat16
    x = torch.randn(2, 16)
    torch.testing.assert_close(bf.proj(x), fp.proj(x))
    from unlimited_ocr.cpu_isa import POLICY
    monkeypatch.setitem(POLICY, "selected", "generic")
    with pytest.warns(UserWarning, match="AVX2 unavailable"):
        fallback = load_model(tmp_path)
    assert not fallback._cpu_quantized_weights
    assert fallback._cpu_weight_format == "bf16_storage"
    torch.testing.assert_close(fallback.proj(x), fp.proj(x))
