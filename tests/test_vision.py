import pytest
import torch
from torch import nn

from unlimited_ocr.cpu_vision import install_vision
from upstream.deepencoder import Attention


@pytest.mark.parametrize("height,width", [(4, 4), (9, 11), (11, 7)])
@pytest.mark.parametrize("use_rel_pos", [False, True])
@pytest.mark.parametrize("chunk", [16, 64, 256])
@torch.no_grad()
def test_chunked_matches_original(height, width, use_rel_pos, chunk):
    torch.manual_seed(7)
    # Different input_size exercises relative-position interpolation, too.
    attention = Attention(24, 3, use_rel_pos=use_rel_pos, input_size=(5, 6)).eval()
    assert isinstance(attention.qkv, nn.Linear)
    if use_rel_pos:
        attention.rel_pos_h.normal_()
        attention.rel_pos_w.normal_()
    x = torch.randn(2, height, width, 24)
    expected = attention(x)
    install_vision(attention, query_chunk=chunk)
    actual = attention(x)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@torch.no_grad()
def test_global_grid_sdpa_has_bounded_queries_and_all_keys(monkeypatch):
    attention = Attention(24, 3, use_rel_pos=True, input_size=(64, 64)).eval()
    install_vision(attention)
    original = torch.nn.functional.scaled_dot_product_attention
    calls = []

    def checked(q, k, v, attn_mask=None):
        assert q.shape[-2] <= 64
        assert k.shape[-2] == v.shape[-2] == 4096
        assert attn_mask.shape == (1, 3, q.shape[-2], 4096)
        calls.append(q.shape[-2])
        return original(q, k, v, attn_mask=attn_mask)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", checked)
    result = attention(torch.randn(1, 64, 64, 24))
    assert result.shape == (1, 64, 64, 24)
    assert torch.isfinite(result).all()
    assert calls == [64] * 64


def test_install_is_local_and_idempotent():
    patched = Attention(24, 3)
    untouched = Attention(24, 3)
    model = nn.Sequential(patched, nn.Linear(24, 24))
    install_vision(model)
    install_vision(model)
    assert patched.forward.__func__ is not Attention.forward
    assert untouched.forward.__func__ is Attention.forward
