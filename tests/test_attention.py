"""Compare against the unmodified upstream functions without loading weights."""

import ast
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from unlimited_ocr.cpu_attention import _decode_sparse, _forward_sparse, install_attention

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reference_dash', ROOT / 'upstream/dash_attention.py')
dash = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dash)

# Decode's module imports the full vision/LM stack. Compile its actual function
# verbatim, supplying only its two globals, to keep these unit tests lightweight.
tree = ast.parse((ROOT / 'upstream/modeling_deepseekv2.py').read_text())
node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_decode_attend')
namespace = {'torch': torch, '_MASK_NEG': dash._MASK_NEG,
             '_llama_repeat_kv': lambda x, n: x.repeat_interleave(n, dim=1)}
exec(compile(ast.Module(body=[node], type_ignores=[]), '<upstream decode>', 'exec'), namespace)
decode_reference = namespace['_decode_attend']


def inputs(length, queries, kv_heads=3, dtype=torch.float32):
    torch.manual_seed(length + queries)
    q = torch.randn(2, 3, queries, 8, dtype=dtype)
    k = torch.randn(2, kv_heads, length, 8, dtype=dtype)
    v = torch.randn_like(k)
    nblk = math.ceil(length / 4)
    topi = torch.randint(nblk, (2, queries, 5))
    bias = torch.randn(2, queries, 5) * 2
    valid = torch.rand(2, queries, 5) > .3
    # Duplicate slots, invalid last writes, local overlap, future selections.
    topi[..., 1] = topi[..., 0]
    valid[0, :, 1] = False
    topi[..., -1] = nblk - 1
    return q, k, v, topi, bias, valid


@pytest.mark.parametrize('length', [1, 3, 4, 5, 17, 33, 257])
@pytest.mark.parametrize('mask_kind', ['none', 'padding', 'all', 'head_bias'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_prefill(length, mask_kind, dtype):
    q, k, v, topi, bias, valid = inputs(length, length, dtype=dtype)
    router = SimpleNamespace(block_size=4, local_window=5, q_chunk=3, _scale=8 ** -.5)
    mask = None
    if mask_kind != 'none':
        mask = torch.zeros(2, 3 if mask_kind == 'head_bias' else 1, length, length)
        if mask_kind == 'padding':
            mask[0, ..., :2] = -float('inf')
        elif mask_kind == 'all':
            mask[:] = -float('inf')
        else:
            mask.normal_(0, .2)
    expected = dash.DashAttention._forward_dense(router, q, k, v, topi, bias, valid,
                                                length, math.ceil(length / 4), mask)
    actual = _forward_sparse(router, q, k, v, topi, bias, valid,
                             length, math.ceil(length / 4), mask)
    torch.testing.assert_close(actual, expected, atol=.016 if dtype == torch.bfloat16 else 2e-6,
                               rtol=.016 if dtype == torch.bfloat16 else 2e-5)


@pytest.mark.parametrize('length', [1, 3, 4, 5, 17, 33])
@pytest.mark.parametrize('kv_heads', [1, 3])
def test_decode(length, kv_heads):
    args = inputs(length, 3, kv_heads)
    positions = torch.tensor([[0, max(0, length - 2), length - 1],
                              [length - 1, 0, length // 2]])
    expected = decode_reference(*args, positions, 4, 5, 8 ** -.5)
    actual = _decode_sparse(*args, positions, 4, 5, 8 ** -.5)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


def test_install():
    class Router:
        pass
    module = SimpleNamespace(DashAttention=Router)
    install_attention(module)
    install_attention(module)
    assert Router._forward_dense is _forward_sparse
    assert module._decode_attend is _decode_sparse


def test_prefill_gathers_bounded_read_set(monkeypatch):
    args = inputs(257, 257)
    router = SimpleNamespace(block_size=4, local_window=5, _scale=8 ** -.5)
    original = torch.nn.functional.scaled_dot_product_attention
    calls = []

    def checked(q, k, v, **kwargs):
        assert q.shape[-2] <= 128
        assert k.shape[-2] <= q.shape[-2] * (5 * 4 + 5)
        assert (kwargs['attn_mask'].isfinite().sum(-1) <= 5 * 4 + 5).all()
        assert kwargs['attn_mask'].shape[-1] == k.shape[-2]
        calls.append(k.shape[-2])
        return original(q, k, v, **kwargs)

    monkeypatch.setattr(torch.nn.functional, 'scaled_dot_product_attention', checked)
    _forward_sparse(router, *args, 257, math.ceil(257 / 4), None)
    assert len(calls) == 2 * math.ceil(257 / 128)


def benchmark():
    """Run manually with `python -m tests.test_attention` (no weights needed)."""
    import time

    for threads in (1, 3):
        torch.set_num_threads(threads)
        for length in (273, 1024):
            torch.manual_seed(0)
            q = torch.randn(1, 10, length, 128)
            k, v = torch.randn_like(q), torch.randn_like(q)
            nblk = math.ceil(length / 32)
            blocks = torch.arange(nblk)[None].expand(length, -1)
            candidates = blocks <= torch.arange(length)[:, None] // 32 - 4
            scores = torch.rand(length, nblk).masked_fill(~candidates, -1)
            topi = scores.topk(min(12, nblk), dim=-1).indices[None]
            valid = candidates[None].gather(-1, topi)
            bias = torch.randn_like(topi, dtype=torch.float32)
            router = SimpleNamespace(block_size=32, local_window=128,
                                     q_chunk=2048, _scale=128 ** -.5)
            args = (router, q, k, v, topi, bias, valid, length, nblk, None)
            outputs = []
            for name, fn in [('dense', dash.DashAttention._forward_dense),
                             ('gathered', _forward_sparse)]:
                with torch.no_grad():
                    outputs.append(fn(*args))
                    times = []
                    for _ in range(3):
                        start = time.perf_counter()
                        fn(*args)
                        times.append(time.perf_counter() - start)
                print(f'threads={threads} S={length} {name}: {min(times) * 1000:.2f} ms')
            torch.testing.assert_close(*outputs, atol=2e-6, rtol=2e-5)


if __name__ == '__main__':
    benchmark()
