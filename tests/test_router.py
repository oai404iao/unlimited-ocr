"""Small weight-free comparisons against the actual upstream scoring equation."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from unlimited_ocr import cpu_router

ROOT = Path(__file__).resolve().parents[1]
tree = ast.parse((ROOT / 'upstream/modeling_deepseekv2.py').read_text())
namespace = dict(torch=torch, F=F, math=math, _ACT_CLAMP=1e4, _SCORE_NEG=-1e9)
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
         and n.name in ('_block_scores_infer', '_infer_prefill_attend')]
exec(compile(ast.Module(body=nodes, type_ignores=[]), '<upstream router>', 'exec'), namespace)
reference = namespace['_block_scores_infer']


def make_module(actual_decode=False):
    # Run the real prefill. Decode's projections/attention are irrelevant to
    # this patch; keep its exact router-key append and score-call lifecycle.
    ns = dict(namespace)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<upstream router>', 'exec'), ns)
    ns['_router_qk'] = lambda router, hs, pe: (hs, hs)
    ns['_infer_route_select'] = lambda sc, cand, p: (sc, cand, None)

    def decode(self, hs, positions, cache):
        keys = torch.cat((cache._dash_idx_k[self.layer_idx], hs), dim=1)
        cache._dash_idx_k[self.layer_idx] = keys
        return ns['_block_scores_infer'](self.dash_router, hs, keys, positions,
                                         math.ceil(keys.shape[1] / self.dash_router.block_size))

    # Module attributes and function globals must share the installed functions.
    class Module:
        def __getattr__(self, name):
            return ns[name]

        def __setattr__(self, name, value):
            ns[name] = value

    module = Module()
    module._dash_decode_forward = decode
    if actual_decode:
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_dash_decode_forward')
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<upstream decode>', 'exec'), ns)
    cpu_router.install_router(module)
    return module


def router():
    return SimpleNamespace(stage0='head_cls', block_size=4, q_chunk=3,
                           local_blocks=1, local_window=4, head_cls=torch.randn(8),
                           _infer_params={}, use_triton=False,
                           _forward_dense=lambda q, k, v, scores, cand, valid, *args: scores)


def prefill(module, attn, cache, keys, positions=None):
    if positions is None:
        positions = torch.arange(keys.shape[1])[None].expand(keys.shape[0], -1)
    q = keys[:, None]
    return module._infer_prefill_attend(attn, attn.dash_router, keys, q, q, q,
                                       None, positions, None, None, cache)


@pytest.mark.parametrize('batch', [1, 2])
@pytest.mark.parametrize('initial', [1, 3, 4, 5, 8])
def test_prefill_incremental_boundaries(batch, initial, monkeypatch):
    torch.manual_seed(11)
    module, r, cache = make_module(), router(), SimpleNamespace()
    attn = SimpleNamespace(layer_idx=0, dash_router=r)
    keys = torch.randn(batch, initial, 8)
    positions = torch.arange(initial)[None].expand(batch, -1)
    got = prefill(module, attn, cache, keys)
    torch.testing.assert_close(got, reference(r, keys, keys, positions, math.ceil(initial / 4)), rtol=0, atol=0)
    lengths = []
    original = cpu_router._summaries

    def record(keys, *args):
        lengths.append(keys.shape[1])
        return original(keys, *args)

    monkeypatch.setattr(cpu_router, '_summaries', record)
    for step in range(12):
        calls_before = len(lengths)
        new = torch.randn(batch, 1, 8)
        keys = torch.cat((keys, new), dim=1)
        # Distinct batch positions, including non-monotone positions, exercise
        # causality/local exclusion independently of key-summary caching.
        positions = torch.tensor([[keys.shape[1] - 1 - b * 3] for b in range(batch)])
        got = module._dash_decode_forward(attn, new, positions, cache)
        expected = reference(r, new, keys, positions, math.ceil(keys.shape[1] / 4))
        torch.testing.assert_close(got, expected, rtol=2e-6, atol=2e-6)
        assert got.dtype == torch.float32
        if batch == 1 and keys.shape[1] % 4:
            assert len(lengths) == calls_before
        else:
            assert len(lengths) == calls_before + 1
            assert lengths[-1] == (keys.shape[1] - 1) % 4 + 1


def test_request_and_chunk_reset_and_cache_identity():
    module, r = make_module(), router()
    attn = SimpleNamespace(layer_idx=0, dash_router=r)
    for cache in [SimpleNamespace(), SimpleNamespace()]:
        for length, batch in [(8, 1), (8, 1), (3, 2), (10, 2)]:
            keys = torch.randn(batch, length, 8)
            prefill(module, attn, cache, keys)
            new = torch.randn(batch, 1, 8)
            positions = torch.full((batch, 1), length)
            got = module._dash_decode_forward(attn, new, positions, cache)
            all_keys = torch.cat((keys, new), 1)
            torch.testing.assert_close(got, reference(r, new, all_keys, positions, math.ceil((length + 1) / 4)))
        # Replacing the keys outside the wrapper invalidates a same-length state.
        cache._dash_idx_k[0] = torch.randn_like(cache._dash_idx_k[0])
        new = torch.randn(2, 1, 8)
        got = module._dash_decode_forward(attn, new, torch.full((2, 1), 11), cache)
        torch.testing.assert_close(got, reference(r, new, cache._dash_idx_k[0], torch.full((2, 1), 11), 3))


@pytest.mark.parametrize('change', ['head_value', 'head_replace', 'block_size', 'stage', 'local'])
def test_configuration_changes(change):
    module, r, cache = make_module(), router(), SimpleNamespace()
    attn = SimpleNamespace(layer_idx=0, dash_router=r)
    keys = torch.randn(2, 12, 8)
    prefill(module, attn, cache, keys)
    if change == 'head_value':
        r.head_cls.add_(4)
    elif change == 'head_replace':
        r.head_cls = torch.randn(8)
    elif change == 'block_size':
        r.block_size = 3
    elif change == 'stage':
        r.stage0 = 'indexer'
    else:
        r.local_blocks = 3
    new, positions = torch.randn(2, 1, 8), torch.tensor([[12], [8]])
    got = module._dash_decode_forward(attn, new, positions, cache)
    torch.testing.assert_close(got, reference(r, new, torch.cat((keys, new), 1), positions, math.ceil(13 / r.block_size)))


@pytest.mark.parametrize('stage', ['head_cls', 'indexer'])
def test_stateless_masks_clamp_and_padding(stage):
    module, r = make_module(), router()
    r.stage0 = stage
    for length in [1, 4, 9]:
        keys, queries = torch.randn(2, length, 8) * 1e3, torch.randn(2, 7, 8) * 1e3
        if stage == 'head_cls':
            keys[0, 0, 0] = float('nan')
            keys[1, 0, 0] = float('inf')
        positions = torch.tensor([[-1, 0, 4, 8, 12, 20, 3], [20, 12, 8, 4, 0, -1, 3]])
        nblk = math.ceil(length / 4) + (2 if stage == 'head_cls' else 0)
        got = module._block_scores_infer(r, queries, keys, positions, nblk)
        torch.testing.assert_close(got, reference(r, queries, keys, positions, nblk), rtol=0, atol=0)


def test_head_dimension_mismatch_and_install_idempotence():
    module, r = make_module(), router()
    installed = module._block_scores_infer
    cpu_router.install_router(module)
    assert module._block_scores_infer is installed
    r.head_cls = torch.randn(7)
    with pytest.raises(RuntimeError):
        module._block_scores_infer(r, torch.randn(1, 1, 8), torch.randn(1, 8, 8), torch.tensor([[7]]), 2)


def test_no_cache_and_inference_mode():
    module, r = make_module(), router()
    attn = SimpleNamespace(layer_idx=0, dash_router=r)
    with torch.inference_mode():
        keys = torch.randn(1, 7, 8)
        a = prefill(module, attn, None, keys)
        b = prefill(module, attn, SimpleNamespace(), keys)
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_actual_upstream_decode_wiring_and_page_candidates():
    module, r = make_module(actual_decode=True), router()
    r.image_only, r._scale = True, 8 ** -.5
    attn = SimpleNamespace(layer_idx=0, dash_router=r, indexer=SimpleNamespace(rope_mode='1d'),
                           config=SimpleNamespace(num_attention_heads=1, num_key_value_heads=1),
                           head_dim=8, main_rope_mode='1d', q_proj=lambda x: x,
                           k_proj=lambda x: x, v_proj=lambda x: x, o_proj=lambda x: x,
                           rotary_emb=lambda v, pos: (None, None),
                           _dash_dec_allow=torch.tensor([[True, False]]))
    module._llama_apply_rotary_pos_emb = lambda q, k, cos, sin: (q, k)
    module._dash_indexer_rope = lambda *args: None
    module._decode_attend = lambda q, *args: q
    cache = SimpleNamespace()
    keys = torch.randn(1, 8, 8)
    prefill(module, attn, cache, keys)

    def update(k, v, layer):
        length = cache._dash_idx_k[layer].shape[1] + k.shape[2]
        return k.new_zeros(1, 1, length, 8), v.new_zeros(1, 1, length, 8)

    cache.update = update
    captured = []

    def select(scores, candidates, params):
        captured.append((scores, candidates))
        return scores, candidates, None

    module._infer_route_select = select
    with torch.inference_mode():
        for length in range(9, 14):
            new, positions = torch.randn(1, 1, 8), torch.tensor([[length - 1]])
            module._dash_decode_forward(attn, new, positions, cache)
            keys = torch.cat((keys, new), 1)
            scores, candidates = captured[-1]
            torch.testing.assert_close(scores, reference(r, new, keys, positions, math.ceil(length / 4)))
            assert candidates.shape == (1, 1, math.ceil(length / 4))
            assert candidates[0, 0, 0] and not candidates[0, 0, 1:].any()


def test_lazy_partial_block_falls_back_for_future_positions_and_no_local_blocks(monkeypatch):
    module, r, cache = make_module(), router(), SimpleNamespace()
    attn = SimpleNamespace(layer_idx=0, dash_router=r)
    keys = torch.randn(2, 4, 8)
    prefill(module, attn, cache, keys)
    calls = []
    original = cpu_router._summaries

    def record(keys, *args):
        calls.append(keys.shape[1])
        return original(keys, *args)

    monkeypatch.setattr(cpu_router, '_summaries', record)
    for length, position, local_blocks in [(5, 4, 1), (6, 20, 1), (7, 6, 1),
                                            (8, 7, 1), (9, 8, 0), (10, 9, 1)]:
        r.local_blocks = local_blocks
        new = torch.randn(2, 1, 8)
        keys = torch.cat((keys, new), 1)
        positions = torch.full((2, 1), position)
        got = module._dash_decode_forward(attn, new, positions, cache)
        torch.testing.assert_close(got, reference(r, new, keys, positions, math.ceil(length / 4)))
    # Ordinary partial steps (5, 7, 10) do no summary work. Future-position
    # and local_blocks=0 calls still compute their visible partial block.
    assert calls == [2, 4, 1]
