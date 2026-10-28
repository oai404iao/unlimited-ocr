"""Exact FP32 head_cls summaries, incrementally cached on the request KV cache.

Only the immutable, completed blocks are reused. Routing, page candidacy and
entmax remain upstream-owned. Calls outside prefill/decode wrappers are stateless.
"""

from contextvars import ContextVar
from functools import wraps
import importlib
import math
import weakref

import torch
import torch.nn.functional as F

_context = ContextVar('cpu_router_request', default=None)


def _summaries(keys, head_cls, block_size, nblk, clamp):
    batch, length, dim = keys.shape
    padded = nblk * block_size
    if padded > length:
        kk = F.pad(keys, (0, 0, 0, padded - length))
        real = F.pad(keys.new_ones(batch, length, dtype=torch.bool),
                     (0, padded - length), value=False)
    else:
        kk = keys[:, :padded]
        real = keys.new_ones(batch, padded, dtype=torch.bool)
    kk = kk.reshape(batch, nblk, block_size, dim)
    mask = real.view(batch, nblk, block_size)
    hc = head_cls.float().view(1, 1, 1, dim)
    aw = ((kk * hc).sum(-1) * (1.0 / math.sqrt(dim))).clamp(-clamp, clamp)
    maximum = torch.where(mask, aw, aw.new_full((), torch.finfo(aw.dtype).min)).amax(-1, keepdim=True)
    maximum = torch.where(mask.any(-1, keepdim=True), maximum, torch.zeros_like(maximum))
    e = torch.exp(aw - maximum) * mask.to(aw.dtype)
    w = e / e.sum(-1, keepdim=True).clamp_min(1e-6)
    summary = torch.nan_to_num((w.unsqueeze(-1) * kk).sum(2), nan=0., posinf=0., neginf=0.)
    return summary, mask.any(-1)


def install_router(modeling_module=None):
    """Install once on the actual loaded modeling_deepseekv2 module.

    Reset at prefill, not configure_dash_inference: modeling_unlimitedocr holds
    an imported alias of the latter. Each prefill chunk replaces upstream's
    router keys, so each chunk must also replace its summaries.
    """
    module = modeling_module or importlib.import_module('upstream.modeling_deepseekv2')
    if getattr(module._block_scores_infer, '_cpu_router', False):
        return
    original_scores = module._block_scores_infer
    original_prefill = module._infer_prefill_attend
    original_decode = module._dash_decode_forward

    @wraps(original_scores)
    @torch.no_grad()
    def scores(router, idx_q, idx_k, q_pos, nblk):
        if router.stage0 != 'head_cls':
            return original_scores(router, idx_q, idx_k, q_pos, nblk)
        batch, length, dim = idx_k.shape
        bs = router.block_size
        signature = (id(router), bs, batch, dim, idx_k.device, idx_k.dtype, module._ACT_CLAMP)
        context = _context.get()
        state = None
        if context is not None:
            cache, layer, previous, appended = context
            states = getattr(cache, '_cpu_router_summaries', None)
            if states is None:
                states = cache._cpu_router_summaries = {}
            state = states.get(layer)
            if state is not None and not (
                state['keys']() is previous and previous is not None
                and state['signature'] == signature
                and length == state['length'] + appended
                and torch.equal(state['head_cls'], router.head_cls)
            ):
                state = None
        # Normal append decode cannot attend its current block. Leave the
        # incomplete block uncomputed until it completes; unusual positions
        # retain the full upstream calculation instead.
        completed_only = (context is not None and previous is not None
                          and idx_q.shape[1] == 1 and router.local_blocks >= 1
                          and bool(torch.all(q_pos == length - 1)))
        summary_blocks = min(nblk, length // bs) if completed_only else nblk
        # Previously padded/partial blocks are never reused, even when a call
        # switches between normal decode and arbitrary query positions.
        start = min(state['length'] // bs, state['summary'].shape[1], summary_blocks) if state else 0
        with torch.autocast(device_type=idx_k.device.type, enabled=False):
            if start < summary_blocks:
                summary, valid = _summaries(idx_k[:, start * bs:].float(), router.head_cls,
                                            bs, summary_blocks - start, module._ACT_CLAMP)
                if start:
                    summary = torch.cat((state['summary'][:, :start], summary), dim=1)
                    valid = torch.cat((state['valid'][:, :start], valid), dim=1)
            elif state is not None:
                summary, valid = state['summary'][:, :summary_blocks], state['valid'][:, :summary_blocks]
            else:
                summary = idx_k.new_empty((batch, 0, dim), dtype=torch.float32)
                valid = idx_k.new_empty((batch, 0), dtype=torch.bool)
            if context is not None:
                states[layer] = dict(keys=weakref.ref(idx_k), length=length, signature=signature,
                                     head_cls=state['head_cls'] if state else router.head_cls.detach().clone(),
                                     summary=summary, valid=valid)
            blocks = torch.arange(summary_blocks, device=idx_k.device)
            out = []
            for c0 in range(0, idx_q.shape[1], router.q_chunk):
                c1 = min(c0 + router.q_chunk, idx_q.shape[1])
                z = torch.matmul(idx_q[:, c0:c1].float(), summary.transpose(-1, -2)) * (1.0 / math.sqrt(dim))
                z = z.clamp(-module._ACT_CLAMP, module._ACT_CLAMP)
                qblk = q_pos[:, c0:c1] // bs
                keep = (blocks[None, None, :] <= qblk[:, :, None] - router.local_blocks) & valid[:, None, :]
                z = z.masked_fill(~keep, module._SCORE_NEG)
                if summary_blocks < nblk:
                    z = F.pad(z, (0, nblk - summary_blocks), value=module._SCORE_NEG)
                out.append(z)
            return torch.cat(out, dim=1)

    @wraps(original_prefill)
    def prefill(self, router, hidden_states, q, k, v, idx_pe, pos_ids,
                block_cand, attn_mask, past_kv):
        if past_kv is not None:
            getattr(past_kv, '_cpu_router_summaries', {}).pop(self.layer_idx, None)
        token = _context.set((past_kv, self.layer_idx, None, 0) if past_kv is not None else None)
        try:
            return original_prefill(self, router, hidden_states, q, k, v, idx_pe,
                                    pos_ids, block_cand, attn_mask, past_kv)
        finally:
            _context.reset(token)

    @wraps(original_decode)
    def decode(self, hidden_states, position_ids, past_kv):
        previous = getattr(past_kv, '_dash_idx_k', {}).get(self.layer_idx)
        token = _context.set((past_kv, self.layer_idx, previous, hidden_states.shape[1]))
        try:
            return original_decode(self, hidden_states, position_ids, past_kv)
        finally:
            _context.reset(token)

    scores._cpu_router = True
    module._block_scores_infer = scores
    module._infer_prefill_attend = prefill
    module._dash_decode_forward = decode
