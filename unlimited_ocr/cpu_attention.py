"""Gathered DASH attention for CPU inference; routing remains upstream-owned.

Small query chunks share a gathered union of their selected blocks and local
windows. No sequence-by-sequence attention scores or block-bias map is built.
"""

import importlib

import torch
import torch.nn.functional as F

_MASK_NEG = -1.0e9


def _selected_tokens(topi, bias, valid, position, length, block_size, window):
    # scatter's CPU duplicate semantics: the LAST slot wins, even if invalid.
    slots = {block: slot for slot, block in enumerate(topi.tolist())}
    indices, biases = [], []
    local_start = max(0, position - window + 1)
    for block, slot in sorted(slots.items()):
        if not valid[slot] or not bias[slot] > _MASK_NEG * 0.5:
            continue
        start = max(0, block * block_size)
        end = min((block + 1) * block_size, length, position + 1, local_start)
        if start < end:
            tokens = torch.arange(start, end, device=topi.device)
            indices.append(tokens)
            biases.append(bias[slot].expand(tokens.numel()))
    local = torch.arange(min(local_start, length), min(position + 1, length),
                         device=topi.device)
    indices.append(local)
    biases.append(bias.new_zeros(local.numel()))
    return torch.cat(indices), torch.cat(biases)


def _forward_sparse_rows(self, q, k, v, topi, blk_bias, valid, S, nblk, attention_mask):
    batches = []
    for b in range(q.shape[0]):
        rows = []
        for row in range(S):
            tokens, bias = _selected_tokens(
                topi[b, row], blk_bias[b, row], valid[b, row], row, S,
                self.block_size, self.local_window)
            mask = bias[None, None, :]
            extra = None
            if attention_mask is not None:
                extra = attention_mask[b:b + 1, :, row:row + 1, :S].to(bias.dtype)
                mask = mask + extra[0].index_select(-1, tokens)
            # The reference uses a FINITE sentinel. If every selected key is
            # masked, excluded keys can regain mass. Reproduce that exceptional
            # case with one full row (O(S), never O(S**2)).
            if not tokens.numel() or (mask <= _MASK_NEG * 0.5).all(-1).any():
                full = bias.new_full((S,), _MASK_NEG)
                full[tokens] = bias
                tokens = torch.arange(S, device=q.device)
                mask = full[None, None, :]
                if extra is not None:
                    mask = mask + extra[0]
            mask = mask.clamp_min(_MASK_NEG).to(q.dtype)
            rows.append(F.scaled_dot_product_attention(
                q[b, :, row:row + 1], k[b].index_select(1, tokens),
                v[b].index_select(1, tokens), attn_mask=mask,
                is_causal=False, scale=self._scale))
        batches.append(torch.cat(rows, dim=1))
    return torch.stack(batches)


def _forward_sparse(self, q, k, v, topi, blk_bias, valid, S, nblk, attention_mask):
    batches = []
    block_offsets = torch.arange(self.block_size, device=q.device)
    for b in range(q.shape[0]):
        chunks = []
        for start in range(0, S, 128):
            end = min(start + 128, S)
            blocks = topi[b, start:end]
            prior = blk_bias[b, start:end]
            positions = torch.arange(start, end, device=q.device)[:, None]
            local_blocks = torch.arange(max(0, start - self.local_window + 1) // self.block_size,
                                        (end - 1) // self.block_size + 1, device=q.device)
            union_blocks = torch.unique(torch.cat((blocks.flatten(), local_blocks)), sorted=True)
            mapped = torch.searchsorted(union_blocks, blocks)
            base = prior.new_full((end - start, union_blocks.numel()), float('-inf'))
            active = valid[b, start:end] & (prior > _MASK_NEG * .5)
            # CPU scatter preserves the reference's last-write duplicate rule.
            base.scatter_(-1, mapped, prior.masked_fill(~active, float('-inf')))
            union = (union_blocks[:, None] * self.block_size + block_offsets).flatten()
            mask = base.repeat_interleave(self.block_size, dim=-1)
            causal = (union[None] <= positions) & (union[None] < S) & (union[None] >= 0)
            local = causal & (union[None] > positions - self.local_window)
            mask = torch.where(local, 0., mask).masked_fill(~causal, float('-inf'))
            # Gather only the actual union, dropping future/invalid selections.
            used = torch.isfinite(mask).any(0)
            union, mask = union[used], mask[:, used]
            selected = torch.isfinite(mask)
            mask = mask[None]
            if attention_mask is not None:
                mask = mask + attention_mask[b, :, start:end, :S].index_select(-1, union).to(mask.dtype)
            # Preserve the reference's finite-sentinel all-masked behavior.
            if not union.numel() or (mask <= _MASK_NEG * .5).all(-1).any():
                return _forward_sparse_rows(self, q, k, v, topi, blk_bias, valid,
                                            S, nblk, attention_mask)
            mask = mask.clamp_min(_MASK_NEG).to(q.dtype).masked_fill(~selected[None], float('-inf'))
            if int(union[-1] - union[0]) + 1 == union.numel():
                # Short prefixes often select a contiguous union. Slicing is
                # the same gathered read set without copying H*D key/value data.
                first, last = int(union[0]), int(union[-1]) + 1
                kk, vv = k[b:b + 1, :, first:last], v[b:b + 1, :, first:last]
            else:
                kk, vv = k[b:b + 1].index_select(2, union), v[b:b + 1].index_select(2, union)
            chunks.append(F.scaled_dot_product_attention(
                q[b:b + 1, :, start:end], kk, vv, attn_mask=mask[None],
                is_causal=False, scale=self._scale)[0])
        batches.append(torch.cat(chunks, dim=1))
    return torch.stack(batches)


@torch.no_grad()
def _decode_sparse(q, k_all, v_all, topi, bias, valid, q_pos,
                   block_size, local_window, scale):
    batches = []
    heads = torch.arange(q.shape[1], device=q.device) // (q.shape[1] // k_all.shape[1])
    for b in range(q.shape[0]):
        rows = []
        for row in range(q.shape[2]):
            tokens, prior = _selected_tokens(
                topi[b, row], bias[b, row], valid[b, row], int(q_pos[b, row]),
                k_all.shape[2], block_size, local_window)
            if not tokens.numel():
                # Decode's -inf mask yields NaN for an entirely empty read set.
                rows.append(q.new_full((q.shape[1], 1, v_all.shape[-1]),
                                       float('nan'), dtype=torch.float32))
                continue
            kk = k_all[b].index_select(1, tokens).index_select(0, heads).float()
            vv = v_all[b].index_select(1, tokens).index_select(0, heads).float()
            logits = q[b, :, row:row + 1].float() @ kk.transpose(-1, -2)
            weights = torch.softmax(logits * scale + prior.float()[None, None], dim=-1)
            rows.append(weights @ vv)
        batches.append(torch.cat(rows, dim=1))
    return torch.stack(batches)


@torch.no_grad()
def _decode_sdpa(q, k_all, v_all, topi, bias, valid, q_pos,
                 block_size, local_window, scale):
    """Reuse PyTorch CPU SDPA; vectorized read-set selection, no custom math kernel."""
    batches = []
    offsets = torch.arange(block_size, device=q.device)
    slots = torch.arange(topi.shape[-1], device=q.device)
    for b in range(q.shape[0]):
        rows = []
        for row in range(q.shape[2]):
            blocks = topi[b, row]
            prior = bias[b, row].float()
            position = int(q_pos[b, row])
            length = k_all.shape[2]
            local_start = max(0, position - local_window + 1)
            # A later duplicate shadows earlier slots, even when invalid.
            last = ~((blocks[:, None] == blocks[None, :]) & (slots[:, None] < slots[None, :])).any(-1)
            remote = blocks[:, None] * block_size + offsets
            active = last & valid[b, row] & (prior > _MASK_NEG * .5)
            keep = active[:, None] & (remote >= 0) & (remote < min(length, position + 1, local_start))
            tokens = remote[keep]
            biases = prior[:, None].expand_as(remote)[keep]
            order = tokens.argsort()
            local = torch.arange(min(local_start, length), max(0, min(position + 1, length)), device=q.device)
            tokens = torch.cat((tokens[order], local))
            biases = torch.cat((biases[order], prior.new_zeros(local.numel())))
            if not tokens.numel():
                rows.append(q.new_full((q.shape[1], 1, v_all.shape[-1]), float('nan'), dtype=torch.float32))
                continue
            kk, vv = k_all[b:b + 1].index_select(2, tokens).float(), v_all[b:b + 1].index_select(2, tokens).float()
            groups = q.shape[1] // kk.shape[1]
            if groups != 1:
                kk, vv = kk.repeat_interleave(groups, 1), vv.repeat_interleave(groups, 1)
            rows.append(F.scaled_dot_product_attention(
                q[b:b + 1, :, row:row + 1].float(), kk, vv,
                attn_mask=biases[None, None, None], scale=scale)[0])
        batches.append(torch.cat(rows, dim=1))
    return torch.stack(batches)


def install_attention(modeling_module=None):
    """Patch the loaded upstream modeling module (or the local upstream package).

    Pass the actual modeling_deepseekv2 module when loading via Transformers'
    dynamic module cache. Calling this repeatedly is harmless.
    """
    if modeling_module is None:
        modeling_module = importlib.import_module('upstream.modeling_deepseekv2')
    modeling_module.DashAttention._forward_dense = _forward_sparse
    modeling_module._decode_attend = _decode_sparse
