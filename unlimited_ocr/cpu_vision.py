"""Memory-bounded query chunking for the SAM encoder's CPU inference."""

from types import MethodType

import torch
import torch.nn.functional as F

from upstream.deepencoder import Attention, add_decomposed_rel_pos


@torch.no_grad()
def attention_forward(self, x):
    """Keep all keys, materializing relative bias for only a query chunk.

    At a 64x64 grid with 12 heads and batch size one, the float32 bias
    chunk of 64 is 12 MiB (256 is 48 MiB), versus the original full 768 MiB bias.
    Compact height/width terms and QKV still grow with the image grid.
    """
    batch, height, width, _ = x.shape
    tokens = height * width
    qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, -1)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

    if self.use_rel_pos:
        rel_h, rel_w = add_decomposed_rel_pos(
            q.reshape(batch * self.num_heads, tokens, -1),
            self.rel_pos_h,
            self.rel_pos_w,
            (height, width),
            (height, width),
        )
        rel_h = rel_h.reshape(batch, self.num_heads, tokens, height, 1)
        rel_w = rel_w.reshape(batch, self.num_heads, tokens, 1, width)

    output = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    chunk = getattr(self, "cpu_query_chunk", 64)
    for start in range(0, tokens, chunk):
        end = min(start + chunk, tokens)
        bias = None
        if self.use_rel_pos:
            # Slice BEFORE broadcasting; never form [B, heads, tokens, tokens].
            bias = (rel_h[:, :, start:end] + rel_w[:, :, start:end]).reshape(
                batch, self.num_heads, end - start, tokens
            )
        output[:, :, start:end].copy_(
            F.scaled_dot_product_attention(q[:, :, start:end], k, v, attn_mask=bias)
        )
        del bias

    output = output.transpose(1, 2).reshape(batch, height, width, -1)
    return self.proj(output)


def install_vision(model, query_chunk=64):
    """Patch only this model's SAM Attention instances, not global classes."""
    if query_chunk < 1:
        raise ValueError("query_chunk must be positive")
    for module in model.modules():
        if isinstance(module, Attention):
            module.cpu_query_chunk = query_chunk
            module.forward = MethodType(attention_forward, module)
