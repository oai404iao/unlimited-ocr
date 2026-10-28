"""Append-only inference KV cache for transformers 4.52.1.

Storage doubles when necessary; memory still grows with sequence length. No
tokens are evicted: sparse attention may refer back to remote image tokens.
Use ``CPUCache`` explicitly in place of ``DynamicCache`` (no global patch).
"""

import torch
from transformers.cache_utils import DynamicCache


class CPUCache(DynamicCache):
    """Single-request cache with amortized append cost instead of per-token cat.

    Public key_cache/value_cache entries are views of the populated prefix,
    never the unused capacity. Like any mutable inference cache, returned views
    should not be treated as immutable snapshots. Training and beam/batch
    transformations are not supported.
    """

    def __init__(self, _distributed_cache_data=None):
        super().__init__()
        self._key_buffers = []
        self._value_buffers = []
        if _distributed_cache_data is not None:
            for layer_idx, (keys, values) in enumerate(_distributed_cache_data):
                self.update(keys, values, layer_idx)

    @torch.no_grad()
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        # cache_kwargs is intentionally ignored, as in DynamicCache: appends
        # preserve every position rather than implementing a sliding window.
        if key_states.shape[0] != 1 or value_states.shape[0] != 1:
            raise ValueError("CPUCache supports a single request (batch size 1)")
        if key_states.shape[-2] != value_states.shape[-2]:
            raise ValueError("Key and value sequence lengths must match")
        if layer_idx < 0:
            raise ValueError("layer_idx must be nonnegative")

        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(torch.tensor([]))
            self.value_cache.append(torch.tensor([]))
            self._key_buffers.append(None)
            self._value_buffers.append(None)

        length = self.get_seq_length(layer_idx)
        new_length = length + key_states.shape[-2]
        keys = self._key_buffers[layer_idx]
        values = self._value_buffers[layer_idx]
        capacity = keys.shape[-2] if keys is not None else 0
        if keys is None or new_length > capacity:
            capacity = max(new_length, max(1, capacity * 2))
            key_shape = (*key_states.shape[:-2], capacity, key_states.shape[-1])
            value_shape = (*value_states.shape[:-2], capacity, value_states.shape[-1])
            keys = key_states.new_empty(key_shape)
            values = value_states.new_empty(value_shape)
            if length:
                keys[..., :length, :].copy_(self.key_cache[layer_idx])
                values[..., :length, :].copy_(self.value_cache[layer_idx])
            self._key_buffers[layer_idx] = keys
            self._value_buffers[layer_idx] = values

        keys[..., length:new_length, :].copy_(key_states)
        values[..., length:new_length, :].copy_(value_states)
        self.key_cache[layer_idx] = keys[..., :new_length, :]
        self.value_cache[layer_idx] = values[..., :new_length, :]
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx):
        raise NotImplementedError("CPUCache does not support beam search")

    def batch_split(self, full_batch_size, split_size):
        raise NotImplementedError("CPUCache does not support batch transformations")

    def batch_repeat_interleave(self, repeats):
        raise NotImplementedError("CPUCache does not support batch transformations")

    def batch_select_indices(self, indices):
        raise NotImplementedError("CPUCache does not support batch transformations")
