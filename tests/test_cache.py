import pytest
import torch
from transformers.cache_utils import DynamicCache

from unlimited_ocr.cpu_cache import CPUCache


def states(length, offset=0):
    keys = torch.arange(2 * length * 3, dtype=torch.float32).reshape(1, 2, length, 3) + offset
    values = torch.arange(2 * length * 5, dtype=torch.float32).reshape(1, 2, length, 5) - offset
    return keys, values


def test_append_matches_dynamic_cache_multiple_layers():
    actual, expected = CPUCache(), DynamicCache()
    for step, length in enumerate([3, 1, 1, 1, 4, 1, 13]):
        for layer in range(3):
            keys, values = states(length, step * 100 + layer * 1000)
            a = actual.update(keys, values, layer, {"cache_position": None})
            b = expected.update(keys, values, layer)
            for got, want in zip(a, b):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
            assert actual.get_seq_length(layer) == expected.get_seq_length(layer)
        assert actual._seen_tokens == expected._seen_tokens
    assert actual.get_max_cache_shape() is None
    assert len(actual) == 3
    assert list(actual)[1][0] is actual.key_cache[1]
    assert actual[2][1] is actual.value_cache[2]


def test_storage_reused_until_geometric_growth(monkeypatch):
    cache = CPUCache()
    for layer in range(2):
        cache.update(*states(3, layer), layer)

    def no_cat(*args, **kwargs):
        raise AssertionError("Cache append must not call torch.cat")

    monkeypatch.setattr(torch, "cat", no_cat)
    for layer in range(2):
        original = cache.key_cache[layer].untyped_storage().data_ptr()
        cache.update(*states(1, 10), layer)
        grown = cache.key_cache[layer].untyped_storage().data_ptr()
        value_ptr = cache.value_cache[layer].untyped_storage().data_ptr()
        assert grown != original
        assert cache._key_buffers[layer].shape[-2] == 6
        for _ in range(2):
            cache.update(*states(1, 20), layer)
            assert cache.key_cache[layer].untyped_storage().data_ptr() == grown
            assert cache.value_cache[layer].untyped_storage().data_ptr() == value_ptr
        cache.update(*states(1, 30), layer)
        assert cache._key_buffers[layer].shape[-2] == 12
        assert cache.key_cache[layer].untyped_storage().data_ptr() != grown


def test_skipped_layers_and_legacy_roundtrip():
    cache = CPUCache()
    assert cache.get_seq_length() == 0
    cache.update(*states(3), 2)
    assert cache.get_seq_length(0) == 0
    assert cache.get_seq_length(1) == 0
    assert cache.get_seq_length(2) == 3
    cache.update(*states(3), 0)
    cache.update(*states(3), 1)
    restored = CPUCache.from_legacy_cache(cache.to_legacy_cache())
    assert isinstance(restored, CPUCache)
    assert restored._seen_tokens == 3
    for a, b in zip(cache, restored):
        for got, want in zip(a, b):
            torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("crop_length", [2, -2, 0])
def test_crop_then_append(crop_length):
    actual, expected = CPUCache(), DynamicCache()
    for cache in (actual, expected):
        for layer in range(2):
            cache.update(*states(4), layer)
        cache.crop(crop_length)
        for layer in range(2):
            cache.update(*states(2, 100), layer)
    assert actual._seen_tokens == expected._seen_tokens
    for a, b in zip(actual, expected):
        for got, want in zip(a, b):
            torch.testing.assert_close(got, want, rtol=0, atol=0)


def test_single_request_only():
    cache = CPUCache()
    keys, values = states(2)
    with pytest.raises(ValueError, match="single request"):
        cache.update(keys.repeat(2, 1, 1, 1), values.repeat(2, 1, 1, 1), 0)
    with pytest.raises(NotImplementedError, match="beam"):
        cache.reorder_cache(torch.tensor([0]))
