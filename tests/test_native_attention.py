"""Build native_attention.so before running these CPU numerical tests."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from unlimited_ocr.cpu_attention import _decode_sparse, _decode_sdpa, install_attention
from unlimited_ocr.cpu_native_attention import decode_native, install_native_attention


def case(length=83, width=13, kv_heads=2, strided=False):
    torch.manual_seed(length + width)
    q = torch.randn(2, 4, 3, width)
    k = torch.randn(2, kv_heads, length + 17, width)
    v = torch.randn(2, kv_heads, length + 17, width + 2)
    k, v = k[:, :, :length], v[:, :, :length]
    if strided:
        q = torch.randn(2, 3, 4, width * 2).transpose(1, 2)[..., ::2]
        k = torch.randn(2, length + 17, kv_heads, width * 2).transpose(1, 2)[:, :, :length, ::2]
        v = torch.randn(2, length + 17, kv_heads, (width + 2) * 2).transpose(1, 2)[:, :, :length, ::2]
    topi = torch.randint((length + 7) // 8, (2, 3, 7))
    topi[..., 1] = topi[..., 0]
    topi[..., -1] = (length + 7) // 8 - 1
    bias = torch.randn(2, 3, 7) * 2
    bias[1, 0, 2] = -5e8
    valid = torch.rand(2, 3, 7) > .2
    valid[0, :, 1] = False
    positions = torch.tensor([[0, length // 2, length - 1], [1, length // 3, length - 1]])
    return (q, k, v, topi, bias, valid, positions, 8, 11, width ** -.5)


@pytest.mark.parametrize('length', [1, 7, 32, 83, 300])
@pytest.mark.parametrize('width', [7, 13, 128])
@pytest.mark.parametrize('kv_heads', [1, 2, 4])
@pytest.mark.parametrize('strided', [False, True])
def test_matches_sparse(length, width, kv_heads, strided):
    args = case(length, width, kv_heads, strided)
    torch.testing.assert_close(decode_native(*args), _decode_sparse(*args),
                               atol=3e-6, rtol=3e-5, equal_nan=True)


@pytest.mark.parametrize('strided', [False, True])
def test_sdpa_reuse(strided):
    args = case(strided=strided)
    torch.testing.assert_close(_decode_sdpa(*args), _decode_sparse(*args), atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize('index,value', [(7, 2**64), (8, 2**64), (7, 0), (8, -1)])
def test_scalar_integer_bounds(index, value):
    args = list(case())
    args[index] = value
    with pytest.raises(ValueError):
        decode_native(*args)


def test_extreme_positions_do_not_overflow():
    args = list(case())
    args[6].fill_(torch.iinfo(torch.int64).max)
    # With a future position and finite local window, only remote keys remain.
    actual = decode_native(*args)
    args[6].fill_(10000)
    torch.testing.assert_close(actual, decode_native(*args), equal_nan=True)
    args[6].fill_(torch.iinfo(torch.int64).min)
    assert decode_native(*args).isnan().all()


@pytest.mark.parametrize('window', [0, 1, 19])
def test_duplicate_invalid_last_and_local_precedence(window):
    q = torch.zeros(1, 2, 2, 9)
    k = torch.zeros(1, 1, 19, 9)
    v = torch.arange(19.).view(1, 1, 19, 1)
    # Invalid last duplicate suppresses block 0; block 2 is future at row 0.
    topi = torch.tensor([[[0, 0, 1, 2], [0, 0, 1, 2]]])
    bias = torch.tensor([[[9., 2., -1., 20.], [9., 2., -1., 20.]]])
    valid = torch.tensor([[[True, False, True, True], [True, True, True, True]]])
    args = (q, k, v, topi, bias, valid, torch.tensor([[5, 18]]), 8, window, 1.)
    torch.testing.assert_close(decode_native(*args), _decode_sparse(*args), equal_nan=True)


@pytest.mark.parametrize('length', [0, 20])
def test_empty_read_set(length):
    args = (torch.zeros(1, 2, 1, 9), torch.zeros(1, 1, length, 9),
            torch.zeros(1, 1, length, 7), torch.empty(1, 1, 0, dtype=torch.int64),
            torch.empty(1, 1, 0), torch.empty(1, 1, 0, dtype=torch.bool),
            torch.zeros(1, 1, dtype=torch.int64), 8, 0, 1.)
    assert decode_native(*args).isnan().all()


def test_noncontiguous_routes_and_expanded_positions():
    args = list(case())
    args[3:6] = [x.transpose(0, 1).contiguous().transpose(0, 1) for x in args[3:6]]
    args[6] = args[6][:1].expand(2, -1)
    torch.testing.assert_close(decode_native(*args), _decode_sparse(*args), atol=3e-6, rtol=3e-5)


def test_concurrent_calls():
    args = case()
    expected = _decode_sparse(*args)
    with ThreadPoolExecutor(max_workers=3) as pool:
        outputs = list(pool.map(lambda _: decode_native(*args), range(12)))
    for actual in outputs:
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)


def test_install_only_replaces_decode():
    module = SimpleNamespace(DashAttention=type('DashAttention', (), {}))
    install_attention(module)
    forward = module.DashAttention._forward_dense
    install_native_attention(module)
    assert module._decode_attend is decode_native
    assert module.DashAttention._forward_dense is forward


def test_validation_and_no_thread_change():
    args = list(case())
    threads = torch.get_num_threads()
    decode_native(*args)
    assert torch.get_num_threads() == threads
    args[0] = args[0].to(torch.bfloat16)
    with pytest.raises(ValueError, match='FP32'):
        decode_native(*args)
    args = list(case())
    args[1] = args[1][:, :, :, :-1]
    with pytest.raises(ValueError, match='shape'):
        decode_native(*args)
