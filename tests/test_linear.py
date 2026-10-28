import pytest

from unlimited_ocr import cpu_linear
import torch


def legacy_linear(weight, bias=None):
    # Deliberately test-only: independent reference for the deprecated backend.
    from torch.ao.nn.quantized.dynamic import Linear
    weight = weight.float()
    scales = weight.abs().amax(dim=1).clamp_min(1e-8) / 127
    quantized = torch.quantize_per_channel(weight, scales, torch.zeros(weight.shape[0], dtype=torch.int64),
                                         0, torch.qint8)
    result = Linear(weight.shape[1], weight.shape[0], bias is not None)
    result.set_weight_bias(quantized, None if bias is None else bias.float())
    return result


@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("dims", [(8, 6), (1280, 896)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_fbgemm_numerical_parity(batch, dims, with_bias):
    if not cpu_linear.native_available():
        pytest.skip("Build native INT8 kernel on an AVX2-capable CPU")
    torch.backends.quantized.engine = "fbgemm"
    torch.manual_seed(7)
    k, n = dims
    weight = torch.randn(n, k)
    bias = torch.randn(n) if with_bias else None
    expected = legacy_linear(weight, bias)
    actual = cpu_linear.CPUInt8Linear.from_float(weight, bias)
    torch.testing.assert_close(actual.weight, expected.weight().int_repr(), rtol=0, atol=0)
    torch.testing.assert_close(actual.scales.double(), expected.weight().q_per_channel_scales(), rtol=0, atol=0)
    x = torch.randn(batch, k)
    torch.testing.assert_close(actual(x), expected(x), rtol=0, atol=0)


def test_plain_state_roundtrip_rebuilds_handle():
    torch.manual_seed(12)
    model = cpu_linear.CPUInt8Linear.from_float(torch.randn(6, 8), torch.randn(6))
    other = cpu_linear.CPUInt8Linear.from_float(torch.zeros(6, 8), torch.zeros(6))
    old = other._packed
    state = model.state_dict()
    assert set(state) == {"weight", "scales", "bias"}
    assert all(not tensor.is_quantized for tensor in state.values())
    other.load_state_dict(state, assign=True)
    if old is not None:
        assert other._packed is not old
    x = torch.randn(3, 8)
    torch.testing.assert_close(other(x), model(x), rtol=0, atol=0)


@pytest.mark.parametrize("magnitude,offset", [(0., 0.), (1e-8, 0.), (1e-8, 1e-7), (1e-8, -1e-7), (1., 2.)])
def test_activation_quantization_edge_cases(magnitude, offset):
    if not cpu_linear.native_available():
        pytest.skip("Native FBGEMM unavailable")
    torch.backends.quantized.engine = "fbgemm"
    weight = torch.tensor([[-127., -126.5, -.5, 0., .5, 1.5, 127.], [0.] * 7])
    old = legacy_linear(weight)
    new = cpu_linear.CPUInt8Linear.from_float(weight)
    torch.testing.assert_close(new.weight, old.weight().int_repr(), rtol=0, atol=0)
    x = torch.randn(3, 7) * magnitude + offset
    torch.testing.assert_close(new(x), old(x), rtol=0, atol=0)


def test_generic_no_packing(monkeypatch):
    monkeypatch.setattr(cpu_linear, "native_available", lambda: False)
    model = cpu_linear.CPUInt8Linear.from_float(torch.randn(6, 8), torch.randn(6))
    assert model._packed is None
    x = torch.randn(2, 8)
    expected = torch.nn.functional.linear(x, model.weight.float() * model.scales[:, None], model.bias)
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)


def test_module_transforms_rebuild_or_reject_dtype():
    model = cpu_linear.CPUInt8Linear.from_float(torch.randn(6, 8))
    old = model._packed
    assert model.cpu() is model
    if old is not None:
        assert model._packed is not old
    with pytest.raises((RuntimeError, ValueError), match="scales"):
        model.double()
    assert model._packed is None


@pytest.mark.parametrize("bad", ["weight-dtype", "weight-rank", "scale-shape", "scale-zero", "bias-shape"])
def test_bad_parameters_raise(bad):
    weight, scales, bias = torch.zeros(6, 8, dtype=torch.int8), torch.ones(6), None
    if bad == "weight-dtype":
        weight = weight.float()
    elif bad == "weight-rank":
        weight = weight.flatten()
    elif bad == "scale-shape":
        scales = torch.ones(8)
    elif bad == "scale-zero":
        scales.zero_()
    else:
        bias = torch.zeros(7)
    with pytest.raises((RuntimeError, ValueError)):
        cpu_linear.CPUInt8Linear(weight, scales, bias)


def test_input_shapes_and_guard():
    model = cpu_linear.CPUInt8Linear.from_float(torch.randn(6, 8))
    assert model(torch.randn(8)).shape == (6,)
    assert model(torch.randn(2, 3, 8)).shape == (2, 3, 6)
    assert model(torch.empty(0, 8)).shape == (0, 6)
    x = torch.randn(8, 4).t()
    torch.testing.assert_close(model(x), model(x.contiguous()), rtol=0, atol=0)
    for bad in (torch.randn(2, 7), torch.ones(2, 8, dtype=torch.float64), torch.tensor(1.0)):
        with pytest.raises((RuntimeError, ValueError)):
            model(bad)
