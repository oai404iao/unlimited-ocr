import pytest

from unlimited_ocr import cpu_moe_native
from unlimited_ocr.cpu_linear import CPUInt8Linear, native_available
import torch
from upstream._sparse_base_impl import DeepseekV2MLP, DeepseekV2MoE
from tests.test_moe import config


@pytest.fixture
def model():
    if not native_available():
        pytest.skip("Native FBGEMM unavailable")
    torch.manual_seed(41)
    model = DeepseekV2MoE(config())
    for module in model.modules():
        if isinstance(module, DeepseekV2MLP):
            for name in ("gate_proj", "up_proj", "down_proj"):
                linear = getattr(module, name)
                setattr(module, name, CPUInt8Linear.from_float(linear.weight, linear.bias))
    return model.eval()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_dispatch_exact_order_and_shared_handles(model, dtype):
    x = torch.randn(1, 32)
    ids = torch.tensor([[7, 0, 3]])
    weights = torch.tensor([[.2, .3, .5]], dtype=dtype)
    before = model.state_dict()
    reference = model.moe_infer(x, ids, weights)
    assert cpu_moe_native.install_native_moe(model) == 1
    assert cpu_moe_native.install_native_moe(model) == 0
    assert model.cpu_native_experts.gates[0] is model.experts[0].gate_proj._packed
    assert not model.cpu_native_experts.state_dict()
    assert set(model.state_dict()) == set(before)
    torch.testing.assert_close(model.moe_infer(x, ids, weights), reference, rtol=0, atol=0)


def test_prefill_uses_original(model):
    x = torch.randn(1, 5, 32)
    expected = model(x)
    assert cpu_moe_native.install_native_moe(model) == 1
    original = model._cpu_native_original
    calls = []

    def record(*args):
        calls.append(True)
        return original(*args)

    model._cpu_native_original = record
    torch.testing.assert_close(model(x), expected, rtol=0, atol=0)
    assert calls == [True]


@pytest.mark.parametrize("ids", [torch.tensor([[5, 1, 5]]), torch.tensor([[5, 0, 1, 0, 5, 0]])[:, ::2]])
def test_duplicate_and_strided_ids(model, ids):
    x = torch.randn(1, 32)
    weights = torch.tensor([[.2, .3, .5]], dtype=torch.float64)
    outputs = torch.cat([model.experts[i](x) for i in ids[0].tolist()])
    expected = (outputs.view(1, 3, -1).to(weights.dtype)
                .mul_(weights.unsqueeze(-1)).sum(1).float())
    cpu_moe_native.install_native_moe(model)
    torch.testing.assert_close(model.moe_infer(x, ids, weights), expected, rtol=0, atol=0)


@pytest.mark.parametrize("bad", ["negative", "bounds", "dtype", "shape", "empty", "weights", "lists", "width"])
def test_native_operator_guards(model, bad):
    cpu_moe_native.install_native_moe(model)
    experts = model.cpu_native_experts
    x, ids, weights = torch.randn(1, 32), torch.tensor([[1, 2, 3]]), torch.ones(1, 3)
    gates, ups, downs = experts.gates, experts.ups, experts.downs
    if bad == "negative":
        ids[0, 0] = -1
    elif bad == "bounds":
        ids[0, 0] = 8
    elif bad == "dtype":
        ids = ids.int()
    elif bad == "shape":
        ids = ids.flatten()
    elif bad == "empty":
        ids = torch.empty(1, 0, dtype=torch.int64)
    elif bad == "weights":
        weights = weights.half()
    elif bad == "lists":
        downs = downs[:-1]
    else:
        ups = list(ups)
        ups[1] = CPUInt8Linear.from_float(torch.randn(25, 32))._packed
    with pytest.raises(RuntimeError):
        packed = torch.classes.unlimited_ocr.PackedExperts(gates, ups, downs)
        torch.ops.unlimited_ocr.moe(x, ids, weights, packed)


def test_unavailable_or_missing_handle_skips_install(model, monkeypatch):
    model.experts[0].gate_proj._packed = None
    assert cpu_moe_native.install_native_moe(model) == 0
    monkeypatch.setattr(cpu_moe_native, "native_available", lambda: False)
    assert cpu_moe_native.install_native_moe(model) == 0
    assert not hasattr(model, "_cpu_native_original")
