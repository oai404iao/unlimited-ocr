from types import SimpleNamespace

import pytest
import torch

from unlimited_ocr.cpu_moe import install_moe
from unlimited_ocr.cpu_runtime import pack_linear
from upstream._sparse_base_impl import DeepseekV2MLP, DeepseekV2MoE


def config():
    return SimpleNamespace(hidden_size=32, intermediate_size=48,
                           moe_intermediate_size=24, hidden_act="silu",
                           num_experts_per_tok=3, n_routed_experts=8,
                           n_shared_experts=1, routed_scaling_factor=1.0,
                           scoring_func="softmax", aux_loss_alpha=0.0,
                           seq_aux=False, topk_method="greedy", n_group=1,
                           topk_group=1, norm_topk_prob=True)


def quantize(model):
    torch.backends.quantized.engine = "fbgemm"
    for module in model.modules():
        if isinstance(module, DeepseekV2MLP):
            for name in ("gate_proj", "up_proj", "down_proj"):
                linear = getattr(module, name)
                setattr(module, name, pack_linear(linear.weight, linear.bias))
    return model.eval()


@pytest.mark.parametrize("fuse", [False, True])
@pytest.mark.parametrize("shape", [(1, 1, 32), (1, 7, 32), (2, 1, 32)])
def test_quantized_moe(shape, fuse):
    torch.manual_seed(41)
    model = quantize(DeepseekV2MoE(config()))
    x = torch.randn(shape)
    expected = model(x)
    assert install_moe(model, fuse=fuse) == 1
    assert install_moe(model, fuse=fuse) == 1
    torch.testing.assert_close(model(x), expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dispatch_order_and_selected_only(dtype):
    torch.manual_seed(42)
    model = quantize(DeepseekV2MoE(config()))
    x = torch.randn(1, 32)
    ids = torch.tensor([[7, 1, 5]])
    weights = torch.tensor([[0.1, 0.3, 0.6]], dtype=dtype)
    expected = model.moe_infer(x, ids, weights)
    calls = []
    hooks = [expert.register_forward_hook(lambda m, a, o, i=i: calls.append(i))
             for i, expert in enumerate(model.experts)]
    install_moe(model)
    torch.testing.assert_close(model.moe_infer(x, ids, weights), expected, atol=0, rtol=0)
    assert calls == [7, 1, 5]
    for hook in hooks:
        hook.remove()


def test_duplicate_selection():
    # Gate top-k never produces duplicates; unlike the upstream scatter-based
    # histogram, direct dispatch also has a well-defined result for duplicates.
    model = quantize(DeepseekV2MoE(config()))
    x = torch.randn(1, 32)
    ids = torch.tensor([[5, 1, 5]])
    weights = torch.tensor([[0.2, 0.3, 0.5]])
    outputs = torch.cat([model.experts[i](x) for i in ids[0]])
    expected = (outputs * weights[0, :, None]).sum(0, keepdim=True)
    install_moe(model)
    torch.testing.assert_close(model.moe_infer(x, ids, weights), expected, atol=0, rtol=0)


def test_prefill_uses_original():
    model = quantize(DeepseekV2MoE(config()))
    install_moe(model)
    calls = []
    original = model._cpu_original_moe_infer

    def record(*args):
        calls.append(True)
        return original(*args)

    model._cpu_original_moe_infer = record
    model(torch.randn(1, 4, 32))
    assert calls == [True]


def test_fusion_keeps_quantized_storage():
    model = quantize(DeepseekV2MLP(config()))
    gate, up = model.gate_proj.weight(), model.up_proj.weight()
    x = torch.randn(2, 5, 32)
    expected = model(x)
    install_moe(model, fuse=True)
    fused = model.gate_up_proj.weight()
    assert not hasattr(model, "gate_proj") and not hasattr(model, "up_proj")
    assert torch.equal(fused.int_repr(), torch.cat((gate.int_repr(), up.int_repr())))
    assert torch.equal(fused.q_per_channel_scales(),
                       torch.cat((gate.q_per_channel_scales(), up.q_per_channel_scales())))
    assert torch.equal(fused.q_per_channel_zero_points(),
                       torch.cat((gate.q_per_channel_zero_points(), up.q_per_channel_zero_points())))
    torch.testing.assert_close(model(x), expected, atol=0, rtol=0)
