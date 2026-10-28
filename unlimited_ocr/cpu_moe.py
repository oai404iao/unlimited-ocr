"""Instance-local CPU decode dispatch and optional packed INT8 MLP fusion."""
from types import MethodType

import torch
from torch.ao.nn.quantized.dynamic import Linear as Int8Linear


@torch.no_grad()
def _moe_infer(self, x, topk_ids, topk_weight):
    if x.shape[0] != 1 or x.device.type != "cpu" or self.ep_size != 1:
        return self._cpu_original_moe_infer(x, topk_ids, topk_weight)
    # Keep routing order and the upstream multiply/sum (including dtype casts),
    # rather than accumulating weighted experts one at a time.
    outputs = torch.cat([self.experts[i](x) for i in topk_ids[0].tolist()], dim=0)
    return (outputs.view(*topk_ids.shape, -1).to(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(-1)).sum(dim=1).to(outputs.dtype))


def _fused_mlp(self, x):
    gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
    return self.down_proj(self.act_fn(gate) * up)


def _fuse_mlp(module):
    gate, up = module.gate_proj, module.up_proj
    if not isinstance(gate, Int8Linear) or not isinstance(up, Int8Linear):
        return
    gate_weight, up_weight = gate.weight(), up.weight()
    if (gate_weight.qscheme() != torch.per_channel_affine
            or up_weight.qscheme() != torch.per_channel_affine
            or gate_weight.q_per_channel_axis() != 0
            or up_weight.q_per_channel_axis() != 0):
        raise ValueError("Gate/up fusion requires per-output-channel INT8 weights")
    # Concatenate integer storage and quantization parameters, never dequantize
    # or requantize. Both dynamic projections see the same activation range.
    weight = torch._make_per_channel_quantized_tensor(
        torch.cat((gate_weight.int_repr(), up_weight.int_repr())),
        torch.cat((gate_weight.q_per_channel_scales(), up_weight.q_per_channel_scales())),
        torch.cat((gate_weight.q_per_channel_zero_points(), up_weight.q_per_channel_zero_points())),
        0,
    )
    gate_bias, up_bias = gate.bias(), up.bias()
    bias = None
    if gate_bias is not None or up_bias is not None:
        bias = torch.cat((gate_bias if gate_bias is not None else torch.zeros(gate.out_features),
                          up_bias if up_bias is not None else torch.zeros(up.out_features)))
    fused = Int8Linear(gate.in_features, gate.out_features + up.out_features, bias is not None)
    fused.set_weight_bias(weight, bias)
    fused.train(module.training)
    module.gate_up_proj = fused
    del module.gate_proj, module.up_proj
    module.forward = MethodType(_fused_mlp, module)


def install_moe(model, fuse=False):
    """Patch single-token CPU inference; return the number of MoE modules.

    Prefill/distributed dispatch keeps the original method, also available as
    ``module._cpu_original_moe_infer`` for A/B measurements. ``fuse=True``
    additionally replaces packed gate/up projections in all DeepseekV2MLPs.
    Fusion is one-way (reload to undo), releasing old weights one MLP at a time.
    Repeated installation is safe. Intended for eval-mode CPU inference.
    """
    from upstream._sparse_base_impl import DeepseekV2MLP, DeepseekV2MoE

    count = 0
    for module in model.modules():
        if isinstance(module, DeepseekV2MoE):
            if not hasattr(module, "_cpu_original_moe_infer"):
                module._cpu_original_moe_infer = module.moe_infer
                module.moe_infer = MethodType(_moe_infer, module)
            count += 1
        if fuse and isinstance(module, DeepseekV2MLP) and not hasattr(module, "gate_up_proj"):
            _fuse_mlp(module)
    return count
