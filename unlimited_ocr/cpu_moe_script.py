"""Reuse TorchScript dispatch and existing FBGEMM packed operators, no new GEMM.

Inference-only single-token experts. Prefill keeps upstream grouping. Packed
objects are shared references, not copied/requantized weight tensors.
"""
from typing import List
from types import MethodType

import torch
from torch.ao.nn.quantized.dynamic import Linear as Int8Linear


@torch.jit.script
def _experts(x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor,
             gates: List[torch.classes.quantized.LinearPackedParamsBase],
             ups: List[torch.classes.quantized.LinearPackedParamsBase],
             downs: List[torch.classes.quantized.LinearPackedParamsBase]):
    outputs = torch.jit.annotate(List[torch.Tensor], [])
    for slot in range(ids.size(1)):
        expert = int(ids[0, slot])
        gate = torch.ops.quantized.linear_dynamic(x, gates[expert], True)
        up = torch.ops.quantized.linear_dynamic(x, ups[expert], True)
        output = torch.ops.quantized.linear_dynamic(torch.nn.functional.silu(gate) * up,
                                                     downs[expert], True)
        outputs.append(output)
    values = torch.cat(outputs, dim=0)
    return (values.view(1, ids.size(1), -1).to(weights.dtype)
            .mul_(weights.unsqueeze(-1)).sum(dim=1).to(x.dtype))


class ScriptExperts(torch.nn.Module):
    gates: List[torch.classes.quantized.LinearPackedParamsBase]
    ups: List[torch.classes.quantized.LinearPackedParamsBase]
    downs: List[torch.classes.quantized.LinearPackedParamsBase]

    def __init__(self, experts):
        super().__init__()
        self.gates = [e.gate_proj._packed_params._packed_params for e in experts]
        self.ups = [e.up_proj._packed_params._packed_params for e in experts]
        self.downs = [e.down_proj._packed_params._packed_params for e in experts]

    def forward(self, x: torch.Tensor, ids: torch.Tensor, weights: torch.Tensor):
        return _experts(x, ids, weights, self.gates, self.ups, self.downs)


def _infer(self, x, ids, weights):
    if x.shape[0] != 1 or x.dtype != torch.float32 or x.device.type != "cpu":
        return self._cpu_script_original(x, ids, weights)
    return self.cpu_script_experts(x, ids, weights)


def install_scripted_moe(model):
    from upstream._sparse_base_impl import DeepseekV2MoE
    count = 0
    for module in model.modules():
        if not isinstance(module, DeepseekV2MoE) or module.ep_size != 1:
            continue
        if hasattr(module, "_cpu_script_original"):
            continue
        if not all(isinstance(getattr(expert, name, None), Int8Linear)
                   for expert in module.experts for name in ("gate_proj", "up_proj", "down_proj")):
            raise ValueError("Scripted MoE requires unfused dynamic INT8 expert projections")
        module.cpu_script_experts = torch.jit.script(ScriptExperts(module.experts))
        module._cpu_script_original = module.moe_infer
        module.moe_infer = MethodType(_infer, module)
        count += 1
    return count
