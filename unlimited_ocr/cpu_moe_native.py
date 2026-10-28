"""Single-token selected experts via torch.library; weights/handles are shared."""
from types import MethodType

from unlimited_ocr.cpu_linear import CPUInt8Linear, native_available
import torch


class NativeExperts(torch.nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.gates = [expert.gate_proj._packed for expert in experts]
        self.ups = [expert.up_proj._packed for expert in experts]
        self.downs = [expert.down_proj._packed for expert in experts]
        self._packed = torch.classes.unlimited_ocr.PackedExperts(self.gates, self.ups, self.downs)

    def forward(self, x, ids, weights):
        return torch.ops.unlimited_ocr.moe(x, ids, weights, self._packed)


def _infer(self, x, ids, weights):
    if x.shape[0] != 1 or x.dtype != torch.float32 or x.device.type != "cpu":
        return self._cpu_native_original(x, ids, weights)
    return self.cpu_native_experts(x, ids, weights)


def install_native_moe(model):
    """Install after loading final weights; prefill and generic paths stay original.

    Returns the number installed. Packed handles are references, not serialized
    weights; do not mutate/reload the model's weights after installing dispatch.
    """
    if not native_available():
        return 0
    from upstream._sparse_base_impl import DeepseekV2MoE

    count = 0
    for module in model.modules():
        if not isinstance(module, DeepseekV2MoE) or module.ep_size != 1:
            continue
        if hasattr(module, "_cpu_native_original"):
            continue
        if not all(isinstance(getattr(expert, name, None), CPUInt8Linear)
                   and getattr(expert, name)._packed is not None
                   for expert in module.experts for name in ("gate_proj", "up_proj", "down_proj")):
            continue
        module.cpu_native_experts = NativeExperts(module.experts)
        module._cpu_native_original = module.moe_infer
        module.moe_infer = MethodType(_infer, module)
        count += 1
    return count
