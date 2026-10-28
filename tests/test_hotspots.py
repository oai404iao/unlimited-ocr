import pytest
import torch

from unlimited_ocr.cpu_sam_layout import install_sam_layout
from upstream.deepencoder import ImageEncoderViT
from upstream._sparse_base_impl import DeepseekV2MoE
from tests.test_moe import config, quantize


@pytest.mark.parametrize("tokens", [1, 5])
def test_scripted_moe_matches_packed_reference(tokens):
    from unlimited_ocr.cpu_moe_script import install_scripted_moe
    torch.manual_seed(4)
    model = quantize(DeepseekV2MoE(config()))
    x = torch.randn(1, tokens, 32)
    with torch.inference_mode():
        reference = model(x)
        assert install_scripted_moe(model) == 1
        assert install_scripted_moe(model) == 0
        torch.testing.assert_close(model(x), reference, atol=0, rtol=0)


def test_scripted_moe_shares_weights_and_preserves_fp64_reduction():
    from unlimited_ocr.cpu_moe_script import install_scripted_moe
    model = quantize(DeepseekV2MoE(config()))
    x = torch.randn(1, 32)
    ids = torch.tensor([[7, 0, 3]])
    weights = torch.tensor([[.2, .3, .5]], dtype=torch.float64)
    with torch.inference_mode():
        reference = model.moe_infer(x, ids, weights)
        install_scripted_moe(model)
        assert torch._C._is_alias_of(model.cpu_script_experts.gates[0],
                                    model.experts[0].gate_proj._packed_params._packed_params)
        torch.testing.assert_close(model.moe_infer(x, ids, weights), reference, atol=0, rtol=0)


def test_sam_layout_preserves_values_and_removes_norm_copies():
    torch.manual_seed(9)
    model = ImageEncoderViT(img_size=64, patch_size=8, embed_dim=24, depth=2,
                           num_heads=3, mlp_ratio=2, use_rel_pos=True,
                           window_size=4, global_attn_indexes=(1,)).eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.inference_mode():
        reference = model(x)
        install_sam_layout(model)
        install_sam_layout(model)
        contiguous = []
        hooks = [block.norm1.register_forward_pre_hook(
            lambda module, args: contiguous.append(args[0].is_contiguous())) for block in model.blocks]
        actual = model(x)
        for hook in hooks:
            hook.remove()
        assert all(contiguous)
        torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
