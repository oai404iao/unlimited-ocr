import torch

from unlimited_ocr.cpu_sam_layout import install_sam_layout
from upstream.deepencoder import ImageEncoderViT


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
