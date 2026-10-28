"""Keep SAM's logical BHWC transformer activations physically contiguous."""
from functools import wraps


def install_sam_layout(model):
    """Break inherited NCHW strides once instead of copying before every norm.

    No changed image size, attention domain, quantization, or arithmetic.
    Accept the SAM encoder itself or the full OCR model.
    """
    sam = model.model.sam_model if hasattr(model, "model") else model
    if getattr(sam.patch_embed, "_cpu_contiguous_bhwc", False):
        return
    original = sam.patch_embed.forward

    @wraps(original)
    def forward(x):
        return original(x).contiguous()

    sam.patch_embed.forward = forward
    sam.patch_embed._cpu_contiguous_bhwc = True
