"""Stage-specific thread counts for a single-request CPU process.

PyTorch thread count is process-global. Do not use this with concurrent model
requests or independent workloads in the same process.
"""
from functools import wraps

import torch


def install_stage_threads(model, vision=3, prefill=3, decode=3):
    if min(vision, prefill, decode) < 1:
        raise ValueError("Thread counts must be positive")
    model._cpu_stage_threads = dict(vision=vision, prefill=prefill, decode=decode)
    if getattr(model, "_cpu_threads_installed", False):
        return

    original = model.model.forward

    @wraps(original)
    def forward(*args, **kwargs):
        ids = kwargs.get("input_ids")
        count = model._cpu_stage_threads["decode" if ids is not None and ids.shape[1] == 1 else "prefill"]
        if torch.get_num_threads() != count:
            torch.set_num_threads(count)
        return original(*args, **kwargs)

    model.model.forward = forward

    def wrap_vision(module):
        original = module.forward

        @wraps(original)
        def forward(*args, **kwargs):
            previous = torch.get_num_threads()
            desired = model._cpu_stage_threads["vision"]
            if previous != desired:
                torch.set_num_threads(desired)
            try:
                return original(*args, **kwargs)
            finally:
                if previous != desired:
                    torch.set_num_threads(previous)
        module.forward = forward

    for module in (model.model.sam_model, model.model.vision_model):
        wrap_vision(module)
    model._cpu_threads_installed = True
