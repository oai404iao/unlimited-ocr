from types import SimpleNamespace

import pytest
import torch

from unlimited_ocr.cpu_profile import StageProfiler
from unlimited_ocr.cpu_threads import install_stage_threads


def test_stage_threads_and_exception_restoration():
    old = torch.get_num_threads()
    counts = []
    def vision(*args, **kwargs):
        counts.append(torch.get_num_threads())
        if kwargs.get("fail"):
            raise RuntimeError("test")
    sam = SimpleNamespace(forward=vision)
    clip = SimpleNamespace(forward=vision)
    def forward(input_ids):
        counts.append(torch.get_num_threads())
        if input_ids.shape[1] > 1:
            sam.forward()
            clip.forward()
        counts.append(torch.get_num_threads())
    model = SimpleNamespace(model=SimpleNamespace(forward=forward, sam_model=sam, vision_model=clip))
    try:
        install_stage_threads(model, vision=2, prefill=3, decode=1)
        install_stage_threads(model, vision=2, prefill=3, decode=1)
        model.model.forward(input_ids=torch.zeros(1, 5))
        assert counts == [3, 2, 2, 3]
        counts.clear()
        model.model.forward(input_ids=torch.zeros(1, 1))
        assert counts == [1, 1]
        with pytest.raises(RuntimeError):
            sam.forward(fail=True)
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(old)


def test_profile_nested_attribution_and_restore():
    inner = SimpleNamespace(forward=lambda: 42)
    outer = SimpleNamespace(forward=lambda: inner.forward())
    original = outer.forward
    profiler = StageProfiler(None)
    profiler.wrap(outer, "forward", "outer")
    profiler.wrap(inner, "forward", "inner")
    assert outer.forward() == 42
    result = profiler.report()
    assert result["prefill/outer"]["calls"] == 1
    assert result["prefill/outer"]["inclusive_seconds"] >= result["prefill/inner"]["inclusive_seconds"]
    assert result["prefill/outer"]["exclusive_seconds"] >= 0
    profiler.__exit__()
    assert outer.forward is original
