"""Reproducible component timings; not a substitute for document accuracy tests."""
import argparse
import json
import time
from pathlib import Path

from unlimited_ocr.cpu_runtime import configure_cpu, pack_linear
from unlimited_ocr.cpu_norm import rmsnorm, backend_name
import torch


def median_time(fn, iterations=30):
    for _ in range(5):
        fn()
    values = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        values.append(time.perf_counter() - start)
    return sorted(values)[len(values) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--output", default="results/kernels.json")
    args = parser.parse_args()
    configure_cpu(args.threads)
    torch.manual_seed(42)
    result = {"threads": args.threads, "torch": torch.__version__,
              "capability": torch.backends.cpu.get_cpu_capability(), "linear": []}
    with torch.inference_mode():
        for rows, output in [(1, 896), (1, 6848), (1, 129280), (273, 896)]:
            x = torch.randn(rows, 1280)
            w = torch.randn(output, 1280) * 0.02
            packed = pack_linear(w)
            baseline = lambda: torch.nn.functional.linear(x, w)
            quantized = lambda: packed(x)
            reference, actual = baseline(), quantized()
            fp_time, int_time = median_time(baseline), median_time(quantized)
            result["linear"].append(dict(shape=[rows, 1280, output], fp32_ms=fp_time*1000,
                int8_ms=int_time*1000, speedup=fp_time/int_time,
                relative_rmse=float((reference-actual).square().mean().sqrt()/reference.square().mean().sqrt())))
            del w, packed, reference, actual
        x, w = torch.randn(1, 1, 1280), torch.randn(1280)
        ref = lambda: x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * w
        base, fused = median_time(ref), median_time(lambda: rmsnorm(x, w, 1e-6))
        result["rmsnorm"] = dict(torch_us=base*1e6, selected_us=fused*1e6,
                                 backend=backend_name(), speedup=base/fused)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
