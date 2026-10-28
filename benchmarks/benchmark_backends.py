"""Compare the selected native backend with FP32 reference paths."""
import json
from pathlib import Path

from unlimited_ocr.cpu_runtime import configure_cpu, pack_linear
import torch
from torch.utils.benchmark import Timer
from unlimited_ocr.cpu_linear import backend_name


def main():
    configure_cpu(3)
    torch.manual_seed(42)
    results = []
    with torch.inference_mode():
        for rows, columns, output, threads in ((4096, 768, 2304, 6), (4096, 3072, 768, 6), (1, 1280, 896, 3)):
            x = torch.randn(rows, columns)
            weight, bias = torch.randn(output, columns) * .02, torch.randn(output) * .01
            original = pack_linear(weight, bias)
            expected = torch.nn.functional.linear(x, weight, bias)
            def dequantized(value):
                return torch.nn.functional.linear(value, original.weight.float() * original.scales[:, None], original.bias)
            def reference(value):
                return torch.nn.functional.linear(value, weight, bias)
            for backend, layer in ((backend_name(), original), ("fp32_dequantize", dequantized), ("fp32", reference)):
                actual = layer(x)
                seconds = Timer("layer(x)", globals={"layer": layer, "x": x}, num_threads=threads
                                ).blocked_autorange(min_run_time=.5).median
                results.append(dict(shape=[rows, columns, output], threads=threads, backend=backend,
                                    milliseconds=seconds * 1000,
                                    max_abs_difference=float((expected - actual).abs().max()),
                                    relative_rmse=float((expected - actual).square().mean().sqrt() / expected.square().mean().sqrt())))
    path = Path("results/linear-backends.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    print(path.read_text())


if __name__ == "__main__":
    main()
