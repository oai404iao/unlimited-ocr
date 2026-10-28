"""Compare mature packed dynamic INT8 backends, same per-channel weights."""
import json
from pathlib import Path

from unlimited_ocr.cpu_runtime import configure_cpu, pack_linear
import torch
from torch.utils.benchmark import Timer
from torch.ao.nn.quantized.dynamic import Linear


def main():
    configure_cpu(3)
    torch.manual_seed(42)
    results = []
    with torch.inference_mode():
        for rows, columns, output, threads in ((4096, 768, 2304, 6), (4096, 3072, 768, 6), (1, 1280, 896, 3)):
            x = torch.randn(rows, columns)
            weight, bias = torch.randn(output, columns) * .02, torch.randn(output) * .01
            torch.backends.quantized.engine = "fbgemm"
            original = pack_linear(weight, bias)
            expected = original(x)
            for backend in ("fbgemm", "onednn", "x86"):
                torch.backends.quantized.engine = backend
                layer = Linear(columns, output, True).eval()
                layer.set_weight_bias(original.weight(), original.bias())
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
