"""Serial old gather vs mature PyTorch SDPA vs direct-read native comparison."""
import json
from pathlib import Path

from unlimited_ocr.cpu_runtime import configure_cpu
import torch
from torch.utils.benchmark import Timer
from unlimited_ocr.cpu_attention import _decode_sparse, _decode_sdpa
from unlimited_ocr.cpu_native_attention import decode_native


def main():
    configure_cpu(3)
    torch.manual_seed(42)
    results = []
    for length in (300, 1600, 8192):
        q = torch.randn(1, 10, 1, 128)
        k = torch.randn(1, 10, length + 20, 128)[:, :, :length]
        v = torch.randn_like(k)
        topi = torch.randperm((length + 31) // 32)[:16][None, None]
        bias = torch.randn_like(topi, dtype=torch.float32)
        valid = torch.ones_like(topi, dtype=torch.bool)
        args = (q, k, v, topi, bias, valid, torch.tensor([[length - 1]]), 32, 128, 128 ** -.5)
        reference = _decode_sparse(*args)
        result = dict(S=length)
        for function in (_decode_sparse, _decode_sdpa, decode_native):
            torch.testing.assert_close(function(*args), reference, rtol=2e-5, atol=2e-6)
            result[function.__name__ + "_us"] = Timer(
                "fn(*args)", globals={"fn": function, "args": args}, num_threads=3
            ).blocked_autorange(min_run_time=.5).median * 1e6
        results.append(result)
    path = Path("results/decode-kernels-serial.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
