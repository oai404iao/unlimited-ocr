"""Export a self-contained offline INT8 model directory."""
import argparse
import time

from unlimited_ocr.cpu_runtime import configure_cpu
from unlimited_ocr.cpu_checkpoint import export_quantized_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="models/unlimited-ocr-2")
    parser.add_argument("--output", required=True, help="New directory; existing directories are not overwritten")
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    configure_cpu(args.threads)
    started = time.perf_counter()
    export_quantized_model(args.source, args.output)
    print(f"Exported {args.output}; elapsed={time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
