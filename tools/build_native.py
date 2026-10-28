"""Build baseline-safe libraries containing explicitly targeted SIMD helpers."""

from pathlib import Path
from importlib.util import find_spec
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    sources = root / "native"
    output = root / "unlimited_ocr"
    for name in ("native", "native_attention"):
        subprocess.run(
            [
                "g++", "-O3", "-std=c++17", "-shared", "-fPIC",
                str(sources / f"{name}.cpp"), "-o", str(output / f"{name}.so"),
            ],
            check=True,
        )
    torch_root = Path(next(iter(find_spec("torch").submodule_search_locations)))
    subprocess.run([
        "g++", "-O2", "-std=c++20", "-shared", "-fPIC",
        str(sources / "isa_probe.cpp"), "-I", str(torch_root / "include"),
        "-L", str(torch_root / "lib"), "-ltorch_cpu",
        f"-Wl,-rpath,{torch_root / 'lib'}", "-o", str(output / "isa_probe.so"),
    ], check=True)
    import torch
    subprocess.run([
        "g++", "-O3", "-std=c++20", "-shared", "-fPIC", "-fopenmp",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        str(sources / "linear.cpp"), "-I", str(torch_root / "include"),
        "-I", str(torch_root / "include/torch/csrc/api/include"),
        "-L", str(torch_root / "lib"), "-ltorch_cpu", "-ltorch", "-lc10",
        f"-Wl,-rpath,{torch_root / 'lib'}", "-o", str(output / "linear.so"),
    ], check=True)


if __name__ == "__main__":
    main()
