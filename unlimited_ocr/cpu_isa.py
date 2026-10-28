"""Linux CPU ISA policy, evaluated before importing torch.

UOCR_CPU_ISA: auto (tuned for the measured CPUs), avx2, avx512 (base),
vnni256, vnni512, or generic (PyTorch reference path).
Unsupported requests downgrade safely.
Raw library dispatch environment variables are owned by this policy.
"""
import os
from pathlib import Path
import warnings


def cpu_flags():
    try:
        lines = Path("/proc/cpuinfo").read_text().splitlines()
    except OSError:
        return set()
    flags = [set(line.split(":", 1)[1].split()) for line in lines if line.startswith("flags")]
    return set.intersection(*flags) if flags else set()


def select_isa(requested, flags):
    if requested not in ("auto", "avx2", "avx512", "vnni256", "vnni512", "generic"):
        raise ValueError("UOCR_CPU_ISA must be auto, avx2, avx512, vnni256, vnni512 or generic")
    avx2 = {"avx2", "fma"} <= flags
    avx512 = avx2 and {"avx512f", "avx512bw", "avx512dq", "avx512vl"} <= flags
    selected = "avx512" if avx512 else "avx2" if avx2 else "generic"
    if requested == "generic":
        selected = "generic"
    elif requested == "avx2" and avx2:
        selected = "avx2"
    has_vnni = avx512 and "avx512_vnni" in flags
    vnni = requested in ("auto", "vnni256", "vnni512") and selected == "avx512" and has_vnni
    bf16 = vnni and "avx512_bf16" in flags
    satisfied = (requested in ("auto", "generic") or requested == "avx2" and avx2
                 or requested == "avx512" and avx512 or requested.startswith("vnni") and has_vnni)
    return dict(requested=requested, selected=selected, vnni=vnni, bf16=bf16,
                downgraded=not satisfied,
                fbgemm=("AVX512_E1" if requested == "vnni512" else "AVX512_E1_256") if vnni else
                       "AVX512" if selected == "avx512" else "AVX2" if selected == "avx2" else None,
                onednn="AVX512_CORE_BF16" if bf16 else "AVX512_CORE_VNNI" if vnni else
                       "AVX512_CORE" if selected == "avx512" else "AVX2" if selected == "avx2" else "SSE41")


POLICY = select_isa(os.environ.get("UOCR_CPU_ISA", "auto").lower(), cpu_flags())
if POLICY["downgraded"]:
    warnings.warn(f"Requested {POLICY['requested']} unavailable; using {POLICY['selected']}")
os.environ["ATEN_CPU_CAPABILITY"] = "default" if POLICY["selected"] == "generic" else POLICY["selected"]
os.environ["ONEDNN_MAX_CPU_ISA"] = POLICY["onednn"]
os.environ["DNNL_MAX_CPU_ISA"] = POLICY["onednn"]
if POLICY["fbgemm"] is not None:
    os.environ["FBGEMM_ENABLE_INSTRUCTIONS"] = POLICY["fbgemm"]
else:
    os.environ.pop("FBGEMM_ENABLE_INSTRUCTIONS", None)


def isa_report():
    """Configured limits plus observable dispatch. Does not claim each op's ISA."""
    import ctypes
    import torch
    report = dict(POLICY, aten=torch.backends.cpu.get_cpu_capability())
    probe = Path(__file__).with_name("isa_probe.so")
    if probe.exists() and POLICY["selected"] != "generic":
        try:
            library = ctypes.CDLL(str(probe))
            library.fbgemm_selected_isa.restype = ctypes.c_char_p
            report["fbgemm_observed"] = library.fbgemm_selected_isa().decode()
        except (OSError, AttributeError) as error:
            report["probe_error"] = str(error)
    return report
