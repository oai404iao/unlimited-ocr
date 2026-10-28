"""CPU-only OCR CLI. Use --dense-attention for an identical-weight baseline."""
import argparse
import contextlib
import json
import resource
import time
from pathlib import Path

from unlimited_ocr.cpu_runtime import configure_cpu, load_model
import torch
from PIL import Image
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--model-dir", default="models/unlimited-ocr-2", help="Original BF16 or exported offline INT8 directory")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--vision-threads", type=int, default=6)
    parser.add_argument("--decode-threads", type=int, help="Defaults to --threads")
    parser.add_argument("--vision-chunk", type=int, default=256)
    parser.add_argument("--vision-bf16", action="store_true", help="Experimental native BF16 SAM; falls back when unsupported")
    parser.add_argument("--baseline", action="store_true", help="Gathered decode, chunk64, uniform threads, no native MoE")
    parser.add_argument("--optimize-moe", action="store_true", help="Experimental selected-expert dispatch and packed gate/up fusion")
    parser.add_argument("--profile", action="store_true", help="Add stage attribution (has measurement overhead)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--tiles", action="store_true", help="Enable upstream dynamic tiling (more expensive)")
    parser.add_argument("--dense-attention", action="store_true")
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--fp32", action="store_true", help="Unquantized baseline; needs >14 GiB RAM")
    reference.add_argument("--bf16-reference", action="store_true", help="Unquantized BF16 storage, FP32 compute (casts weights on each call)")
    parser.add_argument("--output", default="results/ocr.json")
    args = parser.parse_args()
    from unlimited_ocr.cpu_checkpoint import is_quantized_checkpoint
    if args.vision_bf16 and is_quantized_checkpoint(args.model_dir):
        parser.error("--vision-bf16 requires original BF16 weights, not an offline INT8 directory")
    configure_cpu(args.threads)
    t = time.perf_counter()
    quantized = not (args.fp32 or args.bf16_reference)
    model = load_model(args.model_dir, quantized=quantized, bf16_storage=args.bf16_reference)
    quantized = model._cpu_quantized_weights
    if not args.dense_attention:
        from unlimited_ocr.cpu_attention import install_attention
        install_attention()
        if not args.baseline:
            from unlimited_ocr.cpu_native_attention import install_native_attention
            install_native_attention()
    from unlimited_ocr.cpu_norm import install_norm
    fused_norms = install_norm(model)
    from unlimited_ocr.cpu_vision import install_vision
    install_vision(model, query_chunk=64 if args.baseline else args.vision_chunk)
    native_moe = 0
    if not args.baseline:
        from unlimited_ocr.cpu_sam_layout import install_sam_layout
        from unlimited_ocr.cpu_router import install_router
        install_sam_layout(model)
        install_router()
        if quantized and not args.optimize_moe:
            from unlimited_ocr.cpu_moe_native import install_native_moe
            native_moe = install_native_moe(model)
    if args.optimize_moe:
        from unlimited_ocr.cpu_moe import install_moe
        install_moe(model, fuse=True)
    bf16_vision = False
    if args.vision_bf16:
        from unlimited_ocr.cpu_bf16_vision import install_bf16_sam
        bf16_vision = install_bf16_sam(model, directory=args.model_dir)
    vision_threads = args.threads if args.baseline else args.vision_threads
    decode_threads = args.threads if args.baseline else (args.decode_threads or args.threads)
    from unlimited_ocr.cpu_threads import install_stage_threads
    install_stage_threads(model, vision_threads, args.threads, decode_threads)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    from upstream.modeling_unlimitedocr import _uocr_build_inputs
    images = [Image.open(path).convert("RGB") for path in args.images]
    prompt = "<PAGE><image>" * len(images) + "Multi page parsing."
    inputs = _uocr_build_inputs(tokenizer, prompt, images, crop_mode=args.tiles)
    load_time = time.perf_counter() - t
    from unlimited_ocr.cpu_profile import StageProfiler
    observer = StageProfiler(model) if args.profile else contextlib.nullcontext()
    with torch.inference_mode(), observer as timing:
        started = time.perf_counter()
        result = model._uocr_generate(tokenizer, *inputs, max_new_tokens=args.max_new_tokens,
                                     ngram_size=35, ngram_window=128, stream=True)
        wall_seconds = time.perf_counter() - started
    if args.profile:
        result["profile"] = timing.report()
    from unlimited_ocr.cpu_isa import isa_report
    from unlimited_ocr.cpu_norm import backend_name as norm_backend
    from unlimited_ocr.cpu_native_attention import backend_name as attention_backend
    result.update(threads=args.threads, quantized=quantized, bf16_reference=args.bf16_reference,
                  sparse_attention=not args.dense_attention, fused_norms=fused_norms,
                  native_attention=not (args.baseline or args.dense_attention),
                  optimize_moe=args.optimize_moe, vision_threads=vision_threads, decode_threads=decode_threads,
                  native_moe=native_moe, cached_router=not args.baseline, contiguous_sam=not args.baseline,
                  vision_chunk=64 if args.baseline else args.vision_chunk, wall_seconds=wall_seconds,
                  load_seconds=load_time, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                  cpu_capability=torch.backends.cpu.get_cpu_capability(),
                  torch_version=torch.__version__)
    result.update(isa=isa_report(), weight_format=model._cpu_weight_format,
                  linear_backend="fbgemm" if quantized else "fp32_dequantize" if model._cpu_weight_format == "int8_dequantize" else "torch",
                  bf16_vision=bf16_vision,
                  norm_backend=norm_backend(),
                  attention_backend="dense" if args.dense_attention else "gather" if args.baseline else attention_backend())
    result["native_attention"] = result["attention_backend"] in ("avx2", "avx512")
    decode = result["time"] - (result["prefill_time"] or result["time"])
    result["decode_tokens_per_second"] = max(0, len(result["token_ids"]) - 1) / decode if decode > 0 else None
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved {path}; elapsed={result['time']:.2f}s peak RSS={result['peak_rss_mib']:.0f} MiB")


if __name__ == "__main__":
    main()
