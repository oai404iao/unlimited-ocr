"""Warm current-model benchmark. Run different ISA settings in fresh processes."""
import argparse
import json
from pathlib import Path
import statistics
import time

from unlimited_ocr.cpu_runtime import configure_cpu, load_model
from unlimited_ocr.cpu_isa import isa_report
import torch
from PIL import Image
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--vision-threads", type=int, default=6)
    parser.add_argument("--decode-threads", type=int)
    parser.add_argument("--vision-bf16", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    configure_cpu(args.threads)
    started = time.perf_counter()
    model = load_model()
    from unlimited_ocr.cpu_attention import install_attention
    from unlimited_ocr.cpu_native_attention import install_native_attention, backend_name as attention_backend
    from unlimited_ocr.cpu_norm import install_norm, backend_name as norm_backend
    from unlimited_ocr.cpu_vision import install_vision
    from unlimited_ocr.cpu_sam_layout import install_sam_layout
    from unlimited_ocr.cpu_router import install_router
    from unlimited_ocr.cpu_threads import install_stage_threads
    install_attention()
    install_native_attention()
    install_norm(model)
    install_vision(model, query_chunk=256)
    install_sam_layout(model)
    install_router()
    bf16_vision = False
    if args.vision_bf16:
        from unlimited_ocr.cpu_bf16_vision import install_bf16_sam
        bf16_vision = install_bf16_sam(model)
    if model._cpu_quantized_weights:
        from unlimited_ocr.cpu_moe_script import install_scripted_moe
        install_scripted_moe(model)
    decode_threads = args.decode_threads or args.threads
    install_stage_threads(model, args.vision_threads, args.threads, decode_threads)
    tokenizer = AutoTokenizer.from_pretrained("models/unlimited-ocr-2")
    from upstream.modeling_unlimitedocr import _uocr_build_inputs
    inputs = _uocr_build_inputs(tokenizer, "<PAGE><image>" * len(args.images) + "Multi page parsing.",
                                [Image.open(p).convert("RGB") for p in args.images], crop_mode=False)
    result = dict(isa=isa_report(), weight_format=model._cpu_weight_format,
                  bf16_vision=bf16_vision,
                  prefill_threads=args.threads, vision_threads=args.vision_threads,
                  decode_threads=decode_threads, load_seconds=time.perf_counter() - started,
                  torch_version=torch.__version__, runs=[])
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    for i in range(args.repeats + 1):
        with torch.inference_mode():
            start = time.perf_counter()
            run = model._uocr_generate(tokenizer, *inputs, max_new_tokens=args.max_new_tokens,
                                      ngram_size=35, ngram_window=128, stream=False)
            run["wall_seconds"] = time.perf_counter() - start
        run["label"] = "warmup" if i == 0 else f"measured-{i}"
        run["decode_seconds"] = run["time"] - run["prefill_time"]
        result["runs"].append(run)
        result.update(attention_backend=attention_backend(), norm_backend=norm_backend())
        if i:
            measured = result["runs"][1:]
            result["median"] = {key: statistics.median(r[key] for r in measured)
                                for key in ("wall_seconds", "prefill_time", "decode_seconds")}
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"{run['label']} {run['wall_seconds']:.3f}s prefill={run['prefill_time']:.3f}s "
              f"tokens={len(run['token_ids'])} finish={run['finish_reason']}", flush=True)
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
