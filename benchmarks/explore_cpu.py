"""Same-process, warm-model comparisons. No parallel benchmark processes."""
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
    parser.add_argument("--output", default="results/exploration.json")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--mode", choices=["profile", "decode", "vision", "threads", "final", "hotspots", "remaining"], default="profile")
    args = parser.parse_args()
    configure_cpu(3)
    model = load_model()
    from unlimited_ocr.cpu_attention import install_attention
    from unlimited_ocr.cpu_norm import install_norm
    from unlimited_ocr.cpu_vision import install_vision
    from unlimited_ocr.cpu_profile import StageProfiler
    install_attention()
    install_norm(model)
    install_vision(model)
    if args.mode in ("hotspots", "remaining"):
        from unlimited_ocr.cpu_native_attention import install_native_attention
        from unlimited_ocr.cpu_threads import install_stage_threads
        install_native_attention()
        install_vision(model, query_chunk=256)
        install_stage_threads(model, 6, 3, 3)
    tokenizer = AutoTokenizer.from_pretrained("models/unlimited-ocr-2")
    from upstream.modeling_unlimitedocr import _uocr_build_inputs
    inputs = _uocr_build_inputs(tokenizer, "<PAGE><image>" * len(args.images) + "Multi page parsing.",
                                [Image.open(path).convert("RGB") for path in args.images], crop_mode=False)
    results = []
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    def run(label, profile=False):
        observer = StageProfiler(model, detailed=args.mode in ("hotspots", "remaining")) if profile else contextlib.nullcontext()
        with torch.inference_mode(), observer as timing:
            start = time.perf_counter()
            result = model._uocr_generate(tokenizer, *inputs, max_new_tokens=args.max_new_tokens,
                                         ngram_size=35, ngram_window=128, stream=False)
            wall = time.perf_counter() - start
        result.update(label=label, wall_seconds=wall,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        if profile:
            result["profile"] = timing.report()
        results.append(result)
        output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"{label}: {wall:.3f}s prefill={result['prefill_time']:.3f}s tokens={len(result['token_ids'])}", flush=True)

    # Warm the whole pipeline, excluding load/packing and first-request effects.
    run("warmup")
    for i in range(args.repeats):
        run(f"baseline-{i}")
    if args.mode == "profile":
        run("profile", profile=True)
    elif args.mode == "decode":
        from unlimited_ocr.cpu_moe import install_moe
        install_moe(model, fuse=False)
        for i in range(args.repeats):
            run(f"moe-{i}")
        from unlimited_ocr.cpu_native_attention import install_native_attention
        install_native_attention()
        for i in range(args.repeats):
            run(f"native-{i}")
        install_moe(model, fuse=True)
        for i in range(args.repeats):
            run(f"fused-{i}")
        run("optimized-profile", profile=True)
    elif args.mode == "vision":
        pixels = inputs[1][1].float()
        reference = None
        for threads in (1, 3, 6):
            torch.set_num_threads(threads)
            for chunk in (64, 128, 256, 512):
                install_vision(model, query_chunk=chunk)
                times = []
                with torch.inference_mode():
                    for _ in range(args.repeats):
                        start = time.perf_counter()
                        actual = model.model.sam_model(pixels)
                        times.append(time.perf_counter() - start)
                    if reference is None:
                        reference = actual.clone()
                    error = float((actual - reference).abs().max())
                item = dict(label="sam-sweep", threads=threads, chunk=chunk, seconds=times, max_abs_error=error)
                results.append(item)
                output.write_text(json.dumps(results, indent=2, ensure_ascii=False))
                print(item, flush=True)
    elif args.mode == "threads":
        from unlimited_ocr.cpu_threads import install_stage_threads
        from unlimited_ocr.cpu_native_attention import install_native_attention
        from unlimited_ocr.cpu_moe import install_moe
        install_native_attention()
        install_moe(model, fuse=True)
        install_vision(model, query_chunk=256)
        for vision, prefill, decode in ((6, 3, 3), (6, 3, 1), (6, 3, 2), (6, 3, 4), (6, 3, 6)):
            install_stage_threads(model, vision, prefill, decode)
            for i in range(args.repeats):
                run(f"threads-{vision}-{prefill}-{decode}-{i}")
        install_stage_threads(model, 6, 3, 3)
        run("stage-profile", profile=True)
    elif args.mode == "final":
        from unlimited_ocr.cpu_threads import install_stage_threads
        from unlimited_ocr.cpu_native_attention import install_native_attention
        install_native_attention()
        install_vision(model, query_chunk=256)
        install_stage_threads(model, 6, 3, 3)
        for i in range(args.repeats):
            run(f"final-{i}")
        run("final-profile", profile=True)
    elif args.mode == "hotspots":
        run("detailed-profile", profile=True)
        with torch.inference_mode(), torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True) as profiler:
            model.model.sam_model(inputs[1][1].float())
        events = profiler.key_averages(group_by_input_shape=True)
        output.with_suffix(".sam.txt").write_text(events.table(sort_by="self_cpu_time_total", row_limit=40))
    elif args.mode == "remaining":
        from unlimited_ocr.cpu_sam_layout import install_sam_layout
        from unlimited_ocr.cpu_router import install_router
        from unlimited_ocr.cpu_moe_script import install_scripted_moe
        install_sam_layout(model)
        for i in range(args.repeats):
            run(f"layout-{i}")
        install_router()
        for i in range(args.repeats):
            run(f"router-{i}")
        install_scripted_moe(model)
        for i in range(args.repeats):
            run(f"scripted-moe-{i}")
        run("remaining-profile", profile=True)


if __name__ == "__main__":
    main()
