"""Load once, recognize multiple documents; run from the repository root."""
import argparse
import json
from pathlib import Path

# Import the ISA policy before torch initializes its CPU dispatch.
from unlimited_ocr.cpu_runtime import configure_cpu, load_model
from unlimited_ocr.cpu_isa import isa_report
import torch
from PIL import Image
from transformers import AutoTokenizer


def load_ocr(threads=3, vision_threads=6, decode_threads=3, model_dir="models/unlimited-ocr-2"):
    """Call once per process; returned model owns reusable packed weights."""
    configure_cpu(threads)
    model = load_model(model_dir, quantized=True)
    from unlimited_ocr.cpu_attention import install_attention
    from unlimited_ocr.cpu_native_attention import install_native_attention
    from unlimited_ocr.cpu_norm import install_norm
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
    if model._cpu_quantized_weights:
        from unlimited_ocr.cpu_moe_native import install_native_moe
        install_native_moe(model)
    install_stage_threads(model, vision_threads, threads, decode_threads)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    return model, tokenizer


@torch.inference_mode()
def recognize(model, tokenizer, image_paths, max_new_tokens=256):
    """One document per call; image_paths is its ordered page list, not a batch."""
    if not image_paths:
        raise ValueError("Provide at least one page image")
    from upstream.modeling_unlimitedocr import _uocr_build_inputs

    images = []
    for path in image_paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    prompt = "<PAGE><image>" * len(images) + "Multi page parsing."
    inputs = _uocr_build_inputs(tokenizer, prompt, images, crop_mode=False)
    return model._uocr_generate(
        tokenizer, *inputs, max_new_tokens=max_new_tokens,
        ngram_size=35, ngram_window=128, stream=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+")
    parser.add_argument("--model-dir", default="models/unlimited-ocr-2")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--output", default="results/example.json")
    args = parser.parse_args()
    model, tokenizer = load_ocr(model_dir=args.model_dir)
    result = recognize(model, tokenizer, args.images, args.max_new_tokens)
    result.update(isa=isa_report(), weight_format=model._cpu_weight_format)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(result["text"])
    print(f"finish_reason={result['finish_reason']}; saved {path}")


if __name__ == "__main__":
    main()
