"""Generate a deterministic OCR smoke-test page (not an accuracy benchmark)."""
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

lines = ["CPU OCR TEST", "Unlimited OCR 2", "Invoice number: 2026-001", "Total: 123.45 USD",
         "中文识别测试", "使用 AVX2 加速 CPU 推理"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", required=True, help="Path to a font supporting Chinese and Latin text")
    args = parser.parse_args()
    font = ImageFont.truetype(args.font, 36)
    path = Path("samples")
    path.mkdir(exist_ok=True)
    image = Image.new("RGB", (1024, 1024), "white")
    draw = ImageDraw.Draw(image)
    for i, text in enumerate(lines):
        draw.text((80, 100 + 100*i), text, font=font, fill="black")
    image.save(path / "smoke.png")
    (path / "smoke.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
