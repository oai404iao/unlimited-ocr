"""Download only data files; audited model code is vendored in upstream/."""
from huggingface_hub import snapshot_download

REVISION = "04647c7a6098d8d58915f003536a741c82eafb40"

if __name__ == "__main__":
    snapshot_download("baidu/Unlimited-OCR-2", revision=REVISION, local_dir="models/unlimited-ocr-2",
                      allow_patterns=["*.safetensors", "*.json"], max_workers=2)
