"""Qwen3-VL-32B-Instruct-FP8 provider.

Qwen's 32B vision-language model, pre-quantized to FP8 (e4m3, block size 128).
Performance nearly identical to BF16 at half the memory (~35.5 GB).

Note: Transformers does not support loading these FP8 weights directly.
      Use vLLM or SGLang for serving.

Download:  python src/casadei/providers/qwen3_vl_32b.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from casadei import MODELS_DIR

MODEL_ID = "Qwen/Qwen3-VL-32B-Instruct-FP8"
LOCAL_DIR = MODELS_DIR / "Qwen3-VL-32B-Instruct-FP8"


def download_model() -> Path:
    """Download the model to LOCAL_DIR if not already present."""
    from huggingface_hub import snapshot_download

    if LOCAL_DIR.exists() and any(LOCAL_DIR.glob("*.safetensors")):
        print(f"Model already exists at {LOCAL_DIR}")
        return LOCAL_DIR

    print(f"Downloading {MODEL_ID} to {LOCAL_DIR} ...")
    print("This is ~35.5 GB and will take a while.")
    snapshot_download(
        MODEL_ID,
        local_dir=str(LOCAL_DIR),
    )
    print(f"Download complete: {LOCAL_DIR}")
    return LOCAL_DIR


if __name__ == "__main__":
    download_model()
