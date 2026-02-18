"""HunyuanImage-3.0-Instruct provider.

Tencent's 80B MoE autoregressive image generation/editing model.
Uses transformers (not diffusers) — AutoModelForCausalLM with trust_remote_code.

Download:  python src/casadei/providers/hunyuan_image3_instruct.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from casadei import MODELS_DIR

MODEL_ID = "tencent/HunyuanImage-3.0-Instruct"
# Directory name must not contain dots (transformers limitation)
LOCAL_DIR = MODELS_DIR / "HunyuanImage-3-Instruct"


def download_model() -> Path:
    """Download the model to LOCAL_DIR if not already present."""
    from huggingface_hub import snapshot_download

    existing = sorted(LOCAL_DIR.glob("model-*-of-*.safetensors"))
    expected = 32
    if len(existing) >= expected:
        print(f"All {expected} shards present at {LOCAL_DIR}")
        return LOCAL_DIR

    if existing:
        print(f"Incomplete: {len(existing)}/{expected} shards. Resuming...")
    else:
        print(f"Downloading {MODEL_ID} to {LOCAL_DIR} ...")
        print("This is ~157 GB and will take a while.")
    snapshot_download(
        MODEL_ID,
        local_dir=str(LOCAL_DIR),
    )
    print(f"Download complete: {LOCAL_DIR}")
    return LOCAL_DIR


if __name__ == "__main__":
    download_model()
