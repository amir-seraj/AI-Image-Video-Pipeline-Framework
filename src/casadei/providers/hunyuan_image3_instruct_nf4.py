"""HunyuanImage-3.0-Instruct NF4 provider.

Loads EricRollei/HunyuanImage-3.0-Instruct-NF4 — a pre-quantized NF4 (4-bit)
version of Tencent's 83B MoE image generation model. Uses ~45 GB memory,
fits comfortably on Jetson Thor (128 GB unified).

Uses bitsandbytes NF4 quantization with standard transformers API.
No FlashInfer dependency — uses eager MoE + PyTorch SDPA attention.

Download + first run:  python src/casadei/providers/hunyuan_image3_instruct_nf4.py
"""

from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

import torch
from PIL import Image as PILImage

from casadei import MODELS_DIR
from casadei.models.base import ModelCapability, ImageConstraint, TextConstraint
from casadei.models.image_edit import ImageEditModel

logger = logging.getLogger(__name__)

MODEL_ID = "EricRollei/HunyuanImage-3.0-Instruct-NF4"
LOCAL_DIR = MODELS_DIR / "HunyuanImage-3-Instruct-NF4"


def download_model() -> Path:
    """Download the NF4 model to LOCAL_DIR if not already present."""
    from huggingface_hub import snapshot_download

    existing = sorted(LOCAL_DIR.glob("model-*-of-*.safetensors"))
    expected = 10
    if len(existing) >= expected:
        print(f"All {expected} shards present at {LOCAL_DIR}")
        return LOCAL_DIR

    if existing:
        print(f"Incomplete: {len(existing)}/{expected} shards. Resuming...")
    else:
        print(f"Downloading {MODEL_ID} to {LOCAL_DIR} ...")
        print("This is ~45 GB (NF4 pre-quantized) and will take a while.")
    snapshot_download(MODEL_ID, local_dir=str(LOCAL_DIR))
    print(f"Download complete: {LOCAL_DIR}")
    return LOCAL_DIR


class HunyuanImage3InstructNF4(ImageEditModel):
    """HunyuanImage-3.0-Instruct — NF4 quantized (bitsandbytes 4-bit).

    83B MoE model with NF4 quantization on FFN/expert layers, attention
    projections and VAE kept in BF16. Uses ~45 GB memory total.
    Supports text-to-image and image editing via generate_image().
    """

    capability = ModelCapability(
        inputs=[
            ImageConstraint(
                required=False,
                max_count=4,
                supported_formats=["png", "jpg", "jpeg", "webp"],
            ),
            TextConstraint(required=True),
        ],
        outputs=[
            ImageConstraint(required=True, max_count=1),
        ],
    )

    DEFAULT_PARAMS = {
        "diff_infer_steps": 50,
        "seed": 42,
        "image_size": "auto",
        "use_system_prompt": "en_unified",
        "bot_task": "think_recaption",
    }

    def __init__(self) -> None:
        super().__init__()
        self._model = None

    def load_model(self) -> None:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        model_path = LOCAL_DIR if LOCAL_DIR.exists() else MODEL_ID

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        logger.info("Loading %s with NF4 quantization...", model_path)
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        model.load_tokenizer(str(model_path))
        self._model = model
        logger.info("Model loaded. GPU memory: %.2f GB", torch.cuda.memory_allocated() / 1024**3)

    def unload_model(self) -> None:
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _edit(
        self,
        images: list[PILImage.Image],
        prompt: str,
        negative_prompt: str,
        **kwargs,
    ) -> PILImage.Image:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        params = {**self.DEFAULT_PARAMS, **kwargs}

        gen_kwargs = {
            "prompt": prompt,
            "seed": params.get("seed", 42),
            "image_size": params.get("image_size", "auto"),
            "use_system_prompt": params.get("use_system_prompt", "en_unified"),
            "bot_task": params.get("bot_task", "think_recaption"),
            "diff_infer_steps": params.get("diff_infer_steps", 50),
        }

        if images:
            gen_kwargs["image"] = images
            gen_kwargs["infer_align_image_size"] = True

        cot_text, samples = self._model.generate_image(**gen_kwargs)
        return samples[0]


if __name__ == "__main__":
    download_model()
