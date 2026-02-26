"""LongCat-Image-Edit provider.

Meituan's 14.6B image editing model (8.3B Qwen2.5-VL text encoder + 6.3B DiT).
Uses LongCatImageEditPipeline from diffusers. ~29 GB in BF16, fits easily on
Jetson Thor (128 GB unified) without quantization.

Download + first run:  python src/casadei/providers/longcat_image_edit.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image as PILImage

from casadei import MODELS_DIR
from casadei.models.base import ModelCapability, ImageConstraint, TextConstraint
from casadei.models.image_edit import ImageEditModel
from casadei.providers._base import clamp_steps

logger = logging.getLogger(__name__)

MODEL_ID = "meituan-longcat/LongCat-Image-Edit"

try:
    from diffusers import LongCatImageEditPipeline
except ImportError:
    LongCatImageEditPipeline = None


def download_model() -> None:
    """Download the model to HF cache if not already present."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {MODEL_ID} (~29 GB) ...")
    snapshot_download(MODEL_ID, cache_dir=MODELS_DIR)
    print("Download complete.")


class LongCatImageEdit(ImageEditModel):
    """meituan-longcat/LongCat-Image-Edit — 14.6B image editor.

    8.3B Qwen2.5-VL text encoder + 6.3B DiT transformer + VAE.
    Accepts 1 image and a text prompt, produces 1 edited image.
    ~29 GB in BF16.
    """

    MODEL_ID = MODEL_ID

    capability = ModelCapability(
        inputs=[
            ImageConstraint(
                required=True,
                max_count=1,
                supported_formats=["png", "jpg", "jpeg", "webp"],
            ),
            TextConstraint(required=True),
        ],
        outputs=[
            ImageConstraint(required=True, max_count=1),
        ],
    )

    PIPELINE_CLS = LongCatImageEditPipeline

    DEFAULT_PARAMS = {
        "num_inference_steps": 50,
        "guidance_scale": 4.5,
        "negative_prompt": "",
        "num_images_per_prompt": 1,
    }

    MIN_STEPS = 1
    MAX_STEPS = 50

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = None

    def load_model(self) -> None:
        if LongCatImageEditPipeline is None:
            raise ImportError(
                "diffusers with LongCatImageEditPipeline is required. "
                "Install: pip install git+https://github.com/huggingface/diffusers"
            )

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        pipe = LongCatImageEditPipeline.from_pretrained(
            self.MODEL_ID, torch_dtype=torch_dtype, cache_dir=MODELS_DIR
        )
        if torch.cuda.is_available():
            pipe.to("cuda")

        self._pipeline = pipe

    def unload_model(self) -> None:
        self._pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _edit(
        self,
        images: list[PILImage.Image],
        prompt: str,
        negative_prompt: str,
        **kwargs,
    ) -> PILImage.Image:
        if self._pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        params = {**self.DEFAULT_PARAMS, **kwargs}
        if negative_prompt:
            params["negative_prompt"] = negative_prompt
        clamp_steps(params, "num_inference_steps", self.MIN_STEPS, self.MAX_STEPS)

        target_size = images[0].size

        with torch.inference_mode():
            output = self._pipeline(
                images[0],
                prompt,
                **params,
            )

        result = output.images[0]
        if result.size != target_size:
            result = result.resize(target_size, PILImage.LANCZOS)
        return result


if __name__ == "__main__":
    download_model()
