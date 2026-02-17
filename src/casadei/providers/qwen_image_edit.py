"""Qwen Image Edit model provider."""

from __future__ import annotations

import torch
from PIL import Image as PILImage

from casadei import MODELS_DIR
from casadei.models.base import ModelCapability, ImageConstraint, TextConstraint
from casadei.models.image_edit import ImageEditModel

try:
    from diffusers import QwenImageEditPlusPipeline
except ImportError:
    QwenImageEditPlusPipeline = None


class QwenImageEdit(ImageEditModel):
    """Qwen/Qwen-Image-Edit-2511 model.

    Accepts up to 2 images and a text prompt, produces 1 edited image.
    Requires CUDA GPU with bfloat16 support.
    """

    MODEL_ID = "Qwen/Qwen-Image-Edit-2511"

    capability = ModelCapability(
        inputs=[
            ImageConstraint(
                required=True,
                max_count=2,
                supported_formats=["png", "jpg", "jpeg", "webp"],
            ),
            TextConstraint(required=True),
        ],
        outputs=[
            ImageConstraint(required=True, max_count=1),
        ],
    )

    DEFAULT_PARAMS = {
        "num_inference_steps": 40,
        "guidance_scale": 1.0,
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_images_per_prompt": 1,
    }

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = None

    def load_model(self) -> None:
        if QwenImageEditPlusPipeline is None:
            raise ImportError(
                "diffusers with QwenImageEditPlusPipeline is required. "
                "Install: pip install git+https://github.com/huggingface/diffusers"
            )
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            self.MODEL_ID, torch_dtype=torch.bfloat16, cache_dir=MODELS_DIR
        )
        pipe.enable_sequential_cpu_offload()
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
        params["negative_prompt"] = negative_prompt or params["negative_prompt"]

        with torch.inference_mode():
            output = self._pipeline(
                image=images,
                prompt=prompt,
                **params,
            )
        return output.images[0]
