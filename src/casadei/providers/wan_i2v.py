"""Wan2.1 Image-to-Video model provider."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image as PILImage

from casadei import MODELS_DIR
from casadei.models.base import ModelCapability, ImageConstraint, TextConstraint, VideoConstraint
from casadei.models.image_to_video import ImageToVideoModel

try:
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from transformers import CLIPVisionModel
except ImportError:
    AutoencoderKLWan = None
    WanImageToVideoPipeline = None
    CLIPVisionModel = None


class WanImageToVideo(ImageToVideoModel):
    """Wan-AI/Wan2.1-I2V-14B-720P image-to-video model.

    Accepts 1 image and a text prompt, produces a video.
    Uses bfloat16 precision.
    """

    MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"

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
            VideoConstraint(
                required=True,
                max_count=1,
                max_width=1280,
                max_height=720,
            ),
        ],
    )

    PIPELINE_CLS = WanImageToVideoPipeline

    DEFAULT_PARAMS = {
        "num_frames": 81,
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "height": 720,
        "width": 1280,
    }

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = None

    def load_model(self) -> None:
        if WanImageToVideoPipeline is None:
            raise ImportError(
                "diffusers with WanImageToVideoPipeline is required. "
                "Install: pip install diffusers transformers"
            )

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        image_encoder = CLIPVisionModel.from_pretrained(
            self.MODEL_ID, subfolder="image_encoder",
            torch_dtype=torch.float32, cache_dir=MODELS_DIR,
        )
        vae = AutoencoderKLWan.from_pretrained(
            self.MODEL_ID, subfolder="vae",
            torch_dtype=torch.float32, cache_dir=MODELS_DIR,
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
            self.MODEL_ID,
            vae=vae,
            image_encoder=image_encoder,
            torch_dtype=torch_dtype,
            cache_dir=MODELS_DIR,
        )
        if torch.cuda.is_available():
            pipe.to("cuda")
        self._pipeline = pipe

    def unload_model(self) -> None:
        self._pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _resize_for_vae(self, image: PILImage.Image, height: int, width: int) -> tuple[PILImage.Image, int, int]:
        """Resize image to match target resolution while respecting VAE patch constraints."""
        max_area = height * width
        aspect_ratio = image.height / image.width
        mod_value = (
            self._pipeline.vae_scale_factor_spatial
            * self._pipeline.transformer.config.patch_size[1]
        )
        h = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        w = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        return image.resize((w, h)), h, w

    def _generate(
        self,
        image: PILImage.Image,
        prompt: str,
        negative_prompt: str,
        **kwargs,
    ) -> list[np.ndarray]:
        if self._pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        params = {**self.DEFAULT_PARAMS, **kwargs}
        height = params.pop("height", 720)
        width = params.pop("width", 1280)

        resized_image, h, w = self._resize_for_vae(image, height, width)

        if negative_prompt:
            params["negative_prompt"] = negative_prompt

        with torch.inference_mode():
            output = self._pipeline(
                image=resized_image,
                prompt=prompt,
                height=h,
                width=w,
                **params,
            )

        return output.frames[0]
