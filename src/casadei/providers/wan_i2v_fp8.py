"""Wan2.1 Image-to-Video FP8 variant — FP8 dynamic quantization + torch.compile.

Uses Float8DynamicActivationFloat8WeightConfig (torchao) for actual FP8 tensor
core matmuls, combined with torch.compile(backend="inductor") for Triton-based
kernel fusion.

Requirements:
  - pytorch-triton (cu130 aarch64 wheel): provides Triton for SM 110
  - TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas  (system CUDA 13.0 ptxas)
"""

from __future__ import annotations

import logging
import os
import numpy as np
import torch
from pathlib import Path
from PIL import Image as PILImage

from casadei import MODELS_DIR
from casadei.models.base import ModelCapability, ImageConstraint, TextConstraint, VideoConstraint
from casadei.models.image_to_video import ImageToVideoModel

logger = logging.getLogger(__name__)

FP8_CACHE_DIR = Path(MODELS_DIR) / "fp8_cache"

if "TRITON_PTXAS_PATH" not in os.environ:
    _system_ptxas = Path("/usr/local/cuda/bin/ptxas")
    if _system_ptxas.exists():
        os.environ["TRITON_PTXAS_PATH"] = str(_system_ptxas)

try:
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from transformers import CLIPVisionModel
except ImportError:
    AutoencoderKLWan = None
    WanImageToVideoPipeline = None
    CLIPVisionModel = None

try:
    from torchao.quantization import quantize_, Float8WeightOnlyConfig
except ImportError:
    quantize_ = None
    Float8WeightOnlyConfig = None

try:
    from diffusers.hooks import apply_taylorseer_cache, TaylorSeerCacheConfig
except ImportError:
    apply_taylorseer_cache = None
    TaylorSeerCacheConfig = None


class WanImageToVideoFP8(ImageToVideoModel):
    """Wan-AI/Wan2.1-I2V-14B-720P — FP8 dynamic quantization + compiled.

    Applies FP8 dynamic activation + weight quantization (torchao) and
    torch.compile(backend="inductor") for fused FP8 tensor core inference.
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

            if quantize_ is not None and Float8WeightOnlyConfig is not None:
                logger.info("Applying FP8 weight-only quantization...")
                quantize_(pipe.transformer, Float8WeightOnlyConfig())
            else:
                logger.warning("torchao not available, skipping FP8 quantization")

            try:
                pipe.transformer = torch.compile(
                    pipe.transformer, mode="default"
                )
                logger.info("torch.compile(mode='default') applied")
            except Exception:
                logger.warning(
                    "torch.compile failed, running in eager mode",
                    exc_info=True,
                )

            if apply_taylorseer_cache is not None:
                apply_taylorseer_cache(
                    pipe.transformer,
                    TaylorSeerCacheConfig(
                        cache_interval=5,
                        disable_cache_before_step=3,
                        max_order=1,
                    ),
                )
                logger.info("TaylorSeer cache applied (cache_interval=5)")

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
