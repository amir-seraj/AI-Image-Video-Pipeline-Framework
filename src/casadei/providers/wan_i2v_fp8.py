"""Wan2.1 Image-to-Video model provider with FP8 quantization and torch.compile."""

from __future__ import annotations

import logging
import numpy as np
import torch
from pathlib import Path
from PIL import Image as PILImage

from casadei import MODELS_DIR
from casadei.models.base import ModelCapability, ImageConstraint, TextConstraint, VideoConstraint
from casadei.models.image_to_video import ImageToVideoModel

logger = logging.getLogger(__name__)

FP8_CACHE_DIR = Path(MODELS_DIR) / "fp8_cache"

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
    import triton  # noqa: F401
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


class WanImageToVideoFP8(ImageToVideoModel):
    """Wan-AI/Wan2.1-I2V-14B-720P image-to-video model with FP8 quantization.

    Identical to WanImageToVideo but applies FP8 weight-only quantization
    and torch.compile to the transformer for faster inference on supported hardware.
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

            # FP8 quantization with disk cache for the state_dict
            fp8_cache_path = FP8_CACHE_DIR / "wan_i2v_14b_720p_fp8_state.pt"
            self._load_or_quantize(pipe, fp8_cache_path)

            # Apply torch.compile for optimized inference (requires Triton)
            if _HAS_TRITON:
                # Use system ptxas (CUDA 13.0) instead of Triton's bundled one
                import os
                system_ptxas = "/usr/local/cuda/bin/ptxas"
                if os.path.exists(system_ptxas):
                    os.environ["TRITON_PTXAS_PATH"] = system_ptxas
                try:
                    pipe.transformer = torch.compile(pipe.transformer, mode="default")
                except Exception:
                    logger.warning("torch.compile failed, running without it", exc_info=True)
            else:
                logger.warning("Triton not available, skipping torch.compile (FP8 quantization still active)")

        self._pipeline = pipe

    @staticmethod
    def _load_or_quantize(pipe, cache_path: Path) -> None:
        """Load cached FP8 state_dict or quantize and cache."""
        if quantize_ is None or Float8WeightOnlyConfig is None:
            logger.warning("torchao not available, skipping FP8 quantization")
            return

        # Always apply quantize_ to set up FP8 module structure
        logger.info("Applying FP8 quantization structure...")
        quantize_(pipe.transformer, Float8WeightOnlyConfig())

        if cache_path.exists():
            logger.info("Loading cached FP8 weights from %s", cache_path)
            try:
                cached_state = torch.load(cache_path, map_location="cuda", weights_only=True)
                pipe.transformer.load_state_dict(cached_state)
                logger.info("FP8 cache loaded successfully")
                return
            except Exception:
                logger.warning("Failed to load FP8 cache, will overwrite", exc_info=True)
                cache_path.unlink(missing_ok=True)

        # Save the freshly quantized state_dict (just tensors, no pickle issues)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(pipe.transformer.state_dict(), cache_path)
            size_gb = cache_path.stat().st_size / 1024**3
            logger.info("Saved FP8 state_dict cache to %s (%.1f GB)", cache_path, size_gb)
        except Exception:
            logger.warning("Failed to save FP8 cache", exc_info=True)
            cache_path.unlink(missing_ok=True)

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
