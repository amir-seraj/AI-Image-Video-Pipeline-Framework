"""Wan2.1 Video-to-Video editing model provider with FP8 quantization + torch.compile."""

from __future__ import annotations

import logging
import numpy as np
import torch
from pathlib import Path
from PIL import Image as PILImage

from casadei import MODELS_DIR
from casadei.models.base import ModelCapability, TextConstraint, VideoConstraint
from casadei.models.video_edit import VideoEditModel

logger = logging.getLogger(__name__)

FP8_CACHE_DIR = Path(MODELS_DIR) / "fp8_cache"

try:
    from diffusers import AutoencoderKLWan, WanVideoToVideoPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
except ImportError:
    AutoencoderKLWan = None
    WanVideoToVideoPipeline = None
    UniPCMultistepScheduler = None

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


class WanVideoEditFP8(VideoEditModel):
    """Wan-AI/Wan2.1-T2V-14B with FP8 quantization and torch.compile.

    Identical to :class:`~casadei.providers.wan_video_edit.WanVideoEdit` but
    applies FP8 weight-only quantization (via torchao) and ``torch.compile``
    to the transformer for reduced VRAM usage and faster inference.
    """

    MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"

    capability = ModelCapability(
        inputs=[
            VideoConstraint(required=True, max_count=1),
            TextConstraint(required=True),
        ],
        outputs=[
            VideoConstraint(required=True, max_count=1),
        ],
    )

    DEFAULT_PARAMS = {
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "strength": 0.7,
    }

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = None

    def load_model(self) -> None:
        if WanVideoToVideoPipeline is None:
            raise ImportError(
                "diffusers with WanVideoToVideoPipeline is required. "
                "Install: pip install diffusers"
            )

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        vae = AutoencoderKLWan.from_pretrained(
            self.MODEL_ID, subfolder="vae",
            torch_dtype=torch.float32, cache_dir=MODELS_DIR,
        )
        pipe = WanVideoToVideoPipeline.from_pretrained(
            self.MODEL_ID,
            vae=vae,
            torch_dtype=torch_dtype,
            cache_dir=MODELS_DIR,
        )
        flow_shift = 5.0
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=flow_shift
        )
        if torch.cuda.is_available():
            pipe.to("cuda")

            # FP8 quantization with disk cache for the state_dict
            fp8_cache_path = FP8_CACHE_DIR / "wan_t2v_14b_fp8_state.pt"
            self._load_or_quantize(pipe, fp8_cache_path)

            # torch.compile for max performance (requires Triton)
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

    def _edit(
        self,
        video_frames: list[np.ndarray],
        prompt: str,
        negative_prompt: str,
        **kwargs,
    ) -> list[np.ndarray]:
        if self._pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        params = {**self.DEFAULT_PARAMS, **kwargs}
        if negative_prompt:
            params["negative_prompt"] = negative_prompt

        pil_frames = [PILImage.fromarray(f) for f in video_frames]
        h, w = video_frames[0].shape[:2]

        with torch.inference_mode():
            output = self._pipeline(
                video=pil_frames,
                prompt=prompt,
                height=h,
                width=w,
                **params,
            )

        return output.frames[0]
