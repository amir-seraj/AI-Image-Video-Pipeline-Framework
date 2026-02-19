"""Wan2.1 Video-to-Video FP8 variant — FP8 dynamic quantization + torch.compile.

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
from casadei.models.base import ModelCapability, TextConstraint, VideoConstraint
from casadei.models.video_edit import VideoEditModel

logger = logging.getLogger(__name__)

FP8_CACHE_DIR = Path(MODELS_DIR) / "fp8_cache"

if "TRITON_PTXAS_PATH" not in os.environ:
    _system_ptxas = Path("/usr/local/cuda/bin/ptxas")
    if _system_ptxas.exists():
        os.environ["TRITON_PTXAS_PATH"] = str(_system_ptxas)

try:
    from diffusers import AutoencoderKLWan, WanVideoToVideoPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
except ImportError:
    AutoencoderKLWan = None
    WanVideoToVideoPipeline = None
    UniPCMultistepScheduler = None

try:
    from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
except ImportError:
    quantize_ = None
    Float8DynamicActivationFloat8WeightConfig = None


class WanVideoEditFP8(VideoEditModel):
    """Wan-AI/Wan2.1-T2V-14B — FP8 dynamic quantization + compiled.

    Applies FP8 dynamic activation + weight quantization (torchao) and
    torch.compile(backend="inductor") for fused FP8 tensor core inference.
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

    PIPELINE_CLS = WanVideoToVideoPipeline

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

            if quantize_ is not None and Float8DynamicActivationFloat8WeightConfig is not None:
                logger.info("Applying FP8 dynamic activation + weight quantization...")
                quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig())
            else:
                logger.warning("torchao not available, skipping FP8 quantization")

            try:
                torch._inductor.config.coordinate_descent_tuning = False
                torch._inductor.config.max_autotune = False
                pipe.transformer = torch.compile(
                    pipe.transformer, backend="inductor", mode="reduce-overhead"
                )
                logger.info("torch.compile(backend='inductor', mode='reduce-overhead') applied")
            except Exception:
                logger.warning(
                    "torch.compile failed, running in eager mode",
                    exc_info=True,
                )

        self._pipeline = pipe

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
