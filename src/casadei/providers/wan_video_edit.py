"""Wan2.1 Video-to-Video editing model provider."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image as PILImage

from casadei import MODELS_DIR
from casadei.models.base import ModelCapability, TextConstraint, VideoConstraint
from casadei.models.video_edit import VideoEditModel

try:
    from diffusers import AutoencoderKLWan, WanVideoToVideoPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
except ImportError:
    AutoencoderKLWan = None
    WanVideoToVideoPipeline = None
    UniPCMultistepScheduler = None


class WanVideoEdit(VideoEditModel):
    """Wan-AI/Wan2.1-T2V-14B text-guided video-to-video editing model.

    Accepts 1 video and a text prompt, produces an edited video.
    The ``strength`` parameter controls how much the output diverges
    from the input (0.0 = no change, 1.0 = complete regeneration).
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
