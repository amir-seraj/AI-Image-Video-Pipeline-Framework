"""Model provider implementations."""

from casadei.providers.qwen_image_edit import QwenImageEdit
from casadei.providers.wan_i2v import WanImageToVideo
from casadei.providers.wan_i2v_fp8 import WanImageToVideoFP8
from casadei.providers.wan_video_edit import WanVideoEdit
from casadei.providers.wan_video_edit_fp8 import WanVideoEditFP8

__all__ = [
    "QwenImageEdit",
    "WanImageToVideo", "WanImageToVideoFP8",
    "WanVideoEdit", "WanVideoEditFP8",
]
