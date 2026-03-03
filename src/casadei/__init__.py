"""Casadei — Flexible AI pipeline framework."""

from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

from casadei.media import ImageMedia, TextMedia, VideoMedia, MediaBundle
from casadei.models.base import (
    AIModel, ModelCapability, MediaConstraint,
    ImageConstraint, TextConstraint, VideoConstraint,
)
from casadei.models.image_edit import ImageEditModel
from casadei.models.image_to_multiview import ImageToMultiViewModel
from casadei.models.image_to_video import ImageToVideoModel
from casadei.models.video_edit import VideoEditModel
from casadei.models.reference_inpaint import ReferenceInpaintModel
from casadei.models.vision_language import VisionLanguageModel
from casadei.models.registry import ModelRegistry, default_registry
from casadei.providers.qwen_image_edit import QwenImageEdit
from casadei.providers.qwen3_vl_8b import Qwen3VL8B
from casadei.providers.qwen3_vl_30b import Qwen3VL30B
from casadei.providers.wan_i2v import WanImageToVideo
from casadei.providers.wan_i2v_fp8 import WanImageToVideoFP8
from casadei.providers.wan_video_edit import WanVideoEdit
from casadei.providers.wan_video_edit_fp8 import WanVideoEditFP8
from casadei.agent import Agent, AgentConfig, load_agent, save_agent
from casadei.pipeline import AgentStep, CodeStep, PipelineStep, Pipeline
from casadei.loop import LoopStep, LoopIteration, LoopResult
from casadei.logging import ExecutionLog, StepLog, LoggedPipeline
from casadei.visualization import to_mermaid

__all__ = [
    "MODELS_DIR",
    "ImageMedia", "TextMedia", "VideoMedia", "MediaBundle",
    "AIModel", "ModelCapability", "MediaConstraint",
    "ImageConstraint", "TextConstraint", "VideoConstraint",
    "ImageEditModel",
    "ImageToMultiViewModel",
    "ImageToVideoModel",
    "VideoEditModel",
    "ReferenceInpaintModel",
    "VisionLanguageModel",
    "ModelRegistry", "default_registry",
    "QwenImageEdit",
    "Qwen3VL8B",
    "Qwen3VL30B",
    "WanImageToVideo", "WanImageToVideoFP8",
    "WanVideoEdit", "WanVideoEditFP8",
    "Agent", "AgentConfig", "load_agent", "save_agent",
    "AgentStep", "CodeStep", "PipelineStep", "Pipeline",
    "LoopStep", "LoopIteration", "LoopResult",
    "ExecutionLog", "StepLog", "LoggedPipeline",
    "to_mermaid",
]
