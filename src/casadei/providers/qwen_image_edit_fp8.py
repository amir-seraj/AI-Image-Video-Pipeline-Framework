"""Qwen Image Edit FP8 variant — FP8 dynamic quantization + torch.compile.

Uses Float8DynamicActivationFloat8WeightConfig (torchao) for actual FP8 tensor
core matmuls, combined with torch.compile(backend="inductor") for Triton-based
kernel fusion. This combination is critical: FP8 without fusion is SLOWER than
BF16 due to per-layer quantization overhead, but with fusion the FP8 tensor
cores deliver ~1.5x speedup.

Requirements:
  - pytorch-triton (cu130 aarch64 wheel): provides Triton for SM 110
  - TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas  (system CUDA 13.0 ptxas)
  - The bundled ptxas in Triton 3.5.x is from CUDA 12.8 and doesn't know
    SM 110a. The system ptxas from CUDA 13.0 does.
"""

from __future__ import annotations

import logging
import os
import torch
from pathlib import Path
from PIL import Image as PILImage

from casadei import MODELS_DIR
from casadei.models.base import ModelCapability, ImageConstraint, TextConstraint
from casadei.models.image_edit import ImageEditModel
from casadei.providers._base import (
    clamp_steps,
    decode_and_save_step_latents,
    make_step_callback,
)

logger = logging.getLogger(__name__)

FP8_CACHE_DIR = Path(MODELS_DIR) / "fp8_cache"

# Ensure system ptxas is used for Triton compilation on Jetson Thor.
# The bundled ptxas (CUDA 12.8) doesn't support SM 110a; the system
# ptxas (CUDA 13.0) does.
if "TRITON_PTXAS_PATH" not in os.environ:
    _system_ptxas = Path("/usr/local/cuda/bin/ptxas")
    if _system_ptxas.exists():
        os.environ["TRITON_PTXAS_PATH"] = str(_system_ptxas)

try:
    from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
except ImportError:
    QwenImageEditPlusPipeline = None
    QwenImageTransformer2DModel = None

try:
    from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
except ImportError:
    quantize_ = None
    Float8DynamicActivationFloat8WeightConfig = None


class QwenImageEditFP8(ImageEditModel):
    """Qwen/Qwen-Image-Edit-2511 — FP8 dynamic quantization + compiled.

    Applies FP8 dynamic activation + weight quantization (torchao) and
    torch.compile(backend="inductor") for fused FP8 tensor core inference.
    ~1.5x faster than BF16 eager on Jetson Thor.
    """

    BASE_MODEL_ID = "Qwen/Qwen-Image-Edit-2511"

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

    PIPELINE_CLS = QwenImageEditPlusPipeline

    DEFAULT_PARAMS = {
        "num_inference_steps": 40,
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_images_per_prompt": 1,
    }

    MIN_STEPS = 1
    MAX_STEPS = 50

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = None
        self.save_steps_dir: Path | None = None
        self.save_steps_interval: int = 1

    def load_model(self) -> None:
        if QwenImageEditPlusPipeline is None or QwenImageTransformer2DModel is None:
            raise ImportError(
                "diffusers with QwenImageEditPlusPipeline and "
                "QwenImageTransformer2DModel is required."
            )

        pipe = QwenImageEditPlusPipeline.from_pretrained(
            self.BASE_MODEL_ID, torch_dtype=torch.bfloat16, cache_dir=MODELS_DIR
        )

        if torch.cuda.is_available():
            pipe.to("cuda")

            # FP8 dynamic activation + weight quantization for actual FP8
            # tensor core matmuls via _scaled_mm.
            if quantize_ is not None and Float8DynamicActivationFloat8WeightConfig is not None:
                logger.info("Applying FP8 dynamic activation + weight quantization...")
                quantize_(pipe.transformer, Float8DynamicActivationFloat8WeightConfig())
            else:
                logger.warning("torchao not available, skipping FP8 quantization")

            # Compile individual transformer blocks (not the whole model).
            # This avoids compiling pos_embed (which has CPU tensor issues)
            # and focuses on the hot path (60 repeated blocks = 95%+ compute).
            # Since all blocks share the same structure, Triton compiles
            # unique kernels once for block 0, then reuses for blocks 1-59.
            self._compile_transformer_blocks(pipe.transformer)

        self._pipeline = pipe

    @staticmethod
    def _compile_transformer_blocks(transformer: torch.nn.Module) -> None:
        """Compile individual transformer blocks for Triton kernel fusion.

        Compiles each block in transformer_blocks individually rather than
        the whole model. Benefits:
          - Avoids compiling pos_embed (has CPU tensors that break graphs)
          - All 60 blocks share the same structure, so Triton compiles
            unique kernel shapes only once (block 0), then cache-hits for 1-59
          - Much faster compilation than compiling the entire model
        """
        blocks = getattr(transformer, "transformer_blocks", None)
        if blocks is None:
            logger.warning("No transformer_blocks found, skipping compilation")
            return

        torch._inductor.config.coordinate_descent_tuning = False
        torch._inductor.config.max_autotune = False

        n = len(blocks)
        logger.info("Compiling %d transformer blocks...", n)
        compiled = 0
        for i, block in enumerate(blocks):
            try:
                blocks[i] = torch.compile(block, backend="inductor")
                compiled += 1
            except Exception:
                logger.warning("Failed to compile block %d", i, exc_info=True)
        logger.info("Compiled %d/%d transformer blocks", compiled, n)

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
        clamp_steps(params, "num_inference_steps", self.MIN_STEPS, self.MAX_STEPS)

        target_size = images[0].size
        saved_latents: list[tuple[int, torch.Tensor]] = []

        if self.save_steps_dir is not None:
            steps_dir = Path(self.save_steps_dir)
            steps_dir.mkdir(parents=True, exist_ok=True)
            params["callback_on_step_end"] = make_step_callback(
                saved_latents, self.save_steps_interval
            )
            params["callback_on_step_end_tensor_inputs"] = ["latents"]

        with torch.inference_mode():
            output = self._pipeline(
                image=images,
                prompt=prompt,
                **params,
            )

        if saved_latents and self.save_steps_dir is not None:
            decode_and_save_step_latents(
                saved_latents, self._pipeline,
                output.images[0], Path(self.save_steps_dir),
            )

        result = output.images[0]
        if result.size != target_size:
            result = result.resize(target_size, PILImage.LANCZOS)
        return result
