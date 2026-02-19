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
import numpy as np
import torch
from pathlib import Path
from PIL import Image as PILImage

from casadei import MODELS_DIR
from casadei.models.base import ModelCapability, ImageConstraint, TextConstraint
from casadei.models.image_edit import ImageEditModel

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

        target_size = images[0].size

        # Collect latents during inference for deferred VAE decode
        saved_latents: list[tuple[int, torch.Tensor]] = []

        if self.save_steps_dir is not None:
            steps_dir = Path(self.save_steps_dir)
            steps_dir.mkdir(parents=True, exist_ok=True)
            interval = self.save_steps_interval

            def callback_on_step_end(_pipe, step_index, timestep, callback_kwargs):
                if step_index % interval != 0:
                    return callback_kwargs
                saved_latents.append(
                    (step_index, callback_kwargs["latents"].detach().cpu())
                )
                print(f"  step {step_index} latents captured")
                return callback_kwargs

            params["callback_on_step_end"] = callback_on_step_end
            params["callback_on_step_end_tensor_inputs"] = ["latents"]

        with torch.inference_mode():
            output = self._pipeline(
                image=images,
                prompt=prompt,
                **params,
            )

        # Decode saved latents through VAE now that the transformer is offloaded
        if saved_latents and self.save_steps_dir is not None:
            steps_dir = Path(self.save_steps_dir)
            pipe = self._pipeline
            device = pipe._execution_device
            dtype = pipe.vae.dtype
            z_dim = pipe.vae.config.get("z_dim", 16)
            _mean = pipe.vae.config["latents_mean"]
            _std = pipe.vae.config["latents_std"]

            # 5D mean/std for the Qwen causal-3D VAE: (1, C, 1, 1, 1)
            latents_mean = torch.tensor(
                _mean, device=device, dtype=dtype
            ).view(1, z_dim, 1, 1, 1)
            latents_std = torch.tensor(
                _std, device=device, dtype=dtype
            ).view(1, z_dim, 1, 1, 1)

            # Use output image dimensions for correct unpack geometry
            result_img = output.images[0]
            img_h, img_w = result_img.size[1], result_img.size[0]

            # Unpack all saved latents from packed transformer format
            # (B, num_patches, C*4) -> (B, C, 1, H_lat, W_lat)
            unpacked = []
            step_indices = []
            for step_index, lat in saved_latents:
                lat = lat.to(device=device, dtype=dtype)
                lat_5d = pipe._unpack_latents(
                    lat, img_h, img_w, pipe.vae_scale_factor
                )
                unpacked.append(lat_5d)
                step_indices.append(step_index)

            # Batch decode: single VAE GPU transfer instead of N
            batch = torch.cat(unpacked, dim=0)
            denormed = batch * latents_std + latents_mean
            del unpacked, batch

            print(f"  Decoding {len(step_indices)} step latents via VAE...")
            with torch.no_grad():
                decoded = pipe.vae.decode(denormed, return_dict=False)[0]
            del denormed

            # Remove temporal dim and postprocess to PIL
            images_4d = decoded[:, :, 0]
            for i, step_index in enumerate(step_indices):
                img = pipe.image_processor.postprocess(
                    images_4d[i : i + 1], output_type="pil"
                )[0]
                img.save(steps_dir / f"step_{step_index:03d}.png")
                print(f"  step {step_index} saved")

            del saved_latents, decoded, images_4d
            torch.cuda.empty_cache()

        result = output.images[0]
        if result.size != target_size:
            result = result.resize(target_size, PILImage.LANCZOS)
        return result
