"""Qwen Image Edit FP8 (float8_e4m3fn) quantized model provider.

Uses torchao FP8 quantization + torch.compile on the transformer for actual
FP8 tensor core compute via torch._scaled_mm, giving real speedup over BF16.
"""

from __future__ import annotations

import logging
import numpy as np
import torch
from pathlib import Path
from PIL import Image as PILImage

from casadei import MODELS_DIR
from casadei.models.base import ModelCapability, ImageConstraint, TextConstraint
from casadei.models.image_edit import ImageEditModel

logger = logging.getLogger(__name__)

FP8_CACHE_DIR = Path(MODELS_DIR) / "fp8_cache"

try:
    from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel
except ImportError:
    QwenImageEditPlusPipeline = None
    QwenImageTransformer2DModel = None

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


class QwenImageEditFP8(ImageEditModel):
    """FP8-quantized Qwen/Qwen-Image-Edit-2511 model.

    Loads the base BF16 pipeline, then applies torchao FP8 weight-only
    quantization to the transformer. This replaces nn.Linear layers with
    FP8-aware versions that use torch._scaled_mm for native FP8 tensor core
    compute, giving real speedup from reduced memory bandwidth and FP8 matmuls.
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
        if quantize_ is None or Float8WeightOnlyConfig is None:
            raise ImportError(
                "torchao is required for FP8 quantization. "
                "Install: pip install torchao"
            )

        # Load full pipeline in BF16
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            self.BASE_MODEL_ID, torch_dtype=torch.bfloat16, cache_dir=MODELS_DIR
        )

        # Move to CUDA before quantizing. Using pipe.to() instead of
        # enable_model_cpu_offload() because torchao's Float8Tensor subclass
        # is incompatible with accelerate's CPU offload hooks (they fail on
        # cross-device storage aliasing). On Jetson unified memory, CPU
        # offloading provides no memory benefit anyway.
        if torch.cuda.is_available():
            pipe.to("cuda")

            # FP8 quantization with disk cache for the state_dict
            fp8_cache_path = FP8_CACHE_DIR / "qwen_image_edit_fp8_state.pt"
            self._load_or_quantize(pipe, fp8_cache_path)

            # torch.compile fuses dequant + matmul into optimized kernels,
            # without it FP8 is actually slower than BF16 due to eager
            # dequantization overhead on every linear layer forward pass.
            if _HAS_TRITON:
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
        # Apply quantize_ to set up FP8 module structure
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

        # Save the freshly quantized state_dict
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
