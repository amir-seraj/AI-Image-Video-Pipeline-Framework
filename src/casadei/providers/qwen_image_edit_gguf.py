"""Qwen Image Edit GGUF variant — Q8_0 transformer via diffusers GGUF loader.

Loads the transformer weights from a local GGUF file using diffusers'
GGUFQuantizationConfig, then assembles the full pipeline from the base model
(VAE, text encoder, scheduler, etc.).  The GGUF transformer uses 8-bit integer
weights, cutting VRAM roughly in half versus BF16 while staying accurate.

Requirements:
  - pip install gguf          (Python GGUF reader used by diffusers)
  - GGUF snapshot present at models/models--unsloth--Qwen-Image-Edit-2511-GGUF
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from PIL import Image as PILImage

from casadei import MODELS_DIR
from casadei.models.base import ImageConstraint, ModelCapability, TextConstraint
from casadei.models.image_edit import ImageEditModel

logger = logging.getLogger(__name__)

# Snapshot that contains the Q8_0 GGUF file
_GGUF_REPO_DIR = (
    Path(MODELS_DIR)
    / "models--unsloth--Qwen-Image-Edit-2511-GGUF"
    / "snapshots"
)
_GGUF_FILENAME = "qwen-image-edit-2511-Q8_0.gguf"

# Base model supplies VAE, scheduler, image processor, tokeniser, etc.
_BASE_MODEL_ID = "Qwen/Qwen-Image-Edit-2511"

try:
    from diffusers import (
        GGUFQuantizationConfig,
        QwenImageEditPlusPipeline,
        QwenImageTransformer2DModel,
    )
except ImportError:
    GGUFQuantizationConfig = None  # type: ignore[assignment,misc]
    QwenImageEditPlusPipeline = None  # type: ignore[assignment,misc]
    QwenImageTransformer2DModel = None  # type: ignore[assignment,misc]


def _find_gguf_path() -> Path:
    """Return the absolute path to the GGUF file.

    Searches all snapshot sub-directories so this is robust to hash changes.
    Falls back to any .gguf file found if the expected filename is absent.
    """
    if not _GGUF_REPO_DIR.is_dir():
        raise FileNotFoundError(
            f"GGUF repo not found at {_GGUF_REPO_DIR}. "
            "Download with: huggingface-cli download unsloth/Qwen-Image-Edit-2511-GGUF"
        )
    # Prefer exact filename match
    matches = list(_GGUF_REPO_DIR.rglob(_GGUF_FILENAME))
    if not matches:
        matches = list(_GGUF_REPO_DIR.rglob("*.gguf"))
    if not matches:
        raise FileNotFoundError(
            f"No .gguf files found under {_GGUF_REPO_DIR}"
        )
    return matches[0]


# Cached base model snapshots directory (houses the transformer/ sub-folder)
_BASE_CACHE_DIR = (
    Path(MODELS_DIR)
    / "models--Qwen--Qwen-Image-Edit-2511"
    / "snapshots"
)


def _find_transformer_config_dir() -> Path:
    """Return the local transformer/ config directory from the cached base model.

    from_single_file needs a config source so it knows the transformer
    architecture without downloading anything from HF.
    """
    if _BASE_CACHE_DIR.is_dir():
        matches = list(_BASE_CACHE_DIR.rglob("transformer/config.json"))
        if matches:
            return matches[0].parent

    # Fallback: let diffusers resolve it from HF (requires network)
    logger.warning(
        "Local transformer config not found under %s; "
        "falling back to HF model ID %s (requires network).",
        _BASE_CACHE_DIR,
        _BASE_MODEL_ID,
    )
    return Path(_BASE_MODEL_ID)  # type: ignore[return-value]


class QwenImageEditGGUF(ImageEditModel):
    """Qwen/Qwen-Image-Edit-2511 — GGUF Q8_0 transformer.

    The transformer is loaded from a local GGUF file via diffusers'
    GGUFQuantizationConfig.  All other pipeline components (VAE, scheduler,
    image processor) are loaded from the base HuggingFace model.
    """

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        if QwenImageEditPlusPipeline is None or QwenImageTransformer2DModel is None:
            raise ImportError(
                "diffusers with QwenImageEditPlusPipeline and "
                "QwenImageTransformer2DModel is required. "
                "Install: pip install git+https://github.com/huggingface/diffusers"
            )
        if GGUFQuantizationConfig is None:
            raise ImportError(
                "diffusers GGUFQuantizationConfig is not available. "
                "Upgrade diffusers: pip install --upgrade diffusers"
            )

        gguf_path = _find_gguf_path()
        logger.info("Loading GGUF transformer from %s", gguf_path)

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        # Resolve the transformer config directory from the locally-cached base
        # model so that from_single_file knows the architecture without hitting HF.
        transformer_config_dir = _find_transformer_config_dir()
        logger.info("Using transformer config from %s", transformer_config_dir)

        # Load transformer from GGUF; weights stay as quantised integers,
        # compute is performed in bfloat16 (dequant on the fly).
        quant_config = GGUFQuantizationConfig(compute_dtype=torch_dtype)
        transformer = QwenImageTransformer2DModel.from_single_file(
            str(gguf_path),
            config=str(transformer_config_dir),
            quantization_config=quant_config,
            torch_dtype=torch_dtype,
        )

        logger.info("Loading pipeline from base model %s (GGUF transformer injected)", _BASE_MODEL_ID)
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            _BASE_MODEL_ID,
            transformer=transformer,
            torch_dtype=torch_dtype,
            cache_dir=MODELS_DIR,
        )

        if torch.cuda.is_available():
            pipe.enable_model_cpu_offload()

        self._pipeline = pipe

    def unload_model(self) -> None:
        self._pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

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

        import numpy as np

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

            latents_mean = torch.tensor(_mean, device=device, dtype=dtype).view(
                1, z_dim, 1, 1, 1
            )
            latents_std = torch.tensor(_std, device=device, dtype=dtype).view(
                1, z_dim, 1, 1, 1
            )

            result_img = output.images[0]
            img_h, img_w = result_img.size[1], result_img.size[0]

            unpacked = []
            step_indices = []
            for step_index, lat in saved_latents:
                lat = lat.to(device=device, dtype=dtype)
                lat_5d = pipe._unpack_latents(lat, img_h, img_w, pipe.vae_scale_factor)
                unpacked.append(lat_5d)
                step_indices.append(step_index)

            batch = torch.cat(unpacked, dim=0)
            denormed = batch * latents_std + latents_mean
            del unpacked, batch

            print(f"  Decoding {len(step_indices)} step latents via VAE...")
            with torch.no_grad():
                decoded = pipe.vae.decode(denormed, return_dict=False)[0]
            del denormed

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
