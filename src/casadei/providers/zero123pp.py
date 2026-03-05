"""Zero123++ multi-view generation provider.

Generates 6 views from a single image using sudo-ai/zero123plus-v1.2.
Output is a 3x2 grid that gets split into 6 individual view images.
"""

from __future__ import annotations

import logging

import torch
from PIL import Image as PILImage

from casadei import MODELS_DIR
from casadei.models.base import ModelCapability, ImageConstraint
from casadei.models.image_to_multiview import ImageToMultiViewModel

try:
    from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
except ImportError:
    DiffusionPipeline = None
    EulerAncestralDiscreteScheduler = None

logger = logging.getLogger(__name__)


class _CPUOffloadVAE:
    """Transparent proxy that runs VAE on CPU but reports a CUDA device.

    CUBLAS has persistent bugs with linear layers inside the VAE attention
    on Blackwell GPUs (SM 110) / CUDA 13.0 — affects float16, bfloat16,
    *and* float32.  Running the lightweight VAE encode/decode on CPU
    side-steps the issue while the heavy UNet denoising stays on GPU.

    The proxy reports ``device`` as CUDA so the pipeline routes tensors
    correctly (e.g. ``image_2`` must reach the CLIP vision encoder on CUDA).
    """

    def __init__(
        self, vae: torch.nn.Module, report_device: torch.device, report_dtype: torch.dtype,
    ) -> None:
        object.__setattr__(self, "_vae", vae.to(device="cpu", dtype=torch.float32))
        object.__setattr__(self, "_report_device", report_device)
        object.__setattr__(self, "_report_dtype", report_dtype)

    # ------------------------------------------------------------------
    # Properties the pipeline reads for device/dtype routing
    # ------------------------------------------------------------------
    @property
    def device(self) -> torch.device:
        return self._report_device

    @property
    def dtype(self) -> torch.dtype:
        return self._report_dtype

    # ------------------------------------------------------------------
    # Core ops — move inputs to CPU, run, return CPU results
    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor, **kwargs):
        return self._vae.encode(x.cpu().float(), **kwargs)

    def decode(self, x: torch.Tensor, **kwargs):
        return self._vae.decode(x.cpu().float(), **kwargs)

    # ------------------------------------------------------------------
    # Proxy everything else (config, scaling_factor, etc.)
    # ------------------------------------------------------------------
    def __getattr__(self, name: str):
        return getattr(self._vae, name)

# Grid layout: 3 columns x 2 rows, each tile 320x320
GRID_COLS = 3
GRID_ROWS = 2
TILE_SIZE = 320


def _split_grid(grid_image: PILImage.Image) -> list[PILImage.Image]:
    """Split a 3x2 grid image into 6 individual tiles."""
    w, h = grid_image.size
    tile_w = w // GRID_COLS
    tile_h = h // GRID_ROWS
    tiles = []
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            box = (col * tile_w, row * tile_h, (col + 1) * tile_w, (row + 1) * tile_h)
            tiles.append(grid_image.crop(box))
    return tiles


class Zero123PlusPlus(ImageToMultiViewModel):
    """sudo-ai/zero123plus-v1.2 multi-view generation.

    Takes a single image and generates 6 views at different angles
    (azimuths 30, 90, 150, 210, 270, 330 degrees).
    Output is a 3x2 grid that gets split into individual frames.
    """

    MODEL_ID = "sudo-ai/zero123plus-v1.2"
    CUSTOM_PIPELINE = "sudo-ai/zero123plus-pipeline"

    capability = ModelCapability(
        inputs=[
            ImageConstraint(
                required=True,
                max_count=1,
                supported_formats=["png", "jpg", "jpeg", "webp"],
            ),
        ],
        outputs=[
            ImageConstraint(required=True, max_count=6),
        ],
    )

    DEFAULT_PARAMS = {
        "num_inference_steps": 28,
    }

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = None

    def load_model(self) -> None:
        if DiffusionPipeline is None:
            raise ImportError(
                "diffusers is required. Install: pip install diffusers transformers"
            )

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        pipe = DiffusionPipeline.from_pretrained(
            self.MODEL_ID,
            custom_pipeline=self.CUSTOM_PIPELINE,
            torch_dtype=torch_dtype,
            cache_dir=MODELS_DIR,
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
        if torch.cuda.is_available():
            pipe.to("cuda")

            # CUBLAS crashes in VAE attention on Blackwell (SM 110) / CUDA 13.0.
            # Offload VAE to CPU.  The proxy reports the pipeline's device/dtype
            # so the custom pipeline routes image_2 to CUDA+bfloat16 for CLIP.
            cuda_dev = torch.device("cuda")
            pipe.vae = _CPUOffloadVAE(
                pipe.vae, report_device=cuda_dev, report_dtype=torch_dtype,
            )

            # Patch encode_condition_image so encoded latents land on CUDA
            # in the pipeline's dtype for the UNet.
            def _encode_cond_image(image: torch.Tensor, _pipe=pipe) -> torch.Tensor:
                lat = _pipe.vae.encode(image).latent_dist.sample()
                return lat.to(device=cuda_dev, dtype=torch_dtype)

            pipe.encode_condition_image = _encode_cond_image
            logger.info("Zero123++ loaded (model: %s, vae: CPU offload)", self.MODEL_ID)
        else:
            logger.info("Zero123++ loaded (model: %s, cpu)", self.MODEL_ID)

        self._pipeline = pipe

    def unload_model(self) -> None:
        self._pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Zero123++ unloaded")

    def _generate_views(
        self,
        image: PILImage.Image,
        num_views: int = 6,
        **kwargs,
    ) -> list[PILImage.Image]:
        if self._pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Zero123++ expects a square input, recommended >= 320x320
        if image.size[0] != image.size[1]:
            size = max(image.size)
            canvas = PILImage.new("RGB", (size, size), (255, 255, 255))
            canvas.paste(image, ((size - image.size[0]) // 2, (size - image.size[1]) // 2))
            image = canvas

        if image.size[0] < TILE_SIZE:
            image = image.resize((TILE_SIZE, TILE_SIZE), PILImage.LANCZOS)

        params = {**self.DEFAULT_PARAMS, **kwargs}

        with torch.inference_mode():
            result = self._pipeline(image, **params)

        grid_image = result.images[0]
        views = _split_grid(grid_image)
        return views[:num_views]
