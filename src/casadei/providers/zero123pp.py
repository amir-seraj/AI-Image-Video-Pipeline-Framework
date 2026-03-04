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
        self._pipeline = pipe
        logger.info("Zero123++ loaded (model: %s)", self.MODEL_ID)

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
