"""Gemini Flash multi-view generation provider.

API-based multi-view generation using Google Gemini. Sends 6 sequential
requests with angle-specific prompts. No local GPU required.
"""

from __future__ import annotations

import io
import logging

from PIL import Image as PILImage

from casadei.models.base import ModelCapability, ImageConstraint
from casadei.models.image_to_multiview import ImageToMultiViewModel

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

logger = logging.getLogger(__name__)

MODEL_ID = "gemini-3.1-flash-image-preview"

# Viewing angles for 6-view generation (degrees)
VIEW_ANGLES = [0, 60, 120, 180, 240, 300]

VIEW_PROMPTS = [
    "Show this exact product from a front view (0 degrees). Maintain the exact same appearance, materials, colors, and details. Plain white background. Product photography style.",
    "Show this exact product rotated 60 degrees to the right. Maintain the exact same appearance, materials, colors, and details. Plain white background. Product photography style.",
    "Show this exact product rotated 120 degrees to the right. Maintain the exact same appearance, materials, colors, and details. Plain white background. Product photography style.",
    "Show this exact product from the back view (180 degrees). Maintain the exact same appearance, materials, colors, and details. Plain white background. Product photography style.",
    "Show this exact product rotated 240 degrees to the right. Maintain the exact same appearance, materials, colors, and details. Plain white background. Product photography style.",
    "Show this exact product rotated 300 degrees to the right. Maintain the exact same appearance, materials, colors, and details. Plain white background. Product photography style.",
]


class GeminiFlashMultiView(ImageToMultiViewModel):
    """Google Gemini Flash — API-based multi-view generation.

    Sends 6 sequential API calls with angle-specific prompts.
    No local GPU required. Reads GEMINI_API_KEY from environment.
    """

    MODEL_ID = MODEL_ID

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

    DEFAULT_PARAMS: dict = {}

    def __init__(self) -> None:
        super().__init__()
        self._client = None

    def load_model(self) -> None:
        if genai is None:
            raise ImportError(
                "google-genai is required. Install: pip install google-genai"
            )
        self._client = genai.Client()
        logger.info("Gemini multi-view client initialized (model: %s)", self.MODEL_ID)

    def unload_model(self) -> None:
        self._client = None

    def _generate_views(
        self,
        image: PILImage.Image,
        num_views: int = 6,
        **kwargs,
    ) -> list[PILImage.Image]:
        if self._client is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Resize input to max 1024 on longest side
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (round(image.size[0] * ratio), round(image.size[1] * ratio))
            image = image.resize(new_size, PILImage.LANCZOS)

        views = []
        for i in range(min(num_views, len(VIEW_PROMPTS))):
            prompt = VIEW_PROMPTS[i]

            logger.debug("Generating view %d/%d (angle %d\u00b0)", i + 1, num_views, VIEW_ANGLES[i])

            response = self._client.models.generate_content(
                model=self.MODEL_ID,
                contents=[prompt, image],
                config=genai_types.GenerateContentConfig(
                    image_config=genai_types.ImageConfig(
                        image_size="1K",
                    ),
                ),
            )

            view_image = None
            for part in response.parts:
                if part.inline_data is not None:
                    view_image = PILImage.open(io.BytesIO(part.inline_data.data))
                    break

            if view_image is None:
                raise RuntimeError(
                    f"No image returned for view {i + 1} (angle {VIEW_ANGLES[i]}\u00b0). "
                    "The model may have refused the request."
                )

            views.append(view_image)

        return views
