"""Gemini 3.1 Flash Image Preview provider ("Nano Banana 2").

Google's API-based image generation and editing model. No local weights —
requires GEMINI_API_KEY environment variable.

Model code: gemini-3.1-flash-image-preview
Input: up to 10 images + text prompt
Output: 1 edited/generated image
"""

from __future__ import annotations

import io
import logging

from PIL import Image as PILImage

from casadei.models.base import ModelCapability, ImageConstraint, TextConstraint
from casadei.models.image_edit import ImageEditModel
from casadei.providers.gemini_pricing import extract_token_usage

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:
    genai = None
    genai_types = None

logger = logging.getLogger(__name__)

MODEL_ID = "gemini-3.1-flash-image-preview"
MAX_INPUT_SIZE = 1024

# Aspect ratios supported by the Gemini image API (w:h)
_SUPPORTED_RATIOS: list[tuple[int, int]] = [
    (1, 1), (1, 4), (1, 8),
    (2, 3), (3, 2), (3, 4), (4, 3),
    (4, 5), (5, 4),
    (8, 1), (9, 16), (16, 9), (21, 9),
]


def _find_ratio(w: int, h: int) -> tuple[int, int]:
    """Return the closest supported aspect ratio tuple for given dimensions."""
    target = w / h
    return min(_SUPPORTED_RATIOS, key=lambda r: abs(r[0] / r[1] - target))


def _pad_to_ratio(img: PILImage.Image, ratio: tuple[int, int]) -> PILImage.Image:
    """Scale image so its longest side = MAX_INPUT_SIZE, then pad the shorter
    side with white to exactly match the given aspect ratio.

    The image is never cropped — only expanded with white padding.
    Returns a canvas whose longest side is exactly MAX_INPUT_SIZE and whose
    dimensions match ratio exactly (up to integer rounding).
    """
    wr, hr = ratio
    orig_w, orig_h = img.size

    # Compute canvas dimensions that contain the image without cropping
    if orig_w / orig_h <= wr / hr:
        # Image is narrower than ratio → pad width
        canvas_h = orig_h
        canvas_w = round(orig_h * wr / hr)
    else:
        # Image is wider than ratio → pad height
        canvas_w = orig_w
        canvas_h = round(orig_w * hr / wr)

    # Scale canvas so longest side = MAX_INPUT_SIZE
    scale = MAX_INPUT_SIZE / max(canvas_w, canvas_h)
    final_w = round(canvas_w * scale)
    final_h = round(canvas_h * scale)

    # Scale the original image by the same factor
    img_w = round(orig_w * scale)
    img_h = round(orig_h * scale)
    scaled = img.resize((img_w, img_h), PILImage.LANCZOS)

    # Paste centered on a white canvas
    canvas = PILImage.new("RGB", (final_w, final_h), (255, 255, 255))
    canvas.paste(scaled, ((final_w - img_w) // 2, (final_h - img_h) // 2))
    return canvas


class GeminiFlashImageEdit(ImageEditModel):
    """Google Gemini 3.1 Flash Image Preview — API-based image editor.

    Accepts up to 10 reference images and a text prompt. Produces 1 edited image.
    Reads GEMINI_API_KEY from the environment via the google-genai SDK.
    No local model weights or GPU required.

    Before sending to the API each input image is scaled so its longest side
    is 1024px, then white-padded to the closest supported Gemini aspect ratio.
    The same ratio and "1K" size are requested for the output, so the model
    receives and returns images with consistent dimensions.
    """

    MODEL_ID = MODEL_ID

    capability = ModelCapability(
        inputs=[
            ImageConstraint(
                required=True,
                max_count=10,
                supported_formats=["png", "jpg", "jpeg", "webp"],
            ),
            TextConstraint(required=True, max_count=1),
        ],
        outputs=[
            ImageConstraint(required=True, max_count=1),
        ],
    )

    DEFAULT_TEMPERATURE = 1.0  # Gemini API default
    DEFAULT_PARAMS: dict = {}

    def __init__(self) -> None:
        super().__init__()
        self._client = None
        self.last_token_usage: dict[str, int] | None = None

    def load_model(self) -> None:
        if genai is None:
            raise ImportError(
                "google-genai is required. Install: pip install google-genai"
            )
        self._client = genai.Client()
        logger.info("Gemini client initialized (model: %s)", self.MODEL_ID)

    def unload_model(self) -> None:
        self._client = None

    def _edit(
        self,
        images: list[PILImage.Image],
        prompt: str,
        negative_prompt: str,
        **kwargs,
    ) -> PILImage.Image:
        if self._client is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Pick the aspect ratio that best fits the primary (first) image
        ratio = _find_ratio(*images[0].size)
        aspect_ratio_str = f"{ratio[0]}:{ratio[1]}"

        # Pad every input to that ratio at MAX_INPUT_SIZE on the longest side
        padded = [_pad_to_ratio(img, ratio) for img in images]
        target_size = padded[0].size

        logger.debug(
            "Input %s → padded %s  ratio %s",
            images[0].size, target_size, aspect_ratio_str,
        )

        temperature = kwargs.get("temperature", self.DEFAULT_TEMPERATURE)

        response = self._client.models.generate_content(
            model=self.MODEL_ID,
            contents=[prompt] + padded,
            config=genai_types.GenerateContentConfig(
                temperature=temperature,
                image_config=genai_types.ImageConfig(
                    image_size="1K",
                    aspect_ratio=aspect_ratio_str,
                ),
            ),
        )
        self.last_token_usage = extract_token_usage(
            getattr(response, "usage_metadata", None)
        )

        for part in response.parts:
            if part.inline_data is not None:
                result = PILImage.open(io.BytesIO(part.inline_data.data))
                if result.size != target_size:
                    result = result.resize(target_size, PILImage.LANCZOS)
                return result

        raise RuntimeError(
            "No image returned by Gemini API. "
            "The model may have refused the request or returned text only."
        )
