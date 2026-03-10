"""Gemini 3.1 Flash Lite vision-language model provider.

Google's API-based lightweight multimodal model. No local weights —
requires GEMINI_API_KEY environment variable.

Model code: gemini-3.1-flash-lite-preview
Input: 0–14 images + text prompt
Output: text response
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from PIL import Image as PILImage

from casadei.models.base import ModelCapability, ImageConstraint, TextConstraint
from casadei.models.vision_language import VisionLanguageModel
from casadei.providers.gemini_pricing import extract_token_usage

try:
    from google import genai
except ImportError:
    genai = None

logger = logging.getLogger(__name__)

MODEL_ID = "gemini-3.1-flash-lite-preview"


class GeminiFlashLite(VisionLanguageModel):
    """Google Gemini 3.1 Flash Lite — API-based vision-language model.

    Accepts up to 14 images and a text prompt, produces a text response.
    Reads GEMINI_API_KEY from the environment via the google-genai SDK.
    No local model weights or GPU required.
    """

    MODEL_ID = MODEL_ID

    capability = ModelCapability(
        inputs=[
            ImageConstraint(
                required=False,
                max_count=14,
                supported_formats=["png", "jpg", "jpeg", "webp"],
            ),
            TextConstraint(required=True, max_count=1),
        ],
        outputs=[
            TextConstraint(required=True, max_count=1),
        ],
    )

    DEFAULT_PARAMS: dict = {}

    def __init__(self) -> None:
        super().__init__()
        self._client = None
        self.last_token_usage: dict[str, int] | None = None
        self.last_thinking: str | None = None

    def load_model(self) -> None:
        if genai is None:
            raise ImportError(
                "google-genai is required. Install: pip install google-genai"
            )
        self._client = genai.Client()
        logger.info("Gemini client initialized (model: %s)", self.MODEL_ID)

    def unload_model(self) -> None:
        self._client = None

    def _generate_text(
        self,
        images: list[PILImage.Image],
        prompt: str,
        **kwargs,
    ) -> str:
        if self._client is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        contents = [prompt] + images

        config: dict = {
            "temperature": 0.7,
            "thinking_config": {"thinking_budget": 10000},
        }
        if "response_mime_type" in kwargs:
            config["response_mime_type"] = kwargs["response_mime_type"]
        if "response_json_schema" in kwargs:
            config["response_json_schema"] = kwargs["response_json_schema"]

        response = self._client.models.generate_content(
            model=self.MODEL_ID,
            contents=contents,
            config=config,
        )
        self.last_token_usage = extract_token_usage(
            getattr(response, "usage_metadata", None)
        )
        thinking_parts: list[str] = []
        try:
            for part in response.candidates[0].content.parts:
                if getattr(part, "thought", False) and part.text:
                    thinking_parts.append(part.text)
        except Exception:
            pass
        self.last_thinking = "\n".join(thinking_parts) if thinking_parts else None
        return response.text or ""

    def _generate_text_streaming(
        self,
        images: list[PILImage.Image],
        prompt: str,
        **kwargs,
    ) -> Iterator[str]:
        if self._client is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        contents = [prompt] + images

        last_chunk = None
        for chunk in self._client.models.generate_content_stream(
            model=self.MODEL_ID,
            contents=contents,
            config={"temperature": 0.7},
        ):
            last_chunk = chunk
            if chunk.text:
                yield chunk.text
        self.last_token_usage = extract_token_usage(
            getattr(last_chunk, "usage_metadata", None) if last_chunk else None
        )
