"""Google Veo 3.1 video generation provider.

API-based video generation model. No local weights —
requires GEMINI_API_KEY environment variable.

Model code: veo-3.1-fast-generate-preview
Input: text prompt + optional image (first frame) + optional reference images
Output: video (8s, 720p/1080p/4k, 16:9 or 9:16)

Supports:
- Text-to-video generation
- Image-to-video generation (optional first frame)
- Reference images (up to 3) for subject consistency
- First/last frame interpolation
- Aspect ratio control (16:9, 9:16)
- Resolution control (720p, 1080p, 4k)
"""

from __future__ import annotations

import logging
import time

import numpy as np
from PIL import Image as PILImage

from casadei.media import (
    ImageMedia,
    MediaBundle,
    TextMedia,
    VideoMedia,
)
from casadei.models.base import (
    AIModel,
    ImageConstraint,
    ModelCapability,
    TextConstraint,
    VideoConstraint,
)

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

logger = logging.getLogger(__name__)

MODEL_ID = "veo-3.1-fast-generate-preview"

_DEFAULT_POLL_INTERVAL = 10  # seconds


class VeoVideoGenerate(AIModel):
    """Google Veo 3.1 — API-based video generation model.

    Generates 8-second videos from a text prompt and optional image.
    Reads GEMINI_API_KEY from the environment via the google-genai SDK.
    No local model weights or GPU required.

    Supports reference images (up to 3) to preserve subject appearance,
    and first/last frame interpolation for controlled video generation.
    """

    MODEL_ID = MODEL_ID

    capability = ModelCapability(
        inputs=[
            TextConstraint(required=True, max_count=1),
            ImageConstraint(
                required=False,
                max_count=4,  # 1 first frame + up to 3 reference images
                supported_formats=["png", "jpg", "jpeg", "webp"],
            ),
        ],
        outputs=[
            VideoConstraint(required=True, max_count=1),
        ],
    )

    DEFAULT_PARAMS: dict = {
        "aspect_ratio": "16:9",
        "poll_interval": _DEFAULT_POLL_INTERVAL,
    }

    def __init__(self) -> None:
        super().__init__()
        self._client = None

    def load_model(self) -> None:
        if genai is None:
            raise ImportError(
                "google-genai is required. Install: pip install google-genai"
            )
        self._client = genai.Client()
        logger.info("Gemini client initialized (model: %s)", self.MODEL_ID)

    def unload_model(self) -> None:
        self._client = None

    def run(self, inputs: MediaBundle, **kwargs) -> MediaBundle:
        """Run video generation.

        Special input keys:
        - "prompt": text prompt (required)
        - "image": first frame image (optional)
        - "last_frame": last frame for interpolation (optional)
        - "reference_0", "reference_1", "reference_2": reference images (optional)

        Kwargs:
        - aspect_ratio: "16:9" (default) or "9:16"
        - resolution: "720p", "1080p", or "4k"
        - number_of_videos: int
        - poll_interval: seconds between status polls (default 10)
        - reference_images: list of PIL images (alternative to input keys)
        - reference_type: "ASSET" (default) or "STYLE" for reference images
        """
        errors = self.capability.validate_inputs(inputs)
        if errors:
            raise ValueError("; ".join(errors))

        if self._client is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Extract prompt
        prompt = ""
        if "prompt" in inputs.items and isinstance(inputs.items["prompt"], TextMedia):
            prompt = inputs.items["prompt"].text
        if not prompt:
            text_items = [
                v for v in inputs.items.values() if isinstance(v, TextMedia)
            ]
            if text_items:
                prompt = text_items[0].text

        # Extract optional first-frame image
        first_frame = None
        if "image" in inputs.items and isinstance(inputs.items["image"], ImageMedia):
            first_frame = inputs.items["image"].image

        # Extract optional last-frame image
        last_frame = None
        if "last_frame" in inputs.items and isinstance(inputs.items["last_frame"], ImageMedia):
            last_frame = inputs.items["last_frame"].image

        # Extract reference images from input keys
        reference_pil_images = []
        for key in ("reference_0", "reference_1", "reference_2"):
            if key in inputs.items and isinstance(inputs.items[key], ImageMedia):
                reference_pil_images.append(inputs.items[key].image)

        # Build config from kwargs merged with defaults
        merged = {**self.DEFAULT_PARAMS, **kwargs}
        poll_interval = merged.pop("poll_interval", _DEFAULT_POLL_INTERVAL)
        reference_type = merged.pop("reference_type", "ASSET")

        # Allow passing reference images directly via kwargs
        kwarg_refs = merged.pop("reference_images", None)
        if kwarg_refs and not reference_pil_images:
            reference_pil_images = list(kwarg_refs)

        config_kwargs = {}
        if "aspect_ratio" in merged:
            config_kwargs["aspect_ratio"] = merged.pop("aspect_ratio")
        if "resolution" in merged:
            config_kwargs["resolution"] = merged.pop("resolution")
        if "number_of_videos" in merged:
            config_kwargs["number_of_videos"] = merged.pop("number_of_videos")

        # Build reference images config
        if reference_pil_images:
            ref_objects = []
            for ref_img in reference_pil_images[:3]:
                ref_objects.append(
                    types.VideoGenerationReferenceImage(
                        image=types.Image(
                            image_bytes=self._pil_to_bytes(ref_img),
                            mime_type="image/png",
                        ),
                        reference_type=reference_type,
                    )
                )
            config_kwargs["reference_images"] = ref_objects

        # Add last frame to config if provided
        if last_frame is not None:
            config_kwargs["last_frame"] = types.Image(
                image_bytes=self._pil_to_bytes(last_frame),
                mime_type="image/png",
            )

        config = types.GenerateVideosConfig(**config_kwargs) if config_kwargs else None

        # Start video generation
        generate_kwargs = {
            "model": self.MODEL_ID,
            "prompt": prompt,
        }
        if first_frame is not None:
            generate_kwargs["image"] = types.Image(
                image_bytes=self._pil_to_bytes(first_frame),
                mime_type="image/png",
            )
        if config is not None:
            generate_kwargs["config"] = config

        logger.info(
            "Starting video generation (refs=%d, first_frame=%s, last_frame=%s)",
            len(reference_pil_images),
            first_frame is not None,
            last_frame is not None,
        )
        operation = self._client.models.generate_videos(**generate_kwargs)

        # Poll until done
        while not operation.done:
            logger.info("Waiting for video generation to complete...")
            time.sleep(poll_interval)
            operation = self._client.operations.get(operation)

        # Download the generated video
        response = operation.response
        logger.info("Operation response: %s", response)
        if not response or not response.generated_videos:
            raise RuntimeError(
                f"Video generation failed or was filtered. Response: {response}"
            )
        generated_video = response.generated_videos[0]
        self._client.files.download(file=generated_video.video)

        # Convert to numpy frames via temporary file
        video_bytes = generated_video.video.video_bytes
        frames = self._bytes_to_frames(video_bytes)

        return MediaBundle(items={
            "video": VideoMedia.from_frames(frames),
        })

    @staticmethod
    def _pil_to_bytes(image: PILImage.Image) -> bytes:
        """Convert a PIL image to PNG bytes."""
        import io

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _bytes_to_frames(video_bytes: bytes) -> list[np.ndarray]:
        """Decode raw video bytes into a list of numpy frames."""
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(video_bytes)
            tmp_path = Path(tmp.name)

        try:
            from diffusers.utils import load_video

            frames_pil = load_video(str(tmp_path))
            return [np.array(f) for f in frames_pil]
        finally:
            tmp_path.unlink(missing_ok=True)
