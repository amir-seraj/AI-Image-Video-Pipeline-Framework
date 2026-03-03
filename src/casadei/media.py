"""Media types that flow through pipelines."""

from __future__ import annotations

import re
from pathlib import Path
from string import Template
from typing import Any

from PIL import Image as PILImage
import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator


class Media(BaseModel):
    """Base for all media flowing through the pipeline."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class ImageMedia(Media):
    """An image with optional format metadata."""

    image: PILImage.Image
    format: str = "png"

    @property
    def width(self) -> int:
        return self.image.size[0]

    @property
    def height(self) -> int:
        return self.image.size[1]

    def save(self, path: Path) -> None:
        self.image.save(path)

    @classmethod
    def load(cls, path: Path) -> ImageMedia:
        img = PILImage.open(path)
        img.load()
        suffix = path.suffix.lstrip(".") or "png"
        return cls(image=img, format=suffix)


class TextMedia(Media):
    """A text string with optional $variable template support.

    Use $variable syntax for placeholders. Call .fill() to substitute values.
    Check .is_complete to verify all variables have been filled.
    Check .variables to see which variables remain unfilled.
    """

    text: str

    @property
    def variables(self) -> set[str]:
        """Return the set of unfilled $variable names."""
        matches = re.findall(r'(?<!\$)\$(?:\{(\w+)\}|(\w+))', self.text)
        return {m[0] or m[1] for m in matches}

    @property
    def is_complete(self) -> bool:
        """True if no unfilled $variables remain."""
        return len(self.variables) == 0

    def fill(self, **kwargs: Any) -> TextMedia:
        """Return a new TextMedia with $variables substituted.

        Uses safe_substitute so unfilled variables remain as-is.
        """
        template = Template(self.text)
        return TextMedia(text=template.safe_substitute(**kwargs))


class EmbeddingMedia(Media):
    """A list of numeric embeddings (vectors).

    Stored as a list of lists of floats. Each inner list corresponds to
    the embedding of a single document/text.
    """

    embeddings: list[list[float]]

    @property
    def batch_size(self) -> int:
        return len(self.embeddings)

    @property
    def dimension(self) -> int:
        if not self.embeddings:
            return 0
        return len(self.embeddings[0])


class VideoMedia(Media):
    """A video stored as file path, in-memory frames, or both.

    At least one of ``path`` or ``frames`` must be provided.
    Frames are numpy arrays of shape (H, W, 3) with uint8 dtype,
    matching the output format of diffusers video pipelines.
    """

    path: Path | None = None
    frames: list[Any] | None = None
    fps: int = 16
    format: str = "mp4"

    @model_validator(mode="after")
    def _must_have_path_or_frames(self) -> VideoMedia:
        if self.path is None and self.frames is None:
            raise ValueError("VideoMedia requires at least one of 'path' or 'frames'")
        if self.path is not None and not self.path.exists() and self.frames is None:
            raise ValueError(f"Video path does not exist: {self.path}")
        return self

    @property
    def frame_count(self) -> int:
        if self.frames is not None:
            return len(self.frames)
        return 0

    @property
    def duration_seconds(self) -> float:
        if self.frames is not None and self.fps > 0:
            return len(self.frames) / self.fps
        return 0.0

    def save(self, path: Path) -> Path:
        """Export in-memory frames to a video file."""
        if self.frames is None:
            raise ValueError("No in-memory frames to save")
        from diffusers.utils import export_to_video
        export_to_video(self.frames, str(path), fps=self.fps)
        self.path = path
        return path

    @classmethod
    def load(cls, path: Path, fps: int = 16) -> VideoMedia:
        """Load a video file into in-memory frames."""
        from diffusers.utils import load_video
        frames_pil = load_video(str(path))
        frames_np = [np.array(f) for f in frames_pil]
        return cls(path=path, frames=frames_np, fps=fps)

    @classmethod
    def from_frames(cls, frames: list, fps: int = 16) -> VideoMedia:
        """Create VideoMedia from a list of numpy array frames."""
        return cls(frames=frames, fps=fps)


class MediaBundle(Media):
    """A named collection of media items.

    Can hold multiple items of any type under different keys.
    For example: multiple images, multiple texts, mixed media.
    """

    items: dict[str, Media]

    def __getitem__(self, key: str) -> Media:
        return self.items[key]

