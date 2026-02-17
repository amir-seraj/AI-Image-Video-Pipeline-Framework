"""Media types that flow through pipelines."""

from __future__ import annotations

import re
from pathlib import Path
from string import Template
from typing import Any

from PIL import Image as PILImage
from pydantic import BaseModel, ConfigDict, field_validator


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


class VideoMedia(Media):
    """A video referenced by file path."""

    path: Path

    @field_validator("path")
    @classmethod
    def path_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"Video path does not exist: {v}")
        return v


class MediaBundle(Media):
    """A named collection of media items.

    Can hold multiple items of any type under different keys.
    For example: multiple images, multiple texts, mixed media.
    """

    items: dict[str, Media]

    def __getitem__(self, key: str) -> Media:
        return self.items[key]
