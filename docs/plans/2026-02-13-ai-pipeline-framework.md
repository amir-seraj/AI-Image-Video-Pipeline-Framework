# AI Pipeline Framework Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a flexible, composable pipeline framework for AI image/video generation and editing with pluggable model providers, reusable agents, workflow composition, execution logging, and pipeline visualization.

**Architecture:** A layered system: Media types (with template support) define data flowing through pipelines. Model classes form a hierarchy (AIModel → ImageEditModel → QwenImageEdit) encoding capabilities and constraints. Agents wrap model classes with specific configs and are serializable to YAML. Pipelines chain steps (agent, code, or sub-pipeline) via input/output mapping with shared context, and pipelines compose into larger pipelines. Execution logging tracks timing and metrics per step and per pipeline. Visualization renders pipeline structure as diagrams.

**Tech Stack:** Python 3.13, conda (local env), Pydantic v2 (validation/serialization), PyYAML, Pillow, PyTorch + diffusers (model inference), pytest

**Hardware:** NVIDIA RTX 4060 (8GB VRAM) — bfloat16 required for model loading

---

## Task 1: Project Scaffolding

**Files:**
- Create: `environment.yml`
- Create: `pyproject.toml`
- Create: `src/casadei/__init__.py`
- Create: `src/casadei/media.py` (empty placeholder)
- Create: `src/casadei/models/__init__.py`
- Create: `src/casadei/models/base.py` (empty placeholder)
- Create: `src/casadei/models/image_edit.py` (empty placeholder)
- Create: `src/casadei/models/registry.py` (empty placeholder)
- Create: `src/casadei/providers/__init__.py`
- Create: `src/casadei/providers/qwen_image_edit.py` (empty placeholder)
- Create: `src/casadei/agent.py` (empty placeholder)
- Create: `src/casadei/pipeline.py` (empty placeholder)
- Create: `src/casadei/logging.py` (empty placeholder)
- Create: `src/casadei/visualization.py` (empty placeholder)
- Create: `agents/.gitkeep`
- Create: `workflows/.gitkeep`
- Create: `tests/conftest.py`
- Create: `.gitignore`

**Step 1: Initialize git repo**

Run: `cd /home/innovina/Documents/casadei && git init`

**Step 2: Create `.gitignore`**

```
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
dist/
build/
.env
*.pt
*.bin
*.safetensors
.conda_env/
```

**Step 3: Create `environment.yml`**

```yaml
name: casadei
prefix: ./.conda_env
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.13
  - pytorch
  - torchvision
  - pytorch-cuda=12.6
  - pip
  - pip:
    - pydantic>=2.0
    - pyyaml
    - pillow
    - "diffusers @ git+https://github.com/huggingface/diffusers"
    - pytest
    - pytest-cov
```

**Step 4: Create conda environment**

Run: `cd /home/innovina/Documents/casadei && conda env create -f environment.yml`
Expected: Environment created at `.conda_env/`

**Step 5: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "casadei"
version = "0.1.0"
description = "Flexible AI image/video pipeline framework"
requires-python = ">=3.13"
dependencies = [
    "pydantic>=2.0",
    "pyyaml",
    "pillow",
    "torch",
    "diffusers",
]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

**Step 6: Create directory structure with placeholder files**

Create all `__init__.py` files and empty placeholder modules listed above. Each placeholder should contain only a docstring:

```python
"""Module docstring."""
```

For `src/casadei/__init__.py`:
```python
"""Casadei — Flexible AI pipeline framework."""
```

For `tests/conftest.py`:
```python
"""Shared test fixtures."""
```

**Step 7: Install package in editable mode**

Run: `cd /home/innovina/Documents/casadei && conda run -p .conda_env pip install -e ".[dev]"`
Expected: Successfully installed casadei

**Step 8: Verify setup**

Run: `cd /home/innovina/Documents/casadei && conda run -p .conda_env pytest --co -q`
Expected: "no tests ran" (no test files yet, but pytest finds the project)

**Step 9: Commit**

```bash
git add -A
git commit -m "chore: scaffold project with conda env, pyproject.toml, and directory structure"
```

---

## Task 2: Media Types (with TextMedia Templates)

**Files:**
- Create: `src/casadei/media.py`
- Create: `tests/test_media.py`

TextMedia supports `$variable` template syntax (Python `string.Template`). A TextMedia can be a complete text or a template that needs `.fill()` before use. MediaBundle can hold multiple TextMedia items under different keys (prompt, negative_prompt, caption, etc.).

**Step 1: Write failing tests for media types**

```python
# tests/test_media.py
import pytest
from pathlib import Path
from PIL import Image as PILImage

from casadei.media import ImageMedia, TextMedia, VideoMedia, MediaBundle


class TestImageMedia:
    def test_create_from_pil(self):
        img = PILImage.new("RGB", (512, 512), color="red")
        media = ImageMedia(image=img)
        assert media.image.size == (512, 512)
        assert media.format == "png"

    def test_create_with_format(self):
        img = PILImage.new("RGB", (256, 256))
        media = ImageMedia(image=img, format="jpg")
        assert media.format == "jpg"

    def test_save_and_load(self, tmp_path):
        img = PILImage.new("RGB", (100, 100), color="blue")
        media = ImageMedia(image=img)
        path = tmp_path / "test.png"
        media.save(path)
        loaded = ImageMedia.load(path)
        assert loaded.image.size == (100, 100)

    def test_width_height(self):
        img = PILImage.new("RGB", (640, 480))
        media = ImageMedia(image=img)
        assert media.width == 640
        assert media.height == 480


class TestTextMedia:
    def test_create(self):
        media = TextMedia(text="hello world")
        assert media.text == "hello world"

    def test_empty_text(self):
        media = TextMedia(text="")
        assert media.text == ""

    def test_is_complete_plain_text(self):
        media = TextMedia(text="no variables here")
        assert media.is_complete is True

    def test_is_complete_with_unfilled_variables(self):
        media = TextMedia(text="I want to add $item in the image")
        assert media.is_complete is False

    def test_variables_returns_variable_names(self):
        media = TextMedia(text="Add $item near the $location")
        assert media.variables == {"item", "location"}

    def test_variables_empty_for_plain_text(self):
        media = TextMedia(text="plain text")
        assert media.variables == set()

    def test_fill_replaces_variables(self):
        media = TextMedia(text="I want to add $item in the image")
        filled = media.fill(item="a red car")
        assert filled.text == "I want to add a red car in the image"
        assert filled.is_complete is True

    def test_fill_multiple_variables(self):
        media = TextMedia(text="Replace $source with $target")
        filled = media.fill(source="the cat", target="a dog")
        assert filled.text == "Replace the cat with a dog"

    def test_fill_partial_leaves_remaining(self):
        media = TextMedia(text="Add $item near the $location")
        filled = media.fill(item="tree")
        assert filled.text == "Add tree near the $location"
        assert filled.is_complete is False
        assert filled.variables == {"location"}

    def test_fill_returns_new_instance(self):
        media = TextMedia(text="Add $item")
        filled = media.fill(item="tree")
        assert media.text == "Add $item"  # original unchanged
        assert filled.text == "Add tree"

    def test_fill_unknown_variable_ignored(self):
        media = TextMedia(text="Hello $name")
        filled = media.fill(name="world", extra="ignored")
        assert filled.text == "Hello world"


class TestVideoMedia:
    def test_create(self, tmp_path):
        path = tmp_path / "video.mp4"
        path.touch()
        media = VideoMedia(path=path)
        assert media.path == path

    def test_nonexistent_path_raises(self):
        with pytest.raises(ValueError):
            VideoMedia(path=Path("/nonexistent/video.mp4"))


class TestMediaBundle:
    def test_create_bundle(self):
        img = PILImage.new("RGB", (100, 100))
        bundle = MediaBundle(items={
            "image": ImageMedia(image=img),
            "prompt": TextMedia(text="edit this"),
        })
        assert "image" in bundle.items
        assert "prompt" in bundle.items

    def test_getitem(self):
        text = TextMedia(text="hello")
        bundle = MediaBundle(items={"greeting": text})
        assert bundle["greeting"].text == "hello"

    def test_missing_key_raises(self):
        bundle = MediaBundle(items={})
        with pytest.raises(KeyError):
            bundle["missing"]

    def test_bundle_with_multiple_texts(self):
        bundle = MediaBundle(items={
            "prompt": TextMedia(text="make it blue"),
            "negative_prompt": TextMedia(text="blurry, ugly"),
            "caption": TextMedia(text="A beautiful landscape"),
        })
        assert len([v for v in bundle.items.values() if isinstance(v, TextMedia)]) == 3

    def test_bundle_with_multiple_images(self):
        bundle = MediaBundle(items={
            "source": ImageMedia(image=PILImage.new("RGB", (100, 100))),
            "reference": ImageMedia(image=PILImage.new("RGB", (200, 200))),
            "mask": ImageMedia(image=PILImage.new("L", (100, 100))),
        })
        assert len([v for v in bundle.items.values() if isinstance(v, ImageMedia)]) == 3

    def test_bundle_with_template_text(self):
        bundle = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100))),
            "prompt": TextMedia(text="Add $item to the scene"),
        })
        prompt = bundle["prompt"]
        assert isinstance(prompt, TextMedia)
        assert prompt.is_complete is False
        filled = prompt.fill(item="a sunset")
        assert filled.is_complete is True
```

**Step 2: Run tests to verify they fail**

Run: `conda run -p .conda_env pytest tests/test_media.py -v`
Expected: FAIL — `ImportError: cannot import name 'ImageMedia' from 'casadei.media'`

**Step 3: Implement media types**

```python
# src/casadei/media.py
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
        # Match $identifier or ${identifier} but not $$
        return set(re.findall(r'(?<!\$)\$(?:\{(\w+)\}|(\w+))', self.text)
                   |> (lambda matches: [m[0] or m[1] for m in matches]))

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
```

**IMPORTANT:** The `variables` property above has a duplicate — only keep the second one (with `re.findall` returning a set comprehension). The first one with the pipe operator is invalid Python. The correct implementation:

```python
    @property
    def variables(self) -> set[str]:
        """Return the set of unfilled $variable names."""
        matches = re.findall(r'(?<!\$)\$(?:\{(\w+)\}|(\w+))', self.text)
        return {m[0] or m[1] for m in matches}
```

**Step 4: Run tests to verify they pass**

Run: `conda run -p .conda_env pytest tests/test_media.py -v`
Expected: All 21 tests PASS

**Step 5: Commit**

```bash
git add src/casadei/media.py tests/test_media.py
git commit -m "feat: add media types with TextMedia template support ($variable syntax)"
```

---

## Task 3: Model Base Classes and Capability System

**Files:**
- Create: `src/casadei/models/base.py`
- Create: `tests/test_model_base.py`

**Step 1: Write failing tests**

```python
# tests/test_model_base.py
import pytest
from PIL import Image as PILImage

from casadei.media import ImageMedia, TextMedia, MediaBundle
from casadei.models.base import (
    AIModel,
    ModelCapability,
    ImageConstraint,
    TextConstraint,
    VideoConstraint,
)


class TestModelCapability:
    def test_create_capability(self):
        cap = ModelCapability(
            inputs=[
                ImageConstraint(required=True, max_count=2),
                TextConstraint(required=True),
            ],
            outputs=[
                ImageConstraint(required=True, max_count=1),
            ],
        )
        assert len(cap.inputs) == 2
        assert len(cap.outputs) == 1

    def test_image_constraint_defaults(self):
        c = ImageConstraint()
        assert c.required is True
        assert c.max_count == 1
        assert c.max_width is None
        assert c.max_height is None
        assert "png" in c.supported_formats

    def test_text_constraint_defaults(self):
        c = TextConstraint()
        assert c.required is True
        assert c.max_length is None

    def test_validate_inputs_valid(self):
        cap = ModelCapability(
            inputs=[
                ImageConstraint(required=True, max_count=1),
                TextConstraint(required=True),
            ],
            outputs=[ImageConstraint()],
        )
        bundle = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100))),
            "prompt": TextMedia(text="edit this"),
        })
        errors = cap.validate_inputs(bundle)
        assert errors == []

    def test_validate_inputs_missing_required(self):
        cap = ModelCapability(
            inputs=[
                ImageConstraint(required=True, max_count=1),
                TextConstraint(required=True),
            ],
            outputs=[ImageConstraint()],
        )
        bundle = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100))),
        })
        errors = cap.validate_inputs(bundle)
        assert len(errors) > 0
        assert "TextMedia" in errors[0] or "text" in errors[0].lower()

    def test_validate_inputs_too_many(self):
        cap = ModelCapability(
            inputs=[ImageConstraint(required=True, max_count=1)],
            outputs=[ImageConstraint()],
        )
        bundle = MediaBundle(items={
            "img1": ImageMedia(image=PILImage.new("RGB", (100, 100))),
            "img2": ImageMedia(image=PILImage.new("RGB", (100, 100))),
        })
        errors = cap.validate_inputs(bundle)
        assert len(errors) > 0
        assert "too many" in errors[0].lower() or "max" in errors[0].lower()

    def test_validate_multiple_text_inputs_allowed(self):
        cap = ModelCapability(
            inputs=[TextConstraint(required=True, max_count=3)],
            outputs=[TextConstraint()],
        )
        bundle = MediaBundle(items={
            "prompt": TextMedia(text="do this"),
            "negative": TextMedia(text="not this"),
            "style": TextMedia(text="like this"),
        })
        errors = cap.validate_inputs(bundle)
        assert errors == []


class TestAIModel:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            AIModel()

    def test_subclass_must_define_capability(self):
        class BadModel(AIModel):
            capability = None

            def load_model(self):
                pass

            def unload_model(self):
                pass

            def run(self, inputs):
                return inputs

        with pytest.raises(TypeError):
            BadModel()

    def test_valid_subclass(self):
        class GoodModel(AIModel):
            capability = ModelCapability(
                inputs=[TextConstraint(required=True)],
                outputs=[TextConstraint()],
            )

            def load_model(self):
                self._loaded = True

            def unload_model(self):
                self._loaded = False

            def run(self, inputs: MediaBundle) -> MediaBundle:
                return inputs

        model = GoodModel()
        assert model.capability is not None
```

**Step 2: Run tests to verify they fail**

Run: `conda run -p .conda_env pytest tests/test_model_base.py -v`
Expected: FAIL — `ImportError`

**Step 3: Implement model base classes**

```python
# src/casadei/models/base.py
"""Base model classes and capability system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from casadei.media import Media, ImageMedia, TextMedia, VideoMedia, MediaBundle


class MediaConstraint(BaseModel):
    """Base constraint for a media type in model capabilities."""

    required: bool = True
    max_count: int = 1

    def media_type(self) -> type[Media]:
        raise NotImplementedError

    def find_matching(self, bundle: MediaBundle) -> list[Media]:
        mt = self.media_type()
        return [v for v in bundle.items.values() if isinstance(v, mt)]


class ImageConstraint(MediaConstraint):
    """Constraints on image inputs/outputs."""

    max_width: int | None = None
    max_height: int | None = None
    supported_formats: list[str] = ["png", "jpg", "jpeg", "webp"]

    def media_type(self) -> type[Media]:
        return ImageMedia


class TextConstraint(MediaConstraint):
    """Constraints on text inputs/outputs."""

    max_length: int | None = None

    def media_type(self) -> type[Media]:
        return TextMedia


class VideoConstraint(MediaConstraint):
    """Constraints on video inputs/outputs."""

    max_duration_seconds: float | None = None
    max_width: int | None = None
    max_height: int | None = None

    def media_type(self) -> type[Media]:
        return VideoMedia


class ModelCapability(BaseModel):
    """Defines what a model accepts and produces."""

    inputs: list[MediaConstraint]
    outputs: list[MediaConstraint]

    def validate_inputs(self, bundle: MediaBundle) -> list[str]:
        errors = []
        for constraint in self.inputs:
            matches = constraint.find_matching(bundle)
            mt_name = constraint.media_type().__name__
            if constraint.required and len(matches) == 0:
                errors.append(f"Required {mt_name} input is missing")
            if len(matches) > constraint.max_count:
                errors.append(
                    f"Too many {mt_name} inputs: got {len(matches)}, max {constraint.max_count}"
                )
        return errors


class AIModel(ABC):
    """Abstract base for all AI models."""

    capability: ModelCapability

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if not getattr(cls, '__abstractmethods__', None):
            if not isinstance(getattr(cls, 'capability', None), ModelCapability):
                raise TypeError(
                    f"{cls.__name__} must define 'capability' as a ModelCapability instance"
                )

    def __init__(self) -> None:
        if not isinstance(getattr(self, 'capability', None), ModelCapability):
            raise TypeError(
                f"{type(self).__name__} must define 'capability' as a ModelCapability instance"
            )

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights into memory/GPU."""

    @abstractmethod
    def unload_model(self) -> None:
        """Unload model weights to free resources."""

    @abstractmethod
    def run(self, inputs: MediaBundle) -> MediaBundle:
        """Run inference. Inputs should match self.capability."""
```

**Step 4: Update `src/casadei/models/__init__.py`**

```python
"""Model base classes and capability system."""

from casadei.models.base import (
    AIModel,
    ModelCapability,
    MediaConstraint,
    ImageConstraint,
    TextConstraint,
    VideoConstraint,
)

__all__ = [
    "AIModel",
    "ModelCapability",
    "MediaConstraint",
    "ImageConstraint",
    "TextConstraint",
    "VideoConstraint",
]
```

**Step 5: Run tests to verify they pass**

Run: `conda run -p .conda_env pytest tests/test_model_base.py -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add src/casadei/models/ tests/test_model_base.py
git commit -m "feat: add AIModel base, ModelCapability, and constraint system"
```

---

## Task 4: ImageEditModel Base Class

**Files:**
- Create: `src/casadei/models/image_edit.py`
- Create: `tests/test_image_edit.py`

**Step 1: Write failing tests**

```python
# tests/test_image_edit.py
import pytest
from PIL import Image as PILImage

from casadei.media import ImageMedia, TextMedia, MediaBundle
from casadei.models.base import ModelCapability, ImageConstraint, TextConstraint
from casadei.models.image_edit import ImageEditModel


class TestImageEditModel:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            ImageEditModel()

    def test_subclass_inherits_defaults(self):
        class MockEditor(ImageEditModel):
            capability = ModelCapability(
                inputs=[
                    ImageConstraint(required=True, max_count=1),
                    TextConstraint(required=True),
                ],
                outputs=[ImageConstraint(required=True, max_count=1)],
            )

            def load_model(self):
                pass

            def unload_model(self):
                pass

            def _edit(self, images, prompt, negative_prompt, **kwargs):
                return images[0]

        editor = MockEditor()
        assert isinstance(editor, ImageEditModel)

    def test_run_delegates_to_edit(self):
        class MockEditor(ImageEditModel):
            capability = ModelCapability(
                inputs=[
                    ImageConstraint(required=True, max_count=1),
                    TextConstraint(required=True),
                ],
                outputs=[ImageConstraint(required=True, max_count=1)],
            )

            def load_model(self):
                pass

            def unload_model(self):
                pass

            def _edit(self, images, prompt, negative_prompt, **kwargs):
                return PILImage.new("RGB", (100, 100), color="green")

        editor = MockEditor()
        bundle = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100), color="red")),
            "prompt": TextMedia(text="make it green"),
        })
        result = editor.run(bundle)
        assert "image" in result.items
        output_img = result["image"]
        assert isinstance(output_img, ImageMedia)
        assert output_img.image.getpixel((50, 50)) == (0, 128, 0)

    def test_run_validates_inputs(self):
        class MockEditor(ImageEditModel):
            capability = ModelCapability(
                inputs=[
                    ImageConstraint(required=True, max_count=1),
                    TextConstraint(required=True),
                ],
                outputs=[ImageConstraint(required=True, max_count=1)],
            )

            def load_model(self):
                pass

            def unload_model(self):
                pass

            def _edit(self, images, prompt, negative_prompt, **kwargs):
                return images[0]

        editor = MockEditor()
        bundle = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100))),
        })
        with pytest.raises(ValueError, match="[Rr]equired.*[Tt]ext"):
            editor.run(bundle)

    def test_run_with_multiple_text_inputs(self):
        """prompt and negative_prompt are both TextMedia in the bundle."""
        class MockEditor(ImageEditModel):
            capability = ModelCapability(
                inputs=[
                    ImageConstraint(required=True, max_count=1),
                    TextConstraint(required=True, max_count=2),
                ],
                outputs=[ImageConstraint(required=True, max_count=1)],
            )

            def load_model(self):
                pass

            def unload_model(self):
                pass

            def _edit(self, images, prompt, negative_prompt, **kwargs):
                assert prompt == "make it green"
                assert negative_prompt == "blurry"
                return PILImage.new("RGB", (100, 100), color="green")

        editor = MockEditor()
        bundle = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100))),
            "prompt": TextMedia(text="make it green"),
            "negative_prompt": TextMedia(text="blurry"),
        })
        result = editor.run(bundle)
        assert "image" in result.items
```

**Step 2: Run tests to verify they fail**

Run: `conda run -p .conda_env pytest tests/test_image_edit.py -v`
Expected: FAIL — `ImportError`

**Step 3: Implement ImageEditModel**

```python
# src/casadei/models/image_edit.py
"""Base class for image editing models."""

from __future__ import annotations

from abc import abstractmethod

from PIL import Image as PILImage

from casadei.media import ImageMedia, TextMedia, MediaBundle
from casadei.models.base import AIModel


class ImageEditModel(AIModel):
    """Base class for models that edit images given a text prompt.

    Subclasses must implement `_edit()` with model-specific inference logic
    and define `capability` with their specific constraints.
    """

    @abstractmethod
    def _edit(
        self,
        images: list[PILImage.Image],
        prompt: str,
        negative_prompt: str,
        **kwargs,
    ) -> PILImage.Image:
        """Run model-specific image editing inference."""

    def run(self, inputs: MediaBundle) -> MediaBundle:
        errors = self.capability.validate_inputs(inputs)
        if errors:
            raise ValueError("; ".join(errors))

        images = [
            v.image for v in inputs.items.values() if isinstance(v, ImageMedia)
        ]

        # Extract prompt and negative_prompt by key name convention,
        # falling back to positional ordering of TextMedia items
        prompt = ""
        negative_prompt = ""
        if "prompt" in inputs.items and isinstance(inputs.items["prompt"], TextMedia):
            prompt = inputs.items["prompt"].text
        if "negative_prompt" in inputs.items and isinstance(inputs.items["negative_prompt"], TextMedia):
            negative_prompt = inputs.items["negative_prompt"].text

        # Fallback: if no named keys, use first/second TextMedia found
        if not prompt:
            text_items = [v for v in inputs.items.values() if isinstance(v, TextMedia)]
            if text_items:
                prompt = text_items[0].text
            if len(text_items) > 1 and not negative_prompt:
                negative_prompt = text_items[1].text

        result_image = self._edit(
            images=images,
            prompt=prompt,
            negative_prompt=negative_prompt,
        )
        return MediaBundle(items={
            "image": ImageMedia(image=result_image),
        })
```

**Step 4: Run tests to verify they pass**

Run: `conda run -p .conda_env pytest tests/test_image_edit.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/casadei/models/image_edit.py tests/test_image_edit.py
git commit -m "feat: add ImageEditModel base class with multi-text input support"
```

---

## Task 5: QwenImageEdit Provider

**Files:**
- Create: `src/casadei/providers/qwen_image_edit.py`
- Create: `tests/test_qwen.py`

**Step 1: Write failing tests (unit tests with mocked pipeline)**

```python
# tests/test_qwen.py
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image as PILImage

from casadei.media import ImageMedia, TextMedia, MediaBundle
from casadei.models.base import ImageConstraint, TextConstraint
from casadei.providers.qwen_image_edit import QwenImageEdit


class TestQwenImageEditCapability:
    def test_accepts_up_to_2_images(self):
        img_constraints = [
            c for c in QwenImageEdit.capability.inputs
            if isinstance(c, ImageConstraint)
        ]
        assert len(img_constraints) == 1
        assert img_constraints[0].max_count == 2

    def test_requires_text_prompt(self):
        txt_constraints = [
            c for c in QwenImageEdit.capability.inputs
            if isinstance(c, TextConstraint)
        ]
        assert len(txt_constraints) >= 1
        assert txt_constraints[0].required is True

    def test_outputs_single_image(self):
        img_constraints = [
            c for c in QwenImageEdit.capability.outputs
            if isinstance(c, ImageConstraint)
        ]
        assert len(img_constraints) == 1
        assert img_constraints[0].max_count == 1

    def test_is_image_edit_model(self):
        from casadei.models.image_edit import ImageEditModel
        assert issubclass(QwenImageEdit, ImageEditModel)


class TestQwenImageEditInference:
    @patch("casadei.providers.qwen_image_edit.QwenImageEditPlusPipeline")
    def test_load_model(self, mock_pipeline_cls):
        mock_pipe = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe

        model = QwenImageEdit()
        model.load_model()

        mock_pipeline_cls.from_pretrained.assert_called_once()
        mock_pipe.to.assert_called_once_with("cuda")

    @patch("casadei.providers.qwen_image_edit.QwenImageEditPlusPipeline")
    def test_edit_calls_pipeline(self, mock_pipeline_cls):
        fake_output_img = PILImage.new("RGB", (512, 512), color="green")
        mock_pipe = MagicMock()
        mock_pipe.return_value.images = [fake_output_img]
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe
        mock_pipe.to.return_value = mock_pipe

        model = QwenImageEdit()
        model.load_model()

        input_img = PILImage.new("RGB", (512, 512), color="red")
        result = model._edit(
            images=[input_img],
            prompt="make it green",
            negative_prompt=" ",
        )
        assert result.size == (512, 512)
        mock_pipe.assert_called_once()

    @patch("casadei.providers.qwen_image_edit.QwenImageEditPlusPipeline")
    def test_run_end_to_end(self, mock_pipeline_cls):
        fake_output_img = PILImage.new("RGB", (512, 512), color="blue")
        mock_pipe = MagicMock()
        mock_pipe.return_value.images = [fake_output_img]
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe
        mock_pipe.to.return_value = mock_pipe

        model = QwenImageEdit()
        model.load_model()

        bundle = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (512, 512))),
            "prompt": TextMedia(text="make it blue"),
        })
        result = model.run(bundle)
        assert "image" in result.items
        assert isinstance(result["image"], ImageMedia)

    @patch("casadei.providers.qwen_image_edit.QwenImageEditPlusPipeline")
    def test_unload_model(self, mock_pipeline_cls):
        mock_pipe = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe
        mock_pipe.to.return_value = mock_pipe

        model = QwenImageEdit()
        model.load_model()
        model.unload_model()
        assert model._pipeline is None

    def test_edit_without_load_raises(self):
        model = QwenImageEdit()
        with pytest.raises(RuntimeError, match="[Nn]ot loaded"):
            model._edit(
                images=[PILImage.new("RGB", (100, 100))],
                prompt="test",
                negative_prompt="",
            )
```

**Step 2: Run tests to verify they fail**

Run: `conda run -p .conda_env pytest tests/test_qwen.py -v`
Expected: FAIL — `ImportError`

**Step 3: Implement QwenImageEdit**

```python
# src/casadei/providers/qwen_image_edit.py
"""Qwen Image Edit model provider."""

from __future__ import annotations

import torch
from PIL import Image as PILImage

from casadei.models.base import ModelCapability, ImageConstraint, TextConstraint
from casadei.models.image_edit import ImageEditModel

try:
    from diffusers import QwenImageEditPlusPipeline
except ImportError:
    QwenImageEditPlusPipeline = None


class QwenImageEdit(ImageEditModel):
    """Qwen/Qwen-Image-Edit-2511 model.

    Accepts up to 2 images and a text prompt, produces 1 edited image.
    Requires CUDA GPU with bfloat16 support.
    """

    MODEL_ID = "Qwen/Qwen-Image-Edit-2511"

    capability = ModelCapability(
        inputs=[
            ImageConstraint(required=True, max_count=2, supported_formats=["png", "jpg", "jpeg", "webp"]),
            TextConstraint(required=True),
        ],
        outputs=[
            ImageConstraint(required=True, max_count=1),
        ],
    )

    DEFAULT_PARAMS = {
        "num_inference_steps": 40,
        "guidance_scale": 1.0,
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_images_per_prompt": 1,
    }

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = None

    def load_model(self) -> None:
        if QwenImageEditPlusPipeline is None:
            raise ImportError(
                "diffusers with QwenImageEditPlusPipeline is required. "
                "Install: pip install git+https://github.com/huggingface/diffusers"
            )
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            self.MODEL_ID, torch_dtype=torch.bfloat16
        )
        self._pipeline = pipe.to("cuda")

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

        with torch.inference_mode():
            output = self._pipeline(
                image=images,
                prompt=prompt,
                **params,
            )
        return output.images[0]
```

**Step 4: Update `src/casadei/providers/__init__.py`**

```python
"""Model provider implementations."""

from casadei.providers.qwen_image_edit import QwenImageEdit

__all__ = ["QwenImageEdit"]
```

**Step 5: Run tests to verify they pass**

Run: `conda run -p .conda_env pytest tests/test_qwen.py -v`
Expected: All 8 tests PASS

**Step 6: Commit**

```bash
git add src/casadei/providers/ tests/test_qwen.py
git commit -m "feat: add QwenImageEdit provider with capability constraints"
```

---

## Task 6: Model Registry

**Files:**
- Create: `src/casadei/models/registry.py`
- Create: `tests/test_registry.py`

**Step 1: Write failing tests**

```python
# tests/test_registry.py
import pytest

from casadei.models.registry import ModelRegistry
from casadei.models.base import AIModel, ModelCapability, TextConstraint
from casadei.media import MediaBundle


class TestModelRegistry:
    def setup_method(self):
        self.registry = ModelRegistry()

    def test_register_and_get(self):
        class DummyModel(AIModel):
            capability = ModelCapability(
                inputs=[TextConstraint()], outputs=[TextConstraint()]
            )
            def load_model(self): pass
            def unload_model(self): pass
            def run(self, inputs: MediaBundle) -> MediaBundle: return inputs

        self.registry.register("dummy", DummyModel)
        assert self.registry.get("dummy") is DummyModel

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="dummy"):
            self.registry.get("dummy")

    def test_list_models(self):
        class DummyModel(AIModel):
            capability = ModelCapability(
                inputs=[TextConstraint()], outputs=[TextConstraint()]
            )
            def load_model(self): pass
            def unload_model(self): pass
            def run(self, inputs: MediaBundle) -> MediaBundle: return inputs

        self.registry.register("dummy", DummyModel)
        assert "dummy" in self.registry.list_models()

    def test_builtin_registry_has_qwen(self):
        from casadei.models.registry import default_registry
        cls = default_registry.get("qwen_image_edit")
        from casadei.providers.qwen_image_edit import QwenImageEdit
        assert cls is QwenImageEdit
```

**Step 2: Run tests to verify they fail**

Run: `conda run -p .conda_env pytest tests/test_registry.py -v`
Expected: FAIL — `ImportError`

**Step 3: Implement registry**

```python
# src/casadei/models/registry.py
"""Model class registry for lookup by name."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from casadei.models.base import AIModel


class ModelRegistry:
    """Maps string names to AIModel subclasses."""

    def __init__(self) -> None:
        self._models: dict[str, type[AIModel]] = {}

    def register(self, name: str, model_cls: type[AIModel]) -> None:
        self._models[name] = model_cls

    def get(self, name: str) -> type[AIModel]:
        if name not in self._models:
            raise KeyError(f"Unknown model: '{name}'. Available: {list(self._models)}")
        return self._models[name]

    def list_models(self) -> list[str]:
        return list(self._models.keys())


def _build_default_registry() -> ModelRegistry:
    registry = ModelRegistry()
    from casadei.providers.qwen_image_edit import QwenImageEdit
    registry.register("qwen_image_edit", QwenImageEdit)
    return registry


default_registry = _build_default_registry()
```

**Step 4: Run tests to verify they pass**

Run: `conda run -p .conda_env pytest tests/test_registry.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/casadei/models/registry.py tests/test_registry.py
git commit -m "feat: add model registry with built-in Qwen registration"
```

---

## Task 7: Agent System

**Files:**
- Create: `src/casadei/agent.py`
- Create: `tests/test_agent.py`

The Agent now uses TextMedia's template system. When `prompt_template` has `$variables`, they get filled via `TextMedia.fill()`.

**Step 1: Write failing tests**

```python
# tests/test_agent.py
import pytest
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image as PILImage

from casadei.agent import Agent, AgentConfig, load_agent, save_agent
from casadei.media import ImageMedia, TextMedia, MediaBundle


class TestAgentConfig:
    def test_create_config(self):
        config = AgentConfig(
            name="bg_remover",
            model="qwen_image_edit",
            prompt_template="Remove the background from this image",
            params={"num_inference_steps": 30},
        )
        assert config.name == "bg_remover"
        assert config.model == "qwen_image_edit"

    def test_config_defaults(self):
        config = AgentConfig(name="test", model="qwen_image_edit")
        assert config.prompt_template == ""
        assert config.negative_prompt == ""
        assert config.params == {}
        assert config.description == ""

    def test_serialize_to_dict(self):
        config = AgentConfig(
            name="test",
            model="qwen_image_edit",
            prompt_template="do something",
        )
        d = config.model_dump()
        assert d["name"] == "test"
        assert d["model"] == "qwen_image_edit"

    def test_deserialize_from_dict(self):
        d = {
            "name": "test",
            "model": "qwen_image_edit",
            "prompt_template": "do something",
        }
        config = AgentConfig(**d)
        assert config.name == "test"

    def test_template_with_dollar_variables(self):
        config = AgentConfig(
            name="adder",
            model="qwen_image_edit",
            prompt_template="I want to add $item in the image",
        )
        assert config.prompt_template == "I want to add $item in the image"


class TestAgent:
    @patch("casadei.agent.default_registry")
    def test_create_agent_from_config(self, mock_registry):
        mock_model_cls = MagicMock()
        mock_registry.get.return_value = mock_model_cls

        config = AgentConfig(name="test", model="qwen_image_edit")
        agent = Agent(config=config)
        assert agent.config.name == "test"

    @patch("casadei.agent.default_registry")
    def test_agent_fills_template_variables(self, mock_registry):
        mock_model = MagicMock()
        mock_model.run.return_value = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100))),
        })
        mock_model_cls = MagicMock(return_value=mock_model)
        mock_registry.get.return_value = mock_model_cls

        config = AgentConfig(
            name="adder",
            model="qwen_image_edit",
            prompt_template="I want to add $item in the image",
        )
        agent = Agent(config=config)
        agent.load()

        bundle = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100))),
        })
        result = agent.execute(bundle, item="a red car")

        # Verify the prompt was filled and injected
        call_args = mock_model.run.call_args[0][0]
        prompt_items = [v for v in call_args.items.values() if isinstance(v, TextMedia)]
        assert any("a red car" in p.text for p in prompt_items)
        # No $item should remain
        assert all("$item" not in p.text for p in prompt_items)

    @patch("casadei.agent.default_registry")
    def test_agent_passes_raw_prompt_when_no_template(self, mock_registry):
        mock_model = MagicMock()
        mock_model.run.return_value = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100))),
        })
        mock_model_cls = MagicMock(return_value=mock_model)
        mock_registry.get.return_value = mock_model_cls

        config = AgentConfig(name="raw", model="qwen_image_edit")
        agent = Agent(config=config)
        agent.load()

        bundle = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100))),
            "prompt": TextMedia(text="do this specific thing"),
        })
        result = agent.execute(bundle)

        call_args = mock_model.run.call_args[0][0]
        prompt_items = [v for v in call_args.items.values() if isinstance(v, TextMedia)]
        assert any("do this specific thing" in p.text for p in prompt_items)

    @patch("casadei.agent.default_registry")
    def test_agent_incomplete_template_raises(self, mock_registry):
        """If template has unfilled variables after fill(), raise an error."""
        mock_model = MagicMock()
        mock_model_cls = MagicMock(return_value=mock_model)
        mock_registry.get.return_value = mock_model_cls

        config = AgentConfig(
            name="needs_two_vars",
            model="qwen_image_edit",
            prompt_template="Replace $source with $target",
        )
        agent = Agent(config=config)
        agent.load()

        bundle = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100))),
        })
        # Only providing one of two required variables
        with pytest.raises(ValueError, match="unfilled"):
            agent.execute(bundle, source="cat")


class TestAgentPersistence:
    def test_save_and_load(self, tmp_path):
        config = AgentConfig(
            name="bg_remover",
            model="qwen_image_edit",
            description="Removes backgrounds",
            prompt_template="Remove the background, leaving only the subject",
            negative_prompt="blurry",
            params={"num_inference_steps": 30},
        )
        filepath = tmp_path / "bg_remover.yaml"
        save_agent(config, filepath)

        loaded = load_agent(filepath)
        assert loaded.name == "bg_remover"
        assert loaded.model == "qwen_image_edit"
        assert loaded.prompt_template == "Remove the background, leaving only the subject"
        assert loaded.params["num_inference_steps"] == 30

    def test_load_from_agents_directory(self, tmp_path):
        config = AgentConfig(name="test_agent", model="qwen_image_edit")
        save_agent(config, tmp_path / "test_agent.yaml")

        loaded = load_agent(tmp_path / "test_agent.yaml")
        assert loaded.name == "test_agent"

    def test_saved_file_is_valid_yaml(self, tmp_path):
        config = AgentConfig(
            name="test",
            model="qwen_image_edit",
            prompt_template="Add $item to the scene",
        )
        filepath = tmp_path / "test.yaml"
        save_agent(config, filepath)

        with open(filepath) as f:
            data = yaml.safe_load(f)
        assert data["name"] == "test"
        assert data["prompt_template"] == "Add $item to the scene"
```

**Step 2: Run tests to verify they fail**

Run: `conda run -p .conda_env pytest tests/test_agent.py -v`
Expected: FAIL — `ImportError`

**Step 3: Implement agent system**

```python
# src/casadei/agent.py
"""Agent system — configured model instances for reuse in pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from casadei.media import MediaBundle, TextMedia
from casadei.models.registry import default_registry


class AgentConfig(BaseModel):
    """Serializable configuration for an agent.

    prompt_template supports $variable syntax (Python string.Template).
    Variables are filled at execution time via keyword arguments.
    """

    name: str
    model: str
    description: str = ""
    prompt_template: str = ""
    negative_prompt: str = ""
    params: dict[str, Any] = {}


class Agent:
    """A configured model instance ready for use in pipelines.

    Wraps a model class with specific prompts, templates, and parameters.
    Uses TextMedia's $variable template system for prompt filling.
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._model_cls = default_registry.get(config.model)
        self._model = None

    def load(self) -> None:
        self._model = self._model_cls()
        self._model.load_model()

    def unload(self) -> None:
        if self._model is not None:
            self._model.unload_model()
            self._model = None

    def execute(self, inputs: MediaBundle, **template_kwargs: Any) -> MediaBundle:
        if self._model is None:
            raise RuntimeError("Agent not loaded. Call load() first.")

        prepared = self._prepare_inputs(inputs, **template_kwargs)
        return self._model.run(prepared)

    def _prepare_inputs(
        self, inputs: MediaBundle, **template_kwargs: Any
    ) -> MediaBundle:
        items = dict(inputs.items)

        if self.config.prompt_template:
            prompt = TextMedia(text=self.config.prompt_template)
            filled = prompt.fill(**template_kwargs)
            if not filled.is_complete:
                unfilled = filled.variables
                raise ValueError(
                    f"Prompt template has unfilled variables: {unfilled}. "
                    f"Provide them as keyword arguments to execute()."
                )
            items["prompt"] = filled

        if self.config.negative_prompt and "negative_prompt" not in items:
            items["negative_prompt"] = TextMedia(text=self.config.negative_prompt)

        return MediaBundle(items=items)


def save_agent(config: AgentConfig, path: Path) -> None:
    """Save an agent config to a YAML file."""
    with open(path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)


def load_agent(path: Path) -> AgentConfig:
    """Load an agent config from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return AgentConfig(**data)
```

**Step 4: Run tests to verify they pass**

Run: `conda run -p .conda_env pytest tests/test_agent.py -v`
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add src/casadei/agent.py tests/test_agent.py
git commit -m "feat: add Agent system with $variable template support and YAML persistence"
```

---

## Task 8: Pipeline with AgentStep, CodeStep, and PipelineStep

**Files:**
- Create: `src/casadei/pipeline.py`
- Create: `tests/test_pipeline.py`

The pipeline supports three step types:
- **AgentStep** — runs an Agent
- **CodeStep** — runs an arbitrary Python callable
- **PipelineStep** — runs a nested Pipeline (composition)

All share a common `Step` base with `execute(context) -> outputs`.

**Step 1: Write failing tests**

```python
# tests/test_pipeline.py
import pytest
from unittest.mock import MagicMock
from PIL import Image as PILImage

from casadei.media import ImageMedia, TextMedia, Media, MediaBundle
from casadei.pipeline import AgentStep, CodeStep, PipelineStep, Pipeline


class TestAgentStep:
    def test_create_step(self):
        agent = MagicMock()
        step = AgentStep(
            name="edit",
            agent=agent,
            input_map={"image": "source_image"},
            output_map={"image": "edited_image"},
        )
        assert step.name == "edit"
        assert step.input_map == {"image": "source_image"}

    def test_execute_maps_inputs(self):
        agent = MagicMock()
        agent.execute.return_value = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100))),
        })

        step = AgentStep(
            name="edit",
            agent=agent,
            input_map={"image": "source_image"},
            output_map={"image": "result"},
        )

        context = {
            "source_image": ImageMedia(image=PILImage.new("RGB", (200, 200))),
        }
        outputs = step.execute(context)
        assert "result" in outputs

        call_bundle = agent.execute.call_args[0][0]
        assert "image" in call_bundle.items

    def test_passes_template_kwargs(self):
        agent = MagicMock()
        agent.execute.return_value = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100))),
        })

        step = AgentStep(
            name="style",
            agent=agent,
            input_map={"image": "source"},
            output_map={"image": "result"},
            template_kwargs={"style": "watercolor"},
        )

        context = {"source": ImageMedia(image=PILImage.new("RGB", (100, 100)))}
        step.execute(context)
        _, kwargs = agent.execute.call_args
        assert kwargs.get("style") == "watercolor"

    def test_missing_input_raises(self):
        agent = MagicMock()
        step = AgentStep(
            name="edit",
            agent=agent,
            input_map={"image": "nonexistent"},
            output_map={"image": "out"},
        )
        with pytest.raises(KeyError, match="nonexistent"):
            step.execute({})


class TestCodeStep:
    def test_simple_function(self):
        def resize(context: dict[str, Media]) -> dict[str, Media]:
            img = context["image"]
            assert isinstance(img, ImageMedia)
            resized = img.image.resize((64, 64))
            return {"small_image": ImageMedia(image=resized)}

        step = CodeStep(name="resize", fn=resize)
        context = {"image": ImageMedia(image=PILImage.new("RGB", (256, 256)))}
        outputs = step.execute(context)
        assert "small_image" in outputs
        assert outputs["small_image"].width == 64

    def test_code_step_can_access_full_context(self):
        def combine(context: dict[str, Media]) -> dict[str, Media]:
            prompt = context["prompt"]
            suffix = context["suffix"]
            assert isinstance(prompt, TextMedia)
            assert isinstance(suffix, TextMedia)
            combined = TextMedia(text=f"{prompt.text} {suffix.text}")
            return {"full_prompt": combined}

        step = CodeStep(name="combine", fn=combine)
        context = {
            "prompt": TextMedia(text="hello"),
            "suffix": TextMedia(text="world"),
        }
        outputs = step.execute(context)
        assert outputs["full_prompt"].text == "hello world"

    def test_code_step_error_propagates(self):
        def bad_fn(context):
            raise ValueError("something broke")

        step = CodeStep(name="bad", fn=bad_fn)
        with pytest.raises(ValueError, match="something broke"):
            step.execute({})


class TestPipelineStep:
    def test_nested_pipeline(self):
        agent = MagicMock()
        agent.execute.return_value = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100), color="green")),
        })

        inner_pipeline = Pipeline(
            name="inner",
            steps=[AgentStep(
                name="edit",
                agent=agent,
                input_map={"image": "input_image"},
                output_map={"image": "output_image"},
            )],
        )

        step = PipelineStep(
            name="sub_pipeline",
            pipeline=inner_pipeline,
            input_map={"input_image": "raw_image"},
            output_map={"output_image": "processed_image"},
        )

        context = {"raw_image": ImageMedia(image=PILImage.new("RGB", (200, 200)))}
        outputs = step.execute(context)
        assert "processed_image" in outputs


class TestPipeline:
    def test_create_pipeline(self):
        pipeline = Pipeline(name="test")
        assert pipeline.name == "test"
        assert len(pipeline.steps) == 0

    def test_add_step(self):
        pipeline = Pipeline(name="test")
        agent = MagicMock()
        step = AgentStep(name="s", agent=agent, input_map={}, output_map={})
        pipeline.add_step(step)
        assert len(pipeline.steps) == 1

    def test_run_single_agent_step(self):
        output_img = PILImage.new("RGB", (100, 100), color="green")
        agent = MagicMock()
        agent.execute.return_value = MediaBundle(items={
            "image": ImageMedia(image=output_img),
        })

        step = AgentStep(
            name="edit",
            agent=agent,
            input_map={"image": "input_image"},
            output_map={"image": "output_image"},
        )

        pipeline = Pipeline(name="test", steps=[step])
        result = pipeline.run({
            "input_image": ImageMedia(image=PILImage.new("RGB", (200, 200))),
        })
        assert "output_image" in result

    def test_run_chained_steps(self):
        agent1 = MagicMock()
        agent1.execute.return_value = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100), color="blue")),
        })
        agent2 = MagicMock()
        agent2.execute.return_value = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100), color="red")),
        })

        pipeline = Pipeline(name="two_step", steps=[
            AgentStep(
                name="clean",
                agent=agent1,
                input_map={"image": "raw_image"},
                output_map={"image": "clean_image"},
            ),
            AgentStep(
                name="style",
                agent=agent2,
                input_map={"image": "clean_image"},
                output_map={"image": "final_image"},
            ),
        ])
        result = pipeline.run({
            "raw_image": ImageMedia(image=PILImage.new("RGB", (300, 300))),
        })
        assert "final_image" in result
        assert result["final_image"].image.getpixel((50, 50)) == (255, 0, 0)

    def test_mixed_step_types(self):
        """Pipeline with AgentStep, CodeStep, and PipelineStep together."""
        # CodeStep: resize input
        def resize_fn(ctx):
            img = ctx["raw"]
            return {"resized": ImageMedia(image=img.image.resize((256, 256)))}

        # AgentStep: edit image
        agent = MagicMock()
        agent.execute.return_value = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (256, 256), color="blue")),
        })

        # PipelineStep: a nested pipeline that does a final touch
        final_agent = MagicMock()
        final_agent.execute.return_value = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (256, 256), color="green")),
        })
        inner = Pipeline(name="final_touch", steps=[
            AgentStep(
                name="polish",
                agent=final_agent,
                input_map={"image": "edit_input"},
                output_map={"image": "edit_output"},
            ),
        ])

        pipeline = Pipeline(name="full", steps=[
            CodeStep(name="resize", fn=resize_fn),
            AgentStep(
                name="edit",
                agent=agent,
                input_map={"image": "resized"},
                output_map={"image": "edited"},
            ),
            PipelineStep(
                name="final",
                pipeline=inner,
                input_map={"edit_input": "edited"},
                output_map={"edit_output": "final_result"},
            ),
        ])

        result = pipeline.run({
            "raw": ImageMedia(image=PILImage.new("RGB", (1024, 1024))),
        })
        assert "final_result" in result
        assert result["final_result"].image.getpixel((0, 0)) == (0, 128, 0)


class TestPipelineCompose:
    def test_compose_two_pipelines(self):
        agent1 = MagicMock()
        agent1.execute.return_value = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100), color="blue")),
        })
        agent2 = MagicMock()
        agent2.execute.return_value = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100), color="red")),
        })

        p1 = Pipeline(name="first", steps=[
            AgentStep(name="a", agent=agent1,
                      input_map={"image": "input_image"},
                      output_map={"image": "mid_image"}),
        ])
        p2 = Pipeline(name="second", steps=[
            AgentStep(name="b", agent=agent2,
                      input_map={"image": "mid_image"},
                      output_map={"image": "final_image"}),
        ])

        composed = Pipeline.compose("combined", [p1, p2])
        result = composed.run({
            "input_image": ImageMedia(image=PILImage.new("RGB", (200, 200))),
        })
        assert "final_image" in result

    def test_pipeline_load_and_unload(self):
        agent = MagicMock()
        step = AgentStep(name="s", agent=agent, input_map={}, output_map={})
        pipeline = Pipeline(name="test", steps=[step])

        pipeline.load()
        agent.load.assert_called_once()

        pipeline.unload()
        agent.unload.assert_called_once()
```

**Step 2: Run tests to verify they fail**

Run: `conda run -p .conda_env pytest tests/test_pipeline.py -v`
Expected: FAIL — `ImportError`

**Step 3: Implement pipeline**

```python
# src/casadei/pipeline.py
"""Pipeline system — composable chains of agent, code, and sub-pipeline steps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from casadei.agent import Agent
from casadei.media import Media, MediaBundle


class Step(ABC):
    """Base class for all pipeline steps."""

    name: str

    @abstractmethod
    def execute(self, context: dict[str, Media]) -> dict[str, Media]:
        """Execute this step given the pipeline context.

        Args:
            context: All media items available so far in the pipeline.

        Returns:
            New media items to add to the context.
        """


@dataclass
class AgentStep(Step):
    """A step that runs an Agent.

    Maps context keys to agent input names, runs the agent,
    and maps agent output names back to context keys.
    """

    name: str
    agent: Agent
    input_map: dict[str, str]   # agent_input_name -> context_key
    output_map: dict[str, str]  # agent_output_name -> context_key
    template_kwargs: dict[str, Any] = field(default_factory=dict)

    def execute(self, context: dict[str, Media]) -> dict[str, Media]:
        agent_inputs = {}
        for agent_key, context_key in self.input_map.items():
            if context_key not in context:
                raise KeyError(
                    f"AgentStep '{self.name}': input '{context_key}' not found in context. "
                    f"Available: {list(context.keys())}"
                )
            agent_inputs[agent_key] = context[context_key]

        bundle = MediaBundle(items=agent_inputs)
        result = self.agent.execute(bundle, **self.template_kwargs)

        outputs = {}
        for agent_key, context_key in self.output_map.items():
            if agent_key in result.items:
                outputs[context_key] = result[agent_key]
        return outputs


@dataclass
class CodeStep(Step):
    """A step that runs an arbitrary Python function.

    The function receives the full context dict and returns
    a dict of new media items to add to the context.
    """

    name: str
    fn: Callable[[dict[str, Media]], dict[str, Media]]

    def execute(self, context: dict[str, Media]) -> dict[str, Media]:
        return self.fn(context)


@dataclass
class PipelineStep(Step):
    """A step that runs a nested Pipeline.

    Maps context keys into the sub-pipeline's input namespace
    and maps sub-pipeline outputs back to the parent context.
    """

    name: str
    pipeline: Pipeline
    input_map: dict[str, str]   # sub_pipeline_key -> parent_context_key
    output_map: dict[str, str]  # sub_pipeline_key -> parent_context_key

    def execute(self, context: dict[str, Media]) -> dict[str, Media]:
        sub_inputs = {}
        for sub_key, parent_key in self.input_map.items():
            if parent_key not in context:
                raise KeyError(
                    f"PipelineStep '{self.name}': input '{parent_key}' not found in context. "
                    f"Available: {list(context.keys())}"
                )
            sub_inputs[sub_key] = context[parent_key]

        sub_result = self.pipeline.run(sub_inputs)

        outputs = {}
        for sub_key, parent_key in self.output_map.items():
            if sub_key in sub_result:
                outputs[parent_key] = sub_result[sub_key]
        return outputs


@dataclass
class Pipeline:
    """An ordered sequence of steps with shared context.

    Steps execute sequentially. Each step reads from and writes to
    a shared context dict, enabling data flow between steps.
    Steps can be AgentStep, CodeStep, or PipelineStep.
    """

    name: str
    steps: list[Step] = field(default_factory=list)

    def add_step(self, step: Step) -> None:
        self.steps.append(step)

    def run(self, inputs: dict[str, Media]) -> dict[str, Media]:
        context = dict(inputs)
        for step in self.steps:
            outputs = step.execute(context)
            context.update(outputs)
        return context

    def load(self) -> None:
        for step in self.steps:
            if isinstance(step, AgentStep):
                step.agent.load()
            elif isinstance(step, PipelineStep):
                step.pipeline.load()

    def unload(self) -> None:
        for step in self.steps:
            if isinstance(step, AgentStep):
                step.agent.unload()
            elif isinstance(step, PipelineStep):
                step.pipeline.unload()

    @classmethod
    def compose(cls, name: str, pipelines: list[Pipeline]) -> Pipeline:
        """Compose multiple pipelines into one by flattening their steps."""
        all_steps = []
        for p in pipelines:
            all_steps.extend(p.steps)
        return cls(name=name, steps=all_steps)
```

**Step 4: Run tests to verify they pass**

Run: `conda run -p .conda_env pytest tests/test_pipeline.py -v`
Expected: All 13 tests PASS

**Step 5: Commit**

```bash
git add src/casadei/pipeline.py tests/test_pipeline.py
git commit -m "feat: add Pipeline with AgentStep, CodeStep, and PipelineStep"
```

---

## Task 9: Execution Logging

**Files:**
- Create: `src/casadei/logging.py`
- Create: `tests/test_logging.py`

Tracks per-step and per-pipeline execution timing. Each pipeline run produces an `ExecutionLog` with step-level detail.

**Step 1: Write failing tests**

```python
# tests/test_logging.py
import pytest
import time
from unittest.mock import MagicMock
from PIL import Image as PILImage

from casadei.media import ImageMedia, TextMedia, Media, MediaBundle
from casadei.pipeline import AgentStep, CodeStep, PipelineStep, Pipeline
from casadei.logging import ExecutionLog, StepLog, LoggedPipeline


class TestStepLog:
    def test_step_log_fields(self):
        log = StepLog(
            step_name="edit",
            step_type="AgentStep",
            duration_ms=1234.5,
            input_keys=["image", "prompt"],
            output_keys=["image"],
        )
        assert log.step_name == "edit"
        assert log.duration_ms == 1234.5
        assert log.step_type == "AgentStep"

    def test_step_log_str(self):
        log = StepLog(
            step_name="edit",
            step_type="AgentStep",
            duration_ms=150.3,
            input_keys=["image"],
            output_keys=["result"],
        )
        s = str(log)
        assert "edit" in s
        assert "150.3" in s or "150" in s


class TestExecutionLog:
    def test_execution_log_fields(self):
        log = ExecutionLog(
            pipeline_name="test",
            total_duration_ms=5000.0,
            step_logs=[
                StepLog("s1", "AgentStep", 2000.0, ["a"], ["b"]),
                StepLog("s2", "CodeStep", 3000.0, ["b"], ["c"]),
            ],
        )
        assert log.pipeline_name == "test"
        assert log.total_duration_ms == 5000.0
        assert len(log.step_logs) == 2

    def test_execution_log_summary(self):
        log = ExecutionLog(
            pipeline_name="test",
            total_duration_ms=5000.0,
            step_logs=[
                StepLog("s1", "AgentStep", 2000.0, ["a"], ["b"]),
                StepLog("s2", "CodeStep", 3000.0, ["b"], ["c"]),
            ],
        )
        summary = log.summary()
        assert "test" in summary
        assert "s1" in summary
        assert "s2" in summary
        assert "5000" in summary or "5.0" in summary


class TestLoggedPipeline:
    def test_logged_pipeline_returns_result_and_log(self):
        agent = MagicMock()
        agent.execute.return_value = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100))),
        })

        pipeline = Pipeline(name="test", steps=[
            AgentStep(
                name="edit",
                agent=agent,
                input_map={"image": "input"},
                output_map={"image": "output"},
            ),
        ])

        logged = LoggedPipeline(pipeline)
        result, log = logged.run({"input": ImageMedia(image=PILImage.new("RGB", (200, 200)))})

        assert "output" in result
        assert isinstance(log, ExecutionLog)
        assert log.pipeline_name == "test"
        assert len(log.step_logs) == 1
        assert log.step_logs[0].step_name == "edit"
        assert log.step_logs[0].step_type == "AgentStep"
        assert log.total_duration_ms >= 0

    def test_logged_pipeline_times_code_step(self):
        def slow_fn(ctx):
            time.sleep(0.05)  # 50ms
            return {"result": TextMedia(text="done")}

        pipeline = Pipeline(name="timed", steps=[
            CodeStep(name="slow", fn=slow_fn),
        ])

        logged = LoggedPipeline(pipeline)
        result, log = logged.run({})

        assert log.step_logs[0].duration_ms >= 40  # at least 40ms
        assert log.total_duration_ms >= 40

    def test_logged_pipeline_nested(self):
        agent = MagicMock()
        agent.execute.return_value = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (50, 50))),
        })

        inner = Pipeline(name="inner", steps=[
            AgentStep(name="inner_edit", agent=agent,
                      input_map={"image": "in"}, output_map={"image": "out"}),
        ])

        outer = Pipeline(name="outer", steps=[
            PipelineStep(
                name="sub",
                pipeline=inner,
                input_map={"in": "raw"},
                output_map={"out": "final"},
            ),
        ])

        logged = LoggedPipeline(outer)
        result, log = logged.run({"raw": ImageMedia(image=PILImage.new("RGB", (100, 100)))})

        assert "final" in result
        assert len(log.step_logs) == 1
        assert log.step_logs[0].step_name == "sub"
        assert log.step_logs[0].step_type == "PipelineStep"
```

**Step 2: Run tests to verify they fail**

Run: `conda run -p .conda_env pytest tests/test_logging.py -v`
Expected: FAIL — `ImportError`

**Step 3: Implement execution logging**

```python
# src/casadei/logging.py
"""Execution logging for pipeline runs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from casadei.media import Media
from casadei.pipeline import Pipeline, Step, AgentStep, CodeStep, PipelineStep


@dataclass
class StepLog:
    """Timing and metadata for a single step execution."""

    step_name: str
    step_type: str
    duration_ms: float
    input_keys: list[str]
    output_keys: list[str]

    def __str__(self) -> str:
        return (
            f"  [{self.step_type}] {self.step_name}: "
            f"{self.duration_ms:.1f}ms "
            f"({', '.join(self.input_keys)} -> {', '.join(self.output_keys)})"
        )


@dataclass
class ExecutionLog:
    """Full log for a pipeline execution."""

    pipeline_name: str
    total_duration_ms: float
    step_logs: list[StepLog] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Pipeline '{self.pipeline_name}' — {self.total_duration_ms:.1f}ms total",
            f"  Steps: {len(self.step_logs)}",
        ]
        for sl in self.step_logs:
            lines.append(str(sl))
        return "\n".join(lines)


class LoggedPipeline:
    """Wraps a Pipeline to add execution logging.

    Usage:
        logged = LoggedPipeline(pipeline)
        result, log = logged.run(inputs)
        print(log.summary())
    """

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    def run(self, inputs: dict[str, Media]) -> tuple[dict[str, Media], ExecutionLog]:
        context = dict(inputs)
        step_logs = []
        pipeline_start = time.perf_counter()

        for step in self.pipeline.steps:
            step_start = time.perf_counter()
            input_keys = list(context.keys())

            outputs = step.execute(context)
            context.update(outputs)

            step_end = time.perf_counter()
            duration_ms = (step_end - step_start) * 1000

            step_logs.append(StepLog(
                step_name=step.name,
                step_type=type(step).__name__,
                duration_ms=duration_ms,
                input_keys=input_keys,
                output_keys=list(outputs.keys()),
            ))

        pipeline_end = time.perf_counter()
        total_ms = (pipeline_end - pipeline_start) * 1000

        log = ExecutionLog(
            pipeline_name=self.pipeline.name,
            total_duration_ms=total_ms,
            step_logs=step_logs,
        )
        return context, log
```

**Step 4: Run tests to verify they pass**

Run: `conda run -p .conda_env pytest tests/test_logging.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/casadei/logging.py tests/test_logging.py
git commit -m "feat: add execution logging with per-step timing"
```

---

## Task 10: Pipeline Visualization

**Files:**
- Create: `src/casadei/visualization.py`
- Create: `tests/test_visualization.py`

Renders pipeline structure as Mermaid diagrams (text format, can be rendered in markdown, GitHub, etc.). Shows step types, connections, and nested pipelines.

**Step 1: Write failing tests**

```python
# tests/test_visualization.py
import pytest
from unittest.mock import MagicMock
from PIL import Image as PILImage

from casadei.media import ImageMedia, TextMedia, Media, MediaBundle
from casadei.pipeline import AgentStep, CodeStep, PipelineStep, Pipeline
from casadei.visualization import to_mermaid


class TestMermaidVisualization:
    def test_single_agent_step(self):
        agent = MagicMock()
        pipeline = Pipeline(name="simple", steps=[
            AgentStep(name="edit", agent=agent,
                      input_map={"image": "raw"},
                      output_map={"image": "edited"}),
        ])
        mermaid = to_mermaid(pipeline)
        assert "graph" in mermaid or "flowchart" in mermaid
        assert "edit" in mermaid
        assert "raw" in mermaid
        assert "edited" in mermaid

    def test_chained_steps(self):
        agent1 = MagicMock()
        agent2 = MagicMock()
        pipeline = Pipeline(name="chain", steps=[
            AgentStep(name="clean", agent=agent1,
                      input_map={"image": "raw"},
                      output_map={"image": "clean"}),
            AgentStep(name="style", agent=agent2,
                      input_map={"image": "clean"},
                      output_map={"image": "styled"}),
        ])
        mermaid = to_mermaid(pipeline)
        assert "clean" in mermaid
        assert "style" in mermaid
        # Should show data flow: raw -> clean_step -> clean -> style_step -> styled
        assert "raw" in mermaid
        assert "styled" in mermaid

    def test_mixed_step_types(self):
        agent = MagicMock()

        def my_fn(ctx):
            return {}

        inner = Pipeline(name="inner", steps=[
            AgentStep(name="inner_edit", agent=agent,
                      input_map={}, output_map={}),
        ])

        pipeline = Pipeline(name="mixed", steps=[
            CodeStep(name="preprocess", fn=my_fn),
            AgentStep(name="edit", agent=agent,
                      input_map={"image": "preprocessed"},
                      output_map={"image": "edited"}),
            PipelineStep(name="postprocess", pipeline=inner,
                         input_map={"in": "edited"},
                         output_map={"out": "final"}),
        ])
        mermaid = to_mermaid(pipeline)
        assert "preprocess" in mermaid
        assert "edit" in mermaid
        assert "postprocess" in mermaid

    def test_nested_pipeline_shows_subgraph(self):
        agent = MagicMock()
        inner = Pipeline(name="inner_pipeline", steps=[
            AgentStep(name="sub_step", agent=agent,
                      input_map={"image": "x"},
                      output_map={"image": "y"}),
        ])
        outer = Pipeline(name="outer", steps=[
            PipelineStep(name="nested", pipeline=inner,
                         input_map={"x": "input"},
                         output_map={"y": "output"}),
        ])
        mermaid = to_mermaid(pipeline=outer, expand_nested=True)
        assert "subgraph" in mermaid
        assert "inner_pipeline" in mermaid
        assert "sub_step" in mermaid

    def test_output_is_valid_mermaid_syntax(self):
        agent = MagicMock()
        pipeline = Pipeline(name="test", steps=[
            AgentStep(name="step1", agent=agent,
                      input_map={"a": "x"}, output_map={"b": "y"}),
        ])
        mermaid = to_mermaid(pipeline)
        # Basic syntax checks
        lines = mermaid.strip().split("\n")
        assert lines[0].startswith("flowchart") or lines[0].startswith("graph")
```

**Step 2: Run tests to verify they fail**

Run: `conda run -p .conda_env pytest tests/test_visualization.py -v`
Expected: FAIL — `ImportError`

**Step 3: Implement visualization**

```python
# src/casadei/visualization.py
"""Pipeline visualization — render pipeline structure as Mermaid diagrams."""

from __future__ import annotations

from casadei.pipeline import Pipeline, Step, AgentStep, CodeStep, PipelineStep


def _step_shape(step: Step) -> tuple[str, str]:
    """Return mermaid node shape brackets based on step type."""
    if isinstance(step, AgentStep):
        return "[", "]"  # rectangle for agents
    elif isinstance(step, CodeStep):
        return "{{", "}}"  # hexagon for code
    elif isinstance(step, PipelineStep):
        return "[[", "]]"  # double bracket for sub-pipelines
    return "[", "]"


def _step_label(step: Step) -> str:
    """Return a descriptive label for the step."""
    if isinstance(step, AgentStep):
        return f"{step.name}\\n(Agent)"
    elif isinstance(step, CodeStep):
        return f"{step.name}\\n(Code)"
    elif isinstance(step, PipelineStep):
        return f"{step.name}\\n(Pipeline: {step.pipeline.name})"
    return step.name


def _sanitize_id(name: str) -> str:
    """Make a string safe for use as a mermaid node ID."""
    return name.replace(" ", "_").replace("-", "_")


def to_mermaid(
    pipeline: Pipeline,
    expand_nested: bool = False,
    direction: str = "TD",
) -> str:
    """Render a pipeline as a Mermaid flowchart diagram.

    Args:
        pipeline: The pipeline to visualize.
        expand_nested: If True, expand PipelineSteps into subgraphs.
        direction: Flow direction — "TD" (top-down) or "LR" (left-right).

    Returns:
        Mermaid diagram as a string.
    """
    lines = [f"flowchart {direction}"]
    _render_pipeline(pipeline, lines, expand_nested=expand_nested, indent="    ")
    return "\n".join(lines)


def _render_pipeline(
    pipeline: Pipeline,
    lines: list[str],
    expand_nested: bool,
    indent: str,
    prefix: str = "",
) -> None:
    """Recursively render pipeline steps into mermaid lines."""
    steps = pipeline.steps
    if not steps:
        return

    prev_output_nodes: list[str] = []

    for i, step in enumerate(steps):
        step_id = _sanitize_id(f"{prefix}{step.name}")
        left, right = _step_shape(step)
        label = _step_label(step)

        # If it's a PipelineStep and we want to expand it
        if isinstance(step, PipelineStep) and expand_nested:
            lines.append(f"{indent}subgraph {step_id}_sub [{step.pipeline.name}]")
            _render_pipeline(
                step.pipeline, lines, expand_nested=True,
                indent=indent + "    ", prefix=f"{step_id}_"
            )
            lines.append(f"{indent}end")

            # Connect previous step to subgraph
            if prev_output_nodes:
                inner_first = step.pipeline.steps[0] if step.pipeline.steps else None
                if inner_first:
                    inner_id = _sanitize_id(f"{step_id}_{inner_first.name}")
                    for pn in prev_output_nodes:
                        lines.append(f"{indent}{pn} --> {inner_id}")

            # Track inner pipeline's last step as output
            if step.pipeline.steps:
                last_inner = step.pipeline.steps[-1]
                prev_output_nodes = [_sanitize_id(f"{step_id}_{last_inner.name}")]
            else:
                prev_output_nodes = []
        else:
            # Regular step node
            lines.append(f"{indent}{step_id}{left}\"{label}\"{right}")

            # Add input data nodes for the first step
            if i == 0 and isinstance(step, (AgentStep, PipelineStep)):
                input_map = step.input_map if hasattr(step, 'input_map') else {}
                for agent_key, ctx_key in input_map.items():
                    data_id = _sanitize_id(f"{prefix}data_{ctx_key}")
                    lines.append(f"{indent}{data_id}(({ctx_key}))")
                    lines.append(f"{indent}{data_id} --> {step_id}")

            # Connect from previous step
            if prev_output_nodes:
                for pn in prev_output_nodes:
                    lines.append(f"{indent}{pn} --> {step_id}")

            # Add output data nodes
            if isinstance(step, (AgentStep, PipelineStep)):
                output_map = step.output_map if hasattr(step, 'output_map') else {}
                output_nodes = []
                for agent_key, ctx_key in output_map.items():
                    data_id = _sanitize_id(f"{prefix}data_{ctx_key}")
                    lines.append(f"{indent}{data_id}(({ctx_key}))")
                    lines.append(f"{indent}{step_id} --> {data_id}")
                    output_nodes.append(data_id)
                prev_output_nodes = output_nodes if output_nodes else [step_id]
            else:
                prev_output_nodes = [step_id]
```

**Step 4: Run tests to verify they pass**

Run: `conda run -p .conda_env pytest tests/test_visualization.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/casadei/visualization.py tests/test_visualization.py
git commit -m "feat: add pipeline visualization as Mermaid diagrams"
```

---

## Task 11: Public API and Package Exports

**Files:**
- Modify: `src/casadei/__init__.py`
- Create: `tests/test_public_api.py`

**Step 1: Write failing tests**

```python
# tests/test_public_api.py
"""Verify the public API is clean and accessible."""

import pytest


class TestPublicAPI:
    def test_import_media(self):
        from casadei import ImageMedia, TextMedia, VideoMedia, MediaBundle
        assert ImageMedia is not None

    def test_import_models(self):
        from casadei import AIModel, ImageEditModel, ModelCapability
        assert AIModel is not None

    def test_import_constraints(self):
        from casadei import ImageConstraint, TextConstraint, VideoConstraint
        assert ImageConstraint is not None

    def test_import_providers(self):
        from casadei import QwenImageEdit
        assert QwenImageEdit is not None

    def test_import_agent(self):
        from casadei import Agent, AgentConfig, load_agent, save_agent
        assert Agent is not None

    def test_import_pipeline(self):
        from casadei import Pipeline, AgentStep, CodeStep, PipelineStep
        assert Pipeline is not None
        assert CodeStep is not None
        assert PipelineStep is not None

    def test_import_logging(self):
        from casadei import LoggedPipeline, ExecutionLog, StepLog
        assert LoggedPipeline is not None

    def test_import_visualization(self):
        from casadei import to_mermaid
        assert to_mermaid is not None

    def test_import_registry(self):
        from casadei import default_registry
        assert default_registry is not None
```

**Step 2: Run tests to verify they fail**

Run: `conda run -p .conda_env pytest tests/test_public_api.py -v`
Expected: FAIL — `ImportError`

**Step 3: Update `src/casadei/__init__.py`**

```python
"""Casadei — Flexible AI pipeline framework."""

from casadei.media import ImageMedia, TextMedia, VideoMedia, MediaBundle
from casadei.models.base import (
    AIModel,
    ModelCapability,
    MediaConstraint,
    ImageConstraint,
    TextConstraint,
    VideoConstraint,
)
from casadei.models.image_edit import ImageEditModel
from casadei.models.registry import ModelRegistry, default_registry
from casadei.providers.qwen_image_edit import QwenImageEdit
from casadei.agent import Agent, AgentConfig, load_agent, save_agent
from casadei.pipeline import AgentStep, CodeStep, PipelineStep, Pipeline
from casadei.logging import ExecutionLog, StepLog, LoggedPipeline
from casadei.visualization import to_mermaid

__all__ = [
    # Media
    "ImageMedia",
    "TextMedia",
    "VideoMedia",
    "MediaBundle",
    # Models
    "AIModel",
    "ModelCapability",
    "MediaConstraint",
    "ImageConstraint",
    "TextConstraint",
    "VideoConstraint",
    "ImageEditModel",
    # Registry
    "ModelRegistry",
    "default_registry",
    # Providers
    "QwenImageEdit",
    # Agent
    "Agent",
    "AgentConfig",
    "load_agent",
    "save_agent",
    # Pipeline
    "AgentStep",
    "CodeStep",
    "PipelineStep",
    "Pipeline",
    # Logging
    "ExecutionLog",
    "StepLog",
    "LoggedPipeline",
    # Visualization
    "to_mermaid",
]
```

**Step 4: Run tests to verify they pass**

Run: `conda run -p .conda_env pytest tests/test_public_api.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add src/casadei/__init__.py tests/test_public_api.py
git commit -m "feat: expose full public API including logging and visualization"
```

---

## Task 12: Integration Test — Full Workflow

**Files:**
- Create: `tests/test_integration.py`

Demonstrates the full workflow: agent configs with `$variable` templates, a pipeline mixing all three step types, execution logging, and visualization.

**Step 1: Write the integration test**

```python
# tests/test_integration.py
"""Integration test: full workflow from agent configs to pipeline execution."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image as PILImage

from casadei import (
    Agent,
    AgentConfig,
    Pipeline,
    AgentStep,
    CodeStep,
    PipelineStep,
    ImageMedia,
    TextMedia,
    MediaBundle,
    LoggedPipeline,
    save_agent,
    load_agent,
    to_mermaid,
)
from casadei.media import Media


class TestFullWorkflow:
    """Simulates: create agent configs -> save to YAML -> load -> build pipeline -> run with logging."""

    def test_agent_config_roundtrip_and_pipeline(self, tmp_path):
        # 1. Create and save agent configs with $variable templates
        cleaner_config = AgentConfig(
            name="image_cleaner",
            model="qwen_image_edit",
            description="Cleans and enhances images",
            prompt_template="Clean up this image, focusing on $focus_area",
            negative_prompt="blurry, noisy",
            params={"num_inference_steps": 30},
        )
        styler_config = AgentConfig(
            name="style_transfer",
            model="qwen_image_edit",
            description="Applies artistic style",
            prompt_template="Apply $style artistic style to this image",
            params={"num_inference_steps": 50},
        )

        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        save_agent(cleaner_config, agents_dir / "image_cleaner.yaml")
        save_agent(styler_config, agents_dir / "style_transfer.yaml")

        # 2. Load agent configs from disk
        loaded_cleaner = load_agent(agents_dir / "image_cleaner.yaml")
        loaded_styler = load_agent(agents_dir / "style_transfer.yaml")
        assert loaded_cleaner.name == "image_cleaner"
        assert "$style" in loaded_styler.prompt_template

        # 3. Create agents from loaded configs (with mocked models)
        with patch("casadei.agent.default_registry") as mock_registry:
            def make_mock_model():
                model = MagicMock()
                model.run.side_effect = lambda bundle: MediaBundle(items={
                    "image": ImageMedia(
                        image=PILImage.new("RGB", (512, 512), color="green")
                    ),
                })
                return model

            mock_cls = MagicMock(side_effect=lambda: make_mock_model())
            mock_registry.get.return_value = mock_cls

            cleaner_agent = Agent(config=loaded_cleaner)
            styler_agent = Agent(config=loaded_styler)
            cleaner_agent.load()
            styler_agent.load()

            # 4. Build pipeline with mixed step types
            def resize_fn(ctx: dict[str, Media]) -> dict[str, Media]:
                img = ctx["raw_image"]
                assert isinstance(img, ImageMedia)
                resized = img.image.resize((512, 512))
                return {"resized_image": ImageMedia(image=resized)}

            pipeline = Pipeline(
                name="clean_and_style",
                steps=[
                    CodeStep(name="resize", fn=resize_fn),
                    AgentStep(
                        name="clean",
                        agent=cleaner_agent,
                        input_map={"image": "resized_image"},
                        output_map={"image": "clean_image"},
                        template_kwargs={"focus_area": "background noise"},
                    ),
                    AgentStep(
                        name="stylize",
                        agent=styler_agent,
                        input_map={"image": "clean_image"},
                        output_map={"image": "styled_image"},
                        template_kwargs={"style": "impressionist"},
                    ),
                ],
            )

            # 5. Run pipeline with logging
            logged = LoggedPipeline(pipeline)
            input_image = ImageMedia(
                image=PILImage.new("RGB", (1024, 1024), color="red")
            )
            result, log = logged.run({"raw_image": input_image})

            # 6. Verify results
            assert "styled_image" in result
            assert isinstance(result["styled_image"], ImageMedia)

            # 7. Verify logging
            assert log.pipeline_name == "clean_and_style"
            assert len(log.step_logs) == 3
            assert log.step_logs[0].step_name == "resize"
            assert log.step_logs[0].step_type == "CodeStep"
            assert log.step_logs[1].step_name == "clean"
            assert log.step_logs[1].step_type == "AgentStep"
            assert log.total_duration_ms >= 0
            summary = log.summary()
            assert "clean_and_style" in summary

            # 8. Verify visualization
            mermaid = to_mermaid(pipeline)
            assert "resize" in mermaid
            assert "clean" in mermaid
            assert "stylize" in mermaid

    def test_pipeline_composition_with_nested(self, tmp_path):
        """Test composing pipelines with PipelineStep."""
        with patch("casadei.agent.default_registry") as mock_registry:
            mock_model = MagicMock()
            mock_model.run.return_value = MediaBundle(items={
                "image": ImageMedia(image=PILImage.new("RGB", (256, 256))),
            })
            mock_cls = MagicMock(return_value=mock_model)
            mock_registry.get.return_value = mock_cls

            # Inner pipeline: preprocessing
            agent_a = Agent(config=AgentConfig(name="preprocess", model="qwen_image_edit"))
            agent_a.load()
            preprocessing = Pipeline(
                name="preprocessing",
                steps=[AgentStep(
                    name="pre",
                    agent=agent_a,
                    input_map={"image": "raw"},
                    output_map={"image": "preprocessed"},
                )],
            )

            # Outer pipeline uses inner as a step
            agent_b = Agent(config=AgentConfig(name="edit", model="qwen_image_edit"))
            agent_b.load()
            full_pipeline = Pipeline(
                name="full_workflow",
                steps=[
                    PipelineStep(
                        name="preprocess_step",
                        pipeline=preprocessing,
                        input_map={"raw": "input_image"},
                        output_map={"preprocessed": "clean_image"},
                    ),
                    AgentStep(
                        name="final_edit",
                        agent=agent_b,
                        input_map={"image": "clean_image"},
                        output_map={"image": "result"},
                    ),
                ],
            )

            # Run with logging
            logged = LoggedPipeline(full_pipeline)
            result, log = logged.run({
                "input_image": ImageMedia(image=PILImage.new("RGB", (512, 512))),
            })
            assert "result" in result
            assert len(log.step_logs) == 2

            # Visualize with expanded nested pipeline
            mermaid = to_mermaid(full_pipeline, expand_nested=True)
            assert "subgraph" in mermaid
            assert "preprocessing" in mermaid
```

**Step 2: Run tests to verify they pass**

Run: `conda run -p .conda_env pytest tests/test_integration.py -v`
Expected: All 2 tests PASS

**Step 3: Run full test suite**

Run: `conda run -p .conda_env pytest -v --tb=short`
Expected: All tests PASS (approximately 70+ tests)

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for full workflow with logging and visualization"
```

---

## Task 13: Example Agent Configs

**Files:**
- Create: `agents/qwen_background_remover.yaml`
- Create: `agents/qwen_style_transfer.yaml`
- Create: `agents/qwen_object_replacer.yaml`

**Step 1: Create example agent configs**

```yaml
# agents/qwen_background_remover.yaml
name: background_remover
model: qwen_image_edit
description: Removes backgrounds from images, isolating the main subject
prompt_template: "Remove the background completely, leaving only the main subject on a $background_color background"
negative_prompt: "blurry, artifacts, incomplete removal"
params:
  num_inference_steps: 40
  true_cfg_scale: 4.0
```

```yaml
# agents/qwen_style_transfer.yaml
name: style_transfer
model: qwen_image_edit
description: Applies artistic style transformations to images
prompt_template: "Transform this image into $style style while preserving the composition and main subjects"
negative_prompt: "distorted, low quality, blurry"
params:
  num_inference_steps: 50
  true_cfg_scale: 3.5
```

```yaml
# agents/qwen_object_replacer.yaml
name: object_replacer
model: qwen_image_edit
description: Replaces specific objects in an image with something else
prompt_template: "Replace the $source_object with a $target_object, maintaining realistic lighting and perspective"
negative_prompt: "unrealistic, floating, disconnected shadows"
params:
  num_inference_steps: 40
  true_cfg_scale: 4.0
```

**Step 2: Commit**

```bash
git add agents/
git commit -m "feat: add example agent configs with $variable templates"
```

---

## Summary

| Task | What | Tests |
|------|------|-------|
| 1 | Project scaffolding (git, conda, pyproject.toml, dirs) | Setup verified |
| 2 | Media types with TextMedia `$variable` templates | 21 tests |
| 3 | AIModel base, ModelCapability, constraint system | 9 tests |
| 4 | ImageEditModel base class with multi-text support | 5 tests |
| 5 | QwenImageEdit provider (mocked) | 8 tests |
| 6 | Model registry | 4 tests |
| 7 | Agent system with template filling + validation | 10 tests |
| 8 | Pipeline with AgentStep, CodeStep, PipelineStep | 13 tests |
| 9 | Execution logging (per-step + per-pipeline timing) | 6 tests |
| 10 | Pipeline visualization (Mermaid diagrams) | 5 tests |
| 11 | Public API exports | 9 tests |
| 12 | Integration test (full workflow) | 2 tests |
| 13 | Example agent configs | N/A |

**Total: ~92 tests, 13 tasks, TDD throughout**

### Architecture at a Glance

```
Media Types             Models                    Agents              Pipeline
──────────────         ──────────────            ──────────          ────────────────
ImageMedia       ──→   AIModel (ABC)       ──→   AgentConfig   ──→  AgentStep
TextMedia ($var)       ├─ ImageEditModel         Agent               CodeStep (fn)
VideoMedia             │  └─ QwenImageEdit       save/load YAML      PipelineStep
MediaBundle            ├─ ImageGenModel (future)                     Pipeline
                       ├─ LLMModel (future)                          compose()
                       └─ ImageVideoModel (future)
                                                                Logging
                                                                ──────────
                                                                LoggedPipeline
                                                                ExecutionLog
                                                                StepLog

                                                                Visualization
                                                                ──────────────
                                                                to_mermaid()
```

### Key Design Decisions

- **TextMedia `$variable` syntax**: Uses Python `string.Template` — `$item` or `${item}`. `fill()` returns a new instance (immutable). `safe_substitute` leaves unfilled vars as-is for partial filling. Agent validates all vars are filled before running.
- **Three step types**: AgentStep (runs models), CodeStep (arbitrary Python functions for custom logic between model calls), PipelineStep (nested pipelines for composition).
- **Logging is opt-in**: `LoggedPipeline` wraps a `Pipeline` — no overhead when you don't need it.
- **Visualization is text-based**: Mermaid output can be rendered in GitHub, VS Code, or any markdown viewer. `expand_nested=True` shows sub-pipeline internals.

### Future Extension Points (not implemented now, but the architecture supports them)

- **New model types**: Subclass `AIModel` → e.g., `ImageGenModel`, `LLMModel`, `ImageVideoModel`
- **New providers**: Subclass model types → e.g., `StableDiffusionXL(ImageGenModel)`
- **Workflow YAML**: Serialize entire pipelines to YAML files in `workflows/`
- **Conditional steps**: Add `ConditionalStep` that branches based on context values
- **Parallel steps**: Steps with no data dependencies could run concurrently
- **Web UI**: Pipeline builder with drag-and-drop (reads the same YAML configs)
- **Log persistence**: Save `ExecutionLog` to disk or database for pipeline analytics
