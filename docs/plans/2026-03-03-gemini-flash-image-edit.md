# GeminiFlashImageEdit Provider Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `GeminiFlashImageEdit` provider wrapping Google's `gemini-3.1-flash-image-preview` ("Nano Banana 2") API for image editing, registered as `gemini_flash_image_edit` in the model registry.

**Architecture:** API-only provider (no local weights, no GPU) — subclasses `ImageEditModel`, passes PIL images + prompt directly to the `google-genai` SDK, extracts the returned PIL image via `part.as_image()`. Mirrors the Voyage embedding provider pattern for API-based clients.

**Tech Stack:** `google-genai>=1.0`, `GEMINI_API_KEY` env var, Python `unittest.mock` for tests.

---

### Task 1: Add google-genai dependency

**Files:**
- Modify: `pyproject.toml:10-27`

**Step 1: Add the dependency**

In `pyproject.toml`, add `"google-genai>=1.0",` to the `dependencies` list, after the `voyageai` line:

```toml
    "voyageai>=0.3.0",
    "google-genai>=1.0",
```

**Step 2: Install it**

Run:
```bash
pip install -e .
```

Expected: installs `google-genai` without errors, `casadei` reinstalled in editable mode.

**Step 3: Verify import works**

Run:
```bash
python -c "from google import genai; print('ok')"
```

Expected: prints `ok`.

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add google-genai dependency for Nano Banana 2 provider"
```

---

### Task 2: Create the provider with tests

**Files:**
- Create: `src/casadei/providers/gemini_flash_image_edit.py`
- Create: `tests/test_gemini_flash_image_edit.py`

**Step 1: Write the failing tests**

Create `tests/test_gemini_flash_image_edit.py`:

```python
"""Tests for GeminiFlashImageEdit provider."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage

from casadei.media import ImageMedia, TextMedia, MediaBundle
from casadei.models.image_edit import ImageEditModel
from casadei.models.base import ImageConstraint, TextConstraint


class TestGeminiFlashImageEditClass:
    def test_is_image_edit_model(self):
        from casadei.providers.gemini_flash_image_edit import GeminiFlashImageEdit
        assert issubclass(GeminiFlashImageEdit, ImageEditModel)

    def test_capability_requires_image_and_text(self):
        from casadei.providers.gemini_flash_image_edit import GeminiFlashImageEdit
        cap = GeminiFlashImageEdit.capability
        input_types = [type(c) for c in cap.inputs]
        assert ImageConstraint in input_types
        assert TextConstraint in input_types

    def test_capability_outputs_one_image(self):
        from casadei.providers.gemini_flash_image_edit import GeminiFlashImageEdit
        cap = GeminiFlashImageEdit.capability
        assert len(cap.outputs) == 1
        assert isinstance(cap.outputs[0], ImageConstraint)
        assert cap.outputs[0].max_count == 1

    def test_capability_accepts_up_to_10_images(self):
        from casadei.providers.gemini_flash_image_edit import GeminiFlashImageEdit
        cap = GeminiFlashImageEdit.capability
        img_constraint = next(c for c in cap.inputs if isinstance(c, ImageConstraint))
        assert img_constraint.max_count == 10


class TestGeminiFlashImageEditLoadUnload:
    def test_load_model_creates_client(self):
        from casadei.providers.gemini_flash_image_edit import GeminiFlashImageEdit

        with patch("casadei.providers.gemini_flash_image_edit.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client

            provider = GeminiFlashImageEdit()
            provider.load_model()

            mock_genai.Client.assert_called_once()
            assert provider._client is mock_client

    def test_unload_model_clears_client(self):
        from casadei.providers.gemini_flash_image_edit import GeminiFlashImageEdit

        with patch("casadei.providers.gemini_flash_image_edit.genai"):
            provider = GeminiFlashImageEdit()
            provider._client = MagicMock()
            provider.unload_model()
            assert provider._client is None

    def test_edit_raises_if_not_loaded(self):
        from casadei.providers.gemini_flash_image_edit import GeminiFlashImageEdit

        provider = GeminiFlashImageEdit()
        # _client is None by default
        img = PILImage.new("RGB", (64, 64), color="red")
        with pytest.raises(RuntimeError, match="not loaded"):
            provider._edit(images=[img], prompt="test", negative_prompt="")


class TestGeminiFlashImageEditEdit:
    def _make_fake_response(self, pil_image: PILImage.Image):
        """Build a mock genai response that returns pil_image from part.as_image()."""
        fake_part = MagicMock()
        fake_part.text = None
        fake_part.inline_data = MagicMock()  # non-None so it's treated as image part
        fake_part.as_image.return_value = pil_image

        fake_response = MagicMock()
        fake_response.parts = [fake_part]
        return fake_response

    def test_edit_returns_pil_image(self):
        from casadei.providers.gemini_flash_image_edit import GeminiFlashImageEdit

        input_img = PILImage.new("RGB", (64, 64), color="red")
        output_img = PILImage.new("RGB", (64, 64), color="blue")
        fake_response = self._make_fake_response(output_img)

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = fake_response

        with patch("casadei.providers.gemini_flash_image_edit.genai"):
            provider = GeminiFlashImageEdit()
            provider._client = mock_client

            result = provider._edit(
                images=[input_img], prompt="make it blue", negative_prompt=""
            )

        assert isinstance(result, PILImage.Image)
        mock_client.models.generate_content.assert_called_once()

    def test_edit_passes_prompt_and_images_to_api(self):
        from casadei.providers.gemini_flash_image_edit import GeminiFlashImageEdit

        input_img = PILImage.new("RGB", (64, 64), color="red")
        output_img = PILImage.new("RGB", (64, 64), color="blue")
        fake_response = self._make_fake_response(output_img)

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = fake_response

        with patch("casadei.providers.gemini_flash_image_edit.genai"):
            provider = GeminiFlashImageEdit()
            provider._client = mock_client

            provider._edit(
                images=[input_img], prompt="make it blue", negative_prompt=""
            )

        call_kwargs = mock_client.models.generate_content.call_args
        contents = call_kwargs.kwargs["contents"]
        assert "make it blue" in contents
        assert input_img in contents

    def test_edit_raises_if_no_image_in_response(self):
        from casadei.providers.gemini_flash_image_edit import GeminiFlashImageEdit

        # Response with only text part, no inline_data
        text_only_part = MagicMock()
        text_only_part.text = "I cannot do that"
        text_only_part.inline_data = None

        fake_response = MagicMock()
        fake_response.parts = [text_only_part]

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = fake_response

        input_img = PILImage.new("RGB", (64, 64), color="red")

        with patch("casadei.providers.gemini_flash_image_edit.genai"):
            provider = GeminiFlashImageEdit()
            provider._client = mock_client

            with pytest.raises(RuntimeError, match="[Nn]o image"):
                provider._edit(
                    images=[input_img], prompt="test", negative_prompt=""
                )

    def test_edit_resizes_output_to_input_dimensions(self):
        from casadei.providers.gemini_flash_image_edit import GeminiFlashImageEdit

        input_img = PILImage.new("RGB", (128, 64), color="red")
        # API returns a different size
        wrong_size_output = PILImage.new("RGB", (512, 512), color="blue")
        fake_response = self._make_fake_response(wrong_size_output)

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = fake_response

        with patch("casadei.providers.gemini_flash_image_edit.genai"):
            provider = GeminiFlashImageEdit()
            provider._client = mock_client

            result = provider._edit(
                images=[input_img], prompt="test", negative_prompt=""
            )

        assert result.size == (128, 64)
```

**Step 2: Run the tests to verify they all fail**

Run:
```bash
pytest tests/test_gemini_flash_image_edit.py -v
```

Expected: `ImportError` — `cannot import name 'GeminiFlashImageEdit'`.

**Step 3: Implement the provider**

Create `src/casadei/providers/gemini_flash_image_edit.py`:

```python
"""Gemini 3.1 Flash Image Preview provider ("Nano Banana 2").

Google's API-based image generation and editing model. No local weights —
requires GEMINI_API_KEY environment variable.

Model code: gemini-3.1-flash-image-preview
Input: up to 10 images + text prompt
Output: 1 edited/generated image
"""

from __future__ import annotations

import logging

from PIL import Image as PILImage

from casadei.models.base import ModelCapability, ImageConstraint, TextConstraint
from casadei.models.image_edit import ImageEditModel

try:
    from google import genai
except ImportError:
    genai = None

logger = logging.getLogger(__name__)

MODEL_ID = "gemini-3.1-flash-image-preview"


class GeminiFlashImageEdit(ImageEditModel):
    """Google Gemini 3.1 Flash Image Preview — API-based image editor.

    Accepts up to 10 reference images and a text prompt. Produces 1 edited image.
    Reads GEMINI_API_KEY from the environment via the google-genai SDK.
    No local model weights or GPU required.
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

        target_size = images[0].size

        # SDK accepts PIL Images directly alongside the text prompt
        contents = [prompt] + images

        response = self._client.models.generate_content(
            model=self.MODEL_ID,
            contents=contents,
        )

        for part in response.parts:
            if part.inline_data is not None:
                result = part.as_image()
                if result.size != target_size:
                    result = result.resize(target_size, PILImage.LANCZOS)
                return result

        raise RuntimeError(
            "No image returned by Gemini API. "
            "The model may have refused the request or returned text only."
        )
```

**Step 4: Run the tests to verify they all pass**

Run:
```bash
pytest tests/test_gemini_flash_image_edit.py -v
```

Expected: all tests PASS.

**Step 5: Commit**

```bash
git add src/casadei/providers/gemini_flash_image_edit.py tests/test_gemini_flash_image_edit.py
git commit -m "feat: add GeminiFlashImageEdit provider (Nano Banana 2)"
```

---

### Task 3: Register in the default registry

**Files:**
- Modify: `src/casadei/models/registry.py:57` (before `return registry`)
- Modify: `tests/test_registry.py`

**Step 1: Write the failing test**

Add to `tests/test_registry.py`:

```python
def test_builtin_registry_has_gemini_flash_image_edit():
    from casadei.models.registry import default_registry
    from casadei.providers.gemini_flash_image_edit import GeminiFlashImageEdit
    cls = default_registry.get("gemini_flash_image_edit")
    assert cls is GeminiFlashImageEdit
```

**Step 2: Run the test to verify it fails**

Run:
```bash
pytest tests/test_registry.py::TestModelRegistry::test_builtin_registry_has_gemini_flash_image_edit -v
```

Expected: FAIL with `KeyError: "Unknown model: 'gemini_flash_image_edit'"`.

**Step 3: Register the provider**

In `src/casadei/models/registry.py`, add before `return registry` (after the last existing `registry.register(...)` call):

```python
    from casadei.providers.gemini_flash_image_edit import GeminiFlashImageEdit
    registry.register("gemini_flash_image_edit", GeminiFlashImageEdit)
```

**Step 4: Run the test to verify it passes**

Run:
```bash
pytest tests/test_registry.py -v
```

Expected: all tests PASS.

**Step 5: Commit**

```bash
git add src/casadei/models/registry.py tests/test_registry.py
git commit -m "feat: register gemini_flash_image_edit in default model registry"
```

---

### Task 4: Export from providers __init__

**Files:**
- Modify: `src/casadei/providers/__init__.py`

**Step 1: Add the import and export**

In `src/casadei/providers/__init__.py`, add:

```python
from casadei.providers.gemini_flash_image_edit import GeminiFlashImageEdit
```

And add `"GeminiFlashImageEdit"` to `__all__`.

The file should look like:

```python
"""Model provider implementations."""

from casadei.providers.qwen_image_edit import QwenImageEdit
from casadei.providers.wan_i2v import WanImageToVideo
from casadei.providers.wan_i2v_fp8 import WanImageToVideoFP8
from casadei.providers.wan_video_edit import WanVideoEdit
from casadei.providers.wan_video_edit_fp8 import WanVideoEditFP8
from casadei.providers.voyage_embedding import VoyageEmbeddingProvider
from casadei.providers.gemini_flash_image_edit import GeminiFlashImageEdit

__all__ = [
    "QwenImageEdit",
    "WanImageToVideo", "WanImageToVideoFP8",
    "WanVideoEdit", "WanVideoEditFP8",
    "VoyageEmbeddingProvider",
    "GeminiFlashImageEdit",
]
```

**Step 2: Verify the import works**

Run:
```bash
python -c "from casadei.providers import GeminiFlashImageEdit; print(GeminiFlashImageEdit.MODEL_ID)"
```

Expected: prints `gemini-3.1-flash-image-preview`.

**Step 3: Run the full test suite**

Run:
```bash
pytest tests/test_gemini_flash_image_edit.py tests/test_registry.py -v
```

Expected: all tests PASS.

**Step 4: Commit**

```bash
git add src/casadei/providers/__init__.py
git commit -m "feat: export GeminiFlashImageEdit from providers package"
```
