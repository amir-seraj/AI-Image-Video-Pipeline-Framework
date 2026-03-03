# 360° Spin Frame Generation — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the placeholder `_run_generate_360()` with real frame generation using Zero123++ (local GPU) and Gemini Flash (API) providers.

**Architecture:** Two new providers implement a shared `ImageToMultiViewModel` base class. The endpoint accepts a `provider` query param to select which one. Each provider takes a single input image and produces 6 multi-view images. Follows existing model registry + provider patterns exactly.

**Tech Stack:** diffusers (Zero123++ custom pipeline), google-genai (Gemini Flash), PIL, torch, pydantic

---

### Task 1: ImageToMultiViewModel base class

**Files:**
- Create: `src/casadei/models/image_to_multiview.py`

**Step 1: Create the base class**

```python
"""Base class for single-image to multi-view generation models."""

from __future__ import annotations

from abc import abstractmethod

from PIL import Image as PILImage

from casadei.media import ImageMedia, MediaBundle
from casadei.models.base import AIModel


class ImageToMultiViewModel(AIModel):
    """Base class for models that generate multiple views from a single image.

    Subclasses must implement ``_generate_views()`` with model-specific logic
    and define ``capability`` with their specific constraints.
    """

    @abstractmethod
    def _generate_views(
        self,
        image: PILImage.Image,
        num_views: int = 6,
        **kwargs,
    ) -> list[PILImage.Image]:
        """Run model-specific multi-view generation.

        Returns a list of PIL images, one per viewing angle.
        """

    def run(self, inputs: MediaBundle, **kwargs) -> MediaBundle:
        errors = self.capability.validate_inputs(inputs)
        if errors:
            raise ValueError("; ".join(errors))

        image_items = [
            v for v in inputs.items.values() if isinstance(v, ImageMedia)
        ]
        if not image_items:
            raise ValueError("No ImageMedia input found")
        source_image = image_items[0].image

        num_views = kwargs.pop("num_views", 6)
        views = self._generate_views(
            image=source_image,
            num_views=num_views,
            **kwargs,
        )
        return MediaBundle(items={
            f"view_{i}": ImageMedia(image=v) for i, v in enumerate(views)
        })
```

**Step 2: Export from models `__init__.py`**

In `src/casadei/models/__init__.py`, add:

```python
from casadei.models.image_to_multiview import ImageToMultiViewModel
```

And add `"ImageToMultiViewModel"` to `__all__`.

**Step 3: Export from package `__init__.py`**

In `src/casadei/__init__.py`, add:

```python
from casadei.models.image_to_multiview import ImageToMultiViewModel
```

And add `"ImageToMultiViewModel"` to `__all__`.

**Step 4: Commit**

```bash
git add src/casadei/models/image_to_multiview.py src/casadei/models/__init__.py src/casadei/__init__.py
git commit -m "feat: add ImageToMultiViewModel base class for multi-view generation"
```

---

### Task 2: Zero123++ provider

**Files:**
- Create: `src/casadei/providers/zero123pp.py`

**Step 1: Create the provider**

Zero123++ outputs a single 960x640 grid image (3 columns x 2 rows, each tile 320x320).
The 6 views are at azimuths 30°, 90°, 150°, 210°, 270°, 330° relative to input.

```python
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

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

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
```

**Step 2: Commit**

```bash
git add src/casadei/providers/zero123pp.py
git commit -m "feat: add Zero123++ multi-view provider"
```

---

### Task 3: Gemini Flash multi-view provider

**Files:**
- Create: `src/casadei/providers/gemini_flash_multiview.py`

**Step 1: Create the provider**

Follows the same pattern as `GeminiFlashImageEdit` but sends 6 requests with angle prompts.

```python
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

            logger.debug("Generating view %d/%d (angle %d°)", i + 1, num_views, VIEW_ANGLES[i])

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
                    f"No image returned for view {i + 1} (angle {VIEW_ANGLES[i]}°). "
                    "The model may have refused the request."
                )

            views.append(view_image)

        return views
```

**Step 2: Commit**

```bash
git add src/casadei/providers/gemini_flash_multiview.py
git commit -m "feat: add Gemini Flash multi-view provider"
```

---

### Task 4: Register both providers

**Files:**
- Modify: `src/casadei/providers/__init__.py`
- Modify: `src/casadei/models/registry.py:61` (before `return registry`)

**Step 1: Update providers `__init__.py`**

Add to imports:
```python
from casadei.providers.zero123pp import Zero123PlusPlus
from casadei.providers.gemini_flash_multiview import GeminiFlashMultiView
```

Add to `__all__`:
```python
"Zero123PlusPlus",
"GeminiFlashMultiView",
```

**Step 2: Update registry**

In `src/casadei/models/registry.py`, inside `_build_default_registry()`, before the `return registry` line, add:

```python
from casadei.providers.zero123pp import Zero123PlusPlus
registry.register("zero123pp", Zero123PlusPlus)
from casadei.providers.gemini_flash_multiview import GeminiFlashMultiView
registry.register("gemini_multiview", GeminiFlashMultiView)
```

**Step 3: Commit**

```bash
git add src/casadei/providers/__init__.py src/casadei/models/registry.py
git commit -m "feat: register Zero123++ and Gemini multi-view providers"
```

---

### Task 5: Update the generate-360 endpoint

**Files:**
- Modify: `src/casadei/api/app.py:1494-1546` (the generate-360 endpoint + `_run_generate_360`)

**Step 1: Replace the placeholder endpoint and background function**

Replace lines 1494-1546 in `app.py` with:

```python
    @app.post(
        "/api/products/{product_id}/variations/{variation_id}/generate-360",
        status_code=202,
    )
    def generate_360(
        product_id: str,
        variation_id: str,
        provider: str = "zero123pp",
    ) -> dict:
        product = store.get_product(product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")

        variation = None
        for v in product.variations:
            if v.id == variation_id:
                variation = v
                break
        if not variation:
            raise HTTPException(
                status_code=404, detail="Variation not found"
            )

        if not variation.results:
            raise HTTPException(
                status_code=400,
                detail="Variation has no rendered images to generate 360 from",
            )

        job_id = job_manager.create(product_id, variation.id)

        thread = threading.Thread(
            target=_run_generate_360,
            args=(product, variation, job_id, provider),
            daemon=True,
        )
        thread.start()

        return {"job_id": job_id}

    def _run_generate_360(
        prod: Product,
        var: Variation,
        jid: str,
        provider_name: str,
    ) -> None:
        """Background thread: generates 360° spin frames for a variation."""
        try:
            from casadei import ImageMedia
            from casadei.models.registry import default_registry

            job_manager.update_progress(jid, 0.05, "Loading source image...")

            # Load the first result image as source
            var_results_dir = results_dir / prod.id / var.id
            source_path = var_results_dir / var.results[0].filename
            source = ImageMedia.load(source_path)

            job_manager.update_progress(jid, 0.1, f"Loading {provider_name} model...")

            model_cls = default_registry.get(provider_name)
            model = model_cls()
            model.load_model()

            try:
                job_manager.update_progress(jid, 0.2, "Generating views...")
                result = model.run(
                    MediaBundle(items={"image": source}),
                    num_views=6,
                )

                job_manager.update_progress(jid, 0.8, "Saving frames...")

                spin_dir = var_results_dir
                spin_dir.mkdir(parents=True, exist_ok=True)

                spin_files = []
                for i, (key, media) in enumerate(sorted(result.items.items())):
                    if isinstance(media, ImageMedia):
                        fname = f"spin_frame_{i}.png"
                        media.save(spin_dir / fname)
                        spin_files.append(ResultFile(filename=fname))

                var.spin_frames = spin_files
                var.status = JobStatus.completed
                store.save_product(prod)

                job_manager.update_progress(jid, 1.0, "Done!")
                job_manager.complete(jid)

            finally:
                model.unload_model()

        except Exception as e:
            import traceback
            traceback.print_exc()
            var.status = JobStatus.failed
            store.save_product(prod)
            job_manager.fail(jid, str(e))
```

Note: The `_run_generate_360` function needs `MediaBundle` imported. Add to the import at the top of the function body:
```python
from casadei import ImageMedia
from casadei.media import MediaBundle
```

**Step 2: Verify the import of `ResultFile`**

Check that `ResultFile` is already imported at the top of `app.py` (line 18-40). It should be — it's used in `_run_variation`.

**Step 3: Commit**

```bash
git add src/casadei/api/app.py
git commit -m "feat: implement real 360° generation endpoint with provider selection"
```

---

### Task 6: Write tests

**Files:**
- Create: `tests/test_multiview.py`

**Step 1: Write unit tests for the base class and grid splitting**

```python
"""Tests for multi-view generation components."""

from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest
from PIL import Image

from casadei.media import ImageMedia, MediaBundle
from casadei.models.base import ModelCapability, ImageConstraint
from casadei.models.image_to_multiview import ImageToMultiViewModel
from casadei.providers.zero123pp import _split_grid, GRID_COLS, GRID_ROWS, TILE_SIZE


class ConcreteMultiView(ImageToMultiViewModel):
    """Test-only concrete implementation."""

    capability = ModelCapability(
        inputs=[ImageConstraint(required=True, max_count=1)],
        outputs=[ImageConstraint(required=True, max_count=6)],
    )

    def load_model(self):
        pass

    def unload_model(self):
        pass

    def _generate_views(self, image, num_views=6, **kwargs):
        return [Image.new("RGB", (320, 320), "red") for _ in range(num_views)]


class TestImageToMultiViewModel:
    def test_run_returns_6_views(self):
        model = ConcreteMultiView()
        bundle = MediaBundle(items={"image": ImageMedia(image=Image.new("RGB", (320, 320)))})
        result = model.run(bundle, num_views=6)
        assert len(result.items) == 6
        assert all(isinstance(v, ImageMedia) for v in result.items.values())

    def test_run_validates_inputs(self):
        model = ConcreteMultiView()
        bundle = MediaBundle(items={})
        with pytest.raises(ValueError, match="Required ImageMedia input is missing"):
            model.run(bundle)

    def test_run_custom_num_views(self):
        model = ConcreteMultiView()
        bundle = MediaBundle(items={"image": ImageMedia(image=Image.new("RGB", (320, 320)))})
        result = model.run(bundle, num_views=3)
        assert len(result.items) == 3


class TestSplitGrid:
    def test_split_produces_6_tiles(self):
        grid = Image.new("RGB", (TILE_SIZE * GRID_COLS, TILE_SIZE * GRID_ROWS))
        tiles = _split_grid(grid)
        assert len(tiles) == 6

    def test_each_tile_correct_size(self):
        grid = Image.new("RGB", (TILE_SIZE * GRID_COLS, TILE_SIZE * GRID_ROWS))
        tiles = _split_grid(grid)
        for tile in tiles:
            assert tile.size == (TILE_SIZE, TILE_SIZE)

    def test_split_preserves_content(self):
        """Each tile should contain the correct region of the grid."""
        grid = Image.new("RGB", (TILE_SIZE * GRID_COLS, TILE_SIZE * GRID_ROWS), "white")
        # Paint top-left tile red
        for x in range(TILE_SIZE):
            for y in range(TILE_SIZE):
                grid.putpixel((x, y), (255, 0, 0))
        tiles = _split_grid(grid)
        # First tile should be red
        assert tiles[0].getpixel((0, 0)) == (255, 0, 0)
        # Second tile should be white
        assert tiles[1].getpixel((0, 0)) == (255, 255, 255)


class TestGenerate360Endpoint:
    """Test the API endpoint (uses mocked model)."""

    @pytest.fixture
    def client(self, tmp_path):
        from casadei.api.app import create_app
        from fastapi.testclient import TestClient
        app = create_app(data_dir=tmp_path)
        return TestClient(app)

    def test_404_product_not_found(self, client):
        resp = client.post("/api/products/bad/variations/bad/generate-360")
        assert resp.status_code == 404

    def test_404_variation_not_found(self, client):
        create = client.post("/api/products", json={"name": "Test"})
        pid = create.json()["id"]
        resp = client.post(f"/api/products/{pid}/variations/bad/generate-360")
        assert resp.status_code == 404
```

**Step 2: Run tests**

```bash
cd /home/innovina/Documents/casadei && python -m pytest tests/test_multiview.py -v
```

Expected: All tests pass.

**Step 3: Commit**

```bash
git add tests/test_multiview.py
git commit -m "test: add unit tests for multi-view generation"
```

---

### Task 7: Verify build

**Step 1: Run full test suite**

```bash
cd /home/innovina/Documents/casadei && python -m pytest tests/ -v
```

Expected: All tests pass, no import errors.

**Step 2: Verify model registration**

```bash
cd /home/innovina/Documents/casadei && python -c "from casadei.models.registry import default_registry; print(default_registry.list_models())"
```

Expected: Output includes `'zero123pp'` and `'gemini_multiview'`.

**Step 3: Commit final**

```bash
git add -A
git commit -m "chore: verify 360° spin generation implementation"
```
