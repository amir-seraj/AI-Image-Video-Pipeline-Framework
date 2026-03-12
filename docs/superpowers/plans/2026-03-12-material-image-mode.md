# Material-Image Mode Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add material-image mode to the sketch-to-shoe pipeline so users can upload material/color images mapped to shoe parts instead of describing materials with text.

**Architecture:** Auto-detected via `spec["materials"]`. Grid builder in shared utils assembles labeled tiles. New prompt template in sketch_to_shoe_gemini/ references materials by name. Pipeline detects mode and switches template + input. API gets upload/delete endpoints for material images with JSON metadata.

**Tech Stack:** Python 3.12+, PIL/Pillow, FastAPI, YAML, Gemini Flash Image Edit

---

## File Structure

| File | Responsibility |
|------|---------------|
| `workflows/shared/image_utils.py` | Add `build_material_grid()` — assemble labeled material tiles into a square grid image |
| `workflows/sketch_to_shoe_gemini/prompt_materials.yaml` | New file — prompt template for material-image mode with `$materials_instructions` placeholder |
| `workflows/sketch_to_shoe_gemini/pipeline.py` | Add `build_materials_prompt()`, `load_materials_prompt_config()`, extend `build_pipeline` to auto-detect and handle materials mode |
| `src/casadei/api/app.py` | Add `ref-material` upload + delete endpoints, extend `_run_variation_gemini` to load `materials_meta.json` |

---

## Chunk 1: Grid Builder + Prompt Template

### Task 1: `build_material_grid()` in `workflows/shared/image_utils.py`

**Files:**
- Modify: `workflows/shared/image_utils.py:215` (append after `build_sketch_grid`)

- [ ] **Step 1: Write the test**

Create `tests/test_material_grid.py`:

```python
"""Tests for build_material_grid in workflows/shared/image_utils.py."""
import sys
from pathlib import Path
from PIL import Image as PILImage

# Add shared utils to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "workflows" / "shared"))
from image_utils import build_material_grid


def _solid(color, size=(200, 300)):
    """Create a solid color test image."""
    return PILImage.new("RGB", size, color)


def test_single_material_grid():
    materials = [{"name": "Leather", "image": _solid((139, 69, 19)), "is_color": False}]
    grid, names = build_material_grid(materials)
    assert names == ["Leather"]
    assert grid.width == grid.height  # square output
    assert grid.width >= 500  # at least one tile wide


def test_auto_naming_material_and_color():
    materials = [
        {"name": None, "image": _solid((200, 0, 0)), "is_color": False},
        {"name": None, "image": _solid((0, 0, 255)), "is_color": True},
        {"name": None, "image": _solid((0, 128, 0)), "is_color": False},
    ]
    grid, names = build_material_grid(materials)
    assert names == ["Material 1", "Color 1", "Material 2"]
    assert grid.width == grid.height  # square


def test_custom_name_preserved():
    materials = [
        {"name": "Suede A", "image": _solid((100, 100, 100)), "is_color": False},
        {"name": None, "image": _solid((50, 50, 50)), "is_color": False},
    ]
    grid, names = build_material_grid(materials)
    assert names == ["Suede A", "Material 1"]


def test_two_materials_grid_is_2x1():
    materials = [
        {"name": "M1", "image": _solid((255, 0, 0)), "is_color": False},
        {"name": "M2", "image": _solid((0, 255, 0)), "is_color": False},
    ]
    grid, names = build_material_grid(materials)
    # 2 cols x 1 row = 1000 x 530, padded to square = 1000 x 1000
    assert grid.width == grid.height


def test_resolution_cap():
    # 5x5 grid = 25 materials, 2500x2650 raw → should be downscaled
    materials = [
        {"name": f"M{i}", "image": _solid((i * 10, i * 10, i * 10)), "is_color": False}
        for i in range(25)
    ]
    grid, names = build_material_grid(materials)
    assert grid.width <= 2048
    assert grid.height <= 2048
    assert len(names) == 25


def test_empty_raises():
    try:
        build_material_grid([])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/innovina/Documents/casadei && python -m pytest tests/test_material_grid.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_material_grid'`

- [ ] **Step 3: Implement `build_material_grid`**

Append to `workflows/shared/image_utils.py` after line 214 (end of `build_sketch_grid`):

```python
# ---------------------------------------------------------------------------
# Material grid assembly
# ---------------------------------------------------------------------------

_TILE_SIZE = 500
_LABEL_HEIGHT = 30
_LABEL_FONT_SIZE = 28
_MAX_GRID_PX = 2048


def _resolve_material_names(materials: list[dict]) -> list[str]:
    """Assign default names to materials that have name=None.

    Materials get "Material N", colors get "Color N".
    Numbering is per-type, independent.
    """
    mat_counter = 0
    color_counter = 0
    names: list[str] = []
    for entry in materials:
        if entry.get("name"):
            names.append(entry["name"])
        elif entry.get("is_color"):
            color_counter += 1
            names.append(f"Color {color_counter}")
        else:
            mat_counter += 1
            names.append(f"Material {mat_counter}")
    return names


def _build_labeled_tile(
    img: PILImage.Image,
    label: str,
    tile_w: int = _TILE_SIZE,
    tile_h: int = _TILE_SIZE,
    label_h: int = _LABEL_HEIGHT,
    font_size: int = _LABEL_FONT_SIZE,
) -> PILImage.Image:
    """Build a single tile: label strip on top, image fitted below."""
    from PIL import ImageDraw, ImageFont

    total_h = label_h + tile_h
    tile = PILImage.new("RGB", (tile_w, total_h), (255, 255, 255))

    # Draw label
    draw = ImageDraw.Draw(tile)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
        )
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_x = (tile_w - text_w) // 2
    draw.text((text_x, 2), label, fill=(0, 0, 0), font=font)

    # Fit image into tile_w x tile_h box preserving aspect ratio
    ow, oh = img.size
    scale = min(tile_w / ow, tile_h / oh)
    new_w = round(ow * scale)
    new_h = round(oh * scale)
    resized = img.convert("RGB").resize((new_w, new_h), PILImage.LANCZOS)
    paste_x = (tile_w - new_w) // 2
    paste_y = label_h + (tile_h - new_h) // 2
    tile.paste(resized, (paste_x, paste_y))

    return tile


def build_material_grid(
    materials: list[dict],
) -> tuple[PILImage.Image, list[str]]:
    """Assemble material/color images into a labeled square grid.

    Each tile: label strip (~30px) above a 500x500 image box.
    Grid is padded to square. Downscaled if > 2048px on any side.

    Args:
        materials: List of dicts with 'image' (PIL.Image), 'name' (str|None),
                   'is_color' (bool). Other fields ignored here.

    Returns:
        (grid_image, resolved_names) — square PIL image and list of assigned names.
    """
    if not materials:
        raise ValueError("No materials provided.")

    names = _resolve_material_names(materials)

    tiles = []
    for entry, name in zip(materials, names):
        tile = _build_labeled_tile(entry["image"], name)
        tiles.append(tile)

    n = len(tiles)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    tile_w = tiles[0].width
    tile_h = tiles[0].height

    grid_w = cols * tile_w
    grid_h = rows * tile_h
    grid = PILImage.new("RGB", (grid_w, grid_h), (255, 255, 255))

    for idx, tile in enumerate(tiles):
        row = idx // cols
        col = idx % cols
        grid.paste(tile, (col * tile_w, row * tile_h))

    # Pad to square
    gw, gh = grid.size
    if gw != gh:
        size = max(gw, gh)
        square = PILImage.new("RGB", (size, size), (255, 255, 255))
        square.paste(grid, ((size - gw) // 2, (size - gh) // 2))
        grid = square

    # Downscale if exceeds max resolution
    w, h = grid.size
    if max(w, h) > _MAX_GRID_PX:
        scale = _MAX_GRID_PX / max(w, h)
        grid = grid.resize(
            (round(w * scale), round(h * scale)), PILImage.LANCZOS
        )

    return grid, names
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/innovina/Documents/casadei && python -m pytest tests/test_material_grid.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add workflows/shared/image_utils.py tests/test_material_grid.py
git commit -m "feat: add build_material_grid to shared image utils"
```

---

### Task 2: Create `prompt_materials.yaml`

**Files:**
- Create: `workflows/sketch_to_shoe_gemini/prompt_materials.yaml`

- [ ] **Step 1: Create the prompt template file**

Write `workflows/sketch_to_shoe_gemini/prompt_materials.yaml`:

```yaml
prompt_template: |
  Generate a studio product photo of the shoe shown in the sketch.
  Study the sketch carefully before generating. Reproduce every line and every open area exactly as drawn — if a structural part is not drawn in the sketch, it must not appear in the photo. Do not add, complete, or assume any structure that is absent from the sketch. Apply the materials/colors from the reference image on top of the sketch shape. Shoot at the specified camera angle — do not copy the sketch's viewpoint.

  $materials_instructions

  Camera angle: $camera_desc
  Staging: $staging_desc
  $extra_specs
  Clean white background, professional studio lighting, sharp focus. $foot_framing
  $feedback

default_params:
  temperature: 0.8

negative_prompt: ""
```

- [ ] **Step 2: Verify YAML loads**

Run: `cd /home/innovina/Documents/casadei && python -c "import yaml; d = yaml.safe_load(open('workflows/sketch_to_shoe_gemini/prompt_materials.yaml')); print('OK:', list(d.keys())); assert 'materials_instructions' in d['prompt_template']"`
Expected: `OK: ['prompt_template', 'default_params', 'negative_prompt']`

- [ ] **Step 3: Commit**

```bash
git add workflows/sketch_to_shoe_gemini/prompt_materials.yaml
git commit -m "feat: add material-image prompt template for sketch-to-shoe"
```

---

## Chunk 2: Pipeline Extension

### Task 3: Add `build_materials_prompt` and `load_materials_prompt_config` to pipeline.py

**Files:**
- Modify: `workflows/sketch_to_shoe_gemini/pipeline.py:33` (add import of `build_material_grid`)
- Modify: `workflows/sketch_to_shoe_gemini/pipeline.py:44-54` (add materials prompt loader)
- Modify: `workflows/sketch_to_shoe_gemini/pipeline.py:66` (add `build_materials_prompt` function before `build_pipeline`)

- [ ] **Step 1: Write the test**

Create `tests/test_materials_prompt.py`:

```python
"""Tests for build_materials_prompt in sketch_to_shoe_gemini pipeline."""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project / "workflows" / "shared"))
sys.path.insert(0, str(_project / "workflows" / "sketch_to_shoe" / "scripts"))

from workflows.sketch_to_shoe_gemini.pipeline import build_materials_prompt


def test_single_material_no_placement():
    materials = [{"name": None, "placement": None, "note": None, "is_color": False}]
    names = ["Material 1"]
    result = build_materials_prompt(materials, names)
    assert "material" in result.lower()
    assert "reference image" in result.lower()
    assert "Material 1" not in result  # single mode doesn't use name


def test_single_color_no_placement():
    materials = [{"name": None, "placement": None, "note": None, "is_color": True}]
    names = ["Color 1"]
    result = build_materials_prompt(materials, names)
    assert "color" in result.lower()


def test_single_material_with_placement():
    materials = [{"name": "Suede", "placement": "toe", "note": None, "is_color": False}]
    names = ["Suede"]
    result = build_materials_prompt(materials, names)
    assert "toe" in result


def test_single_material_with_note():
    materials = [{"name": None, "placement": None, "note": "matte finish in daylight", "is_color": False}]
    names = ["Material 1"]
    result = build_materials_prompt(materials, names)
    assert "matte finish in daylight" in result


def test_multiple_materials():
    materials = [
        {"name": "Suede A", "placement": "toe", "note": "soft nubuck", "is_color": False},
        {"name": None, "placement": "heel", "note": None, "is_color": True},
    ]
    names = ["Suede A", "Color 1"]
    result = build_materials_prompt(materials, names)
    assert '"Suede A"' in result
    assert '"Color 1"' in result
    assert "toe" in result
    assert "heel" in result
    assert "soft nubuck" in result
    assert "material" in result.lower()
    assert "color" in result.lower()


def test_multiple_all_colors():
    materials = [
        {"name": None, "placement": "upper", "note": None, "is_color": True},
        {"name": None, "placement": "sole", "note": None, "is_color": True},
    ]
    names = ["Color 1", "Color 2"]
    result = build_materials_prompt(materials, names)
    assert '"Color 1"' in result
    assert '"Color 2"' in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/innovina/Documents/casadei && python -m pytest tests/test_materials_prompt.py -v`
Expected: FAIL with `ImportError: cannot import name 'build_materials_prompt'`

- [ ] **Step 3: Add imports and `load_materials_prompt_config` to pipeline.py**

In `workflows/sketch_to_shoe_gemini/pipeline.py`:

Add to the import line 33 (after `build_sketch_grid`):

```python
from image_utils import get_camera_preset, get_judge_notes, foot_framing, build_sketch_grid, build_material_grid  # noqa: E402
```

Add after `load_prompt_config()` (after line 54), before `build_extra_specs_text`:

```python
_MATERIALS_PROMPT_PATH = _WORKFLOW_DIR / "prompt_materials.yaml"
_cached_materials_prompt: dict | None = None


def load_materials_prompt_config() -> dict:
    """Load and cache the prompt_materials.yaml config."""
    global _cached_materials_prompt
    if _cached_materials_prompt is None:
        with open(_MATERIALS_PROMPT_PATH) as f:
            _cached_materials_prompt = yaml.safe_load(f)
    return _cached_materials_prompt
```

- [ ] **Step 4: Add `build_materials_prompt` function**

Add after `build_extra_specs_text` (after line 65), before `MAX_ITERATIONS`:

```python
def build_materials_prompt(
    materials: list[dict],
    resolved_names: list[str],
) -> str:
    """Build the $materials_instructions text for the prompt template.

    Single material (no placement): "Apply the material/color shown in the reference image to the shoe."
    Single material (with placement): "Apply the material/color shown in the reference image to the {placement} of the shoe."
    Multiple materials: bullet list mapping each named material to its placement.
    """
    n = len(materials)

    if n == 1:
        entry = materials[0]
        kind = "color" if entry.get("is_color") else "material"
        placement = entry.get("placement")
        note = entry.get("note")

        if placement:
            line = f"Apply the {kind} shown in the reference image to the {placement} of the shoe."
        else:
            line = f"Apply the {kind} shown in the reference image to the shoe."

        parts = [line]
        if note:
            parts.append(note)
        return "\n".join(parts)

    # Multiple materials
    lines = ["The reference image contains labeled material/color swatches. Apply each as follows:"]
    for entry, name in zip(materials, resolved_names):
        kind = "color" if entry.get("is_color") else "material"
        placement = entry.get("placement", "")
        note = entry.get("note")
        bullet = f'- Apply the {kind} shown in "{name}" to the {placement}.'
        if note:
            bullet += f" {note}"
        lines.append(bullet)
    return "\n".join(lines)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /home/innovina/Documents/casadei && python -m pytest tests/test_materials_prompt.py -v`
Expected: All 7 tests PASS

- [ ] **Step 6: Commit**

```bash
git add workflows/sketch_to_shoe_gemini/pipeline.py tests/test_materials_prompt.py
git commit -m "feat: add build_materials_prompt and materials prompt loader"
```

---

### Task 4: Extend `build_pipeline` for materials mode

**Files:**
- Modify: `workflows/sketch_to_shoe_gemini/pipeline.py:75-224` (`build_pipeline` function)

- [ ] **Step 1: Write the test**

Add to `tests/test_materials_prompt.py`:

```python
from PIL import Image as PILImage
from workflows.sketch_to_shoe_gemini.pipeline import build_pipeline


def _solid(color=(128, 128, 128), size=(200, 200)):
    return PILImage.new("RGB", size, color)


def test_build_pipeline_materials_mode_returns_pipeline():
    """build_pipeline with spec['materials'] should use materials prompt template."""
    from casadei import ImageMedia
    spec = {
        "material": "ignored",
        "camera_angle": "3/4",
        "extra": {},
        "materials": [
            {"name": "Test Mat", "image": _solid(), "placement": "toe", "note": None, "is_color": False},
        ],
    }
    # We need a VLMSession mock — just check it doesn't crash during pipeline construction
    # VLMSession is only used for best_fn, which is called at runtime not construction
    import unittest.mock as mock
    vlm = mock.MagicMock()
    pipeline, agent, sessions = build_pipeline(spec, vlm, foot="pair", temperature=0.8)
    assert pipeline.name == "sketch_to_shoe_gemini"
    # The agent's prompt template should contain materials_instructions
    assert "materials" in agent.config.prompt_template.lower() or "reference image" in agent.config.prompt_template.lower()


def test_build_pipeline_text_mode_unchanged():
    """build_pipeline without spec['materials'] should use text prompt template."""
    import unittest.mock as mock
    spec = {
        "material": "red leather",
        "camera_angle": "3/4",
        "extra": {},
    }
    vlm = mock.MagicMock()
    pipeline, agent, sessions = build_pipeline(spec, vlm, foot="pair")
    assert "Materials: " in agent.config.prompt_template or "$material" in agent.config.prompt_template
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/innovina/Documents/casadei && python -m pytest tests/test_materials_prompt.py::test_build_pipeline_materials_mode_returns_pipeline -v`
Expected: FAIL (materials mode not yet handled in `build_pipeline`)

- [ ] **Step 3: Modify `build_pipeline` to handle materials mode**

Replace `build_pipeline` body (lines 75-224 of `pipeline.py`). The key changes are in the first section — detect materials, switch prompt template, build grid, build materials_instructions, adjust input_map. The judges/loop section is unchanged.

The modified `build_pipeline` should:

1. At the top (after line 93 `prompt_config = load_prompt_config()`), add materials detection:

```python
    # --- Detect materials mode ---
    materials_list = spec.get("materials")
    use_materials_mode = bool(materials_list)

    if use_materials_mode:
        prompt_config = load_materials_prompt_config()
        grid_image, resolved_names = build_material_grid(materials_list)
        materials_instructions = build_materials_prompt(materials_list, resolved_names)
    else:
        materials_instructions = None
        grid_image = None

    prompt_template = prompt_config["prompt_template"]
```

Replace lines 94-95 (`prompt_config = load_prompt_config()` and `prompt_template = ...`) with the block above. Note `load_prompt_config()` is still called as the first line for text mode — we override `prompt_config` only in materials mode.

2. In the `ref_images` handling section (lines 99-118), wrap it so ref_images are skipped in materials mode:

```python
    # Handle optional reference images in spec (text mode only)
    ref_images: dict = {} if use_materials_mode else spec.get("ref_images", {})
    material_ref: ImageMedia | None = ref_images.get("material_ref")
    color_ref: ImageMedia | None = ref_images.get("color_ref")
```

The rest of the ref_images block (lines 104-118) stays the same — it just won't run since `ref_images` is empty in materials mode.

3. In the `input_map` section (lines 131-136), add materials grid:

```python
    # Build input_map — sketch first (determines aspect ratio), then materials grid or ref images
    input_map: dict[str, str] = {"image": "sketch"}
    if use_materials_mode:
        input_map["materials_grid"] = "materials_grid"
    else:
        if material_ref is not None:
            input_map["material_ref"] = "material_ref"
        if color_ref is not None:
            input_map["color_ref"] = "color_ref"
```

4. In the `template_kwargs` section (lines 143-149), switch between `material` and `materials_instructions`:

```python
    template_kwargs = {
        **camera_preset,
        "extra_specs": extra_specs_text,
        "foot_framing": foot_framing(foot, emphatic=False),
        "feedback": "",
    }
    if use_materials_mode:
        template_kwargs["materials_instructions"] = materials_instructions
    else:
        template_kwargs["material"] = spec.get("material", "black patent leather")
```

5. The function also needs to return `grid_image` so the caller (`_run_variation_gemini`) can add it to the context. Update the return type and docstring:

Change return signature to: `tuple[Pipeline, Agent, list[VLMSession], PILImage.Image | None]`

At the end (line 224): `return Pipeline(name="sketch_to_shoe_gemini", steps=[loop]), gemini_agent, [session_camera, session_count], grid_image`

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/innovina/Documents/casadei && python -m pytest tests/test_materials_prompt.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Run existing tests to check no regressions**

Run: `cd /home/innovina/Documents/casadei && python -m pytest tests/test_material_grid.py tests/test_materials_prompt.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add workflows/sketch_to_shoe_gemini/pipeline.py tests/test_materials_prompt.py
git commit -m "feat: extend build_pipeline to auto-detect materials mode"
```

---

## Chunk 3: API Endpoints + Integration

### Task 5: Update `_run_variation_gemini` to handle materials + updated return type

**Files:**
- Modify: `src/casadei/api/app.py:1538-1666` (`_run_variation_gemini`)

- [ ] **Step 1: Update `_run_variation_gemini` to load materials**

In `_run_variation_gemini`, after the `ref_images` loading block (after line 1624) and before the `spec = {` line (line 1626), add materials loading:

```python
            # Load material images if available (material-image mode)
            materials_list = None
            materials_meta_path = var_results_dir / "materials_meta.json"
            if materials_meta_path.exists():
                import json as _json
                meta_list = _json.loads(materials_meta_path.read_text())
                if meta_list:
                    materials_list = []
                    for entry in sorted(meta_list, key=lambda e: e["index"]):
                        img_path = var_results_dir / f"ref_mat_{entry['index']}.png"
                        if img_path.exists():
                            materials_list.append({
                                "name": entry.get("name"),
                                "image": PILImage.open(img_path).convert("RGB"),
                                "placement": entry.get("placement"),
                                "note": entry.get("note"),
                                "is_color": entry.get("is_color", False),
                            })
```

Then update the `spec` dict (line 1626-1631) to include materials:

```python
            spec = {
                "material": material_str,
                "camera_angle": "3/4",
                "extra": extra_spec,
                "ref_images": ref_images,
            }
            if materials_list:
                spec["materials"] = materials_list
```

- [ ] **Step 2: Update `build_pipeline` call to handle new return type**

The `build_pipeline` return type now includes `grid_image` as fourth element. Update line 1653:

```python
                    pipeline, edit_agent, _vlm_sessions, grid_image = build_pipeline(
                        spec=spec,
                        vlm_session=vlm_session,
                        foot="pair",
                        temperature=0.8,
                    )
```

And update the context building (lines 1660-1666) to add the grid image:

```python
                    context: dict = {
                        "sketch": sketch_media,
                        "image": sketch_media,
                    }
                    # Add materials grid or reference images to context
                    if grid_image is not None:
                        context["materials_grid"] = ImageMedia(image=grid_image)
                    else:
                        for ref_key, ref_media in ref_images.items():
                            context[ref_key] = ref_media
```

- [ ] **Step 3: Verify the import still works**

Run: `cd /home/innovina/Documents/casadei && python -c "import sys; sys.path.insert(0, 'workflows/shared'); sys.path.insert(0, 'workflows/sketch_to_shoe/scripts'); import casadei.api.app; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add src/casadei/api/app.py
git commit -m "feat: extend _run_variation_gemini to load materials and pass grid to pipeline"
```

---

### Task 6: Add `ref-material` upload endpoint

**Files:**
- Modify: `src/casadei/api/app.py` (add after the existing `ref-image` endpoint, ~line 1843)

- [ ] **Step 1: Add the upload endpoint**

Insert after the `upload_variation_ref_image` endpoint (after line 1843):

```python
    @app.post(
        "/api/products/{product_id}/variations/{variation_id}/ref-material",
        status_code=200,
    )
    async def upload_variation_ref_material(
        product_id: str,
        variation_id: str,
        file: UploadFile,
        index: int = fastapi.Query(...),
        name: str = fastapi.Query(""),
        placement: str = fastapi.Query(""),
        note: str = fastapi.Query(""),
        is_color: bool = fastapi.Query(False),
        request: Request = None,
    ):
        """Upload a material/color reference image for a variation."""
        _get_current_user(request)
        product = store.get_product(product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        variation = None
        for v in product.variations:
            if v.id == variation_id:
                variation = v
                break
        if not variation:
            raise HTTPException(status_code=404, detail="Variation not found")

        var_dir = results_dir / product_id / variation_id
        var_dir.mkdir(parents=True, exist_ok=True)

        # Save image
        filename = f"ref_mat_{index}.png"
        content = await file.read()
        from PIL import Image as PILImage
        import io
        img = PILImage.open(io.BytesIO(content)).convert("RGB")
        img.save(str(var_dir / filename), "PNG")

        # Update materials_meta.json
        import json as _json
        meta_path = var_dir / "materials_meta.json"
        meta_list = []
        if meta_path.exists():
            meta_list = _json.loads(meta_path.read_text())

        # Remove existing entry with same index (overwrite)
        meta_list = [e for e in meta_list if e["index"] != index]
        meta_list.append({
            "index": index,
            "name": name or None,
            "placement": placement or None,
            "note": note or None,
            "is_color": is_color,
        })
        meta_list.sort(key=lambda e: e["index"])
        meta_path.write_text(_json.dumps(meta_list, indent=2))

        return {"filename": filename, "index": index}
```

- [ ] **Step 2: Verify endpoint loads**

Run: `cd /home/innovina/Documents/casadei && python -c "import sys; sys.path.insert(0, 'workflows/shared'); sys.path.insert(0, 'workflows/sketch_to_shoe/scripts'); from casadei.api.app import create_app; app = create_app(); routes = [r.path for r in app.routes if hasattr(r, 'path')]; assert '/api/products/{product_id}/variations/{variation_id}/ref-material' in routes; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/casadei/api/app.py
git commit -m "feat: add ref-material upload endpoint"
```

---

### Task 7: Add `ref-material` delete endpoint

**Files:**
- Modify: `src/casadei/api/app.py` (add after the upload endpoint)

- [ ] **Step 1: Add the delete endpoint**

Insert after the `upload_variation_ref_material` endpoint:

```python
    @app.delete(
        "/api/products/{product_id}/variations/{variation_id}/ref-material/{index}",
        status_code=200,
    )
    def delete_variation_ref_material(
        product_id: str,
        variation_id: str,
        index: int,
        request: Request = None,
    ):
        """Delete a material/color reference image from a variation."""
        _get_current_user(request)
        product = store.get_product(product_id)
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        variation = None
        for v in product.variations:
            if v.id == variation_id:
                variation = v
                break
        if not variation:
            raise HTTPException(status_code=404, detail="Variation not found")

        var_dir = results_dir / product_id / variation_id

        # Remove image file
        img_path = var_dir / f"ref_mat_{index}.png"
        if img_path.exists():
            img_path.unlink()

        # Update materials_meta.json
        import json as _json
        meta_path = var_dir / "materials_meta.json"
        if meta_path.exists():
            meta_list = _json.loads(meta_path.read_text())
            meta_list = [e for e in meta_list if e["index"] != index]
            if meta_list:
                meta_path.write_text(_json.dumps(meta_list, indent=2))
            else:
                meta_path.unlink()

        return {"deleted": index}
```

- [ ] **Step 2: Verify both endpoints load**

Run: `cd /home/innovina/Documents/casadei && python -c "import sys; sys.path.insert(0, 'workflows/shared'); sys.path.insert(0, 'workflows/sketch_to_shoe/scripts'); from casadei.api.app import create_app; app = create_app(); routes = [r.path for r in app.routes if hasattr(r, 'path')]; print([r for r in routes if 'ref-material' in r])"`
Expected: `['/api/products/{product_id}/variations/{variation_id}/ref-material', '/api/products/{product_id}/variations/{variation_id}/ref-material/{index}']`

- [ ] **Step 3: Commit**

```bash
git add src/casadei/api/app.py
git commit -m "feat: add ref-material delete endpoint"
```

---

### Task 8: End-to-end verification

**Files:** None (verification only)

- [ ] **Step 1: Verify all imports work together**

Run:
```bash
cd /home/innovina/Documents/casadei && python -c "
import sys, os
os.environ.setdefault('GEMINI_API_KEY', 'dummy')
for p in ['workflows/shared', 'workflows/sketch_to_shoe/scripts']:
    if p not in sys.path:
        sys.path.insert(0, p)

# Test grid builder
from image_utils import build_material_grid
from PIL import Image as PILImage
materials = [
    {'name': 'Suede', 'image': PILImage.new('RGB', (300, 400), (139, 69, 19)), 'is_color': False},
    {'name': None, 'image': PILImage.new('RGB', (200, 200), (0, 0, 255)), 'is_color': True},
]
grid, names = build_material_grid(materials)
print(f'Grid: {grid.size}, Names: {names}')
assert grid.width == grid.height
assert names == ['Suede', 'Color 1']

# Test prompt builder
from workflows.sketch_to_shoe_gemini.pipeline import build_materials_prompt
prompt = build_materials_prompt(
    [{'placement': 'toe', 'note': 'soft', 'is_color': False}, {'placement': 'heel', 'note': None, 'is_color': True}],
    names,
)
print(f'Prompt: {prompt[:100]}...')
assert 'Suede' in prompt
assert 'Color 1' in prompt

# Test pipeline materials mode detection
import unittest.mock as mock
from workflows.sketch_to_shoe_gemini.pipeline import build_pipeline
spec = {
    'material': 'ignored',
    'camera_angle': '3/4',
    'extra': {},
    'materials': [{'name': 'Test', 'image': PILImage.new('RGB', (100,100)), 'placement': 'toe', 'note': None, 'is_color': False}],
}
pipeline, agent, sessions, grid_img = build_pipeline(spec, mock.MagicMock())
print(f'Pipeline: {pipeline.name}, Grid returned: {grid_img is not None}')
assert grid_img is not None

# Test app loads
import casadei.api.app
print('App module: OK')

print()
print('=== ALL CHECKS PASSED ===')
"
```

Expected: `=== ALL CHECKS PASSED ===`

- [ ] **Step 2: Run all tests**

Run: `cd /home/innovina/Documents/casadei && python -m pytest tests/test_material_grid.py tests/test_materials_prompt.py -v`
Expected: All tests PASS

- [ ] **Step 3: Final commit if any cleanup needed**

```bash
git status
# If clean, no commit needed
```
