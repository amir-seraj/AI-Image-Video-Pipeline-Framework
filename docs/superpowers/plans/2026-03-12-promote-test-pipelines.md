# Promote Test Pipelines to Workflows — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract reusable pipeline logic from three test scripts into proper `workflows/` modules with YAML-driven config, then update `app.py` to import from workflows instead of tests.

**Architecture:** Data (camera presets, prompts) lives in YAML files under `workflows/shared/` and per-workflow dirs. Utilities (image processing) live in `workflows/shared/image_utils.py`. Each workflow's `pipeline.py` is compact orchestration that imports from shared modules. `app.py` rewires imports from `tests/` to `workflows/`. Test scripts are NOT modified.

**Tech Stack:** Python 3.12+, PyYAML, PIL/Pillow, FastAPI, Gemini API (google-genai)

---

## File Map

**Create:**
- `workflows/shared/camera_presets.yaml` — all camera presets, aliases, judge notes, canonical angles, pair/single sets
- `workflows/shared/image_utils.py` — `pad_to_ratio`, `find_ratio`, `build_sketch_grid`, `foot_framing`, `load_camera_presets`, `get_camera_preset`
- `workflows/sketch_to_shoe_gemini_direct/prompt.yaml` — prompt template + default params
- `workflows/sketch_to_shoe_gemini_direct/pipeline.py` — `build_pipeline`, `save_results`
- `workflows/shoe_tryon_gemini/prompt.yaml` — prompt template + default params
- `workflows/shoe_tryon_gemini/pipeline.py` — `build_pipeline`, `save_results`
- `workflows/shoe_angles/prompt.yaml` — prompt template
- `workflows/shoe_angles/pipeline.py` — `generate_angle`, `generate_angle_with_judge`

**Modify:**
- `src/casadei/api/app.py` — rewire imports, add `mode` param to try-on, add `judged` param to angles

**Do NOT modify:**
- `tests/run_sketch_to_shoe_gemini_direct.py`
- `tests/run_shoe_tryon_gemini.py`
- `tests/run_shoe_angles.py`

---

## Chunk 1: Shared Infrastructure

### Task 1: Create `workflows/shared/camera_presets.yaml`

**Files:**
- Create: `workflows/shared/camera_presets.yaml`

This is the canonical source of truth for all camera angle data. Content extracted from `tests/run_shoe_angles.py` (most complete/current version).

- [ ] **Step 1: Create the YAML file**

The YAML must contain:
- `presets` — the full `CAMERA_PRESETS` dict (all 9 angles x 3 foot variants), each with `camera_desc` and `staging_desc`
- `aliases` — mapping of alias names to preset keys (e.g. `"3/4 view": "3/4"`, `"hero": "hero-front-right"`)
- `judge_notes` — the `_CAMERA_JUDGE_NOTES` multi-line string (shared by sketch_to_shoe and shoe_angles)
- `canonical_angles` — ordered list of angle names
- `pair_angles` — list of angles that default to showing a pair
- `single_angles` — list of angles that default to showing a single shoe

Source data: copy from `tests/run_shoe_angles.py` lines 71-414 (the `CAMERA_PRESETS` dict, `CANONICAL_ANGLES`, `PAIR_ANGLES`, `SINGLE_ANGLES`), and lines 379-402 (the `_CAMERA_JUDGE_NOTES`). Also pick up the aliases from lines 370-375.

Structure:
```yaml
judge_notes: |
  EVALUATE camera_angle using these steps in order:
  ...entire judge_notes text...

canonical_angles:
  - "3/4"
  - side
  - front
  - back
  - top
  - hero-front-right
  - hero-front-left
  - hero-back-right
  - hero-back-left

pair_angles:
  - "3/4"
  - front

single_angles:
  - side
  - back
  - top
  - hero-front-right
  - hero-front-left
  - hero-back-right
  - hero-back-left

aliases:
  "3/4 view": "3/4"
  "side view": side
  "front view": front
  "back view": back
  "top view": top
  hero: hero-front-right

presets:
  "3/4":
    pair:
      camera_desc: >-
        Low, ground-level angle, shooting almost parallel to the platform.
        The camera is positioned slightly to the right, looking leftwards
        at the shoes (3/4).
      staging_desc: >-
        A pair of shoes perfectly aligned and parallel, touching side by side,
        both pointing in the same direction at the same angle.
    left:
      camera_desc: >-
        ...
      staging_desc: >-
        ...
    right:
      camera_desc: >-
        ...
      staging_desc: >-
        ...
  side:
    ...
  # (all 9 angles with pair/left/right variants)
```

- [ ] **Step 2: Verify YAML loads correctly**

Run: `python -c "import yaml; d = yaml.safe_load(open('workflows/shared/camera_presets.yaml')); print(len(d['presets']), 'angles loaded'); print(d['canonical_angles']); print('judge_notes length:', len(d['judge_notes']))"`

Expected: `9 angles loaded`, the canonical list, and judge_notes length > 500.

- [ ] **Step 3: Commit**

```bash
git add workflows/shared/camera_presets.yaml
git commit -m "feat: add shared camera_presets.yaml — single source of truth for all angle data"
```

---

### Task 2: Create `workflows/shared/image_utils.py`

**Files:**
- Create: `workflows/shared/image_utils.py`

This module provides image processing utilities and camera preset loading. All functions are pure (no side effects beyond image manipulation).

- [ ] **Step 1: Create the module**

Contents — extracted from `tests/run_shoe_angles.py` and `tests/run_sketch_to_shoe_gemini_direct.py`:

```python
"""Shared image utilities for Casadei workflows."""
from __future__ import annotations

import math
from pathlib import Path

import yaml
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# Camera preset loading
# ---------------------------------------------------------------------------

_PRESETS_PATH = Path(__file__).parent / "camera_presets.yaml"
_cached_presets: dict | None = None


def load_camera_presets() -> dict:
    """Load and cache camera presets from YAML."""
    global _cached_presets
    if _cached_presets is None:
        with open(_PRESETS_PATH) as f:
            _cached_presets = yaml.safe_load(f)
    return _cached_presets


def get_camera_preset(angle: str, foot: str = "pair") -> dict[str, str]:
    """Resolve a camera preset by angle name and foot variant.

    Returns dict with 'camera_desc' and 'staging_desc' keys.
    Falls back to using the angle string as-is if not found.
    """
    data = load_camera_presets()
    key = angle.lower().strip()
    # Resolve aliases
    aliases = data.get("aliases", {})
    if key in aliases:
        key = aliases[key]
    presets = data.get("presets", {})
    if key in presets:
        return presets[key][foot]
    return {
        "camera_desc": angle,
        "staging_desc": "The shoe(s) placed on the white surface.",
    }


def get_judge_notes() -> str:
    """Return the camera angle judge evaluation rubric."""
    return load_camera_presets()["judge_notes"]


def get_canonical_angles() -> list[str]:
    """Return the ordered list of canonical angle names."""
    return load_camera_presets()["canonical_angles"]


def get_pair_angles() -> set[str]:
    """Return angles that default to showing a pair."""
    return set(load_camera_presets()["pair_angles"])


def get_single_angles() -> set[str]:
    """Return angles that default to showing a single shoe."""
    return set(load_camera_presets()["single_angles"])


def foot_for_angle(angle: str, foot: str, single: bool = False) -> str:
    """Return the effective foot variant for a given angle.

    Pair angles use 'pair' unless single=True.
    Single angles use the provided foot value.
    """
    if single:
        return foot
    key = angle.lower().strip()
    if key in get_pair_angles():
        return "pair"
    return foot


# ---------------------------------------------------------------------------
# Foot framing prompt fragments
# ---------------------------------------------------------------------------

def foot_framing(foot: str, emphatic: bool = True) -> str:
    """Return a foot-specific prompt fragment.

    emphatic=True (default): detailed instructions for angle generation
    where reference images may show pairs.
    emphatic=False: simpler instructions for sketch-to-shoe generation.
    """
    if emphatic:
        if foot == "pair":
            return (
                "IMPORTANT — NUMBER OF SHOES: The output MUST contain exactly TWO shoes "
                "(a matching pair — left and right) placed side by side. "
                "The reference image shows a pair; keep both shoes in the output. "
                "Do NOT show only one shoe."
            )
        elif foot == "left":
            return (
                "IMPORTANT — NUMBER OF SHOES: The output MUST contain exactly ONE shoe — "
                "the LEFT shoe only, centered in the frame. "
                "Even though the reference image may show two shoes, generate ONLY the "
                "left shoe. Do NOT include the right shoe. Only one shoe in the image."
            )
        else:
            return (
                "IMPORTANT — NUMBER OF SHOES: The output MUST contain exactly ONE shoe — "
                "the RIGHT shoe only, centered in the frame. "
                "Even though the reference image may show two shoes, generate ONLY the "
                "right shoe. Do NOT include the left shoe. Only one shoe in the image."
            )
    else:
        if foot == "pair":
            return (
                "Show a matching pair of shoes — both left and right — "
                "centered side by side."
            )
        elif foot == "left":
            return "Show the left shoe only, centered and fully visible."
        else:
            return "Show the right shoe only, centered and fully visible."


# ---------------------------------------------------------------------------
# Aspect ratio helpers
# ---------------------------------------------------------------------------

MAX_INPUT_SIZE = 1024

SUPPORTED_RATIOS: list[tuple[int, int]] = [
    (1, 1), (1, 4), (1, 8),
    (2, 3), (3, 2), (3, 4), (4, 3),
    (4, 5), (5, 4),
    (8, 1), (9, 16), (16, 9), (21, 9),
]


def find_ratio(w: int, h: int) -> tuple[int, int]:
    """Find the nearest supported aspect ratio for dimensions w x h."""
    target = w / h
    return min(SUPPORTED_RATIOS, key=lambda r: abs(r[0] / r[1] - target))


def pad_to_ratio(
    img: PILImage.Image,
    ratio: tuple[int, int],
    max_size: int = MAX_INPUT_SIZE,
) -> PILImage.Image:
    """Pad and scale an image to the given aspect ratio, fitting within max_size."""
    wr, hr = ratio
    orig_w, orig_h = img.size

    if orig_w / orig_h <= wr / hr:
        canvas_h = orig_h
        canvas_w = round(orig_h * wr / hr)
    else:
        canvas_w = orig_w
        canvas_h = round(orig_w * hr / wr)

    scale = max_size / max(canvas_w, canvas_h)
    final_w = round(canvas_w * scale)
    final_h = round(canvas_h * scale)

    img_w = round(orig_w * scale)
    img_h = round(orig_h * scale)
    scaled = img.resize((img_w, img_h), PILImage.LANCZOS)

    canvas = PILImage.new("RGB", (final_w, final_h), (255, 255, 255))
    canvas.paste(scaled, ((final_w - img_w) // 2, (final_h - img_h) // 2))
    return canvas


# ---------------------------------------------------------------------------
# Sketch grid assembly
# ---------------------------------------------------------------------------

def build_sketch_grid(
    images: list[PILImage.Image],
    spacing: int = 20,
) -> PILImage.Image:
    """Assemble multiple sketch images into a square grid."""
    if not images:
        raise ValueError("No sketch images provided.")

    n = len(images)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)

    cell_w = max(img.width for img in images)
    cell_h = max(img.height for img in images)

    total_w = cols * cell_w + (cols + 1) * spacing
    total_h = rows * cell_h + (rows + 1) * spacing
    grid = PILImage.new("RGB", (total_w, total_h), (255, 255, 255))

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        x = spacing + col * (cell_w + spacing) + (cell_w - img.width) // 2
        y = spacing + row * (cell_h + spacing) + (cell_h - img.height) // 2
        grid.paste(img, (x, y))

    gw, gh = grid.size
    if gw != gh:
        size = max(gw, gh)
        square = PILImage.new("RGB", (size, size), (255, 255, 255))
        square.paste(grid, ((size - gw) // 2, (size - gh) // 2))
        return square

    return grid
```

- [ ] **Step 2: Verify module imports correctly**

Run: `cd /home/innovina/Documents/casadei && python -c "import sys; sys.path.insert(0, 'workflows/shared'); from image_utils import load_camera_presets, get_camera_preset, build_sketch_grid, pad_to_ratio, find_ratio, foot_framing; p = get_camera_preset('3/4', 'pair'); print('OK:', p['camera_desc'][:40])"`

Expected: `OK: Low, ground-level angle, shooting almo`

- [ ] **Step 3: Commit**

```bash
git add workflows/shared/image_utils.py
git commit -m "feat: add shared image_utils.py — camera preset loading, image processing, foot framing"
```

---

## Chunk 2: Workflow Modules

### Task 3: Create `workflows/sketch_to_shoe_gemini_direct/`

**Files:**
- Create: `workflows/sketch_to_shoe_gemini_direct/prompt.yaml`
- Create: `workflows/sketch_to_shoe_gemini_direct/pipeline.py`

- [ ] **Step 1: Create prompt.yaml**

Extract from `tests/run_sketch_to_shoe_gemini_direct.py` lines 51-66:

```yaml
prompt_template: >-
  Generate a studio product photo of the shoe shown in the sketch.
  Study the sketch carefully before generating.
  Reproduce every line and every open area exactly as drawn —
  if a structural part is not drawn in the sketch, it must not appear in the photo.
  Do not add, complete, or assume any structure that is absent from the sketch.
  Apply the materials below on top of the sketch shape.
  Shoot at the specified camera angle — do not copy the sketch's viewpoint.

  Materials: $material

  Camera angle: $camera_desc
  Staging: $staging_desc
  $extra_specs
  Clean white background, professional studio lighting, sharp focus.
  $foot_framing
  $feedback

default_params:
  temperature: 0.8

negative_prompt: ""
```

Note: The prompt uses `$variable` syntax (Python `string.Template`).

- [ ] **Step 2: Create pipeline.py**

This module contains `build_pipeline` and `save_results`, extracted from `tests/run_sketch_to_shoe_gemini_direct.py`. It imports camera presets and utilities from `workflows/shared/`.

Key functions:
- `load_prompt_template()` — loads from `prompt.yaml`
- `build_extra_specs_text(extra)` — formats extra spec dict into text
- `build_pipeline(spec, vlm_session, foot, temperature)` — returns `(Pipeline, Agent, [VLMSession])`
- `save_results(run_dir, loop_result, ...)` — saves images + JSON + summary

The pipeline.py must:
1. Add `workflows/shared` to `sys.path` for `image_utils` import
2. Add `workflows/sketch_to_shoe/scripts` to `sys.path` for `judge` import
3. Import from `image_utils`: `get_camera_preset`, `get_judge_notes`, `foot_framing`, `build_sketch_grid`
4. Import from `judge`: `VLMSession`, `make_spec_judge`, `make_shoe_count_judge`, `make_best_fn`
5. Import from `casadei`: `Agent`, `AgentConfig`, `AgentStep`, `ImageMedia`, `Pipeline`
6. Import from `casadei.loop`: `LoopStep`, `LoopResult`

Source code: extract from `tests/run_sketch_to_shoe_gemini_direct.py`:
- `_build_extra_specs_text` (lines 457-460)
- `MAX_ITERATIONS = 3` (line 467)
- `_promote_spec_metadata` (lines 495-507) — only if still used
- `build_pipeline` (lines 510-615) — replace inline `PROMPT_TEMPLATE` and `_CAMERA_JUDGE_NOTES` with loaded versions, replace `_get_camera_preset` with `get_camera_preset` from shared, replace `_foot_framing(foot)` with `foot_framing(foot, emphatic=False)` from shared
- `save_results` (lines 622-734)

The pipeline.py should be ~150 lines. Use `yaml.safe_load` to read `prompt.yaml` at module load.

- [ ] **Step 3: Verify pipeline module loads**

Run: `cd /home/innovina/Documents/casadei && python -c "import sys; sys.path.insert(0, 'workflows/sketch_to_shoe_gemini_direct'); from pipeline import build_pipeline, save_results; print('sketch_to_shoe pipeline OK')"`

Expected: `sketch_to_shoe pipeline OK`

- [ ] **Step 4: Commit**

```bash
git add workflows/sketch_to_shoe_gemini_direct/
git commit -m "feat: add sketch_to_shoe_gemini_direct workflow module"
```

---

### Task 4: Create `workflows/shoe_tryon_gemini/`

**Files:**
- Create: `workflows/shoe_tryon_gemini/prompt.yaml`
- Create: `workflows/shoe_tryon_gemini/pipeline.py`

- [ ] **Step 1: Create prompt.yaml**

Extract from `tests/run_shoe_tryon_gemini.py` lines 44-51:

```yaml
prompt_template: >-
  Replace the shoes on the person's feet with the exact shoe shown in the
  first image. Match every visible detail of the reference shoe: color,
  material, heel shape, toe shape, straps, and hardware.
  Replace BOTH shoes — left foot AND right foot.
  Preserve the person's pose, legs, clothing, and background exactly.
  $feedback

default_params:
  max_iterations: 3
  tolerance: moderate

negative_prompt: ""
```

Note: `max_iterations` default changed to 3 per user request.

- [ ] **Step 2: Create pipeline.py**

Extract from `tests/run_shoe_tryon_gemini.py`. This module is simpler (no camera presets needed).

Key functions:
- `load_prompt_template()` — loads from `prompt.yaml`
- `build_pipeline(max_iterations, vlm_session, features, tolerance)` — returns `(Pipeline, Agent)`
- `save_results(run_dir, loop_result, ...)` — saves images + JSON + summary

The pipeline.py must:
1. Add `workflows/shoe_tryon_loop/scripts` to `sys.path` for `judge` import
2. Import from `judge`: `VLMSession`, `make_judge`, `make_best_fn`, `extract_features`
3. Import from `casadei`: `Agent`, `AgentConfig`, `AgentStep`, `ImageMedia`, `TextMedia`, `Pipeline`
4. Import from `casadei.loop`: `LoopStep`, `LoopResult`
5. Import from `casadei.providers.gemini_pricing`: `format_usage_summary`

Source code: extract from `tests/run_shoe_tryon_gemini.py`:
- `build_pipeline` (lines 54-111) — change `max_iterations` default from 5 to 3
- `save_results` (lines 114-250)

- [ ] **Step 3: Verify pipeline module loads**

Run: `cd /home/innovina/Documents/casadei && python -c "import sys; sys.path.insert(0, 'workflows/shoe_tryon_gemini'); from pipeline import build_pipeline, save_results; print('shoe_tryon_gemini pipeline OK')"`

Expected: `shoe_tryon_gemini pipeline OK`

- [ ] **Step 4: Commit**

```bash
git add workflows/shoe_tryon_gemini/
git commit -m "feat: add shoe_tryon_gemini workflow module (default 3 iterations)"
```

---

### Task 5: Create `workflows/shoe_angles/`

**Files:**
- Create: `workflows/shoe_angles/prompt.yaml`
- Create: `workflows/shoe_angles/pipeline.py`

- [ ] **Step 1: Create prompt.yaml**

Extract from `tests/run_shoe_angles.py` lines 505-518:

```yaml
prompt_template: >-
  The first image is the original shoe design sketch.
  The second image is a photorealistic product photograph of this exact shoe —
  use it as the definitive reference for everything: material, color, texture,
  proportions, and design details.

  Generate the exact same shoe from this camera angle:
  - Camera angle: {camera_desc}
  - Shoe alignment and staging: {staging_desc}

  {foot_framing}

  The result must be a studio-quality photograph: clean white background,
  professional product lighting, sharp focus, no shadows on background.
  The shoe must look identical to the reference — only the viewing angle changes.
  {feedback}
```

Note: This prompt uses `{variable}` syntax (Python `.format()`), different from sketch_to_shoe's `$variable`.

- [ ] **Step 2: Create pipeline.py**

Extract from `tests/run_shoe_angles.py`. This is the most complex module — it has both `generate_angle` (single-shot) and `generate_angle_with_judge` (iterative with 3 judges).

Key functions:
- `load_prompt_template()` — loads from `prompt.yaml`
- `generate_angle(client, sketch, reference, angle, foot, single, feedback)` — single-shot generation, returns `(angle, image, usage_dict)`
- `generate_angle_with_judge(client, sketch, reference, angle, foot, single, output_dir)` — iterative with camera+reference+count judges, returns `(angle, image, cost_usd)`
- `annotate_image(img, angle, iteration, angle_ok, ref_ok, count_ok)` — debug banner overlay

The pipeline.py must:
1. Add `workflows/shared` to `sys.path` for `image_utils` import
2. Add `workflows/sketch_to_shoe/scripts` to `sys.path` for `judge` import
3. Import from `image_utils`: `get_camera_preset`, `get_judge_notes`, `get_canonical_angles`, `get_pair_angles`, `foot_for_angle`, `foot_framing`, `find_ratio`, `pad_to_ratio`
4. Import from `judge`: `VLMSession`, `make_spec_judge`, `make_reference_fidelity_judge`, `make_shoe_count_judge`
5. Import from `casadei.media`: `ImageMedia`
6. Import from `casadei.providers.gemini_pricing`: `extract_token_usage`, `calculate_cost`
7. Import `google.genai` and `google.genai.types`

Source code: extract from `tests/run_shoe_angles.py`:
- `_GENERATION_MODEL_ID` (line 65)
- `MAX_JUDGE_ITERATIONS = 3` (line 377)
- `generate_angle` (lines 527-598) — replace inline `CAMERA_PRESETS` with `get_camera_preset()`, replace `_foot_framing` with `foot_framing()`, replace `_find_ratio`/`_pad_to_ratio` with shared versions
- `_annotate_image` (lines 601-626)
- `generate_angle_with_judge` (lines 629-809) — same replacements plus use `get_judge_notes()` instead of inline `_CAMERA_JUDGE_NOTES`

- [ ] **Step 3: Verify pipeline module loads**

Run: `cd /home/innovina/Documents/casadei && python -c "import sys; sys.path.insert(0, 'workflows/shoe_angles'); from pipeline import generate_angle, generate_angle_with_judge; print('shoe_angles pipeline OK')"`

Expected: `shoe_angles pipeline OK`

- [ ] **Step 4: Commit**

```bash
git add workflows/shoe_angles/
git commit -m "feat: add shoe_angles workflow module with judged generation"
```

---

## Chunk 3: App.py Integration

### Task 6: Rewire `_run_variation_gemini` imports

**Files:**
- Modify: `src/casadei/api/app.py:1538-1560` (import section of `_run_variation_gemini`)

- [ ] **Step 1: Update imports**

Change from:
```python
_project_root = Path(__file__).resolve().parent.parent.parent.parent
_scripts_dir = _project_root / "workflows" / "sketch_to_shoe" / "scripts"
_tests_dir = _project_root / "tests"
for p in [str(_scripts_dir), str(_tests_dir)]:
    if p not in _sys.path:
        _sys.path.insert(0, p)

from run_sketch_to_shoe_gemini_direct import build_pipeline, _build_sketch_grid
from judge import VLMSession
```

To:
```python
_project_root = Path(__file__).resolve().parent.parent.parent.parent
_workflow_dir = _project_root / "workflows" / "sketch_to_shoe_gemini_direct"
_shared_dir = _project_root / "workflows" / "shared"
_scripts_dir = _project_root / "workflows" / "sketch_to_shoe" / "scripts"
for p in [str(_workflow_dir), str(_shared_dir), str(_scripts_dir)]:
    if p not in _sys.path:
        _sys.path.insert(0, p)

from pipeline import build_pipeline
from image_utils import build_sketch_grid
from judge import VLMSession
```

Also update the one call site: change `_build_sketch_grid(raw_sketches)` to `build_sketch_grid(raw_sketches)` at line 1575.

- [ ] **Step 2: Verify the app still starts**

Run: `cd /home/innovina/Documents/casadei && python -c "from casadei.api.app import create_app; app = create_app(); print('app OK')"`

Expected: `app OK`

- [ ] **Step 3: Commit**

```bash
git add src/casadei/api/app.py
git commit -m "refactor: rewire _run_variation_gemini to import from workflows/"
```

---

### Task 7: Rewire `_run_generate_angles` and add `judged` param

**Files:**
- Modify: `src/casadei/api/app.py:1864-1982` (`_run_generate_angles` + endpoint)

- [ ] **Step 1: Update imports in `_run_generate_angles`**

Change from:
```python
_angles_script = _project_root / "tests"
if str(_angles_script) not in _sys.path:
    _sys.path.insert(0, str(_angles_script))

from run_shoe_angles import (
    generate_angle,
    CANONICAL_ANGLES,
)
```

To:
```python
_workflow_dir = _project_root / "workflows" / "shoe_angles"
_shared_dir = _project_root / "workflows" / "shared"
_scripts_dir = _project_root / "workflows" / "sketch_to_shoe" / "scripts"
for p in [str(_workflow_dir), str(_shared_dir), str(_scripts_dir)]:
    if p not in _sys.path:
        _sys.path.insert(0, p)

from pipeline import generate_angle, generate_angle_with_judge
from image_utils import get_canonical_angles
```

Change `CANONICAL_ANGLES` references to `get_canonical_angles()`.

- [ ] **Step 2: Add `judged` parameter to `_run_generate_angles`**

Add `judged: bool = True` parameter to `_run_generate_angles` function signature.

In the `_gen_angle` inner function, switch between:
```python
def _gen_angle(angle: str) -> tuple[str, PILImage.Image | None]:
    try:
        if job_manager.is_cancelled(job_id):
            return angle, None
        if judged:
            _, img, _cost = generate_angle_with_judge(
                client, sketch, reference, angle, foot, single=single,
                output_dir=var_results_dir,
            )
        else:
            _, img, _usage = generate_angle(client, sketch, reference, angle, foot, single=single)
        return angle, img
    except Exception:
        return angle, None
```

- [ ] **Step 3: Add `judged` query param to the endpoint**

Update the `generate_angles` endpoint (around line 1987):
```python
def generate_angles(
    product_id: str,
    variation_id: str,
    foot: str = Query("right", pattern="^(left|right)$"),
    single: bool = Query(False),
    angles: list[str] = Query(None, alias="angle"),
    judged: bool = Query(True),
    request: Request = None,
) -> dict:
```

And pass `judged=judged` to the `_run_generate_angles` kwargs.

- [ ] **Step 4: Verify the endpoint accepts the new param**

Run: `cd /home/innovina/Documents/casadei && python -c "from casadei.api.app import create_app; app = create_app(); print('app OK')"`

Expected: `app OK`

- [ ] **Step 5: Commit**

```bash
git add src/casadei/api/app.py
git commit -m "feat: rewire angles to workflows/, add judged=true|false param (default true)"
```

---

### Task 8: Add `mode=fast|quality` to try-on endpoint

**Files:**
- Modify: `src/casadei/api/app.py:2748-2822` (try-on endpoint + `_run_try_on`)

- [ ] **Step 1: Add mode param to endpoint**

Update the `try_on_model` endpoint:
```python
async def try_on_model(
    product_id: str,
    variation_id: str,
    model_photo: UploadFile = fastapi.File(...),
    mode: str = fastapi.Query("fast", pattern="^(fast|quality)$"),
    request: Request = None,
) -> dict:
```

Pass `mode` through to `_run_try_on`:
```python
thread = threading.Thread(
    target=_run_try_on,
    args=(product, variation, job_id, photo_path, user.id, mode),
    daemon=True,
)
```

- [ ] **Step 2: Update `_run_try_on` to support quality mode**

Add `mode: str = "fast"` to `_run_try_on` signature.

For `mode == "quality"`, add the import and pipeline call:

```python
def _run_try_on(
    prod: Product,
    var: Variation,
    jid: str,
    photo_path: Path,
    user_id: str = "",
    mode: str = "fast",
) -> None:
    try:
        from PIL import Image as PILImage

        job_manager.update_progress(jid, 0.1, "Loading images...")

        model_img = PILImage.open(photo_path).convert("RGB")
        shoe_path = results_dir / prod.id / var.id / var.results[0].filename
        shoe_img = PILImage.open(shoe_path).convert("RGB")

        if mode == "quality":
            # Agentic loop with VLM judge (max 3 iterations)
            import sys as _sys
            _project_root = Path(__file__).resolve().parent.parent.parent.parent
            _workflow_dir = _project_root / "workflows" / "shoe_tryon_gemini"
            _shared_dir = _project_root / "workflows" / "shared"
            _scripts_dir = _project_root / "workflows" / "shoe_tryon_loop" / "scripts"
            for p in [str(_workflow_dir), str(_shared_dir), str(_scripts_dir)]:
                if p not in _sys.path:
                    _sys.path.insert(0, p)

            from pipeline import build_pipeline as build_tryon_pipeline
            from judge import VLMSession, extract_features
            from casadei import ImageMedia, LoggedPipeline

            job_manager.update_progress(jid, 0.15, "Extracting shoe features...")
            vlm_session = VLMSession("gemini_flash_lite")
            shoe_media = ImageMedia(image=shoe_img)
            features = extract_features(vlm_session, shoe_media)

            job_manager.update_progress(jid, 0.2, "Running quality try-on loop...")
            pipeline, edit_agent = build_tryon_pipeline(
                max_iterations=3,
                vlm_session=vlm_session,
                features=features,
                tolerance="moderate",
            )
            logged = LoggedPipeline(pipeline)
            person_media = ImageMedia(image=model_img)
            context = {
                "person": person_media,
                "shoe": shoe_media,
                "image": person_media,
            }

            try:
                result, exec_log = logged.run(context)
            finally:
                vlm_session.unload()

            final_img = result.get("image")
            if final_img is not None and isinstance(final_img, ImageMedia):
                import io as _io
                buf = _io.BytesIO()
                final_img.image.save(buf, format="PNG")
                img_bytes = buf.getvalue()
            else:
                job_manager.fail(jid, "Quality try-on produced no output")
                return

            # Log costs
            token_records = vlm_session.token_usage_log + edit_agent.token_usage_log
            for usage_entry in token_records:
                cost = calculate_cost(usage_entry.get("model", ""), usage_entry)
                _log_cost(user_id=user_id, operation="try_on_quality",
                          model=usage_entry.get("model", ""),
                          product_id=prod.id, variation_id=var.id,
                          input_tokens=usage_entry.get("input_tokens", 0),
                          output_tokens=usage_entry.get("output_tokens", 0),
                          thinking_tokens=usage_entry.get("thinking_tokens", 0),
                          cost_usd=cost)

        else:
            # Fast mode: single Gemini edit call (existing behavior)
            job_manager.update_progress(jid, 0.2, "Generating try-on with Gemini...")

            desc = f"{var.material} {var.color}".strip() or "the shoe"
            prompt = (
                f"Replace the shoes the model is wearing with {desc} shown in the "
                f"second image. Keep the model's pose, outfit, and background exactly "
                f"the same. Only change the shoes. The result should look like a natural "
                f"fashion photograph. Photorealistic, high quality."
            )

            img_bytes, usage = _gemini_edit([model_img, shoe_img], prompt)

            cost = calculate_cost("gemini-3.1-flash-image-preview", usage)
            _log_cost(user_id=user_id, operation="try_on", model="gemini-3.1-flash-image-preview",
                      product_id=prod.id, variation_id=var.id,
                      input_tokens=usage.get("input_tokens", 0),
                      output_tokens=usage.get("output_tokens", 0),
                      thinking_tokens=usage.get("thinking_tokens", 0), cost_usd=cost)

        job_manager.update_progress(jid, 0.9, "Saving result...")
        _save_gemini_result(prod, var, img_bytes, "try-on", f"gemini_tryon_{mode}")

        job_manager.update_progress(jid, 1.0, "Done!")
        job_manager.complete(jid)

    except Exception as e:
        import traceback
        traceback.print_exc()
        job_manager.fail(jid, str(e))
```

- [ ] **Step 3: Verify the endpoint accepts the new param**

Run: `cd /home/innovina/Documents/casadei && python -c "from casadei.api.app import create_app; app = create_app(); print('app OK')"`

Expected: `app OK`

- [ ] **Step 4: Commit**

```bash
git add src/casadei/api/app.py
git commit -m "feat: add mode=fast|quality to try-on endpoint (default fast, quality uses 3-iter loop)"
```

---

## Chunk 4: Final Verification

### Task 9: End-to-end verification

- [ ] **Step 1: Verify all workflow modules import cleanly**

```bash
cd /home/innovina/Documents/casadei
python -c "
import sys
sys.path.insert(0, 'workflows/shared')
sys.path.insert(0, 'workflows/sketch_to_shoe/scripts')
sys.path.insert(0, 'workflows/shoe_tryon_loop/scripts')

# Test shared
from image_utils import load_camera_presets, get_camera_preset, build_sketch_grid, pad_to_ratio, find_ratio, foot_framing, get_canonical_angles
presets = load_camera_presets()
assert len(presets['presets']) == 9
assert get_camera_preset('3/4', 'pair')['camera_desc'].startswith('Low')
print('shared: OK')

# Test sketch_to_shoe_gemini_direct
sys.path.insert(0, 'workflows/sketch_to_shoe_gemini_direct')
from pipeline import build_pipeline as s2s_build
print('sketch_to_shoe_gemini_direct: OK')

# Test shoe_tryon_gemini
sys.path.insert(0, 'workflows/shoe_tryon_gemini')
from pipeline import build_pipeline as tryon_build
print('shoe_tryon_gemini: OK')

# Test shoe_angles
sys.path.insert(0, 'workflows/shoe_angles')
from pipeline import generate_angle, generate_angle_with_judge
print('shoe_angles: OK')

print('All workflow modules load successfully')
"
```

- [ ] **Step 2: Verify app.py loads cleanly**

```bash
cd /home/innovina/Documents/casadei && python -c "from casadei.api.app import create_app; app = create_app(); print('app OK')"
```

- [ ] **Step 3: Run existing tests**

```bash
cd /home/innovina/Documents/casadei && pytest tests/test_api_app.py -v -x 2>&1 | head -50
```

All existing tests should pass — we haven't changed behavior, only moved code.

- [ ] **Step 4: Verify test scripts are unmodified**

```bash
cd /home/innovina/Documents/casadei && git diff tests/run_sketch_to_shoe_gemini_direct.py tests/run_shoe_tryon_gemini.py tests/run_shoe_angles.py
```

Expected: no output (no changes).

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: verify all workflow modules and app integration"
```
