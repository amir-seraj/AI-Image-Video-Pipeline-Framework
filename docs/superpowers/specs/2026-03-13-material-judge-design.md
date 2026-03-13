# Material Judge Design

Add a material compliance judge to the sketch-to-shoe Gemini pipeline. The judge verifies that generated shoes use the correct materials/colors. Operates in two modes (text and image) matching the pipeline's two material input modes.

## Overview

The pipeline currently judges only camera angle + shoe count. Materials rely on the generation prompt alone. This adds a third parallel judge that verifies material compliance, and upgrades the protection logic so the "don't change materials" safeguard is backed by actual material verification rather than assuming materials are fine when shoe count passes.

## Factory Function

**Location:** `workflows/sketch_to_shoe/scripts/judge.py`

```python
def make_material_judge(
    session: VLMSession,
    material_spec: str,
    grid_image: ImageMedia | None = None,
    candidate_key: str = "image",
    tolerance: str = "generous",
) -> JudgeCallable
```

**Parameters:**
- `material_spec` — text mode: the material description string (e.g. "red leather"); image mode: the `materials_instructions` text from `build_materials_prompt()` (includes names, placements, notes)
- `grid_image` — `None` for text mode; the grid `ImageMedia` for image mode. Presence auto-detects the mode.
- `tolerance` — reuses existing `TOLERANCE_CONFIGS` ("generous", "moderate", "strict")

**Returns:** A `JudgeCallable` that returns `(accepted: bool, repair: str)`.

## Mode Behavior

### Text Mode (`grid_image=None`)

**Input to VLM:** candidate image + text prompt.

**What it checks:** Does the shoe's visible material/color/finish match the text description?

**Prompt approach:**
> You are a material inspector for a luxury shoe studio. The shoe should be made of: {material_spec}.
>
> Examine the generated shoe and score how well the visible material, color, and finish match the specification.
>
> STEP 1 — OBSERVE: Describe the material, color, and finish you see on the shoe.
> STEP 2 — SCORE: Rate material compliance 1-5 (1=completely wrong, 5=precise match).
> STEP 3 — REPAIR: For score 3 or below, state the flaw and give a concrete material/color fix instruction. For 4-5, write "none".

**Scored attributes:** Single attribute `material` (1-5 scale).

### Image Mode — Single Material (`grid_image` provided, no bullet points in `material_spec`)

**Input to VLM:** candidate image + grid image + text prompt.

**What it checks:** Does the shoe use the material/color shown in the reference swatch?

**Prompt approach:**
> You are a material inspector for a luxury shoe studio. You are given two images:
> - SHOE IMAGE: a generated photorealistic shoe product photo.
> - MATERIAL REFERENCE: a labeled material/color swatch that should be applied to the shoe.
>
> The generator was instructed: {material_spec}
>
> Examine the SHOE IMAGE and verify that the material/color from the MATERIAL REFERENCE has been correctly applied to the shoe.
>
> STEP 1 — OBSERVE: Describe the material/color in the MATERIAL REFERENCE. Then describe what you see on the shoe in the SHOE IMAGE.
> STEP 2 — SCORE: Rate how well the SHOE IMAGE matches the MATERIAL REFERENCE (1-5).
> STEP 3 — REPAIR: For score 3 or below, state the discrepancy and give a concrete fix. For 4-5, write "none".

**Scored attributes:** Single attribute `material_match` (1-5 scale).

### Image Mode — Multiple Materials (`grid_image` provided, bullet points detected in `material_spec`)

Same as single-material image mode, but with placement verification emphasis added to the prompt:

> - MATERIAL REFERENCE: labeled material/color swatches with names. Each should be applied to a specific part of the shoe as described below.
>
> ...
>
> STEP 1 — OBSERVE: For each material listed below, describe what the MATERIAL REFERENCE shows for that swatch, then describe what you see on that part of the shoe in the SHOE IMAGE.
> STEP 2 — SCORE: Rate each material on two aspects: (a) material_match — does the material/color match the swatch? (b) placement — is it applied to the correct part of the shoe? Score each 1-5.
> STEP 3 — REPAIR: For any score 3 or below, state which material on which part is wrong and give a concrete fix. For all 4-5, write "none".

**Scored attributes:** Per-material `material_match` and `placement` scores. Acceptance requires all scores to meet tolerance thresholds.

**Multi-material detection:** Check if `material_spec` contains `"\n- "` (bullet point lines from `build_materials_prompt`). This is reliable because single-material mode never produces bullet points.

## Structured Output Schema

### Text Mode & Single-Image Mode

```python
class _MaterialJudgeResult(BaseModel):
    observation: str  # what the judge sees
    score: int        # 1-5
    repair: str       # fix instruction or "none"
```

### Multi-Material Image Mode

```python
class _MultiMaterialJudgeResult(BaseModel):
    observations: dict[str, str]  # per-material observation
    scores: dict[str, int]        # per-material scores (material_match and placement per entry)
    repair: str                   # combined fix instructions or "none"
```

## Judge Infrastructure

Follows the same patterns as existing judges in `judge.py`:

- **Hash-based dedup:** MD5 of candidate image; auto-accepts if unchanged from previous iteration (no token spend)
- **Tolerance thresholds:** Uses `TOLERANCE_CONFIGS` — `avg_threshold` and `min_floor` from the chosen tolerance level
- **Context metadata:** Writes `_judge_metadata_material` to context with scores, avg, lowest
- **VLM session:** Uses `session.acquire()` / `session.release()` with `_call_vlm_structured` for JSON schema output
- **Retry logic:** Inherits from `_call_vlm_structured` (transient error retry with backoff)

## Pipeline Integration

### `_combined_judge` Changes in `pipeline.py`

The material judge becomes the third parallel judge:

```python
# In build_pipeline, create the material judge:
session_material = VLMSession("gemini_flash")

material_judge = make_material_judge(
    session=session_material,
    material_spec=materials_instructions if use_materials_mode else spec.get("material", "black patent leather"),
    grid_image=ImageMedia(image=grid_image) if grid_image is not None else None,
    candidate_key="image",
    tolerance="generous",
)
```

In `_combined_judge`:
- Run all three judges in parallel via `ThreadPoolExecutor(max_workers=3)`
- **Protection logic change:** Replace `if count_accepted:` with `if material_accepted:` for the "CRITICAL: Keep ALL materials..." safeguard. This is more precise — we now know materials are actually correct rather than inferring from shoe count.

```python
def _combined_judge(context):
    # ... (existing force-load for threads)
    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_cam = pool.submit(camera_judge, ctx_cam)
        fut_count = pool.submit(count_judge, ctx_count)
        fut_material = pool.submit(material_judge, ctx_material)
        cam_accepted, cam_fb = fut_cam.result()
        count_accepted, count_fb = fut_count.result()
        mat_accepted, mat_fb = fut_material.result()

    accepted = cam_accepted and count_accepted and mat_accepted
    parts = []
    if not count_accepted and count_fb and count_fb != "none":
        parts.append(f"Shoe count issue: {count_fb}")
    if not mat_accepted and mat_fb and mat_fb != "none":
        parts.append(f"Material issue: {mat_fb}")
    if not cam_accepted and cam_fb and cam_fb != "none":
        if mat_accepted:
            parts.append(
                "CRITICAL: Keep ALL materials, colors, textures, and design elements "
                "IDENTICAL to the current image — do NOT change any design aspect. "
                "Only correct the camera angle as described below."
            )
        parts.append(f"Camera angle issue: {cam_fb}")
    return accepted, "\n".join(parts) if parts else "none"
```

### Return Type

`build_pipeline` return type and the VLM sessions list are updated to include `session_material`:

```python
return (pipeline, gemini_agent,
        [session_camera, session_count, session_material],
        grid_image)
```

## What Does NOT Change

- `judge.py` existing judges (`make_spec_judge`, `make_sketch_judge`, `make_shoe_count_judge`, etc.) — untouched
- `build_pipeline` return type signature — already `tuple[Pipeline, Agent, list[VLMSession], PILImage.Image | None]`
- `app.py` — no changes needed; it already passes the sessions list through
- `image_utils.py` — no changes
- `prompt.yaml` / `prompt_materials.yaml` — no changes
- The generation step, input_map, template_kwargs — all untouched

## VLM Model Choice

`session_material = VLMSession("gemini_flash")` — same model as the camera judge. It needs visual reasoning capability to compare materials/colors accurately.
