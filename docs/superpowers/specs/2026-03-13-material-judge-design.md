# Material Judge Design

Add a material compliance judge to the sketch-to-shoe Gemini pipeline. The judge verifies that generated shoes use the correct materials/colors. Operates in three modes — text, single-image, and multi-material image — matching the pipeline's material input modes.

## Overview

The pipeline currently judges only camera angle + shoe count. Materials rely on the generation prompt alone. This adds a third parallel judge that verifies material compliance, and upgrades the protection logic so the "don't change materials" safeguard is backed by actual material verification rather than assuming materials are fine when shoe count passes.

## Factory Function

**Location:** `workflows/sketch_to_shoe/scripts/judge.py`

```python
def make_material_judge(
    session: VLMSession,
    material_spec: str,
    grid_image: ImageMedia | None = None,
    material_names: list[str] | None = None,
    candidate_key: str = "image",
    tolerance: str = "generous",
) -> JudgeCallable
```

**Parameters:**
- `material_spec` — text mode: the material description string (e.g. "red leather"); image mode: the `materials_instructions` text from `build_materials_prompt()` (includes names, placements, notes)
- `grid_image` — `None` for text mode; the grid `ImageMedia` for image mode. Presence auto-detects text vs image mode.
- `material_names` — `None` for text mode and single-image mode. For multi-material image mode: the list of resolved names (e.g. `["Suede A", "Color 1"]`). When provided and `len > 1`, enables placement verification. This is explicit — no string-heuristic detection.
- `tolerance` — reuses existing `TOLERANCE_CONFIGS` ("generous", "moderate", "strict")
- `material_spec` must be non-empty.

**Returns:** A `JudgeCallable` that returns `(accepted: bool, repair: str)`.

**Mode selection logic:**
- `grid_image is None` → text mode
- `grid_image` provided, `material_names is None or len <= 1` → single-material image mode
- `grid_image` provided, `len(material_names) > 1` → multi-material image mode

## Mode Behavior

### Text Mode (`grid_image=None`)

**Input to VLM:** candidate image + text prompt.

**What it checks:** Does the shoe's visible material/color/finish match the text description?

**Bundle construction:**
```python
bundle = MediaBundle(items={
    "candidate": candidate,
    "prompt": TextMedia(text=prompt_text),
})
```

**Prompt approach:**
> You are a material inspector for a luxury shoe studio. The shoe should be made of: {material_spec}.
>
> Examine the generated shoe and score how well the visible material, color, and finish match the specification.
>
> STEP 1 — OBSERVE: Describe the material, color, and finish you see on the shoe.
> STEP 2 — SCORE: Rate material compliance 1-5 (1=completely wrong material/color, 2=fundamentally different, 3=partially correct with clear discrepancy, 4=substantially matches with minor variation, 5=precise match).
> STEP 3 — REPAIR: For score 3 or below, state the flaw and give a concrete material/color fix instruction. For 4-5, write "none".

**Scored attributes:** Single attribute `material` (1-5 scale).

### Image Mode — Single Material (`grid_image` provided, `material_names` is None or len <= 1)

**Input to VLM:** candidate image + grid image + text prompt.

**What it checks:** Does the shoe use the material/color shown in the reference swatch?

**Bundle construction:**
```python
bundle = MediaBundle(items={
    "candidate": candidate,
    "material_ref": grid_image,
    "prompt": TextMedia(text=prompt_text),
})
```

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

**Scored attributes:** Single attribute `material` (1-5 scale). Uses `_MaterialJudgeResult`.

### Image Mode — Multiple Materials (`grid_image` provided, `len(material_names) > 1`)

Same prompt structure but with placement verification emphasis and per-material scoring.

**Bundle construction:** Same as single-image mode.

**Prompt approach:**
> You are a material inspector for a luxury shoe studio. You are given two images:
> - SHOE IMAGE: a generated photorealistic shoe product photo.
> - MATERIAL REFERENCE: labeled material/color swatches with names. Each should be applied to a specific part of the shoe as described below.
>
> The generator was instructed:
> {material_spec}
>
> Examine the SHOE IMAGE and verify that each material/color from the MATERIAL REFERENCE has been correctly applied to the correct part of the shoe.
>
> STEP 1 — OBSERVE: For each material listed, describe what the MATERIAL REFERENCE shows for that swatch, then describe what you see on that part of the shoe in the SHOE IMAGE.
> STEP 2 — SCORE: Rate each material on two aspects. Use these exact attribute names:
> {score_attributes}
> Score each 1-5 (1=completely wrong, 5=precise match).
> STEP 3 — REPAIR: For any attribute scored 3 or below, state which material on which part is wrong and give a concrete fix. For all 4-5, write "none".

**Score attribute names** are generated from `material_names` and injected into the prompt:
```python
# For material_names=["Suede A", "Color 1"]:
score_attributes = [
    "Suede_A_match", "Suede_A_placement",
    "Color_1_match", "Color_1_placement",
]
```
Key format: `{sanitized_name}_match` and `{sanitized_name}_placement` (spaces replaced with underscores).

**Acceptance:** All scores must meet `avg_threshold` (average) and `min_floor` (lowest individual). Uses `_SpecJudgeResult` schema (same as spec judge — `observations: dict[str, str]`, `scores: dict[str, int]`, `repair: str`).

## Structured Output Schema

### Text Mode & Single-Image Mode

```python
class _MaterialJudgeResult(BaseModel):
    observation: str = Field(
        description="Describe the material, color, and finish visible on the shoe"
    )
    score: int = Field(
        description="Material compliance score 1-5",
        ge=1, le=5,
    )
    repair: str = Field(
        description="For score 3 or below: flaw + fix instruction. For 4-5: 'none'"
    )
```

### Multi-Material Image Mode

Reuses the existing `_SpecJudgeResult` schema from `judge.py` — it already has `observations: dict[str, str]`, `scores: dict[str, int]`, and `repair: str`. The attribute names are injected into the prompt via `{score_attributes}`.

## Judge Infrastructure

Follows the same patterns as existing judges in `judge.py`:

- **Hash-based dedup:** MD5 of candidate image; auto-accepts if unchanged from previous iteration (no token spend)
- **Tolerance thresholds:** Uses `TOLERANCE_CONFIGS` — `avg_threshold` and `min_floor` from the chosen tolerance level
- **Context metadata:** Writes `_judge_metadata_material` to context with scores, avg, lowest
- **Usage tracking:** Calls `session.record_usage("Material Judge")` after the VLM call
- **VLM session:** Uses `session.acquire()` / `session.release()` with `_call_vlm_structured` for JSON schema output
- **Retry logic:** Inherits from `_call_vlm_structured` (transient error retry with backoff)
- **Parse failure handling:** Wrap `model_validate_json` in try/except. On parse failure, return `(False, "Material judge parse error — regenerate with the specified materials.")` so the loop can continue rather than crashing.

## Pipeline Integration

### Constants

Extract the default material to a shared constant in `pipeline.py`:

```python
DEFAULT_MATERIAL = "black patent leather"
```

Used in both `template_kwargs` (replacing the existing hardcoded `"black patent leather"` string) and the material judge construction.

### `_combined_judge` Changes in `pipeline.py`

The material judge becomes the third parallel judge:

```python
# In build_pipeline, create the material judge:
session_material = VLMSession("gemini_flash")

material_judge = make_material_judge(
    session=session_material,
    material_spec=materials_instructions if use_materials_mode else spec.get("material", DEFAULT_MATERIAL),
    grid_image=ImageMedia(image=grid_image) if grid_image is not None else None,
    material_names=resolved_names if use_materials_mode and len(materials_list) > 1 else None,
    candidate_key="image",
    tolerance="generous",
)
```

In `_combined_judge`:
- Run all three judges in parallel via `ThreadPoolExecutor(max_workers=3)`
- Each judge gets its own context dict (`ctx_cam`, `ctx_count`, `ctx_material`) with the candidate image copied in
- Propagate all `_`-prefixed metadata from all three contexts back to the main context
- **Protection logic:** The "CRITICAL: Keep ALL materials..." safeguard fires only when both material AND count pass but camera fails. This avoids contradictory repair instructions.

```python
def _combined_judge(context):
    image = context.get("image")
    if isinstance(image, ImageMedia):
        image.image.load()
    ctx_cam: dict = {"image": image}
    ctx_count: dict = {"image": image}
    ctx_material: dict = {"image": image}

    with ThreadPoolExecutor(max_workers=3) as pool:
        fut_cam = pool.submit(camera_judge, ctx_cam)
        fut_count = pool.submit(count_judge, ctx_count)
        fut_material = pool.submit(material_judge, ctx_material)
        cam_accepted, cam_fb = fut_cam.result()
        count_accepted, count_fb = fut_count.result()
        mat_accepted, mat_fb = fut_material.result()

    # Propagate metadata from all judges
    context.update({k: v for k, v in ctx_cam.items() if k.startswith("_")})
    context.update({k: v for k, v in ctx_count.items() if k.startswith("_")})
    context.update({k: v for k, v in ctx_material.items() if k.startswith("_")})

    # Promote camera + material metadata for best_fn
    meta = context.pop("_judge_metadata_spec", {})
    mat_meta = context.pop("_judge_metadata_material", {})
    context["_judge_metadata"] = {
        "sketch_avg": None,
        "spec_scores": meta.get("scores", {}),
        "spec_avg": meta.get("avg_score"),
        "material_scores": mat_meta.get("scores", {}),
        "material_avg": mat_meta.get("avg_score"),
    }

    accepted = cam_accepted and count_accepted and mat_accepted
    parts = []
    if not count_accepted and count_fb and count_fb != "none":
        parts.append(f"Shoe count issue: {count_fb}")
    if not mat_accepted and mat_fb and mat_fb != "none":
        parts.append(f"Material issue: {mat_fb}")
    if not cam_accepted and cam_fb and cam_fb != "none":
        if mat_accepted and count_accepted:
            parts.append(
                "CRITICAL: Keep ALL materials, colors, textures, and design elements "
                "IDENTICAL to the current image — do NOT change any design aspect. "
                "Only correct the camera angle as described below."
            )
        parts.append(f"Camera angle issue: {cam_fb}")
    return accepted, "\n".join(parts) if parts else "none"
```

### `best_fn` Scoring Update

The existing `make_best_fn` ranks candidates by `(sketch_avg + spec_avg) / 2.0`. Since `sketch_avg` is always `None` in this pipeline, only camera scores drive selection. With the new material judge, `best_fn` must also factor in `material_avg`:

```python
sketch_avg = record.metadata.get("sketch_avg") or 0.0
spec_avg = record.metadata.get("spec_avg") or 0.0
material_avg = record.metadata.get("material_avg") or 0.0
combined = (sketch_avg + spec_avg + material_avg) / 3.0
```

This is a one-line change in `make_best_fn` inside `judge.py`. When `material_avg` is absent (e.g. older pipelines not passing it), it falls back to `0.0` — same behavior as `sketch_avg` today. No breaking change.

### `save_results` Note

`save_results` in `pipeline.py` currently logs `camera_score` and `spec_avg` from iteration metadata. The new `material_scores` and `material_avg` fields will also be available in `_judge_metadata` but capturing them in `save_results` is NOT required for this task — it's a nice-to-have that can be added later.

### Return Type

`build_pipeline` return type and the VLM sessions list are updated to include `session_material`:

```python
return (pipeline, gemini_agent,
        [session_camera, session_count, session_material],
        grid_image)
```

## What Does NOT Change

- `judge.py` existing judges (`make_spec_judge`, `make_sketch_judge`, `make_shoe_count_judge`, etc.) — untouched except `make_best_fn` scoring formula (one-line addition)
- `build_pipeline` return type signature — already `tuple[Pipeline, Agent, list[VLMSession], PILImage.Image | None]`
- `app.py` — no changes needed; it already passes the sessions list through
- `image_utils.py` — no changes
- `prompt.yaml` / `prompt_materials.yaml` — no changes
- The generation step, input_map, template_kwargs — all untouched

## VLM Model Choice

`session_material = VLMSession("gemini_flash")` — same model as the camera judge. It needs visual reasoning capability to compare materials/colors accurately.
