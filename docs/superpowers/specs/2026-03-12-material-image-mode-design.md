# Material-Image Mode for Sketch-to-Shoe

## Summary

Add a material-image mode to the sketch-to-shoe generation pipeline. Instead of describing materials with text (e.g., "red patent leather"), users upload actual material/color images, each mapped to a part of the shoe. The system assembles a labeled grid image of all materials, builds a prompt referencing each by name, and passes both the sketch and the material grid to the Gemini model.

Auto-detected: if `spec["materials"]` is present, use material-image mode; otherwise use existing text mode. No new endpoints for generation — the same `create_variation` flow handles both.

## Data Model

### Material Entry

Each material in the list:

| Field       | Type           | Required | Default             | Description |
|-------------|----------------|----------|---------------------|-------------|
| `name`      | `str \| None`  | No       | "Material N" / "Color N" | Label shown on grid tile and referenced in prompt |
| `image`     | `PIL.Image`    | Yes      | —                   | Material texture photo or solid color swatch |
| `placement` | `str \| None`  | If >1 material | None          | Shoe part (e.g., "toe", "heel strap", "upper") |
| `note`      | `str \| None`  | No       | None                | Free text about this material (any context) |
| `is_color`  | `bool`         | Yes      | `False`             | `True` = solid color from color picker; `False` = real material image |

**Constraints:**
- `placement` is required when there are 2+ materials. Optional (can be None) when exactly 1 material.
- `name` defaults to "Material 1", "Material 2", ... for `is_color=False` and "Color 1", "Color 2", ... for `is_color=True`. Numbering is counted per type independently (e.g., `[material, color, material]` → "Material 1", "Color 1", "Material 2").

### Spec Dict Extension

```python
spec = {
    "material": "...",          # text mode (ignored when materials present)
    "camera_angle": "3/4",
    "extra": {"note": "..."},   # overall request note (existing)
    "ref_images": {...},        # existing single ref images
    "materials": [              # NEW — if present and non-empty, triggers material-image mode
        {"name": "Suede A", "image": <PIL>, "placement": "toe", "note": "matte finish", "is_color": False},
        {"name": None, "image": <PIL>, "placement": "heel", "note": None, "is_color": True},
    ],
}
```

## Image Preprocessing Interaction

The Gemini provider (`GeminiFlashImageEdit`) scales all input images to a max side of 1024px via `_pad_to_ratio()`. The aspect ratio is determined by the first image. This has two implications for the material grid:

**Aspect ratio**: The first image in the agent's `MediaBundle` is always the sketch (see Pipeline Changes below). Since `build_sketch_grid` produces square images, the aspect ratio will be 1:1. The materials grid will be padded to match. **Therefore, the grid builder must produce a square output** — pad to square with white fill after assembling the grid. This avoids wasting pixel budget on padding.

**Label legibility**: After the provider scales the grid to fit within 1024px, tile labels must remain readable. With 500px tiles:
- 1-4 materials (up to 2×2 = 1000×1060) → fits ~1024px with minimal scaling. Labels remain clear.
- 5-9 materials (up to 3×3 = 1500×1590) → scales to ~645px per tile side. ~20px label text, still readable.
- 10-16 materials (up to 4×4 = 2000×2120) → scales to ~485px per tile side. ~15px label, borderline.

**Mitigation**: Use bold, large text (minimum 28px before scaling) for labels. The 2048px resolution cap combined with the 1024px scaling pipeline means practical legibility up to ~16 materials, which is more than sufficient for shoe design. The `build_material_grid` function uses a label font size proportional to tile size (at least 28px at 500px tiles).

## Material Grid Builder

New function `build_material_grid(materials)` in `workflows/shared/image_utils.py`.

### Tile Layout

Each tile consists of:
- **Label strip**: ~30px tall, white background, black bold text, material name centered (minimum 28px font)
- **Image box**: 500×500px, material image resized to fit (aspect-preserved), white padding

Total tile size: 500px wide × ~530px tall.

### Grid Layout

- `cols = ceil(sqrt(n))`
- `rows = ceil(n / cols)`
- Empty tiles (when n < cols × rows) filled white
- Examples: 1→1×1, 2→2×1, 3→2×2, 4→2×2, 5→3×2, 6→3×2, 9→3×3
- **Final grid is padded to a square** (white fill) to match the sketch image's 1:1 aspect ratio

### Resolution Limit

- If either dimension exceeds 2048px, downscale the final grid preserving aspect ratio
- Never upscale small grids
- With 500px tiles, 2048px cap allows up to 4×4 = 16 materials before downscaling

### Returns

`(grid_image: PIL.Image, resolved_names: list[str])` — the square grid image and the list of final names assigned to each material (so the prompt builder knows what labels are on the image).

## Prompt Templates

### File: `workflows/sketch_to_shoe_gemini/prompt_materials.yaml`

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

The shared preamble (sketch faithfulness, camera angle, staging, background) is identical to the text-mode prompt. Only the materials section differs — `$material` is replaced by `$materials_instructions`.

### Prompt Builder: `build_materials_prompt(materials, resolved_names)`

In `workflows/sketch_to_shoe_gemini/pipeline.py`. Generates the `$materials_instructions` text.

**Single material, no placement:**
```
Apply the material shown in the reference image to the shoe.
```
or if `is_color=True`:
```
Apply the color shown in the reference image to the shoe.
```
Plus per-material note and overall note if provided.

**Single material with placement:**
```
Apply the material shown in the reference image to the toe of the shoe.
```
Plus per-material note if provided.

**Multiple materials:**
```
The reference image contains labeled material/color swatches. Apply each as follows:
- Apply the material shown in "Suede A" to the toe. matte finish
- Apply the color shown in "Color 1" to the heel strap.
- Apply the material shown in "Material 2" to the upper. glossy patent leather
```
Plus overall note if provided.

Wording uses "material" vs "color" based on `is_color` per entry.

## Pipeline Changes

In `workflows/sketch_to_shoe_gemini/pipeline.py`, `build_pipeline`:

1. Check `spec.get("materials")` — if present and non-empty, enter material-image mode
2. Load `prompt_materials.yaml` instead of `prompt.yaml`
3. Call `build_material_grid(materials)` to produce the grid image + resolved names
4. Call `build_materials_prompt(materials, resolved_names)` to generate `$materials_instructions`
5. Add grid image to pipeline context as `"materials_grid"`
6. Build `input_map` with sketch first: `{"image": "sketch", "materials_grid": "materials_grid"}` — ordering ensures the sketch (square) determines the aspect ratio
7. When materials mode is active, `ref_images` (`material_ref`, `color_ref`) are **excluded** from the `input_map` and context to avoid sending unlabeled extra images to the model
8. Everything else unchanged: same agent model, same judges, same loop, same `save_results`

The agent receives: sketch image (first) + materials grid image (second) + prompt with material instructions.

## API Changes

### Material Image Upload

New endpoint:

`POST /api/products/{product_id}/variations/{variation_id}/ref-material`

Accepts:
- `file`: image upload (material texture or color swatch)
- `index`: int (ordering position)
- `name`: str (optional)
- `placement`: str (optional)
- `note`: str (optional)
- `is_color`: bool (default false)

Storage:
- Image saved as `ref_mat_{index}.png` in `data/results/{product_id}/{variation_id}/`
- Re-uploading with the same `index` overwrites the previous image and metadata entry
- Metadata written to `materials_meta.json` in the same directory

### Delete Material

`DELETE /api/products/{product_id}/variations/{variation_id}/ref-material/{index}`

Removes `ref_mat_{index}.png` and the corresponding entry from `materials_meta.json`.

### `materials_meta.json` Schema

```json
[
  {"index": 0, "name": "Suede A", "placement": "toe", "note": "matte finish", "is_color": false},
  {"index": 1, "name": null, "placement": "heel", "note": null, "is_color": true}
]
```

Array of objects, ordered by `index`. Each object mirrors the Material Entry fields (minus `image`, which is stored as a separate file).

### Loading in `_run_variation_gemini`

- Check for `materials_meta.json` in the variation's results directory
- If found, load each `ref_mat_{index}.png` + metadata
- Build `spec["materials"]` list from the loaded data
- Pass to `build_pipeline` — auto-detects material-image mode

### Existing Behavior

- `material_ref` / `color_ref` single-image uploads continue to work as before for text mode
- When materials mode is active (materials_meta.json exists with entries), `ref_images` are excluded from the pipeline input (materials takes precedence)

## Files Changed

| File | Change |
|------|--------|
| `workflows/shared/image_utils.py` | Add `build_material_grid()` |
| `workflows/sketch_to_shoe_gemini/prompt_materials.yaml` | New file — material-image prompt template |
| `workflows/sketch_to_shoe_gemini/pipeline.py` | Add `build_materials_prompt()`, extend `build_pipeline` to detect and handle materials mode |
| `src/casadei/api/app.py` | Add `ref-material` upload + delete endpoints, extend `_run_variation_gemini` to load materials |

## Constraints

- Existing text mode unchanged — no regressions
- Test scripts in `tests/` not modified
- Same judges, loop, and save_results for both modes
- Grid builder is a general utility in `shared/` (reusable)
- Placement values are free text — no validation beyond requiring non-empty when >1 material
