# Promote Test Pipelines to Workflows

## Summary

Move reusable pipeline logic from three test scripts into proper `workflows/` modules with data-driven configuration (YAML for prompts/presets, Python modules for utilities). Test scripts become thin CLI wrappers. `app.py` imports from workflows instead of tests.

**Test scripts (unchanged, stay in `tests/`):**
- `tests/run_sketch_to_shoe_gemini_direct.py`
- `tests/run_shoe_tryon_gemini.py`
- `tests/run_shoe_angles.py`

## New Directory Structure

```
workflows/
  shared/
    camera_presets.yaml      # All camera presets + aliases + judge_notes
    image_utils.py           # pad_to_ratio, find_ratio, build_sketch_grid, foot_framing

  sketch_to_shoe_gemini_direct/
    pipeline.py              # build_pipeline, save_results (~100-150 lines)
    prompt.yaml              # PROMPT_TEMPLATE + default params

  shoe_tryon_gemini/
    pipeline.py              # build_pipeline, save_results
    prompt.yaml              # PROMPT_TEMPLATE + default params

  shoe_angles/
    pipeline.py              # generate_angle, generate_angle_with_judge
    prompt.yaml              # PROMPT_TEMPLATE
```

## What Goes Where

### `shared/camera_presets.yaml`
- Full `CAMERA_PRESETS` dict (all angles x all foot variants)
- Aliases (`"3/4 view"` -> `"3/4"`, `"hero"` -> `"hero-front-right"`, etc.)
- `judge_notes` — the `_CAMERA_JUDGE_NOTES` evaluation rubric
- `CANONICAL_ANGLES` list
- `PAIR_ANGLES` / `SINGLE_ANGLES` sets
- Single source of truth — currently duplicated in `run_sketch_to_shoe_gemini_direct.py` and `run_shoe_angles.py`

### `shared/image_utils.py`
- `pad_to_ratio(img, ratio)` — pad image to target aspect ratio (from shoe_angles)
- `find_ratio(w, h)` — find nearest supported aspect ratio (from shoe_angles)
- `build_sketch_grid(images, spacing)` — assemble multiple sketches into a grid (from sketch_to_shoe)
- `foot_framing(foot)` — return foot-specific prompt fragment (from both scripts)
- `load_camera_presets()` — load and return the YAML data
- `get_camera_preset(angle, foot)` — resolve preset by angle name + foot

### Per-workflow `prompt.yaml`
Each contains:
- `prompt_template` — the `$variable` template string
- `default_params` — default generation parameters (temperature, etc.)
- `negative_prompt` — if applicable

### Per-workflow `pipeline.py`
Compact orchestration:
- `build_pipeline(...)` — wires agents, steps, judges, loop
- `save_results(...)` — output serialization
- Judge composition (combined judge closures)
- Imports everything else from `shared/`

## API Changes in `app.py`

### Try-on endpoint
`POST /api/products/{id}/variations/{id}/try-on`
- New query param: `mode=fast|quality` (default: `fast`)
- `fast` — current single-call Gemini edit (no loop, no judge)
- `quality` — uses `shoe_tryon_gemini/pipeline.py` `build_pipeline()` with `max_iterations=3`

### Angles endpoint
`POST /api/products/{id}/variations/{id}/generate-angles`
- New query param: `judged=true|false` (default: `true`)
- `true` — uses `generate_angle_with_judge` (up to 3 iterations per angle)
- `false` — current single `generate_angle` call

### Sketch-to-shoe
`_run_variation_gemini` — rewire imports from `tests/` to `workflows/sketch_to_shoe_gemini_direct/pipeline.py`. No behavior change.

### Import cleanup
All `sys.path.insert(0, tests_dir)` in app.py replaced with `sys.path.insert(0, workflows_dir / "...")`.

## Test Script Changes

Each test script becomes a thin CLI wrapper (~50-80 lines):
- `argparse` setup
- Load images
- Call `build_pipeline()` / `generate_angle_with_judge()` from workflow module
- Call `save_results()` from workflow module
- Print summary

Test scripts add workflow dirs to `sys.path` instead of (or in addition to) current paths.

## Constraints
- Test scripts must NOT be modified or removed from `tests/`
- Existing judge scripts (`workflows/sketch_to_shoe/scripts/judge.py`, `workflows/shoe_tryon_loop/scripts/judge.py`) stay where they are
- Pipeline modules add judge `scripts/` dirs to `sys.path` as needed
