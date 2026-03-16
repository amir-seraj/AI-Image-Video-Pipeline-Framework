# Low-Fi to High-Fi Sketch Workflow

## Summary

A single-pass Gemini-based workflow that transforms rough hand-drawn sketches into
high-fidelity professional pencil sketches. Optionally accepts a 3D volume image to
improve shape understanding. Always outputs a single shoe.

## Inputs

| Key      | Type  | Required | Description                                      |
|----------|-------|----------|--------------------------------------------------|
| `sketch` | image | yes      | Low-fidelity hand-drawn sketch                   |
| `volume` | image | no       | 3D volume/last image showing shoe shape           |

## Output

A single high-fidelity pencil sketch image — clean black-and-white rendering on
off-white paper, professional fashion sketch quality, fine line work with shading.

## Directory Structure

```
workflows/lowfi_to_hifi/
├── pipeline.py              # build_pipeline(), save_results()
├── prompt_with_volume.yaml  # prompt when both volume + sketch provided
└── prompt_sketch_only.yaml  # prompt when only sketch provided
```

## Prompt Strategy

Two prompt templates, selected dynamically based on whether a volume image is provided.

### `prompt_with_volume.yaml`

```yaml
prompt_template: |
  You are given two images: a 3D volume showing the shoe's shape and proportions,
  and a rough hand-drawn sketch showing the design. Study the volume to understand
  the depth, curves, and structure. Study the sketch to understand every design
  detail — lines, straps, openings, and structural elements. Generate a
  high-fidelity professional pencil sketch of a single shoe that combines the 3D
  understanding from the volume with the design from the sketch.

  Output style: clean black-and-white pencil rendering on off-white paper,
  professional fashion sketch quality, single shoe, fine line work with shading
  for depth.
  $extra_specs

default_params:
  temperature: 0.8

negative_prompt: ""
```

### `prompt_sketch_only.yaml`

```yaml
prompt_template: |
  You are given a rough hand-drawn sketch of a shoe design. Study every line,
  strap, opening, and structural element carefully. Generate a high-fidelity
  professional pencil sketch of a single shoe that faithfully renders the design.

  Output style: clean black-and-white pencil rendering on off-white paper,
  professional fashion sketch quality, single shoe, fine line work with shading
  for depth.
  $extra_specs

default_params:
  temperature: 0.8

negative_prompt: ""
```

## Pipeline Construction (`pipeline.py`)

### `build_pipeline(spec) -> tuple[Pipeline, Agent]`

1. Check if `spec` contains a `"volume"` key with an image
2. Load `prompt_with_volume.yaml` or `prompt_sketch_only.yaml` accordingly
3. Create a Gemini agent:
   ```python
   Agent(AgentConfig(
       name="gemini_lowfi_to_hifi",
       model="gemini_flash_image_edit",
       description="Gemini Flash image edit for lowfi-to-hifi sketch generation",
       prompt_template=prompt_config["prompt_template"],
       negative_prompt="",
       params={"temperature": temperature},
   ))
   ```
4. Build input map dynamically:
   - Sketch only: `{"image": "sketch"}`
   - With volume: `{"image": "sketch", "volume": "volume"}`
5. Create single `AgentStep` → `Pipeline` (no loop)
6. Return `(pipeline, agent)`

### `save_results(run_dir, result_context, result_image, spec, total_elapsed)`

Simple result saver:
- Save result image as `result.png`
- Save metadata JSON with timestamp, elapsed time, model, spec, mode (with/without volume)
- Print summary to stdout

## What This Workflow Does NOT Have

- No loop or judges (single-pass generation)
- No camera angle system
- No material grid
- No foot framing logic (always single shoe)
- No multi-angle generation

## Model

Uses `gemini_flash_image_edit` — same as `sketch_to_shoe_gemini`.

## Pattern Reference

Follows the same `pipeline.py` + prompt YAML pattern as `workflows/sketch_to_shoe_gemini/`.
