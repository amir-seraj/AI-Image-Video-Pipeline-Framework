# Sketch-to-Shoe Agentic Loop — Design

**Date:** 2026-02-26
**Status:** Approved

## Overview

A new workflow that takes one or more shoe design sketches plus a set of design specifications (material, color, camera angle, and open-ended extras) and produces a professional studio-quality product photograph of the shoe. Uses the same `LoopStep` agentic pattern as `shoe_tryon_loop`: FireRed generates, two VLM judges evaluate, feedback is injected into the next iteration.

---

## Section 1: Inputs & Sketch Grid

### CLI Arguments

```
--sketches path1.png [path2.png ...]   # one or more sketch images (required)
--material "leather"                    # fixed spec attribute
--color "black"                         # fixed spec attribute
--camera-angle "3/4 view"              # fixed spec; choices: "3/4 view", "side view",
                                        #   "front view", "top view", or any custom string
--spec KEY=VALUE [KEY=VALUE ...]       # open-ended extras: style="elegant" note="chunky sole"
--max-iter 5                            # loop cap (default: 5)
--steps 30                              # FireRed inference steps (default: 30)
--tolerance strict                      # judge tolerance: generous / moderate / strict
--keep-both                             # keep all models loaded (needs ~74GB+ VRAM)
--scale 1.0                             # resize factor for sketch images
```

### Sketch Grid Assembly

1. Load all sketch images, convert to RGB.
2. Arrange into a rectangular grid (row-major, minimum rows to keep aspect landscape-or-square):
   - 1 sketch → 1×1 grid
   - 2 sketches → 1×2 grid
   - 3–4 sketches → 2×2 grid
   - 5–6 sketches → 2×3 grid
   - etc.
3. Apply fixed 20px white padding between cells and around the border.
4. If the resulting grid image is not square (width ≠ height), center it on a white square canvas.

The final sketch image passed to FireRed is always square.

---

## Section 2: Generation (FireRed Agent)

**Model:** `firered_image_edit`
**Image 1:** sketch grid (constant across all iterations — the design reference)
**Image 2:** previous iteration's generated photo (first iteration: same as Image 1)

### Prompt Template

```
The image shows a shoe design sketch. Convert it into a professional,
photorealistic product photograph of the final shoe.

Design specifications:
- Material: $material
- Color: $color
- Camera angle: $camera_angle
$extra_specs

The result must be a studio-quality photograph: clean white background,
professional product lighting, sharp focus, no shadows on background,
shoe centered and fully visible.
$feedback
```

- `$extra_specs` expands open-ended key-value pairs, one per line as `- Key: value`
- `$feedback` is empty on iteration 1; filled with structured dual-judge feedback on subsequent iterations

### Negative Prompt

```
blurry, distorted, low quality, sketch, drawing, illustration, flat, cartoon,
dark background, cluttered background, bad lighting, overexposed, underexposed
```

---

## Section 3: Dual-Judge Loop

Two VLM judges run sequentially each iteration using `qwen3_vl_8b`. Both must accept for the iteration to pass.

### Judge 1 — Sketch Fidelity

Compares the generated photo against the original sketch to verify design faithfulness.

- **IMAGE 1:** sketch grid (reference)
- **IMAGE 2:** generated photo (candidate)
- **Attributes:** extracted once from the sketch via VLM before the loop starts (same `extract_features` pattern as `shoe_tryon_loop`). Typical attributes: `shape`, `proportions`, `toe_shape`, `heel_style`, `sole_design`, `silhouette`.
- **Score format:** `SCORES: shape=N, proportions=N, ...` + `REPAIR: <feedback>`

### Judge 2 — Spec Compliance

Checks if the generated photo matches the user's design specification and studio quality requirements.

- **IMAGE:** generated photo only
- **TEXT:** full spec (material, color, camera angle, extra key-value pairs)
- **Attributes:** one per spec key, plus fixed photo-quality attributes: `white_background`, `lighting`, `sharpness`
- **Score format:** `SCORES: material=N, color=N, camera_angle=N, white_background=N, ...` + `REPAIR: <feedback>`

### Accept / Reject Logic

Both judges use the same tolerance thresholds (from `TOLERANCE_CONFIGS`):

```
generous:  avg >= 2.5, min >= 1.5
moderate:  avg >= 3.5, min >= 2.5
strict:    avg >= 4.5, min >= 3.5
```

Accepted = Judge 1 passes **AND** Judge 2 passes.
Both judges always run every iteration (no early-exit skip), so both feedbacks are always available.

### Feedback Injection

Combined feedback injected into `$feedback` in the generation prompt:

```
[Sketch feedback]: <Judge 1 REPAIR text>
[Spec feedback]: <Judge 2 REPAIR text>
```

Stale-feedback guardrail (same as `shoe_tryon_loop`): if the lowest-scoring attribute repeats across N iterations, a stale nudge is added to that judge's prompt.

### Best-of-N Selection

If max iterations is reached without acceptance, the VLM (Judge 1's session) picks the best candidate from all iterations by viewing a labeled side-by-side grid. Falls back to the last candidate if parse fails.

---

## Section 4: File Structure

```
workflows/sketch_to_shoe/
└── scripts/
    └── judge.py          # SketchJudge + SpecJudge classes + VLMSession + best_fn

tests/
└── run_sketch_to_shoe_loop.py   # main entry point
```

### Output Layout

`tests/output/sketch_to_shoe_loop/{N}iter_{S}steps_{timestamp}/`

| File | Contents |
|------|----------|
| `input_sketch_grid.png` | Assembled sketch grid fed to FireRed |
| `iter_00_candidate.png` … | Per-iteration generated photos |
| `final_result.png` | Accepted or best-selected result |
| `results.json` | Both judges' scores, feedback, timings per iteration |
| `summary.txt` | Human-readable report |

---

## Key Differences from `shoe_tryon_loop`

| | shoe_tryon_loop | sketch_to_shoe_loop |
|---|---|---|
| Generation model | FireRed | FireRed |
| Image 1 | Reference shoe photo | Sketch grid |
| Image 2 | Person photo (seeded with last gen) | Last generated photo (seeded with sketch) |
| Judge count | 1 | 2 (sketch fidelity + spec compliance) |
| Judge reference | Reference shoe photo | Sketch grid / spec text |
| Spec parameters | None | material, color, camera_angle + open-ended |
| Sketch grid logic | N/A | Multi-sketch concat + square-pad |
