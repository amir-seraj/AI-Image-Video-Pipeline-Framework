# Structured Judge Design — Attribute-Based Scoring

**Date:** 2026-02-26
**Status:** Approved
**Problem:** The VLM judge in the shoe try-on loop (1) gives repetitive feedback across iterations, (2) never accepts because there are no explicit acceptance criteria.

## Design

### Overview

Replace the free-text ACCEPT/REJECT judge with a two-phase structured scoring system:

1. **Feature extraction** (once, before the loop) — VLM identifies key visual attributes from the reference shoe
2. **Per-attribute scoring** (each iteration) — VLM scores each attribute 1-5, code decides accept/reject based on configurable tolerance

### Phase 1: Feature Extraction

New function `extract_features(session, shoe_image) -> list[str]`.

- Sends reference shoe image to VLM
- Prompt asks for a JSON list of short attribute names (e.g. `["red color", "block heel", "platform sole", "ankle strap"]`)
- Fallback on parse failure: `["color", "shape", "heel", "material", "details"]`
- Called once before the loop starts, result passed to `make_judge()`

### Phase 2: Scoring Judge

**Prompt (assembled dynamically):**
```
You are a quality inspector for a virtual shoe try-on system.

IMAGE 1 is the REFERENCE shoe — the original product photo.
IMAGE 2 is the TRY-ON RESULT — a person who should be wearing
the exact same shoe from IMAGE 1 on their feet.

Your job: examine the shoes on the person's feet in IMAGE 2 and
rate how closely they match the reference shoe in IMAGE 1.

This is attempt {iteration} of {max_iterations}.
{Previous feedback: "..."}
{Stale nudge if applicable}

Score each attribute from 1 to 5:
  1 = completely different from the reference
  2 = vaguely similar but clearly wrong
  3 = recognizably the same shoe type but noticeable differences
  4 = close match with only minor differences
  5 = near-identical to the reference

Attributes to score: {comma-separated features}

Reply in this exact format:
SCORES: attr1=N, attr2=N, ...
REPAIR: <describe what is wrong in the result, then tell the model
to look at the reference shoe image and match that part — do NOT
give direct orders like "make it X", instead point out the mismatch
and direct the model to use the reference shoe image to fix it>
```

**Accept/reject logic (code-side):**

Two conditions must BOTH be true to accept:
1. **Average threshold:** `avg(scores) >= avg_threshold`
2. **Minimum per-attribute floor:** every individual score `>= min_floor`

```
tolerance configs = {
    "generous":  {"avg_threshold": 2.5, "min_floor": 1.5},
    "moderate":  {"avg_threshold": 3.5, "min_floor": 2.5},
    "strict":    {"avg_threshold": 4.5, "min_floor": 3.5},
}
avg = sum(scores.values()) / len(scores)
lowest_val = min(scores.values())
accepted = avg >= avg_threshold and lowest_val >= min_floor
```

This prevents e.g. scores `[1, 1, 5]` from passing on generous (avg 2.3 < 2.5, and min 1 < 1.5).

Lowest-scoring attribute is computed by code, not requested from VLM.

**Parse failure handling:**
1. Retry up to 3 times with error feedback: "Your response could not be parsed. Error: {error}. Please reply in the exact format: SCORES: attr1=N, ... REPAIR: ..."
2. After 3 failed retries, fall back to REJECT with raw text as feedback

### Phase 3: Stale Feedback Guardrail

Track the lowest-scoring attribute across iterations via closure state. If the same attribute is lowest for 2+ consecutive iterations, append to the judge prompt:

> "The attribute '{attr}' has been the weakest for {N} consecutive attempts. In your REPAIR instruction, clearly describe what is wrong with '{attr}' in the generated result, then tell the model to look at the reference shoe image and match that specific part. Do NOT give direct orders like 'make it X'. Instead, point out the mismatch and direct the model to use the reference shoe image as the source of truth."

### Iteration Context

`LoopStep.execute()` sets `working["loop_iteration"]` and `working["loop_max_iterations"]` before calling the judge. The judge reads these from the context dict.

Previous feedback is tracked in the judge closure (not from context) to keep it self-contained.

### CLI: `--tolerance` Flag

New argument in `run_shoe_tryon_loop.py`:
- `--tolerance generous` (default) — avg >= 2.5, each attribute >= 1.5
- `--tolerance moderate` — avg >= 3.5, each attribute >= 2.5
- `--tolerance strict` — avg >= 4.5, each attribute >= 3.5

Passed through `build_pipeline()` to `make_judge()`.

## Files Changed

| File | Change |
|------|--------|
| `workflows/shoe_tryon_loop/scripts/judge.py` | Add `extract_features()`, rewrite `make_judge()` with scoring prompt, parse logic, stale guardrail, tolerance |
| `src/casadei/loop.py` | Set `loop_iteration` and `loop_max_iterations` in working context |
| `tests/run_shoe_tryon_loop.py` | Add `--tolerance` arg, call `extract_features()` before pipeline, pass to `make_judge()` |

## Streaming

VLM responses continue to stream to terminal via `_stream_vlm()`. The scoring response is collected and parsed after streaming completes.
