# Material Judge Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a material compliance judge to the sketch-to-shoe Gemini pipeline that verifies generated shoes use the correct materials/colors in three modes (text, single-image, multi-material image).

**Architecture:** A `make_material_judge` factory function in `judge.py` returns a `JudgeCallable`. It auto-selects mode based on `grid_image` and `material_names` parameters. The pipeline's `_combined_judge` runs it as a third parallel judge alongside camera and count judges.

**Tech Stack:** Python, Pydantic v2 (structured output schema), Gemini VLM (via `_call_vlm_structured`), `ThreadPoolExecutor` for parallel judge execution.

**Spec:** `docs/superpowers/specs/2026-03-13-material-judge-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `workflows/sketch_to_shoe/scripts/judge.py` | Modify (lines 183-266, 975-998) | Add `_MaterialJudgeResult` schema, `make_material_judge` factory, update `make_best_fn` scoring |
| `workflows/sketch_to_shoe_gemini/pipeline.py` | Modify (lines 34, 126, 218, 240-294) | Add `DEFAULT_MATERIAL` constant, import `make_material_judge`, update `_combined_judge` and return value |
| `tests/test_material_judge.py` | Create | Unit tests for `make_material_judge` and pipeline integration |

---

## Chunk 1: Material Judge in judge.py

### Task 1: Add `_MaterialJudgeResult` schema

**Files:**
- Modify: `workflows/sketch_to_shoe/scripts/judge.py:247` (after `_ShoeCountResult`)
- Test: `tests/test_material_judge.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_material_judge.py`:

```python
"""Tests for the material judge."""
import sys
from pathlib import Path

_project = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project / "workflows" / "shared"))
sys.path.insert(0, str(_project / "workflows" / "sketch_to_shoe" / "scripts"))

from judge import _MaterialJudgeResult


def test_material_judge_result_valid():
    result = _MaterialJudgeResult(
        observation="Black patent leather with glossy finish",
        score=4,
        repair="none",
    )
    assert result.score == 4
    assert result.repair == "none"


def test_material_judge_result_score_bounds():
    import pytest
    with pytest.raises(Exception):
        _MaterialJudgeResult(observation="test", score=0, repair="none")
    with pytest.raises(Exception):
        _MaterialJudgeResult(observation="test", score=6, repair="none")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_material_judge.py::test_material_judge_result_valid -v`
Expected: FAIL with `ImportError: cannot import name '_MaterialJudgeResult'`

- [ ] **Step 3: Write the schema**

In `workflows/sketch_to_shoe/scripts/judge.py`, after the `_ShoeCountResult` class (after line 265), add:

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

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_material_judge.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add workflows/sketch_to_shoe/scripts/judge.py tests/test_material_judge.py
git commit -m "feat: add _MaterialJudgeResult schema for material judge"
```

---

### Task 2: Add `make_material_judge` factory — text mode

**Files:**
- Modify: `workflows/sketch_to_shoe/scripts/judge.py` (after `make_shoe_count_judge`, before `make_dual_judge`)
- Test: `tests/test_material_judge.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_material_judge.py`:

```python
import unittest.mock as mock
from unittest.mock import MagicMock, patch
from PIL import Image as PILImage
from casadei.media import ImageMedia, TextMedia

from judge import make_material_judge, _MaterialJudgeResult, _SpecJudgeResult


def _make_candidate():
    """Create a dummy candidate ImageMedia."""
    return ImageMedia(image=PILImage.new("RGB", (100, 100), (0, 0, 0)))


def _mock_vlm_text_mode(raw_json):
    """Return a mock VLM session that returns the given JSON string."""
    session = MagicMock()
    model = MagicMock()
    session.acquire.return_value = model
    model.run.return_value = MagicMock(
        items={"text": TextMedia(text=raw_json)}
    )
    return session, model


def test_text_mode_accept():
    """Text mode: score >= threshold → accept."""
    result_json = _MaterialJudgeResult(
        observation="Red leather upper with matte finish",
        score=4,
        repair="none",
    ).model_dump_json()

    session, model = _mock_vlm_text_mode(result_json)
    judge = make_material_judge(
        session=session,
        material_spec="red leather",
        grid_image=None,
        tolerance="generous",  # avg_threshold=3.0, min_floor=2.0
    )
    ctx = {"image": _make_candidate()}
    accepted, feedback = judge(ctx)
    assert accepted is True
    assert feedback == "none"
    assert "_judge_metadata_material" in ctx


def test_text_mode_reject():
    """Text mode: score below threshold → reject with repair."""
    result_json = _MaterialJudgeResult(
        observation="Blue suede instead of red leather",
        score=2,
        repair="Change material from blue suede to red leather",
    ).model_dump_json()

    session, model = _mock_vlm_text_mode(result_json)
    judge = make_material_judge(
        session=session,
        material_spec="red leather",
        grid_image=None,
        tolerance="moderate",  # avg_threshold=3.5, min_floor=2.5
    )
    ctx = {"image": _make_candidate()}
    accepted, feedback = judge(ctx)
    assert accepted is False
    assert "red leather" in feedback


def test_text_mode_no_grid_image():
    """Text mode is selected when grid_image is None."""
    result_json = _MaterialJudgeResult(
        observation="Black patent leather", score=5, repair="none",
    ).model_dump_json()
    session, model = _mock_vlm_text_mode(result_json)
    judge = make_material_judge(
        session=session,
        material_spec="black patent leather",
        grid_image=None,
    )
    ctx = {"image": _make_candidate()}
    judge(ctx)
    # Bundle should NOT contain material_ref key
    call_args = model.run.call_args
    bundle = call_args[0][0]
    assert "material_ref" not in bundle.items
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_material_judge.py::test_text_mode_accept -v`
Expected: FAIL with `ImportError: cannot import name 'make_material_judge'`

- [ ] **Step 3: Write the text mode implementation**

In `workflows/sketch_to_shoe/scripts/judge.py`, after `make_shoe_count_judge` (after line 925), add the full `make_material_judge` function. For now, implement only the text mode path — image mode paths will return `(False, "not implemented")` as placeholder:

```python
# ---------------------------------------------------------------------------
# Material Compliance Judge
# ---------------------------------------------------------------------------

_MATERIAL_TEXT_PROMPT = """\
You are a material inspector for a luxury shoe studio. The shoe should be made of: {material_spec}.

Examine the generated shoe and score how well the visible material, color, and finish match the specification.

STEP 1 — OBSERVE: Describe the material, color, and finish you see on the shoe.
STEP 2 — SCORE: Rate material compliance 1-5 (1=completely wrong material/color, \
2=fundamentally different, 3=partially correct with clear discrepancy, \
4=substantially matches with minor variation, 5=precise match).
STEP 3 — REPAIR: For score 3 or below, state the flaw and give a concrete \
material/color fix instruction. For 4-5, write "none".
"""

_MATERIAL_IMAGE_SINGLE_PROMPT = """\
You are a material inspector for a luxury shoe studio. You are given two images:
- SHOE IMAGE: a generated photorealistic shoe product photo.
- MATERIAL REFERENCE: a labeled material/color swatch that should be applied to the shoe.

The generator was instructed: {material_spec}

Examine the SHOE IMAGE and verify that the material/color from the MATERIAL REFERENCE \
has been correctly applied to the shoe.

STEP 1 — OBSERVE: Describe the material/color in the MATERIAL REFERENCE. \
Then describe what you see on the shoe in the SHOE IMAGE.
STEP 2 — SCORE: Rate how well the SHOE IMAGE matches the MATERIAL REFERENCE (1-5).
STEP 3 — REPAIR: For score 3 or below, state the discrepancy and give a concrete fix. \
For 4-5, write "none".
"""

_MATERIAL_IMAGE_MULTI_PROMPT = """\
You are a material inspector for a luxury shoe studio. You are given two images:
- SHOE IMAGE: a generated photorealistic shoe product photo.
- MATERIAL REFERENCE: labeled material/color swatches with names. Each should be applied \
to a specific part of the shoe as described below.

The generator was instructed:
{material_spec}

Examine the SHOE IMAGE and verify that each material/color from the MATERIAL REFERENCE \
has been correctly applied to the correct part of the shoe.

STEP 1 — OBSERVE: For each material listed, describe what the MATERIAL REFERENCE shows \
for that swatch, then describe what you see on that part of the shoe in the SHOE IMAGE.
STEP 2 — SCORE: Rate each material on two aspects. Use these exact attribute names:
{score_attributes}
Score each 1-5 (1=completely wrong, 5=precise match).
STEP 3 — REPAIR: For any attribute scored 3 or below, state which material on which part \
is wrong and give a concrete fix. For all 4-5, write "none".
"""


def make_material_judge(
    session: VLMSession,
    material_spec: str,
    grid_image: ImageMedia | None = None,
    material_names: list[str] | None = None,
    candidate_key: str = "image",
    tolerance: str = "moderate",
) -> JudgeCallable:
    """Return a JudgeCallable that verifies material compliance.

    Mode selection:
    - grid_image is None → text mode (single score)
    - grid_image provided, material_names is None or len <= 1 → single-image mode (single score)
    - grid_image provided, len(material_names) > 1 → multi-material image mode (per-material scores)
    """
    if not material_spec:
        raise ValueError("material_spec must be non-empty")

    tol = TOLERANCE_CONFIGS.get(tolerance, TOLERANCE_CONFIGS["moderate"])
    avg_threshold = tol["avg_threshold"]
    min_floor = tol["min_floor"]

    _prev_candidate_hash: list[str | None] = [None]

    # Determine mode
    is_multi = (
        grid_image is not None
        and material_names is not None
        and len(material_names) > 1
    )
    is_image = grid_image is not None

    # Build score attributes for multi-material mode
    if is_multi:
        score_attributes = []
        for name in material_names:
            sanitized = name.replace(" ", "_")
            score_attributes.append(f"{sanitized}_match")
            score_attributes.append(f"{sanitized}_placement")
    else:
        score_attributes = []

    def judge(context: dict[str, Media]) -> tuple[bool, str]:
        candidate = context.get(candidate_key)
        if not isinstance(candidate, ImageMedia):
            return False, f"Missing candidate image (key='{candidate_key}')."

        current_hash = _image_hash(candidate)
        if _prev_candidate_hash[0] is not None and current_hash == _prev_candidate_hash[0]:
            print("  [Material Judge] Image unchanged — auto-accepting (no token spend)")
            return True, "none"
        _prev_candidate_hash[0] = current_hash

        # --- Build prompt and bundle based on mode ---
        if is_multi:
            prompt_text = _MATERIAL_IMAGE_MULTI_PROMPT.format(
                material_spec=material_spec,
                score_attributes=", ".join(score_attributes),
            )
            bundle = MediaBundle(items={
                "candidate": candidate,
                "material_ref": grid_image,
                "prompt": TextMedia(text=prompt_text),
            })
            schema = _SpecJudgeResult.model_json_schema()
        elif is_image:
            prompt_text = _MATERIAL_IMAGE_SINGLE_PROMPT.format(
                material_spec=material_spec,
            )
            bundle = MediaBundle(items={
                "candidate": candidate,
                "material_ref": grid_image,
                "prompt": TextMedia(text=prompt_text),
            })
            schema = _MaterialJudgeResult.model_json_schema()
        else:
            prompt_text = _MATERIAL_TEXT_PROMPT.format(
                material_spec=material_spec,
            )
            bundle = MediaBundle(items={
                "candidate": candidate,
                "prompt": TextMedia(text=prompt_text),
            })
            schema = _MaterialJudgeResult.model_json_schema()

        # --- Call VLM ---
        model = session.acquire()
        try:
            raw_json = _call_vlm_structured(
                model, bundle,
                schema=schema,
                label="Material Judge",
            )
            session.record_usage("Material Judge")

            # --- Parse and score ---
            if is_multi:
                try:
                    parsed = _SpecJudgeResult.model_validate_json(raw_json)
                except Exception:
                    return False, "Material judge parse error — regenerate with the specified materials."

                scores = {k: float(v) for k, v in parsed.scores.items()}
                repair = parsed.repair
                avg = sum(scores.values()) / len(scores) if scores else 0.0
                lowest_val = min(scores.values()) if scores else 0.0
                accepted = avg >= avg_threshold and lowest_val >= min_floor

                context["_judge_metadata_material"] = {
                    "scores": scores,
                    "avg_score": round(avg, 2),
                    "lowest_score": round(lowest_val, 2),
                }

                verdict = "ACCEPT" if accepted else "REJECT"
                scores_str = ", ".join(f"{k}={v}" for k, v in scores.items())
                print(f"  [Material Judge {verdict}] avg={avg:.1f} min={lowest_val:.1f} "
                      f"(threshold: avg>={avg_threshold}, min>={min_floor})")
                print(f"  Scores: {scores_str}")
                if not accepted:
                    print(f"  Repair: {repair}")
                return accepted, repair
            else:
                try:
                    parsed = _MaterialJudgeResult.model_validate_json(raw_json)
                except Exception:
                    return False, "Material judge parse error — regenerate with the specified materials."

                score = float(parsed.score)
                repair = parsed.repair
                accepted = score >= avg_threshold and score >= min_floor

                context["_judge_metadata_material"] = {
                    "scores": {"material": score},
                    "avg_score": round(score, 2),
                    "lowest_score": round(score, 2),
                }

                verdict = "ACCEPT" if accepted else "REJECT"
                print(f"  [Material Judge {verdict}] score={score:.1f} "
                      f"(threshold: avg>={avg_threshold}, min>={min_floor})")
                if not accepted:
                    print(f"  Repair: {repair}")
                return accepted, repair
        finally:
            session.release()

    return judge
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_material_judge.py -v`
Expected: 5 PASS (2 from Task 1 + 3 new)

- [ ] **Step 5: Commit**

```bash
git add workflows/sketch_to_shoe/scripts/judge.py tests/test_material_judge.py
git commit -m "feat: add make_material_judge factory with text mode"
```

---

### Task 3: Test image modes (single + multi-material)

**Files:**
- Test: `tests/test_material_judge.py`

- [ ] **Step 1: Write image mode tests**

Append to `tests/test_material_judge.py`:

```python
def test_single_image_mode_accept():
    """Image mode with single material: grid_image provided, no material_names."""
    result_json = _MaterialJudgeResult(
        observation="Material matches the swatch — brown leather",
        score=5,
        repair="none",
    ).model_dump_json()

    session, model = _mock_vlm_text_mode(result_json)
    grid = ImageMedia(image=PILImage.new("RGB", (500, 530), (200, 150, 100)))
    judge = make_material_judge(
        session=session,
        material_spec="Apply the material shown in the reference image to the shoe.",
        grid_image=grid,
        material_names=None,
        tolerance="generous",
    )
    ctx = {"image": _make_candidate()}
    accepted, feedback = judge(ctx)
    assert accepted is True
    # Bundle should contain material_ref
    call_args = model.run.call_args
    bundle = call_args[0][0]
    assert "material_ref" in bundle.items


def test_single_image_mode_with_one_name():
    """Image mode with material_names=["Suede A"] → still single mode (len <= 1)."""
    result_json = _MaterialJudgeResult(
        observation="Suede matches", score=4, repair="none",
    ).model_dump_json()
    session, model = _mock_vlm_text_mode(result_json)
    grid = ImageMedia(image=PILImage.new("RGB", (500, 530), (200, 150, 100)))
    judge = make_material_judge(
        session=session,
        material_spec="Apply the material shown in the reference image to the shoe.",
        grid_image=grid,
        material_names=["Suede A"],
        tolerance="generous",
    )
    ctx = {"image": _make_candidate()}
    accepted, _ = judge(ctx)
    assert accepted is True
    # Metadata should have single "material" key, not per-name keys
    meta = ctx["_judge_metadata_material"]
    assert "material" in meta["scores"]


def test_multi_image_mode_accept():
    """Multi-material image mode: grid_image + len(material_names) > 1."""
    result_json = _SpecJudgeResult(
        observations={
            "Suede_A_match": "Brown suede matches swatch",
            "Suede_A_placement": "Applied to upper correctly",
            "Color_1_match": "Red matches swatch",
            "Color_1_placement": "Applied to heel correctly",
        },
        scores={
            "Suede_A_match": 5,
            "Suede_A_placement": 4,
            "Color_1_match": 5,
            "Color_1_placement": 5,
        },
        repair="none",
    ).model_dump_json()

    session, model = _mock_vlm_text_mode(result_json)
    grid = ImageMedia(image=PILImage.new("RGB", (1000, 530), (200, 150, 100)))
    judge = make_material_judge(
        session=session,
        material_spec="Apply Suede A to upper, Color 1 to heel.",
        grid_image=grid,
        material_names=["Suede A", "Color 1"],
        tolerance="generous",
    )
    ctx = {"image": _make_candidate()}
    accepted, feedback = judge(ctx)
    assert accepted is True
    meta = ctx["_judge_metadata_material"]
    assert "Suede_A_match" in meta["scores"]
    assert "Color_1_placement" in meta["scores"]


def test_multi_image_mode_reject():
    """Multi-material: one score below min_floor → reject."""
    result_json = _SpecJudgeResult(
        observations={
            "Suede_A_match": "Wrong material", "Suede_A_placement": "Wrong spot",
            "Color_1_match": "Good", "Color_1_placement": "Good",
        },
        scores={
            "Suede_A_match": 1, "Suede_A_placement": 2,
            "Color_1_match": 5, "Color_1_placement": 5,
        },
        repair="Suede A is completely wrong — use brown suede on the upper.",
    ).model_dump_json()
    session, model = _mock_vlm_text_mode(result_json)
    grid = ImageMedia(image=PILImage.new("RGB", (1000, 530), (200, 150, 100)))
    judge = make_material_judge(
        session=session,
        material_spec="Apply Suede A to upper, Color 1 to heel.",
        grid_image=grid,
        material_names=["Suede A", "Color 1"],
        tolerance="moderate",
    )
    ctx = {"image": _make_candidate()}
    accepted, feedback = judge(ctx)
    assert accepted is False
    assert "suede" in feedback.lower() or "Suede" in feedback
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_material_judge.py -v`
Expected: 9 PASS (5 previous + 4 new)

- [ ] **Step 3: Commit**

```bash
git add tests/test_material_judge.py
git commit -m "test: add image mode tests for material judge"
```

---

### Task 4: Test edge cases (hash dedup, parse failure, empty spec)

**Files:**
- Test: `tests/test_material_judge.py`

- [ ] **Step 1: Write edge case tests**

Append to `tests/test_material_judge.py`:

```python
def test_hash_dedup_auto_accepts():
    """Same image called twice → second call auto-accepts without VLM call."""
    result_json = _MaterialJudgeResult(
        observation="ok", score=4, repair="none",
    ).model_dump_json()
    session, model = _mock_vlm_text_mode(result_json)
    judge = make_material_judge(
        session=session,
        material_spec="red leather",
        grid_image=None,
        tolerance="generous",
    )
    candidate = _make_candidate()
    ctx1 = {"image": candidate}
    judge(ctx1)
    assert model.run.call_count == 1

    # Same image again
    ctx2 = {"image": candidate}
    accepted, feedback = judge(ctx2)
    assert accepted is True
    assert feedback == "none"
    assert model.run.call_count == 1  # No second VLM call


def test_parse_failure_returns_rejection():
    """VLM returns garbage → graceful rejection, not crash."""
    session, model = _mock_vlm_text_mode("not valid json at all")
    judge = make_material_judge(
        session=session,
        material_spec="red leather",
        grid_image=None,
        tolerance="generous",
    )
    ctx = {"image": _make_candidate()}
    accepted, feedback = judge(ctx)
    assert accepted is False
    assert "parse error" in feedback.lower()


def test_empty_material_spec_raises():
    """material_spec="" should raise ValueError."""
    import pytest
    session = MagicMock()
    with pytest.raises(ValueError, match="non-empty"):
        make_material_judge(session=session, material_spec="")
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `pytest tests/test_material_judge.py -v`
Expected: 12 PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_material_judge.py
git commit -m "test: add edge case tests for material judge (dedup, parse failure, validation)"
```

---

### Task 5: Update `make_best_fn` scoring formula

**Files:**
- Modify: `workflows/sketch_to_shoe/scripts/judge.py:996-998`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_material_judge.py`:

```python
from judge import make_best_fn
from casadei.loop import LoopIteration


def test_best_fn_includes_material_avg():
    """best_fn should factor material_avg into candidate ranking."""
    session = MagicMock()
    best_fn = make_best_fn(session=session, output_key="image")

    img_good_camera = _make_candidate()
    img_good_material = _make_candidate()
    # Modify the second image so it has a different identity
    img_good_material.image.putpixel((0, 0), (255, 0, 0))

    history = [
        LoopIteration(
            index=0,
            accepted=False,
            feedback="camera bad",
            duration_ms=100.0,
            outputs={"image": img_good_camera},
            metadata={"sketch_avg": None, "spec_avg": 5.0, "material_avg": 1.0},
        ),
        LoopIteration(
            index=1,
            accepted=False,
            feedback="camera ok-ish",
            duration_ms=100.0,
            outputs={"image": img_good_material},
            metadata={"sketch_avg": None, "spec_avg": 3.0, "material_avg": 5.0},
        ),
    ]
    result = best_fn(history, {})
    # iter 0: non-zero components = [5.0, 1.0] → avg = 3.0
    # iter 1: non-zero components = [3.0, 5.0] → avg = 4.0
    # iter 1 should win
    assert result.get("best_selection_index") == 2  # 1-indexed


def test_best_fn_backward_compatible_no_material():
    """best_fn still works when material_avg is absent (older pipelines)."""
    session = MagicMock()
    best_fn = make_best_fn(session=session, output_key="image")

    img = _make_candidate()
    history = [
        LoopIteration(
            index=0,
            accepted=False,
            feedback="rejected",
            duration_ms=100.0,
            outputs={"image": img},
            metadata={"sketch_avg": None, "spec_avg": 4.0},
        ),
    ]
    result = best_fn(history, {})
    assert result.get("best_selection_index") == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_material_judge.py::test_best_fn_includes_material_avg -v`
Expected: FAIL — iter 0 would currently win with `(0 + 5) / 2 = 2.5` vs iter 1 `(0 + 3) / 2 = 1.5`

- [ ] **Step 3: Update the scoring formula**

In `workflows/sketch_to_shoe/scripts/judge.py`, find `make_best_fn` (line 975). Replace lines 996-998:

Old:
```python
            sketch_avg = record.metadata.get("sketch_avg") or 0.0
            spec_avg = record.metadata.get("spec_avg") or 0.0
            combined = (sketch_avg + spec_avg) / 2.0
```

New:
```python
            sketch_avg = record.metadata.get("sketch_avg") or 0.0
            spec_avg = record.metadata.get("spec_avg") or 0.0
            material_avg = record.metadata.get("material_avg") or 0.0
            components = [v for v in (sketch_avg, spec_avg, material_avg) if v > 0]
            combined = sum(components) / len(components) if components else 0.0
```

This uses only non-zero components in the divisor, so pipelines that don't provide `material_avg` keep their original `(sketch_avg + spec_avg) / 2.0` behavior. No absolute score changes for existing code.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_material_judge.py -v`
Expected: 14 PASS

- [ ] **Step 5: Commit**

```bash
git add workflows/sketch_to_shoe/scripts/judge.py tests/test_material_judge.py
git commit -m "feat: update best_fn to include material_avg in scoring"
```

---

## Chunk 2: Pipeline Integration in pipeline.py

### Task 6: Add `DEFAULT_MATERIAL` constant and import `make_material_judge`

**Files:**
- Modify: `workflows/sketch_to_shoe_gemini/pipeline.py:34,126,218`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_material_judge.py`:

```python
from workflows.sketch_to_shoe_gemini.pipeline import DEFAULT_MATERIAL, build_pipeline


def test_default_material_constant():
    assert DEFAULT_MATERIAL == "black patent leather"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_material_judge.py::test_default_material_constant -v`
Expected: FAIL with `ImportError: cannot import name 'DEFAULT_MATERIAL'`

- [ ] **Step 3: Make the changes**

In `workflows/sketch_to_shoe_gemini/pipeline.py`:

1. Update the import on line 34 to include `make_material_judge`:
```python
from judge import VLMSession, make_spec_judge, make_shoe_count_judge, make_material_judge, make_best_fn  # noqa: E402
```

2. Add `DEFAULT_MATERIAL` constant after `MAX_ITERATIONS` (after line 126):
```python
DEFAULT_MATERIAL = "black patent leather"
```

3. Replace the hardcoded `"black patent leather"` in `template_kwargs` (line 218):

Old:
```python
            **({"materials_instructions": materials_instructions} if use_materials_mode else {"material": spec.get("material", "black patent leather")}),
```

New:
```python
            **({"materials_instructions": materials_instructions} if use_materials_mode else {"material": spec.get("material", DEFAULT_MATERIAL)}),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_material_judge.py::test_default_material_constant -v`
Expected: PASS

Run: `pytest tests/test_materials_prompt.py -v`
Expected: All existing tests still PASS (no regressions)

- [ ] **Step 5: Commit**

```bash
git add workflows/sketch_to_shoe_gemini/pipeline.py tests/test_material_judge.py
git commit -m "feat: add DEFAULT_MATERIAL constant and import make_material_judge"
```

---

### Task 7: Update `_combined_judge` and return value

**Files:**
- Modify: `workflows/sketch_to_shoe_gemini/pipeline.py:240-294`
- Test: `tests/test_material_judge.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_material_judge.py`:

```python
def test_build_pipeline_returns_three_sessions():
    """build_pipeline should return 3 VLM sessions (camera, count, material)."""
    spec = {
        "material": "red leather",
        "camera_angle": "3/4",
        "extra": {},
    }
    vlm = MagicMock()
    pipeline, agent, sessions, grid_img = build_pipeline(spec, vlm, foot="pair")
    assert len(sessions) == 3


def test_build_pipeline_materials_mode_returns_three_sessions():
    """build_pipeline with materials mode should also return 3 sessions."""
    spec = {
        "material": "ignored",
        "camera_angle": "3/4",
        "extra": {},
        "materials": [
            {"name": "Test Mat", "image": PILImage.new("RGB", (200, 200), (128, 128, 128)),
             "placement": "toe", "note": None, "is_color": False},
        ],
    }
    vlm = MagicMock()
    pipeline, agent, sessions, grid_img = build_pipeline(spec, vlm, foot="pair")
    assert len(sessions) == 3
    assert grid_img is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_material_judge.py::test_build_pipeline_returns_three_sessions -v`
Expected: FAIL — currently returns 2 sessions

- [ ] **Step 3: Update `_combined_judge` and return value**

In `workflows/sketch_to_shoe_gemini/pipeline.py`, replace the section from `session_camera = VLMSession(...)` through the `return` statement (lines 222-294) with:

```python
    session_camera = VLMSession("gemini_flash")
    session_count = VLMSession("gemini_flash_lite")
    session_material = VLMSession("gemini_flash")

    camera_judge = make_spec_judge(
        session=session_camera,
        candidate_key="image",
        spec={"camera_angle": camera_preset["camera_desc"] + " " + camera_preset["staging_desc"]},
        tolerance="generous",
        include_quality_features=False,
        judge_notes=get_judge_notes(),
    )

    count_judge = make_shoe_count_judge(
        session=session_count,
        foot=foot,
        candidate_key="image",
    )

    material_judge = make_material_judge(
        session=session_material,
        material_spec=materials_instructions if use_materials_mode else spec.get("material", DEFAULT_MATERIAL),
        grid_image=ImageMedia(image=grid_image) if grid_image is not None else None,
        material_names=resolved_names if use_materials_mode and len(materials_list) > 1 else None,
        candidate_key="image",
        tolerance="moderate",
    )

    def _combined_judge(context):
        image = context.get("image")
        # Force PIL lazy-load before threads start — prevents concurrent load() race
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

    loop = LoopStep(
        name="angle_correction_loop",
        body=[edit_step],
        judge=_combined_judge,
        max_iterations=MAX_ITERATIONS,
        best_fn=make_best_fn(
            session=vlm_session,
            output_key="image",
        ),
        swap_models=True,
        output_key="image",
        feedback_template_var="feedback",
    )

    return Pipeline(name="sketch_to_shoe_gemini", steps=[loop]), gemini_agent, [session_camera, session_count, session_material], grid_image
```

Note: The `material_judge` construction references `resolved_names` and `materials_list` which are only defined when `use_materials_mode` is True. When `use_materials_mode` is False, the condition `use_materials_mode and len(materials_list) > 1` short-circuits and `material_names` is `None`. This is safe.

- [ ] **Step 4: Run all tests to verify they pass**

Run: `pytest tests/test_material_judge.py tests/test_materials_prompt.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add workflows/sketch_to_shoe_gemini/pipeline.py tests/test_material_judge.py
git commit -m "feat: integrate material judge as third parallel judge in pipeline"
```

---

### Task 8: Final verification

- [ ] **Step 1: Run all project tests**

Run: `pytest tests/ -v`
Expected: All PASS — no regressions

- [ ] **Step 2: Verify imports work end-to-end**

Run: `python -c "from workflows.sketch_to_shoe_gemini.pipeline import build_pipeline, DEFAULT_MATERIAL; print('OK', DEFAULT_MATERIAL)"`
Expected: `OK black patent leather`

- [ ] **Step 3: Final commit if any cleanup needed**

Only if there are uncommitted changes from minor fixes during verification.
