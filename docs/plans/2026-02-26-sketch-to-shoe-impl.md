# Sketch-to-Shoe Agentic Loop — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a sketch-to-realistic-shoe agentic loop that converts design sketches + spec (material, color, camera angle, extras) into a studio-quality product photo, using FireRed for generation and dual VLM judges for sketch fidelity and spec compliance.

**Architecture:** Two new files: `workflows/sketch_to_shoe/scripts/judge.py` holds the dual-judge logic (VLMSession, sketch fidelity judge, spec compliance judge, dual-judge combiner, best_fn); `tests/run_sketch_to_shoe_loop.py` is the main entry point with CLI, sketch grid assembly, spec parsing, pipeline construction, and result saving. Unit tests in `tests/test_sketch_to_shoe.py` cover all pure functions without GPU.

**Tech Stack:** Python 3.12+, PIL (Pillow), FireRed via `casadei.providers.firered_image_edit`, Qwen3-VL-8B via `casadei.providers.qwen3_vl_8b`, `LoopStep` from `casadei.loop`, same VLMSession/judge patterns as `workflows/shoe_tryon_loop/scripts/judge.py`.

---

### Task 1: Scaffold judge module skeleton

**Files:**
- Create: `workflows/sketch_to_shoe/scripts/__init__.py`
- Create: `workflows/sketch_to_shoe/scripts/judge.py`

**Step 1: Create the directory and empty `__init__.py`**

```bash
mkdir -p workflows/sketch_to_shoe/scripts
touch workflows/sketch_to_shoe/scripts/__init__.py
```

**Step 2: Write `judge.py` with imports, constants, helpers, and VLMSession**

Create `workflows/sketch_to_shoe/scripts/judge.py`:

```python
"""Judge functions for the sketch-to-shoe agentic loop.

Two independent judges run per iteration:
  1. Sketch Fidelity Judge  — IMAGE1=sketch, IMAGE2=generated → scores design faithfulness
  2. Spec Compliance Judge  — IMAGE=generated, TEXT=spec → scores material/color/angle/quality

make_dual_judge() wraps both into one JudgeCallable for LoopStep.
"""

from __future__ import annotations

import gc
import re
import sys
from PIL import Image as PILImage, ImageDraw, ImageFont

from casadei.loop import BestFn, JudgeCallable, LoopIteration
from casadei.media import ImageMedia, Media, TextMedia, MediaBundle

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOLERANCE_CONFIGS = {
    "generous":  {"avg_threshold": 2.5, "min_floor": 1.5},
    "moderate":  {"avg_threshold": 3.5, "min_floor": 2.5},
    "strict":    {"avg_threshold": 4.5, "min_floor": 3.5},
}

_MAX_RETRIES = 3
_FALLBACK_SKETCH_FEATURES = ["shape", "proportions", "toe_shape", "heel_style", "sole_design"]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _unload_and_cleanup(model) -> None:
    model.unload_model()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _stream_vlm(model, bundle: MediaBundle, label: str = "VLM") -> str:
    sys.stdout.write(f"\n  [{label}] ")
    sys.stdout.flush()
    chunks: list[str] = []
    for chunk in model.run_streaming(bundle):
        sys.stdout.write(chunk)
        sys.stdout.flush()
        chunks.append(chunk)
    sys.stdout.write("\n")
    sys.stdout.flush()
    return "".join(chunks).strip()


# ---------------------------------------------------------------------------
# VLMSession (identical lifecycle to shoe_tryon_loop)
# ---------------------------------------------------------------------------

class VLMSession:
    """Shared VLM model lifecycle. Swap mode by default; persistent if load() called."""

    def __init__(self, model_name: str = "qwen3_vl_8b"):
        self._model_name = model_name
        self._model = None
        self._persistent = False

    def load(self) -> None:
        if self._model is None:
            from casadei.models.registry import default_registry
            model_cls = default_registry.get(self._model_name)
            self._model = model_cls()
            self._model.load_model()
        self._persistent = True
        print(f"  [VLMSession] Pre-loaded {self._model_name}")

    def unload(self) -> None:
        if self._model is not None:
            _unload_and_cleanup(self._model)
            self._model = None
        self._persistent = False

    def acquire(self):
        if self._model is None:
            from casadei.models.registry import default_registry
            model_cls = default_registry.get(self._model_name)
            self._model = model_cls()
            self._model.load_model()
        return self._model

    def release(self) -> None:
        if not self._persistent and self._model is not None:
            _unload_and_cleanup(self._model)
            self._model = None


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------

def _parse_scores(text: str) -> dict[str, float]:
    scores_match = re.search(r"SCORES:\s*(.+)", text, re.IGNORECASE)
    if not scores_match:
        raise ValueError("No 'SCORES:' line found")
    scores_text = scores_match.group(1).strip()
    scores = {}
    for pair in re.split(r",\s*", scores_text):
        pair = pair.strip()
        if not pair:
            continue
        m = re.match(r"(.+?)\s*=\s*([0-9]+(?:\.[0-9]+)?)", pair)
        if m:
            scores[m.group(1).strip()] = float(m.group(2))
    if not scores:
        raise ValueError(f"Could not parse any scores from: {scores_text}")
    return scores


def _parse_repair(text: str) -> str:
    repair_match = re.search(r"REPAIR:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if repair_match:
        return repair_match.group(1).strip()
    return text


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SKETCH_FEATURE_PROMPT = """\
Look at this shoe design sketch. List 4-6 short visual attribute names that \
describe its key design features (e.g. toe shape, heel type, sole style, \
silhouette, ankle height, strap count).

Reply with ONLY a JSON array of short strings. Example:
["pointed toe", "block heel", "platform sole", "ankle strap", "side zipper"]
"""

_SKETCH_SCORE_PROMPT = """\
You are a design fidelity inspector for a shoe manufacturing system.

IMAGE 1 is the ORIGINAL SKETCH — the designer's intended shoe design.
IMAGE 2 is the GENERATED PHOTO — a photorealistic rendering that should \
faithfully represent the sketch.

Your job: compare the shoe in IMAGE 2 to the design in IMAGE 1 and score \
how closely the generated photo matches the sketch's design intent.

{iteration_context}\
{previous_feedback}\
{stale_nudge}\
Score each attribute from 1 to 5:
  1 = completely different from the sketch
  2 = vaguely similar but clearly wrong
  3 = recognizably the same design but noticeable differences
  4 = close match with only minor differences
  5 = near-identical to the sketch design

Attributes to score: {features}

Reply in this exact format:
SCORES: {score_format}
REPAIR: <describe what is wrong in the generated photo compared to the sketch, \
then tell the model to look at the sketch and correct that specific element>
"""

_SPEC_SCORE_PROMPT = """\
You are a product specification inspector for a shoe manufacturing system.

The image shows a generated photorealistic shoe product photo.

Your job: check whether the shoe in the image matches the following \
design specifications, and evaluate the overall photo quality.

Design specifications:
{spec_text}

{iteration_context}\
{previous_feedback}\
{stale_nudge}\
Score each attribute from 1 to 5:
  1 = completely fails this specification
  2 = vaguely meets it but clearly wrong
  3 = partially meets it with noticeable issues
  4 = mostly meets it with only minor issues
  5 = fully and clearly meets this specification

Attributes to score: {features}

Reply in this exact format:
SCORES: {score_format}
REPAIR: <describe what does not match the specifications, then tell the model \
specifically what to change to meet them>
"""

_BEST_PROMPT = """\
You are selecting the best sketch-to-shoe rendering result.

IMAGE 1 is the ORIGINAL SKETCH — the designer's intended shoe design.
IMAGE 2 shows {n} candidate photorealistic renderings side by side, labeled \
Option 1 through Option {n} from left to right.

Which option best represents the shoe design from the sketch in a \
professional product photograph style?

Reply with ONLY a single number (1 to {n}) on the first line.
Then on the next line, briefly explain why.
"""


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_sketch_features(session: VLMSession, sketch_image: ImageMedia) -> list[str]:
    """Ask VLM to identify visual design attributes from the sketch.

    Returns list of short attribute names for the sketch fidelity judge.
    Falls back to _FALLBACK_SKETCH_FEATURES on parse failure.
    """
    import json as _json

    bundle = MediaBundle(items={
        "sketch": sketch_image,
        "prompt": TextMedia(text=_SKETCH_FEATURE_PROMPT),
    })

    model = session.acquire()
    try:
        response = _stream_vlm(model, bundle, label="Sketch Feature Extraction")
    finally:
        session.release()

    try:
        start = response.index("[")
        end = response.index("]", start) + 1
        features = _json.loads(response[start:end])
        if isinstance(features, list) and all(isinstance(f, str) for f in features) and len(features) >= 2:
            print(f"  Extracted sketch features: {features}")
            return features
    except (ValueError, _json.JSONDecodeError):
        pass

    print(f"  Feature extraction parse failed, using fallback: {_FALLBACK_SKETCH_FEATURES}")
    return list(_FALLBACK_SKETCH_FEATURES)


# ---------------------------------------------------------------------------
# Sketch Fidelity Judge
# ---------------------------------------------------------------------------

def make_sketch_judge(
    session: VLMSession,
    sketch_key: str = "sketch",
    candidate_key: str = "image",
    features: list[str] | None = None,
    tolerance: str = "strict",
) -> JudgeCallable:
    """Return a JudgeCallable that scores sketch design fidelity.

    IMAGE 1 = sketch grid (reference design)
    IMAGE 2 = generated photo (candidate)
    """
    if features is None:
        features = list(_FALLBACK_SKETCH_FEATURES)

    tol = TOLERANCE_CONFIGS.get(tolerance, TOLERANCE_CONFIGS["strict"])
    avg_threshold = tol["avg_threshold"]
    min_floor = tol["min_floor"]

    prev_lowest_attr: list[str | None] = [None]
    stale_count: list[int] = [0]
    prev_feedback: list[str] = [""]
    score_format = ", ".join(f"{f}=N" for f in features)

    def judge(context: dict[str, Media]) -> tuple[bool, str]:
        sketch = context.get(sketch_key)
        candidate = context.get(candidate_key)

        if not isinstance(sketch, ImageMedia):
            return False, f"Missing sketch image (key='{sketch_key}')."
        if not isinstance(candidate, ImageMedia):
            return False, f"Missing candidate image (key='{candidate_key}')."

        iteration = context.get("loop_iteration", 0)
        max_iterations = context.get("loop_max_iterations", 5)

        iteration_context = f"This is attempt {iteration + 1} of {max_iterations}.\n"
        previous_feedback = f"Previous feedback: \"{prev_feedback[0]}\"\n" if prev_feedback[0] else ""
        stale_nudge = ""
        if stale_count[0] >= 2 and prev_lowest_attr[0]:
            attr = prev_lowest_attr[0]
            stale_nudge = (
                f"The attribute '{attr}' has been the weakest for {stale_count[0]} consecutive "
                f"attempts. In your REPAIR instruction, specifically describe what is wrong "
                f"with '{attr}' in the generated photo compared to the sketch.\n"
            )

        prompt_text = _SKETCH_SCORE_PROMPT.format(
            iteration_context=iteration_context,
            previous_feedback=previous_feedback,
            stale_nudge=stale_nudge,
            features=", ".join(features),
            score_format=score_format,
        )

        bundle = MediaBundle(items={
            "sketch": sketch,
            "candidate": candidate,
            "prompt": TextMedia(text=prompt_text),
        })

        model = session.acquire()
        try:
            last_error = ""
            raw_response = ""

            for attempt in range(_MAX_RETRIES + 1):
                if attempt > 0 and last_error:
                    retry_prompt = (
                        f"{prompt_text}\n\nYour previous response could not be parsed. "
                        f"Error: {last_error}\nPlease reply in the exact format:\n"
                        f"SCORES: {score_format}\nREPAIR: ..."
                    )
                    bundle = MediaBundle(items={
                        "sketch": sketch,
                        "candidate": candidate,
                        "prompt": TextMedia(text=retry_prompt),
                    })

                raw_response = _stream_vlm(model, bundle, label="Sketch Judge")

                try:
                    scores = _parse_scores(raw_response)
                    repair = _parse_repair(raw_response)
                    avg = sum(scores.values()) / len(scores)
                    lowest_val = min(scores.values())
                    lowest_attr = min(scores, key=scores.get)
                    accepted = avg >= avg_threshold and lowest_val >= min_floor

                    if lowest_attr == prev_lowest_attr[0]:
                        stale_count[0] += 1
                    else:
                        prev_lowest_attr[0] = lowest_attr
                        stale_count[0] = 1
                    prev_feedback[0] = repair

                    context["_judge_metadata_sketch"] = {
                        "scores": scores,
                        "avg_score": round(avg, 2),
                        "lowest_score": round(lowest_val, 2),
                        "lowest_attr": lowest_attr,
                        "stale_count": stale_count[0],
                    }

                    verdict = "ACCEPT" if accepted else "REJECT"
                    scores_str = ", ".join(f"{k}={v}" for k, v in scores.items())
                    print(f"  [Sketch Judge {verdict}] avg={avg:.1f} min={lowest_val:.1f} "
                          f"(threshold: avg>={avg_threshold}, min>={min_floor})")
                    print(f"  Scores: {scores_str}")
                    if not accepted:
                        print(f"  Repair: {repair[:200]}")
                    return accepted, repair

                except ValueError as e:
                    last_error = str(e)
                    print(f"  [Sketch Judge] Parse error: {last_error} "
                          f"(attempt {attempt + 1}/{_MAX_RETRIES + 1})")

            print("  [Sketch Judge] Fallback: treating unparseable response as REJECT")
            prev_feedback[0] = raw_response or "Could not parse VLM response."
            return False, prev_feedback[0]
        finally:
            session.release()

    return judge


# ---------------------------------------------------------------------------
# Spec Compliance Judge
# ---------------------------------------------------------------------------

def make_spec_judge(
    session: VLMSession,
    candidate_key: str = "image",
    spec: dict[str, str] | None = None,
    tolerance: str = "strict",
) -> JudgeCallable:
    """Return a JudgeCallable that scores spec compliance and photo quality.

    IMAGE = generated photo only (no sketch shown)
    TEXT = spec dict (material, color, camera_angle, extras) + quality attributes
    """
    if spec is None:
        spec = {}

    tol = TOLERANCE_CONFIGS.get(tolerance, TOLERANCE_CONFIGS["strict"])
    avg_threshold = tol["avg_threshold"]
    min_floor = tol["min_floor"]

    quality_features = ["white_background", "lighting", "sharpness"]
    features = list(spec.keys()) + [f for f in quality_features if f not in spec]

    spec_lines = [f"- {k.capitalize()}: {v}" for k, v in spec.items()]
    spec_text = "\n".join(spec_lines) if spec_lines else "(no additional specs)"

    prev_lowest_attr: list[str | None] = [None]
    stale_count: list[int] = [0]
    prev_feedback: list[str] = [""]
    score_format = ", ".join(f"{f}=N" for f in features)

    def judge(context: dict[str, Media]) -> tuple[bool, str]:
        candidate = context.get(candidate_key)

        if not isinstance(candidate, ImageMedia):
            return False, f"Missing candidate image (key='{candidate_key}')."

        iteration = context.get("loop_iteration", 0)
        max_iterations = context.get("loop_max_iterations", 5)

        iteration_context = f"This is attempt {iteration + 1} of {max_iterations}.\n"
        previous_feedback = f"Previous feedback: \"{prev_feedback[0]}\"\n" if prev_feedback[0] else ""
        stale_nudge = ""
        if stale_count[0] >= 2 and prev_lowest_attr[0]:
            attr = prev_lowest_attr[0]
            stale_nudge = (
                f"The attribute '{attr}' has been below standard for {stale_count[0]} consecutive "
                f"attempts. Focus your REPAIR instruction specifically on '{attr}'.\n"
            )

        prompt_text = _SPEC_SCORE_PROMPT.format(
            spec_text=spec_text,
            iteration_context=iteration_context,
            previous_feedback=previous_feedback,
            stale_nudge=stale_nudge,
            features=", ".join(features),
            score_format=score_format,
        )

        bundle = MediaBundle(items={
            "candidate": candidate,
            "prompt": TextMedia(text=prompt_text),
        })

        model = session.acquire()
        try:
            last_error = ""
            raw_response = ""

            for attempt in range(_MAX_RETRIES + 1):
                if attempt > 0 and last_error:
                    retry_prompt = (
                        f"{prompt_text}\n\nYour previous response could not be parsed. "
                        f"Error: {last_error}\nPlease reply in the exact format:\n"
                        f"SCORES: {score_format}\nREPAIR: ..."
                    )
                    bundle = MediaBundle(items={
                        "candidate": candidate,
                        "prompt": TextMedia(text=retry_prompt),
                    })

                raw_response = _stream_vlm(model, bundle, label="Spec Judge")

                try:
                    scores = _parse_scores(raw_response)
                    repair = _parse_repair(raw_response)
                    avg = sum(scores.values()) / len(scores)
                    lowest_val = min(scores.values())
                    lowest_attr = min(scores, key=scores.get)
                    accepted = avg >= avg_threshold and lowest_val >= min_floor

                    if lowest_attr == prev_lowest_attr[0]:
                        stale_count[0] += 1
                    else:
                        prev_lowest_attr[0] = lowest_attr
                        stale_count[0] = 1
                    prev_feedback[0] = repair

                    context["_judge_metadata_spec"] = {
                        "scores": scores,
                        "avg_score": round(avg, 2),
                        "lowest_score": round(lowest_val, 2),
                        "lowest_attr": lowest_attr,
                        "stale_count": stale_count[0],
                    }

                    verdict = "ACCEPT" if accepted else "REJECT"
                    scores_str = ", ".join(f"{k}={v}" for k, v in scores.items())
                    print(f"  [Spec Judge {verdict}] avg={avg:.1f} min={lowest_val:.1f} "
                          f"(threshold: avg>={avg_threshold}, min>={min_floor})")
                    print(f"  Scores: {scores_str}")
                    if not accepted:
                        print(f"  Repair: {repair[:200]}")
                    return accepted, repair

                except ValueError as e:
                    last_error = str(e)
                    print(f"  [Spec Judge] Parse error: {last_error} "
                          f"(attempt {attempt + 1}/{_MAX_RETRIES + 1})")

            print("  [Spec Judge] Fallback: treating unparseable response as REJECT")
            prev_feedback[0] = raw_response or "Could not parse VLM response."
            return False, prev_feedback[0]
        finally:
            session.release()

    return judge


# ---------------------------------------------------------------------------
# Dual Judge Combiner
# ---------------------------------------------------------------------------

def make_dual_judge(
    sketch_judge: JudgeCallable,
    spec_judge: JudgeCallable,
) -> JudgeCallable:
    """Combine sketch fidelity and spec compliance judges into one JudgeCallable.

    Runs both judges sequentially. Accepted only if both accept.
    Merges metadata from both judges into context['_judge_metadata'].
    Feedback format:
        [Sketch feedback]: ...
        [Spec feedback]: ...
    """
    def judge(context: dict[str, Media]) -> tuple[bool, str]:
        accepted1, feedback1 = sketch_judge(context)
        meta1 = context.pop("_judge_metadata_sketch", {})

        accepted2, feedback2 = spec_judge(context)
        meta2 = context.pop("_judge_metadata_spec", {})

        context["_judge_metadata"] = {
            "sketch_scores": meta1.get("scores", {}),
            "sketch_avg": meta1.get("avg_score"),
            "sketch_lowest_attr": meta1.get("lowest_attr"),
            "spec_scores": meta2.get("scores", {}),
            "spec_avg": meta2.get("avg_score"),
            "spec_lowest_attr": meta2.get("lowest_attr"),
        }

        combined_feedback = (
            f"[Sketch feedback]: {feedback1}\n"
            f"[Spec feedback]: {feedback2}"
        )
        return accepted1 and accepted2, combined_feedback

    return judge


# ---------------------------------------------------------------------------
# Best-of-N candidate grid helper
# ---------------------------------------------------------------------------

def _concat_candidates(images: list[PILImage.Image], label_height: int = 40) -> PILImage.Image:
    if not images:
        raise ValueError("No images to concat")
    target_h = max(img.height for img in images)
    resized = []
    for img in images:
        if img.height != target_h:
            ratio = target_h / img.height
            new_w = int(img.width * ratio)
            resized.append(img.resize((new_w, target_h), PILImage.LANCZOS))
        else:
            resized.append(img)
    total_w = sum(img.width for img in resized)
    canvas = PILImage.new("RGB", (total_w, target_h + label_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
    except (OSError, IOError):
        font = ImageFont.load_default()
    x_offset = 0
    for i, img in enumerate(resized):
        label = f"Option {i + 1}"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        draw.text((x_offset + (img.width - text_w) // 2, 6), label, fill=(0, 0, 0), font=font)
        canvas.paste(img, (x_offset, label_height))
        x_offset += img.width
    return canvas


# ---------------------------------------------------------------------------
# Best-of-N Selection
# ---------------------------------------------------------------------------

def make_best_fn(
    session: VLMSession,
    sketch_key: str = "sketch",
    output_key: str = "image",
) -> BestFn:
    """Return a BestFn that picks the best candidate relative to the sketch."""

    def best_fn(
        history: list[LoopIteration],
        context: dict[str, Media],
    ) -> dict[str, Media]:
        candidates: list[ImageMedia] = []
        for record in history:
            img = record.outputs.get(output_key)
            if isinstance(img, ImageMedia):
                candidates.append(img)

        if not candidates:
            return {}
        if len(candidates) == 1:
            print("  [VLM Best-of-N] Only 1 candidate, selecting it.")
            return {
                output_key: candidates[0],
                "best_selection_index": 1,
                "best_selection_reason": TextMedia(text="Only one candidate."),
            }

        sketch = context.get(sketch_key)
        if not isinstance(sketch, ImageMedia):
            return {output_key: candidates[-1]}

        n = len(candidates)
        grid_image = _concat_candidates([c.image for c in candidates])
        print(f"\n  [VLM Best-of-N] Selecting from {n} candidates")

        prompt_text = _BEST_PROMPT.format(n=n)
        bundle = MediaBundle(items={
            "sketch": sketch,
            "candidates_grid": ImageMedia(image=grid_image),
            "prompt": TextMedia(text=prompt_text),
        })

        model = session.acquire()
        try:
            chosen_1based = None
            response = ""
            for attempt in range(_MAX_RETRIES + 1):
                response = _stream_vlm(model, bundle, label="VLM Best-of-N")
                first_line = response.splitlines()[0] if response else ""
                match = re.search(r"\d+", first_line)
                if match:
                    val = int(match.group())
                    if 1 <= val <= n:
                        chosen_1based = val
                        break
                print(f"  [VLM Best-of-N] Could not parse option 1-{n} "
                      f"(attempt {attempt + 1}/{_MAX_RETRIES + 1})")
            if chosen_1based is None:
                chosen_1based = n
                print(f"  [VLM Best-of-N] Fallback: selecting last candidate (Option {n})")
        finally:
            session.release()

        chosen_0based = chosen_1based - 1
        print(f"  Selected: Option {chosen_1based} of {n}")
        return {
            output_key: candidates[chosen_0based],
            "best_selection_index": chosen_1based,
            "best_selection_reason": TextMedia(text=response),
        }

    return best_fn
```

**Step 3: Commit**

```bash
git add workflows/sketch_to_shoe/
git commit -m "feat(sketch-to-shoe): add complete judge module"
```

---

### Task 2: Write unit tests for judge.py pure functions

**Files:**
- Create: `tests/test_sketch_to_shoe.py`

**Step 1: Write the test file**

Create `tests/test_sketch_to_shoe.py`:

```python
"""Unit tests for sketch-to-shoe — pure functions, no GPU required."""

from __future__ import annotations

import sys
from pathlib import Path
import pytest
from PIL import Image as PILImage
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "workflows" / "sketch_to_shoe" / "scripts"))


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------

class TestParseScores:
    def test_parses_valid_scores(self):
        from judge import _parse_scores
        result = _parse_scores("SCORES: shape=4, proportions=3, toe_shape=5")
        assert result == {"shape": 4.0, "proportions": 3.0, "toe_shape": 5.0}

    def test_raises_on_missing_scores_line(self):
        from judge import _parse_scores
        with pytest.raises(ValueError, match="No 'SCORES:'"):
            _parse_scores("REPAIR: fix the toe")

    def test_raises_on_empty_scores(self):
        from judge import _parse_scores
        with pytest.raises(ValueError):
            _parse_scores("SCORES: ")

    def test_float_scores(self):
        from judge import _parse_scores
        result = _parse_scores("SCORES: color=4.5, material=3.0")
        assert result["color"] == 4.5


class TestParseRepair:
    def test_extracts_repair_text(self):
        from judge import _parse_repair
        result = _parse_repair("SCORES: shape=3\nREPAIR: The toe is too round.")
        assert result == "The toe is too round."

    def test_returns_full_text_when_no_repair_tag(self):
        from judge import _parse_repair
        result = _parse_repair("something without repair tag")
        assert result == "something without repair tag"


# ---------------------------------------------------------------------------
# Feature extraction fallback
# ---------------------------------------------------------------------------

class TestExtractSketchFeaturesFallback:
    def _make_session(self, response_text: str):
        session = MagicMock()
        model = MagicMock()
        model.run_streaming.return_value = iter([response_text])
        session.acquire.return_value = model
        return session

    def test_returns_fallback_on_invalid_json(self):
        from judge import extract_sketch_features, _FALLBACK_SKETCH_FEATURES
        from casadei.media import ImageMedia
        session = self._make_session("not json at all")
        sketch = ImageMedia(image=PILImage.new("RGB", (64, 64)))
        result = extract_sketch_features(session, sketch)
        assert result == list(_FALLBACK_SKETCH_FEATURES)

    def test_returns_extracted_features_on_valid_json(self):
        from judge import extract_sketch_features
        from casadei.media import ImageMedia
        session = self._make_session('["toe shape", "heel height", "sole thickness"]')
        sketch = ImageMedia(image=PILImage.new("RGB", (64, 64)))
        result = extract_sketch_features(session, sketch)
        assert result == ["toe shape", "heel height", "sole thickness"]

    def test_returns_fallback_on_too_few_features(self):
        from judge import extract_sketch_features, _FALLBACK_SKETCH_FEATURES
        from casadei.media import ImageMedia
        session = self._make_session('["only one"]')
        sketch = ImageMedia(image=PILImage.new("RGB", (64, 64)))
        result = extract_sketch_features(session, sketch)
        assert result == list(_FALLBACK_SKETCH_FEATURES)


# ---------------------------------------------------------------------------
# Dual judge combiner
# ---------------------------------------------------------------------------

class TestMakeDualJudge:
    def _img(self):
        from casadei.media import ImageMedia
        return ImageMedia(image=PILImage.new("RGB", (64, 64)))

    def test_both_accept_returns_accepted(self):
        from judge import make_dual_judge
        dual = make_dual_judge(lambda ctx: (True, "Sketch OK."), lambda ctx: (True, "Spec OK."))
        accepted, feedback = dual({"sketch": self._img(), "image": self._img()})
        assert accepted is True
        assert "[Sketch feedback]" in feedback
        assert "[Spec feedback]" in feedback

    def test_sketch_reject_returns_rejected(self):
        from judge import make_dual_judge
        dual = make_dual_judge(lambda ctx: (False, "Wrong toe."), lambda ctx: (True, "Spec OK."))
        accepted, _ = dual({"sketch": self._img(), "image": self._img()})
        assert accepted is False

    def test_spec_reject_returns_rejected(self):
        from judge import make_dual_judge
        dual = make_dual_judge(lambda ctx: (True, "Sketch OK."), lambda ctx: (False, "Wrong material."))
        accepted, _ = dual({"sketch": self._img(), "image": self._img()})
        assert accepted is False

    def test_both_reject_returns_rejected(self):
        from judge import make_dual_judge
        dual = make_dual_judge(lambda ctx: (False, "Bad sketch."), lambda ctx: (False, "Bad spec."))
        accepted, _ = dual({"sketch": self._img(), "image": self._img()})
        assert accepted is False

    def test_metadata_merged_into_context(self):
        from judge import make_dual_judge

        def sketch_j(ctx):
            ctx["_judge_metadata_sketch"] = {"scores": {"shape": 4.0}, "avg_score": 4.0}
            return True, "OK"

        def spec_j(ctx):
            ctx["_judge_metadata_spec"] = {"scores": {"material": 5.0}, "avg_score": 5.0}
            return True, "OK"

        dual = make_dual_judge(sketch_j, spec_j)
        ctx = {"sketch": self._img(), "image": self._img()}
        dual(ctx)
        assert "_judge_metadata" in ctx
        assert "sketch_scores" in ctx["_judge_metadata"]
        assert "spec_scores" in ctx["_judge_metadata"]
        # Temp keys cleaned up
        assert "_judge_metadata_sketch" not in ctx
        assert "_judge_metadata_spec" not in ctx
```

**Step 2: Run tests to verify they pass**

```bash
pytest tests/test_sketch_to_shoe.py -v
```

Expected: All 11 tests PASS

**Step 3: Commit**

```bash
git add tests/test_sketch_to_shoe.py
git commit -m "test(sketch-to-shoe): add unit tests for judge pure functions"
```

---

### Task 3: Write and test the main runner script

**Files:**
- Create: `tests/run_sketch_to_shoe_loop.py`

**Step 1: Write `tests/test_sketch_to_shoe.py` additions for grid/spec utilities**

Add to `tests/test_sketch_to_shoe.py`:

```python
# ---------------------------------------------------------------------------
# Sketch grid assembly
# ---------------------------------------------------------------------------

class TestBuildSketchGrid:
    def _sketch(self, w, h, color=(200, 200, 200)):
        return PILImage.new("RGB", (w, h), color)

    def test_single_square_sketch_unchanged(self):
        from run_sketch_to_shoe_loop import _build_sketch_grid
        result = _build_sketch_grid([self._sketch(100, 100)], spacing=0)
        assert result.size == (100, 100)

    def test_single_non_square_padded_to_square(self):
        from run_sketch_to_shoe_loop import _build_sketch_grid
        result = _build_sketch_grid([self._sketch(100, 60)], spacing=0)
        w, h = result.size
        assert w == h, f"Expected square, got {w}x{h}"

    def test_two_sketches_output_is_square(self):
        from run_sketch_to_shoe_loop import _build_sketch_grid
        result = _build_sketch_grid([self._sketch(100, 100), self._sketch(100, 100)], spacing=0)
        w, h = result.size
        assert w == h

    def test_four_sketches_form_2x2_grid(self):
        from run_sketch_to_shoe_loop import _build_sketch_grid
        result = _build_sketch_grid([self._sketch(50, 50) for _ in range(4)], spacing=0)
        assert result.size == (100, 100)

    def test_padding_area_is_white(self):
        from run_sketch_to_shoe_loop import _build_sketch_grid
        result = _build_sketch_grid([self._sketch(50, 80, (0, 0, 0))], spacing=0)
        assert result.getpixel((0, 0)) == (255, 255, 255)

    def test_spacing_adds_white_gap(self):
        from run_sketch_to_shoe_loop import _build_sketch_grid
        sketches = [self._sketch(50, 50, (0, 0, 0)), self._sketch(50, 50, (0, 0, 0))]
        result = _build_sketch_grid(sketches, spacing=20)
        # Gap is at x=70 (spacing + 50 + spacing/2) in the pre-pad grid
        # Grid: [20 border][50 tile][20 gap][50 tile][20 border] = 160w x 90h (with borders)
        # After square pad: 160x160. Gap center at x=95, y=45
        pixel = result.getpixel((95, 45))
        assert pixel == (255, 255, 255)


class TestParseSpecArgs:
    def test_parses_key_value_pairs(self):
        from run_sketch_to_shoe_loop import _parse_spec_args
        result = _parse_spec_args(["style=elegant", "note=chunky sole"])
        assert result == {"style": "elegant", "note": "chunky sole"}

    def test_empty_list_returns_empty_dict(self):
        from run_sketch_to_shoe_loop import _parse_spec_args
        assert _parse_spec_args([]) == {}

    def test_ignores_entries_without_equals(self):
        from run_sketch_to_shoe_loop import _parse_spec_args
        result = _parse_spec_args(["valid=yes", "no_equals", "also=good"])
        assert "valid" in result and "also" in result
        assert "no_equals" not in result

    def test_value_with_equals_preserved(self):
        from run_sketch_to_shoe_loop import _parse_spec_args
        result = _parse_spec_args(["desc=a=b"])
        assert result["desc"] == "a=b"


class TestBuildExtraSpecsText:
    def test_formats_as_bullet_lines(self):
        from run_sketch_to_shoe_loop import _build_extra_specs_text
        result = _build_extra_specs_text({"style": "elegant", "note": "chunky"})
        assert "- Style: elegant" in result
        assert "- Note: chunky" in result

    def test_empty_dict_returns_empty_string(self):
        from run_sketch_to_shoe_loop import _build_extra_specs_text
        assert _build_extra_specs_text({}) == ""
```

**Step 2: Run new tests — they should FAIL (module not yet created)**

```bash
pytest tests/test_sketch_to_shoe.py::TestBuildSketchGrid tests/test_sketch_to_shoe.py::TestParseSpecArgs tests/test_sketch_to_shoe.py::TestBuildExtraSpecsText -v
```

Expected: ImportError / ModuleNotFoundError

**Step 3: Write `tests/run_sketch_to_shoe_loop.py`**

Create `tests/run_sketch_to_shoe_loop.py` with the full content below:

```python
"""Sketch-to-shoe agentic loop — integration runner.

Converts design sketches + spec (material, color, camera angle, extras) into
a photorealistic studio product photograph using FireRed + dual VLM judges.

Usage:
    python tests/run_sketch_to_shoe_loop.py --sketches tests/Image/sketch001.png
    python tests/run_sketch_to_shoe_loop.py \\
        --sketches s1.png s2.png \\
        --material suede --color beige --camera-angle "3/4 view" \\
        --spec style=elegant note="chunky platform"
    python tests/run_sketch_to_shoe_loop.py --max-iter 3 --steps 20
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image as PILImage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "workflows" / "sketch_to_shoe" / "scripts"))

from casadei import (
    Agent, AgentConfig, AgentStep, ImageMedia, TextMedia,
    LoggedPipeline, Pipeline,
)
from casadei.loop import LoopStep, LoopResult
from judge import (
    VLMSession,
    extract_sketch_features,
    make_sketch_judge,
    make_spec_judge,
    make_dual_judge,
    make_best_fn,
)

IMAGE_DIR = Path(__file__).parent / "Image"
OUTPUT_DIR = Path(__file__).parent / "output" / "sketch_to_shoe_loop"

PROMPT_TEMPLATE = (
    "The image shows a shoe design sketch. Convert it into a professional, "
    "photorealistic product photograph of the final shoe.\n\n"
    "Design specifications:\n"
    "- Material: $material\n"
    "- Color: $color\n"
    "- Camera angle: $camera_angle\n"
    "$extra_specs\n\n"
    "The result must be a studio-quality photograph: clean white background, "
    "professional product lighting, sharp focus, no shadows on background, "
    "shoe centered and fully visible. $feedback"
)


# ---------------------------------------------------------------------------
# Sketch grid assembly
# ---------------------------------------------------------------------------

def _build_sketch_grid(
    images: list[PILImage.Image],
    spacing: int = 20,
) -> PILImage.Image:
    """Arrange sketch images in a rectangular grid, then pad to square.

    1. Compute grid dimensions: cols = ceil(sqrt(n)), rows = ceil(n / cols).
    2. Common cell size = max width x max height across all inputs.
    3. Paste tiles into grid with `spacing` px white gaps (also used as outer border).
    4. If resulting grid is not square, center on a white square canvas.
    """
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


# ---------------------------------------------------------------------------
# Spec utilities
# ---------------------------------------------------------------------------

def _parse_spec_args(spec_list: list[str]) -> dict[str, str]:
    """Parse KEY=VALUE strings from --spec CLI args. Ignores entries without '='."""
    result = {}
    for item in spec_list:
        if "=" in item:
            key, _, value = item.partition("=")
            result[key.strip()] = value.strip()
    return result


def _build_extra_specs_text(extra: dict[str, str]) -> str:
    """Format extra spec dict as bullet lines for the prompt."""
    if not extra:
        return ""
    return "\n".join(f"- {k.capitalize()}: {v}" for k, v in extra.items())


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

def build_pipeline(
    sketch_media: ImageMedia,
    spec: dict,
    max_iterations: int = 5,
    num_inference_steps: int = 30,
    swap_models: bool = True,
    vlm_session: VLMSession | None = None,
    sketch_features: list[str] | None = None,
    tolerance: str = "strict",
) -> Pipeline:
    """Build the iterative sketch-to-shoe pipeline."""
    if vlm_session is None:
        vlm_session = VLMSession("qwen3_vl_8b")

    extra_specs_text = _build_extra_specs_text(spec.get("extra", {}))

    firered_agent = Agent(AgentConfig(
        name="firered_sketch_to_shoe",
        model="firered_image_edit",
        description="FireRed sketch-to-shoe with dual-judge feedback repair",
        prompt_template=PROMPT_TEMPLATE,
        negative_prompt=(
            "blurry, distorted, low quality, sketch, drawing, illustration, flat, cartoon, "
            "dark background, cluttered background, bad lighting, overexposed, underexposed"
        ),
        params={"num_inference_steps": num_inference_steps},
    ))

    firered_step = AgentStep(
        name="firered_generate",
        agent=firered_agent,
        input_map={
            "image": "sketch",    # IMAGE 1: sketch grid (constant reference)
            "image_2": "image",   # IMAGE 2: last result (first iter: seeded with sketch)
        },
        output_map={"image": "image"},
        template_kwargs={
            "material": spec.get("material", "leather"),
            "color": spec.get("color", "black"),
            "camera_angle": spec.get("camera_angle", "3/4 view"),
            "extra_specs": extra_specs_text,
            "feedback": "",
        },
    )

    full_spec = {
        "material": spec.get("material", "leather"),
        "color": spec.get("color", "black"),
        "camera_angle": spec.get("camera_angle", "3/4 view"),
        **spec.get("extra", {}),
    }

    dual_judge = make_dual_judge(
        make_sketch_judge(
            session=vlm_session,
            sketch_key="sketch",
            candidate_key="image",
            features=sketch_features,
            tolerance=tolerance,
        ),
        make_spec_judge(
            session=vlm_session,
            candidate_key="image",
            spec=full_spec,
            tolerance=tolerance,
        ),
    )

    loop = LoopStep(
        name="sketch_to_shoe_loop",
        body=[firered_step],
        judge=dual_judge,
        max_iterations=max_iterations,
        best_fn=make_best_fn(
            session=vlm_session,
            sketch_key="sketch",
            output_key="image",
        ),
        swap_models=swap_models,
        output_key="image",
        feedback_template_var="feedback",
    )

    return Pipeline(name="sketch_to_shoe", steps=[loop])


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def save_results(
    run_dir: Path,
    loop_result: LoopResult,
    result_context: dict,
    sketch_grid: PILImage.Image,
    final_img: ImageMedia | None,
    spec: dict,
    total_elapsed: float,
    peak_gb: float,
    sketch_features: list[str] | None = None,
    tolerance: str = "strict",
) -> None:
    """Save all results, intermediates, and metrics to run_dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    sketch_grid.save(run_dir / "input_sketch_grid.png")

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "total_elapsed_s": total_elapsed,
        "peak_vram_gb": peak_gb,
        "spec": spec,
        "sketch_features": sketch_features or [],
        "tolerance": tolerance,
        "total_iterations": len(loop_result.iterations),
        "iterations": [],
    }

    for it in loop_result.iterations:
        iter_data = {
            "index": it.index,
            "accepted": it.accepted,
            "feedback": it.feedback,
            "duration_ms": it.duration_ms,
        }
        if it.metadata:
            iter_data["sketch_scores"] = it.metadata.get("sketch_scores", {})
            iter_data["sketch_avg"] = it.metadata.get("sketch_avg")
            iter_data["spec_scores"] = it.metadata.get("spec_scores", {})
            iter_data["spec_avg"] = it.metadata.get("spec_avg")

        candidate_img = it.outputs.get("image")
        if isinstance(candidate_img, ImageMedia):
            img_path = run_dir / f"iter_{it.index:02d}_candidate.png"
            candidate_img.image.save(img_path)
            iter_data["image_path"] = str(img_path.name)
        results_data["iterations"].append(iter_data)

    if final_img is not None and isinstance(final_img, ImageMedia):
        final_img.image.save(run_dir / "final_result.png")
        results_data["final_result"] = "final_result.png"

    if loop_result.iterations:
        last = loop_result.iterations[-1]
        if last.accepted:
            results_data["final_verdict"] = f"accepted_at_iteration_{last.index}"
        else:
            results_data["final_verdict"] = "max_reached_best_selected"
            best_idx = result_context.get("best_selection_index")
            best_reason = result_context.get("best_selection_reason")
            if best_idx is not None:
                results_data["best_selected_candidate"] = best_idx
            if isinstance(best_reason, TextMedia):
                results_data["best_selection_vlm_response"] = best_reason.text

    (run_dir / "results.json").write_text(
        json.dumps(results_data, indent=2, default=str)
    )

    lines = [
        "Sketch-to-Shoe Loop Results",
        "=" * 60,
        f"Date: {datetime.now().isoformat()}",
        f"Total time: {total_elapsed:.1f}s",
        f"Peak VRAM: {peak_gb:.2f} GB",
        f"Material: {spec.get('material')}  Color: {spec.get('color')}  Angle: {spec.get('camera_angle')}",
        f"Sketch features: {sketch_features or []}",
        f"Tolerance: {tolerance}",
        f"Iterations: {len(loop_result.iterations)}",
        "",
    ]
    for it in loop_result.iterations:
        verdict = "ACCEPT" if it.accepted else "REJECT"
        lines.append(f"  Iteration {it.index}: {verdict} ({it.duration_ms:.1f}ms)")
        if it.metadata:
            if it.metadata.get("sketch_scores"):
                ss = it.metadata["sketch_scores"]
                lines.append(f"    Sketch: {', '.join(f'{k}={v}' for k,v in ss.items())} "
                              f"(avg={it.metadata.get('sketch_avg')})")
            if it.metadata.get("spec_scores"):
                ss = it.metadata["spec_scores"]
                lines.append(f"    Spec:   {', '.join(f'{k}={v}' for k,v in ss.items())} "
                              f"(avg={it.metadata.get('spec_avg')})")
        lines.append(f"    Feedback: {it.feedback[:200]}")
        lines.append("")

    lines.append(f"Final verdict: {results_data.get('final_verdict', 'unknown')}")
    lines.append(f"Output: {run_dir}")
    summary = "\n".join(lines)
    print(f"\n{summary}")
    (run_dir / "summary.txt").write_text(summary)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sketch-to-shoe agentic loop")
    parser.add_argument("--sketches", type=str, nargs="+", required=True,
        help="Path(s) to sketch image(s)")
    parser.add_argument("--material", type=str, default="leather",
        help="Shoe material (e.g. leather, suede, canvas)")
    parser.add_argument("--color", type=str, default="black",
        help="Shoe color (e.g. black, white, red)")
    parser.add_argument("--camera-angle", type=str, default="3/4 view",
        dest="camera_angle",
        help="Camera angle: '3/4 view', 'side view', 'front view', 'top view', or custom")
    parser.add_argument("--spec", type=str, nargs="*", default=[],
        metavar="KEY=VALUE",
        help="Open-ended extras, e.g. style=elegant note='chunky sole'")
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--keep-both", action="store_true",
        help="Keep all models loaded (needs ~74GB+ VRAM)")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--tolerance", type=str, default="strict",
        choices=["generous", "moderate", "strict"])
    parser.add_argument("--spacing", type=int, default=20,
        help="Pixel spacing between sketches in grid (default: 20)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        return

    print("=== Sketch-to-Shoe Agentic Loop ===")
    print(f"Sketches: {args.sketches}")
    print(f"Material: {args.material}")
    print(f"Color: {args.color}")
    print(f"Camera angle: {args.camera_angle}")
    extra_spec = _parse_spec_args(args.spec)
    if extra_spec:
        print(f"Extra spec: {extra_spec}")
    print(f"Max iterations: {args.max_iter}")
    print(f"Inference steps: {args.steps}")
    print(f"Memory mode: {'keep-both' if args.keep_both else 'swap'}")
    print(f"Scale: {args.scale}x")
    print(f"Tolerance: {args.tolerance}")
    print()

    raw_sketches = []
    for path in args.sketches:
        img = PILImage.open(path).convert("RGB")
        if args.scale != 1.0:
            img = img.resize(
                (int(img.width * args.scale), int(img.height * args.scale)),
                PILImage.LANCZOS,
            )
        raw_sketches.append(img)

    sketch_grid = _build_sketch_grid(raw_sketches, spacing=args.spacing)
    print(f"Sketch grid: {sketch_grid.size[0]}x{sketch_grid.size[1]} px "
          f"({len(raw_sketches)} sketch(es))")

    spec = {
        "material": args.material,
        "color": args.color,
        "camera_angle": args.camera_angle,
        "extra": extra_spec,
    }

    vlm_session = VLMSession("qwen3_vl_8b")
    sketch_media = ImageMedia(image=sketch_grid)

    print("Extracting sketch design features...")
    sketch_features = extract_sketch_features(vlm_session, sketch_media)
    print(f"Features: {sketch_features}")
    print()

    pipeline = build_pipeline(
        sketch_media=sketch_media,
        spec=spec,
        max_iterations=args.max_iter,
        num_inference_steps=args.steps,
        swap_models=not args.keep_both,
        vlm_session=vlm_session,
        sketch_features=sketch_features,
        tolerance=args.tolerance,
    )
    logged = LoggedPipeline(pipeline)

    context = {
        "sketch": sketch_media,
        "image": sketch_media,  # seed for first iteration
    }

    if args.keep_both:
        print("Loading all models (keep-both mode)...")
        vlm_session.load()
        pipeline.load()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    try:
        result, exec_log = logged.run(context)
    finally:
        vlm_session.unload()

    torch.cuda.synchronize()
    total_elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

    print(exec_log.summary())

    loop_result = result.get("sketch_to_shoe_loop_history")
    final_img = result.get("image")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"{args.max_iter}iter_{args.steps}steps_{ts}"

    save_results(
        run_dir=run_dir,
        loop_result=loop_result if isinstance(loop_result, LoopResult) else LoopResult(),
        result_context=result,
        sketch_grid=sketch_grid,
        final_img=final_img,
        spec=spec,
        total_elapsed=total_elapsed,
        peak_gb=peak_gb,
        sketch_features=sketch_features,
        tolerance=args.tolerance,
    )

    gc.collect()
    torch.cuda.empty_cache()
    print(f"\nDone. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
```

**Step 4: Run all tests**

```bash
pytest tests/test_sketch_to_shoe.py -v
```

Expected: All tests PASS

**Step 5: Verify imports work without GPU**

```bash
python -c "
import sys
sys.path.insert(0, 'tests')
from run_sketch_to_shoe_loop import _build_sketch_grid, _parse_spec_args, _build_extra_specs_text
print('run_sketch_to_shoe_loop.py imports OK')
"
```

```bash
python -c "
import sys
sys.path.insert(0, 'workflows/sketch_to_shoe/scripts')
from judge import VLMSession, make_sketch_judge, make_spec_judge, make_dual_judge, make_best_fn, extract_sketch_features
print('judge.py imports OK')
"
```

Expected: Both print `... imports OK`

**Step 6: Run full test suite to confirm no regressions**

```bash
pytest tests/test_sketch_to_shoe.py tests/test_loop_step.py -v
```

Expected: All tests PASS

**Step 7: Commit**

```bash
git add tests/run_sketch_to_shoe_loop.py tests/test_sketch_to_shoe.py
git commit -m "feat(sketch-to-shoe): add runner script and full unit test suite"
```

---

### Final summary

**Files created:**

| File | Purpose |
|------|---------|
| `workflows/sketch_to_shoe/scripts/__init__.py` | Package marker |
| `workflows/sketch_to_shoe/scripts/judge.py` | VLMSession, both judges, dual combiner, best_fn |
| `tests/run_sketch_to_shoe_loop.py` | CLI runner: grid assembly, spec parsing, pipeline, saving |
| `tests/test_sketch_to_shoe.py` | Unit tests for all pure functions (no GPU) |

**Example run:**

```bash
python tests/run_sketch_to_shoe_loop.py \
    --sketches tests/Image/sketch_front.png tests/Image/sketch_side.png \
    --material suede \
    --color "off-white" \
    --camera-angle "3/4 view" \
    --spec style=casual note="thick gum sole" \
    --max-iter 5 \
    --steps 30 \
    --tolerance strict
```
