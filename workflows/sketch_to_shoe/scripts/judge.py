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
    # Detect if VLM echoed back the placeholder instead of filling in numbers
    if re.search(r"=\[1-5\]", scores_text) or re.search(r"=N\b", scores_text):
        raise ValueError(
            "Scores contain unfilled placeholders ([1-5] or N). "
            "Replace each placeholder with an actual integer 1-5."
        )
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
You are looking at a shoe design sketch. Extract the key structural and \
design elements that another agent will use to score whether a generated \
photo faithfully matches this sketch.

Each tag must be a specific, visually verifiable design element — not a \
general category. For example, "pointed stiletto toe" is verifiable; \
"toe shape" is not. Do NOT include any numbers or measurements \
(no centimeters, no estimates like "~10cm") — use descriptive terms only \
(e.g. "thin", "high", "chunky", "thick"). Focus on elements that are \
clearly drawn in the sketch:
- Silhouette / shoe type  (e.g. "ankle boot", "slingback pump", "platform sneaker")
- Toe shape  (e.g. "extreme pointed toe", "open square toe", "rounded almond toe")
- Heel  (e.g. "thin stiletto heel", "chunky block heel", "wedge heel", "flat")
- Sole / platform  (e.g. "thick platform sole", "thin leather sole", "lug sole")
- Ankle height  (e.g. "above-ankle cut", "low-cut", "knee-high shaft")
- Closure / straps  (e.g. "ankle strap with buckle", "side zip", "lace-up front", "slip-on")
- Notable structural details  (e.g. "cut-out side panel", "oversized buckle", "wrapped heel")

Reply with ONLY a JSON array of 5–8 descriptor strings. \
No explanation, no markdown — just the array.

Example:
["pointed stiletto pump", "open peep-toe", "thin stiletto heel", \
"slim platform sole", "ankle strap with gold buckle", "patent leather upper"]
"""

_SKETCH_SCORE_PROMPT = """\
You are a design fidelity inspector for a shoe manufacturing system.

The first image is the ORIGINAL SKETCH — the designer's intended shoe design.
The second image is the GENERATED PHOTO — {result_description}

Your job: compare the shoe in the generated photo to the design in the original \
sketch and score how closely the generated photo matches the sketch's design intent.

{iteration_context}\
{stale_nudge}\
Score each attribute from 1 to 5:
  1 = completely different from the sketch
  2 = vaguely similar but clearly wrong
  3 = recognizably the same design but noticeable differences
  4 = close match with minor differences
  5 = near-identical to the sketch design

Attributes to score: {features}

Reply in this exact format. Replace each [1-5] with your actual integer score:
SCORES: {score_format}
REPAIR: <Look at the CURRENT image. Do NOT copy the previous issues verbatim. \
For each attribute scored below 5, describe what you actually see NOW vs. the sketch. \
If a previously-identified issue has been RESOLVED in this image, say so explicitly. \
If it persists, describe the exact mismatch you see in the current image, \
then tell the model to look at the sketch and correct that specific element.>

Example of a correctly filled response (scores are illustrative):
SCORES: {example_format}
REPAIR: The toe shape in the generated photo is rounded but the sketch shows a pointed toe ...
"""

_SPEC_SCORE_PROMPT = """\
You are a product specification inspector for a shoe manufacturing system.

The image shows a generated photorealistic shoe product photo — {result_description}

Your job: check whether the shoe in the image matches the following \
design specifications, and evaluate the overall photo quality.

Design specifications:
{spec_text}

{iteration_context}\
{stale_nudge}\
Score each attribute from 1 to 5:
  1 = completely fails this specification
  2 = vaguely meets it but clearly wrong
  3 = partially meets it with noticeable issues
  4 = mostly meets it with only minor issues
  5 = fully and clearly meets this specification

Attributes to score: {features}

Reply in this exact format. Replace each [1-5] with your actual integer score:
SCORES: {score_format}
REPAIR: <Look at the CURRENT image. Do NOT copy the previous issues verbatim. \
For each attribute scored below 5, describe what you actually see NOW vs. the specification. \
If a previously-identified issue has been RESOLVED in this image, say so explicitly. \
If it persists, describe exactly what is still wrong and tell the model what to change.>

Example of a correctly filled response (scores are illustrative):
SCORES: {example_format}
REPAIR: The material appears to be canvas rather than the specified leather ...
"""

_BEST_PROMPT = """\
You are selecting the best sketch-to-shoe rendering result.

The first image is the ORIGINAL SKETCH — the designer's intended shoe design.
The second image shows {n} candidate photorealistic renderings side by side, \
labeled Option 1 through Option {n} from left to right.

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
    score_format = ", ".join(f"{f}=[1-5]" for f in features)
    example_format = ", ".join(f"{f}=3" for f in features)

    def judge(context: dict[str, Media]) -> tuple[bool, str]:
        sketch = context.get(sketch_key)
        candidate = context.get(candidate_key)

        if not isinstance(sketch, ImageMedia):
            return False, f"Missing sketch image (key='{sketch_key}')."
        if not isinstance(candidate, ImageMedia):
            return False, f"Missing candidate image (key='{candidate_key}')."

        iteration = context.get("loop_iteration", 0)
        max_iterations = context.get("loop_max_iterations", 5)

        if iteration == 0:
            result_description = (
                "a first attempt at converting the sketch into a photorealistic shoe photo."
            )
            iteration_context = (
                f"This is the first rendering (1 of {max_iterations}). "
                f"Check how well the generated photo captures the sketch's design.\n"
            )
        else:
            result_description = (
                "a refined attempt that should more closely follow the sketch than before."
            )
            iteration_context = (
                f"This is refinement attempt {iteration + 1} of {max_iterations}. "
                f"Check whether this iteration improved fidelity to the sketch.\n"
            )

        stale_nudge = ""
        if stale_count[0] >= 2 and prev_lowest_attr[0]:
            attr = prev_lowest_attr[0]
            stale_nudge = (
                f"The attribute '{attr}' has been the weakest for {stale_count[0]} consecutive "
                f"attempts. In your REPAIR instruction, specifically describe what is wrong "
                f"with '{attr}' in the generated photo compared to the sketch.\n"
            )

        prompt_text = _SKETCH_SCORE_PROMPT.format(
            result_description=result_description,
            iteration_context=iteration_context,
            stale_nudge=stale_nudge,
            features=", ".join(features),
            score_format=score_format,
            example_format=example_format,
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
                        f"{prompt_text}\n\n"
                        f"Your previous response could not be parsed ({last_error}).\n"
                        f"You MUST replace every [1-5] with an actual integer. "
                        f"Do not output [1-5] literally.\n"
                        f"Correct example:\n"
                        f"SCORES: {example_format}\n"
                        f"REPAIR: ..."
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
            return False, raw_response or "Could not parse VLM response."
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
    score_format = ", ".join(f"{f}=[1-5]" for f in features)
    example_format = ", ".join(f"{f}=3" for f in features)

    def judge(context: dict[str, Media]) -> tuple[bool, str]:
        candidate = context.get(candidate_key)

        if not isinstance(candidate, ImageMedia):
            return False, f"Missing candidate image (key='{candidate_key}')."

        iteration = context.get("loop_iteration", 0)
        max_iterations = context.get("loop_max_iterations", 5)

        if iteration == 0:
            result_description = "a first attempt at generating the shoe from the design spec."
            iteration_context = (
                f"This is the first rendering (1 of {max_iterations}). "
                f"Check how well the generated photo meets each specification.\n"
            )
        else:
            result_description = (
                "a refined attempt that should better meet the specifications than before."
            )
            iteration_context = (
                f"This is refinement attempt {iteration + 1} of {max_iterations}. "
                f"Check whether this iteration improved spec compliance.\n"
            )

        stale_nudge = ""
        if stale_count[0] >= 2 and prev_lowest_attr[0]:
            attr = prev_lowest_attr[0]
            stale_nudge = (
                f"The attribute '{attr}' has been below standard for {stale_count[0]} consecutive "
                f"attempts. Focus your REPAIR instruction specifically on '{attr}'.\n"
            )

        prompt_text = _SPEC_SCORE_PROMPT.format(
            result_description=result_description,
            spec_text=spec_text,
            iteration_context=iteration_context,
            stale_nudge=stale_nudge,
            features=", ".join(features),
            score_format=score_format,
            example_format=example_format,
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
                        f"{prompt_text}\n\n"
                        f"Your previous response could not be parsed ({last_error}).\n"
                        f"You MUST replace every [1-5] with an actual integer. "
                        f"Do not output [1-5] literally.\n"
                        f"Correct example:\n"
                        f"SCORES: {example_format}\n"
                        f"REPAIR: ..."
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
            return False, raw_response or "Could not parse VLM response."
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

        iteration = context.get("loop_iteration", 0)
        if iteration == 0:
            framing = (
                "The shoe was generated but does not yet fully match the sketch and "
                "specifications. Refine it — do not start over from scratch. "
            )
        else:
            framing = (
                "The previous refinement still does not fully meet the design requirements. "
                "Continue improving the existing rendering. "
            )

        combined_feedback = (
            f"{framing}"
            f"[Sketch feedback]: {feedback1} "
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
