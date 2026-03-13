"""Judge functions for the sketch-to-shoe agentic loop.

Two independent judges run per iteration:
  1. Sketch Fidelity Judge  — IMAGE1=sketch, IMAGE2=generated → scores design faithfulness
  2. Spec Compliance Judge  — IMAGE=generated, TEXT=spec → scores material/color/angle/quality

make_dual_judge() wraps both into one JudgeCallable for LoopStep.
"""

from __future__ import annotations

import gc
import hashlib
import io
import sys
import time
from typing import Dict, List
from pydantic import BaseModel, Field
from PIL import Image as PILImage, ImageDraw, ImageFont

from casadei.loop import BestFn, JudgeCallable, LoopIteration
from casadei.media import ImageMedia, Media, TextMedia, MediaBundle

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOLERANCE_CONFIGS = {
    "generous":  {"avg_threshold": 3.0, "min_floor": 2.0},
    "moderate":  {"avg_threshold": 3.5, "min_floor": 2.5},
    "strict":    {"avg_threshold": 4.0, "min_floor": 3.5},
}

_MAX_RETRIES = 3
_FALLBACK_SKETCH_FEATURES = ["shape", "proportions", "toe_shape", "heel_style", "sole_design"]

# ---------------------------------------------------------------------------
# Image identity helper
# ---------------------------------------------------------------------------

def _image_hash(img: ImageMedia) -> str:
    """MD5 hash of raw PNG bytes — fast identity check for unchanged images."""
    buf = io.BytesIO()
    img.image.save(buf, format="PNG")
    return hashlib.md5(buf.getvalue()).hexdigest()

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


def _stream_vlm(model, bundle: MediaBundle, label: str = "VLM",
                max_api_retries: int = 3, retry_delay: float = 10.0) -> str:
    for api_attempt in range(max_api_retries + 1):
        try:
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
        except Exception as e:
            err_str = str(e)
            is_transient = any(code in err_str for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
            if is_transient and api_attempt < max_api_retries:
                wait = retry_delay * (api_attempt + 1)
                print(f"\n  [{label}] API error ({err_str[:80]}), retrying in {wait:.0f}s "
                      f"(attempt {api_attempt + 1}/{max_api_retries})...")
                time.sleep(wait)
            else:
                raise


def _call_vlm_structured(
    model,
    bundle: MediaBundle,
    schema: dict,
    label: str = "VLM",
    max_api_retries: int = 3,
    retry_delay: float = 10.0,
) -> str:
    """Call the VLM with JSON-schema-enforced structured output.

    Returns the raw JSON string. The caller is responsible for parsing it
    with the appropriate Pydantic model.
    """
    for api_attempt in range(max_api_retries + 1):
        try:
            sys.stdout.write(f"\n  [{label}] calling... ")
            sys.stdout.flush()
            result = model.run(
                bundle,
                response_mime_type="application/json",
                response_json_schema=schema,
            )
            text = result.items["text"].text if "text" in result.items else ""
            sys.stdout.write("done\n")
            sys.stdout.flush()
            if text:
                print(f"  [{label}] {text}")
            return text
        except Exception as e:
            err_str = str(e)
            is_transient = any(code in err_str for code in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED"))
            if is_transient and api_attempt < max_api_retries:
                wait = retry_delay * (api_attempt + 1)
                print(f"\n  [{label}] API error ({err_str[:80]}), retrying in {wait:.0f}s "
                      f"(attempt {api_attempt + 1}/{max_api_retries})...")
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# VLMSession (identical lifecycle to shoe_tryon_loop)
# ---------------------------------------------------------------------------

class VLMSession:
    """Shared VLM model lifecycle. Swap mode by default; persistent if load() called."""

    def __init__(self, model_name: str = "qwen3_vl_8b"):
        self._model_name = model_name
        self._model = None
        self._persistent = False
        self.token_usage_log: list[dict] = []

    def record_usage(self, label: str = "") -> None:
        """Append the model's last_token_usage (if any) to the log."""
        model = self._model
        if model is None:
            return
        usage = getattr(model, "last_token_usage", None)
        if usage:
            self.token_usage_log.append({
                "label": label,
                "model": getattr(model, "MODEL_ID", self._model_name),
                **usage,
            })

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
# Judge output schemas (structured output via Gemini JSON schema)
# ---------------------------------------------------------------------------

class _SketchJudgeResult(BaseModel):
    holistic: str = Field(
        description="Key structural differences in one sentence, or 'none' if the structures match"
    )
    sketch_observations: str = Field(
        description="Summary of every structural element seen in the REFERENCE SKETCH"
    )
    photo_observations: str = Field(
        description=(
            "Summary of the same structural elements in the GENERATED PHOTO — "
            "describe only what is visually present, no assumptions or inferences"
        )
    )
    score: int = Field(
        description=(
            "Holistic structural match score: "
            "1=fundamentally different, 2=major differences, 3=one clear mismatch, "
            "4=matches with minor variation, 5=precise match"
        ),
        ge=1, le=5,
    )
    repair: str = Field(
        description=(
            "For score 3 or below: per-mismatch fix with sketch region reference. "
            "For score 4-5: 'none'"
        )
    )


class _ReferenceFidelityResult(BaseModel):
    reference_observations: str = Field(
        description="Key design attributes of the REFERENCE SHOE: material, color, texture, heel, toe, straps"
    )
    generated_observations: str = Field(
        description="Same attributes observed in the GENERATED SHOE — describe only what is visually present"
    )
    score: int = Field(
        description=(
            "Design fidelity score: "
            "1=completely different shoe, 2=major discrepancies in key elements, "
            "3=some clear visible differences, 4=mostly faithful with minor variation, 5=identical"
        ),
        ge=1, le=5,
    )
    repair: str = Field(
        description="For score 3 or below: per-element discrepancy + concrete fix instruction. For score 4-5: 'none'"
    )


class _SpecJudgeResult(BaseModel):
    observations: Dict[str, str] = Field(
        description="Per-attribute factual observation — key is attribute name, value is one sentence of what you see"
    )
    scores: Dict[str, int] = Field(
        description="Per-attribute score 1-5 — key is attribute name, value is integer 1-5"
    )
    repair: str = Field(
        description="For attributes scored 3 or below: flaw + fix instruction per attribute. Otherwise 'none'"
    )


class _ShoeCountResult(BaseModel):
    observation: str = Field(
        description="State exactly how many shoes are visible in the image."
    )
    correct: int = Field(
        description=(
            "1 if the image shows exactly the required number of shoes, "
            "0 if the count is wrong (too many or too few)."
        ),
        ge=0, le=1,
    )
    repair: str = Field(
        description=(
            "If correct=0: state how many shoes are visible and give a concrete instruction "
            "(e.g. 'two shoes shown instead of one — generate only a single shoe'). "
            "If correct=1: 'none'."
        )
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SKETCH_FEATURE_PROMPT = """\
You are looking at a design sketch. Extract its key STRUCTURAL features \
as a flat list of short, positive descriptions.

RULES:
- STRUCTURE ONLY — no materials, colors, or finishes.
- Only describe what is VISUALLY PRESENT in the sketch. \
  Never assume a feature exists because it is typical for this design type. \
  Look at what is actually drawn, line by line.
- Use positive language for EVERY feature — whether a part is solid/enclosed \
  OR open/exposed. Describe what you actually see: if a region is visibly open \
  or exposed, name that openness as a feature. If a part is visibly enclosed or \
  covered, name that closure. Never write "no X" — always describe what IS there.
- Short, concrete, unambiguous descriptors. 4–8 features total.

Reply with ONLY valid JSON — no explanation, no markdown:
{"features": [...]}
"""

_SKETCH_SCORE_PROMPT = """\
You are a structural inspector for a luxury shoe design studio. \
Your job is to verify that a generated photo matches the SHAPE and CONSTRUCTION of the sketch. \
Score STRUCTURE ONLY — not materials, colors, or finishes (those are judged separately). \
Only penalize differences in shape, size, presence, or absence of structural elements.

REFERENCE SKETCH — the designer's ground truth.
GENERATED PHOTO — {result_description}

Follow these steps in order. Do not skip any step.

STEP 1 — REFERENCE SKETCH OBSERVATION
Look carefully at the REFERENCE SKETCH only. List every visible structural element: \
strap positions, heel type, toe shape, open/closed areas, sole profile, \
and any exposed or enclosed sections. Do not look at the GENERATED PHOTO yet.

STEP 2 — GENERATED PHOTO OBSERVATION
Look carefully at the GENERATED PHOTO only. For each structural element listed in STEP 1, \
describe exactly what you see — do not assume or infer. If a feature is ambiguous or partially \
visible, say so. Do not describe what you expect to see based on the sketch — only \
describe what is visually present in the GENERATED PHOTO.

STEP 3 — SCORE
Compare your STEP 1 and STEP 2 observations and give a single holistic score \
for how well the rendering matches the sketch construction:
  1 = fundamentally different structure — multiple major elements wrong or missing
  2 = significant structural differences — one or more key elements clearly wrong
  3 = mostly correct with one clear, visible structural difference
  4 = matches the sketch — at most a minor geometric variation
  5 = matches the sketch construction precisely

Do NOT reduce the score for material or color differences. \
Do NOT inflate the score because you expect a feature to be there — only score 4 or 5 if you \
can clearly see in the GENERATED PHOTO that the structural element matches the REFERENCE SKETCH. \
When in doubt, score lower.

STEP 4 — REPAIR
If score is 3 or below: for each structural mismatch, (a) name the specific region or element \
that is wrong, (b) give a concrete fix instruction (shape, placement, or presence/absence), and \
(c) explicitly direct the generator to re-examine that exact region of the REFERENCE SKETCH and \
implement it faithfully as drawn there. Do NOT give material or color instructions. \
If score is 4 or 5, write "none".
"""

_SPEC_SCORE_PROMPT = """\
You are a quality inspector for a luxury shoe manufacturing studio. \
Your job is to verify that a generated shoe photo meets the design specification. \
Be precise and fair — neither inflate scores for elements that are clearly wrong, \
nor penalize elements for minor stylistic or rendering variations that preserve the design intent. \
Give the benefit of the doubt when the spec is substantially met.

The image shows a generated photorealistic shoe product photo — {result_description}

Design specifications:
{spec_text}

{judge_notes_section}\
{iteration_context}\
{stale_nudge}\
Follow these three steps in order. Do not skip steps.

STEP 1 — OBSERVE (write before scoring)
For each attribute below, write one factual sentence describing exactly what you see \
in the image. Name the actual material, color, finish, and shape you observe. \
Do not score yet.

Attributes: {features}

STEP 2 — SCORE
Based only on your observations above, score each attribute 1–5 against the spec:
  1 = completely absent or opposite of the specification
  2 = fundamentally wrong — clearly a different material class, color family, or structural type
  3 = partially correct but with a clear, visible discrepancy
  4 = clearly meets the specification — at most a minor stylistic variation
  5 = fully and precisely meets the specification

When in doubt between 3 and 4, choose 4 if the spec intent is substantially met. \
A score may NOT be higher than what your Step 1 observation supports. \
If your observation describes a genuine problem with the core spec requirement, the score must be 3 or below.

STEP 3 — REPAIR
For each attribute scored 3 or below: state the flaw in one sentence, \
then give a concrete generation instruction (material, color, texture, shape, placement). \
Skip attributes scored 4 or 5. If all pass, write "none".

Use the attribute names exactly as listed: {features}
"""


_REFERENCE_FIDELITY_PROMPT = """\
The image shows two panels side by side.

LEFT panel labeled "REFERENCE SHOE": the original photorealistic shoe (top) and its design sketch (bottom).
RIGHT panel labeled "GENERATED SHOE": the newly generated shoe at a different camera angle.

Your task: verify that the GENERATED SHOE faithfully reproduces the same shoe as the REFERENCE SHOE.
Compare design fidelity only: materials, colors, textures, straps, closures, toe shape, heel style, sole design.
Ignore differences in camera angle, lighting direction, shoe orientation, and staging.

STEP 1 — OBSERVE
Describe the key design attributes of the REFERENCE SHOE (material, color, texture, heel, toe, straps).
Then describe the same attributes in the GENERATED SHOE — only what is visually present, no assumptions.

STEP 2 — SCORE
Rate how faithfully the GENERATED SHOE reproduces the REFERENCE SHOE:
  1 = completely different shoe — wrong material class or fundamentally different design
  2 = major discrepancies — key design elements are clearly wrong
  3 = partially correct — some visible differences in color, material, or design
  4 = mostly faithful — at most a minor stylistic variation
  5 = identical in all design aspects

STEP 3 — REPAIR
For score 3 or below: list each discrepancy and give a concrete fix instruction so the generator can correct it.
For score 4-5: write "none".
"""


def _make_comparison_image(
    reference: PILImage.Image,
    sketch: PILImage.Image,
    candidate: PILImage.Image,
    panel_width: int = 512,
    label_height: int = 40,
    padding: int = 10,
    gap: int = 6,
) -> PILImage.Image:
    """Build a 2-panel side-by-side image.

    Left panel: reference photo (top) + sketch (bottom), labeled 'REFERENCE SHOE'.
    Right panel: candidate, labeled 'GENERATED SHOE'.
    """
    def _fit(img: PILImage.Image, width: int) -> PILImage.Image:
        scale = width / img.width
        return img.resize((width, int(img.height * scale)), PILImage.LANCZOS)

    ref = _fit(reference.convert("RGB"), panel_width)
    sk  = _fit(sketch.convert("RGB"),    panel_width)
    cand = _fit(candidate.convert("RGB"), panel_width)

    left_h  = ref.height + gap + sk.height
    right_h = cand.height
    content_h = max(left_h, right_h)
    panel_h = label_height + content_h + padding * 2
    total_w = (panel_width + padding * 2) * 2 + gap

    composite = PILImage.new("RGB", (total_w, panel_h), (220, 220, 220))
    draw = ImageDraw.Draw(composite)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    # Left panel
    lx = 0
    draw.rectangle([lx, 0, lx + panel_width + padding * 2, panel_h], fill=(255, 255, 255))
    draw.text((lx + padding, padding // 2 + (label_height - 24) // 2), "REFERENCE SHOE",
              fill=(20, 20, 20), font=font)
    composite.paste(ref,  (lx + padding, label_height + padding))
    composite.paste(sk,   (lx + padding, label_height + padding + ref.height + gap))

    # Right panel
    rx = panel_width + padding * 2 + gap
    draw.rectangle([rx, 0, rx + panel_width + padding * 2, panel_h], fill=(255, 255, 255))
    draw.text((rx + padding, padding // 2 + (label_height - 24) // 2), "GENERATED SHOE",
              fill=(20, 20, 20), font=font)
    composite.paste(cand, (rx + padding, label_height + padding))

    return composite


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class _SketchFeatures(BaseModel):
    features: List[str] = Field(
        description=(
            "Structural features of the shoe sketch as positive descriptions. "
            "Open/exposed areas use positive language ('open back', 'open toe'). "
            "No materials or colors."
        )
    )


def extract_sketch_features(session: VLMSession, sketch_image: ImageMedia) -> list[str]:
    """Ask VLM to identify visual design attributes from the sketch.

    Uses Gemini structured output (response_json_schema) to get a typed
    _SketchFeatures response. Features are a flat list of positive descriptions
    including open/exposed areas (e.g. "open back", "open toe").

    Falls back to _FALLBACK_SKETCH_FEATURES on parse failure.
    """
    bundle = MediaBundle(items={
        "sketch": sketch_image,
        "prompt": TextMedia(text=_SKETCH_FEATURE_PROMPT),
    })

    model = session.acquire()
    try:
        print(f"\n  [Sketch Feature Extraction] ", end="", flush=True)
        result = model.run(
            bundle,
            response_mime_type="application/json",
            response_json_schema=_SketchFeatures.model_json_schema(),
        )
        session.record_usage("Sketch Feature Extraction")
        response = result.items.get("text")
        response_text = response.text if isinstance(response, TextMedia) else ""
        print(response_text)
    finally:
        session.release()

    try:
        data = _SketchFeatures.model_validate_json(response_text)
        if data.features:
            print(f"  Extracted sketch features: {data.features}")
            return data.features
    except Exception:
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
    features: list[str] | None = None,  # unused — kept for API compatibility
    tolerance: str = "strict",
) -> JudgeCallable:
    """Return a JudgeCallable that scores sketch design fidelity as a single holistic score.

    IMAGE 1 = sketch grid (reference design)
    IMAGE 2 = generated photo (candidate)
    """
    tol = TOLERANCE_CONFIGS.get(tolerance, TOLERANCE_CONFIGS["strict"])
    avg_threshold = tol["avg_threshold"]
    min_floor = tol["min_floor"]

    _prev_candidate_hash: list[str | None] = [None]

    def judge(context: dict[str, Media]) -> tuple[bool, str]:
        sketch = context.get(sketch_key)
        candidate = context.get(candidate_key)

        if not isinstance(sketch, ImageMedia):
            return False, f"Missing sketch image (key='{sketch_key}')."
        if not isinstance(candidate, ImageMedia):
            return False, f"Missing candidate image (key='{candidate_key}')."

        current_hash = _image_hash(candidate)
        if _prev_candidate_hash[0] is not None and current_hash == _prev_candidate_hash[0]:
            print("  [Sketch Judge] Image unchanged — auto-accepting (no token spend)")
            return True, "none"
        _prev_candidate_hash[0] = current_hash

        prompt_text = _SKETCH_SCORE_PROMPT.format(
            result_description="a photorealistic shoe photo generated from the sketch.",
        )

        bundle = MediaBundle(items={
            "sketch": sketch,
            "candidate": candidate,
            "prompt": TextMedia(text=prompt_text),
        })

        model = session.acquire()
        try:
            raw_json = _call_vlm_structured(
                model, bundle,
                schema=_SketchJudgeResult.model_json_schema(),
                label="Sketch Judge",
            )
            session.record_usage("Sketch Judge")

            parsed = _SketchJudgeResult.model_validate_json(raw_json)
            score = float(parsed.score)
            repair = parsed.repair
            accepted = score >= avg_threshold and score >= min_floor

            context["_judge_metadata_sketch"] = {
                "scores": {"sketch": score},
                "avg_score": round(score, 2),
                "lowest_score": round(score, 2),
            }

            verdict = "ACCEPT" if accepted else "REJECT"
            print(f"  [Sketch Judge {verdict}] score={score:.1f} "
                  f"(threshold: >={avg_threshold})")
            if not accepted:
                print(f"  Repair: {repair}")
            return accepted, repair
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
    judge_notes: str = "",
    include_quality_features: bool = True,
) -> JudgeCallable:
    """Return a JudgeCallable that scores spec compliance and photo quality.

    IMAGE = generated photo only (no sketch shown)
    TEXT = spec dict (material, color, camera_angle, extras) + quality attributes

    Set include_quality_features=False to score only the attributes in spec,
    omitting the default white_background / lighting / sharpness checks.
    """
    if spec is None:
        spec = {}

    tol = TOLERANCE_CONFIGS.get(tolerance, TOLERANCE_CONFIGS["strict"])
    avg_threshold = tol["avg_threshold"]
    min_floor = tol["min_floor"]

    _quality = ["white_background", "lighting", "sharpness"]
    quality_features = _quality if include_quality_features else []
    features = list(spec.keys()) + [f for f in quality_features if f not in spec]

    spec_lines = [f"- {k.capitalize()}: {v}" for k, v in spec.items()]
    spec_text = "\n".join(spec_lines) if spec_lines else "(no additional specs)"

    prev_lowest_attr: list[str | None] = [None]
    stale_count: list[int] = [0]
    _prev_candidate_hash: list[str | None] = [None]

    def judge(context: dict[str, Media]) -> tuple[bool, str]:
        candidate = context.get(candidate_key)

        if not isinstance(candidate, ImageMedia):
            return False, f"Missing candidate image (key='{candidate_key}')."

        current_hash = _image_hash(candidate)
        if _prev_candidate_hash[0] is not None and current_hash == _prev_candidate_hash[0]:
            print("  [Spec Judge] Image unchanged — auto-accepting (no token spend)")
            return True, "none"
        _prev_candidate_hash[0] = current_hash

        iteration = context.get("loop_iteration", 0)
        max_iterations = context.get("loop_max_iterations", 5)

        result_description = (
            "a photorealistic shoe photo generated from the design spec."
        )
        iteration_context = (
            f"This is rendering {iteration + 1} of {max_iterations}. "
            f"Score each attribute based on what you clearly see. "
            f"Default to 4 when the spec intent is substantially met — only score 3 or below "
            f"for clear, unambiguous spec violations.\n"
        )

        stale_nudge = ""
        if stale_count[0] >= 2 and prev_lowest_attr[0]:
            attr = prev_lowest_attr[0]
            stale_nudge = (
                f"The attribute '{attr}' has been below standard for {stale_count[0]} consecutive "
                f"attempts. Focus your REPAIR instruction specifically on '{attr}'.\n"
            )

        judge_notes_section = ""
        if judge_notes:
            judge_notes_section = (
                f"SPECIAL SCORING RULES — read carefully before evaluating:\n"
                f"{judge_notes}\n\n"
            )

        prompt_text = _SPEC_SCORE_PROMPT.format(
            result_description=result_description,
            spec_text=spec_text,
            judge_notes_section=judge_notes_section,
            iteration_context=iteration_context,
            stale_nudge=stale_nudge,
            features=", ".join(features),
        )

        bundle = MediaBundle(items={
            "candidate": candidate,
            "prompt": TextMedia(text=prompt_text),
        })

        model = session.acquire()
        try:
            raw_json = _call_vlm_structured(
                model, bundle,
                schema=_SpecJudgeResult.model_json_schema(),
                label="Spec Judge",
            )
            session.record_usage("Spec Judge")

            parsed = _SpecJudgeResult.model_validate_json(raw_json)
            scores = {k: float(v) for k, v in parsed.scores.items()}
            repair = parsed.repair
            avg = sum(scores.values()) / len(scores) if scores else 0.0
            lowest_val = min(scores.values()) if scores else 0.0
            lowest_attr = min(scores, key=scores.get) if scores else ""
            accepted = avg >= avg_threshold and lowest_val >= min_floor

            if lowest_attr == prev_lowest_attr[0]:
                stale_count[0] += 1
            else:
                prev_lowest_attr[0] = lowest_attr
                stale_count[0] = 1

            context["_judge_detail_spec"] = {
                "observations": parsed.observations,
                "scores": {k: int(v) for k, v in parsed.scores.items()},
                "repair": parsed.repair,
                "thinking": getattr(model, "last_thinking", None),
            }

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
                print(f"  Repair: {repair}")
            return accepted, repair
        finally:
            session.release()

    return judge


# ---------------------------------------------------------------------------
# Reference Fidelity Judge
# ---------------------------------------------------------------------------

def make_reference_fidelity_judge(
    session: VLMSession,
    reference_image: ImageMedia,
    sketch_image: ImageMedia,
    candidate_key: str = "image",
    tolerance: str = "generous",
    save_dir=None,
    angle_name: str = "",
) -> JudgeCallable:
    """Return a JudgeCallable that checks if the generated shoe matches the reference shoe.

    Builds a 2-panel composite (REFERENCE SHOE = ref photo + sketch | GENERATED SHOE = candidate)
    and sends it as a single image so the model can compare side-by-side.
    """
    tol = TOLERANCE_CONFIGS.get(tolerance, TOLERANCE_CONFIGS["generous"])
    avg_threshold = tol["avg_threshold"]
    min_floor = tol["min_floor"]

    _prev_candidate_hash: list[str | None] = [None]
    _iter_count: list[int] = [0]

    def judge(context: dict[str, Media]) -> tuple[bool, str]:
        candidate = context.get(candidate_key)

        if not isinstance(candidate, ImageMedia):
            return False, f"Missing candidate image (key='{candidate_key}')."

        current_hash = _image_hash(candidate)
        if _prev_candidate_hash[0] is not None and current_hash == _prev_candidate_hash[0]:
            print("  [Reference Judge] Image unchanged — auto-accepting (no token spend)")
            return True, "none"
        _prev_candidate_hash[0] = current_hash
        _iter_count[0] += 1

        composite = _make_comparison_image(
            reference=reference_image.image,
            sketch=sketch_image.image,
            candidate=candidate.image,
        )

        if save_dir is not None:
            try:
                safe = angle_name or "angle"
                comp_path = save_dir / f"{safe}_comparison_iter{_iter_count[0]}.png"
                composite.save(comp_path)
            except Exception as _e:
                print(f"  [Reference Judge] Warning: could not save composite: {_e}")

        bundle = MediaBundle(items={
            "comparison": ImageMedia(image=composite),
            "prompt": TextMedia(text=_REFERENCE_FIDELITY_PROMPT),
        })

        model = session.acquire()
        try:
            raw_json = _call_vlm_structured(
                model, bundle,
                schema=_ReferenceFidelityResult.model_json_schema(),
                label="Reference Judge",
            )
            session.record_usage("Reference Judge")

            parsed = _ReferenceFidelityResult.model_validate_json(raw_json)
            score = float(parsed.score)
            repair = parsed.repair
            accepted = score >= avg_threshold and score >= min_floor

            context["_judge_detail_ref"] = {
                "reference_observations": parsed.reference_observations,
                "generated_observations": parsed.generated_observations,
                "score": int(parsed.score),
                "repair": parsed.repair,
                "thinking": getattr(model, "last_thinking", None),
            }

            verdict = "ACCEPT" if accepted else "REJECT"
            print(f"  [Reference Judge {verdict}] score={score:.1f} (threshold: >={avg_threshold})")
            if not accepted:
                print(f"  Repair: {repair}")
            return accepted, repair
        finally:
            session.release()

    return judge


# ---------------------------------------------------------------------------
# Shoe Count / Foot Judge
# ---------------------------------------------------------------------------

_SHOE_COUNT_PROMPT = """\
You are a product photo verifier. Your only task is to count how many shoes \
are visible in the generated image.

Required: {requirement}

Count every distinct shoe visible in the image. Do NOT check which foot — \
only count the number of shoes.

STEP 1 — OBSERVE
State exactly how many shoes are visible in the image.

STEP 2 — VERIFY
Does the count match the requirement?
  correct = 1  →  the image shows EXACTLY the required number of shoes
  correct = 0  →  the count is wrong (too many or too few)

STEP 3 — REPAIR (only if correct = 0)
State how many shoes are visible and give a single concrete instruction. \
Use plain terms like "generate only a single shoe" or \
"generate a pair of two shoes". Do not mention camera angle, foot, or materials.
"""

_SHOE_COUNT_REQUIREMENTS = {
    "right": "exactly 1 shoe",
    "left":  "exactly 1 shoe",
    "pair":  "exactly 2 shoes",
}


def make_shoe_count_judge(
    session: VLMSession,
    foot: str,
    candidate_key: str = "image",
) -> JudgeCallable:
    """Return a JudgeCallable that checks shoe count and foot (left/right/pair).

    Uses binary scoring: correct=1 passes, correct=0 fails.
    foot must be one of 'left', 'right', 'pair'.
    """
    requirement = _SHOE_COUNT_REQUIREMENTS.get(
        foot.lower(),
        f"exactly the shoe(s) matching: {foot}",
    )
    prompt_text = _SHOE_COUNT_PROMPT.format(requirement=requirement)

    _prev_candidate_hash: list[str | None] = [None]

    def judge(context: dict[str, Media]) -> tuple[bool, str]:
        candidate = context.get(candidate_key)
        if not isinstance(candidate, ImageMedia):
            return False, f"Missing candidate image (key='{candidate_key}')."

        current_hash = _image_hash(candidate)
        if _prev_candidate_hash[0] is not None and current_hash == _prev_candidate_hash[0]:
            print("  [Count Judge] Image unchanged — auto-accepting (no token spend)")
            return True, "none"
        _prev_candidate_hash[0] = current_hash

        bundle = MediaBundle(items={
            "candidate": candidate,
            "prompt": TextMedia(text=prompt_text),
        })

        model = session.acquire()
        try:
            raw_json = _call_vlm_structured(
                model, bundle,
                schema=_ShoeCountResult.model_json_schema(),
                label="Count Judge",
            )
            session.record_usage("Count Judge")

            parsed = _ShoeCountResult.model_validate_json(raw_json)
            accepted = parsed.correct == 1

            context["_judge_detail_count"] = {
                "observation": parsed.observation,
                "correct": parsed.correct,
                "repair": parsed.repair,
                "thinking": getattr(model, "last_thinking", None),
            }

            verdict = "ACCEPT" if accepted else "REJECT"
            print(f"  [Count Judge {verdict}] correct={parsed.correct} | {parsed.observation}")
            if not accepted:
                print(f"  Repair: {parsed.repair}")
            return accepted, parsed.repair
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
            f"\n\nREFINEMENT INSTRUCTIONS — do not start over, refine the existing rendering:\n"
            f"Sketch fidelity issues to fix:\n{feedback1}\n\n"
            f"Spec compliance issues to fix:\n{feedback2}"
        )
        return accepted1 and accepted2, combined_feedback

    return judge



# ---------------------------------------------------------------------------
# Best-of-N Selection
# ---------------------------------------------------------------------------

def make_best_fn(
    session: VLMSession,
    sketch_key: str = "sketch",
    output_key: str = "image",
) -> BestFn:
    """Return a BestFn that picks the best candidate by combined score.

    Computes (sketch_avg + spec_avg) / 2 for each iteration and selects
    the highest-scoring one. No VLM call — uses scores already recorded
    by the dual judge.
    """

    def best_fn(
        history: list[LoopIteration],
        context: dict[str, Media],
    ) -> dict[str, Media]:
        candidates: list[tuple[int, ImageMedia, float]] = []  # (index, image, score)
        for i, record in enumerate(history):
            img = record.outputs.get(output_key)
            if not isinstance(img, ImageMedia):
                continue
            sketch_avg = record.metadata.get("sketch_avg") or 0.0
            spec_avg = record.metadata.get("spec_avg") or 0.0
            combined = (sketch_avg + spec_avg) / 2.0
            candidates.append((i, img, combined))

        if not candidates:
            return {}

        if len(candidates) == 1:
            idx, img, score = candidates[0]
            print(f"  [Score Best-of-N] Only 1 candidate (score={score:.2f}), selecting it.")
            return {
                output_key: img,
                "best_selection_index": idx + 1,
                "best_selection_reason": TextMedia(text=f"Only one candidate (combined score={score:.2f})."),
            }

        best_idx, best_img, best_score = max(candidates, key=lambda t: t[2])
        print(f"\n  [Score Best-of-N] Scores: " +
              ", ".join(f"iter{i}={s:.2f}" for i, _, s in candidates))
        print(f"  Selected: iter {best_idx} with combined score {best_score:.2f}")

        reason = (
            f"Score-based selection: iter {best_idx} had the highest combined score "
            f"({best_score:.2f}) across {len(candidates)} candidates. "
            f"Scores: {', '.join(f'iter{i}={s:.2f}' for i, _, s in candidates)}"
        )
        return {
            output_key: best_img,
            "best_selection_index": best_idx + 1,
            "best_selection_reason": TextMedia(text=reason),
        }

    return best_fn
