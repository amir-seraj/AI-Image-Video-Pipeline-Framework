"""Judge and best-selection functions for the shoe replacement loop.

Provides factory functions that return callables matching the
JudgeCallable and BestFn protocols defined in casadei.loop.
"""

from __future__ import annotations

import gc
import re
import sys

from PIL import Image as PILImage, ImageDraw, ImageFont

from casadei.loop import BestFn, JudgeCallable, LoopIteration
from casadei.media import ImageMedia, Media, TextMedia, MediaBundle

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SCORE_PROMPT = """\
You are a strict quality inspector comparing shoes across two images.

The first image is the REFERENCE SHOE — a product photo of the target shoe.
The second image shows a PERSON WEARING SHOES — examine only the shoes \
on their feet.

Your job: for each listed attribute, score how closely the shoes on the \
person's feet match the same attribute in the reference shoe.

IMPORTANT: The shoes on the person are AI-generated replacements. They \
often look correct at first glance but have subtle differences in \
proportions, trim details, strap attachment points, edge finishing, or \
color shade. Examine each attribute at the fine-detail level, not just \
the category level. "Both are red" does not earn a 5 — check the exact \
shade, glossiness, and reflections.

{iteration_context}\
{stale_nudge}\
Scoring scale:
  1 = completely different from the reference
  2 = vaguely similar but clearly wrong
  3 = same general type but noticeable differences in detail
  4 = close match with only minor differences visible on close inspection
  5 = indistinguishable — identical shape, proportions, color shade, and \
every visible detail

A score of 5 means you cannot find ANY difference for that attribute. \
If you can spot even one small difference, the score must be 4 or lower.

Attributes to score: {features}

Check BOTH feet individually. For "both feet match": score 1 if the two \
feet show different shoes (one matches the reference, the other does not); \
score 5 only if both feet wear identical shoes that match the reference.

First, describe what you observe in the reference shoe and on the \
person's feet for each attribute. Then provide your scores.

Reply in this exact format:
OBSERVATIONS: <For each attribute, state what you see in the reference \
vs. what you see on the person's feet. Note any differences no matter \
how small.>
SCORES: {score_format}
REPAIR: <For each attribute scored below 5, describe the specific \
mismatch between what you see on the person's feet and the reference. \
Do NOT copy previous issues verbatim — describe what you actually see NOW.>

Example of a correctly filled response (scores are illustrative):
SCORES: {example_format}
REPAIR: The heel appears lower than the reference ...
"""

_FEATURE_PROMPT = """\
You are looking at a product shoe photo. Extract a complete and precise \
description of this shoe as a list of attribute tags.

These tags will be used as scoring criteria by another agent: given only \
your tag list and a new image, that agent must be able to check each tag \
individually and say whether the shoe in the new image matches or not. \
So every tag must be specific and verifiable — "red patent leather" is \
verifiable; "nice material" is not.

Cover every visible dimension — record the concrete value you observe, \
not just the category name. Do NOT include any numbers or measurements \
(no centimeters, no estimates like "~10cm") — use descriptive terms only \
(e.g. "thin", "high", "chunky", "thick"). Only extract EXTERIOR features \
visible when the shoe is worn on a foot — do NOT include interior features \
(lining, insole, brand label) that would be hidden while worn:
- Type / silhouette  (e.g. "stiletto pump", "over-the-knee boot", "platform mule")
- Color(s) / pattern  (e.g. "deep burgundy", "zebra print", "two-tone black and white")
- Material / texture  (e.g. "croc-embossed patent leather", "shearling lining", "mesh upper")
- Heel  (e.g. "clear lucite stiletto heel", "stacked wooden block heel", "flat")
- Toe  (e.g. "extreme pointed toe", "open square toe", "closed almond toe")
- Closure / straps  (e.g. "criss-cross ankle lacing", "elasticated strap", "side zip")
- Sole / platform  (e.g. "transparent platform sole", "rubber lug sole", "thin leather sole")
- Hardware / details  (e.g. "oversized gold chain trim", "crystal toe cap", "logo plate")

Reply with ONLY a JSON array of 6–10 descriptor strings. \
No explanation, no markdown — just the array.

Example:
["deep red croc-embossed patent leather pump", "clear lucite stiletto heel", \
"transparent platform sole", "open square toe", \
"ankle wrap lacing with gold rings", "gold hardware throughout"]
"""

_FALLBACK_FEATURES = [
    "shoe type and silhouette",
    "color and pattern",
    "heel type and height",
    "material and texture",
    "toe shape",
    "closure and straps",
    "sole style",
]

TOLERANCE_CONFIGS = {
    "generous":  {"avg_threshold": 2.5, "min_floor": 1.5},
    "moderate":  {"avg_threshold": 3.5, "min_floor": 2.5},
    "strict":    {"avg_threshold": 4.5, "min_floor": 3.5},
}

_BEST_PROMPT = """\
You are selecting the best shoe replacement result.

The first image is the REFERENCE SHOE PHOTO — the target shoe design.
The second image shows {n} candidate results side by side, labeled Option 1 \
through Option {n} from left to right.

Which option's shoes best match the reference shoe photo in style, color, \
heel shape, material, and natural placement on the person?

Reply with ONLY a single number (1 to {n}) on the first line.
Then on the next line, briefly explain why.
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_RETRIES = 2  # extra attempts on unparseable VLM output


def _unload_and_cleanup(model) -> None:
    """Unload a model and clear GPU memory."""
    model.unload_model()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _stream_vlm(model, bundle: MediaBundle, label: str = "VLM") -> str:
    """Run VLM with streaming, printing tokens to terminal in real time.

    Returns the full collected response text.
    """
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


def _concat_candidates(images: list[PILImage.Image], label_height: int = 40) -> PILImage.Image:
    """Stitch candidate images side by side with 'Option N' labels on top."""
    if not images:
        raise ValueError("No images to concat")

    # Resize all to same height for a clean grid
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
        text_x = x_offset + (img.width - text_w) // 2
        draw.text((text_x, 6), label, fill=(0, 0, 0), font=font)
        canvas.paste(img, (x_offset, label_height))
        x_offset += img.width

    return canvas


# ---------------------------------------------------------------------------
# Score Parsing
# ---------------------------------------------------------------------------


def _parse_scores(text: str) -> dict[str, float]:
    """Parse 'SCORES: attr1=4, attr2=2, ...' from VLM response."""
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
    """Extract REPAIR text from VLM response."""
    repair_match = re.search(r"REPAIR:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if repair_match:
        return repair_match.group(1).strip()
    return text


# ---------------------------------------------------------------------------
# VLM Session
# ---------------------------------------------------------------------------

class VLMSession:
    """Shared VLM model lifecycle for judge and best_fn.

    In swap mode (default), the model is loaded and unloaded on each
    acquire/release pair. In keep-loaded mode (call ``load()`` first),
    the model stays resident and ``release()`` is a no-op.
    """

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
        """Pre-load the model (keep-both mode). Call once before the loop."""
        if self._model is None:
            from casadei.models.registry import default_registry
            model_cls = default_registry.get(self._model_name)
            self._model = model_cls()
            self._model.load_model()
        self._persistent = True
        print(f"  [VLMSession] Pre-loaded {self._model_name} (keep-both mode)")

    def unload(self) -> None:
        """Unload the model and free GPU memory."""
        if self._model is not None:
            _unload_and_cleanup(self._model)
            self._model = None
        self._persistent = False

    def acquire(self):
        """Get the model, loading it if not already loaded."""
        if self._model is None:
            from casadei.models.registry import default_registry
            model_cls = default_registry.get(self._model_name)
            self._model = model_cls()
            self._model.load_model()
        return self._model

    def release(self) -> None:
        """Release the model. Unloads only if not in persistent mode."""
        if not self._persistent and self._model is not None:
            _unload_and_cleanup(self._model)
            self._model = None


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------


def extract_features(session: VLMSession, shoe_image: ImageMedia) -> list[str]:
    """Ask the VLM to identify visual attributes of the reference shoe.

    Called once before the loop starts. Returns a list of short attribute names.
    Falls back to generic features on parse failure.
    """
    import json as _json

    bundle = MediaBundle(items={
        "shoe": shoe_image,
        "prompt": TextMedia(text=_FEATURE_PROMPT),
    })

    model = session.acquire()
    try:
        response = _stream_vlm(model, bundle, label="Feature Extraction")
        session.record_usage("Feature Extraction")
    finally:
        session.release()

    try:
        start = response.index("[")
        end = response.index("]", start) + 1
        features = _json.loads(response[start:end])
        if isinstance(features, list) and all(isinstance(f, str) for f in features) and len(features) >= 2:
            print(f"  Extracted features: {features}")
            return features
    except (ValueError, _json.JSONDecodeError):
        pass

    print(f"  Feature extraction parse failed, using fallback: {_FALLBACK_FEATURES}")
    return list(_FALLBACK_FEATURES)


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def make_judge(
    session: VLMSession,
    shoe_key: str = "shoe",
    candidate_key: str = "image",
    features: list[str] | None = None,
    tolerance: str = "strict",
) -> JudgeCallable:
    """Return a structured per-attribute scoring JudgeCallable.

    Scores each feature 1-5 and accepts when avg >= avg_threshold AND
    every attribute >= min_floor (thresholds set by tolerance level).
    Includes iteration context, previous-feedback echo, and a stale-
    attribute guardrail to break repetitive repair loops.
    """
    if features is None:
        features = list(_FALLBACK_FEATURES)

    # Always include consistency check — catches partial replacements
    _CONSISTENCY_ATTR = "both feet match"
    if _CONSISTENCY_ATTR not in features:
        features = list(features) + [_CONSISTENCY_ATTR]

    tol = TOLERANCE_CONFIGS.get(tolerance, TOLERANCE_CONFIGS["strict"])
    avg_threshold = tol["avg_threshold"]
    min_floor = tol["min_floor"]

    # Closure state for stale guardrail
    prev_lowest_attr: list[str | None] = [None]
    stale_count: list[int] = [0]

    # Use [1-5] as placeholder — unambiguous for VLMs (N was being echoed literally)
    score_format = ", ".join(f"{f}=[1-5]" for f in features)
    # Concrete filled example using the real attribute names — shows the model exactly
    # what a valid SCORES line looks like (values are illustrative, not real scores)
    example_format = ", ".join(f"{f}=3" for f in features)

    def judge(context: dict[str, Media]) -> tuple[bool, str]:
        candidate = context.get(candidate_key)
        reference = context.get(shoe_key)

        if not isinstance(candidate, ImageMedia):
            return False, f"Missing candidate image (key='{candidate_key}')."
        if not isinstance(reference, ImageMedia):
            return False, f"Missing reference shoe image (key='{shoe_key}')."

        iteration = context.get("loop_iteration", 0)
        max_iterations = context.get("loop_max_iterations", 5)

        if iteration == 0:
            iteration_context = (
                f"This is comparison 1 of {max_iterations}.\n"
            )
        else:
            iteration_context = (
                f"This is comparison {iteration + 1} of {max_iterations}. "
                f"Check whether the shoes are a closer match than in the previous comparison.\n"
            )

        stale_nudge = ""
        if stale_count[0] >= 2 and prev_lowest_attr[0]:
            attr = prev_lowest_attr[0]
            n = stale_count[0]
            stale_nudge = (
                f"Note: '{attr}' has scored lowest for {n} comparisons in a row. "
                f"Pay close attention to this attribute and describe the mismatch "
                f"in precise detail.\n"
            )

        prompt_text = _SCORE_PROMPT.format(
            iteration_context=iteration_context,
            stale_nudge=stale_nudge,
            features=", ".join(features),
            score_format=score_format,
            example_format=example_format,
        )

        bundle = MediaBundle(items={
            "reference": reference,
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
                        "reference": reference,
                        "candidate": candidate,
                        "prompt": TextMedia(text=retry_prompt),
                    })

                raw_response = _stream_vlm(model, bundle, label="VLM Judge")
                session.record_usage("VLM Judge")

                try:
                    scores = _parse_scores(raw_response)
                    repair = _parse_repair(raw_response)

                    avg = sum(scores.values()) / len(scores)
                    lowest_val = min(scores.values())
                    lowest_attr = min(scores, key=scores.get)
                    accepted = avg >= avg_threshold and lowest_val >= min_floor

                    # Update stale guardrail
                    if lowest_attr == prev_lowest_attr[0]:
                        stale_count[0] += 1
                    else:
                        prev_lowest_attr[0] = lowest_attr
                        stale_count[0] = 1

                    # Frame feedback for the generation model
                    if iteration == 0:
                        generation_feedback = (
                            "The shoes in the second image were replaced in the previous "
                            "attempt but do not fully match the reference shoe photo yet. "
                            "Ensure BOTH feet have the correct shoe and refine to better "
                            "match the reference. "
                            f"Specifically: {repair}"
                        )
                    else:
                        generation_feedback = (
                            "The previous refinement still does not fully match the "
                            "reference shoe. Continue improving the shoe match. "
                            f"Specifically: {repair}"
                        )

                    context["_judge_metadata"] = {
                        "scores": scores,
                        "avg_score": round(avg, 2),
                        "lowest_score": round(lowest_val, 2),
                        "lowest_attr": lowest_attr,
                        "stale_count": stale_count[0],
                    }

                    verdict = "ACCEPT" if accepted else "REJECT"
                    scores_str = ", ".join(f"{k}={v}" for k, v in scores.items())
                    print(f"  [{verdict}] avg={avg:.1f} min={lowest_val:.1f} "
                          f"(threshold: avg>={avg_threshold}, min>={min_floor})")
                    print(f"  Scores: {scores_str}")
                    if not accepted:
                        print(f"  Repair: {repair[:200]}")

                    return accepted, generation_feedback

                except ValueError as e:
                    last_error = str(e)
                    print(f"  [VLM Judge] Parse error: {last_error} "
                          f"(attempt {attempt + 1}/{_MAX_RETRIES + 1})")

            print(f"  [VLM Judge] Fallback: treating unparseable response as REJECT")
            return False, raw_response or "Could not parse VLM response."
        finally:
            session.release()

    return judge


# ---------------------------------------------------------------------------
# Multi-Judge (specialized agents)
# ---------------------------------------------------------------------------

_MULTI_AGENT_PROMPT = """\
You are an expert {domain} inspector comparing shoes across two images.

The first image is the REFERENCE SHOE — a product photo of the target shoe.
The second image shows a PERSON WEARING SHOES — examine only the shoes \
on their feet.

{domain_instructions}

IMPORTANT: The shoes on the person are AI-generated replacements. They \
often look correct at first glance but have subtle differences. Examine \
each attribute at the fine-detail level, not just the category level.

Scoring scale:
  1 = completely different from the reference
  2 = vaguely similar but clearly wrong
  3 = same general type but noticeable differences in detail
  4 = close match with only minor differences visible on close inspection
  5 = indistinguishable — identical in every visible detail

A score of 5 means you cannot find ANY difference for that attribute. \
If you can spot even one small difference, the score must be 4 or lower.

Attributes to score: {features}

First, describe what you observe in the reference shoe and on the \
person's feet for each attribute. Then provide your scores.

Reply in this exact format:
OBSERVATIONS: <For each attribute, state what you see in the reference \
vs. what you see on the person's feet. Note any differences no matter \
how small.>
SCORES: {score_format}
REPAIR: <For each attribute scored below 5, describe the specific \
mismatch between what you see on the person's feet and the reference. \
If all scores are 5, write "No issues.">

Example of a correctly filled response (scores are illustrative):
SCORES: {example_format}
REPAIR: The shade is slightly warmer than the reference ...
"""

_SYNTHESIS_PROMPT = """\
You are directing FireRed, an AI image editor, that will see the REFERENCE SHOE \
image above while making edits to a generated photo.

Specialist judges found these remaining issues in the generated shoe:
{failing_reports}

These aspects are already correct and must NOT be changed:
{passing_domains}

Write a short instruction (under 80 words) for FireRed. Your instruction must:
- Point to a specific visible area of the reference shoe image
- Say exactly what to replicate from that area
- Confirm what is already correct and should stay unchanged
- Sound like directions to a skilled photo retoucher, not a judge report

Write only the instruction. Do not use agent category names or bullet points.
"""


def _synthesize_firered_prompt(
    model,
    reference: ImageMedia,
    failing_agents: list[tuple[str, dict, str]],
    passing_names: list[str],
) -> str | None:
    """Call VLM with the reference shoe image to produce a focused FireRed directive.

    The VLM sees the reference shoe and the failing agent reports, then writes
    a short natural-language instruction pointing FireRed at the specific parts
    of the reference it needs to replicate more closely.

    Returns None on failure (caller falls back to raw repair text).
    """
    if not failing_agents:
        return None

    failing_reports = "\n".join(
        f"- {repair}"
        for _, _, repair in failing_agents
    )
    passing_text = (
        ", ".join(passing_names)
        if passing_names
        else "None — all aspects need improvement"
    )
    prompt_text = _SYNTHESIS_PROMPT.format(
        failing_reports=failing_reports,
        passing_domains=passing_text,
    )
    bundle = MediaBundle(items={
        "reference": reference,
        "prompt": TextMedia(text=prompt_text),
    })
    try:
        result = _stream_vlm(model, bundle, label="Synthesis→FireRed")
        return result.strip() if result.strip() else None
    except Exception as exc:
        print(f"  [Synthesis] Failed ({exc}), using raw repair text")
        return None


_MULTI_AGENTS = [
    {
        "name": "Color & Material",
        "domain": "color and material",
        "domain_instructions": (
            "Your expertise is COLOR, PATTERN, MATERIAL, and SURFACE FINISH.\n\n"
            "Focus on:\n"
            "- Exact color shade (not just 'red' — is it cherry red, wine red, coral?)\n"
            "- Color distribution and gradients across the shoe\n"
            "- Pattern accuracy (print scale, orientation, placement)\n"
            "- Material type (leather, suede, patent, fabric, synthetic)\n"
            "- Surface texture (grain, weave, embossing depth)\n"
            "- Glossiness and reflections (matte vs satin vs high-gloss)\n\n"
            "Compare each at the pixel level. 'Both are red' is NOT sufficient — "
            "check whether the exact shade, saturation, and surface sheen match."
        ),
        "features": [
            "color shade accuracy",
            "pattern fidelity",
            "material type",
            "texture and grain",
            "glossiness and reflections",
        ],
    },
    {
        "name": "Shape & Structure",
        "domain": "shape and structural",
        "domain_instructions": (
            "Your expertise is SILHOUETTE, PROPORTIONS, and STRUCTURAL GEOMETRY.\n\n"
            "Focus on:\n"
            "- Overall silhouette and profile shape\n"
            "- Heel type (stiletto, block, wedge, kitten, flat)\n"
            "- Heel height relative to the rest of the shoe\n"
            "- Heel angle and taper\n"
            "- Toe box shape (pointed, round, square, almond, open)\n"
            "- Sole thickness and platform height\n"
            "- Arch curve and instep line\n\n"
            "Compare proportions and angles precisely. A heel that is "
            "slightly too short or a toe box that is slightly too wide "
            "should score below 5."
        ),
        "features": [
            "silhouette and profile",
            "heel type and height",
            "heel angle and shape",
            "toe shape and proportions",
            "sole and platform",
        ],
    },
    {
        "name": "Details & Hardware",
        "domain": "detail and hardware",
        "domain_instructions": (
            "Your expertise is SMALL DETAILS, HARDWARE, and DECORATIVE ELEMENTS.\n\n"
            "Focus on:\n"
            "- Closure type (zip, buckle, lace, slip-on, velcro)\n"
            "- Strap count, width, placement, and routing\n"
            "- Buckle/clasp shape, size, color, and placement\n"
            "- Metal hardware finish (gold, silver, gunmetal, brushed, polished)\n"
            "- Decorative trim (chains, studs, crystals, bows, logos)\n"
            "- Edge finishing (piping, stitching, raw-cut, rolled)\n"
            "- Seam lines and construction details\n\n"
            "These small elements are where AI-generated shoes most often "
            "differ from the reference. Examine attachment points, hardware "
            "count, and decorative placement carefully."
        ),
        "features": [
            "closure and straps",
            "buckles and clasps",
            "hardware finish and color",
            "decorative trim",
            "edge finishing and seams",
        ],
    },
    {
        "name": "Consistency",
        "domain": "consistency and realism",
        "domain_instructions": (
            "Your expertise is LEFT-RIGHT CONSISTENCY and PLACEMENT REALISM.\n\n"
            "Focus on:\n"
            "- Do BOTH feet show the same shoe? Compare left foot vs right foot.\n"
            "- Are both shoes the same color, shape, and style?\n"
            "- Do the shoes sit naturally on the person's feet?\n"
            "- Is the scale/size proportionate to the person's body?\n"
            "- Do the shoes make proper contact with the ground/surface?\n"
            "- Are there any impossible geometry artifacts (floating, clipping)?\n\n"
            "AI-generated replacements often get one foot right and the other "
            "wrong, or produce shoes that float or clip through the ground."
        ),
        "features": [
            "both feet match",
            "natural foot placement",
            "scale and proportion to body",
            "ground contact realism",
        ],
    },
]


def make_multi_judge(
    session: VLMSession,
    shoe_key: str = "shoe",
    candidate_key: str = "image",
    tolerance: str = "strict",
) -> JudgeCallable:
    """Return a multi-agent JudgeCallable with 4 specialized domain agents.

    Each agent focuses on a narrow set of attributes (color/material,
    shape/structure, details/hardware, consistency). Runs sequentially
    on the same model. Scores are combined for accept/reject decision.
    """
    tol = TOLERANCE_CONFIGS.get(tolerance, TOLERANCE_CONFIGS["strict"])
    avg_threshold = tol["avg_threshold"]
    min_floor = tol["min_floor"]

    # Closure state for stale guardrail
    prev_lowest_attr: list[str | None] = [None]
    stale_count: list[int] = [0]

    def judge(context: dict[str, Media]) -> tuple[bool, str]:
        candidate = context.get(candidate_key)
        reference = context.get(shoe_key)

        if not isinstance(candidate, ImageMedia):
            return False, f"Missing candidate image (key='{candidate_key}')."
        if not isinstance(reference, ImageMedia):
            return False, f"Missing reference shoe image (key='{shoe_key}')."

        iteration = context.get("loop_iteration", 0)

        all_scores: dict[str, float] = {}
        # Per-agent tracking: name -> (scores_dict, repair_text)
        agent_results: dict[str, tuple[dict[str, float], str]] = {}

        model = session.acquire()
        try:
            for agent_def in _MULTI_AGENTS:
                agent_name = agent_def["name"]
                features = agent_def["features"]
                score_format = ", ".join(f"{f}=[1-5]" for f in features)
                example_format = ", ".join(f"{f}=3" for f in features)

                prompt_text = _MULTI_AGENT_PROMPT.format(
                    domain=agent_def["domain"],
                    domain_instructions=agent_def["domain_instructions"],
                    features=", ".join(features),
                    score_format=score_format,
                    example_format=example_format,
                )

                bundle = MediaBundle(items={
                    "reference": reference,
                    "candidate": candidate,
                    "prompt": TextMedia(text=prompt_text),
                })

                last_error = ""
                parsed = False

                for attempt in range(_MAX_RETRIES + 1):
                    if attempt > 0 and last_error:
                        retry_prompt = (
                            f"{prompt_text}\n\n"
                            f"Your previous response could not be parsed ({last_error}).\n"
                            f"You MUST replace every [1-5] with an actual integer.\n"
                            f"Correct example:\n"
                            f"SCORES: {example_format}\n"
                            f"REPAIR: ..."
                        )
                        bundle = MediaBundle(items={
                            "reference": reference,
                            "candidate": candidate,
                            "prompt": TextMedia(text=retry_prompt),
                        })

                    raw = _stream_vlm(model, bundle, label=agent_name)
                    session.record_usage(agent_name)

                    try:
                        scores = _parse_scores(raw)
                        repair = _parse_repair(raw)
                        all_scores.update(scores)
                        agent_results[agent_name] = (scores, repair)
                        parsed = True
                        break
                    except ValueError as e:
                        last_error = str(e)
                        print(f"  [{agent_name}] Parse error: {last_error} "
                              f"(attempt {attempt + 1}/{_MAX_RETRIES + 1})")

                if not parsed:
                    print(f"  [{agent_name}] All attempts failed, skipping agent")

            # --- Synthesis step (model still loaded) ---
            # Run after all 4 agents while we still hold the model, so we don't
            # need an extra load/unload cycle.
            _FINE_TUNE_THRESHOLD = 3.8
            synthesized_repair: str | None = None
            if all_scores:
                _avg_now = sum(all_scores.values()) / len(all_scores)

                # Identify failing agents (same criteria as accept/reject).
                _failing = [
                    (name, sc, rep)
                    for name, (sc, rep) in agent_results.items()
                    if sc and (
                        min(sc.values()) < min_floor
                        or sum(sc.values()) / len(sc) < avg_threshold
                    )
                    and rep and rep != "No issues."
                ]

                # Fine-tune vs redo: same strategy as before, but now we feed
                # those agents into the synthesis VLM rather than raw text.
                if _avg_now >= _FINE_TUNE_THRESHOLD and _failing:
                    _worst = min(
                        _failing,
                        key=lambda x: (min(x[1].values()), sum(x[1].values()) / len(x[1])),
                    )
                    _agents_for_synth = [_worst]
                else:
                    _agents_for_synth = _failing

                _failing_names = {n for n, _, _ in _agents_for_synth}
                _passing_names = [n for n in agent_results if n not in _failing_names]

                synthesized_repair = _synthesize_firered_prompt(
                    model, reference, _agents_for_synth, _passing_names,
                )
                session.record_usage("Synthesis")
        finally:
            session.release()

        if not all_scores:
            return False, "All judge agents failed to produce scores."

        avg = sum(all_scores.values()) / len(all_scores)
        lowest_val = min(all_scores.values())
        lowest_attr = min(all_scores, key=all_scores.get)
        accepted = avg >= avg_threshold and lowest_val >= min_floor

        # Update stale guardrail
        if lowest_attr == prev_lowest_attr[0]:
            stale_count[0] += 1
        else:
            prev_lowest_attr[0] = lowest_attr
            stale_count[0] = 1

        # Full combined repair for logging only
        all_repairs = [
            f"[{name}] {repair}"
            for name, (_, repair) in agent_results.items()
            if repair and repair != "No issues."
        ]
        combined_repair = " ".join(all_repairs) if all_repairs else "No issues."

        # Identify failing agents (for fallback and logging).
        failing_agents = [
            (name, scores, repair)
            for name, (scores, repair) in agent_results.items()
            if scores and (
                min(scores.values()) < min_floor
                or sum(scores.values()) / len(scores) < avg_threshold
            )
            and repair and repair != "No issues."
        ]

        # Use the VLM-synthesized directive if available, otherwise fall back
        # to the raw repair text with the fine-tune/redo strategy.
        if synthesized_repair:
            focused_repair = synthesized_repair
        else:
            if avg >= _FINE_TUNE_THRESHOLD and failing_agents:
                worst = min(
                    failing_agents,
                    key=lambda x: (
                        min(x[1].values()),
                        sum(x[1].values()) / len(x[1]),
                    ),
                )
                focused_repair = f"[{worst[0]}] {worst[2]}"
            else:
                focused_repair = (
                    " ".join(f"[{n}] {r}" for n, _, r in failing_agents)
                    if failing_agents else "No issues."
                )

        # Frame feedback for the generation model — use focused_repair (worst agent only)
        # not combined_repair, to avoid overwhelming FireRed with contradictory instructions
        if iteration == 0:
            generation_feedback = (
                "The shoes in the second image were replaced in the previous "
                "attempt but do not fully match the reference shoe photo yet. "
                "Ensure BOTH feet have the correct shoe and refine to better "
                "match the reference. "
                f"Specifically: {focused_repair}"
            )
        else:
            generation_feedback = (
                "The previous refinement still does not fully match the "
                "reference shoe. Continue improving the shoe match. "
                f"Specifically: {focused_repair}"
            )

        context["_judge_metadata"] = {
            "scores": all_scores,
            "avg_score": round(avg, 2),
            "lowest_score": round(lowest_val, 2),
            "lowest_attr": lowest_attr,
            "stale_count": stale_count[0],
        }

        verdict = "ACCEPT" if accepted else "REJECT"
        scores_str = ", ".join(f"{k}={v}" for k, v in all_scores.items())
        print(f"  [{verdict}] avg={avg:.1f} min={lowest_val:.1f} "
              f"(threshold: avg>={avg_threshold}, min>={min_floor})")
        print(f"  Scores: {scores_str}")
        if not accepted:
            mode = "fine-tune" if avg >= _FINE_TUNE_THRESHOLD else "redo"
            src = "synthesized" if synthesized_repair else "raw"
            print(f"  Repair ({mode}/{src}): {focused_repair[:300]}")

        return accepted, generation_feedback

    return judge


# ---------------------------------------------------------------------------
# Best-of-N Selection
# ---------------------------------------------------------------------------

def make_best_fn(
    session: VLMSession,
    shoe_key: str = "shoe",
    output_key: str = "image",
) -> BestFn:
    """Return a BestFn that concatenates all candidates into a single labeled
    image and asks the VLM to pick the best shoe replacement result.

    Sends exactly 2 images to the VLM: the reference shoe photo + a grid of
    all candidates with "Option N" labels. Streams the response to stdout.
    Retries on unparseable output; falls back to last candidate.
    """

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
            print(f"\n  [VLM Best-of-N] Only 1 candidate, selecting it.")
            return {
                output_key: candidates[0],
                "best_selection_index": 1,
                "best_selection_reason": TextMedia(text="Only one candidate."),
            }

        reference = context.get(shoe_key)
        if not isinstance(reference, ImageMedia):
            return {output_key: candidates[-1]}

        # Concat all candidates into a single labeled image
        n = len(candidates)
        pil_images = [c.image for c in candidates]
        grid_image = _concat_candidates(pil_images)

        print(f"\n  [VLM Best-of-N] Selecting from {n} candidates (grid: {grid_image.size[0]}x{grid_image.size[1]})")

        prompt_text = _BEST_PROMPT.format(n=n)
        bundle = MediaBundle(items={
            "reference": reference,
            "candidates_grid": ImageMedia(image=grid_image),
            "prompt": TextMedia(text=prompt_text),
        })

        model = session.acquire()
        try:
            chosen_1based = None
            response = ""

            for attempt in range(_MAX_RETRIES + 1):
                response = _stream_vlm(model, bundle, label="VLM Best-of-N")
                session.record_usage("VLM Best-of-N")

                # Parse: find a number 1..n on the first line
                first_line = response.splitlines()[0] if response else ""
                match = re.search(r"\d+", first_line)
                if match:
                    val = int(match.group())
                    if 1 <= val <= n:
                        chosen_1based = val
                        break

                print(f"  [VLM Best-of-N] Could not parse valid option 1-{n} (attempt {attempt + 1}/{_MAX_RETRIES + 1}), retrying...")

            # Fallback: pick last candidate (most refined)
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
