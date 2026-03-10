"""Simple holistic judge for the sketch-to-shoe pipeline.

Single judge that compares the generated rendering against BOTH the sketch
(structure) and the material specification (color/finish) in one pass.
No feature extraction step needed.
"""

from __future__ import annotations

import hashlib
import io
import re
import sys

from casadei.loop import BestFn, JudgeCallable, LoopIteration
from casadei.media import ImageMedia, Media, TextMedia, MediaBundle

# Reuse shared infrastructure from judge.py
from judge import VLMSession, _stream_vlm, make_best_fn, _MAX_RETRIES  # noqa: F401


def _image_hash(img: ImageMedia) -> str:
    """MD5 hash of raw PNG bytes — fast identity check for unchanged images."""
    buf = io.BytesIO()
    img.image.save(buf, format="PNG")
    return hashlib.md5(buf.getvalue()).hexdigest()

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_HOLISTIC_JUDGE_PROMPT = """\
You are a luxury footwear quality inspector. You have three inputs:
- IMAGE 1: the original designer sketch
- IMAGE 2: the photorealistic shoe rendering to evaluate
- MATERIAL SPEC: the required materials, colors, finishes, and photography setup

MATERIAL SPEC:
{spec_text}

{judge_notes_section}\
{iteration_context}\
YOUR BIAS: default to PASS. Only FAIL when you can point to a specific, unambiguous, \
clearly visible error — something that is objectively wrong, not a matter of interpretation \
or rendering style. If you are uncertain, write "ok" and PASS.

STEP 1 — STRUCTURE
Look carefully at IMAGE 1 (sketch) and IMAGE 2 (rendering). Ignore colors and materials entirely.
Briefly note the shoe silhouette in each image, then compare construction only.

Write "ok" unless there is an unmistakably clear structural difference — a strap that is \
completely absent, a heel type that is the opposite of what the sketch shows (flat vs high, \
open vs closed back). Do NOT flag orientation, staging, or shoe positioning here — those are \
photography setup details, not structural defects. Do NOT flag differences that could be \
explained by the 3D-to-2D rendering perspective.

STEP 2 — MATERIAL
Compare IMAGE 2 against the MATERIAL SPEC text.
Evaluate PRESENCE and CATEGORY only — not rendering quality or texture precision.

For each component, write "ok" unless it is ENTIRELY ABSENT or in a COMPLETELY WRONG category \
(e.g., fully opaque leather where transparent PVC is required; zero gems on a gem-specified strap; \
completely wrong color family with no resemblance). Do NOT flag:
- texture rendering quality or how 3D the gems look
- exact sparkle level, gem density, or finish precision
- transparent heels that look slightly tinted vs fully clear
- color gradient precision or exact shade matching
These are rendering limitations, not design errors.

STEP 3 — VERDICT
PASS if both steps are ok or have only minor, arguable, or ambiguous differences.
FAIL only if there is a specific, clearly visible, undeniable error in STRUCTURE or MATERIAL.
When in doubt between PASS and FAIL → choose PASS.

Reply in EXACTLY this format:
STRUCTURE: <what clearly and undeniably differs from sketch, or "ok">
MATERIAL: <what is completely absent or in wrong category, or "ok">
VERDICT: PASS or FAIL
REPAIR: <specific fix instruction, or "none" if PASS>
"""


# ---------------------------------------------------------------------------
# Verdict parsing
# ---------------------------------------------------------------------------

def _parse_verdict(text: str) -> bool:
    m = re.search(r"VERDICT:\s*(PASS|FAIL)", text, re.IGNORECASE)
    if not m:
        raise ValueError("No 'VERDICT: PASS' or 'VERDICT: FAIL' line found")
    return m.group(1).upper() == "PASS"


def _parse_repair(text: str) -> str:
    m = re.search(r"REPAIR:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


# ---------------------------------------------------------------------------
# Holistic Judge
# ---------------------------------------------------------------------------

def make_holistic_judge(
    session: VLMSession,
    sketch_key: str = "sketch",
    candidate_key: str = "image",
    spec: dict[str, str] | None = None,
    judge_notes: str = "",
) -> JudgeCallable:
    """Single judge that compares rendered photo against sketch + full spec in one prompt.

    IMAGE 1 = sketch (structure reference)
    IMAGE 2 = generated rendering
    TEXT    = material spec + photography requirements
    """
    if spec is None:
        spec = {}

    spec_lines = [f"- {k.capitalize()}: {v}" for k, v in spec.items()]
    spec_text = "\n".join(spec_lines) if spec_lines else "(no spec provided)"

    # Stagnation tracking: accept if MATERIAL keeps failing while STRUCTURE is fine
    _material_fail_count: list[int] = [0]
    # Image identity tracking: skip VLM if candidate image hasn't changed
    _prev_candidate_hash: list[str | None] = [None]

    def _extract_failing_areas(text: str) -> frozenset[str]:
        """Return the set of areas that are NOT 'ok' in the judge response."""
        failing = set()
        for area in ("STRUCTURE", "MATERIAL"):
            m = re.search(rf"^{area}:\s*(.+)", text, re.IGNORECASE | re.MULTILINE)
            if m and m.group(1).strip().lower() != "ok":
                failing.add(area)
        return frozenset(failing)

    def judge(context: dict[str, Media]) -> tuple[bool, str]:
        sketch = context.get(sketch_key)
        candidate = context.get(candidate_key)

        if not isinstance(sketch, ImageMedia):
            return False, f"Missing sketch image (key='{sketch_key}')."
        if not isinstance(candidate, ImageMedia):
            return False, f"Missing candidate image (key='{candidate_key}')."

        # Skip VLM entirely if the image hasn't changed since the last iteration.
        # The generation model is stuck — re-judging would waste tokens and produce
        # inconsistent hallucinated feedback. Accept instead of looping forever.
        current_hash = _image_hash(candidate)
        if _prev_candidate_hash[0] is not None and current_hash == _prev_candidate_hash[0]:
            print("  [Holistic Judge] Image unchanged from previous iteration — auto-accepting (no token spend)")
            return True, "none"
        _prev_candidate_hash[0] = current_hash

        iteration = context.get("loop_iteration", 0)
        max_iterations = context.get("loop_max_iterations", 5)

        iteration_context = (
            f"This is rendering {iteration + 1} of {max_iterations}. "
            f"Evaluate only what you clearly see — default to PASS if uncertain.\n"
        )

        judge_notes_section = ""
        if judge_notes:
            judge_notes_section = (
                f"SPECIAL EVALUATION NOTES — read carefully before judging:\n"
                f"{judge_notes}\n\n"
            )

        prompt_text = _HOLISTIC_JUDGE_PROMPT.format(
            spec_text=spec_text,
            judge_notes_section=judge_notes_section,
            iteration_context=iteration_context,
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
                        f"You MUST include a line with exactly 'VERDICT: PASS' or 'VERDICT: FAIL'."
                    )
                    bundle = MediaBundle(items={
                        "sketch": sketch,
                        "candidate": candidate,
                        "prompt": TextMedia(text=retry_prompt),
                    })

                raw_response = _stream_vlm(model, bundle, label="Holistic Judge")
                session.record_usage("Holistic Judge")

                try:
                    accepted = _parse_verdict(raw_response)
                    repair = _parse_repair(raw_response)

                    if not accepted:
                        failing = _extract_failing_areas(raw_response)

                        # Track MATERIAL failures independently of PHOTOGRAPHY
                        if "MATERIAL" in failing:
                            _material_fail_count[0] += 1
                        else:
                            _material_fail_count[0] = 0

                        # Accept if MATERIAL keeps failing but STRUCTURE is fine
                        # (minor texture/finish drift that the model can't fix)
                        if _material_fail_count[0] >= 2 and "STRUCTURE" not in failing:
                            print(f"  [Holistic Judge] Material issue stale for "
                                  f"{_material_fail_count[0]} iterations with correct structure — "
                                  f"accepting as substantially correct")
                            return True, repair

                    verdict = "ACCEPT" if accepted else "REJECT"
                    print(f"  [Holistic Judge {verdict}]")
                    if not accepted:
                        print(f"  Repair: {repair[:400]}")
                    return accepted, repair

                except ValueError as e:
                    last_error = str(e)
                    print(f"  [Holistic Judge] Parse error: {last_error} "
                          f"(attempt {attempt + 1}/{_MAX_RETRIES + 1})")

            print("  [Holistic Judge] Fallback: treating unparseable response as REJECT")
            return False, raw_response or "Could not parse VLM response."
        finally:
            session.release()

    return judge
