"""Generate shoe images at multiple camera angles — Gemini version.

Takes a design sketch and an already-generated photorealistic reference image
(e.g. from run_sketch_to_shoe_gemini.py) and produces new views at specified
camera angles.  Each angle is a single-shot Gemini Flash Image Edit call —
no iterative loop or VLM judging.

When --all-angles is used, all angle requests are fired concurrently via
ThreadPoolExecutor for maximum speed.

Usage:
    # Single angle
    python tests/run_shoe_angles.py \\
        --sketch tests/Image/sketch001.png \\
        --reference tests/output/sketch_to_shoe_gemini/.../final_result.png \\
        --angles side

    # Multiple specific angles
    python tests/run_shoe_angles.py \\
        --sketch sketch.png --reference ref.png \\
        --angles 3/4 side front back top

    # All angles (concurrent)
    python tests/run_shoe_angles.py \\
        --sketch sketch.png --reference ref.png \\
        --all-angles

    # All angles with foot selection
    python tests/run_shoe_angles.py \\
        --sketch sketch.png --reference ref.png \\
        --all-angles --foot left
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("shoe_angles")

from dotenv import load_dotenv
from PIL import Image as PILImage

# Load .env so GEMINI_API_KEY is available before any genai import
load_dotenv()

from google import genai
from google.genai import types as genai_types

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "workflows" / "sketch_to_shoe" / "scripts"))

from casadei.media import ImageMedia
from casadei.providers.gemini_pricing import extract_token_usage, calculate_cost
from judge import VLMSession, make_spec_judge, make_reference_fidelity_judge, make_shoe_count_judge

_GENERATION_MODEL_ID = "gemini-3.1-flash-image-preview"

# ---------------------------------------------------------------------------
# Camera angle presets — same as run_sketch_to_shoe_gemini.py
# ---------------------------------------------------------------------------

CAMERA_PRESETS: dict[str, dict[str, dict[str, str]]] = {
    "3/4": {
        "pair": {
            "camera_desc": (
                "Low, ground-level angle, shooting almost parallel to the platform. "
                "The camera is positioned slightly to the right, looking leftwards "
                "at the shoes (3/4)."
            ),
            "staging_desc": (
                "A pair of shoes perfectly aligned and parallel, touching side by side, "
                "both pointing in the same direction at the same angle."
            ),
        },
        "left": {
            "camera_desc": (
                "Low, ground-level angle, shooting almost parallel to the platform. "
                "The camera is positioned slightly to the left, looking rightwards "
                "at the left shoe (3/4 from the left)."
            ),
            "staging_desc": "The left shoe centered on the white surface.",
        },
        "right": {
            "camera_desc": (
                "Low, ground-level angle, shooting almost parallel to the platform. "
                "The camera is positioned slightly to the right, looking leftwards "
                "at the right shoe (3/4 from the right)."
            ),
            "staging_desc": "The right shoe centered on the white surface.",
        },
    },
    "side": {
        "pair": {
            "camera_desc": (
                "Straight side-on profile view, camera at shoe mid-height, "
                "perpendicular to the shoes' length axis. The full silhouette "
                "from toe to heel is visible for both shoes."
            ),
            "staging_desc": (
                "A pair of shoes placed side by side on the white surface showing "
                "their complete lateral profile."
            ),
        },
        "left": {
            "camera_desc": (
                "Straight side-on profile view from the left (medial side), "
                "camera at shoe mid-height, perpendicular to the left shoe's "
                "length axis."
            ),
            "staging_desc": "The left shoe centered, medial side facing the camera.",
        },
        "right": {
            "camera_desc": (
                "Straight side-on profile view from the right (lateral side), "
                "camera at shoe mid-height, perpendicular to the right shoe's "
                "length axis."
            ),
            "staging_desc": "The right shoe centered, lateral side facing the camera.",
        },
    },
    "front": {
        "pair": {
            "camera_desc": (
                "Head-on front view at shoe mid-height, camera looking straight "
                "at the toe boxes. Both shoes are visible, slightly angled outward."
            ),
            "staging_desc": (
                "A pair of shoes placed next to each other, toe boxes facing "
                "the camera."
            ),
        },
        "left": {
            "camera_desc": (
                "Head-on front view at shoe mid-height, camera looking straight "
                "at the toe box of the left shoe."
            ),
            "staging_desc": "The left shoe centered, toe box facing the camera.",
        },
        "right": {
            "camera_desc": (
                "Head-on front view at shoe mid-height, camera looking straight "
                "at the toe box of the right shoe."
            ),
            "staging_desc": "The right shoe centered, toe box facing the camera.",
        },
    },
    "back": {
        "pair": {
            "camera_desc": (
                "Rear view at shoe mid-height, camera looking straight at the heels. "
                "Both shoes are visible, showing the heel counter and back stitching."
            ),
            "staging_desc": (
                "A pair of shoes placed next to each other, heels facing the camera."
            ),
        },
        "left": {
            "camera_desc": (
                "Rear view at shoe mid-height, camera looking straight at the heel "
                "of the left shoe, showing the heel counter and back stitching."
            ),
            "staging_desc": "The left shoe centered, heel facing the camera.",
        },
        "right": {
            "camera_desc": (
                "Rear view at shoe mid-height, camera looking straight at the heel "
                "of the right shoe, showing the heel counter and back stitching."
            ),
            "staging_desc": "The right shoe centered, heel facing the camera.",
        },
    },
    "top": {
        "pair": {
            "camera_desc": (
                "Directly overhead bird's-eye view, camera pointing straight down. "
                "The full footbed outline from toe to heel is visible for both shoes."
            ),
            "staging_desc": (
                "A pair of shoes placed next to each other on the white surface, "
                "viewed from above."
            ),
        },
        "left": {
            "camera_desc": (
                "Directly overhead bird's-eye view, camera pointing straight down "
                "at the left shoe. The full footbed outline from toe to heel is visible."
            ),
            "staging_desc": "The left shoe centered on the white surface, viewed from above.",
        },
        "right": {
            "camera_desc": (
                "Directly overhead bird's-eye view, camera pointing straight down "
                "at the right shoe. The full footbed outline from toe to heel is visible."
            ),
            "staging_desc": "The right shoe centered on the white surface, viewed from above.",
        },
    },
    "hero-front-right": {
        "pair": {
            "camera_desc": (
                "Dynamic low hero shot (~30° elevation from ground). "
                "In the output image: the toe boxes of both shoes point toward the LEFT side of the frame; "
                "the heels recede toward the RIGHT-BACK. "
                "The RIGHT/OUTER sides of both shoes face the camera — the left/inner sides are turned away and NOT visible."
            ),
            "staging_desc": (
                "A pair of shoes on the white surface filling the frame with dramatic presence. "
                "Slight Dutch tilt."
            ),
        },
        "left": {
            "camera_desc": (
                "Dynamic low hero shot (~30° elevation from ground). "
                "In the output image: the toe box points toward the LEFT side of the frame; "
                "the heel recedes toward the RIGHT-BACK. "
                "The OUTER/LATERAL side of the left shoe faces the camera — "
                "the inner/medial (arch) side is turned away and NOT visible."
            ),
            "staging_desc": (
                "The left shoe filling the frame with dramatic presence. "
                "Slight Dutch tilt."
            ),
        },
        "right": {
            "camera_desc": (
                "Dynamic low hero shot (~30° elevation from ground). "
                "In the output image: the toe box points toward the LEFT side of the frame; "
                "the heel recedes toward the RIGHT-BACK. "
                "The INNER/MEDIAL (arch) side of the right shoe faces the camera — "
                "the outer/lateral side is turned away and NOT visible."
            ),
            "staging_desc": (
                "The right shoe filling the frame with dramatic presence. "
                "Slight Dutch tilt."
            ),
        },
    },
    "hero-front-left": {
        "pair": {
            "camera_desc": (
                "Dynamic low hero shot (~30° elevation from ground). "
                "In the output image: the toe boxes of both shoes point toward the RIGHT side of the frame; "
                "the heels recede toward the LEFT-BACK. "
                "The LEFT/INNER sides of both shoes face the camera — the right/outer sides are turned away and NOT visible."
            ),
            "staging_desc": (
                "A pair of shoes on the white surface filling the frame with dramatic presence. "
                "Slight Dutch tilt."
            ),
        },
        "left": {
            "camera_desc": (
                "Dynamic low hero shot (~30° elevation from ground). "
                "In the output image: the toe box points toward the RIGHT side of the frame; "
                "the heel recedes toward the LEFT-BACK. "
                "The INNER/MEDIAL (arch) side of the left shoe faces the camera — "
                "the outer/lateral side is turned away and NOT visible."
            ),
            "staging_desc": (
                "The left shoe filling the frame with dramatic presence. "
                "Slight Dutch tilt."
            ),
        },
        "right": {
            "camera_desc": (
                "Dynamic low hero shot (~30° elevation from ground). "
                "In the output image: the toe box points toward the RIGHT side of the frame; "
                "the heel recedes toward the LEFT-BACK. "
                "The OUTER/LATERAL side of the right shoe faces the camera — "
                "the inner/medial (arch) side is turned away and NOT visible."
            ),
            "staging_desc": (
                "The right shoe filling the frame with dramatic presence. "
                "Slight Dutch tilt."
            ),
        },
    },
    "hero-back-right": {
        "pair": {
            "camera_desc": (
                "Dynamic low hero shot (~30° elevation from ground). "
                "In the output image: the heel counters are in the LEFT-FRONT area of the frame; "
                "the toe boxes recede toward the RIGHT-BACK. "
                "The RIGHT/OUTER sides of both shoes face the camera — the left/inner sides are turned away and NOT visible."
            ),
            "staging_desc": (
                "A pair of shoes on the white surface filling the frame with dramatic presence. "
                "Slight Dutch tilt."
            ),
        },
        "left": {
            "camera_desc": (
                "Dynamic low hero shot (~30° elevation from ground). "
                "In the output image: the heel counter is in the LEFT-FRONT area of the frame; "
                "the toe recedes toward the RIGHT-BACK. "
                "The OUTER/LATERAL side of the left shoe faces the camera — "
                "the inner/medial (arch) side is turned away and NOT visible."
            ),
            "staging_desc": (
                "The left shoe filling the frame with dramatic presence. "
                "Slight Dutch tilt."
            ),
        },
        "right": {
            "camera_desc": (
                "Dynamic low hero shot (~30° elevation from ground). "
                "In the output image: the heel counter is in the LEFT-FRONT area of the frame; "
                "the toe recedes toward the RIGHT-BACK. "
                "The INNER/MEDIAL (arch) side of the right shoe faces the camera — "
                "the outer/lateral side is turned away and NOT visible."
            ),
            "staging_desc": (
                "The right shoe filling the frame with dramatic presence. "
                "Slight Dutch tilt."
            ),
        },
    },
    "hero-back-left": {
        "pair": {
            "camera_desc": (
                "Dynamic low hero shot (~30° elevation from ground). "
                "In the output image: the heel counters are in the RIGHT-FRONT area of the frame; "
                "the toe boxes recede toward the LEFT-BACK. "
                "The LEFT/INNER sides of both shoes face the camera — the right/outer sides are turned away and NOT visible."
            ),
            "staging_desc": (
                "A pair of shoes on the white surface filling the frame with dramatic presence. "
                "Slight Dutch tilt."
            ),
        },
        "left": {
            "camera_desc": (
                "Dynamic low hero shot (~30° elevation from ground). "
                "In the output image: the heel counter is in the RIGHT-FRONT area of the frame; "
                "the toe recedes toward the LEFT-BACK. "
                "The INNER/MEDIAL (arch) side of the left shoe faces the camera — "
                "the outer/lateral side is turned away and NOT visible."
            ),
            "staging_desc": (
                "The left shoe filling the frame with dramatic presence. "
                "Slight Dutch tilt."
            ),
        },
        "right": {
            "camera_desc": (
                "Dynamic low hero shot (~30° elevation from ground). "
                "In the output image: the heel counter is in the RIGHT-FRONT area of the frame; "
                "the toe recedes toward the LEFT-BACK. "
                "The OUTER/LATERAL side of the right shoe faces the camera — "
                "the inner/medial (arch) side is turned away and NOT visible."
            ),
            "staging_desc": (
                "The right shoe filling the frame with dramatic presence. "
                "Slight Dutch tilt."
            ),
        },
    },
}

# Aliases
CAMERA_PRESETS["3/4 view"] = CAMERA_PRESETS["3/4"]
CAMERA_PRESETS["side view"] = CAMERA_PRESETS["side"]
CAMERA_PRESETS["front view"] = CAMERA_PRESETS["front"]
CAMERA_PRESETS["back view"] = CAMERA_PRESETS["back"]
CAMERA_PRESETS["top view"] = CAMERA_PRESETS["top"]
CAMERA_PRESETS["hero"] = CAMERA_PRESETS["hero-front-right"]

MAX_JUDGE_ITERATIONS = 3

_CAMERA_JUDGE_NOTES = (
    "EVALUATE camera_angle using these steps in order:\n\n"
    "STEP A — OBSERVE FIRST.\n"
    "  Write your answers into observations[\"camera_angle\"] before scoring:\n"
    "  - Which direction does the TOE BOX point in the frame? (LEFT / RIGHT / CENTER / AWAY)\n"
    "  - Which part of the shoe is the primary subject facing the camera? "
    "(toe box front / heel back / outer side / inner side / top)\n\n"
    "STEP B — COMPARE to the spec.\n"
    "  Read the spec carefully. Check ONLY what the spec explicitly states — "
    "do not invent additional requirements that are not mentioned.\n"
    "  Score 4-5: what you observed in Step A satisfies the spec's stated requirements.\n"
    "  Score 3: the angle is mostly right but has a minor deviation from the spec.\n"
    "  Score 1-2: the angle is clearly and fundamentally wrong "
    "(e.g. spec says toe-front but heel is facing camera, or spec says left but toe points right).\n"
    "  IMPORTANT: Be lenient with head-on front/rear/top views. "
    "For a front view, seeing the interior (insole, straps) is normal and correct — "
    "do NOT penalise a front view for showing the inside of the shoe. "
    "Do NOT penalise camera height, exact elevation angle, Dutch tilt, or minor staging variations.\n\n"
    "STEP C — REPAIR (only if score ≤ 3):\n"
    "  State one concrete fix describing the required output state. "
    "For angled shots: name which direction the toe must point (LEFT / RIGHT / CENTER) "
    "and which part of the shoe should face the camera. "
    "Do NOT write 'rotate' or 'adjust the camera' — only describe what the final image must show."
)

CANONICAL_ANGLES = [
    "3/4", "side", "front", "back", "top",
    "hero-front-right", "hero-front-left", "hero-back-right", "hero-back-left",
]

# Angles that naturally show a pair vs a single shoe
PAIR_ANGLES = {"3/4", "front"}
SINGLE_ANGLES = {
    "side", "back", "top",
    "hero-front-right", "hero-front-left", "hero-back-right", "hero-back-left",
}


def _foot_for_angle(angle: str, foot: str, single: bool = False) -> str:
    """Return the foot variation to use for a given angle.

    Pair angles use 'pair' unless --single is set. Single angles use the
    --foot value (left or right).
    """
    if single:
        return foot
    key = angle.lower().strip()
    if key in PAIR_ANGLES:
        return "pair"
    return foot

# ---------------------------------------------------------------------------
# Aspect-ratio helpers (same logic as gemini_flash_image_edit.py)
# ---------------------------------------------------------------------------

MAX_INPUT_SIZE = 1024

_SUPPORTED_RATIOS: list[tuple[int, int]] = [
    (1, 1), (1, 4), (1, 8),
    (2, 3), (3, 2), (3, 4), (4, 3),
    (4, 5), (5, 4),
    (8, 1), (9, 16), (16, 9), (21, 9),
]


def _find_ratio(w: int, h: int) -> tuple[int, int]:
    target = w / h
    return min(_SUPPORTED_RATIOS, key=lambda r: abs(r[0] / r[1] - target))


def _pad_to_ratio(img: PILImage.Image, ratio: tuple[int, int]) -> PILImage.Image:
    wr, hr = ratio
    orig_w, orig_h = img.size

    if orig_w / orig_h <= wr / hr:
        canvas_h = orig_h
        canvas_w = round(orig_h * wr / hr)
    else:
        canvas_w = orig_w
        canvas_h = round(orig_w * hr / wr)

    scale = MAX_INPUT_SIZE / max(canvas_w, canvas_h)
    final_w = round(canvas_w * scale)
    final_h = round(canvas_h * scale)

    img_w = round(orig_w * scale)
    img_h = round(orig_h * scale)
    scaled = img.resize((img_w, img_h), PILImage.LANCZOS)

    canvas = PILImage.new("RGB", (final_w, final_h), (255, 255, 255))
    canvas.paste(scaled, ((final_w - img_w) // 2, (final_h - img_h) // 2))
    return canvas


# ---------------------------------------------------------------------------
# Foot framing
# ---------------------------------------------------------------------------

def _foot_framing(foot: str) -> str:
    if foot == "pair":
        return (
            "IMPORTANT — NUMBER OF SHOES: The output MUST contain exactly TWO shoes "
            "(a matching pair — left and right) placed side by side. "
            "The reference image shows a pair; keep both shoes in the output. "
            "Do NOT show only one shoe."
        )
    elif foot == "left":
        return (
            "IMPORTANT — NUMBER OF SHOES: The output MUST contain exactly ONE shoe — "
            "the LEFT shoe only, centered in the frame. "
            "Even though the reference image may show two shoes, generate ONLY the "
            "left shoe. Do NOT include the right shoe. Only one shoe in the image."
        )
    else:
        return (
            "IMPORTANT — NUMBER OF SHOES: The output MUST contain exactly ONE shoe — "
            "the RIGHT shoe only, centered in the frame. "
            "Even though the reference image may show two shoes, generate ONLY the "
            "right shoe. Do NOT include the left shoe. Only one shoe in the image."
        )


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = (
    "The first image is the original shoe design sketch. "
    "The second image is a photorealistic product photograph of this exact shoe — "
    "use it as the definitive reference for everything: material, color, texture, "
    "proportions, and design details.\n\n"
    "Generate the exact same shoe from this camera angle:\n"
    "- Camera angle: {camera_desc}\n"
    "- Shoe alignment and staging: {staging_desc}\n\n"
    "{foot_framing}\n\n"
    "The result must be a studio-quality photograph: clean white background, "
    "professional product lighting, sharp focus, no shadows on background. "
    "The shoe must look identical to the reference — only the viewing angle changes."
    "{feedback}"
)




# ---------------------------------------------------------------------------
# Single-angle generation
# ---------------------------------------------------------------------------

def generate_angle(
    client: genai.Client,
    sketch: PILImage.Image,
    reference: PILImage.Image,
    angle: str,
    foot: str,
    single: bool = False,
    feedback: str = "",
) -> tuple[str, PILImage.Image, dict]:
    """Generate a single angle. Returns (angle_name, result_image)."""
    effective_foot = _foot_for_angle(angle, foot, single=single)
    key = angle.lower().strip()
    if key in CAMERA_PRESETS:
        preset = CAMERA_PRESETS[key][effective_foot]
    else:
        preset = {
            "camera_desc": angle,
            "staging_desc": "The shoe(s) placed on the white surface.",
        }

    prompt = PROMPT_TEMPLATE.format(
        camera_desc=preset["camera_desc"],
        staging_desc=preset["staging_desc"],
        foot_framing=_foot_framing(effective_foot),
        feedback=f"\n\n{feedback}" if feedback else "",
    )

    logger.info("=== generate_angle: %s (foot=%s) ===", angle, effective_foot)
    logger.info("PROMPT:\n%s", prompt)

    # Pad both images to the same ratio
    ratio = _find_ratio(*reference.size)
    aspect_ratio_str = f"{ratio[0]}:{ratio[1]}"
    padded_sketch = _pad_to_ratio(sketch, ratio)
    padded_ref = _pad_to_ratio(reference, ratio)
    target_size = padded_ref.size

    logger.info("Aspect ratio: %s, target size: %s", aspect_ratio_str, target_size)

    response = client.models.generate_content(
        model="gemini-3.1-flash-image-preview",
        contents=[prompt, padded_sketch, padded_ref],
        config=genai_types.GenerateContentConfig(
            temperature=1.0,
            image_config=genai_types.ImageConfig(
                image_size="1K",
                aspect_ratio=aspect_ratio_str,
            ),
        ),
    )

    # Log any text parts from response
    for part in response.parts:
        if part.text:
            logger.info("API text response for %s: %s", angle, part.text)

    gen_usage = extract_token_usage(getattr(response, "usage_metadata", None))
    gen_usage["model"] = _GENERATION_MODEL_ID

    for part in response.parts:
        if part.inline_data is not None:
            result = PILImage.open(io.BytesIO(bytes(part.inline_data.data)))
            result.load()  # force full pixel load before response/buffer is GC'd
            logger.info("Got image for %s: %sx%s", angle, result.width, result.height)
            if result.size != target_size:
                result = result.resize(target_size, PILImage.LANCZOS)
            return angle, result, gen_usage

    raise RuntimeError(
        f"No image returned by Gemini API for angle '{angle}'. "
        "The model may have refused the request or returned text only."
    )


def _annotate_image(
    img: PILImage.Image,
    angle: str,
    iteration: int,
    angle_ok: bool,
    ref_ok: bool,
    count_ok: bool = True,
) -> PILImage.Image:
    """Return a copy of img with a small info banner at the top."""
    from PIL import ImageDraw, ImageFont
    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    label = (
        f"{angle}  iter={iteration}  "
        f"cam={'OK' if angle_ok else 'FAIL'}  "
        f"ref={'OK' if ref_ok else 'FAIL'}  "
        f"count={'OK' if count_ok else 'FAIL'}"
    )
    banner_h = 28
    draw.rectangle([0, 0, out.width, banner_h], fill=(30, 30, 30))
    draw.text((6, 4), label, fill=(255, 255, 255), font=font)
    return out


def generate_angle_with_judge(
    client: genai.Client,
    sketch: PILImage.Image,
    reference: PILImage.Image,
    angle: str,
    foot: str,
    single: bool = False,
    output_dir: Path | None = None,
) -> tuple[str, PILImage.Image, float]:
    """Generate a single angle with three parallel judges (camera angle + reference fidelity + shoe count).

    All judges run concurrently per iteration. Accepted only if all three pass.
    Combined feedback is injected into the next generation prompt.
    """
    effective_foot = _foot_for_angle(angle, foot, single=single)
    key = angle.lower().strip()
    preset = CAMERA_PRESETS.get(key, {}).get(effective_foot, {
        "camera_desc": angle,
        "staging_desc": "The shoe(s) placed on the white surface.",
    })

    reference_media = ImageMedia(image=reference)
    sketch_media = ImageMedia(image=sketch)
    session_angle = VLMSession("gemini_flash")
    session_ref = VLMSession("gemini_flash")
    session_count = VLMSession("gemini_flash_lite")

    camera_judge = make_spec_judge(
        session=session_angle,
        candidate_key="image",
        spec={"camera_angle": preset["camera_desc"] + " " + preset["staging_desc"]},
        tolerance="generous",
        include_quality_features=False,
        judge_notes=_CAMERA_JUDGE_NOTES,
    )
    angle_safe = angle.replace("/", "_")
    logs_dir = (output_dir / "logs") if output_dir is not None else None
    if logs_dir is not None:
        logs_dir.mkdir(exist_ok=True)
    ref_judge = make_reference_fidelity_judge(
        session=session_ref,
        reference_image=reference_media,
        sketch_image=sketch_media,
        candidate_key="image",
        tolerance="moderate",
        save_dir=logs_dir,
        angle_name=angle_safe,
    )
    count_judge = make_shoe_count_judge(
        session=session_count,
        foot=effective_foot,
        candidate_key="image",
    )

    feedback = ""
    last_img = None
    judge_log: list[dict] = []
    total_gen_cost = 0.0
    total_judge_cost = 0.0
    try:
        for iteration in range(MAX_JUDGE_ITERATIONS):
            # Snapshot token log lengths before this iteration's judges
            len_angle_before = len(session_angle.token_usage_log)
            len_ref_before = len(session_ref.token_usage_log)
            len_count_before = len(session_count.token_usage_log)

            _, img, gen_usage = generate_angle(client, sketch, reference, angle, foot, single, feedback)
            gen_cost = calculate_cost(_GENERATION_MODEL_ID, gen_usage)
            total_gen_cost += gen_cost
            last_img = img
            image_media = ImageMedia(image=img)

            # Run all three judges concurrently
            ctx_angle: dict = {"image": image_media}
            ctx_ref: dict = {"image": image_media}
            ctx_count: dict = {"image": image_media}
            with ThreadPoolExecutor(max_workers=3) as judge_pool:
                fut_angle = judge_pool.submit(camera_judge, ctx_angle)
                fut_ref = judge_pool.submit(ref_judge, ctx_ref)
                fut_count = judge_pool.submit(count_judge, ctx_count)
                angle_accepted, angle_fb = fut_angle.result()
                ref_accepted, ref_fb = fut_ref.result()
                count_accepted, count_fb = fut_count.result()

            camera_detail = ctx_angle.pop("_judge_detail_spec", {})
            ref_detail = ctx_ref.pop("_judge_detail_ref", {})
            count_detail = ctx_count.pop("_judge_detail_count", {})

            # Compute judge costs from new token log entries
            new_judge_records = (
                session_angle.token_usage_log[len_angle_before:]
                + session_ref.token_usage_log[len_ref_before:]
                + session_count.token_usage_log[len_count_before:]
            )
            judge_cost = sum(calculate_cost(r.get("model", ""), r) for r in new_judge_records)
            total_judge_cost += judge_cost

            accepted = angle_accepted and ref_accepted and count_accepted
            verdict = "ACCEPT" if accepted else "REJECT"
            print(f"  [{angle}] iteration {iteration + 1}/{MAX_JUDGE_ITERATIONS}: {verdict} "
                  f"(angle={'OK' if angle_accepted else 'FAIL'}, "
                  f"ref={'OK' if ref_accepted else 'FAIL'}, "
                  f"count={'OK' if count_accepted else 'FAIL'})")

            judge_log.append({
                "iteration": iteration + 1,
                "verdict": verdict,
                "cost": {
                    "generation_usd": round(gen_cost, 6),
                    "judges_usd": round(judge_cost, 6),
                    "total_usd": round(gen_cost + judge_cost, 6),
                },
                "camera_judge": {
                    "accepted": angle_accepted,
                    "feedback": angle_fb,
                    **camera_detail,
                },
                "reference_judge": {
                    "accepted": ref_accepted,
                    "feedback": ref_fb,
                    **ref_detail,
                },
                "count_judge": {
                    "accepted": count_accepted,
                    "feedback": count_fb,
                    **count_detail,
                },
                "feedback_injected": "",
            })

            if logs_dir is not None:
                try:
                    annotated = _annotate_image(
                        img, angle, iteration + 1, angle_accepted, ref_accepted, count_accepted
                    )
                    annotated.save(logs_dir / f"{angle_safe}_generated_iter{iteration + 1}.png")
                except Exception as _e:
                    print(f"  [{angle}] Warning: could not save annotated image: {_e}")

            if accepted:
                total_cost_usd = round(total_gen_cost + total_judge_cost, 6)
                return angle, img, total_cost_usd

            parts = []
            if not count_accepted and count_fb and count_fb != "none":
                parts.append(f"Shoe count/foot issue: {count_fb}")
            if not angle_accepted and angle_fb and angle_fb != "none":
                if ref_accepted and count_accepted:
                    # Materials and count are fine — protect them while fixing angle
                    parts.append(
                        "CRITICAL: Keep ALL materials, colors, textures, and design elements "
                        "IDENTICAL to the reference image — do NOT change any design aspect. "
                        "Only correct the camera angle as described below."
                    )
                parts.append(f"Camera angle issue: {angle_fb}")
            if not ref_accepted and ref_fb and ref_fb != "none":
                parts.append(f"Design fidelity issue: {ref_fb}")
            feedback = "\n".join(parts)
            if judge_log:
                judge_log[-1]["feedback_injected"] = feedback
    finally:
        session_angle.unload()
        session_ref.unload()
        session_count.unload()
        total_cost_usd = round(total_gen_cost + total_judge_cost, 6)
        if logs_dir is not None and judge_log:
            try:
                log_path = logs_dir / f"{angle_safe}_judge_log.json"
                log_data = {
                    "angle": angle,
                    "total_cost_usd": total_cost_usd,
                    "total_generation_cost_usd": round(total_gen_cost, 6),
                    "total_judges_cost_usd": round(total_judge_cost, 6),
                    "iterations": judge_log,
                }
                log_path.write_text(json.dumps(log_data, indent=2))
            except Exception as _e:
                print(f"  [{angle}] Warning: could not save judge log: {_e}")

    # All iterations exhausted without acceptance — return last attempt.
    return angle, last_img, total_cost_usd



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate shoe images at multiple camera angles using Gemini Flash"
    )
    parser.add_argument("--sketch", type=str, required=True,
        help="Path to the design sketch image")
    parser.add_argument("--reference", type=str, required=True,
        help="Path to the photorealistic reference image (e.g. final_result.png)")
    parser.add_argument("--foot", type=str, default="right",
        choices=["left", "right"],
        help="Which foot for single-shoe angles (side, back, top, hero). "
             "Default: right. Pair angles (3/4, front) show both shoes unless --single is used.")
    parser.add_argument("--single", action="store_true",
        help="Force all angles to show only one shoe (the --foot shoe). "
             "Without this flag, 3/4 and front angles show a pair.")

    angle_group = parser.add_mutually_exclusive_group(required=True)
    angle_group.add_argument("--angles", type=str, nargs="+",
        help=f"Camera angle(s) to generate. Available: {', '.join(CANONICAL_ANGLES)}")
    angle_group.add_argument("--all-angles", action="store_true",
        help="Generate all camera angle presets concurrently")

    parser.add_argument("--output-dir", type=str, default=None,
        help="Output directory (default: tests/output/shoe_angles/<timestamp>)")
    parser.add_argument("--workers", type=int, default=10,
        help="Max concurrent API calls when using --all-angles (default: 10)")
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not set. Add it to your .env file.")
        return

    # Determine which angles to generate
    angles = CANONICAL_ANGLES if args.all_angles else args.angles

    # Load images
    sketch = PILImage.open(args.sketch).convert("RGB")
    reference = PILImage.open(args.reference).convert("RGB")

    # Output directory
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(__file__).parent / "output" / "shoe_angles" / ts
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up file logging for this run
    log_path = output_dir / "run.log"
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

    print("=== Shoe Angle Generator — Gemini ===")
    print(f"Sketch:     {args.sketch}")
    print(f"Reference:  {args.reference}")
    print(f"Foot:       {args.foot} ({'all angles' if args.single else 'single-shoe angles only'})")
    print(f"Angles:     {', '.join(angles)}")
    print(f"Concurrent: {len(angles) > 1}")
    print(f"Output:     {output_dir}")
    print(f"Log:        {log_path}")
    print()

    logger.info("Sketch: %s", args.sketch)
    logger.info("Reference: %s", args.reference)
    logger.info("Foot: %s, Single: %s", args.foot, args.single)

    # Save inputs for reference
    sketch.save(output_dir / "input_sketch.png")
    reference.save(output_dir / "input_reference.png")

    client = genai.Client()
    results: dict[str, dict] = {}
    t0 = time.perf_counter()

    logger.info("Angles: %s", angles)

    def _run_and_save(angle: str) -> tuple[str, PILImage.Image | None]:
        """Generate an angle, save it, record result."""
        at = time.perf_counter()
        try:
            _, img, angle_cost = generate_angle_with_judge(
                client, sketch, reference, angle, args.foot,
                single=args.single, output_dir=output_dir,
            )
            elapsed = time.perf_counter() - at
            fname = f"{angle.replace('/', '_')}.png"
            img.save(output_dir / fname)
            results[angle] = {"file": fname, "elapsed_s": round(elapsed, 1), "cost_usd": angle_cost}
            print(f"  {angle}: done ({elapsed:.1f}s)  cost=${angle_cost:.6f}")
            return angle, img
        except Exception as e:
            results[angle] = {"error": str(e)}
            print(f"  {angle}: FAILED — {e}")
            return angle, None

    if len(angles) == 1:
        _run_and_save(angles[0])
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_and_save, a): a for a in angles}
            for fut in as_completed(futures):
                fut.result()

    total_elapsed = time.perf_counter() - t0

    # Aggregate total cost across all angles
    total_cost_usd = round(sum(
        r.get("cost_usd", 0.0) for r in results.values() if "file" in r
    ), 6)

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_elapsed_s": round(total_elapsed, 1),
        "total_cost_usd": total_cost_usd,
        "sketch": args.sketch,
        "reference": args.reference,
        "foot": args.foot,
        "angles": results,
    }
    (output_dir / "results.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    success_count = len([r for r in results.values() if 'file' in r])
    logger.info("SUMMARY: %d/%d angles generated in %.1fs  cost=$%.6f",
                success_count, len(angles), total_elapsed, total_cost_usd)
    logger.info("Results: %s", json.dumps(results, indent=2, default=str))

    print(f"\nDone. {success_count}/{len(angles)} "
          f"angles generated in {total_elapsed:.1f}s  |  total cost: ${total_cost_usd:.6f}")
    print(f"Output: {output_dir}")
    print(f"Log:    {log_path}")


if __name__ == "__main__":
    main()
