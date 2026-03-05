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
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image as PILImage

# Load .env so GEMINI_API_KEY is available before any genai import
load_dotenv()

from google import genai
from google.genai import types as genai_types

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
                "Dynamic hero shot: camera low at roughly 30 degrees from the ground, "
                "positioned at the front-right of the shoes, angled upward. "
                "Slight Dutch tilt for editorial drama. The toe box and right side "
                "of the shoes are prominent."
            ),
            "staging_desc": (
                "A pair of shoes on the white surface, filling the frame with presence."
            ),
        },
        "left": {
            "camera_desc": (
                "Dynamic hero shot: camera low at roughly 30 degrees from the ground, "
                "positioned at the front-right of the left shoe, angled upward. "
                "Slight Dutch tilt for editorial drama. The toe box and outer side "
                "of the left shoe are prominent."
            ),
            "staging_desc": "The left shoe filling the frame with presence.",
        },
        "right": {
            "camera_desc": (
                "Dynamic hero shot: camera low at roughly 30 degrees from the ground, "
                "positioned at the front-right of the right shoe, angled upward. "
                "Slight Dutch tilt for editorial drama. The toe box and outer side "
                "of the right shoe are prominent."
            ),
            "staging_desc": "The right shoe filling the frame with presence.",
        },
    },
    "hero-front-left": {
        "pair": {
            "camera_desc": (
                "Dynamic hero shot: camera low at roughly 30 degrees from the ground, "
                "positioned at the front-left of the shoes, angled upward. "
                "Slight Dutch tilt for editorial drama. The toe box and left side "
                "of the shoes are prominent."
            ),
            "staging_desc": (
                "A pair of shoes on the white surface, filling the frame with presence."
            ),
        },
        "left": {
            "camera_desc": (
                "Dynamic hero shot: camera low at roughly 30 degrees from the ground, "
                "positioned at the front-left of the left shoe, angled upward. "
                "Slight Dutch tilt for editorial drama. The toe box and inner side "
                "of the left shoe are prominent."
            ),
            "staging_desc": "The left shoe filling the frame with presence.",
        },
        "right": {
            "camera_desc": (
                "Dynamic hero shot: camera low at roughly 30 degrees from the ground, "
                "positioned at the front-left of the right shoe, angled upward. "
                "Slight Dutch tilt for editorial drama. The toe box and inner side "
                "of the right shoe are prominent."
            ),
            "staging_desc": "The right shoe filling the frame with presence.",
        },
    },
    "hero-back-right": {
        "pair": {
            "camera_desc": (
                "Dynamic hero shot: camera low at roughly 30 degrees from the ground, "
                "positioned at the back-right of the shoes, angled upward. "
                "Slight Dutch tilt for editorial drama. The heel counter and right side "
                "of the shoes are prominent."
            ),
            "staging_desc": (
                "A pair of shoes on the white surface, filling the frame with presence."
            ),
        },
        "left": {
            "camera_desc": (
                "Dynamic hero shot: camera low at roughly 30 degrees from the ground, "
                "positioned at the back-right of the left shoe, angled upward. "
                "Slight Dutch tilt for editorial drama. The heel and outer side "
                "of the left shoe are prominent."
            ),
            "staging_desc": "The left shoe filling the frame with presence.",
        },
        "right": {
            "camera_desc": (
                "Dynamic hero shot: camera low at roughly 30 degrees from the ground, "
                "positioned at the back-right of the right shoe, angled upward. "
                "Slight Dutch tilt for editorial drama. The heel and outer side "
                "of the right shoe are prominent."
            ),
            "staging_desc": "The right shoe filling the frame with presence.",
        },
    },
    "hero-back-left": {
        "pair": {
            "camera_desc": (
                "Dynamic hero shot: camera low at roughly 30 degrees from the ground, "
                "positioned at the back-left of the shoes, angled upward. "
                "Slight Dutch tilt for editorial drama. The heel counter and left side "
                "of the shoes are prominent."
            ),
            "staging_desc": (
                "A pair of shoes on the white surface, filling the frame with presence."
            ),
        },
        "left": {
            "camera_desc": (
                "Dynamic hero shot: camera low at roughly 30 degrees from the ground, "
                "positioned at the back-left of the left shoe, angled upward. "
                "Slight Dutch tilt for editorial drama. The heel and inner side "
                "of the left shoe are prominent."
            ),
            "staging_desc": "The left shoe filling the frame with presence.",
        },
        "right": {
            "camera_desc": (
                "Dynamic hero shot: camera low at roughly 30 degrees from the ground, "
                "positioned at the back-left of the right shoe, angled upward. "
                "Slight Dutch tilt for editorial drama. The heel and inner side "
                "of the right shoe are prominent."
            ),
            "staging_desc": "The right shoe filling the frame with presence.",
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
) -> tuple[str, PILImage.Image]:
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
    )

    # Pad both images to the same ratio
    ratio = _find_ratio(*reference.size)
    aspect_ratio_str = f"{ratio[0]}:{ratio[1]}"
    padded_sketch = _pad_to_ratio(sketch, ratio)
    padded_ref = _pad_to_ratio(reference, ratio)
    target_size = padded_ref.size

    response = client.models.generate_content(
        model="gemini-3.1-flash-image-preview",
        contents=[prompt, padded_sketch, padded_ref],
        config=genai_types.GenerateContentConfig(
            image_config=genai_types.ImageConfig(
                image_size="1K",
                aspect_ratio=aspect_ratio_str,
            ),
        ),
    )

    for part in response.parts:
        if part.inline_data is not None:
            result = PILImage.open(io.BytesIO(part.inline_data.data))
            if result.size != target_size:
                result = result.resize(target_size, PILImage.LANCZOS)
            return angle, result

    raise RuntimeError(
        f"No image returned by Gemini API for angle '{angle}'. "
        "The model may have refused the request or returned text only."
    )


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

    print("=== Shoe Angle Generator — Gemini ===")
    print(f"Sketch:     {args.sketch}")
    print(f"Reference:  {args.reference}")
    print(f"Foot:       {args.foot} ({'all angles' if args.single else 'single-shoe angles only'})")
    print(f"Angles:     {', '.join(angles)}")
    print(f"Concurrent: {len(angles) > 1}")
    print(f"Output:     {output_dir}")
    print()

    # Save inputs for reference
    sketch.save(output_dir / "input_sketch.png")
    reference.save(output_dir / "input_reference.png")

    client = genai.Client()
    results: dict[str, dict] = {}
    t0 = time.perf_counter()

    if len(angles) == 1:
        # Single angle — no threading needed
        angle = angles[0]
        print(f"  Generating: {angle} ... ", end="", flush=True)
        at = time.perf_counter()
        _, img = generate_angle(
            client, sketch, reference, angle, args.foot, single=args.single,
        )
        elapsed = time.perf_counter() - at
        fname = f"{angle.replace('/', '_')}.png"
        img.save(output_dir / fname)
        results[angle] = {"file": fname, "elapsed_s": round(elapsed, 1)}
        print(f"done ({elapsed:.1f}s)")
    else:
        # Multiple angles — fire all concurrently
        futures = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for angle in angles:
                fut = pool.submit(
                    generate_angle,
                    client, sketch, reference, angle, args.foot, args.single,
                )
                futures[fut] = angle

            for fut in as_completed(futures):
                angle = futures[fut]
                try:
                    _, img = fut.result()
                    elapsed = time.perf_counter() - t0
                    fname = f"{angle.replace('/', '_')}.png"
                    img.save(output_dir / fname)
                    results[angle] = {"file": fname, "elapsed_s": round(elapsed, 1)}
                    print(f"  {angle}: done ({elapsed:.1f}s)")
                except Exception as e:
                    results[angle] = {"error": str(e)}
                    print(f"  {angle}: FAILED — {e}")

    total_elapsed = time.perf_counter() - t0

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_elapsed_s": round(total_elapsed, 1),
        "sketch": args.sketch,
        "reference": args.reference,
        "foot": args.foot,
        "angles": results,
    }
    (output_dir / "results.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    print(f"\nDone. {len([r for r in results.values() if 'file' in r])}/{len(angles)} "
          f"angles generated in {total_elapsed:.1f}s")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
