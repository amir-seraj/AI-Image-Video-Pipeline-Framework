"""Sketch-to-shoe agentic loop — Gemini version.

Converts design sketches + spec (material, color, camera angle, extras) into
a photorealistic studio product photograph using Gemini Flash Image Edit
for generation and Gemini Flash VLM for dual judgment (sketch fidelity +
spec compliance). Both models are API-based (no GPU required).
Reads GEMINI_API_KEY from the .env file.

Usage:
    python tests/run_sketch_to_shoe_gemini.py --sketches tests/Image/sketch001.png
    python tests/run_sketch_to_shoe_gemini.py \\
        --sketches s1.png s2.png \\
        --material "beige suede" --camera-angle "3/4 view" \\
        --spec style=elegant note="chunky platform"
    python tests/run_sketch_to_shoe_gemini.py --max-iter 3 --tolerance moderate
    python tests/run_sketch_to_shoe_gemini.py --foot pair   # both shoes (default)
    python tests/run_sketch_to_shoe_gemini.py --foot left   # left shoe only
    python tests/run_sketch_to_shoe_gemini.py --foot right  # right shoe only
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image as PILImage

# Load .env so GEMINI_API_KEY is available before any genai import
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "workflows" / "sketch_to_shoe" / "scripts"))

from casadei import (
    Agent, AgentConfig, AgentStep, ImageMedia, TextMedia,
    LoggedPipeline, Pipeline,
)
from casadei.loop import LoopStep, LoopResult
from casadei.providers.gemini_pricing import format_usage_summary
from judge import (
    VLMSession,
    make_sketch_judge,
    make_spec_judge,
    make_dual_judge,
    make_best_fn,
)

IMAGE_DIR = Path(__file__).parent / "Image"
OUTPUT_DIR = Path(__file__).parent / "output" / "sketch_to_shoe_gemini"

PROMPT_TEMPLATE = (
    "Generate a studio product photo of the shoe shown in the sketch.\n"
    "Study the sketch carefully before generating. "
    "Reproduce every line and every open area exactly as drawn — "
    "if a structural part is not drawn in the sketch, it must not appear in the photo. "
    "Do not add, complete, or assume any structure that is absent from the sketch. "
    "Apply the materials below on top of the sketch shape. "
    "Shoot at the specified camera angle — do not copy the sketch's viewpoint.\n\n"
    "Materials: $material\n\n"
    "Camera angle: $camera_desc\n"
    "Staging: $staging_desc\n"
    "$extra_specs\n"
    "Clean white background, professional studio lighting, sharp focus. "
    "$foot_framing\n"
    "$feedback"
)

# ---------------------------------------------------------------------------
# Camera angle presets — deterministic prompt fragments per angle
# ---------------------------------------------------------------------------

CAMERA_PRESETS: dict[str, dict[str, dict[str, str]]] = {
    "3/4": {
        "pair": {
            "camera_desc": (
                "Classic luxury product 3/4 angle: the camera is low (near ground level), "
                "positioned to the front-left of the shoes, looking toward the front-right. "
                "The toe boxes face away from the camera diagonally. "
                "This angle reveals the insole, the inner side profile, and the heel "
                "simultaneously — the classic Casadei product shot perspective."
            ),
            "staging_desc": (
                "A matched pair of shoes placed side by side, BOTH pointing in the SAME "
                "direction — toes facing forward-right, heels toward the camera-left. "
                "The shoes are NOT mirrored and NOT facing each other. "
                "They are parallel, with the left shoe slightly in front and the right "
                "shoe slightly behind, touching side by side."
            ),
            "judge_note": (
                "EVALUATION NOTE: In a correct 3/4 pair shot the left shoe naturally shows "
                "its inner arch/insole and the right shoe shows its outer lateral side — "
                "this asymmetry is CORRECT and must NOT be scored as mirrored. "
                "Score 2 ONLY if the shoes form a V-shape with toes pointing toward each other."
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

# Aliases for convenience
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


def _get_camera_preset(angle: str, foot: str = "pair") -> dict[str, str]:
    """Look up camera preset by angle and foot. Falls back to raw angle string."""
    key = angle.lower().strip()
    if key in CAMERA_PRESETS:
        return CAMERA_PRESETS[key][foot]
    # Fallback: use the raw angle as-is (user provided a custom description)
    return {
        "camera_desc": angle,
        "staging_desc": "The shoe(s) placed on the white surface.",
    }


def _foot_framing(foot: str) -> str:
    if foot == "pair":
        return (
            "Show a matching pair of shoes — both left and right — "
            "centered side by side."
        )
    elif foot == "left":
        return "Show the left shoe only, centered and fully visible."
    else:
        return "Show the right shoe only, centered and fully visible."


def _default_camera_angle(foot: str) -> str:
    return "3/4"


# ---------------------------------------------------------------------------
# Sketch grid assembly
# ---------------------------------------------------------------------------

def _build_sketch_grid(
    images: list[PILImage.Image],
    spacing: int = 20,
) -> PILImage.Image:
    """Arrange sketch images in a rectangular grid, then pad to square."""
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
    """Parse KEY=VALUE strings from --spec CLI args."""
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
    vlm_session: VLMSession | None = None,
    sketch_vlm_session: VLMSession | None = None,
    sketch_features: list[str] | None = None,
    tolerance: str = "strict",
    foot: str = "pair",
    temperature: float = 0.8,
) -> tuple[Pipeline, Agent]:
    """Build the iterative sketch-to-shoe pipeline using Gemini models.

    Returns (pipeline, image_edit_agent) so callers can read the agent's
    token_usage_log after the run.
    """
    if vlm_session is None:
        vlm_session = VLMSession("gemini_flash_lite")
    if sketch_vlm_session is None:
        sketch_vlm_session = vlm_session

    extra_specs_text = _build_extra_specs_text(spec.get("extra", {}))

    gemini_agent = Agent(AgentConfig(
        name="gemini_sketch_to_shoe",
        model="gemini_flash_image_edit",
        description="Gemini Flash image edit for sketch-to-shoe generation",
        prompt_template=PROMPT_TEMPLATE,
        negative_prompt="",
        params={"temperature": temperature},
    ))

    edit_step = AgentStep(
        name="gemini_generate",
        agent=gemini_agent,
        input_map={
            "image": "sketch",  # always the original sketch (constant reference)
            "image_2": "image", # first iter: sketch (seeded); subsequent: last generation
        },
        output_map={"image": "image"},
        template_kwargs={
            "material": spec.get("material", "black patent leather"),
            **_get_camera_preset(spec.get("camera_angle", _default_camera_angle(foot)), foot),
            "extra_specs": extra_specs_text,
            "foot_framing": _foot_framing(foot),
            "feedback": "",
        },
    )

    camera_preset = _get_camera_preset(spec.get("camera_angle", _default_camera_angle(foot)), foot)
    _camera_judge_note = camera_preset.get("judge_note", "")
    full_spec = {
        "material": spec.get("material", "black patent leather"),
        "camera_angle": " ".join(filter(None, [
            camera_preset["camera_desc"],
            "Staging: " + camera_preset["staging_desc"],
        ])),
        **spec.get("extra", {}),
    }

    dual_judge = make_dual_judge(
        make_sketch_judge(
            session=sketch_vlm_session,
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
            judge_notes=_camera_judge_note,
        ),
    )

    loop = LoopStep(
        name="sketch_to_shoe_loop",
        body=[edit_step],
        judge=dual_judge,
        max_iterations=max_iterations,
        best_fn=make_best_fn(
            session=vlm_session,
            sketch_key="sketch",
            output_key="image",
        ),
        swap_models=True,
        output_key="image",
        feedback_template_var="feedback",
    )

    return Pipeline(name="sketch_to_shoe_gemini", steps=[loop]), gemini_agent


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
    sketch_features: list[str] | None = None,
    tolerance: str = "strict",
    foot: str = "pair",
    token_records: list[dict] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    sketch_grid.save(run_dir / "input_sketch_grid.png")

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "total_elapsed_s": total_elapsed,
        "models": {
            "image_edit": "gemini_flash_image_edit",
            "vlm_judge": "gemini_flash_lite",
        },
        "foot": foot,
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
        meta = it.metadata
        if meta:
            iter_data["sketch_scores"] = meta.get("sketch_scores", {})
            iter_data["sketch_avg"] = meta.get("sketch_avg")
            iter_data["spec_scores"] = meta.get("spec_scores", {})
            iter_data["spec_avg"] = meta.get("spec_avg")

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

    # Token usage and pricing
    if token_records:
        usage_summary = format_usage_summary(token_records)
        results_data["token_usage"] = {
            "records": token_records,
            "summary": usage_summary,
        }

    (run_dir / "results.json").write_text(
        json.dumps(results_data, indent=2, default=str)
    )

    lines = [
        "Sketch-to-Shoe Loop Results — Gemini",
        "=" * 60,
        f"Date: {datetime.now().isoformat()}",
        f"Total time: {total_elapsed:.1f}s",
        f"Image edit model: gemini_flash_image_edit",
        f"VLM judge model:  gemini_flash_lite",
        f"Foot output:  {foot}",
        f"Material: {spec.get('material')}  Angle: {spec.get('camera_angle')}",
        f"Sketch features: {sketch_features or []}",
        f"Tolerance: {tolerance}",
        f"Iterations: {len(loop_result.iterations)}",
        "",
    ]
    for it in loop_result.iterations:
        verdict = "ACCEPT" if it.accepted else "REJECT"
        lines.append(f"  Iteration {it.index}: {verdict} ({it.duration_ms:.1f}ms)")
        meta = it.metadata
        if meta:
            if meta.get("sketch_scores"):
                ss = meta["sketch_scores"]
                lines.append(f"    Sketch: {', '.join(f'{k}={v}' for k,v in ss.items())} "
                              f"(avg={meta.get('sketch_avg')})")
            if meta.get("spec_scores"):
                ss = meta["spec_scores"]
                lines.append(f"    Spec:   {', '.join(f'{k}={v}' for k,v in ss.items())} "
                              f"(avg={meta.get('spec_avg')})")
        lines.append(f"    Feedback: {it.feedback}")
        lines.append("")

    lines.append(f"Final verdict: {results_data.get('final_verdict', 'unknown')}")

    # Token usage summary
    if token_records:
        usage_summary = format_usage_summary(token_records)
        lines.append("")
        lines.append("Token Usage & Pricing")
        lines.append("-" * 40)
        for mid, totals in usage_summary["by_model"].items():
            lines.append(f"  {mid}:")
            lines.append(f"    Calls:      {totals['calls']}")
            lines.append(f"    Input:      {totals['input_tokens']:,} tokens")
            lines.append(f"    Output:     {totals['output_tokens']:,} tokens")
            if totals['thinking_tokens']:
                lines.append(f"    Thinking:   {totals['thinking_tokens']:,} tokens")
            if totals['cached_tokens']:
                lines.append(f"    Cached:     {totals['cached_tokens']:,} tokens")
            lines.append(f"    Total:      {totals['total_tokens']:,} tokens")
            lines.append(f"    Cost:       ${totals['cost_usd']:.6f}")
        gt = usage_summary["grand_total"]
        lines.append(f"  ---")
        lines.append(f"  Grand total:  {gt['total_tokens']:,} tokens  |  "
                      f"${gt['cost_usd']:.6f}  |  {gt['calls']} API calls")

    lines.append(f"Output: {run_dir}")
    summary = "\n".join(lines)
    print(f"\n{summary}")
    (run_dir / "summary.txt").write_text(summary)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sketch-to-shoe agentic loop — Gemini Flash (image edit + VLM judge)"
    )
    parser.add_argument("--sketches", type=str, nargs="+", required=True,
        help="Path(s) to sketch image(s)")
    parser.add_argument("--material", type=str, default="black patent leather",
        help="Shoe material and color (e.g. 'black leather', 'beige suede', 'white canvas')")
    parser.add_argument("--foot", type=str, default="pair",
        choices=["pair", "left", "right"],
        help="Output: 'pair' for both shoes (default), 'left' or 'right' for a single shoe")
    parser.add_argument("--camera-angle", type=str, default=None,
        dest="camera_angle",
        help="Camera angle preset: 3/4, side, front, back, top, hero "
             "(or any custom string). Default: 3/4. "
             "Foot variation (pair/left/right) is selected via --foot.")
    parser.add_argument("--spec", type=str, nargs="*", default=[],
        metavar="KEY=VALUE",
        help="Open-ended extras, e.g. style=elegant note='chunky sole'")
    parser.add_argument("--max-iter", type=int, default=3)
    parser.add_argument("--tolerance", type=str, default="moderate",
        choices=["generous", "moderate", "strict"])
    parser.add_argument("--scale", type=float, default=1.0,
        help="Scale factor for sketch images (default: 1.0)")
    parser.add_argument("--spacing", type=int, default=20,
        help="Pixel spacing between sketches in grid (default: 20)")
    parser.add_argument("--temperature", type=float, default=0.8,
        help="Generation temperature for Gemini image edit (default: 0.8, model default: 1.0)")
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not set. Add it to your .env file.")
        return

    camera_angle = args.camera_angle or _default_camera_angle(args.foot)

    print("=== Sketch-to-Shoe Agentic Loop — Gemini ===")
    print(f"Sketches:      {args.sketches}")
    print(f"Foot output:   {args.foot}")
    print(f"Material:      {args.material}")
    print(f"Camera angle:  {camera_angle}")
    extra_spec = _parse_spec_args(args.spec)
    if extra_spec:
        print(f"Extra spec:    {extra_spec}")
    print(f"Max iterations:{args.max_iter}")
    print(f"Tolerance:     {args.tolerance}")
    print(f"Scale:         {args.scale}x")
    print(f"Image edit:    gemini_flash_image_edit (Nano Banana 2, temperature={args.temperature})")
    print(f"Sketch judge:  gemini_flash (Gemini 3.0 Flash)")
    print(f"Spec judge:    gemini_flash_lite (Gemini 3.1 Flash Lite)")
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
        "camera_angle": camera_angle,
        "extra": extra_spec,
    }

    vlm_session = VLMSession("gemini_flash_lite")
    sketch_vlm_session = VLMSession("gemini_flash_lite")
    sketch_media = ImageMedia(image=sketch_grid)

    pipeline, edit_agent = build_pipeline(
        sketch_media=sketch_media,
        spec=spec,
        max_iterations=args.max_iter,
        vlm_session=vlm_session,
        sketch_vlm_session=sketch_vlm_session,
        sketch_features=None,
        tolerance=args.tolerance,
        foot=args.foot,
        temperature=args.temperature,
    )
    logged = LoggedPipeline(pipeline)

    context = {
        "sketch": sketch_media,
        "image": sketch_media,  # seed for first iteration
    }

    t0 = time.perf_counter()

    try:
        result, exec_log = logged.run(context)
    finally:
        vlm_session.unload()

    total_elapsed = time.perf_counter() - t0

    print(exec_log.summary())

    loop_result = result.get("sketch_to_shoe_loop_history")
    final_img = result.get("image")

    # Collect all token usage records
    token_records = (
        sketch_vlm_session.token_usage_log
        + vlm_session.token_usage_log
        + edit_agent.token_usage_log
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"{args.max_iter}iter_{ts}"

    save_results(
        run_dir=run_dir,
        loop_result=loop_result if isinstance(loop_result, LoopResult) else LoopResult(),
        result_context=result,
        sketch_grid=sketch_grid,
        final_img=final_img,
        spec=spec,
        total_elapsed=total_elapsed,
        sketch_features=None,
        tolerance=args.tolerance,
        foot=args.foot,
        token_records=token_records,
    )

    print(f"\nDone. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
