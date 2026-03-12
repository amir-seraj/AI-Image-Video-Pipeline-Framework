"""Shoe angles pipeline — generate photorealistic shoe views at multiple camera angles.

Takes a design sketch and a photorealistic reference image and produces new views
at specified camera angles using Gemini Flash Image Edit, with three parallel
judges (camera angle + reference fidelity + shoe count) for iterative refinement.
"""

from __future__ import annotations

import io
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

logger = logging.getLogger("shoe_angles")

# ---------------------------------------------------------------------------
# sys.path setup — shared utilities and judge module
# ---------------------------------------------------------------------------

_WORKFLOW_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_WORKFLOW_ROOT / "shared"))
sys.path.insert(0, str(_WORKFLOW_ROOT / "sketch_to_shoe" / "scripts"))

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import yaml
from PIL import Image as PILImage

from google import genai
from google.genai import types as genai_types

from image_utils import (
    get_camera_preset,
    get_judge_notes,
    get_canonical_angles,
    get_pair_angles,
    foot_for_angle,
    foot_framing,
    find_ratio,
    pad_to_ratio,
)
from judge import VLMSession, make_spec_judge, make_reference_fidelity_judge, make_shoe_count_judge

from casadei.media import ImageMedia
from casadei.providers.gemini_pricing import extract_token_usage, calculate_cost

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_GENERATION_MODEL_ID = "gemini-3.1-flash-image-preview"
MAX_JUDGE_ITERATIONS = 3

# ---------------------------------------------------------------------------
# Prompt config loading
# ---------------------------------------------------------------------------

_PROMPT_CONFIG_PATH = Path(__file__).resolve().parent / "prompt.yaml"
_cached_prompt_config: dict | None = None


def load_prompt_config() -> dict:
    """Load and cache prompt.yaml from this workflow directory."""
    global _cached_prompt_config
    if _cached_prompt_config is None:
        with open(_PROMPT_CONFIG_PATH) as f:
            _cached_prompt_config = yaml.safe_load(f)
    return _cached_prompt_config


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
    """Generate a single angle. Returns (angle_name, result_image, usage_dict)."""
    effective_foot = foot_for_angle(angle, foot, single=single)
    preset = get_camera_preset(angle, effective_foot)

    prompt_template = load_prompt_config()["prompt_template"]
    prompt = prompt_template.format(
        camera_desc=preset["camera_desc"],
        staging_desc=preset["staging_desc"],
        foot_framing=foot_framing(effective_foot),
        feedback=f"\n\n{feedback}" if feedback else "",
    )

    logger.info("=== generate_angle: %s (foot=%s) ===", angle, effective_foot)
    logger.info("PROMPT:\n%s", prompt)

    # Pad both images to the same ratio
    ratio = find_ratio(*reference.size)
    aspect_ratio_str = f"{ratio[0]}:{ratio[1]}"
    padded_sketch = pad_to_ratio(sketch, ratio)
    padded_ref = pad_to_ratio(reference, ratio)
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


# ---------------------------------------------------------------------------
# Debug annotation overlay
# ---------------------------------------------------------------------------

def annotate_image(
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


# ---------------------------------------------------------------------------
# Iterative angle generation with three parallel judges
# ---------------------------------------------------------------------------

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
    effective_foot = foot_for_angle(angle, foot, single=single)
    preset = get_camera_preset(angle, effective_foot)

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
        judge_notes=get_judge_notes(),
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
                    annotated = annotate_image(
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


# Re-export for convenience
__all__ = [
    "load_prompt_config",
    "generate_angle",
    "annotate_image",
    "generate_angle_with_judge",
    "get_canonical_angles",
    "MAX_JUDGE_ITERATIONS",
    "_GENERATION_MODEL_ID",
]
