"""Pipeline module for sketch_to_shoe_gemini workflow.

Provides build_pipeline, save_results, build_extra_specs_text, and load_prompt_config.
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import yaml
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# sys.path setup — allow imports from shared utilities and judge scripts
# ---------------------------------------------------------------------------

_WORKFLOW_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _WORKFLOW_DIR.parent.parent
_SHARED_DIR = _PROJECT_ROOT / "workflows" / "shared"
_JUDGE_DIR = _PROJECT_ROOT / "workflows" / "sketch_to_shoe" / "scripts"

for _p in [str(_SHARED_DIR), str(_JUDGE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from image_utils import get_camera_preset, get_judge_notes, foot_framing, build_sketch_grid, build_material_grid  # noqa: E402
from judge import VLMSession, make_spec_judge, make_shoe_count_judge, make_best_fn  # noqa: E402

from casadei import Agent, AgentConfig, AgentStep, ImageMedia, TextMedia, Pipeline  # noqa: E402
from casadei.loop import LoopStep, LoopResult  # noqa: E402
from casadei.providers.gemini_pricing import format_usage_summary  # noqa: E402

# ---------------------------------------------------------------------------
# Prompt config loading
# ---------------------------------------------------------------------------

_PROMPT_CONFIG_PATH = _WORKFLOW_DIR / "prompt.yaml"
_cached_prompt_config: dict | None = None


def load_prompt_config() -> dict:
    """Load and cache the prompt.yaml config for this workflow."""
    global _cached_prompt_config
    if _cached_prompt_config is None:
        with open(_PROMPT_CONFIG_PATH) as f:
            _cached_prompt_config = yaml.safe_load(f)
    return _cached_prompt_config


_MATERIALS_PROMPT_PATH = _WORKFLOW_DIR / "prompt_materials.yaml"
_cached_materials_prompt: dict | None = None


def load_materials_prompt_config() -> dict:
    """Load and cache the prompt_materials.yaml config."""
    global _cached_materials_prompt
    if _cached_materials_prompt is None:
        with open(_MATERIALS_PROMPT_PATH) as f:
            _cached_materials_prompt = yaml.safe_load(f)
    return _cached_materials_prompt


# ---------------------------------------------------------------------------
# Spec utilities
# ---------------------------------------------------------------------------

def build_extra_specs_text(extra: dict) -> str:
    """Build a formatted string of extra spec key-value pairs."""
    if not extra:
        return ""
    return "\n".join(f"- {k.capitalize()}: {v}" for k, v in extra.items())


def build_materials_prompt(
    materials: list[dict],
    resolved_names: list[str],
) -> str:
    """Build the $materials_instructions text for the prompt template.

    Single material (no placement): "Apply the material/color shown in the reference image to the shoe."
    Single material (with placement): "Apply the material/color shown in the reference image to the {placement} of the shoe."
    Multiple materials: bullet list mapping each named material to its placement.
    """
    n = len(materials)

    if n == 1:
        entry = materials[0]
        kind = "color" if entry.get("is_color") else "material"
        placement = entry.get("placement")
        note = entry.get("note")

        if placement:
            line = f"Apply the {kind} shown in the reference image to the {placement} of the shoe."
        else:
            line = f"Apply the {kind} shown in the reference image to the shoe."

        parts = [line]
        if note:
            parts.append(note)
        return "\n".join(parts)

    # Multiple materials
    lines = ["The reference image contains labeled material/color swatches. Apply each as follows:"]
    for entry, name in zip(materials, resolved_names):
        kind = "color" if entry.get("is_color") else "material"
        placement = entry.get("placement", "")
        note = entry.get("note")
        bullet = f'- Apply the {kind} shown in "{name}" to the {placement}.'
        if note:
            bullet += f" {note}"
        lines.append(bullet)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

MAX_ITERATIONS = 3


def build_pipeline(
    spec: dict,
    vlm_session: VLMSession,
    foot: str = "pair",
    temperature: float = 1.0,
) -> tuple[Pipeline, Agent, list[VLMSession], PILImage.Image | None]:
    """Build the sketch-to-shoe generation pipeline with angle-correction loop.

    Args:
        spec: Dict containing 'material', 'camera_angle', 'extra' (dict of extras),
              and optionally 'ref_images' (dict with 'material_ref' and/or 'color_ref'
              keys mapping to ImageMedia objects).
        vlm_session: VLMSession used for best_fn ranking.
        foot: One of 'pair', 'left', 'right'.
        temperature: Generation temperature for Gemini image edit.

    Returns:
        Tuple of (Pipeline, Agent, list[VLMSession], grid_image_or_None).
    """
    prompt_config = load_prompt_config()

    # --- Detect materials mode ---
    materials_list = spec.get("materials")
    use_materials_mode = bool(materials_list)

    if use_materials_mode:
        prompt_config = load_materials_prompt_config()
        grid_image, resolved_names = build_material_grid(materials_list)
        materials_instructions = build_materials_prompt(materials_list, resolved_names)
    else:
        materials_instructions = None
        grid_image = None

    prompt_template = prompt_config["prompt_template"]

    extra_specs_text = build_extra_specs_text(spec.get("extra", {}))

    # Handle optional reference images in spec
    ref_images: dict = {} if use_materials_mode else spec.get("ref_images", {})
    material_ref: ImageMedia | None = ref_images.get("material_ref")
    color_ref: ImageMedia | None = ref_images.get("color_ref")

    # Build extra_specs_text additions for reference images
    if material_ref is not None or color_ref is not None:
        ref_lines = []
        if material_ref is not None:
            ref_lines.append(
                "- Material reference: use the attached material_ref image to match texture and finish."
            )
        if color_ref is not None:
            ref_lines.append(
                "- Color reference: use the attached color_ref image to match the exact color palette."
            )
        if extra_specs_text:
            extra_specs_text = extra_specs_text + "\n" + "\n".join(ref_lines)
        else:
            extra_specs_text = "\n".join(ref_lines)

    gemini_agent = Agent(AgentConfig(
        name="gemini_sketch_to_shoe",
        model="gemini_flash_image_edit",
        description="Gemini Flash image edit for sketch-to-shoe generation",
        prompt_template=prompt_template,
        negative_prompt="",
        params={"temperature": temperature},
    ))

    camera_preset = get_camera_preset(spec.get("camera_angle", "3/4"), foot)

    # Build input_map — sketch first (determines aspect ratio), then materials grid or ref images
    input_map: dict[str, str] = {"image": "sketch"}
    if use_materials_mode:
        input_map["materials_grid"] = "materials_grid"
    else:
        if material_ref is not None:
            input_map["material_ref"] = "material_ref"
        if color_ref is not None:
            input_map["color_ref"] = "color_ref"

    edit_step = AgentStep(
        name="gemini_generate",
        agent=gemini_agent,
        input_map=input_map,
        output_map={"image": "image"},
        template_kwargs={
            **camera_preset,
            "extra_specs": extra_specs_text,
            "foot_framing": foot_framing(foot, emphatic=False),
            "feedback": "",
            **({"materials_instructions": materials_instructions} if use_materials_mode else {"material": spec.get("material", "black patent leather")}),
        },
    )

    session_camera = VLMSession("gemini_flash")
    session_count = VLMSession("gemini_flash_lite")

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

    def _combined_judge(context):
        image = context.get("image")
        # Force PIL lazy-load before threads start — prevents concurrent load() race
        if isinstance(image, ImageMedia):
            image.image.load()
        ctx_cam: dict = {"image": image}
        ctx_count: dict = {"image": image}

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_cam = pool.submit(camera_judge, ctx_cam)
            fut_count = pool.submit(count_judge, ctx_count)
            cam_accepted, cam_fb = fut_cam.result()
            count_accepted, count_fb = fut_count.result()

        # Propagate metadata written by judges back to main context
        context.update({k: v for k, v in ctx_cam.items() if k.startswith("_")})
        context.update({k: v for k, v in ctx_count.items() if k.startswith("_")})

        # Promote camera metadata for best_fn
        meta = context.pop("_judge_metadata_spec", {})
        context["_judge_metadata"] = {
            "sketch_avg": None,
            "spec_scores": meta.get("scores", {}),
            "spec_avg": meta.get("avg_score"),
        }
        accepted = cam_accepted and count_accepted
        parts = []
        if not count_accepted and count_fb and count_fb != "none":
            parts.append(f"Shoe count issue: {count_fb}")
        if not cam_accepted and cam_fb and cam_fb != "none":
            if count_accepted:
                # Count is fine — protect materials while fixing angle
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

    return Pipeline(name="sketch_to_shoe_gemini", steps=[loop]), gemini_agent, [session_camera, session_count], grid_image


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def save_results(
    run_dir: Path,
    loop_result: LoopResult,
    result_context: dict,
    result_image: ImageMedia | None,
    sketch_grid: PILImage.Image,
    spec: dict,
    total_elapsed: float,
    foot: str = "pair",
    token_records: list[dict] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    sketch_grid.save(run_dir / "input_sketch_grid.png")

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "total_elapsed_s": total_elapsed,
        "model": "gemini_flash_image_edit",
        "angle_judge": "gemini_flash",
        "foot": foot,
        "spec": spec,
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
        if meta and meta.get("spec_scores"):
            iter_data["camera_score"] = meta.get("spec_scores", {}).get("camera_angle")
            iter_data["spec_avg"] = meta.get("spec_avg")
        candidate_img = it.outputs.get("image")
        if isinstance(candidate_img, ImageMedia):
            img_path = run_dir / f"iter_{it.index:02d}_candidate.png"
            candidate_img.image.save(img_path)
            iter_data["image_path"] = str(img_path.name)
        results_data["iterations"].append(iter_data)

    if result_image is not None and isinstance(result_image, ImageMedia):
        result_image.image.save(run_dir / "result.png")
        results_data["result"] = "result.png"

    if loop_result.iterations:
        last = loop_result.iterations[-1]
        if last.accepted:
            results_data["final_verdict"] = f"accepted_at_iteration_{last.index}"
        else:
            results_data["final_verdict"] = "max_reached_best_selected"

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
        "Sketch-to-Shoe Direct + Angle Judge — Gemini",
        "=" * 60,
        f"Date: {datetime.now().isoformat()}",
        f"Total time: {total_elapsed:.1f}s",
        f"Model: gemini_flash_image_edit",
        f"Angle judge: gemini_flash  (camera angle only, max {MAX_ITERATIONS} iter)",
        f"Foot output:  {foot}",
        f"Material: {spec.get('material')}  Angle: {spec.get('camera_angle')}",
        f"Iterations: {len(loop_result.iterations)}",
        "",
    ]
    for it in loop_result.iterations:
        verdict = "ACCEPT" if it.accepted else "REJECT"
        lines.append(f"  Iteration {it.index}: {verdict} ({it.duration_ms:.1f}ms)")
        meta = it.metadata
        if meta and meta.get("spec_scores"):
            score = meta["spec_scores"].get("camera_angle", "?")
            lines.append(f"    Camera angle score: {score}")
        lines.append(f"    Feedback: {it.feedback}")
        lines.append("")
    lines.append(f"Final verdict: {results_data.get('final_verdict', 'unknown')}")
    lines.append("")

    if token_records:
        usage_summary = format_usage_summary(token_records)
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
