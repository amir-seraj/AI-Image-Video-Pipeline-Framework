"""Shoe replacement pipeline — Gemini version.

Provides build_pipeline and save_results extracted from
tests/run_shoe_tryon_gemini.py, loading prompt configuration from
prompt.yaml in this directory.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import yaml
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Explicit file-based import of judge module to avoid collision with
# workflows/sketch_to_shoe/scripts/judge.py when both are loaded in the
# same process (e.g. app.py).
# ---------------------------------------------------------------------------

_WORKFLOW_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _WORKFLOW_DIR.parent.parent
_JUDGE_PATH = _PROJECT_ROOT / "workflows" / "shoe_tryon_loop" / "scripts" / "judge.py"

_spec = importlib.util.spec_from_file_location("tryon_judge", _JUDGE_PATH)
_judge_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_judge_mod)

from casadei import Agent, AgentConfig, AgentStep, ImageMedia, TextMedia, Pipeline
from casadei.loop import LoopStep, LoopResult
from casadei.providers.gemini_pricing import format_usage_summary

VLMSession = _judge_mod.VLMSession
make_judge = _judge_mod.make_judge
make_best_fn = _judge_mod.make_best_fn
extract_features = _judge_mod.extract_features


@lru_cache(maxsize=1)
def load_prompt_config() -> dict:
    """Load and cache prompt configuration from prompt.yaml."""
    config_path = _WORKFLOW_DIR / "prompt.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_pipeline(
    max_iterations: int = 3,
    vlm_session: VLMSession | None = None,
    features: list[str] | None = None,
    tolerance: str = "moderate",
) -> tuple[Pipeline, Agent]:
    """Build the iterative shoe replacement pipeline using Gemini models.

    Returns (pipeline, image_edit_agent) so callers can read the agent's
    token_usage_log after the run.
    """
    config = load_prompt_config()
    prompt_template = config["prompt_template"].strip()

    if vlm_session is None:
        vlm_session = VLMSession("gemini_flash_lite")

    gemini_edit_agent = Agent(AgentConfig(
        name="gemini_shoe_replace",
        model="gemini_flash_image_edit",
        description="Gemini Flash image edit for shoe replacement",
        prompt_template=prompt_template,
        negative_prompt="",
        params={},
    ))

    edit_step = AgentStep(
        name="gemini_edit",
        agent=gemini_edit_agent,
        input_map={
            "image": "shoe",    # first image = reference shoe
            "image_2": "image", # second image = person / last generation
        },
        output_map={"image": "image"},
        template_kwargs={"feedback": ""},
    )

    judge_fn = make_judge(
        session=vlm_session,
        shoe_key="shoe",
        candidate_key="image",
        features=features,
        tolerance=tolerance,
    )

    loop = LoopStep(
        name="tryon_loop",
        body=[edit_step],
        judge=judge_fn,
        max_iterations=max_iterations,
        best_fn=make_best_fn(
            session=vlm_session,
            shoe_key="shoe",
            output_key="image",
        ),
        swap_models=True,
        output_key="image",
        feedback_template_var="feedback",
    )

    return Pipeline(name="shoe_tryon_gemini", steps=[loop]), gemini_edit_agent


def save_results(
    run_dir: Path,
    loop_result: LoopResult,
    result_context: dict,
    source_img: PILImage.Image,
    shoes_img: PILImage.Image,
    final_img: ImageMedia | None,
    total_elapsed: float,
    features: list[str] | None = None,
    tolerance: str = "strict",
    token_records: list[dict] | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    source_img.save(run_dir / "input_person.png")
    shoes_img.save(run_dir / "input_shoes.png")

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "total_elapsed_s": total_elapsed,
        "models": {
            "image_edit": "gemini_flash_image_edit",
            "vlm_judge": "gemini_flash_lite",
        },
        "features": features or [],
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
            iter_data["scores"] = it.metadata.get("scores", {})
            iter_data["avg_score"] = it.metadata.get("avg_score")
            iter_data["lowest_attr"] = it.metadata.get("lowest_attr")

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
        "Shoe Replacement Loop Results — Gemini",
        "=" * 60,
        f"Date: {datetime.now().isoformat()}",
        f"Total time: {total_elapsed:.1f}s",
        f"Image edit model: gemini_flash_image_edit",
        f"VLM judge model:  gemini_flash_lite",
        f"Features: {features or []}",
        f"Tolerance: {tolerance}",
        f"Iterations: {len(loop_result.iterations)}",
        "",
    ]
    for it in loop_result.iterations:
        verdict = "ACCEPT" if it.accepted else "REJECT"
        lines.append(f"  Iteration {it.index}: {verdict} ({it.duration_ms:.1f}ms)")
        if it.metadata and it.metadata.get("scores"):
            scores = it.metadata["scores"]
            scores_str = ", ".join(f"{k}={v}" for k, v in scores.items())
            avg = it.metadata.get("avg_score", 0)
            lines.append(f"    Scores: {scores_str} (avg={avg})")
        lines.append(f"    Feedback: {it.feedback[:200]}")
        lines.append("")

    if results_data.get("best_selected_candidate"):
        lines.append(f"Best selected: candidate {results_data['best_selected_candidate']}")
        if results_data.get("best_selection_vlm_response"):
            lines.append(f"  VLM reason: {results_data['best_selection_vlm_response'][:200]}")
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
