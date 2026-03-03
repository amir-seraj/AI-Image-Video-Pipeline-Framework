"""Shoe replacement agentic loop — Gemini version.

Runs the iterative shoe replacement pipeline using Gemini Flash Image Edit
for generation and Gemini Flash VLM for judgment. Both models are API-based
(no GPU required). Reads GEMINI_API_KEY from the .env file.

Usage:
    python tests/run_shoe_tryon_gemini.py
    python tests/run_shoe_tryon_gemini.py --source tests/Image/model001.jpeg --shoes tests/Image/shoes001.jpeg
    python tests/run_shoe_tryon_gemini.py --max-iter 3 --tolerance moderate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image as PILImage

# Load .env so GEMINI_API_KEY is available before any genai import
load_dotenv()

# Add workflow scripts directory to path for judge helpers
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "workflows" / "shoe_tryon_loop" / "scripts"))

from casadei import (
    Agent, AgentConfig, AgentStep, ImageMedia, TextMedia,
    LoggedPipeline, Pipeline,
)
from casadei.loop import LoopStep, LoopResult
from judge import VLMSession, make_judge, make_best_fn, extract_features

IMAGE_DIR = Path(__file__).parent / "Image"
OUTPUT_DIR = Path(__file__).parent / "output" / "shoe_tryon_gemini"

PROMPT_TEMPLATE = (
    "Replace the shoes on the person's feet with the exact shoe shown in the "
    "first image. Match every visible detail of the reference shoe: color, "
    "material, heel shape, toe shape, straps, and hardware. "
    "Replace BOTH shoes — left foot AND right foot. "
    "Preserve the person's pose, legs, clothing, and background exactly. "
    "$feedback"
)


def build_pipeline(
    max_iterations: int = 5,
    vlm_session: VLMSession | None = None,
    features: list[str] | None = None,
    tolerance: str = "strict",
) -> Pipeline:
    """Build the iterative shoe replacement pipeline using Gemini models."""
    if vlm_session is None:
        vlm_session = VLMSession("gemini_flash")

    gemini_edit_agent = Agent(AgentConfig(
        name="gemini_shoe_replace",
        model="gemini_flash_image_edit",
        description="Gemini Flash image edit for shoe replacement",
        prompt_template=PROMPT_TEMPLATE,
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

    return Pipeline(name="shoe_tryon_gemini", steps=[loop])


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
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    source_img.save(run_dir / "input_person.png")
    shoes_img.save(run_dir / "input_shoes.png")

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "total_elapsed_s": total_elapsed,
        "models": {
            "image_edit": "gemini_flash_image_edit",
            "vlm_judge": "gemini_flash",
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

    (run_dir / "results.json").write_text(
        json.dumps(results_data, indent=2, default=str)
    )

    lines = [
        "Shoe Replacement Loop Results — Gemini",
        "=" * 60,
        f"Date: {datetime.now().isoformat()}",
        f"Total time: {total_elapsed:.1f}s",
        f"Image edit model: gemini_flash_image_edit",
        f"VLM judge model:  gemini_flash",
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
    lines.append(f"Output: {run_dir}")

    summary = "\n".join(lines)
    print(f"\n{summary}")
    (run_dir / "summary.txt").write_text(summary)


def main():
    parser = argparse.ArgumentParser(
        description="Shoe replacement agentic loop — Gemini Flash (image edit + VLM judge)"
    )
    parser.add_argument(
        "--source", type=str,
        default=str(IMAGE_DIR / "model001.jpeg"),
        help="Path to person/model image",
    )
    parser.add_argument(
        "--shoes", type=str,
        default=str(IMAGE_DIR / "shoes001.jpeg"),
        help="Path to reference shoes image",
    )
    parser.add_argument(
        "--max-iter", type=int, default=5,
        help="Maximum loop iterations (default: 5)",
    )
    parser.add_argument(
        "--tolerance", type=str, default="moderate",
        choices=["generous", "moderate", "strict"],
        help="Judge tolerance level (default: moderate)",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="Scale factor for input images (default: 1.0)",
    )
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY not set. Add it to your .env file.")
        return

    print("=== Shoe Replacement Agentic Loop — Gemini ===")
    print(f"Source:        {args.source}")
    print(f"Shoes:         {args.shoes}")
    print(f"Max iterations:{args.max_iter}")
    print(f"Tolerance:     {args.tolerance}")
    print(f"Scale:         {args.scale}x")
    print(f"Image edit:    gemini_flash_image_edit (Nano Banana 2)")
    print(f"VLM judge:     gemini_flash (Gemini 3 Flash)")
    print()

    source = PILImage.open(args.source).convert("RGB")
    shoes = PILImage.open(args.shoes).convert("RGB")

    if args.scale != 1.0:
        source = source.resize(
            (int(source.width * args.scale), int(source.height * args.scale)),
            PILImage.LANCZOS,
        )
        shoes = shoes.resize(
            (int(shoes.width * args.scale), int(shoes.height * args.scale)),
            PILImage.LANCZOS,
        )
        print(f"Resized — source: {source.size}, shoes: {shoes.size}")

    vlm_session = VLMSession("gemini_flash")

    shoe_media = ImageMedia(image=shoes)
    print("Extracting shoe features...")
    features = extract_features(vlm_session, shoe_media)
    print(f"Features: {features}")
    print()

    pipeline = build_pipeline(
        max_iterations=args.max_iter,
        vlm_session=vlm_session,
        features=features,
        tolerance=args.tolerance,
    )
    logged = LoggedPipeline(pipeline)

    person_media = ImageMedia(image=source)
    context = {
        "person": person_media,
        "shoe": ImageMedia(image=shoes),
        "image": person_media,
    }

    t0 = time.perf_counter()

    try:
        result, exec_log = logged.run(context)
    finally:
        vlm_session.unload()

    total_elapsed = time.perf_counter() - t0

    print(exec_log.summary())

    loop_result = result.get("tryon_loop_history")
    final_img = result.get("image")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"{args.max_iter}iter_{ts}"

    save_results(
        run_dir=run_dir,
        loop_result=loop_result if isinstance(loop_result, LoopResult) else LoopResult(),
        result_context=result,
        source_img=source,
        shoes_img=shoes,
        final_img=final_img,
        total_elapsed=total_elapsed,
        features=features,
        tolerance=args.tolerance,
    )

    print(f"\nDone. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
