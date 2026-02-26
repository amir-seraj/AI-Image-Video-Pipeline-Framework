"""Shoe try-on agentic loop — integration test with full logging.

Runs the iterative shoe try-on pipeline using FireRed for generation
and Qwen3-VL for judgment. Saves all intermediate results, VLM
responses, timing data, and the final output.

Usage:
    python tests/run_shoe_tryon_loop.py
    python tests/run_shoe_tryon_loop.py --source tests/Image/model001.jpeg --shoes tests/Image/shoes001.jpeg
    python tests/run_shoe_tryon_loop.py --max-iter 3 --steps 20
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image as PILImage

# Add project root to path so we can import workflow scripts
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "workflows" / "shoe_tryon_loop" / "scripts"))

from casadei import (
    Agent, AgentConfig, AgentStep, ImageMedia, TextMedia,
    LoggedPipeline, Pipeline,
)
from casadei.loop import LoopStep, LoopResult
from judge import VLMSession, make_judge, make_best_fn

IMAGE_DIR = Path(__file__).parent / "Image"
OUTPUT_DIR = Path(__file__).parent / "output" / "shoe_tryon_loop"

PROMPT_TEMPLATE = (
    "The first image is the reference shoe — the target design. "
    "In the second image there is a person wearing shoes. "
    "Replace the shoes on the person's feet with the shoes shown "
    "in the first image. Keep the person's pose, legs, clothing, "
    "and the background exactly the same. $feedback"
)


def _pad_to_square(img: PILImage.Image) -> PILImage.Image:
    w, h = img.size
    if w == h:
        return img
    size = max(w, h)
    sq = PILImage.new("RGB", (size, size), (255, 255, 255))
    sq.paste(img, ((size - w) // 2, (size - h) // 2))
    return sq


def build_pipeline(
    max_iterations: int = 5,
    num_inference_steps: int = 30,
    swap_models: bool = True,
    vlm_session: VLMSession | None = None,
) -> Pipeline:
    """Build the iterative shoe try-on pipeline."""
    if vlm_session is None:
        vlm_session = VLMSession("qwen3_vl_8b")

    firered_agent = Agent(AgentConfig(
        name="firered_tryon",
        model="firered_image_edit",
        description="FireRed shoe try-on with feedback repair",
        prompt_template=PROMPT_TEMPLATE,
        negative_prompt="blurry, distorted, low quality, bad anatomy, missing shoes, extra limbs",
        params={"num_inference_steps": num_inference_steps},
    ))

    firered_step = AgentStep(
        name="firered_edit",
        agent=firered_agent,
        input_map={
            "image": "shoe",        # always original shoes (IMAGE 1 — same as VLM)
            "image_2": "image",     # first iter: original person (seeded); subsequent: last generation
        },
        output_map={"image": "image"},
        template_kwargs={"feedback": ""},
    )

    loop = LoopStep(
        name="tryon_loop",
        body=[firered_step],
        judge=make_judge(
            session=vlm_session,
            shoe_key="shoe",
            candidate_key="image",
        ),
        max_iterations=max_iterations,
        best_fn=make_best_fn(
            session=vlm_session,
            shoe_key="shoe",
            output_key="image",
        ),
        swap_models=swap_models,
        output_key="image",
        feedback_template_var="feedback",
    )

    return Pipeline(name="shoe_tryon_loop", steps=[loop])


def save_results(
    run_dir: Path,
    loop_result: LoopResult,
    result_context: dict,
    source_img: PILImage.Image,
    shoes_img: PILImage.Image,
    final_img: ImageMedia | None,
    total_elapsed: float,
    peak_gb: float,
) -> None:
    """Save all results, intermediates, and metrics to run_dir."""
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save inputs
    source_img.save(run_dir / "input_person.png")
    shoes_img.save(run_dir / "input_shoes.png")

    # Save per-iteration results
    results_data = {
        "timestamp": datetime.now().isoformat(),
        "total_elapsed_s": total_elapsed,
        "peak_vram_gb": peak_gb,
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

        # Save candidate image for this iteration
        candidate_img = it.outputs.get("image")
        if isinstance(candidate_img, ImageMedia):
            img_path = run_dir / f"iter_{it.index:02d}_candidate.png"
            candidate_img.image.save(img_path)
            iter_data["image_path"] = str(img_path.name)

        results_data["iterations"].append(iter_data)

    # Save final result
    if final_img is not None and isinstance(final_img, ImageMedia):
        final_img.image.save(run_dir / "final_result.png")
        results_data["final_result"] = "final_result.png"

    # Determine final verdict and best-selection info
    if loop_result.iterations:
        last = loop_result.iterations[-1]
        if last.accepted:
            results_data["final_verdict"] = f"accepted_at_iteration_{last.index}"
        else:
            results_data["final_verdict"] = "max_reached_best_selected"
            # Capture best-selection metadata from best_fn
            best_idx = result_context.get("best_selection_index")
            best_reason = result_context.get("best_selection_reason")
            if best_idx is not None:
                results_data["best_selected_candidate"] = best_idx
            if isinstance(best_reason, TextMedia):
                results_data["best_selection_vlm_response"] = best_reason.text

    # Write JSON report
    (run_dir / "results.json").write_text(
        json.dumps(results_data, indent=2, default=str)
    )

    # Write human-readable summary
    lines = [
        f"Shoe Try-On Loop Results",
        f"{'=' * 60}",
        f"Date: {datetime.now().isoformat()}",
        f"Total time: {total_elapsed:.1f}s",
        f"Peak VRAM: {peak_gb:.2f} GB",
        f"Iterations: {len(loop_result.iterations)}",
        f"",
    ]
    for it in loop_result.iterations:
        verdict = "ACCEPT" if it.accepted else "REJECT"
        lines.append(f"  Iteration {it.index}: {verdict} ({it.duration_ms:.1f}ms)")
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
    parser = argparse.ArgumentParser(description="Shoe try-on agentic loop test")
    parser.add_argument(
        "--source", type=str,
        default=str(IMAGE_DIR / "legs001.jpeg"),
        help="Path to person/model image",
    )
    parser.add_argument(
        "--shoes", type=str,
        default=str(IMAGE_DIR / "shoes001.jpeg"),
        help="Path to shoes image",
    )
    parser.add_argument(
        "--max-iter", type=int, default=5,
        help="Maximum loop iterations (default: 5)",
    )
    parser.add_argument(
        "--steps", type=int, default=30,
        help="Inference steps per generation (default: 30)",
    )
    parser.add_argument(
        "--keep-both", action="store_true",
        help="Keep both models loaded (needs ~74GB VRAM)",
    )
    parser.add_argument(
        "--no-pad", action="store_true",
        help="Skip square padding — keep original aspect ratio",
    )
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="Scale factor for input images (e.g. 0.5 for half size, default: 1.0)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        return

    print(f"=== Shoe Try-On Agentic Loop ===")
    print(f"Source: {args.source}")
    print(f"Shoes: {args.shoes}")
    print(f"Max iterations: {args.max_iter}")
    print(f"Inference steps: {args.steps}")
    print(f"Memory mode: {'keep-both' if args.keep_both else 'swap'}")
    print(f"Padding: {'off (original aspect ratio)' if args.no_pad else 'square'}")
    print(f"Scale: {args.scale}x")
    print()

    # Load images
    source = PILImage.open(args.source).convert("RGB")
    if not args.no_pad:
        source = _pad_to_square(source)
    shoes = PILImage.open(args.shoes).convert("RGB")

    # Resize if scale != 1.0
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

    # Create shared VLM session — pre-load in keep-both mode
    vlm_session = VLMSession("qwen3_vl_8b")

    # Build pipeline
    pipeline = build_pipeline(
        max_iterations=args.max_iter,
        num_inference_steps=args.steps,
        swap_models=not args.keep_both,
        vlm_session=vlm_session,
    )
    logged = LoggedPipeline(pipeline)

    # Prepare context — seed "image" with the person image so the first
    # iteration reads it as the base; subsequent iterations read the last generation.
    person_media = ImageMedia(image=source)
    context = {
        "person": person_media,
        "shoe": ImageMedia(image=shoes),
        "image": person_media,
    }

    # Pre-load all models in keep-both mode
    if args.keep_both:
        print("Loading all models (keep-both mode)...")
        vlm_session.load()
        pipeline.load()  # loads body agents (FireRed) via LoopStep.load()

    # Run
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    try:
        result, exec_log = logged.run(context)
    finally:
        vlm_session.unload()

    torch.cuda.synchronize()
    total_elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)

    print(exec_log.summary())

    # Extract results
    loop_result = result.get("tryon_loop_history")
    final_img = result.get("image")

    # Save everything
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"{args.max_iter}iter_{args.steps}steps_{ts}"

    save_results(
        run_dir=run_dir,
        loop_result=loop_result if isinstance(loop_result, LoopResult) else LoopResult(),
        result_context=result,
        source_img=source,
        shoes_img=shoes,
        final_img=final_img,
        total_elapsed=total_elapsed,
        peak_gb=peak_gb,
    )

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\nDone. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
