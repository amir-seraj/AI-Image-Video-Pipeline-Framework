"""Sketch-to-shoe agentic loop — integration runner.

Converts design sketches + spec (material, color, camera angle, extras) into
a photorealistic studio product photograph using FireRed + dual VLM judges.

Usage:
    python tests/run_sketch_to_shoe_loop.py --sketches tests/Image/sketch001.png
    python tests/run_sketch_to_shoe_loop.py \\
        --sketches s1.png s2.png \\
        --material suede --color beige --camera-angle "3/4 view" \\
        --spec style=elegant note="chunky platform"
    python tests/run_sketch_to_shoe_loop.py --max-iter 3 --steps 20
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image as PILImage

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "workflows" / "sketch_to_shoe" / "scripts"))

from casadei import (
    Agent, AgentConfig, AgentStep, ImageMedia, TextMedia,
    LoggedPipeline, Pipeline,
)
from casadei.loop import LoopStep, LoopResult
from judge import (
    VLMSession,
    extract_sketch_features,
    make_sketch_judge,
    make_spec_judge,
    make_dual_judge,
    make_best_fn,
)

IMAGE_DIR = Path(__file__).parent / "Image"
OUTPUT_DIR = Path(__file__).parent / "output" / "sketch_to_shoe_loop"

PROMPT_TEMPLATE = (
    "The image shows a shoe design sketch. Convert it into a professional, "
    "photorealistic product photograph of the final shoe.\n\n"
    "Design specifications:\n"
    "- Material: $material\n"
    "- Color: $color\n"
    "- Camera angle: $camera_angle\n"
    "$extra_specs\n\n"
    "The result must be a studio-quality photograph: clean white background, "
    "professional product lighting, sharp focus, no shadows on background, "
    "shoe centered and fully visible. $feedback"
)


# ---------------------------------------------------------------------------
# Sketch grid assembly
# ---------------------------------------------------------------------------

def _build_sketch_grid(
    images: list[PILImage.Image],
    spacing: int = 20,
) -> PILImage.Image:
    """Arrange sketch images in a rectangular grid, then pad to square.

    1. Compute grid dimensions: cols = ceil(sqrt(n)), rows = ceil(n / cols).
    2. Common cell size = max width x max height across all inputs.
    3. Paste tiles with `spacing` px white gaps (also used as outer border).
    4. If resulting grid is not square, center on a white square canvas.
    """
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
    """Parse KEY=VALUE strings from --spec CLI args. Ignores entries without '='."""
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
    num_inference_steps: int = 30,
    swap_models: bool = True,
    vlm_session: VLMSession | None = None,
    sketch_features: list[str] | None = None,
    tolerance: str = "strict",
) -> Pipeline:
    """Build the iterative sketch-to-shoe pipeline."""
    if vlm_session is None:
        vlm_session = VLMSession("qwen3_vl_8b")

    extra_specs_text = _build_extra_specs_text(spec.get("extra", {}))

    firered_agent = Agent(AgentConfig(
        name="firered_sketch_to_shoe",
        model="firered_image_edit",
        description="FireRed sketch-to-shoe with dual-judge feedback repair",
        prompt_template=PROMPT_TEMPLATE,
        negative_prompt=(
            "blurry, distorted, low quality, sketch, drawing, illustration, flat, cartoon, "
            "dark background, cluttered background, bad lighting, overexposed, underexposed"
        ),
        params={"num_inference_steps": num_inference_steps},
    ))

    firered_step = AgentStep(
        name="firered_generate",
        agent=firered_agent,
        input_map={
            "image": "sketch",    # IMAGE 1: sketch grid (constant reference)
            "image_2": "image",   # IMAGE 2: last result (first iter: seeded with sketch)
        },
        output_map={"image": "image"},
        template_kwargs={
            "material": spec.get("material", "leather"),
            "color": spec.get("color", "black"),
            "camera_angle": spec.get("camera_angle", "3/4 view"),
            "extra_specs": extra_specs_text,
            "feedback": "",
        },
    )

    full_spec = {
        "material": spec.get("material", "leather"),
        "color": spec.get("color", "black"),
        "camera_angle": spec.get("camera_angle", "3/4 view"),
        **spec.get("extra", {}),
    }

    dual_judge = make_dual_judge(
        make_sketch_judge(
            session=vlm_session,
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
        ),
    )

    loop = LoopStep(
        name="sketch_to_shoe_loop",
        body=[firered_step],
        judge=dual_judge,
        max_iterations=max_iterations,
        best_fn=make_best_fn(
            session=vlm_session,
            sketch_key="sketch",
            output_key="image",
        ),
        swap_models=swap_models,
        output_key="image",
        feedback_template_var="feedback",
    )

    return Pipeline(name="sketch_to_shoe", steps=[loop])


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
    peak_gb: float,
    sketch_features: list[str] | None = None,
    tolerance: str = "strict",
) -> None:
    """Save all results, intermediates, and metrics to run_dir."""
    run_dir.mkdir(parents=True, exist_ok=True)
    sketch_grid.save(run_dir / "input_sketch_grid.png")

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "total_elapsed_s": total_elapsed,
        "peak_vram_gb": peak_gb,
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
        meta = getattr(it, "metadata", None)
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

    (run_dir / "results.json").write_text(
        json.dumps(results_data, indent=2, default=str)
    )

    lines = [
        "Sketch-to-Shoe Loop Results",
        "=" * 60,
        f"Date: {datetime.now().isoformat()}",
        f"Total time: {total_elapsed:.1f}s",
        f"Peak VRAM: {peak_gb:.2f} GB",
        f"Material: {spec.get('material')}  Color: {spec.get('color')}  Angle: {spec.get('camera_angle')}",
        f"Sketch features: {sketch_features or []}",
        f"Tolerance: {tolerance}",
        f"Iterations: {len(loop_result.iterations)}",
        "",
    ]
    for it in loop_result.iterations:
        verdict = "ACCEPT" if it.accepted else "REJECT"
        lines.append(f"  Iteration {it.index}: {verdict} ({it.duration_ms:.1f}ms)")
        meta = getattr(it, "metadata", None)
        if meta:
            if meta.get("sketch_scores"):
                ss = meta["sketch_scores"]
                lines.append(f"    Sketch: {', '.join(f'{k}={v}' for k,v in ss.items())} "
                              f"(avg={meta.get('sketch_avg')})")
            if meta.get("spec_scores"):
                ss = meta["spec_scores"]
                lines.append(f"    Spec:   {', '.join(f'{k}={v}' for k,v in ss.items())} "
                              f"(avg={meta.get('spec_avg')})")
        lines.append(f"    Feedback: {it.feedback[:200]}")
        lines.append("")

    lines.append(f"Final verdict: {results_data.get('final_verdict', 'unknown')}")
    lines.append(f"Output: {run_dir}")
    summary = "\n".join(lines)
    print(f"\n{summary}")
    (run_dir / "summary.txt").write_text(summary)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sketch-to-shoe agentic loop")
    parser.add_argument("--sketches", type=str, nargs="+", required=True,
        help="Path(s) to sketch image(s)")
    parser.add_argument("--material", type=str, default="leather",
        help="Shoe material (e.g. leather, suede, canvas)")
    parser.add_argument("--color", type=str, default="black",
        help="Shoe color (e.g. black, white, red)")
    parser.add_argument("--camera-angle", type=str, default="3/4 view",
        dest="camera_angle",
        help="Camera angle: '3/4 view', 'side view', 'front view', 'top view', or custom")
    parser.add_argument("--spec", type=str, nargs="*", default=[],
        metavar="KEY=VALUE",
        help="Open-ended extras, e.g. style=elegant note='chunky sole'")
    parser.add_argument("--max-iter", type=int, default=5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--keep-both", action="store_true",
        help="Keep all models loaded (needs ~74GB+ VRAM)")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--tolerance", type=str, default="strict",
        choices=["generous", "moderate", "strict"])
    parser.add_argument("--spacing", type=int, default=20,
        help="Pixel spacing between sketches in grid (default: 20)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        return

    print("=== Sketch-to-Shoe Agentic Loop ===")
    print(f"Sketches: {args.sketches}")
    print(f"Material: {args.material}")
    print(f"Color: {args.color}")
    print(f"Camera angle: {args.camera_angle}")
    extra_spec = _parse_spec_args(args.spec)
    if extra_spec:
        print(f"Extra spec: {extra_spec}")
    print(f"Max iterations: {args.max_iter}")
    print(f"Inference steps: {args.steps}")
    print(f"Memory mode: {'keep-both' if args.keep_both else 'swap'}")
    print(f"Scale: {args.scale}x")
    print(f"Tolerance: {args.tolerance}")
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
        "color": args.color,
        "camera_angle": args.camera_angle,
        "extra": extra_spec,
    }

    vlm_session = VLMSession("qwen3_vl_8b")
    sketch_media = ImageMedia(image=sketch_grid)

    print("Extracting sketch design features...")
    sketch_features = extract_sketch_features(vlm_session, sketch_media)
    print(f"Features: {sketch_features}")
    print()

    pipeline = build_pipeline(
        sketch_media=sketch_media,
        spec=spec,
        max_iterations=args.max_iter,
        num_inference_steps=args.steps,
        swap_models=not args.keep_both,
        vlm_session=vlm_session,
        sketch_features=sketch_features,
        tolerance=args.tolerance,
    )
    logged = LoggedPipeline(pipeline)

    context = {
        "sketch": sketch_media,
        "image": sketch_media,  # seed for first iteration
    }

    if args.keep_both:
        print("Loading all models (keep-both mode)...")
        vlm_session.load()
        pipeline.load()

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

    loop_result = result.get("sketch_to_shoe_loop_history")
    final_img = result.get("image")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"{args.max_iter}iter_{args.steps}steps_{ts}"

    save_results(
        run_dir=run_dir,
        loop_result=loop_result if isinstance(loop_result, LoopResult) else LoopResult(),
        result_context=result,
        sketch_grid=sketch_grid,
        final_img=final_img,
        spec=spec,
        total_elapsed=total_elapsed,
        peak_gb=peak_gb,
        sketch_features=sketch_features,
        tolerance=args.tolerance,
    )

    gc.collect()
    torch.cuda.empty_cache()
    print(f"\nDone. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
