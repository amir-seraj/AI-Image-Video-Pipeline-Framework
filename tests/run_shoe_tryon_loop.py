"""Shoe replacement agentic loop — integration test with full logging.

Runs the iterative shoe replacement pipeline using FireRed for generation
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
from PIL import Image as PILImage, ImageDraw, ImageFont

# Add project root to path so we can import workflow scripts
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "workflows" / "shoe_tryon_loop" / "scripts"))

from casadei import (
    Agent, AgentConfig, AgentStep, ImageMedia, TextMedia,
    LoggedPipeline, Pipeline,
)
from casadei.loop import LoopStep, LoopResult
from judge import VLMSession, make_judge, make_multi_judge, make_best_fn, extract_features

IMAGE_DIR = Path(__file__).parent / "Image"
OUTPUT_DIR = Path(__file__).parent / "output" / "shoe_tryon_loop"

VLM_MODELS = {
    "8b":  "qwen3_vl_8b",
    "8b-thinking": "qwen3_vl_8b_thinking",
    "30b": "qwen3_vl_30b",
}

PROMPT_TEMPLATE = (
    "The first image is the reference shoe photo — the exact shoe design "
    "to reproduce on the person's feet. "
    "The second image shows a person wearing shoes. "
    "Replace BOTH shoes on the person's feet — left foot AND right foot — "
    "with the exact shoe from the reference shoe photo. Every visible shoe "
    "must match the reference. Preserve the person's pose, legs, "
    "clothing, and background unchanged. $feedback"
)


def _pad_to_square(img: PILImage.Image) -> PILImage.Image:
    w, h = img.size
    if w == h:
        return img
    size = max(w, h)
    sq = PILImage.new("RGB", (size, size), (255, 255, 255))
    sq.paste(img, ((size - w) // 2, (size - h) // 2))
    return sq


def _wrap_text(text: str, max_px: int, font) -> list[str]:
    """Word-wrap *text* so each line fits within *max_px* pixels."""
    dummy = PILImage.new("RGB", (1, 1))
    draw = ImageDraw.Draw(dummy)
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        w = draw.textbbox((0, 0), candidate, font=font)[2]
        if w <= max_px:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]


def _build_iteration_grid(
    loop_result: "LoopResult",
    best_selection_index: int | None,
    best_selection_reason: str | None,
    max_img_height: int = 400,
) -> PILImage.Image | None:
    """Build a side-by-side grid of all iteration outputs.

    Each column shows:
      - Title bar:  "Iteration N", green on accept, grey otherwise
      - Candidate image scaled to *max_img_height*
      - Judge panel: avg score + first line of feedback

    If *best_selection_index* is provided (1-based), that column gets a
    gold highlight border and a "★ BEST" title.
    If *best_selection_reason* is provided, a text panel is appended at
    the bottom of the whole image.
    """
    from casadei.media import ImageMedia

    # Gather (LoopIteration, PIL image) pairs
    cells: list[tuple] = []
    for it in loop_result.iterations:
        img = it.outputs.get("image")
        if isinstance(img, ImageMedia):
            cells.append((it, img.image))

    if not cells:
        return None

    # ---- fonts ----------------------------------------------------------------
    _FONT_PATH_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    _FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        font_title = ImageFont.truetype(_FONT_PATH_BOLD, 22)
        font_judge = ImageFont.truetype(_FONT_PATH, 15)
        font_reason = ImageFont.truetype(_FONT_PATH, 15)
    except (OSError, IOError):
        font_title = ImageFont.load_default()
        font_judge = font_title
        font_reason = font_title

    # ---- layout constants -----------------------------------------------------
    TITLE_H = 36        # pixels reserved above each image for the title bar
    JUDGE_H = 52        # pixels reserved below each image for judge text
    BORDER = 6          # highlight border thickness (px)
    HIGHLIGHT = (255, 215, 0)  # gold
    ACCEPTED_BG = (45, 160, 45)
    BEST_BG = (200, 155, 0)
    DEFAULT_BG = (70, 70, 70)
    LINE_H = 18         # judge-text line height (px)
    REASON_PAD = 12

    # ---- scale images to max_img_height --------------------------------------
    scaled: list[PILImage.Image] = []
    for _, img in cells:
        if img.height > max_img_height:
            ratio = max_img_height / img.height
            scaled.append(img.resize((int(img.width * ratio), max_img_height), PILImage.LANCZOS))
        else:
            scaled.append(img)

    img_h = max(s.height for s in scaled)  # tallest image in the set
    cell_h = TITLE_H + img_h + JUDGE_H
    total_w = sum(s.width for s in scaled)

    # ---- reason panel height -------------------------------------------------
    reason_h = 0
    reason_lines: list[str] = []
    if best_selection_reason:
        header = "Selector reason:  "
        full_text = header + best_selection_reason
        reason_lines = _wrap_text(full_text, total_w - 2 * REASON_PAD, font_reason)
        reason_h = len(reason_lines) * LINE_H + 2 * REASON_PAD + 4

    # ---- canvas ---------------------------------------------------------------
    canvas = PILImage.new("RGB", (total_w, cell_h + reason_h), (230, 230, 230))
    draw = ImageDraw.Draw(canvas)

    x = 0
    for col, (it, _) in enumerate(cells):
        img = scaled[col]
        cell_w = img.width
        is_best = best_selection_index is not None and (col + 1) == best_selection_index

        # title bar
        if it.accepted:
            title_bg = ACCEPTED_BG
            title = f"Iteration {it.index + 1}  ✓ ACCEPTED"
        elif is_best:
            title_bg = BEST_BG
            title = f"Iteration {it.index + 1}  ★ BEST"
        else:
            title_bg = DEFAULT_BG
            title = f"Iteration {it.index + 1}"

        draw.rectangle([x, 0, x + cell_w - 1, TITLE_H - 1], fill=title_bg)
        tb = draw.textbbox((0, 0), title, font=font_title)
        draw.text(
            (x + (cell_w - (tb[2] - tb[0])) // 2, (TITLE_H - (tb[3] - tb[1])) // 2),
            title, fill=(255, 255, 255), font=font_title,
        )

        # candidate image (vertically top-aligned within img_h)
        canvas.paste(img, (x, TITLE_H))

        # judge panel
        y_judge = TITLE_H + img_h
        draw.rectangle([x, y_judge, x + cell_w - 1, cell_h - 1], fill=(255, 255, 255))

        judge_lines: list[str] = []
        if it.metadata and it.metadata.get("scores"):
            scores = it.metadata["scores"]
            avg = it.metadata.get("avg_score", 0)
            scores_str = "  ".join(f"{k}={v}" for k, v in scores.items())
            judge_lines.append(f"avg={avg}  |  {scores_str}")
        verdict = "ACCEPT" if it.accepted else "REJECT"
        fb = (it.feedback or "").replace("\n", " ")
        judge_lines.append(f"[{verdict}]  {fb}")

        for li, jline in enumerate(judge_lines[:2]):
            # clip to cell width
            while jline and draw.textbbox((0, 0), jline, font=font_judge)[2] > cell_w - 6:
                jline = jline[:-1]
            draw.text((x + 4, y_judge + 4 + li * LINE_H), jline, fill=(30, 30, 30), font=font_judge)

        # gold highlight border for best selection
        if is_best:
            draw.rectangle([x, 0, x + cell_w - 1, cell_h - 1], outline=HIGHLIGHT, width=BORDER)

        # separator between columns
        if col < len(cells) - 1:
            draw.line([(x + cell_w - 1, 0), (x + cell_w - 1, cell_h - 1)],
                      fill=(160, 160, 160), width=1)
        x += cell_w

    # ---- reason panel --------------------------------------------------------
    if reason_lines:
        y0 = cell_h
        draw.rectangle([0, y0, total_w - 1, cell_h + reason_h - 1], fill=(255, 252, 220))
        draw.line([(0, y0), (total_w - 1, y0)], fill=(180, 160, 0), width=2)
        for li, rline in enumerate(reason_lines):
            draw.text(
                (REASON_PAD, y0 + REASON_PAD + li * LINE_H),
                rline, fill=(20, 20, 20), font=font_reason,
            )

    return canvas


def build_pipeline(
    max_iterations: int = 5,
    num_inference_steps: int = 30,
    swap_models: bool = True,
    vlm_session: VLMSession | None = None,
    features: list[str] | None = None,
    tolerance: str = "strict",
    multi_judge: bool = False,
) -> Pipeline:
    """Build the iterative shoe replacement pipeline."""
    if vlm_session is None:
        vlm_session = VLMSession("qwen3_vl_8b")

    firered_agent = Agent(AgentConfig(
        name="firered_shoe_replace",
        model="firered_image_edit",
        description="FireRed shoe replacement with iterative refinement",
        prompt_template=PROMPT_TEMPLATE,
        negative_prompt="blurry, distorted, low quality, bad anatomy, missing shoes, extra limbs",
        params={"num_inference_steps": num_inference_steps},
    ))

    firered_step = AgentStep(
        name="firered_edit",
        agent=firered_agent,
        input_map={
            "image": "shoe",        # always the reference shoe photo (first image)
            "image_2": "image",     # first iter: original person; subsequent: last generation
        },
        output_map={"image": "image"},
        template_kwargs={"feedback": ""},
    )

    if multi_judge:
        judge_fn = make_multi_judge(
            session=vlm_session,
            shoe_key="shoe",
            candidate_key="image",
            tolerance=tolerance,
        )
    else:
        judge_fn = make_judge(
            session=vlm_session,
            shoe_key="shoe",
            candidate_key="image",
            features=features,
            tolerance=tolerance,
        )

    loop = LoopStep(
        name="tryon_loop",
        body=[firered_step],
        judge=judge_fn,
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
    features: list[str] | None = None,
    tolerance: str = "strict",
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
            iter_data["stale_count"] = it.metadata.get("stale_count", 0)

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

    # Build and save iteration grid image
    grid = _build_iteration_grid(
        loop_result=loop_result,
        best_selection_index=results_data.get("best_selected_candidate"),
        best_selection_reason=results_data.get("best_selection_vlm_response"),
    )
    if grid is not None:
        grid.save(run_dir / "iterations_grid.png")
        results_data["iterations_grid"] = "iterations_grid.png"

    # Write JSON report
    (run_dir / "results.json").write_text(
        json.dumps(results_data, indent=2, default=str)
    )

    # Write human-readable summary
    lines = [
        f"Shoe Replacement Loop Results",
        f"{'=' * 60}",
        f"Date: {datetime.now().isoformat()}",
        f"Total time: {total_elapsed:.1f}s",
        f"Peak VRAM: {peak_gb:.2f} GB",
        f"Features: {features or []}",
        f"Tolerance: {tolerance}",
        f"Iterations: {len(loop_result.iterations)}",
        f"",
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
    parser = argparse.ArgumentParser(description="Shoe replacement agentic loop test")
    parser.add_argument(
        "--source", type=str,
        default=str(IMAGE_DIR / "model001.jpeg"),
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
    parser.add_argument(
        "--vlm", choices=list(VLM_MODELS), default="8b",
        help="VLM variant for judging and best-selection (default: 8b)",
    )
    parser.add_argument(
        "--tolerance", type=str, default="strict",
        choices=["generous", "moderate", "strict"],
        help="Judge tolerance level (default: strict)",
    )
    parser.add_argument(
        "--multi-judge", action="store_true",
        help="Use 4 specialized judge agents instead of one monolithic judge",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        return

    print(f"=== Shoe Replacement Agentic Loop ===")
    print(f"Source: {args.source}")
    print(f"Shoes: {args.shoes}")
    print(f"Max iterations: {args.max_iter}")
    print(f"Inference steps: {args.steps}")
    print(f"Memory mode: {'keep-both' if args.keep_both else 'swap'}")
    print(f"Padding: {'off (original aspect ratio)' if args.no_pad else 'square'}")
    print(f"Scale: {args.scale}x")
    print(f"VLM: Qwen3-VL-{args.vlm.upper()}")
    print(f"Tolerance: {args.tolerance}")
    print(f"Judge: {'multi-agent (4 specialists)' if args.multi_judge else 'single'}")
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
    vlm_session = VLMSession(VLM_MODELS[args.vlm])

    # Extract shoe features for structured judging (skip for multi-judge — it has fixed features)
    shoe_media = ImageMedia(image=shoes)
    if args.multi_judge:
        features = None
        print("Using multi-judge (4 specialized agents — skipping feature extraction)")
    else:
        print("Extracting shoe features...")
        features = extract_features(vlm_session, shoe_media)
        print(f"Features: {features}")
    print()

    # Build pipeline
    pipeline = build_pipeline(
        max_iterations=args.max_iter,
        num_inference_steps=args.steps,
        swap_models=not args.keep_both,
        vlm_session=vlm_session,
        features=features,
        tolerance=args.tolerance,
        multi_judge=args.multi_judge,
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
        features=features,
        tolerance=args.tolerance,
    )

    # Cleanup
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\nDone. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
