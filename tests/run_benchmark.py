"""Benchmark: compare image-edit model variants (standalone runner).

Runs each enabled variant sequentially — load, infer, save, unload —
so only one model is in memory at a time.  All outputs and a timing
summary land in a single timestamped folder.

Unlike test_benchmark.py (pytest), this script imports real GPU packages
directly — conftest.py's diffusers mock does not apply here.

Usage:
    python -u tests/run_benchmark.py
    python -u tests/run_benchmark.py --source-image legs001.jpeg
    python -u tests/run_benchmark.py --steps 10
    python -u tests/run_benchmark.py --models hunyuan_qint4,firered
"""

import argparse
import gc
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image as PILImage

from casadei.models.registry import default_registry

OUTPUT_DIR = Path(__file__).parent / "output" / "benchmarks"
IMAGE_DIR = Path(__file__).parent / "Image"

# (registry_name, label, steps_kwarg, enabled_by_default)
MODELS = [
    ("longcat_image_edit",         "longcat",             "num_inference_steps",  False),
    ("hunyuan_image3_nf4",         "hunyuan_nf4",         "diff_infer_steps",    False),
    ("hunyuan_image3_distil_int8", "hunyuan_distil_int8", "diff_infer_steps",    False),
    ("hunyuan_image3_qint4",       "hunyuan_qint4",       "diff_infer_steps",    True),
    ("firered_image_edit",         "firered",             "num_inference_steps",  True),
    ("qwen_image_edit",            "full",                "num_inference_steps",  True),
    ("qwen_image_edit_fp8",        "fp8",                 "num_inference_steps",  False),
    ("qwen_image_edit_gguf",       "gguf",                "num_inference_steps",  False),
]

PROMPT = (
    "In the first image there is a person wearing shoes. "
    "Replace the shoes on the person's feet with the shoes shown "
    "in the second image. Keep the person's pose, legs, clothing, "
    "and the background exactly the same."
)


@dataclass
class TimingResult:
    label: str
    registry_name: str
    load_s: float
    infer_s: float
    gpu_ms: float
    peak_gb: float
    output_size: str


def _pad_to_square(img: PILImage.Image) -> PILImage.Image:
    w, h = img.size
    if w == h:
        return img
    size = max(w, h)
    sq = PILImage.new("RGB", (size, size), (255, 255, 255))
    sq.paste(img, ((size - w) // 2, (size - h) // 2))
    return sq


def _cuda_timed(fn):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    t0 = time.perf_counter()
    start_ev.record()
    result = fn()
    end_ev.record()
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    return (
        t1 - t0,
        start_ev.elapsed_time(end_ev),
        torch.cuda.max_memory_allocated() / (1024**3),
        result,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-image", default="model001.jpeg",
                        help="Filename in tests/Image/ (default: model001.jpeg)")
    parser.add_argument("--steps", type=int, default=30,
                        help="Inference steps (default: 30)")
    parser.add_argument("--models", default=None,
                        help="Comma-separated list of labels to run (e.g. hunyuan_qint4,firered). "
                             "Overrides the enabled_by_default flags.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.")
        return

    num_steps = args.steps

    # Resolve which models to run
    if args.models:
        requested = set(args.models.split(","))
        enabled = [m for m in MODELS if m[1] in requested or m[0] in requested]
        if not enabled:
            print(f"ERROR: no models matched '{args.models}'. "
                  f"Available labels: {[m[1] for m in MODELS]}")
            return
    else:
        enabled = [m for m in MODELS if m[3]]

    source = PILImage.open(IMAGE_DIR / args.source_image).convert("RGB")
    source = _pad_to_square(source)
    shoes = PILImage.open(IMAGE_DIR / "shoes001.jpeg").convert("RGB")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"{num_steps}steps_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    source.save(run_dir / "input_source.png")
    shoes.save(run_dir / "input_shoes.png")

    results: list[TimingResult] = []

    for idx, (registry_name, label, steps_kwarg, _) in enumerate(enabled, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(enabled)}] {label} ({registry_name})")
        print(f"{'='*60}")

        model = None
        try:
            print("  Loading...", end="", flush=True)
            load_t0 = time.perf_counter()
            cls = default_registry.get(registry_name)
            model = cls()
            model.load_model()
            load_s = time.perf_counter() - load_t0
            print(f" {load_s:.1f}s")

            if hasattr(model, "save_steps_dir"):
                model.save_steps_dir = None

            print(f"  Running ({num_steps} steps)...", end="", flush=True)

            def run(m=model, sk=steps_kwarg, lbl=label):
                img = m._edit(
                    images=[source, shoes],
                    prompt=PROMPT,
                    negative_prompt="",
                    **{sk: num_steps},
                )
                img.save(run_dir / f"{lbl}_output.png")
                return img

            wall, gpu_ms, peak, img = _cuda_timed(run)
            size_str = f"{img.size[0]}x{img.size[1]}"
            print(f" {wall:.2f}s  (GPU {gpu_ms:.0f}ms, peak {peak:.2f}GB)")

            results.append(TimingResult(label, registry_name, load_s, wall, gpu_ms, peak, size_str))

        except Exception as e:
            print(f"\n  SKIPPED — {type(e).__name__}: {e}")

        finally:
            if model is not None:
                try:
                    model.unload_model()
                except Exception:
                    pass
                del model

        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
        free_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / (1024**3)
        print(f"  GPU after cleanup: {torch.cuda.memory_allocated() / (1024**3):.2f} GB used, {free_gb:.2f} GB free")

    if not results:
        print("\nNo models completed successfully.")
        return

    # -- summary table ----------------------------------------------------
    hdr = (
        f"{'Variant':<14} {'Model':<28} {'Load (s)':>9} "
        f"{'Infer (s)':>10} {'GPU (ms)':>10} {'Peak (GB)':>10} {'Size':>12}"
    )
    sep = "-" * 93

    lines = [
        f"Image-Edit Benchmark — {num_steps} inference steps",
        f"Date: {datetime.now().isoformat()}",
        "",
        hdr,
        sep,
    ]
    for r in results:
        lines.append(
            f"{r.label:<14} {r.registry_name:<28} {r.load_s:>9.1f} "
            f"{r.infer_s:>10.2f} {r.gpu_ms:>10.0f} {r.peak_gb:>10.2f} {r.output_size:>12}"
        )
    lines.append(sep)

    infers = [r.infer_s for r in results]
    slowest = max(infers)
    lines.append("")
    lines.append("Relative inference speed (wall-clock):")
    for r in results:
        ratio = slowest / r.infer_s
        bar = "#" * int(ratio * 20)
        lines.append(f"  {r.label:<14} {ratio:>5.2f}x  {bar}")

    lines.append("")
    lines.append(f"Outputs: {run_dir}")

    summary = "\n".join(lines)
    print(f"\n{'='*93}")
    print(summary)
    print(f"{'='*93}")

    (run_dir / "results.txt").write_text(summary + "\n")


if __name__ == "__main__":
    main()
