"""Benchmark: compare all 3 Qwen image-edit variants (full, fp8, gguf).

Runs each variant sequentially — load, infer (5 steps), save, unload —
so only one model is in memory at a time.  All outputs and a timing
summary land in a single timestamped folder.

Usage:
    python -m pytest tests/test_benchmark.py -v -s
    python -m pytest tests/test_benchmark.py -v -s --source-image legs001.jpeg
"""

import gc
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pytest
import torch
from PIL import Image as PILImage

from casadei.models.registry import default_registry

OUTPUT_DIR = Path(__file__).parent / "output" / "benchmarks"
IMAGE_DIR = Path(__file__).parent / "Image"

NUM_STEPS = 15

MODELS = [
    ("qwen_image_edit", "full"),
    ("qwen_image_edit_fp8", "fp8"),
    ("qwen_image_edit_gguf", "gguf"),
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
    """Run *fn* bracketed by CUDA synchronisation.

    Returns (wall_s, gpu_ms, peak_gb, fn_result).
    """
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


class TestBenchmark:

    def test_qwen_variants(self, request):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        source_file = request.config.getoption("--source-image")

        # -- inputs -------------------------------------------------------
        source = PILImage.open(IMAGE_DIR / source_file).convert("RGB")
        # legs image is not square — pad it; model image is fine as-is
        if "legs" in source_file:
            source = _pad_to_square(source)
        shoes = PILImage.open(IMAGE_DIR / "shoes001.jpeg").convert("RGB")

        # -- output dir ---------------------------------------------------
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = OUTPUT_DIR / f"qwen_bench_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # save inputs once for reference
        source.save(run_dir / "input_source.png")
        shoes.save(run_dir / "input_shoes.png")

        results: list[TimingResult] = []

        for idx, (registry_name, label) in enumerate(MODELS, 1):
            print(f"\n{'='*60}")
            print(f"[{idx}/{len(MODELS)}] {label} ({registry_name})")
            print(f"{'='*60}")

            # -- load -----------------------------------------------------
            print("  Loading...", end="", flush=True)
            load_t0 = time.perf_counter()
            cls = default_registry.get(registry_name)
            model = cls()
            model.load_model()
            load_s = time.perf_counter() - load_t0
            print(f" {load_s:.1f}s")

            # no step images
            if hasattr(model, "save_steps_dir"):
                model.save_steps_dir = None

            # -- inference + save (timed) ---------------------------------
            print(f"  Running ({NUM_STEPS} steps)...", end="", flush=True)

            def run(m=model):
                img = m._edit(
                    images=[source, shoes],
                    prompt=PROMPT,
                    negative_prompt="",
                    num_inference_steps=NUM_STEPS,
                )
                img.save(run_dir / f"{label}_output.png")
                return img

            wall, gpu_ms, peak, img = _cuda_timed(run)
            size_str = f"{img.size[0]}x{img.size[1]}"
            print(f" {wall:.2f}s  (GPU {gpu_ms:.0f}ms, peak {peak:.2f}GB)")

            results.append(
                TimingResult(label, registry_name, load_s, wall, gpu_ms, peak, size_str)
            )

            # -- unload ---------------------------------------------------
            model.unload_model()
            del model
            gc.collect()
            torch.cuda.empty_cache()

        # -- summary table ------------------------------------------------
        hdr = (
            f"{'Variant':<10} {'Model':<25} {'Load (s)':>9} "
            f"{'Infer (s)':>10} {'GPU (ms)':>10} {'Peak (GB)':>10} {'Size':>12}"
        )
        sep = "-" * 86

        lines = [
            f"Qwen Image-Edit Benchmark — {NUM_STEPS} inference steps",
            f"Date: {datetime.now().isoformat()}",
            "",
            hdr,
            sep,
        ]
        for r in results:
            lines.append(
                f"{r.label:<10} {r.registry_name:<25} {r.load_s:>9.1f} "
                f"{r.infer_s:>10.2f} {r.gpu_ms:>10.0f} {r.peak_gb:>10.2f} {r.output_size:>12}"
            )
        lines.append(sep)

        # relative speed
        infers = [r.infer_s for r in results]
        slowest = max(infers)
        lines.append("")
        lines.append("Relative inference speed (wall-clock):")
        for r in results:
            ratio = slowest / r.infer_s
            bar = "#" * int(ratio * 20)
            lines.append(f"  {r.label:<10} {ratio:>5.2f}x  {bar}")

        lines.append("")
        lines.append(f"Outputs: {run_dir}")

        summary = "\n".join(lines)
        print(f"\n{'='*86}")
        print(summary)
        print(f"{'='*86}")

        # persist
        (run_dir / "results.txt").write_text(summary + "\n")

        for r in results:
            assert r.infer_s > 0
            assert r.gpu_ms > 0
