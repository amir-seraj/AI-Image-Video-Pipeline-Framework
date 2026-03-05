"""Benchmark: compare image-edit model variants.

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

NUM_STEPS = 30

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

    def test_image_edit_variants(self, request):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        source_file = request.config.getoption("--source-image")

        # -- inputs -------------------------------------------------------
        source = PILImage.open(IMAGE_DIR / source_file).convert("RGB")
        # Qwen produces best results with 1:1 input — pad to square
        source = _pad_to_square(source)
        shoes = PILImage.open(IMAGE_DIR / "shoes001.jpeg").convert("RGB")

        # -- output dir ---------------------------------------------------
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = OUTPUT_DIR / f"{NUM_STEPS}steps_{ts}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # save inputs once for reference
        source.save(run_dir / "input_source.png")
        shoes.save(run_dir / "input_shoes.png")

        enabled = [m for m in MODELS if m[3]]
        results: list[TimingResult] = []

        for idx, (registry_name, label, steps_kwarg, _) in enumerate(enabled, 1):
            print(f"\n{'='*60}")
            print(f"[{idx}/{len(enabled)}] {label} ({registry_name})")
            print(f"{'='*60}")

            model = None
            try:
                # -- load -------------------------------------------------
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

                # -- inference + save (timed) -----------------------------
                print(f"  Running ({NUM_STEPS} steps)...", end="", flush=True)

                def run(m=model, sk=steps_kwarg, lbl=label):
                    img = m._edit(
                        images=[source, shoes],
                        prompt=PROMPT,
                        negative_prompt="",
                        **{sk: NUM_STEPS},
                    )
                    img.save(run_dir / f"{lbl}_output.png")
                    return img

                wall, gpu_ms, peak, img = _cuda_timed(run)
                size_str = f"{img.size[0]}x{img.size[1]}"
                print(f" {wall:.2f}s  (GPU {gpu_ms:.0f}ms, peak {peak:.2f}GB)")

                results.append(
                    TimingResult(label, registry_name, load_s, wall, gpu_ms, peak, size_str)
                )

            except Exception as e:
                print(f"\n  SKIPPED — {type(e).__name__}: {e}")

            finally:
                # always unload — prevents 45GB+ stuck on GPU after errors
                if model is not None:
                    try:
                        model.unload_model()
                    except Exception:
                        pass
                    del model

            # aggressive cleanup so the next model sees a clean GPU
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()  # second pass catches ref-cycles freed by first
            torch.cuda.empty_cache()
            free_gb = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / (1024**3)
            print(f"  GPU after cleanup: {torch.cuda.memory_allocated() / (1024**3):.2f} GB used, {free_gb:.2f} GB free")

        # -- summary table ------------------------------------------------
        hdr = (
            f"{'Variant':<14} {'Model':<28} {'Load (s)':>9} "
            f"{'Infer (s)':>10} {'GPU (ms)':>10} {'Peak (GB)':>10} {'Size':>12}"
        )
        sep = "-" * 93

        lines = [
            f"Image-Edit Benchmark — {NUM_STEPS} inference steps",
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

        # relative speed
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

        # persist
        (run_dir / "results.txt").write_text(summary + "\n")

        for r in results:
            assert r.infer_s > 0
            assert r.gpu_ms > 0
