"""Benchmark: caching strategies for Wan 2.1 I2V FP8.

Tests different diffusers-native caching hooks to measure speedup
over the baseline FP8 pipeline.

Usage:
    python -u tests/bench_cache.py
"""

import gc
import time
import torch
from pathlib import Path
from PIL import Image as PILImage

from casadei import Agent, AgentConfig, AgentStep, ImageMedia, LoggedPipeline, Pipeline, VideoMedia
from casadei.providers.wan_i2v_fp8 import WanImageToVideoFP8

IMAGE_PATH = Path(__file__).parent / "Image" / "legs001.jpeg"
OUTPUT_DIR = Path(__file__).parent / "output" / "cache_bench"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_FRAMES = 17
NUM_STEPS = 15  # Fewer steps for faster benchmarking
PROMPT = (
    "A woman gracefully takes off her black high heel shoe, "
    "slowly unbuckling the ankle strap and sliding the shoe off her foot, "
    "smooth natural motion"
)

WanImageToVideoFP8.DEFAULT_PARAMS["num_frames"] = NUM_FRAMES
WanImageToVideoFP8.DEFAULT_PARAMS["num_inference_steps"] = NUM_STEPS


def load_model():
    """Load the FP8 model once."""
    agent = Agent(
        config=AgentConfig(name="wan_i2v_fp8", model="wan_i2v_fp8", prompt_template="$prompt")
    )
    agent.load()
    return agent


def run_inference(agent, label):
    """Run one inference pass and return timing."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    pipeline = Pipeline(
        name=f"bench_{label}",
        steps=[
            AgentStep(
                name="generate",
                agent=agent,
                input_map={"image": "source_image"},
                output_map={"video": "result_video"},
                template_kwargs={"prompt": PROMPT},
            ),
        ],
    )

    logged = LoggedPipeline(pipeline)
    t0 = time.perf_counter()
    result, log = logged.run({
        "source_image": ImageMedia(image=PILImage.open(IMAGE_PATH)),
    })
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    video = result["result_video"]
    out_path = OUTPUT_DIR / f"{label}.mp4"
    video.save(out_path)

    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    per_step = elapsed / NUM_STEPS
    print(f"  [{label}] {elapsed:.1f}s total, {per_step:.1f}s/step, peak {peak_gb:.1f} GB")
    return elapsed, per_step, peak_gb


def get_pipeline(agent):
    """Get the underlying diffusers pipeline from the agent."""
    return agent._model._pipeline


def main():
    print(f"=== Cache Strategy Benchmark ({NUM_FRAMES} frames, {NUM_STEPS} steps) ===\n")

    # Load model
    print("Loading FP8 model...")
    t0 = time.time()
    agent = load_model()
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    pipe = get_pipeline(agent)
    results = {}

    # --- 1. Baseline: no caching ---
    print("1. Baseline (no caching):")
    results["baseline"] = run_inference(agent, "baseline")

    # --- 2. TaylorSeer Cache ---
    print("\n2. TaylorSeer Cache (cache_interval=5, max_order=1):")
    try:
        from diffusers.hooks import apply_taylorseer_cache, TaylorSeerCacheConfig
        config = TaylorSeerCacheConfig(
            cache_interval=5,
            disable_cache_before_step=3,
            max_order=1,
        )
        apply_taylorseer_cache(pipe.transformer, config)
        results["taylorseer"] = run_inference(agent, "taylorseer")
        pipe.transformer._reset_stateful_cache()
        # Remove hooks
        if hasattr(pipe.transformer, '_hf_hook'):
            pipe.transformer._hf_hook = None
    except Exception as e:
        print(f"  FAILED: {e}")

    # Clear hooks by resetting
    try:
        pipe.transformer._reset_stateful_cache()
    except Exception:
        pass

    # --- 3. FasterCache ---
    print("\n3. FasterCache:")
    try:
        from diffusers.hooks import apply_faster_cache, FasterCacheConfig
        config = FasterCacheConfig(
            spatial_attention_block_skip_range=2,
            tensor_format="BCFHW",
        )
        apply_faster_cache(pipe, config)
        results["fastercache"] = run_inference(agent, "fastercache")
    except Exception as e:
        print(f"  FAILED: {e}")

    # --- 4. First Block Cache ---
    print("\n4. First Block Cache (threshold=0.2):")
    try:
        from diffusers.hooks import apply_first_block_cache, FirstBlockCacheConfig
        config = FirstBlockCacheConfig(threshold=0.2)
        apply_first_block_cache(pipe.transformer, config)
        results["fbc"] = run_inference(agent, "fbc")
    except Exception as e:
        print(f"  FAILED: {e}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"{'Strategy':<20} {'Total (s)':>10} {'Per Step':>10} {'Peak GB':>10} {'Speedup':>10}")
    print(f"{'-'*60}")
    base_total = results.get("baseline", (1, 1, 1))[0]
    for name, (total, per_step, peak) in results.items():
        speedup = base_total / total
        print(f"{name:<20} {total:>10.1f} {per_step:>10.1f} {peak:>10.1f} {speedup:>9.2f}x")
    print(f"{'='*60}")

    agent.unload()


if __name__ == "__main__":
    main()
