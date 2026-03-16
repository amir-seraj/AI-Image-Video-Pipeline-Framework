"""Test runner: 3 parallel lowfi-to-hifi generations with volume + sketch."""
from __future__ import annotations

import sys
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image as PILImage
from casadei.media import ImageMedia
from pipeline import build_pipeline, save_results


SKETCH_PATH = _PROJECT_ROOT / "tests" / "Image" / "lowfisketch001.jpeg"
VOLUME_PATH = _PROJECT_ROOT / "tests" / "Image" / "volume001.jpg"
OUTPUT_BASE = _PROJECT_ROOT / "data" / "results" / "lowfi_to_hifi_test"


def run_once(run_id: int, batch_dir: Path) -> None:
    """Run one generation and save results."""
    print(f"[Run {run_id}] Starting...")
    run_dir = batch_dir / f"run_{run_id:02d}"

    sketch_img = ImageMedia(image=PILImage.open(SKETCH_PATH).convert("RGB"))
    volume_img = ImageMedia(image=PILImage.open(VOLUME_PATH).convert("RGB"))

    spec = {"volume": True}
    pipeline, agent = build_pipeline(spec)

    agent.load()
    try:
        t0 = time.time()
        context = {"sketch": sketch_img, "volume": volume_img}
        result_context = pipeline.run(context)
        elapsed = time.time() - t0

        result_image = result_context.get("image")
        token_records = list(agent.token_usage_log)

        save_results(
            run_dir=run_dir,
            result_image=result_image,
            spec=spec,
            total_elapsed=elapsed,
            token_records=token_records,
        )
        print(f"[Run {run_id}] Done in {elapsed:.1f}s -> {run_dir}")
    finally:
        agent.unload()


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = OUTPUT_BASE / ts
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running 3 parallel generations...")
    print(f"Sketch: {SKETCH_PATH}")
    print(f"Volume: {VOLUME_PATH}")
    print(f"Output: {batch_dir}\n")

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(run_once, i, batch_dir): i for i in range(1, 4)}
        for fut in as_completed(futures):
            run_id = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"[Run {run_id}] FAILED: {e}")


if __name__ == "__main__":
    main()
