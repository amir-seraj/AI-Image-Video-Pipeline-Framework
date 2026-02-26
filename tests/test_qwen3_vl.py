"""Test script for Qwen3-VL vision-language providers.

Run: python tests/test_qwen3_vl.py              # default 8b
     python tests/test_qwen3_vl.py --model 30b  # 30B MoE variant
"""

import argparse
import time
from pathlib import Path

from PIL import Image as PILImage

from casadei import MODELS_DIR
from casadei.media import ImageMedia, TextMedia, MediaBundle

IMAGE_DIR = Path(__file__).parent / "Image"


def stream_and_collect(model, inputs: MediaBundle) -> tuple[str, float, int]:
    """Stream output to stdout and return (full_text, elapsed_seconds, chunk_count)."""
    t0 = time.perf_counter()
    chunks = []
    for chunk in model.run_streaming(inputs):
        print(chunk, end="", flush=True)
        chunks.append(chunk)
    elapsed = time.perf_counter() - t0
    full_text = "".join(chunks)
    print(f"\n--- ({len(chunks)} chunks, {elapsed:.1f}s) ---\n")
    return full_text, elapsed, len(chunks)


MODELS = {
    "8b": ("casadei.providers.qwen3_vl_8b", "Qwen3VL8B"),
    "30b": ("casadei.providers.qwen3_vl_30b", "Qwen3VL30B"),
}


def load_provider(name: str):
    """Dynamically import and instantiate a Qwen3-VL provider by short name."""
    module_path, class_name = MODELS[name]
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)()


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3-VL providers")
    parser.add_argument(
        "--model", choices=list(MODELS), default="8b",
        help="Model variant to test (default: 8b)",
    )
    args = parser.parse_args()

    print(f"=== Qwen3-VL-{args.model.upper()} Test ===")
    model = load_provider(args.model)

    print("\nLoading model...")
    model.load_model()
    print("Model loaded.\n")

    # --- Step 1: Describe shoes001.jpeg ---
    shoes_path = IMAGE_DIR / "shoes001.jpeg"
    shoes_img = PILImage.open(shoes_path)
    print(f"Image: {shoes_path.name} ({shoes_img.size})")

    extract_inputs = MediaBundle(items={
        "image": ImageMedia(image=shoes_img),
        "prompt": TextMedia(
            text=(
                "Extract the visual features of these shoes as a flat comma-separated list. "
                "Include: type, color, material, heel height, toe shape, closure, sole style, "
                "pattern, occasion, and any other notable details. "
                "Output ONLY the comma-separated features, nothing else."
            ),
        ),
    })

    print("--- Features ---")
    stream_and_collect(model, extract_inputs)

    model.unload_model()
    print("Done.")


if __name__ == "__main__":
    main()
