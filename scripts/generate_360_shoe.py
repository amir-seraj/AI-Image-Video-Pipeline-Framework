"""Generate a 360-degree product video of a Casadei shoe using Veo 3.1.

Uses the front 3/4 view as the starting frame and 3 reference images
(front, back, opposite side) to preserve subject appearance while
the shoe rotates on a clean studio background.

Requires GEMINI_API_KEY environment variable.

Usage:
    python scripts/generate_360_shoe.py
"""

from pathlib import Path

from PIL import Image

from casadei.media import ImageMedia, MediaBundle, TextMedia
from casadei.providers.veo_video_generate import VeoVideoGenerate

# ---------- paths ----------
IMAGE_DIR = Path(__file__).resolve().parent.parent.parent / (
    "casadei-front/mvp/Shoe 2/generated/Varitation 1"
)
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data/results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Front 3/4 view → first frame (starting position)
FIRST_FRAME = IMAGE_DIR / "var 1 - 1.png"

# Three most distinct angles as reference images for subject consistency
REFERENCE_IMAGES = [
    IMAGE_DIR / "var 1 - 3.png",   # front view
    IMAGE_DIR / "var 1 - 5.png",   # back view
    IMAGE_DIR / "var 1 -6.png",    # left side (sole visible)
]

PROMPT = (
    "A smooth, continuous 360-degree rotation of a luxury burgundy patent leather "
    "Casadei platform high-heel shoe with ankle strap and peep toe. "
    "The shoe sits on a clean, minimal light gray studio surface and slowly rotates "
    "clockwise, revealing every angle — front, side, back, and sole. "
    "Studio lighting with soft reflections on the glossy patent leather. "
    "No camera movement, only the shoe rotates in place. "
    "Professional e-commerce product video, photorealistic quality."
)


def main():
    # Load images
    first_frame = Image.open(FIRST_FRAME)
    references = [Image.open(p) for p in REFERENCE_IMAGES]

    # Build input bundle — reference images only (no first frame),
    # as the fast model may not support combining both.
    inputs = MediaBundle(items={
        "prompt": TextMedia(text=PROMPT),
        "reference_0": ImageMedia(image=references[0]),
        "reference_1": ImageMedia(image=references[1]),
        "reference_2": ImageMedia(image=references[2]),
    })

    # Initialize and run the model
    model = VeoVideoGenerate()
    model.load_model()

    print("Starting 360 video generation...")
    result = model.run(inputs, aspect_ratio="16:9")

    # Save output
    output_path = OUTPUT_DIR / "shoe2_var1_360.mp4"
    result["video"].save(output_path)
    print(f"360 video saved to {output_path}")

    model.unload_model()


if __name__ == "__main__":
    main()
