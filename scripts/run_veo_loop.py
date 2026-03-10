"""Generate 4K 8s 360 loop using first+last frame interpolation on Veo 3.1 Fast.

Usage:
    python scripts/run_veo_loop.py <image_dir> [output_path]

    image_dir:   folder containing generated shoe views (must have hero-front-left.png)
    output_path: optional, defaults to data/results/360_loop.mp4

Example:
    python scripts/run_veo_loop.py data/results/f6a9dadf901b/3807c85b664f
"""

import sys
from google import genai
from google.genai import types
from PIL import Image
from pathlib import Path
import time, io

# ---------- config ----------
# Always use hero-front-left as the canonical first/last frame.
# It's a 3/4 angle that every shoe variation generates, shows depth,
# and is a single shoe with clean composition.
FRAME_FILENAME = "hero-front-left.png"

PROMPT = (
    "Fast-spinning product turntable video. The luxury shoe shown in the image "
    "spins rapidly on a motorized turntable. "
    "The turntable completes exactly one full 360-degree revolution at constant speed. "
    "The rotation is quick, steady, and never stops or reverses. "
    "Fixed camera, clean gray studio background, soft product lighting. "
    "Professional e-commerce turntable spin, photorealistic."
)

MODEL = "veo-3.1-fast-generate-preview"
ASPECT_RATIO = "16:9"
RESOLUTION = "4k"
POLL_INTERVAL = 10


# ---------- helpers ----------
def pil_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def pad_to_16_9(img):
    """Pad image to 16:9 by extending with the average edge background color."""
    import numpy as np

    w, h = img.size
    target_w = round(h * 16 / 9)
    if target_w <= w:
        return img

    # Sample background from the four corner regions (away from the shoe)
    arr = np.array(img)
    corners = np.concatenate([
        arr[:20, :20].reshape(-1, 3),
        arr[:20, -20:].reshape(-1, 3),
        arr[-20:, :20].reshape(-1, 3),
        arr[-20:, -20:].reshape(-1, 3),
    ])
    bg_color = tuple(int(v) for v in corners.mean(axis=0))

    canvas = Image.new("RGB", (target_w, h), bg_color)
    offset_x = (target_w - w) // 2
    canvas.paste(img, (offset_x, 0))
    return canvas


# ---------- main ----------
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    img_dir = Path(sys.argv[1])
    if not img_dir.is_absolute():
        img_dir = Path.cwd() / img_dir

    frame_path = img_dir / FRAME_FILENAME
    if not frame_path.exists():
        print(f"ERROR: {frame_path} not found. The image directory must contain {FRAME_FILENAME}")
        sys.exit(1)

    default_out = Path(__file__).resolve().parent.parent / "data/results/360_loop.mp4"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else default_out
    out.parent.mkdir(parents=True, exist_ok=True)

    # Load and pad to 16:9
    hero = Image.open(frame_path)
    hero = pad_to_16_9(hero)
    hero_bytes = pil_to_bytes(hero)

    client = genai.Client()

    print(f"Generating {RESOLUTION} 360 loop with {MODEL}...")
    print(f"  Frame: {frame_path}")
    print(f"  Output: {out}")

    op = client.models.generate_videos(
        model=MODEL,
        prompt=PROMPT,
        image=types.Image(image_bytes=hero_bytes, mime_type="image/png"),
        config=types.GenerateVideosConfig(
            aspect_ratio=ASPECT_RATIO,
            resolution=RESOLUTION,
            last_frame=types.Image(image_bytes=hero_bytes, mime_type="image/png"),
        ),
    )
    while not op.done:
        print("  Waiting...")
        time.sleep(POLL_INTERVAL)
        op = client.operations.get(op)

    if not op.response or not op.response.generated_videos:
        print("ERROR:", op.response)
        sys.exit(1)

    vid = op.response.generated_videos[0]
    client.files.download(file=vid.video)
    vid.video.save(str(out))
    print(f"Done! Saved to {out}")


if __name__ == "__main__":
    main()
