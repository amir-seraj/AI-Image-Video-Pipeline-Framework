"""Generate 4K 8s 360 loop using first+last frame interpolation on Veo 3.1 Fast."""

from google import genai
from google.genai import types
from PIL import Image
from pathlib import Path
import time, io

client = genai.Client()
img_dir = Path(__file__).resolve().parent.parent.parent / "casadei-front/mvp/Shoe 2/generated/Varitation 1"

def pil_to_bytes(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

hero = Image.open(img_dir / "hero-front-right.png")
hero_bytes = pil_to_bytes(hero)

prompt = (
    "Fast-spinning product turntable video. A blue patent leather Casadei platform high-heel shoe "
    "with peep toe spins rapidly on a motorized turntable. "
    "The turntable completes exactly one full 360-degree revolution at constant speed. "
    "The rotation is quick, steady, and never stops or reverses. "
    "Fixed camera, clean gray studio background, soft product lighting. "
    "Professional e-commerce turntable spin, photorealistic."
)

print("Generating 4K 8s with first+last frame on veo-3.1-fast-generate-preview...")
op = client.models.generate_videos(
    model="veo-3.1-fast-generate-preview",
    prompt=prompt,
    image=types.Image(image_bytes=hero_bytes, mime_type="image/png"),
    config=types.GenerateVideosConfig(
        aspect_ratio="16:9",
        resolution="4k",
        last_frame=types.Image(image_bytes=hero_bytes, mime_type="image/png"),
    ),
)
while not op.done:
    print("  Waiting...")
    time.sleep(10)
    op = client.operations.get(op)

if not op.response or not op.response.generated_videos:
    print("ERROR:", op.response)
    exit(1)

vid = op.response.generated_videos[0]
client.files.download(file=vid.video)
out = Path(__file__).resolve().parent.parent / "data/results/shoe2_360_4k_loop.mp4"
out.parent.mkdir(parents=True, exist_ok=True)
vid.video.save(str(out))
print(f"Done! Saved to {out}")
