"""Test SV3D: generate multi-view orbital images from a single shoe image.

SV3D (Stable Video 3D) generates 21 frames at known camera poses around an object.
These can then be fed into a multi-view → 3D reconstruction pipeline.
"""

import sys
import time
import math
from pathlib import Path

# Triton cuBLAS workaround for Blackwell
sys.path.insert(0, str(Path(__file__).parent / "src"))
from casadei.providers.triton_linear_patch import patch_linear
patch_linear()

import torch
from PIL import Image

# Paths
SV3D_MODEL = "/home/innovina/Documents/casadei/models/models--chenguolin--sv3d-diffusers/snapshots/cf4c88acda116901e1e21aa9afa61976a21363f8"
INPUT_IMAGE = Path("/home/innovina/Documents/casadei/data/results/f6a9dadf901b/ad2b80eca24f/front.png")
OUTPUT_DIR = Path("/home/innovina/Documents/casadei/data/results/f6a9dadf901b/ad2b80eca24f/_sv3d_views")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Input:  {INPUT_IMAGE}")
print(f"Output: {OUTPUT_DIR}")
print(flush=True)

# Load pipeline components manually
print("Loading SV3D pipeline...", flush=True)
t0 = time.time()

from diffusers import AutoencoderKL, EulerDiscreteScheduler
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from diffusers_sv3d.models.unets.unet_spatio_temporal_condition import SV3DUNetSpatioTemporalConditionModel
from diffusers_sv3d.pipelines import StableVideo3DDiffusionPipeline

# Load each component separately
feature_extractor = CLIPImageProcessor.from_pretrained(SV3D_MODEL, subfolder="feature_extractor")
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    SV3D_MODEL, subfolder="image_encoder", torch_dtype=torch.float16
)
unet = SV3DUNetSpatioTemporalConditionModel.from_pretrained(
    SV3D_MODEL, subfolder="unet", torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained(
    SV3D_MODEL, subfolder="vae", torch_dtype=torch.float16
)
scheduler = EulerDiscreteScheduler.from_pretrained(SV3D_MODEL, subfolder="scheduler")

pipeline = StableVideo3DDiffusionPipeline(
    vae=vae,
    image_encoder=image_encoder,
    unet=unet,
    scheduler=scheduler,
    feature_extractor=feature_extractor,
)
pipeline.to("cuda")
print(f"SV3D pipeline loaded in {time.time() - t0:.1f}s", flush=True)

# Prepare input image
image = Image.open(INPUT_IMAGE).convert("RGB")
# SV3D expects square images, resize to 576x576 (model's native resolution)
image = image.resize((576, 576), Image.LANCZOS)

# Camera poses: 21 orbital views
num_frames = 21
polar_rad = [math.radians(10.0)] * num_frames  # slight elevation
azimuth_rad = [math.radians(i * 360.0 / num_frames) for i in range(num_frames)]

print(f"Generating {num_frames} orbital views...", flush=True)
print(f"  Elevation: 10°, Azimuth: 0°-360° (step {360/num_frames:.1f}°)", flush=True)
t1 = time.time()

with torch.no_grad():
    output = pipeline(
        image=image,
        polars_rad=polar_rad,
        azimuths_rad=azimuth_rad,
        height=576,
        width=576,
        num_frames=num_frames,
        num_inference_steps=25,
        decode_chunk_size=5,
    )

frames = output.frames[0]  # List of PIL images
print(f"Generated {len(frames)} views in {time.time() - t1:.1f}s", flush=True)

# Save individual frames
for i, frame in enumerate(frames):
    azim = i * 360.0 / num_frames
    frame_path = OUTPUT_DIR / f"view_{i:02d}_az{azim:05.1f}.png"
    frame.save(str(frame_path))

# Also save as a contact sheet for quick inspection
cols = 7
rows = math.ceil(num_frames / cols)
thumb_size = 256
sheet = Image.new("RGB", (cols * thumb_size, rows * thumb_size), (0, 0, 0))
for i, frame in enumerate(frames):
    r, c = divmod(i, cols)
    thumb = frame.resize((thumb_size, thumb_size), Image.LANCZOS)
    sheet.paste(thumb, (c * thumb_size, r * thumb_size))
sheet.save(str(OUTPUT_DIR / "contact_sheet.png"))

# Save camera poses for reconstruction
import json
poses = []
for i in range(num_frames):
    poses.append({
        "frame": i,
        "azimuth_deg": i * 360.0 / num_frames,
        "elevation_deg": 10.0,
        "azimuth_rad": azimuth_rad[i],
        "elevation_rad": polar_rad[i],
    })
with open(OUTPUT_DIR / "camera_poses.json", "w") as f:
    json.dump(poses, f, indent=2)

print(flush=True)
print(f"TOTAL TIME: {time.time() - t0:.1f}s", flush=True)
print(f"Output files:", flush=True)
for f in sorted(OUTPUT_DIR.iterdir()):
    if f.is_file():
        print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)", flush=True)
