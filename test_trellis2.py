"""Test TRELLIS.2: single image → textured 3D mesh (GLB) + Gaussian splat."""

import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Use SDPA for both dense and sparse attention (no flash-attn needed)
os.environ["ATTN_BACKEND"] = "sdpa"
os.environ["SPARSE_ATTN_BACKEND"] = "sdpa"

import sys
import time
from pathlib import Path

# Apply Triton cuBLAS workaround for Blackwell
sys.path.insert(0, str(Path(__file__).parent / "src"))
from casadei.providers.triton_linear_patch import patch_linear
patch_linear()

import torch

# Add TRELLIS.2 repo to path
TRELLIS_REPO = Path("/home/innovina/Documents/TRELLIS.2")
sys.path.insert(0, str(TRELLIS_REPO))

INPUT_IMAGE = Path("/home/innovina/Documents/casadei/data/results/f6a9dadf901b/ad2b80eca24f/front.png")
OUTPUT_DIR = Path("/home/innovina/Documents/casadei/data/results/f6a9dadf901b/ad2b80eca24f/_3d_trellis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

from PIL import Image
import imageio

print(f"Input:  {INPUT_IMAGE}")
print(f"Output: {OUTPUT_DIR}")
print()

# 1. Load Pipeline
print("Loading TRELLIS.2 pipeline...")
t0 = time.time()

from trellis2.pipelines import Trellis2ImageTo3DPipeline
pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
pipeline.cuda()
print(f"Pipeline loaded in {time.time() - t0:.1f}s")

# 2. Run inference
print("Generating 3D from image...")
t1 = time.time()
image = Image.open(INPUT_IMAGE).convert("RGB")
mesh = pipeline.run(image)[0]
print(f"3D generation done in {time.time() - t1:.1f}s")

# 3. Simplify mesh for nvdiffrast
mesh.simplify(16777216)

# 4. Export GLB
print("Exporting GLB...")
import o_voxel
glb = o_voxel.postprocess.to_glb(
    vertices=mesh.vertices,
    faces=mesh.faces,
    attr_volume=mesh.attrs,
    coords=mesh.coords,
    attr_layout=mesh.layout,
    voxel_size=mesh.voxel_size,
    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    decimation_target=1000000,
    texture_size=4096,
    remesh=True,
    remesh_band=1,
    remesh_project=0,
    verbose=True
)

glb_path = OUTPUT_DIR / "model_3d.glb"
glb.export(str(glb_path), extension_webp=True)
print(f"GLB saved to {glb_path} ({glb_path.stat().st_size / 1024:.0f} KB)")

# 5. Render preview video
print("Rendering preview video...")
try:
    import cv2
    from trellis2.utils import render_utils
    from trellis2.renderers import EnvMap

    envmap_path = TRELLIS_REPO / "assets" / "hdri" / "forest.exr"
    envmap = EnvMap(torch.tensor(
        cv2.cvtColor(cv2.imread(str(envmap_path), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB),
        dtype=torch.float32, device='cuda'
    ))
    video = render_utils.make_pbr_vis_frames(render_utils.render_video(mesh, envmap=envmap))
    video_path = OUTPUT_DIR / "preview.mp4"
    imageio.mimsave(str(video_path), video, fps=15)
    print(f"Preview video saved to {video_path}")
except Exception as e:
    print(f"Video render failed: {e}")

print()
print(f"TOTAL TIME: {time.time() - t0:.1f}s")
print("Output files:")
for f in sorted(OUTPUT_DIR.iterdir()):
    if f.is_file():
        print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")
