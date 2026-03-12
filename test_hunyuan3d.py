"""Quick test: generate a 3D model from a shoe variation image using Hunyuan3D 2.1.

Includes Blackwell (SM 110) CUBLAS workaround: all matmul ops replaced with Triton kernels.
"""

import sys
import os
import time
import types
from pathlib import Path

# Apply Triton cuBLAS workaround BEFORE importing torch-dependent code
sys.path.insert(0, str(Path(__file__).parent / "src"))
from casadei.providers.triton_linear_patch import patch_linear
patch_linear()

import torch

# Add Hunyuan3D repo to path
HUNYUAN_REPO = Path("/home/innovina/Documents/Hunyuan3D-2.1")
sys.path.insert(0, str(HUNYUAN_REPO / "hy3dshape"))
sys.path.insert(0, str(HUNYUAN_REPO / "hy3dpaint"))
sys.path.insert(0, str(HUNYUAN_REPO))

# Test image: Hollywood Glitter Blade Sandal — plastic / Royal Blue gradient
INPUT_IMAGE = Path("/home/innovina/Documents/casadei/data/results/f6a9dadf901b/ad2b80eca24f/front.png")
OUTPUT_DIR = Path("/home/innovina/Documents/casadei/data/results/f6a9dadf901b/ad2b80eca24f/_3d_test")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Input:  {INPUT_IMAGE}")
print(f"Output: {OUTPUT_DIR}")
print()


# --- Stage 1: Shape generation ---
print("=" * 60)
print("STAGE 1: Loading shape pipeline...")
t0 = time.time()

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
from PIL import Image

shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    "tencent/Hunyuan3D-2.1",
    subfolder="hunyuan3d-dit-v2-1",
)

print(f"Shape pipeline loaded in {time.time() - t0:.1f}s")

image = Image.open(INPUT_IMAGE).convert("RGB")

print("Generating 3D shape (steps=30, octree=256)...")
t1 = time.time()

with torch.no_grad():
    mesh = shape_pipeline(
        image=image,
        num_inference_steps=30,
        guidance_scale=7.5,
        octree_resolution=256,
    )[0]

print(f"Shape generated in {time.time() - t1:.1f}s")
print(f"  Vertices: {len(mesh.vertices)}")
print(f"  Faces: {len(mesh.faces)}")

shape_path = OUTPUT_DIR / "shape.glb"
mesh.export(str(shape_path))
print(f"Untextured mesh saved to {shape_path}")
print(f"  Size: {shape_path.stat().st_size / 1024:.0f} KB")

# --- Stage 2: Texture painting ---
print()
print("=" * 60)
print("STAGE 2: Loading paint pipeline...")
t2 = time.time()

try:
    # Stub bpy before importing paint pipeline (bpy has no aarch64 wheel)
    import types as _types
    _bpy_mock = _types.ModuleType("bpy")
    _bpy_mock.ops = type("ops", (), {"wm": None, "export_scene": None, "object": None})()
    _bpy_mock.context = None
    _bpy_mock.data = None
    sys.modules["bpy"] = _bpy_mock

    from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

    paint_config = Hunyuan3DPaintConfig(max_num_view=6, resolution=512)
    paint_pipeline = Hunyuan3DPaintPipeline(paint_config)
    print(f"Paint pipeline loaded in {time.time() - t2:.1f}s")

    print("Applying PBR textures...")
    t3 = time.time()

    textured_obj = OUTPUT_DIR / "textured.obj"
    paint_pipeline(
        mesh_path=str(shape_path),
        image_path=image,
        output_mesh_path=str(textured_obj),
        save_glb=True,
    )

    print(f"Textures applied in {time.time() - t3:.1f}s")

    # Convert to final GLB using trimesh (bpy not available on aarch64)
    final_glb = OUTPUT_DIR / "model_3d.glb"
    import trimesh
    scene = trimesh.load(str(textured_obj))
    scene.export(str(final_glb))

    print(f"Textured GLB saved to {final_glb}")
    print(f"  Size: {final_glb.stat().st_size / 1024:.0f} KB")

except Exception as e:
    print(f"Paint pipeline failed: {e}")
    print("You still have the untextured mesh at:", shape_path)
    import traceback
    traceback.print_exc()

# --- Summary ---
print()
print("=" * 60)
print(f"TOTAL TIME: {time.time() - t0:.1f}s")
print(f"Output files:")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")
