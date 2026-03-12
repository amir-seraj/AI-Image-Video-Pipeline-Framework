"""Test Stage 2 only: apply PBR textures to existing shape.glb."""

import sys
import time
import types as _types
from pathlib import Path

# Triton patch
sys.path.insert(0, str(Path(__file__).parent / "src"))
from casadei.providers.triton_linear_patch import patch_linear
patch_linear()

import torch

# Hunyuan3D paths
HUNYUAN_REPO = Path("/home/innovina/Documents/Hunyuan3D-2.1")
sys.path.insert(0, str(HUNYUAN_REPO / "hy3dshape"))
sys.path.insert(0, str(HUNYUAN_REPO / "hy3dpaint"))
sys.path.insert(0, str(HUNYUAN_REPO))

# Stub bpy
_bpy = _types.ModuleType("bpy")
_bpy.ops = type("ops", (), {"wm": None, "export_scene": None, "object": None})()
_bpy.context = None
_bpy.data = None
sys.modules["bpy"] = _bpy

# Stub realesrgan/basicsr (won't build on aarch64) — replace with simple PIL upscale
import numpy as np
from PIL import Image as _PILImage

class _FakeImageSuperNet:
    def __init__(self, config):
        pass
    def __call__(self, image):
        w, h = image.size
        return image.resize((w * 4, h * 4), _PILImage.LANCZOS)

# Patch before textureGenPipeline imports it
_isu = _types.ModuleType("utils.image_super_utils")
_isu.imageSuperNet = _FakeImageSuperNet
sys.modules["utils.image_super_utils"] = _isu

# Stub basicsr/realesrgan modules to prevent import errors
for _mod in ["basicsr", "basicsr.utils", "basicsr.utils.registry",
             "basicsr.archs", "basicsr.archs.rrdbnet_arch",
             "basicsr.data", "basicsr.data.degradations",
             "realesrgan", "realesrgan.archs", "realesrgan.data",
             "realesrgan.data.realesrgan_dataset",
             "realesrgan.data.realesrgan_paired_dataset"]:
    if _mod not in sys.modules:
        sys.modules[_mod] = _types.ModuleType(_mod)

INPUT_IMAGE = Path("/home/innovina/Documents/casadei/data/results/f6a9dadf901b/ad2b80eca24f/front.png")
OUTPUT_DIR = Path("/home/innovina/Documents/casadei/data/results/f6a9dadf901b/ad2b80eca24f/_3d_hq")
SHAPE_GLB = OUTPUT_DIR / "shape.glb"

from PIL import Image
image = Image.open(INPUT_IMAGE).convert("RGB")

print("Loading paint pipeline...")
t0 = time.time()

from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

paint_config = Hunyuan3DPaintConfig(max_num_view=10, resolution=1024)
paint_pipeline = Hunyuan3DPaintPipeline(paint_config)
print(f"Paint pipeline loaded in {time.time() - t0:.1f}s")

print("Applying PBR textures to shape.glb...")
t1 = time.time()

textured_obj = OUTPUT_DIR / "textured.obj"
paint_pipeline(
    mesh_path=str(SHAPE_GLB),
    image_path=image,
    output_mesh_path=str(textured_obj),
    save_glb=True,
)

print(f"Textures applied in {time.time() - t1:.1f}s")

# Convert OBJ → GLB using trimesh
final_glb = OUTPUT_DIR / "model_3d.glb"
import trimesh
scene = trimesh.load(str(textured_obj))
scene.export(str(final_glb))
print(f"Textured GLB saved to {final_glb}")
print(f"  Size: {final_glb.stat().st_size / 1024:.0f} KB")

print()
print(f"TOTAL PAINT TIME: {time.time() - t0:.1f}s")
print("Output files:")
for f in sorted(OUTPUT_DIR.iterdir()):
    if f.is_file():
        print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")
