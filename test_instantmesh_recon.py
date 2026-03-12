"""Test InstantMesh: reconstruct a textured 3D mesh from SV3D multi-view images.

Pipeline: SV3D orbital views → select 6 views → InstantMesh FlexiCubes → OBJ + GLB
"""

import sys
import time
import os
from pathlib import Path

# Triton cuBLAS workaround for Blackwell
sys.path.insert(0, str(Path(__file__).parent / "src"))
from casadei.providers.triton_linear_patch import patch_linear
patch_linear()

# Add InstantMesh to path
INSTANTMESH_DIR = Path("/home/innovina/Documents/InstantMesh")
sys.path.insert(0, str(INSTANTMESH_DIR))

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import (
    get_zero123plus_input_cameras,
    spherical_camera_pose,
    FOV_to_intrinsics,
)
from src.utils.mesh_util import save_obj, save_obj_with_mtl

# ── Paths ──────────────────────────────────────────────────────────────────────
SV3D_VIEWS_DIR = Path("/home/innovina/Documents/casadei/data/results/f6a9dadf901b/ad2b80eca24f/_sv3d_views")
OUTPUT_DIR = Path("/home/innovina/Documents/casadei/data/results/f6a9dadf901b/ad2b80eca24f/_3d_instantmesh")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = INSTANTMESH_DIR / "configs" / "instant-mesh-large.yaml"

# ── Select 6 views from SV3D's 21 orbital images ──────────────────────────────
# InstantMesh was trained with Zero123++ camera layout:
#   azimuths: [30, 90, 150, 210, 270, 330]  elevations: [20, -10, 20, -10, 20, -10]
# SV3D views are at 10° elevation, azimuths at i * 360/21 ≈ i * 17.14°
# Select views closest to the Zero123++ azimuths:
#   view 2: 34.3° ≈ 30°,  view 5: 85.7° ≈ 90°,  view 9: 154.3° ≈ 150°
#   view 12: 205.7° ≈ 210°, view 16: 274.3° ≈ 270°, view 19: 325.7° ≈ 330°
SELECTED_VIEWS = [2, 5, 9, 12, 16, 19]

# Zero123++ expected azimuths and elevations (what the model was trained on)
Z123_AZIMUTHS = np.array([30, 90, 150, 210, 270, 330], dtype=np.float64)
Z123_ELEVATIONS = np.array([20, -10, 20, -10, 20, -10], dtype=np.float64)

print(f"SV3D views dir: {SV3D_VIEWS_DIR}")
print(f"Output dir:     {OUTPUT_DIR}")
print(f"Selected views: {SELECTED_VIEWS}")
print(flush=True)

# ── Load and preprocess images ─────────────────────────────────────────────────
print("Loading SV3D views...", flush=True)

# Get all view files sorted
view_files = sorted(SV3D_VIEWS_DIR.glob("view_*.png"))
print(f"Found {len(view_files)} SV3D views")

images = []
for idx in SELECTED_VIEWS:
    img_path = view_files[idx]
    img = Image.open(img_path).convert("RGB")
    # InstantMesh expects 320x320 images
    img = img.resize((320, 320), Image.LANCZOS)
    img_np = np.asarray(img, dtype=np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous().float()
    images.append(img_tensor)
    print(f"  View {idx}: {img_path.name} → {img_tensor.shape}")

# Stack: [6, 3, 320, 320] → [1, 6, 3, 320, 320]
images = torch.stack(images, dim=0).unsqueeze(0)
print(f"Images tensor: {images.shape}", flush=True)

# ── Load model ─────────────────────────────────────────────────────────────────
print("\nLoading InstantMesh model...", flush=True)
t0 = time.time()

config = OmegaConf.load(str(CONFIG_PATH))
config_name = "instant-mesh-large"
model_config = config.model_config
infer_config = config.infer_config

device = torch.device("cuda")

# Load reconstruction model
model = instantiate_from_config(model_config)

# Download or use local checkpoint
if os.path.exists(infer_config.model_path):
    model_ckpt_path = infer_config.model_path
else:
    print("Downloading model weights from HuggingFace...", flush=True)
    model_ckpt_path = hf_hub_download(
        repo_id="TencentARC/InstantMesh",
        filename="instant_mesh_large.ckpt",
        repo_type="model",
    )

print(f"Loading checkpoint: {model_ckpt_path}", flush=True)
state_dict = torch.load(model_ckpt_path, map_location="cpu")["state_dict"]
state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith("lrm_generator.")}
model.load_state_dict(state_dict, strict=True)

model = model.to(device)
model.init_flexicubes_geometry(device, fovy=30.0)
model = model.eval()

print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

# ── Prepare cameras ───────────────────────────────────────────────────────────
# Use the standard Zero123++ camera matrices (what the model was trained on)
input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(device)
print(f"Camera tensor: {input_cameras.shape}", flush=True)  # [1, 6, 16]

# ── Reconstruct ───────────────────────────────────────────────────────────────
print("\nRunning reconstruction...", flush=True)
t1 = time.time()

images = images.to(device)

with torch.no_grad():
    # Step 1: Get triplane features
    print("  Computing triplane features...", flush=True)
    planes = model.forward_planes(images, input_cameras)
    print(f"  Triplane shape: {planes.shape}")

    # Step 2: Extract mesh with texture map
    print("  Extracting textured mesh...", flush=True)
    mesh_out = model.extract_mesh(
        planes,
        use_texture_map=True,
        **infer_config,
    )
    vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out

print(f"Reconstruction done in {time.time() - t1:.1f}s", flush=True)
print(f"  Vertices: {vertices.shape[0]}")
print(f"  Faces: {faces.shape[0]}")
print(f"  Texture map: {tex_map.shape}")

# ── Save OBJ with texture ─────────────────────────────────────────────────────
obj_path = str(OUTPUT_DIR / "shoe.obj")
save_obj_with_mtl(
    vertices.data.cpu().numpy(),
    uvs.data.cpu().numpy(),
    faces.data.cpu().numpy(),
    mesh_tex_idx.data.cpu().numpy(),
    tex_map.permute(1, 2, 0).data.cpu().numpy(),
    obj_path,
)
print(f"\nOBJ saved to {obj_path}")

# ── Also save as GLB via trimesh ──────────────────────────────────────────────
import trimesh

# Load the OBJ we just saved (includes texture)
mesh = trimesh.load(obj_path, process=False)
glb_path = str(OUTPUT_DIR / "shoe.glb")
mesh.export(glb_path, file_type="glb")
print(f"GLB saved to {glb_path}")

# ── Also save vertex-colored version (no texture map) ─────────────────────────
with torch.no_grad():
    mesh_out_vc = model.extract_mesh(
        planes,
        use_texture_map=False,
        **infer_config,
    )
    verts_vc, faces_vc, colors_vc = mesh_out_vc

obj_vc_path = str(OUTPUT_DIR / "shoe_vertex_colors.obj")
save_obj(verts_vc, faces_vc, colors_vc, obj_vc_path)
print(f"Vertex-colored OBJ saved to {obj_vc_path}")

# Save vertex-colored GLB too
from src.utils.mesh_util import save_glb
glb_vc_path = str(OUTPUT_DIR / "shoe_vertex_colors.glb")
save_glb(verts_vc, faces_vc, colors_vc, glb_vc_path)
print(f"Vertex-colored GLB saved to {glb_vc_path}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nTOTAL TIME: {time.time() - t0:.1f}s", flush=True)
print("Output files:", flush=True)
for f in sorted(OUTPUT_DIR.iterdir()):
    if f.is_file():
        size_kb = f.stat().st_size / 1024
        unit = "KB" if size_kb < 1024 else "MB"
        size_val = size_kb if size_kb < 1024 else size_kb / 1024
        print(f"  {f.name} ({size_val:.1f} {unit})", flush=True)
