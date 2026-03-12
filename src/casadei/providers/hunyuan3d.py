"""Hunyuan3D 2.1 — image-to-3D mesh generation provider.

Generates a textured 3D mesh (GLB) from a single product image using
Tencent's Hunyuan3D 2.1 model (shape: 3.3B params, paint: 2B params).

Two-stage pipeline:
1. Shape generation — DiT flow-matching → untextured trimesh mesh
2. PBR texture painting — multi-view texture synthesis with PBR maps

Requires ~10 GB VRAM for shape, ~21 GB for texture. Thor (128 GB) handles both easily.

Includes workarounds for Jetson Thor (Blackwell SM 11.0):
- Triton cuBLAS bypass (triton_linear_patch)
- bpy stub (no aarch64 wheel — OBJ→GLB via trimesh instead)
- realesrgan/basicsr stubs (no aarch64 wheels — PIL Lanczos 4x upscale instead)
"""

from __future__ import annotations

import logging
import os
import sys
import types as _types
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

MODEL_ID = "tencent/Hunyuan3D-2.1"


def _install_stubs() -> None:
    """Install module stubs for packages that have no aarch64 wheels."""

    # Stub bpy (Blender Python — no aarch64 wheel, only used for OBJ→GLB)
    if "bpy" not in sys.modules:
        _bpy = _types.ModuleType("bpy")
        _bpy.ops = type("ops", (), {"wm": None, "export_scene": None, "object": None})()
        _bpy.context = None
        _bpy.data = None
        sys.modules["bpy"] = _bpy

    # Stub realesrgan/basicsr (won't build on aarch64) — replace with PIL Lanczos 4x
    from PIL import Image as _PILImage

    class _FakeImageSuperNet:
        def __init__(self, config):
            pass
        def __call__(self, image):
            w, h = image.size
            return image.resize((w * 4, h * 4), _PILImage.LANCZOS)

    _isu = _types.ModuleType("utils.image_super_utils")
    _isu.imageSuperNet = _FakeImageSuperNet
    sys.modules["utils.image_super_utils"] = _isu

    for _mod in [
        "basicsr", "basicsr.utils", "basicsr.utils.registry",
        "basicsr.archs", "basicsr.archs.rrdbnet_arch",
        "basicsr.data", "basicsr.data.degradations",
        "realesrgan", "realesrgan.archs", "realesrgan.data",
        "realesrgan.data.realesrgan_dataset",
        "realesrgan.data.realesrgan_paired_dataset",
    ]:
        if _mod not in sys.modules:
            sys.modules[_mod] = _types.ModuleType(_mod)


class Hunyuan3DProvider:
    """Image → textured GLB mesh via Hunyuan3D 2.1."""

    def __init__(self, hunyuan3d_repo: str | Path | None = None) -> None:
        self._shape_pipeline = None
        self._paint_pipeline = None
        # Path to cloned Hunyuan3D-2.1 repo (needed for imports + config files)
        self._repo_path = Path(hunyuan3d_repo) if hunyuan3d_repo else None

    def load_model(self) -> None:
        """Load shape and paint pipelines. Downloads weights on first run."""
        # Apply Triton cuBLAS workaround for Blackwell GPUs
        from casadei.providers.triton_linear_patch import patch_linear
        patch_linear()

        # Install stubs for missing aarch64 packages
        _install_stubs()

        if self._repo_path:
            # Add repo subdirectories to sys.path for Hunyuan3D imports
            for subdir in ["hy3dshape", "hy3dpaint", "."]:
                p = str(self._repo_path / subdir) if subdir != "." else str(self._repo_path)
                if p not in sys.path:
                    sys.path.insert(0, p)

        try:
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        except ImportError:
            raise ImportError(
                "Hunyuan3D-2.1 is required. Clone the repo and set "
                "HUNYUAN3D_REPO_PATH env var: "
                "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1"
            )

        logger.info("Loading Hunyuan3D 2.1 shape pipeline...")
        self._shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            MODEL_ID,
            subfolder="hunyuan3d-dit-v2-1",
        )
        logger.info("Hunyuan3D shape pipeline loaded.")

        try:
            from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig

            paint_config = Hunyuan3DPaintConfig(
                max_num_view=6,
                resolution=512,
            )
            self._paint_pipeline = Hunyuan3DPaintPipeline(paint_config)
            logger.info("Hunyuan3D paint pipeline loaded.")
        except Exception as e:
            logger.warning(
                "Hunyuan3D paint pipeline failed to load (will generate "
                "untextured meshes only): %s", e
            )
            self._paint_pipeline = None

    def unload_model(self) -> None:
        self._shape_pipeline = None
        self._paint_pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Hunyuan3D unloaded.")

    def generate(
        self,
        image_path: str | Path,
        output_dir: str | Path,
        *,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        octree_resolution: int = 256,
        seed: int | None = None,
        texture: bool = True,
    ) -> Path:
        """Generate a textured GLB mesh from a single image.

        Args:
            image_path: Path to the input image (PNG/JPG).
            output_dir: Directory to save output files.
            num_inference_steps: Diffusion steps for shape (5-50).
            guidance_scale: Image adherence strength (1.0-15.0).
            octree_resolution: Mesh detail level (128-512).
            seed: Random seed for reproducibility.
            texture: Whether to apply PBR textures (requires paint pipeline).

        Returns:
            Path to the generated GLB file.
        """
        from PIL import Image as PILImage

        if self._shape_pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load and prepare image
        image = PILImage.open(image_path)
        if image.mode == "RGBA":
            # Composite onto white background
            rgb = PILImage.new("RGB", image.size, (255, 255, 255))
            rgb.paste(image, mask=image.split()[3])
            image = rgb
        elif image.mode != "RGB":
            image = image.convert("RGB")

        # Stage 1: Generate untextured mesh
        logger.info("Generating 3D shape (steps=%d, octree=%d)...",
                     num_inference_steps, octree_resolution)

        kwargs = {
            "image": image,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "octree_resolution": octree_resolution,
        }
        if seed is not None:
            kwargs["seed"] = seed

        with torch.no_grad():
            mesh = self._shape_pipeline(**kwargs)[0]

        shape_path = output_dir / "shape.glb"
        mesh.export(str(shape_path))
        logger.info("Shape mesh saved to %s (%d KB)",
                     shape_path, shape_path.stat().st_size // 1024)

        # Stage 2: Apply PBR textures
        if texture and self._paint_pipeline is not None:
            logger.info("Applying PBR textures...")
            textured_obj = output_dir / "textured.obj"

            # Paint pipeline reads config files relative to CWD —
            # must run from repo root
            orig_cwd = os.getcwd()
            if self._repo_path:
                os.chdir(self._repo_path)

            try:
                self._paint_pipeline(
                    mesh_path=str(shape_path),
                    image_path=image,
                    output_mesh_path=str(textured_obj),
                    save_glb=True,
                )
            finally:
                os.chdir(orig_cwd)

            # Convert OBJ → GLB using trimesh
            textured_glb = output_dir / "model_3d.glb"
            import trimesh
            scene = trimesh.load(str(textured_obj))
            scene.export(str(textured_glb))

            logger.info("Textured GLB saved to %s (%d KB)",
                         textured_glb, textured_glb.stat().st_size // 1024)
            return textured_glb
        else:
            # No texture — rename shape as final output
            final_path = output_dir / "model_3d.glb"
            if shape_path != final_path:
                shape_path.rename(final_path)
            logger.info("Untextured GLB saved to %s", final_path)
            return final_path
