"""Tests for the Shoe Virtual Try-On pipeline — every component end-to-end.

Covers:
  1. ReferenceInpaintModel base class
  2. SDXLIPAdapterInpaint registration, capability, params
  3. segment_shoes.py (process, _detect_shoes, _segment_with_sam2, _cleanup_mask)
  4. Pipeline YAML parsing (inputs declaration)
  5. Named context runner (_run_scratch with multi-image)
  6. API endpoint (multi-image upload, legacy single-image, validation)
  7. PipelineInputDeclaration model
  8. Generation rejection of multi-input pipelines
"""

from __future__ import annotations

import io
import json
import time
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest
import yaml
from PIL import Image as PILImage

from casadei.media import ImageMedia, TextMedia, MediaBundle
from casadei.models.base import (
    AIModel,
    ModelCapability,
    ImageConstraint,
    TextConstraint,
)
from casadei.models.reference_inpaint import ReferenceInpaintModel
from casadei.models.registry import default_registry, ModelRegistry
from casadei.api.models import PipelineInputDeclaration, PipelinePreset, PipelineDetailResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rgb_image(w: int = 100, h: int = 100, color: str = "red") -> PILImage.Image:
    return PILImage.new("RGB", (w, h), color=color)


def _gray_image(w: int = 100, h: int = 100, value: int = 255) -> PILImage.Image:
    return PILImage.new("L", (w, h), color=value)


def _save_image(path: Path, img: PILImage.Image | None = None) -> Path:
    if img is None:
        img = _rgb_image()
    img.save(path)
    return path


def _png_bytes(color: str = "red", size: tuple[int, int] = (64, 64)) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 1. ReferenceInpaintModel base class
# ---------------------------------------------------------------------------

class MockInpainter(ReferenceInpaintModel):
    """Concrete subclass that returns a green image."""

    capability = ModelCapability(
        inputs=[
            ImageConstraint(required=True, max_count=3),
            TextConstraint(required=False, max_count=2),
        ],
        outputs=[ImageConstraint(max_count=1)],
    )

    def load_model(self) -> None:
        pass

    def unload_model(self) -> None:
        pass

    def _inpaint(self, image, mask, reference, prompt, negative_prompt, **kwargs):
        self.last_call = {
            "image_size": image.size,
            "mask_size": mask.size,
            "reference_size": reference.size,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
        }
        return PILImage.new("RGB", image.size, color="green")


class TestReferenceInpaintModel:
    def setup_method(self):
        self.model = MockInpainter()

    def test_is_abstract(self):
        with pytest.raises(TypeError):
            ReferenceInpaintModel()

    def test_run_happy_path(self):
        bundle = MediaBundle(items={
            "image": ImageMedia(image=_rgb_image(200, 150)),
            "mask": ImageMedia(image=_gray_image(200, 150, 255)),
            "reference": ImageMedia(image=_rgb_image(64, 64, "blue")),
            "prompt": TextMedia(text="a shoe on a foot"),
            "negative_prompt": TextMedia(text="blurry"),
        })
        result = self.model.run(bundle)
        assert "image" in result.items
        assert isinstance(result["image"], ImageMedia)
        assert result["image"].image.size == (200, 150)
        # Green fill confirms _inpaint was called
        assert result["image"].image.getpixel((50, 50)) == (0, 128, 0)
        # Verify the arguments forwarded correctly
        assert self.model.last_call["prompt"] == "a shoe on a foot"
        assert self.model.last_call["negative_prompt"] == "blurry"
        assert self.model.last_call["reference_size"] == (64, 64)

    def test_run_without_text(self):
        bundle = MediaBundle(items={
            "image": ImageMedia(image=_rgb_image()),
            "mask": ImageMedia(image=_gray_image()),
            "reference": ImageMedia(image=_rgb_image(32, 32)),
        })
        result = self.model.run(bundle)
        assert result["image"].image.size == (100, 100)
        assert self.model.last_call["prompt"] == ""
        assert self.model.last_call["negative_prompt"] == ""

    def test_run_missing_image_raises(self):
        bundle = MediaBundle(items={
            "mask": ImageMedia(image=_gray_image()),
            "reference": ImageMedia(image=_rgb_image()),
        })
        with pytest.raises(ValueError, match="'image' input"):
            self.model.run(bundle)

    def test_run_missing_mask_raises(self):
        bundle = MediaBundle(items={
            "image": ImageMedia(image=_rgb_image()),
            "reference": ImageMedia(image=_rgb_image()),
        })
        with pytest.raises(ValueError, match="'mask' input"):
            self.model.run(bundle)

    def test_run_missing_reference_raises(self):
        bundle = MediaBundle(items={
            "image": ImageMedia(image=_rgb_image()),
            "mask": ImageMedia(image=_gray_image()),
        })
        with pytest.raises(ValueError, match="'reference' input"):
            self.model.run(bundle)


# ---------------------------------------------------------------------------
# 2. SDXLIPAdapterInpaint — registration, capability, params
# ---------------------------------------------------------------------------

class TestSDXLIPAdapterInpaint:
    def test_registered_in_default_registry(self):
        cls = default_registry.get("sdxl_ipadapter_inpaint")
        assert cls is not None
        assert cls.__name__ == "SDXLIPAdapterInpaint"

    def test_is_reference_inpaint_model(self):
        from casadei.providers.sdxl_ipadapter_inpaint import SDXLIPAdapterInpaint
        assert issubclass(SDXLIPAdapterInpaint, ReferenceInpaintModel)

    def test_default_params(self):
        from casadei.providers.sdxl_ipadapter_inpaint import SDXLIPAdapterInpaint
        params = SDXLIPAdapterInpaint.DEFAULT_PARAMS
        assert params["num_inference_steps"] == 40
        assert params["guidance_scale"] == 7.5
        assert params["ip_adapter_scale"] == 0.95
        assert params["strength"] == 0.99
        assert params["crop_padding"] == 0.4

    def test_capability_allows_three_images(self):
        from casadei.providers.sdxl_ipadapter_inpaint import SDXLIPAdapterInpaint
        cap = SDXLIPAdapterInpaint.capability
        image_constraints = [c for c in cap.inputs if isinstance(c, ImageConstraint)]
        assert len(image_constraints) == 1
        assert image_constraints[0].max_count == 3

    def test_capability_text_optional(self):
        from casadei.providers.sdxl_ipadapter_inpaint import SDXLIPAdapterInpaint
        cap = SDXLIPAdapterInpaint.capability
        text_constraints = [c for c in cap.inputs if isinstance(c, TextConstraint)]
        assert len(text_constraints) == 1
        assert text_constraints[0].required is False

    def test_inpaint_without_load_raises(self):
        from casadei.providers.sdxl_ipadapter_inpaint import SDXLIPAdapterInpaint
        model = SDXLIPAdapterInpaint()
        with pytest.raises(RuntimeError, match="not loaded"):
            model._inpaint(
                _rgb_image(), _gray_image(), _rgb_image(),
                "prompt", "neg",
            )

    def test_exported_in_package(self):
        from casadei import ReferenceInpaintModel as RIM
        assert RIM is ReferenceInpaintModel


# ---------------------------------------------------------------------------
# 3. segment_shoes.py — process function and helpers
# ---------------------------------------------------------------------------

class TestSegmentShoes:
    """Tests for the segmentation script with mocked transformer models."""

    def test_process_returns_mask(self):
        """process() should return a dict with 'mask' ImageMedia."""
        mask_array = np.zeros((100, 100), dtype=bool)
        mask_array[60:100, 20:80] = True  # shoe region at bottom
        fake_mask = PILImage.fromarray(
            (mask_array * 255).astype(np.uint8), mode="L"
        )

        script_path = (
            Path(__file__).parent.parent
            / "workflows"
            / "shoe_tryon"
            / "scripts"
            / "segment_shoes.py"
        )
        spec = importlib.util.spec_from_file_location(
            "segment_shoes_process", script_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Patch the internal function after loading
        mod._detect_and_segment = MagicMock(return_value=fake_mask)

        context = {
            "person": ImageMedia(image=_rgb_image(100, 100)),
            "shoe": ImageMedia(image=_rgb_image(64, 64, "blue")),
        }
        result = mod.process(context)

        assert "mask" in result
        assert isinstance(result["mask"], ImageMedia)
        assert result["mask"].image.mode == "L"
        mod._detect_and_segment.assert_called_once()

    def test_process_rejects_non_image(self):
        """process() should raise if 'person' is not an ImageMedia."""
        script_path = (
            Path(__file__).parent.parent
            / "workflows"
            / "shoe_tryon"
            / "scripts"
            / "segment_shoes.py"
        )
        spec = importlib.util.spec_from_file_location("segment_shoes_val", script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        context = {"person": TextMedia(text="not an image")}
        with pytest.raises(ValueError, match="ImageMedia"):
            mod.process(context)

    def test_cleanup_mask_preserves_mode(self):
        """_cleanup_mask should return an 'L' mode image."""
        script_path = (
            Path(__file__).parent.parent
            / "workflows"
            / "shoe_tryon"
            / "scripts"
            / "segment_shoes.py"
        )
        spec = importlib.util.spec_from_file_location("segment_shoes_clean", script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        mask = _gray_image(100, 100, 255)
        result = mod._cleanup_mask(mask)
        assert result.mode == "L"
        assert result.size == (100, 100)

    def test_cleanup_mask_dilates_small_region(self):
        """_cleanup_mask should grow a small white dot via dilation."""
        script_path = (
            Path(__file__).parent.parent
            / "workflows"
            / "shoe_tryon"
            / "scripts"
            / "segment_shoes.py"
        )
        spec = importlib.util.spec_from_file_location("segment_shoes_dilate", script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Single white pixel in center of black mask
        mask = PILImage.new("L", (50, 50), 0)
        mask.putpixel((25, 25), 255)
        white_before = sum(1 for p in mask.getdata() if p > 128)

        result = mod._cleanup_mask(mask)
        white_after = sum(1 for p in result.getdata() if p > 128)
        # Dilation should have grown the region
        assert white_after > white_before

    def test_detect_and_segment_raises_on_no_boxes(self):
        """_detect_and_segment should raise RuntimeError if no shoes found."""
        script_path = (
            Path(__file__).parent.parent
            / "workflows"
            / "shoe_tryon"
            / "scripts"
            / "segment_shoes.py"
        )
        spec = importlib.util.spec_from_file_location("segment_shoes_nobox", script_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        with patch.object(mod, "_detect_shoes", return_value=[]):
            with pytest.raises(RuntimeError, match="No shoes detected"):
                mod._detect_and_segment(_rgb_image())


import importlib.util  # needed for test_process_rejects_non_image


# ---------------------------------------------------------------------------
# 4. Pipeline YAML parsing — inputs declaration
# ---------------------------------------------------------------------------

class TestPipelineYAMLParsing:
    def test_shoe_tryon_yaml_has_inputs(self):
        yaml_path = (
            Path(__file__).parent.parent
            / "workflows"
            / "shoe_tryon"
            / "pipeline.yaml"
        )
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        assert "inputs" in data
        assert "person" in data["inputs"]
        assert "shoe" in data["inputs"]
        assert data["inputs"]["person"]["type"] == "image"
        assert data["inputs"]["person"]["label"] == "Model Photo"
        assert data["inputs"]["shoe"]["type"] == "image"
        assert data["inputs"]["shoe"]["label"] == "Shoe Photo"

    def test_shoe_tryon_yaml_has_code_step(self):
        yaml_path = (
            Path(__file__).parent.parent
            / "workflows"
            / "shoe_tryon"
            / "pipeline.yaml"
        )
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        steps = data["steps"]
        assert steps[0]["type"] == "code"
        assert steps[0]["script"] == "segment_shoes.py"
        assert steps[0]["function"] == "process"

    def test_shoe_tryon_yaml_has_agent_step_with_inputs(self):
        yaml_path = (
            Path(__file__).parent.parent
            / "workflows"
            / "shoe_tryon"
            / "pipeline.yaml"
        )
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        agent_step = data["steps"][1]
        assert agent_step["type"] == "agent"
        assert agent_step["agent"] == "shoe_inpaint"
        assert agent_step["inputs"] == {
            "image": "person",
            "mask": "mask",
            "reference": "shoe",
        }

    def test_shoe_inpaint_agent_yaml(self):
        yaml_path = (
            Path(__file__).parent.parent
            / "workflows"
            / "shoe_tryon"
            / "agents"
            / "shoe_inpaint.yaml"
        )
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        assert data["name"] == "shoe_inpaint"
        assert data["model"] == "sdxl_ipadapter_inpaint"
        assert data["params"]["ip_adapter_scale"] == 0.95
        assert data["params"]["strength"] == 0.99

    def test_legacy_pipeline_has_no_inputs(self):
        yaml_path = (
            Path(__file__).parent.parent
            / "workflows"
            / "clean_and_style"
            / "pipeline.yaml"
        )
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        assert "inputs" not in data or data.get("inputs") is None

    def test_script_file_exists(self):
        script_path = (
            Path(__file__).parent.parent
            / "workflows"
            / "shoe_tryon"
            / "scripts"
            / "segment_shoes.py"
        )
        assert script_path.exists()


# ---------------------------------------------------------------------------
# 5 & 6. API tests — multi-image endpoint, named context, generation rejection
# ---------------------------------------------------------------------------

class TestAPIMultiImage:
    @pytest.fixture
    def client(self, tmp_path: Path):
        from casadei.api.app import create_app
        from starlette.testclient import TestClient

        app = create_app(data_dir=tmp_path)
        return TestClient(app)

    @pytest.fixture
    def client_with_shoe_tryon(self, tmp_path: Path):
        """Client with shoe_tryon pipeline available."""
        from casadei.api.app import create_app
        from starlette.testclient import TestClient

        # Set up workflow directories
        workflows = tmp_path / "workflows"
        shoe_dir = workflows / "shoe_tryon"
        shoe_dir.mkdir(parents=True)
        agents_dir = shoe_dir / "agents"
        agents_dir.mkdir()
        scripts_dir = shoe_dir / "scripts"
        scripts_dir.mkdir()

        # Pipeline YAML
        pipeline_yaml = {
            "id": "shoe_tryon",
            "name": "Shoe Virtual Try-On",
            "description": "Replace shoes",
            "inputs": {
                "person": {"type": "image", "label": "Model Photo"},
                "shoe": {"type": "image", "label": "Shoe Photo"},
            },
            "steps": [
                {"type": "code", "script": "segment_shoes.py", "function": "process"},
                {
                    "type": "agent",
                    "agent": "shoe_inpaint",
                    "inputs": {"image": "person", "mask": "mask", "reference": "shoe"},
                },
            ],
        }
        with open(shoe_dir / "pipeline.yaml", "w") as f:
            yaml.dump(pipeline_yaml, f)

        # Agent YAML
        agent_yaml = {
            "name": "shoe_inpaint",
            "model": "sdxl_ipadapter_inpaint",
            "description": "Inpaint shoes",
            "prompt_template": "a shoe on a foot",
            "negative_prompt": "blurry",
            "params": {"num_inference_steps": 30},
        }
        with open(agents_dir / "shoe_inpaint.yaml", "w") as f:
            yaml.dump(agent_yaml, f)

        # Minimal code step script that just creates a dummy mask
        script_content = textwrap.dedent("""\
            from casadei.media import ImageMedia, Media
            from PIL import Image as PILImage

            def process(context):
                img = context["person"].image
                mask = PILImage.new("L", img.size, 255)
                return {"mask": ImageMedia(image=mask)}
        """)
        (scripts_dir / "segment_shoes.py").write_text(script_content)

        # Also add a legacy pipeline for backward compat testing
        legacy_dir = workflows / "clean_and_style"
        legacy_dir.mkdir(parents=True)
        legacy_yaml = {
            "id": "clean_and_style",
            "name": "Clean & Style",
            "description": "Remove background then style",
            "steps": [
                {"type": "agent", "agent": "background_remover"},
                {"type": "agent", "agent": "style_transfer"},
            ],
        }
        with open(legacy_dir / "pipeline.yaml", "w") as f:
            yaml.dump(legacy_yaml, f)

        # Global agents for legacy pipeline
        global_agents = tmp_path / "agents"
        global_agents.mkdir()
        for name, model in [("background_remover", "qwen_image_edit"), ("style_transfer", "qwen_image_edit")]:
            agent = {
                "name": name,
                "model": model,
                "description": f"{name} agent",
                "prompt_template": f"$style applied",
                "negative_prompt": "",
                "params": {},
            }
            with open(global_agents / f"{name}.yaml", "w") as f:
                yaml.dump(agent, f)

        app = create_app(data_dir=tmp_path, workflows_dir=workflows, agents_dir=global_agents)
        return TestClient(app)

    # --- list_pipelines includes inputs ---

    def test_list_pipelines_includes_inputs(self, client_with_shoe_tryon):
        resp = client_with_shoe_tryon.get("/api/pipelines")
        assert resp.status_code == 200
        pipelines = resp.json()
        shoe = next((p for p in pipelines if p["id"] == "shoe_tryon"), None)
        assert shoe is not None
        assert "inputs" in shoe
        assert "person" in shoe["inputs"]
        assert shoe["inputs"]["person"]["label"] == "Model Photo"
        assert "shoe" in shoe["inputs"]
        assert shoe["inputs"]["shoe"]["label"] == "Shoe Photo"

    def test_list_pipelines_legacy_has_empty_inputs(self, client_with_shoe_tryon):
        resp = client_with_shoe_tryon.get("/api/pipelines")
        pipelines = resp.json()
        legacy = next((p for p in pipelines if p["id"] == "clean_and_style"), None)
        assert legacy is not None
        assert legacy["inputs"] == {}

    def test_get_pipeline_includes_inputs(self, client_with_shoe_tryon):
        resp = client_with_shoe_tryon.get("/api/pipelines/shoe_tryon")
        assert resp.status_code == 200
        data = resp.json()
        assert "inputs" in data
        assert data["inputs"]["person"]["type"] == "image"

    # --- Multi-image upload ---

    def test_multi_image_upload_accepted(self, client_with_shoe_tryon):
        resp = client_with_shoe_tryon.post(
            "/api/run",
            data={
                "type": "pipeline",
                "name": "shoe_tryon",
                "template_variables": "{}",
                "image_keys": json.dumps(["person", "shoe"]),
            },
            files=[
                ("images", ("person.png", _png_bytes("red"), "image/png")),
                ("images", ("shoe.png", _png_bytes("blue"), "image/png")),
            ],
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "job_id" in data
        assert "run_id" in data

    def test_multi_image_mismatched_keys_rejected(self, client_with_shoe_tryon):
        resp = client_with_shoe_tryon.post(
            "/api/run",
            data={
                "type": "pipeline",
                "name": "shoe_tryon",
                "template_variables": "{}",
                "image_keys": json.dumps(["person"]),  # only 1 key
            },
            files=[
                ("images", ("person.png", _png_bytes(), "image/png")),
                ("images", ("shoe.png", _png_bytes(), "image/png")),  # 2 files
            ],
        )
        assert resp.status_code == 400
        assert "must match" in resp.json()["detail"]

    def test_legacy_single_image_still_works(self, client_with_shoe_tryon):
        resp = client_with_shoe_tryon.post(
            "/api/run",
            data={
                "type": "agent",
                "name": "background_remover",
                "template_variables": json.dumps({"style": "clean"}),
            },
            files=[
                ("image", ("test.png", _png_bytes(), "image/png")),
            ],
        )
        assert resp.status_code == 202

    def test_no_image_rejected(self, client_with_shoe_tryon):
        resp = client_with_shoe_tryon.post(
            "/api/run",
            data={
                "type": "pipeline",
                "name": "shoe_tryon",
                "template_variables": "{}",
                "image_keys": "[]",
            },
        )
        assert resp.status_code == 400
        assert "No image" in resp.json()["detail"]

    def test_invalid_type_rejected(self, client_with_shoe_tryon):
        resp = client_with_shoe_tryon.post(
            "/api/run",
            data={
                "type": "invalid",
                "name": "shoe_tryon",
                "template_variables": "{}",
            },
            files=[("image", ("test.png", _png_bytes(), "image/png"))],
        )
        assert resp.status_code == 400

    def test_invalid_template_variables_rejected(self, client_with_shoe_tryon):
        resp = client_with_shoe_tryon.post(
            "/api/run",
            data={
                "type": "pipeline",
                "name": "shoe_tryon",
                "template_variables": "not json",
                "image_keys": json.dumps(["person", "shoe"]),
            },
            files=[
                ("images", ("person.png", _png_bytes(), "image/png")),
                ("images", ("shoe.png", _png_bytes(), "image/png")),
            ],
        )
        assert resp.status_code == 400
        assert "valid JSON" in resp.json()["detail"]

    def test_invalid_image_keys_rejected(self, client_with_shoe_tryon):
        resp = client_with_shoe_tryon.post(
            "/api/run",
            data={
                "type": "pipeline",
                "name": "shoe_tryon",
                "template_variables": "{}",
                "image_keys": "not json",
            },
            files=[
                ("images", ("person.png", _png_bytes(), "image/png")),
            ],
        )
        assert resp.status_code == 400
        assert "image_keys" in resp.json()["detail"]

    # --- Generation rejects multi-input pipelines ---

    def test_generate_rejects_multi_input_pipeline(self, client_with_shoe_tryon):
        """Product page generate should fail for shoe_tryon (multi-input)."""
        # Create a product with a sketch first
        create_resp = client_with_shoe_tryon.post(
            "/api/products", json={"name": "Test Shoe"}
        )
        product_id = create_resp.json()["id"]

        buf = io.BytesIO()
        PILImage.new("RGB", (64, 64), "red").save(buf, format="PNG")
        buf.seek(0)
        client_with_shoe_tryon.post(
            f"/api/products/{product_id}/sketches",
            files={"file": ("sketch.png", buf, "image/png")},
        )

        gen_resp = client_with_shoe_tryon.post(
            f"/api/products/{product_id}/generate",
            json={
                "pipeline": "shoe_tryon",
                "prompt": "red shoes",
            },
        )
        assert gen_resp.status_code == 202

        # Wait for background thread to fail
        job_id = gen_resp.json()["job_id"]
        time.sleep(0.5)

        job_resp = client_with_shoe_tryon.get(f"/api/jobs/{job_id}")
        job_data = job_resp.json()
        assert job_data["status"] == "failed"
        assert "Workbench" in job_data["error"]
        assert "multiple image inputs" in job_data["error"]


# ---------------------------------------------------------------------------
# 7. PipelineInputDeclaration model
# ---------------------------------------------------------------------------

class TestPipelineInputDeclaration:
    def test_basic_creation(self):
        decl = PipelineInputDeclaration(type="image", label="Model Photo")
        assert decl.type == "image"
        assert decl.label == "Model Photo"

    def test_pipeline_preset_with_inputs(self):
        preset = PipelinePreset(
            id="shoe_tryon",
            name="Shoe Virtual Try-On",
            description="Replace shoes",
            agents=["shoe_inpaint"],
            template_variables=[],
            inputs={
                "person": PipelineInputDeclaration(type="image", label="Model Photo"),
                "shoe": PipelineInputDeclaration(type="image", label="Shoe Photo"),
            },
        )
        assert len(preset.inputs) == 2
        assert preset.inputs["person"].label == "Model Photo"

    def test_pipeline_preset_default_empty_inputs(self):
        preset = PipelinePreset(
            id="legacy",
            name="Legacy",
            description="",
            agents=[],
            template_variables=[],
        )
        assert preset.inputs == {}

    def test_pipeline_detail_response_with_inputs(self):
        detail = PipelineDetailResponse(
            id="shoe_tryon",
            name="Shoe Virtual Try-On",
            description="Replace shoes",
            steps=[],
            local_agents=["shoe_inpaint"],
            template_variables=[],
            inputs={
                "person": PipelineInputDeclaration(type="image", label="Model Photo"),
            },
        )
        assert "person" in detail.inputs

    def test_serialization_roundtrip(self):
        preset = PipelinePreset(
            id="test",
            name="Test",
            description="",
            agents=[],
            template_variables=[],
            inputs={
                "a": PipelineInputDeclaration(type="image", label="A"),
            },
        )
        data = json.loads(preset.model_dump_json())
        assert data["inputs"]["a"]["type"] == "image"
        assert data["inputs"]["a"]["label"] == "A"


# ---------------------------------------------------------------------------
# 8. Named context runner — _run_scratch behavior
# ---------------------------------------------------------------------------

class TestNamedContextRunner:
    """Tests for _run_scratch named context logic via the API endpoint.

    These test the full flow including the background thread.
    """

    @pytest.fixture
    def client_code_pipeline(self, tmp_path: Path):
        """Client with a simple code-step-only pipeline for testing context flow."""
        from casadei.api.app import create_app
        from starlette.testclient import TestClient

        workflows = tmp_path / "workflows"

        # Pipeline with code step that reads named context
        p_dir = workflows / "ctx_test"
        p_dir.mkdir(parents=True)
        (p_dir / "scripts").mkdir()

        pipeline_yaml = {
            "id": "ctx_test",
            "name": "Context Test",
            "description": "Test named context",
            "inputs": {
                "alpha": {"type": "image", "label": "Alpha Image"},
                "beta": {"type": "image", "label": "Beta Image"},
            },
            "steps": [
                {"type": "code", "script": "merge.py", "function": "process"},
            ],
        }
        with open(p_dir / "pipeline.yaml", "w") as f:
            yaml.dump(pipeline_yaml, f)

        # Script: check both inputs exist, create output
        script = textwrap.dedent("""\
            from casadei.media import ImageMedia, Media
            from PIL import Image as PILImage

            def process(context):
                assert "alpha" in context, "missing alpha"
                assert "beta" in context, "missing beta"
                assert context["alpha"].image.size == (64, 64)
                assert context["beta"].image.size == (64, 64)
                out = PILImage.new("RGB", (64, 64), "green")
                return {"image": ImageMedia(image=out)}
        """)
        (p_dir / "scripts" / "merge.py").write_text(script)

        # Also add a single-input pipeline
        s_dir = workflows / "single_test"
        s_dir.mkdir(parents=True)
        (s_dir / "scripts").mkdir()

        single_yaml = {
            "id": "single_test",
            "name": "Single Test",
            "description": "Test single image backward compat",
            "steps": [
                {"type": "code", "script": "echo.py", "function": "process"},
            ],
        }
        with open(s_dir / "pipeline.yaml", "w") as f:
            yaml.dump(single_yaml, f)

        echo_script = textwrap.dedent("""\
            from casadei.media import ImageMedia, Media
            from PIL import Image as PILImage

            def process(context):
                # Single-image backward compat: should have "image" key
                assert "image" in context, f"missing 'image' key, have: {list(context.keys())}"
                out = PILImage.new("RGB", (64, 64), "blue")
                return {"image": ImageMedia(image=out)}
        """)
        (s_dir / "scripts" / "echo.py").write_text(echo_script)

        agents_dir = tmp_path / "agents"
        agents_dir.mkdir(exist_ok=True)

        app = create_app(data_dir=tmp_path, workflows_dir=workflows, agents_dir=agents_dir)
        return TestClient(app)

    def test_multi_image_context_passes_both_keys(self, client_code_pipeline):
        resp = client_code_pipeline.post(
            "/api/run",
            data={
                "type": "pipeline",
                "name": "ctx_test",
                "template_variables": "{}",
                "image_keys": json.dumps(["alpha", "beta"]),
            },
            files=[
                ("images", ("alpha.png", _png_bytes("red", (64, 64)), "image/png")),
                ("images", ("beta.png", _png_bytes("blue", (64, 64)), "image/png")),
            ],
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]
        run_id = resp.json()["run_id"]

        # Wait for completion
        for _ in range(20):
            time.sleep(0.3)
            job = client_code_pipeline.get(f"/api/jobs/{job_id}").json()
            if job["status"] in ("completed", "failed"):
                break

        assert job["status"] == "completed", f"Job failed: {job.get('error')}"

    def test_single_image_backward_compat(self, client_code_pipeline):
        """Single image upload with a non-'image' key should alias to 'image'."""
        resp = client_code_pipeline.post(
            "/api/run",
            data={
                "type": "pipeline",
                "name": "single_test",
                "template_variables": "{}",
                "image_keys": json.dumps(["photo"]),
            },
            files=[
                ("images", ("photo.png", _png_bytes("red", (64, 64)), "image/png")),
            ],
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        for _ in range(20):
            time.sleep(0.3)
            job = client_code_pipeline.get(f"/api/jobs/{job_id}").json()
            if job["status"] in ("completed", "failed"):
                break

        assert job["status"] == "completed", f"Job failed: {job.get('error')}"

    def test_legacy_single_image_form_field(self, client_code_pipeline):
        """Legacy 'image' form field should work for single-input pipelines."""
        resp = client_code_pipeline.post(
            "/api/run",
            data={
                "type": "pipeline",
                "name": "single_test",
                "template_variables": "{}",
            },
            files=[
                ("image", ("test.png", _png_bytes("red", (64, 64)), "image/png")),
            ],
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        for _ in range(20):
            time.sleep(0.3)
            job = client_code_pipeline.get(f"/api/jobs/{job_id}").json()
            if job["status"] in ("completed", "failed"):
                break

        assert job["status"] == "completed", f"Job failed: {job.get('error')}"

    def test_code_step_result_merges_into_context(self, client_code_pipeline):
        """Code step output should merge into context and be available for save."""
        resp = client_code_pipeline.post(
            "/api/run",
            data={
                "type": "pipeline",
                "name": "ctx_test",
                "template_variables": "{}",
                "image_keys": json.dumps(["alpha", "beta"]),
            },
            files=[
                ("images", ("alpha.png", _png_bytes("red", (64, 64)), "image/png")),
                ("images", ("beta.png", _png_bytes("blue", (64, 64)), "image/png")),
            ],
        )
        job_id = resp.json()["job_id"]
        run_id = resp.json()["run_id"]

        for _ in range(20):
            time.sleep(0.3)
            job = client_code_pipeline.get(f"/api/jobs/{job_id}").json()
            if job["status"] in ("completed", "failed"):
                break

        assert job["status"] == "completed"

        # The output file should exist (code step returned {"image": ...})
        from casadei.api.app import create_app
        # Check the results directory
        results_path = client_code_pipeline.app.state.results_dir / "scratch" / run_id / "output_0.png"
        # Access through the static file endpoint
        result_resp = client_code_pipeline.get(f"/api/results/scratch/{run_id}/output_0.png")
        assert result_resp.status_code == 200

    def test_unknown_pipeline_fails(self, client_code_pipeline):
        resp = client_code_pipeline.post(
            "/api/run",
            data={
                "type": "pipeline",
                "name": "nonexistent",
                "template_variables": "{}",
            },
            files=[
                ("image", ("test.png", _png_bytes(), "image/png")),
            ],
        )
        assert resp.status_code == 202
        job_id = resp.json()["job_id"]

        for _ in range(20):
            time.sleep(0.3)
            job = client_code_pipeline.get(f"/api/jobs/{job_id}").json()
            if job["status"] in ("completed", "failed"):
                break

        assert job["status"] == "failed"
        assert "Unknown pipeline" in job["error"]
