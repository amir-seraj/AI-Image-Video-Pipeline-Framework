"""Tests for workflows/lowfi_to_hifi/pipeline.py."""
import pytest
from unittest.mock import patch, MagicMock
from PIL import Image as PILImage

from casadei.media import ImageMedia
from casadei.pipeline import AgentStep, Pipeline


# Import path for the module under test
import sys
from pathlib import Path

_WORKFLOW_DIR = Path(__file__).resolve().parent.parent / "workflows" / "lowfi_to_hifi"
if str(_WORKFLOW_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW_DIR))


class TestBuildPipeline:
    def test_sketch_only_mode(self):
        from pipeline import build_pipeline

        spec = {}
        pipeline, agent = build_pipeline(spec)

        assert isinstance(pipeline, Pipeline)
        assert pipeline.name == "lowfi_to_hifi"
        assert len(pipeline.steps) == 1

        step = pipeline.steps[0]
        assert isinstance(step, AgentStep)
        assert step.input_map == {"image": "sketch"}
        assert step.output_map == {"image": "image"}

    def test_volume_mode(self):
        from pipeline import build_pipeline

        spec = {"volume": True}
        pipeline, agent = build_pipeline(spec)

        step = pipeline.steps[0]
        assert step.input_map == {"image": "sketch", "volume": "volume"}

    def test_extra_specs_passed(self):
        from pipeline import build_pipeline

        spec = {"extra": {"style": "minimal", "brand": "Casadei"}}
        pipeline, agent = build_pipeline(spec)

        step = pipeline.steps[0]
        assert "extra_specs" in step.template_kwargs
        assert "Style: minimal" in step.template_kwargs["extra_specs"]
        assert "Brand: Casadei" in step.template_kwargs["extra_specs"]

    def test_custom_temperature(self):
        from pipeline import build_pipeline

        spec = {}
        pipeline, agent = build_pipeline(spec, temperature=1.0)

        assert agent.config.params["temperature"] == 1.0

    def test_default_temperature(self):
        from pipeline import build_pipeline

        spec = {}
        pipeline, agent = build_pipeline(spec)

        assert agent.config.params["temperature"] == 0.8


class TestSaveResults:
    def test_saves_result_image(self, tmp_path):
        from pipeline import save_results

        img = ImageMedia(image=PILImage.new("RGB", (512, 512), "white"))
        save_results(
            run_dir=tmp_path / "run1",
            result_image=img,
            spec={},
            total_elapsed=2.5,
        )
        assert (tmp_path / "run1" / "result.png").exists()
        assert (tmp_path / "run1" / "results.json").exists()
        assert (tmp_path / "run1" / "summary.txt").exists()

    def test_metadata_contains_mode_sketch_only(self, tmp_path):
        import json
        from pipeline import save_results

        img = ImageMedia(image=PILImage.new("RGB", (512, 512), "white"))
        save_results(
            run_dir=tmp_path / "run2",
            result_image=img,
            spec={},
            total_elapsed=1.0,
        )
        data = json.loads((tmp_path / "run2" / "results.json").read_text())
        assert data["mode"] == "sketch_only"

    def test_metadata_contains_mode_with_volume(self, tmp_path):
        import json
        from pipeline import save_results

        img = ImageMedia(image=PILImage.new("RGB", (512, 512), "white"))
        save_results(
            run_dir=tmp_path / "run3",
            result_image=img,
            spec={"volume": True},
            total_elapsed=1.0,
        )
        data = json.loads((tmp_path / "run3" / "results.json").read_text())
        assert data["mode"] == "with_volume"

    def test_token_records_saved(self, tmp_path):
        import json
        from pipeline import save_results

        img = ImageMedia(image=PILImage.new("RGB", (512, 512), "white"))
        records = [{"model": "gemini-3.1-flash-image-preview", "input_tokens": 100, "output_tokens": 50, "thinking_tokens": 0, "cached_tokens": 0, "total_tokens": 150}]
        save_results(
            run_dir=tmp_path / "run4",
            result_image=img,
            spec={},
            total_elapsed=1.0,
            token_records=records,
        )
        data = json.loads((tmp_path / "run4" / "results.json").read_text())
        assert "token_usage" in data
        assert data["token_usage"]["records"] == records

    def test_none_image_no_crash(self, tmp_path):
        from pipeline import save_results

        save_results(
            run_dir=tmp_path / "run5",
            result_image=None,
            spec={},
            total_elapsed=0.5,
        )
        assert (tmp_path / "run5" / "results.json").exists()
        assert not (tmp_path / "run5" / "result.png").exists()
