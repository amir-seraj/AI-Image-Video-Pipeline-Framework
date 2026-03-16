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
