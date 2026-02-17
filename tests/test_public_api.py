"""Verify the public API is clean and accessible."""
import pytest


class TestPublicAPI:
    def test_import_media(self):
        from casadei import ImageMedia, TextMedia, VideoMedia, MediaBundle
        assert ImageMedia is not None

    def test_import_models(self):
        from casadei import AIModel, ImageEditModel, ModelCapability
        assert AIModel is not None

    def test_import_constraints(self):
        from casadei import ImageConstraint, TextConstraint, VideoConstraint
        assert ImageConstraint is not None

    def test_import_providers(self):
        from casadei import QwenImageEdit
        assert QwenImageEdit is not None

    def test_import_agent(self):
        from casadei import Agent, AgentConfig, load_agent, save_agent
        assert Agent is not None

    def test_import_pipeline(self):
        from casadei import Pipeline, AgentStep, CodeStep, PipelineStep
        assert Pipeline is not None
        assert CodeStep is not None
        assert PipelineStep is not None

    def test_import_logging(self):
        from casadei import LoggedPipeline, ExecutionLog, StepLog
        assert LoggedPipeline is not None

    def test_import_visualization(self):
        from casadei import to_mermaid
        assert to_mermaid is not None

    def test_import_registry(self):
        from casadei import default_registry
        assert default_registry is not None
