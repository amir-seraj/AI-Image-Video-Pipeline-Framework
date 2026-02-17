# tests/test_visualization.py
import pytest
from unittest.mock import MagicMock
from PIL import Image as PILImage

from casadei.media import ImageMedia, TextMedia, Media, MediaBundle
from casadei.pipeline import AgentStep, CodeStep, PipelineStep, Pipeline
from casadei.visualization import to_mermaid


class TestMermaidVisualization:
    def test_single_agent_step(self):
        agent = MagicMock()
        pipeline = Pipeline(name="simple", steps=[
            AgentStep(name="edit", agent=agent,
                      input_map={"image": "raw"},
                      output_map={"image": "edited"}),
        ])
        mermaid = to_mermaid(pipeline)
        assert "graph" in mermaid or "flowchart" in mermaid
        assert "edit" in mermaid
        assert "raw" in mermaid
        assert "edited" in mermaid

    def test_chained_steps(self):
        agent1 = MagicMock()
        agent2 = MagicMock()
        pipeline = Pipeline(name="chain", steps=[
            AgentStep(name="clean", agent=agent1,
                      input_map={"image": "raw"},
                      output_map={"image": "clean"}),
            AgentStep(name="style", agent=agent2,
                      input_map={"image": "clean"},
                      output_map={"image": "styled"}),
        ])
        mermaid = to_mermaid(pipeline)
        assert "clean" in mermaid
        assert "style" in mermaid
        assert "raw" in mermaid
        assert "styled" in mermaid

    def test_mixed_step_types(self):
        agent = MagicMock()
        def my_fn(ctx):
            return {}
        inner = Pipeline(name="inner", steps=[
            AgentStep(name="inner_edit", agent=agent,
                      input_map={}, output_map={}),
        ])
        pipeline = Pipeline(name="mixed", steps=[
            CodeStep(name="preprocess", fn=my_fn),
            AgentStep(name="edit", agent=agent,
                      input_map={"image": "preprocessed"},
                      output_map={"image": "edited"}),
            PipelineStep(name="postprocess", pipeline=inner,
                         input_map={"in": "edited"},
                         output_map={"out": "final"}),
        ])
        mermaid = to_mermaid(pipeline)
        assert "preprocess" in mermaid
        assert "edit" in mermaid
        assert "postprocess" in mermaid

    def test_nested_pipeline_shows_subgraph(self):
        agent = MagicMock()
        inner = Pipeline(name="inner_pipeline", steps=[
            AgentStep(name="sub_step", agent=agent,
                      input_map={"image": "x"},
                      output_map={"image": "y"}),
        ])
        outer = Pipeline(name="outer", steps=[
            PipelineStep(name="nested", pipeline=inner,
                         input_map={"x": "input"},
                         output_map={"y": "output"}),
        ])
        mermaid = to_mermaid(pipeline=outer, expand_nested=True)
        assert "subgraph" in mermaid
        assert "inner_pipeline" in mermaid
        assert "sub_step" in mermaid

    def test_output_is_valid_mermaid_syntax(self):
        agent = MagicMock()
        pipeline = Pipeline(name="test", steps=[
            AgentStep(name="step1", agent=agent,
                      input_map={"a": "x"}, output_map={"b": "y"}),
        ])
        mermaid = to_mermaid(pipeline)
        lines = mermaid.strip().split("\n")
        assert lines[0].startswith("flowchart") or lines[0].startswith("graph")
