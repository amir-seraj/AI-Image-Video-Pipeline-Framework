# tests/test_integration.py
"""Integration test: full workflow from agent configs to pipeline execution."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image as PILImage

from casadei import (
    Agent, AgentConfig, Pipeline, AgentStep, CodeStep, PipelineStep,
    ImageMedia, TextMedia, MediaBundle, LoggedPipeline, save_agent, load_agent, to_mermaid,
)
from casadei.media import Media


class TestFullWorkflow:
    def test_agent_config_roundtrip_and_pipeline(self, tmp_path):
        # 1. Create and save agent configs with $variable templates
        cleaner_config = AgentConfig(
            name="image_cleaner", model="qwen_image_edit",
            description="Cleans and enhances images",
            prompt_template="Clean up this image, focusing on $focus_area",
            negative_prompt="blurry, noisy",
            params={"num_inference_steps": 30},
        )
        styler_config = AgentConfig(
            name="style_transfer", model="qwen_image_edit",
            description="Applies artistic style",
            prompt_template="Apply $style artistic style to this image",
            params={"num_inference_steps": 50},
        )

        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        save_agent(cleaner_config, agents_dir / "image_cleaner.yaml")
        save_agent(styler_config, agents_dir / "style_transfer.yaml")

        # 2. Load agent configs from disk
        loaded_cleaner = load_agent(agents_dir / "image_cleaner.yaml")
        loaded_styler = load_agent(agents_dir / "style_transfer.yaml")
        assert loaded_cleaner.name == "image_cleaner"
        assert "$style" in loaded_styler.prompt_template

        # 3. Create agents with mocked models
        with patch("casadei.agent.default_registry") as mock_registry:
            def make_mock_model():
                model = MagicMock()
                model.run.side_effect = lambda bundle: MediaBundle(items={
                    "image": ImageMedia(image=PILImage.new("RGB", (512, 512), color="green")),
                })
                return model

            mock_cls = MagicMock(side_effect=lambda: make_mock_model())
            mock_registry.get.return_value = mock_cls

            cleaner_agent = Agent(config=loaded_cleaner)
            styler_agent = Agent(config=loaded_styler)
            cleaner_agent.load()
            styler_agent.load()

            # 4. Build pipeline with mixed step types
            def resize_fn(ctx: dict[str, Media]) -> dict[str, Media]:
                img = ctx["raw_image"]
                assert isinstance(img, ImageMedia)
                resized = img.image.resize((512, 512))
                return {"resized_image": ImageMedia(image=resized)}

            pipeline = Pipeline(name="clean_and_style", steps=[
                CodeStep(name="resize", fn=resize_fn),
                AgentStep(name="clean", agent=cleaner_agent,
                          input_map={"image": "resized_image"},
                          output_map={"image": "clean_image"},
                          template_kwargs={"focus_area": "background noise"}),
                AgentStep(name="stylize", agent=styler_agent,
                          input_map={"image": "clean_image"},
                          output_map={"image": "styled_image"},
                          template_kwargs={"style": "impressionist"}),
            ])

            # 5. Run with logging
            logged = LoggedPipeline(pipeline)
            input_image = ImageMedia(image=PILImage.new("RGB", (1024, 1024), color="red"))
            result, log = logged.run({"raw_image": input_image})

            # 6. Verify results
            assert "styled_image" in result
            assert isinstance(result["styled_image"], ImageMedia)

            # 7. Verify logging
            assert log.pipeline_name == "clean_and_style"
            assert len(log.step_logs) == 3
            assert log.step_logs[0].step_name == "resize"
            assert log.step_logs[0].step_type == "CodeStep"
            assert log.step_logs[1].step_name == "clean"
            assert log.step_logs[1].step_type == "AgentStep"
            assert log.total_duration_ms >= 0
            summary = log.summary()
            assert "clean_and_style" in summary

            # 8. Verify visualization
            mermaid = to_mermaid(pipeline)
            assert "resize" in mermaid
            assert "clean" in mermaid
            assert "stylize" in mermaid

    def test_pipeline_composition_with_nested(self, tmp_path):
        with patch("casadei.agent.default_registry") as mock_registry:
            mock_model = MagicMock()
            mock_model.run.return_value = MediaBundle(items={
                "image": ImageMedia(image=PILImage.new("RGB", (256, 256))),
            })
            mock_cls = MagicMock(return_value=mock_model)
            mock_registry.get.return_value = mock_cls

            agent_a = Agent(config=AgentConfig(name="preprocess", model="qwen_image_edit"))
            agent_a.load()
            preprocessing = Pipeline(name="preprocessing", steps=[
                AgentStep(name="pre", agent=agent_a,
                          input_map={"image": "raw"},
                          output_map={"image": "preprocessed"}),
            ])

            agent_b = Agent(config=AgentConfig(name="edit", model="qwen_image_edit"))
            agent_b.load()
            full_pipeline = Pipeline(name="full_workflow", steps=[
                PipelineStep(name="preprocess_step", pipeline=preprocessing,
                             input_map={"raw": "input_image"},
                             output_map={"preprocessed": "clean_image"}),
                AgentStep(name="final_edit", agent=agent_b,
                          input_map={"image": "clean_image"},
                          output_map={"image": "result"}),
            ])

            logged = LoggedPipeline(full_pipeline)
            result, log = logged.run({
                "input_image": ImageMedia(image=PILImage.new("RGB", (512, 512))),
            })
            assert "result" in result
            assert len(log.step_logs) == 2

            mermaid = to_mermaid(full_pipeline, expand_nested=True)
            assert "subgraph" in mermaid
            assert "preprocessing" in mermaid
