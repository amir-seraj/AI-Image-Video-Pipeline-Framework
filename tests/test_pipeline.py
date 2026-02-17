# tests/test_pipeline.py
import pytest
from unittest.mock import MagicMock
from PIL import Image as PILImage

from casadei.media import ImageMedia, TextMedia, Media, MediaBundle
from casadei.pipeline import AgentStep, CodeStep, PipelineStep, Pipeline


class TestAgentStep:
    def test_create_step(self):
        agent = MagicMock()
        step = AgentStep(
            name="edit",
            agent=agent,
            input_map={"image": "source_image"},
            output_map={"image": "edited_image"},
        )
        assert step.name == "edit"
        assert step.input_map == {"image": "source_image"}

    def test_execute_maps_inputs(self):
        agent = MagicMock()
        agent.execute.return_value = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100))),
        })
        step = AgentStep(
            name="edit",
            agent=agent,
            input_map={"image": "source_image"},
            output_map={"image": "result"},
        )
        context = {
            "source_image": ImageMedia(image=PILImage.new("RGB", (200, 200))),
        }
        outputs = step.execute(context)
        assert "result" in outputs
        call_bundle = agent.execute.call_args[0][0]
        assert "image" in call_bundle.items

    def test_passes_template_kwargs(self):
        agent = MagicMock()
        agent.execute.return_value = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100))),
        })
        step = AgentStep(
            name="style",
            agent=agent,
            input_map={"image": "source"},
            output_map={"image": "result"},
            template_kwargs={"style": "watercolor"},
        )
        context = {"source": ImageMedia(image=PILImage.new("RGB", (100, 100)))}
        step.execute(context)
        _, kwargs = agent.execute.call_args
        assert kwargs.get("style") == "watercolor"

    def test_missing_input_raises(self):
        agent = MagicMock()
        step = AgentStep(
            name="edit",
            agent=agent,
            input_map={"image": "nonexistent"},
            output_map={"image": "out"},
        )
        with pytest.raises(KeyError, match="nonexistent"):
            step.execute({})


class TestCodeStep:
    def test_simple_function(self):
        def resize(context: dict[str, Media]) -> dict[str, Media]:
            img = context["image"]
            assert isinstance(img, ImageMedia)
            resized = img.image.resize((64, 64))
            return {"small_image": ImageMedia(image=resized)}

        step = CodeStep(name="resize", fn=resize)
        context = {"image": ImageMedia(image=PILImage.new("RGB", (256, 256)))}
        outputs = step.execute(context)
        assert "small_image" in outputs
        assert outputs["small_image"].width == 64

    def test_code_step_can_access_full_context(self):
        def combine(context: dict[str, Media]) -> dict[str, Media]:
            prompt = context["prompt"]
            suffix = context["suffix"]
            assert isinstance(prompt, TextMedia)
            assert isinstance(suffix, TextMedia)
            combined = TextMedia(text=f"{prompt.text} {suffix.text}")
            return {"full_prompt": combined}

        step = CodeStep(name="combine", fn=combine)
        context = {
            "prompt": TextMedia(text="hello"),
            "suffix": TextMedia(text="world"),
        }
        outputs = step.execute(context)
        assert outputs["full_prompt"].text == "hello world"

    def test_code_step_error_propagates(self):
        def bad_fn(context):
            raise ValueError("something broke")
        step = CodeStep(name="bad", fn=bad_fn)
        with pytest.raises(ValueError, match="something broke"):
            step.execute({})


class TestPipelineStep:
    def test_nested_pipeline(self):
        agent = MagicMock()
        agent.execute.return_value = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100), color="green")),
        })
        inner_pipeline = Pipeline(
            name="inner",
            steps=[AgentStep(
                name="edit",
                agent=agent,
                input_map={"image": "input_image"},
                output_map={"image": "output_image"},
            )],
        )
        step = PipelineStep(
            name="sub_pipeline",
            pipeline=inner_pipeline,
            input_map={"input_image": "raw_image"},
            output_map={"output_image": "processed_image"},
        )
        context = {"raw_image": ImageMedia(image=PILImage.new("RGB", (200, 200)))}
        outputs = step.execute(context)
        assert "processed_image" in outputs


class TestPipeline:
    def test_create_pipeline(self):
        pipeline = Pipeline(name="test")
        assert pipeline.name == "test"
        assert len(pipeline.steps) == 0

    def test_add_step(self):
        pipeline = Pipeline(name="test")
        agent = MagicMock()
        step = AgentStep(name="s", agent=agent, input_map={}, output_map={})
        pipeline.add_step(step)
        assert len(pipeline.steps) == 1

    def test_run_single_agent_step(self):
        output_img = PILImage.new("RGB", (100, 100), color="green")
        agent = MagicMock()
        agent.execute.return_value = MediaBundle(items={
            "image": ImageMedia(image=output_img),
        })
        step = AgentStep(
            name="edit",
            agent=agent,
            input_map={"image": "input_image"},
            output_map={"image": "output_image"},
        )
        pipeline = Pipeline(name="test", steps=[step])
        result = pipeline.run({
            "input_image": ImageMedia(image=PILImage.new("RGB", (200, 200))),
        })
        assert "output_image" in result

    def test_run_chained_steps(self):
        agent1 = MagicMock()
        agent1.execute.return_value = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100), color="blue")),
        })
        agent2 = MagicMock()
        agent2.execute.return_value = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100), color="red")),
        })
        pipeline = Pipeline(name="two_step", steps=[
            AgentStep(name="clean", agent=agent1,
                      input_map={"image": "raw_image"},
                      output_map={"image": "clean_image"}),
            AgentStep(name="style", agent=agent2,
                      input_map={"image": "clean_image"},
                      output_map={"image": "final_image"}),
        ])
        result = pipeline.run({
            "raw_image": ImageMedia(image=PILImage.new("RGB", (300, 300))),
        })
        assert "final_image" in result
        assert result["final_image"].image.getpixel((50, 50)) == (255, 0, 0)

    def test_mixed_step_types(self):
        def resize_fn(ctx):
            img = ctx["raw"]
            return {"resized": ImageMedia(image=img.image.resize((256, 256)))}

        agent = MagicMock()
        agent.execute.return_value = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (256, 256), color="blue")),
        })

        final_agent = MagicMock()
        final_agent.execute.return_value = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (256, 256), color="green")),
        })
        inner = Pipeline(name="final_touch", steps=[
            AgentStep(name="polish", agent=final_agent,
                      input_map={"image": "edit_input"},
                      output_map={"image": "edit_output"}),
        ])

        pipeline = Pipeline(name="full", steps=[
            CodeStep(name="resize", fn=resize_fn),
            AgentStep(name="edit", agent=agent,
                      input_map={"image": "resized"},
                      output_map={"image": "edited"}),
            PipelineStep(name="final", pipeline=inner,
                         input_map={"edit_input": "edited"},
                         output_map={"edit_output": "final_result"}),
        ])

        result = pipeline.run({
            "raw": ImageMedia(image=PILImage.new("RGB", (1024, 1024))),
        })
        assert "final_result" in result
        assert result["final_result"].image.getpixel((0, 0)) == (0, 128, 0)


class TestPipelineCompose:
    def test_compose_two_pipelines(self):
        agent1 = MagicMock()
        agent1.execute.return_value = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100), color="blue")),
        })
        agent2 = MagicMock()
        agent2.execute.return_value = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100), color="red")),
        })
        p1 = Pipeline(name="first", steps=[
            AgentStep(name="a", agent=agent1,
                      input_map={"image": "input_image"},
                      output_map={"image": "mid_image"}),
        ])
        p2 = Pipeline(name="second", steps=[
            AgentStep(name="b", agent=agent2,
                      input_map={"image": "mid_image"},
                      output_map={"image": "final_image"}),
        ])
        composed = Pipeline.compose("combined", [p1, p2])
        result = composed.run({
            "input_image": ImageMedia(image=PILImage.new("RGB", (200, 200))),
        })
        assert "final_image" in result

    def test_pipeline_load_and_unload(self):
        agent = MagicMock()
        step = AgentStep(name="s", agent=agent, input_map={}, output_map={})
        pipeline = Pipeline(name="test", steps=[step])
        pipeline.load()
        agent.load.assert_called_once()
        pipeline.unload()
        agent.unload.assert_called_once()
