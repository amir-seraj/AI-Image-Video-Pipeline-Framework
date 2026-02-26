"""Unit tests for LoopStep — the agentic generate-judge-repair loop.

All tests use mocked agents so no GPU or real model weights are needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest
from PIL import Image as PILImage

from casadei.media import ImageMedia, TextMedia, MediaBundle
from casadei.agent import Agent, AgentConfig
from casadei.pipeline import AgentStep, Pipeline
from casadei.loop import LoopStep, LoopIteration, LoopResult
from casadei.logging import LoggedPipeline


def _make_image(color: str = "red") -> ImageMedia:
    colors = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}
    return ImageMedia(image=PILImage.new("RGB", (64, 64), colors.get(color, (128, 128, 128))))


def _make_mock_agent(output_image: ImageMedia | None = None):
    """Create a mock Agent with load/unload/execute."""
    agent = MagicMock(spec=Agent)
    agent.config = AgentConfig(name="mock", model="qwen_image_edit")
    if output_image is None:
        output_image = _make_image("green")
    agent.execute.return_value = MediaBundle(items={"image": output_image})
    return agent


class TestLoopStepPassOnFirst:
    """Test: loop passes on the first iteration."""

    def test_accepts_immediately(self):
        agent = _make_mock_agent()
        step = AgentStep(
            name="gen",
            agent=agent,
            input_map={"image": "person", "image_2": "shoe"},
            output_map={"image": "image"},
            template_kwargs={"feedback": ""},
        )

        def judge(ctx):
            return True, "Looks good."

        loop = LoopStep(
            name="test_loop",
            body=[step],
            judge=judge,
            max_iterations=5,
            swap_models=True,
        )

        context = {
            "person": _make_image("red"),
            "shoe": _make_image("blue"),
        }

        outputs = loop.execute(context)

        # Should have called execute exactly once
        assert agent.execute.call_count == 1
        # Should have loaded and unloaded (swap_models=True)
        assert agent.load.call_count == 1
        assert agent.unload.call_count == 1

        # Result should contain image and history
        assert "image" in outputs
        assert isinstance(outputs["image"], ImageMedia)

        history = outputs["test_loop_history"]
        assert isinstance(history, LoopResult)
        assert len(history.iterations) == 1
        assert history.iterations[0].accepted is True
        assert history.iterations[0].feedback == "Looks good."


class TestLoopStepPassOnSecond:
    """Test: loop fails first, passes on second iteration."""

    def test_repairs_and_accepts(self):
        agent = _make_mock_agent()
        step = AgentStep(
            name="gen",
            agent=agent,
            input_map={"image": "person"},
            output_map={"image": "image"},
            template_kwargs={"feedback": ""},
        )

        call_count = [0]

        def judge(ctx):
            call_count[0] += 1
            if call_count[0] == 1:
                return False, "Shoes are wrong color."
            return True, "Now looks correct."

        loop = LoopStep(
            name="test_loop",
            body=[step],
            judge=judge,
            max_iterations=5,
        )

        context = {"person": _make_image()}
        outputs = loop.execute(context)

        # Two iterations
        assert agent.execute.call_count == 2
        assert agent.load.call_count == 2
        assert agent.unload.call_count == 2

        history = outputs["test_loop_history"]
        assert len(history.iterations) == 2
        assert history.iterations[0].accepted is False
        assert history.iterations[0].feedback == "Shoes are wrong color."
        assert history.iterations[1].accepted is True
        assert history.iterations[1].feedback == "Now looks correct."


class TestLoopStepMaxIterations:
    """Test: loop exhausts max_iterations and calls best_fn."""

    def test_best_fn_called(self):
        agent = _make_mock_agent()
        step = AgentStep(
            name="gen",
            agent=agent,
            input_map={"image": "person"},
            output_map={"image": "image"},
            template_kwargs={"feedback": ""},
        )

        def judge(ctx):
            return False, "Still not good."

        best_image = _make_image("blue")
        best_fn_called = [False]

        def best_fn(history, context):
            best_fn_called[0] = True
            assert len(history) == 3
            return {"image": best_image}

        loop = LoopStep(
            name="test_loop",
            body=[step],
            judge=judge,
            max_iterations=3,
            best_fn=best_fn,
        )

        context = {"person": _make_image()}
        outputs = loop.execute(context)

        assert best_fn_called[0] is True
        assert agent.execute.call_count == 3
        assert outputs["image"] is best_image

        history = outputs["test_loop_history"]
        assert len(history.iterations) == 3
        assert all(not it.accepted for it in history.iterations)


class TestLoopStepMaxIterationsNoBestFn:
    """Test: max iterations without best_fn returns last candidate."""

    def test_returns_last_without_best_fn(self):
        agent = _make_mock_agent()
        step = AgentStep(
            name="gen",
            agent=agent,
            input_map={"image": "person"},
            output_map={"image": "image"},
            template_kwargs={"feedback": ""},
        )

        def judge(ctx):
            return False, "Nope."

        loop = LoopStep(
            name="test_loop",
            body=[step],
            judge=judge,
            max_iterations=2,
            best_fn=None,
        )

        context = {"person": _make_image()}
        outputs = loop.execute(context)

        # Should still have image from last body execution
        assert "image" in outputs
        assert agent.execute.call_count == 2


class TestLoopStepSwapModelsFalse:
    """Test: swap_models=False loads once and unloads once."""

    def test_load_once_unload_once(self):
        agent = _make_mock_agent()
        step = AgentStep(
            name="gen",
            agent=agent,
            input_map={"image": "person"},
            output_map={"image": "image"},
            template_kwargs={"feedback": ""},
        )

        call_count = [0]

        def judge(ctx):
            call_count[0] += 1
            if call_count[0] >= 3:
                return True, "OK."
            return False, "Try again."

        loop = LoopStep(
            name="test_loop",
            body=[step],
            judge=judge,
            max_iterations=5,
            swap_models=False,
        )

        context = {"person": _make_image()}
        outputs = loop.execute(context)

        # swap_models=False: load once at start, unload once at end
        assert agent.load.call_count == 1
        assert agent.unload.call_count == 1
        # But execute still called 3 times
        assert agent.execute.call_count == 3


class TestLoopStepFeedbackInjection:
    """Test: feedback is propagated to template_kwargs between iterations."""

    def test_feedback_updated(self):
        captured_kwargs = []

        def capture_execute(bundle, **kwargs):
            captured_kwargs.append(dict(kwargs))
            return MediaBundle(items={"image": _make_image()})

        agent = _make_mock_agent()
        agent.execute.side_effect = capture_execute

        step = AgentStep(
            name="gen",
            agent=agent,
            input_map={"image": "person"},
            output_map={"image": "image"},
            template_kwargs={"feedback": "initial"},
        )

        call_count = [0]

        def judge(ctx):
            call_count[0] += 1
            if call_count[0] == 1:
                return False, "Fix the laces."
            return True, "Good."

        loop = LoopStep(
            name="test_loop",
            body=[step],
            judge=judge,
            max_iterations=5,
            feedback_template_var="feedback",
        )

        context = {"person": _make_image()}
        loop.execute(context)

        # After first rejection, the template_kwargs should have the feedback
        assert step.template_kwargs["feedback"] == "Fix the laces."


class TestLoopStepCustomHistoryKey:
    """Test: custom history_key is used."""

    def test_custom_key(self):
        agent = _make_mock_agent()
        step = AgentStep(
            name="gen",
            agent=agent,
            input_map={"image": "person"},
            output_map={"image": "image"},
            template_kwargs={"feedback": ""},
        )

        loop = LoopStep(
            name="test_loop",
            body=[step],
            judge=lambda ctx: (True, "OK"),
            max_iterations=5,
            history_key="my_custom_history",
        )

        context = {"person": _make_image()}
        outputs = loop.execute(context)

        assert "my_custom_history" in outputs
        assert isinstance(outputs["my_custom_history"], LoopResult)


class TestLoopStepInPipeline:
    """Test: LoopStep works inside a Pipeline with LoggedPipeline."""

    def test_logged_pipeline(self):
        agent = _make_mock_agent()
        step = AgentStep(
            name="gen",
            agent=agent,
            input_map={"image": "person"},
            output_map={"image": "image"},
            template_kwargs={"feedback": ""},
        )

        loop = LoopStep(
            name="test_loop",
            body=[step],
            judge=lambda ctx: (True, "Good"),
            max_iterations=3,
        )

        pipeline = Pipeline(name="test_pipeline", steps=[loop])
        logged = LoggedPipeline(pipeline)

        context = {"person": _make_image()}
        result, exec_log = logged.run(context)

        # LoggedPipeline should record one step log for the LoopStep
        assert len(exec_log.step_logs) == 1
        assert exec_log.step_logs[0].step_name == "test_loop"
        assert exec_log.step_logs[0].step_type == "LoopStep"
        assert exec_log.step_logs[0].duration_ms > 0

        # Results should be in the context
        assert "image" in result
        assert "test_loop_history" in result


class TestLoopStepPipelineLoadUnload:
    """Test: Pipeline.load/unload handles LoopStep."""

    def test_pipeline_unload_calls_loop_unload(self):
        agent = _make_mock_agent()
        step = AgentStep(
            name="gen",
            agent=agent,
            input_map={"image": "person"},
            output_map={"image": "image"},
            template_kwargs={"feedback": ""},
        )

        loop = LoopStep(
            name="test_loop",
            body=[step],
            judge=lambda ctx: (True, "OK"),
            max_iterations=3,
            swap_models=False,
        )

        pipeline = Pipeline(name="test", steps=[loop])

        # load should call loop.load which loads agents since swap_models=False
        pipeline.load()
        assert agent.load.call_count == 1

        # unload should call loop.unload
        pipeline.unload()
        assert agent.unload.call_count == 1


class TestLoopResult:
    """Test: LoopResult.summary() formatting."""

    def test_summary(self):
        result = LoopResult(
            iterations=[
                LoopIteration(0, {}, False, "Bad shoes", 100.0),
                LoopIteration(1, {}, True, "Good shoes", 200.0),
            ],
            total_ms=300.0,
        )

        summary = result.summary()
        assert "2 iterations" in summary
        assert "FAIL" in summary
        assert "PASS" in summary
        assert "Bad shoes" in summary
        assert "Good shoes" in summary


class TestLoopStepExceptionSafety:
    """Test: models are unloaded even if body raises."""

    def test_unload_on_exception(self):
        agent = _make_mock_agent()
        agent.execute.side_effect = RuntimeError("GPU OOM")

        step = AgentStep(
            name="gen",
            agent=agent,
            input_map={"image": "person"},
            output_map={"image": "image"},
            template_kwargs={"feedback": ""},
        )

        loop = LoopStep(
            name="test_loop",
            body=[step],
            judge=lambda ctx: (True, "OK"),
            max_iterations=3,
            swap_models=True,
        )

        context = {"person": _make_image()}
        with pytest.raises(RuntimeError, match="GPU OOM"):
            loop.execute(context)

        # Even though execute raised, unload should have been called
        assert agent.unload.call_count == 1
