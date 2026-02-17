# tests/test_logging.py
import pytest
import time
from unittest.mock import MagicMock
from PIL import Image as PILImage

from casadei.media import ImageMedia, TextMedia, Media, MediaBundle
from casadei.pipeline import AgentStep, CodeStep, PipelineStep, Pipeline
from casadei.logging import ExecutionLog, StepLog, LoggedPipeline


class TestStepLog:
    def test_step_log_fields(self):
        log = StepLog(
            step_name="edit",
            step_type="AgentStep",
            duration_ms=1234.5,
            input_keys=["image", "prompt"],
            output_keys=["image"],
        )
        assert log.step_name == "edit"
        assert log.duration_ms == 1234.5
        assert log.step_type == "AgentStep"

    def test_step_log_str(self):
        log = StepLog(
            step_name="edit",
            step_type="AgentStep",
            duration_ms=150.3,
            input_keys=["image"],
            output_keys=["result"],
        )
        s = str(log)
        assert "edit" in s
        assert "150.3" in s or "150" in s


class TestExecutionLog:
    def test_execution_log_fields(self):
        log = ExecutionLog(
            pipeline_name="test",
            total_duration_ms=5000.0,
            step_logs=[
                StepLog("s1", "AgentStep", 2000.0, ["a"], ["b"]),
                StepLog("s2", "CodeStep", 3000.0, ["b"], ["c"]),
            ],
        )
        assert log.pipeline_name == "test"
        assert log.total_duration_ms == 5000.0
        assert len(log.step_logs) == 2

    def test_execution_log_summary(self):
        log = ExecutionLog(
            pipeline_name="test",
            total_duration_ms=5000.0,
            step_logs=[
                StepLog("s1", "AgentStep", 2000.0, ["a"], ["b"]),
                StepLog("s2", "CodeStep", 3000.0, ["b"], ["c"]),
            ],
        )
        summary = log.summary()
        assert "test" in summary
        assert "s1" in summary
        assert "s2" in summary
        assert "5000" in summary or "5.0" in summary


class TestLoggedPipeline:
    def test_logged_pipeline_returns_result_and_log(self):
        agent = MagicMock()
        agent.execute.return_value = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100))),
        })
        pipeline = Pipeline(name="test", steps=[
            AgentStep(name="edit", agent=agent,
                      input_map={"image": "input"},
                      output_map={"image": "output"}),
        ])
        logged = LoggedPipeline(pipeline)
        result, log = logged.run({"input": ImageMedia(image=PILImage.new("RGB", (200, 200)))})
        assert "output" in result
        assert isinstance(log, ExecutionLog)
        assert log.pipeline_name == "test"
        assert len(log.step_logs) == 1
        assert log.step_logs[0].step_name == "edit"
        assert log.step_logs[0].step_type == "AgentStep"
        assert log.total_duration_ms >= 0

    def test_logged_pipeline_times_code_step(self):
        def slow_fn(ctx):
            time.sleep(0.05)
            return {"result": TextMedia(text="done")}
        pipeline = Pipeline(name="timed", steps=[
            CodeStep(name="slow", fn=slow_fn),
        ])
        logged = LoggedPipeline(pipeline)
        result, log = logged.run({})
        assert log.step_logs[0].duration_ms >= 40
        assert log.total_duration_ms >= 40

    def test_logged_pipeline_nested(self):
        agent = MagicMock()
        agent.execute.return_value = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (50, 50))),
        })
        inner = Pipeline(name="inner", steps=[
            AgentStep(name="inner_edit", agent=agent,
                      input_map={"image": "in"}, output_map={"image": "out"}),
        ])
        outer = Pipeline(name="outer", steps=[
            PipelineStep(name="sub", pipeline=inner,
                         input_map={"in": "raw"},
                         output_map={"out": "final"}),
        ])
        logged = LoggedPipeline(outer)
        result, log = logged.run({"raw": ImageMedia(image=PILImage.new("RGB", (100, 100)))})
        assert "final" in result
        assert len(log.step_logs) == 1
        assert log.step_logs[0].step_name == "sub"
        assert log.step_logs[0].step_type == "PipelineStep"
