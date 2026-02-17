"""Execution logging for pipeline runs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from casadei.media import Media
from casadei.pipeline import Pipeline, Step, AgentStep, CodeStep, PipelineStep


@dataclass
class StepLog:
    step_name: str
    step_type: str
    duration_ms: float
    input_keys: list[str]
    output_keys: list[str]

    def __str__(self) -> str:
        return (
            f"  [{self.step_type}] {self.step_name}: "
            f"{self.duration_ms:.1f}ms "
            f"({', '.join(self.input_keys)} -> {', '.join(self.output_keys)})"
        )


@dataclass
class ExecutionLog:
    pipeline_name: str
    total_duration_ms: float
    step_logs: list[StepLog] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Pipeline '{self.pipeline_name}' — {self.total_duration_ms:.1f}ms total",
            f"  Steps: {len(self.step_logs)}",
        ]
        for sl in self.step_logs:
            lines.append(str(sl))
        return "\n".join(lines)


class LoggedPipeline:
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    def run(self, inputs: dict[str, Media]) -> tuple[dict[str, Media], ExecutionLog]:
        context = dict(inputs)
        step_logs = []
        pipeline_start = time.perf_counter()
        for step in self.pipeline.steps:
            step_start = time.perf_counter()
            input_keys = list(context.keys())
            outputs = step.execute(context)
            context.update(outputs)
            step_end = time.perf_counter()
            duration_ms = (step_end - step_start) * 1000
            step_logs.append(StepLog(
                step_name=step.name,
                step_type=type(step).__name__,
                duration_ms=duration_ms,
                input_keys=input_keys,
                output_keys=list(outputs.keys()),
            ))
        pipeline_end = time.perf_counter()
        total_ms = (pipeline_end - pipeline_start) * 1000
        log = ExecutionLog(
            pipeline_name=self.pipeline.name,
            total_duration_ms=total_ms,
            step_logs=step_logs,
        )
        return context, log
