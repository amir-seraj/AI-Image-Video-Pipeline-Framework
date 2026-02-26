"""Pipeline system — composable chains of agent, code, and sub-pipeline steps."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from casadei.agent import Agent
from casadei.media import Media, MediaBundle


class Step(ABC):
    """Base class for all pipeline steps."""
    name: str

    @abstractmethod
    def execute(self, context: dict[str, Media]) -> dict[str, Media]:
        """Execute this step given the pipeline context."""


@dataclass
class AgentStep(Step):
    name: str
    agent: Agent
    input_map: dict[str, str]
    output_map: dict[str, str]
    template_kwargs: dict[str, Any] = field(default_factory=dict)

    def execute(self, context: dict[str, Media]) -> dict[str, Media]:
        agent_inputs = {}
        for agent_key, context_key in self.input_map.items():
            if context_key not in context:
                raise KeyError(
                    f"AgentStep '{self.name}': input '{context_key}' not found in context. "
                    f"Available: {list(context.keys())}"
                )
            agent_inputs[agent_key] = context[context_key]
        bundle = MediaBundle(items=agent_inputs)
        result = self.agent.execute(bundle, **self.template_kwargs)
        outputs = {}
        for agent_key, context_key in self.output_map.items():
            if agent_key in result.items:
                outputs[context_key] = result[agent_key]
        return outputs


@dataclass
class CodeStep(Step):
    name: str
    fn: Callable[[dict[str, Media]], dict[str, Media]]

    def execute(self, context: dict[str, Media]) -> dict[str, Media]:
        return self.fn(context)


@dataclass
class PipelineStep(Step):
    name: str
    pipeline: Pipeline
    input_map: dict[str, str]
    output_map: dict[str, str]

    def execute(self, context: dict[str, Media]) -> dict[str, Media]:
        sub_inputs = {}
        for sub_key, parent_key in self.input_map.items():
            if parent_key not in context:
                raise KeyError(
                    f"PipelineStep '{self.name}': input '{parent_key}' not found in context. "
                    f"Available: {list(context.keys())}"
                )
            sub_inputs[sub_key] = context[parent_key]
        sub_result = self.pipeline.run(sub_inputs)
        outputs = {}
        for sub_key, parent_key in self.output_map.items():
            if sub_key in sub_result:
                outputs[parent_key] = sub_result[sub_key]
        return outputs


@dataclass
class Pipeline:
    name: str
    steps: list[Step] = field(default_factory=list)

    def add_step(self, step: Step) -> None:
        self.steps.append(step)

    def run(self, inputs: dict[str, Media]) -> dict[str, Media]:
        context = dict(inputs)
        for step in self.steps:
            outputs = step.execute(context)
            context.update(outputs)
        return context

    def load(self) -> None:
        from casadei.loop import LoopStep
        for step in self.steps:
            if isinstance(step, AgentStep):
                step.agent.load()
            elif isinstance(step, PipelineStep):
                step.pipeline.load()
            elif isinstance(step, LoopStep):
                step.load()

    def unload(self) -> None:
        from casadei.loop import LoopStep
        for step in self.steps:
            if isinstance(step, AgentStep):
                step.agent.unload()
            elif isinstance(step, PipelineStep):
                step.pipeline.unload()
            elif isinstance(step, LoopStep):
                step.unload()

    @classmethod
    def compose(cls, name: str, pipelines: list[Pipeline]) -> Pipeline:
        all_steps = []
        for p in pipelines:
            all_steps.extend(p.steps)
        return cls(name=name, steps=all_steps)
