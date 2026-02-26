"""LoopStep — iterative agentic loop as a first-class pipeline Step.

Encapsulates the generate-judge-repair pattern. Runs a sequence of body
steps repeatedly, evaluating a judge callable after each iteration. If
the judge accepts, the loop exits. If max_iterations is reached without
acceptance and a best_fn is provided, it selects the best candidate.

Memory management is configurable: ``swap_models=True`` loads and unloads
each AgentStep around its execution; ``swap_models=False`` keeps all body
agents loaded across iterations (faster, but needs more VRAM).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from casadei.media import Media, TextMedia
from casadei.pipeline import Step, AgentStep

# Judge: receives full context after body steps, returns (accepted, feedback_text)
JudgeCallable = Callable[[dict[str, Media]], tuple[bool, str]]

# BestFn: receives iteration history + full context, returns output dict
BestFn = Callable[[list["LoopIteration"], dict[str, Media]], dict[str, Media]]


@dataclass
class LoopIteration:
    """Record of one complete loop iteration."""

    index: int
    outputs: dict[str, Media]
    accepted: bool
    feedback: str
    duration_ms: float


@dataclass
class LoopResult:
    """Summary of a full loop execution, stored in context."""

    iterations: list[LoopIteration] = field(default_factory=list)
    total_ms: float = 0.0

    def summary(self) -> str:
        lines = [
            f"LoopResult — {len(self.iterations)} iterations, {self.total_ms:.1f}ms total",
        ]
        for it in self.iterations:
            verdict = "PASS" if it.accepted else "FAIL"
            lines.append(
                f"  [{it.index}] {verdict} {it.duration_ms:.1f}ms — {it.feedback[:80]}"
            )
        return "\n".join(lines)


@dataclass
class LoopStep(Step):
    """Agentic generate-judge-repair loop.

    Runs ``body`` steps repeatedly until the ``judge`` returns accepted=True
    or ``max_iterations`` is reached. After each rejection, places the judge's
    feedback into ``context["loop_feedback"]`` (as TextMedia) so body steps
    can reference it.

    Parameters
    ----------
    name
        Step name (shown in logs).
    body
        Steps to run each iteration (AgentStep, CodeStep, etc.).
    judge
        Callable receiving the full context, returning (accepted, feedback).
    max_iterations
        Hard cap on iterations. Default: 5.
    best_fn
        Called when max_iterations exhausted without acceptance. Receives
        the iteration history and context, returns an output dict.
    swap_models
        If True (default), each AgentStep in body is loaded before and
        unloaded after its execution. Safe for limited VRAM.
        If False, all body agents stay loaded across iterations.
    output_key
        Context key holding the current candidate result. Default: "image".
    history_key
        Context key for the LoopResult. Defaults to "{name}_history".
    feedback_template_var
        Name of the template variable to inject feedback into AgentStep
        template_kwargs. Default: "feedback". Set to "" to disable.
    """

    name: str
    body: list[Step]
    judge: JudgeCallable
    max_iterations: int = 5
    best_fn: BestFn | None = None
    swap_models: bool = True
    output_key: str = "image"
    history_key: str = ""
    feedback_template_var: str = "feedback"

    def _resolved_history_key(self) -> str:
        return self.history_key or f"{self.name}_history"

    def _body_agent_steps(self) -> list[AgentStep]:
        return [s for s in self.body if isinstance(s, AgentStep)]

    def _is_loaded(self) -> bool:
        """Check if body agents are already loaded."""
        agents = self._body_agent_steps()
        return bool(agents) and all(s.agent._model is not None for s in agents)

    def load(self) -> None:
        """Pre-load all body agents. Only called when swap_models=False."""
        if not self.swap_models and not self._is_loaded():
            for step in self._body_agent_steps():
                step.agent.load()

    def unload(self) -> None:
        """Unload all body agents."""
        for step in self._body_agent_steps():
            try:
                step.agent.unload()
            except Exception:
                pass

    def _inject_feedback(self, context: dict[str, Media]) -> None:
        """Sync loop_feedback from context into AgentStep template_kwargs."""
        if not self.feedback_template_var:
            return
        feedback_text = ""
        feedback_media = context.get("loop_feedback")
        if isinstance(feedback_media, TextMedia):
            feedback_text = feedback_media.text
        for step in self.body:
            if isinstance(step, AgentStep) and self.feedback_template_var in step.template_kwargs:
                step.template_kwargs[self.feedback_template_var] = feedback_text

    def _run_body_once(self, context: dict[str, Media]) -> dict[str, Media]:
        """Run all body steps once, returning new/changed keys."""
        self._inject_feedback(context)
        working = dict(context)
        for step in self.body:
            if self.swap_models and isinstance(step, AgentStep):
                step.agent.load()
                try:
                    outputs = step.execute(working)
                finally:
                    step.agent.unload()
            else:
                outputs = step.execute(working)
            working.update(outputs)
        return {k: v for k, v in working.items() if k not in context or context[k] is not v}

    def execute(self, context: dict[str, Media]) -> dict[str, Media]:
        history_key = self._resolved_history_key()
        loop_result = LoopResult()
        working = dict(context)
        # Initialize empty feedback for first iteration
        working["loop_feedback"] = TextMedia(text="")
        accepted = False

        if not self.swap_models:
            self.load()

        try:
            for i in range(self.max_iterations):
                is_last = (i == self.max_iterations - 1)

                print(f"\n{'='*60}")
                print(f"[Loop '{self.name}'] Iteration {i + 1}/{self.max_iterations}")
                print(f"{'='*60}")

                iter_start = time.perf_counter()

                # Generate
                body_outputs = self._run_body_once(working)
                working.update(body_outputs)

                body_ms = (time.perf_counter() - iter_start) * 1000
                print(f"  Body steps: {body_ms:.0f}ms")

                # Always judge — we need the verdict and feedback for logging
                print(f"  Judging...")
                accepted, feedback = self.judge(working)

                iter_ms = (time.perf_counter() - iter_start) * 1000
                print(f"  Iteration total: {iter_ms:.0f}ms")

                candidate = working.get(self.output_key)
                iteration_outputs = {self.output_key: candidate} if candidate is not None else {}

                record = LoopIteration(
                    index=i,
                    outputs=dict(iteration_outputs),
                    accepted=accepted,
                    feedback=feedback,
                    duration_ms=iter_ms,
                )
                loop_result.iterations.append(record)

                if accepted:
                    break

                # Only inject feedback for next body run if not the last iteration
                if not is_last:
                    working["loop_feedback"] = TextMedia(text=feedback)

            best_outputs = {}
            if not accepted and self.best_fn is not None:
                print(f"\n  Selecting best candidate from {len(loop_result.iterations)} iterations...")
                best_outputs = self.best_fn(loop_result.iterations, working)
                working.update(best_outputs)
        finally:
            if not self.swap_models:
                self.unload()

        loop_result.total_ms = sum(it.duration_ms for it in loop_result.iterations)

        outputs: dict[str, Media] = {}
        if self.output_key in working:
            outputs[self.output_key] = working[self.output_key]
        outputs[history_key] = loop_result  # type: ignore[assignment]
        outputs["loop_feedback"] = working.get("loop_feedback", TextMedia(text=""))
        # Forward best_fn metadata (best_selection_index, best_selection_reason, etc.)
        for k, v in best_outputs.items():
            if k != self.output_key:
                outputs[k] = v  # type: ignore[assignment]
        return outputs
