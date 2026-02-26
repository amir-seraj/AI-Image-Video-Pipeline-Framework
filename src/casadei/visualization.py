"""Pipeline visualization — render pipeline structure as Mermaid diagrams."""

from __future__ import annotations

from casadei.pipeline import Pipeline, Step, AgentStep, CodeStep, PipelineStep
from casadei.loop import LoopStep


def _step_shape(step: Step) -> tuple[str, str]:
    """Return the Mermaid node delimiters based on step type."""
    if isinstance(step, AgentStep):
        return "[", "]"
    elif isinstance(step, CodeStep):
        return "{{", "}}"
    elif isinstance(step, PipelineStep):
        return "[[", "]]"
    elif isinstance(step, LoopStep):
        return "(", ")"
    return "[", "]"


def _step_label(step: Step) -> str:
    """Return the display label for a step node."""
    if isinstance(step, AgentStep):
        return f"{step.name}\\n(Agent)"
    elif isinstance(step, CodeStep):
        return f"{step.name}\\n(Code)"
    elif isinstance(step, PipelineStep):
        return f"{step.name}\\n(Pipeline: {step.pipeline.name})"
    elif isinstance(step, LoopStep):
        return f"{step.name}\\n(Loop x{step.max_iterations})"
    return step.name


def _sanitize_id(name: str) -> str:
    """Sanitize a name for use as a Mermaid node identifier."""
    return name.replace(" ", "_").replace("-", "_")


def to_mermaid(
    pipeline: Pipeline,
    expand_nested: bool = False,
    direction: str = "TD",
) -> str:
    """Render a Pipeline as a Mermaid flowchart string.

    Parameters
    ----------
    pipeline:
        The pipeline to visualize.
    expand_nested:
        If ``True``, nested :class:`PipelineStep` instances are expanded
        into Mermaid ``subgraph`` blocks showing their internal steps.
    direction:
        Mermaid flowchart direction (``"TD"``, ``"LR"``, etc.).

    Returns
    -------
    str
        A complete Mermaid flowchart definition.
    """
    lines: list[str] = [f"flowchart {direction}"]
    _render_pipeline(pipeline, lines, expand_nested=expand_nested, indent="    ")
    return "\n".join(lines)


def _render_pipeline(
    pipeline: Pipeline,
    lines: list[str],
    expand_nested: bool,
    indent: str,
    prefix: str = "",
) -> None:
    """Recursively render pipeline steps into Mermaid lines."""
    steps = pipeline.steps
    if not steps:
        return

    prev_output_nodes: list[str] = []

    for i, step in enumerate(steps):
        step_id = _sanitize_id(f"{prefix}{step.name}")
        left, right = _step_shape(step)
        label = _step_label(step)

        if isinstance(step, PipelineStep) and expand_nested:
            # Render as a subgraph containing the inner pipeline's steps.
            lines.append(f"{indent}subgraph {step_id}_sub [{step.pipeline.name}]")
            _render_pipeline(
                step.pipeline,
                lines,
                expand_nested=True,
                indent=indent + "    ",
                prefix=f"{step_id}_",
            )
            lines.append(f"{indent}end")

            # Connect previous outputs to the first inner step.
            if prev_output_nodes:
                inner_first = step.pipeline.steps[0] if step.pipeline.steps else None
                if inner_first:
                    inner_id = _sanitize_id(f"{step_id}_{inner_first.name}")
                    for pn in prev_output_nodes:
                        lines.append(f"{indent}{pn} --> {inner_id}")

            # Track the last inner step as the output node.
            if step.pipeline.steps:
                last_inner = step.pipeline.steps[-1]
                prev_output_nodes = [_sanitize_id(f"{step_id}_{last_inner.name}")]
            else:
                prev_output_nodes = []
        else:
            # Render the step as a regular node.
            lines.append(f'{indent}{step_id}{left}"{label}"{right}')

            # For the first step, render its input data nodes.
            if i == 0 and isinstance(step, (AgentStep, PipelineStep)):
                input_map = step.input_map if hasattr(step, "input_map") else {}
                for _agent_key, ctx_key in input_map.items():
                    data_id = _sanitize_id(f"{prefix}data_{ctx_key}")
                    lines.append(f"{indent}{data_id}(({ctx_key}))")
                    lines.append(f"{indent}{data_id} --> {step_id}")

            # Connect from previous output nodes.
            if prev_output_nodes:
                for pn in prev_output_nodes:
                    lines.append(f"{indent}{pn} --> {step_id}")

            # Render output data nodes and update tracking.
            if isinstance(step, (AgentStep, PipelineStep)):
                output_map = step.output_map if hasattr(step, "output_map") else {}
                output_nodes: list[str] = []
                for _agent_key, ctx_key in output_map.items():
                    data_id = _sanitize_id(f"{prefix}data_{ctx_key}")
                    lines.append(f"{indent}{data_id}(({ctx_key}))")
                    lines.append(f"{indent}{step_id} --> {data_id}")
                    output_nodes.append(data_id)
                prev_output_nodes = output_nodes if output_nodes else [step_id]
            else:
                prev_output_nodes = [step_id]
