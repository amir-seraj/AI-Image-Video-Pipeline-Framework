"""Agent system — configured model instances for reuse in pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from casadei.media import MediaBundle, TextMedia
from casadei.models.registry import default_registry


class AgentConfig(BaseModel):
    """Serializable configuration for an agent.
    prompt_template supports $variable syntax (Python string.Template).
    Variables are filled at execution time via keyword arguments.
    """
    name: str
    model: str
    description: str = ""
    prompt_template: str = ""
    negative_prompt: str = ""
    params: dict[str, Any] = {}


class Agent:
    """A configured model instance ready for use in pipelines.
    Uses TextMedia's $variable template system for prompt filling.
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self._model_cls = default_registry.get(config.model)
        self._model = None

    def load(self) -> None:
        self._model = self._model_cls()
        self._model.load_model()

    def unload(self) -> None:
        if self._model is not None:
            self._model.unload_model()
            self._model = None

    def execute(self, inputs: MediaBundle, **template_kwargs: Any) -> MediaBundle:
        if self._model is None:
            raise RuntimeError("Agent not loaded. Call load() first.")
        prepared = self._prepare_inputs(inputs, **template_kwargs)
        return self._model.run(prepared)

    def _prepare_inputs(self, inputs: MediaBundle, **template_kwargs: Any) -> MediaBundle:
        items = dict(inputs.items)

        if self.config.prompt_template:
            prompt = TextMedia(text=self.config.prompt_template)
            filled = prompt.fill(**template_kwargs)
            if not filled.is_complete:
                unfilled = filled.variables
                raise ValueError(
                    f"Prompt template has unfilled variables: {unfilled}. "
                    f"Provide them as keyword arguments to execute()."
                )
            items["prompt"] = filled

        if self.config.negative_prompt and "negative_prompt" not in items:
            items["negative_prompt"] = TextMedia(text=self.config.negative_prompt)

        return MediaBundle(items=items)


def save_agent(config: AgentConfig, path: Path) -> None:
    with open(path, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)


def load_agent(path: Path) -> AgentConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return AgentConfig(**data)
