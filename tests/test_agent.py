# tests/test_agent.py
import pytest
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch
from PIL import Image as PILImage

from casadei.agent import Agent, AgentConfig, load_agent, save_agent
from casadei.media import ImageMedia, TextMedia, MediaBundle


class TestAgentConfig:
    def test_create_config(self):
        config = AgentConfig(
            name="bg_remover",
            model="qwen_image_edit",
            prompt_template="Remove the background from this image",
            params={"num_inference_steps": 30},
        )
        assert config.name == "bg_remover"
        assert config.model == "qwen_image_edit"

    def test_config_defaults(self):
        config = AgentConfig(name="test", model="qwen_image_edit")
        assert config.prompt_template == ""
        assert config.negative_prompt == ""
        assert config.params == {}
        assert config.description == ""

    def test_serialize_to_dict(self):
        config = AgentConfig(
            name="test",
            model="qwen_image_edit",
            prompt_template="do something",
        )
        d = config.model_dump()
        assert d["name"] == "test"
        assert d["model"] == "qwen_image_edit"

    def test_deserialize_from_dict(self):
        d = {
            "name": "test",
            "model": "qwen_image_edit",
            "prompt_template": "do something",
        }
        config = AgentConfig(**d)
        assert config.name == "test"

    def test_template_with_dollar_variables(self):
        config = AgentConfig(
            name="adder",
            model="qwen_image_edit",
            prompt_template="I want to add $item in the image",
        )
        assert config.prompt_template == "I want to add $item in the image"


class TestAgent:
    @patch("casadei.agent.default_registry")
    def test_create_agent_from_config(self, mock_registry):
        mock_model_cls = MagicMock()
        mock_registry.get.return_value = mock_model_cls

        config = AgentConfig(name="test", model="qwen_image_edit")
        agent = Agent(config=config)
        assert agent.config.name == "test"

    @patch("casadei.agent.default_registry")
    def test_agent_fills_template_variables(self, mock_registry):
        mock_model = MagicMock()
        mock_model.run.return_value = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100))),
        })
        mock_model_cls = MagicMock(return_value=mock_model)
        mock_registry.get.return_value = mock_model_cls

        config = AgentConfig(
            name="adder",
            model="qwen_image_edit",
            prompt_template="I want to add $item in the image",
        )
        agent = Agent(config=config)
        agent.load()

        bundle = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100))),
        })
        result = agent.execute(bundle, item="a red car")

        call_args = mock_model.run.call_args[0][0]
        prompt_items = [v for v in call_args.items.values() if isinstance(v, TextMedia)]
        assert any("a red car" in p.text for p in prompt_items)
        assert all("$item" not in p.text for p in prompt_items)

    @patch("casadei.agent.default_registry")
    def test_agent_passes_raw_prompt_when_no_template(self, mock_registry):
        mock_model = MagicMock()
        mock_model.run.return_value = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100))),
        })
        mock_model_cls = MagicMock(return_value=mock_model)
        mock_registry.get.return_value = mock_model_cls

        config = AgentConfig(name="raw", model="qwen_image_edit")
        agent = Agent(config=config)
        agent.load()

        bundle = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100))),
            "prompt": TextMedia(text="do this specific thing"),
        })
        result = agent.execute(bundle)

        call_args = mock_model.run.call_args[0][0]
        prompt_items = [v for v in call_args.items.values() if isinstance(v, TextMedia)]
        assert any("do this specific thing" in p.text for p in prompt_items)

    @patch("casadei.agent.default_registry")
    def test_agent_incomplete_template_raises(self, mock_registry):
        mock_model = MagicMock()
        mock_model_cls = MagicMock(return_value=mock_model)
        mock_registry.get.return_value = mock_model_cls

        config = AgentConfig(
            name="needs_two_vars",
            model="qwen_image_edit",
            prompt_template="Replace $source with $target",
        )
        agent = Agent(config=config)
        agent.load()

        bundle = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100))),
        })
        with pytest.raises(ValueError, match="unfilled"):
            agent.execute(bundle, source="cat")


class TestAgentPersistence:
    def test_save_and_load(self, tmp_path):
        config = AgentConfig(
            name="bg_remover",
            model="qwen_image_edit",
            description="Removes backgrounds",
            prompt_template="Remove the background, leaving only the subject",
            negative_prompt="blurry",
            params={"num_inference_steps": 30},
        )
        filepath = tmp_path / "bg_remover.yaml"
        save_agent(config, filepath)

        loaded = load_agent(filepath)
        assert loaded.name == "bg_remover"
        assert loaded.model == "qwen_image_edit"
        assert loaded.prompt_template == "Remove the background, leaving only the subject"
        assert loaded.params["num_inference_steps"] == 30

    def test_load_from_agents_directory(self, tmp_path):
        config = AgentConfig(name="test_agent", model="qwen_image_edit")
        save_agent(config, tmp_path / "test_agent.yaml")
        loaded = load_agent(tmp_path / "test_agent.yaml")
        assert loaded.name == "test_agent"

    def test_saved_file_is_valid_yaml(self, tmp_path):
        config = AgentConfig(
            name="test",
            model="qwen_image_edit",
            prompt_template="Add $item to the scene",
        )
        filepath = tmp_path / "test.yaml"
        save_agent(config, filepath)

        with open(filepath) as f:
            data = yaml.safe_load(f)
        assert data["name"] == "test"
        assert data["prompt_template"] == "Add $item to the scene"
