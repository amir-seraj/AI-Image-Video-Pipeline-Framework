"""Tests for GeminiFlash VLM provider."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage

from casadei.media import ImageMedia, TextMedia, MediaBundle
from casadei.models.vision_language import VisionLanguageModel
from casadei.models.base import ImageConstraint, TextConstraint


class TestGeminiFlashClass:
    def test_is_vision_language_model(self):
        from casadei.providers.gemini_flash import GeminiFlash
        assert issubclass(GeminiFlash, VisionLanguageModel)

    def test_capability_requires_text(self):
        from casadei.providers.gemini_flash import GeminiFlash
        cap = GeminiFlash.capability
        text_inputs = [c for c in cap.inputs if isinstance(c, TextConstraint)]
        assert len(text_inputs) == 1
        assert text_inputs[0].required is True

    def test_capability_images_optional(self):
        from casadei.providers.gemini_flash import GeminiFlash
        cap = GeminiFlash.capability
        img_inputs = [c for c in cap.inputs if isinstance(c, ImageConstraint)]
        assert len(img_inputs) == 1
        assert img_inputs[0].required is False
        assert img_inputs[0].max_count == 14

    def test_capability_outputs_text(self):
        from casadei.providers.gemini_flash import GeminiFlash
        cap = GeminiFlash.capability
        assert len(cap.outputs) == 1
        assert isinstance(cap.outputs[0], TextConstraint)


class TestGeminiFlashLoadUnload:
    def test_load_model_creates_client(self):
        from casadei.providers.gemini_flash import GeminiFlash

        with patch("casadei.providers.gemini_flash.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client

            provider = GeminiFlash()
            provider.load_model()

            mock_genai.Client.assert_called_once()
            assert provider._client is mock_client

    def test_unload_model_clears_client(self):
        from casadei.providers.gemini_flash import GeminiFlash

        with patch("casadei.providers.gemini_flash.genai"):
            provider = GeminiFlash()
            provider._client = MagicMock()
            provider.unload_model()
            assert provider._client is None

    def test_generate_text_raises_if_not_loaded(self):
        from casadei.providers.gemini_flash import GeminiFlash

        provider = GeminiFlash()
        with pytest.raises(RuntimeError, match="not loaded"):
            provider._generate_text(images=[], prompt="hello")

    def test_streaming_raises_if_not_loaded(self):
        from casadei.providers.gemini_flash import GeminiFlash

        provider = GeminiFlash()
        with pytest.raises(RuntimeError, match="not loaded"):
            list(provider._generate_text_streaming(images=[], prompt="hello"))


class TestGeminiFlashGenerateText:
    def test_generate_text_returns_string(self):
        from casadei.providers.gemini_flash import GeminiFlash

        mock_response = MagicMock()
        mock_response.text = "This is a shoe."

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch("casadei.providers.gemini_flash.genai"):
            provider = GeminiFlash()
            provider._client = mock_client

            result = provider._generate_text(images=[], prompt="What is this?")

        assert result == "This is a shoe."

    def test_generate_text_passes_prompt_in_contents(self):
        from casadei.providers.gemini_flash import GeminiFlash

        mock_response = MagicMock()
        mock_response.text = "ok"

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        img = PILImage.new("RGB", (64, 64), color="red")

        with patch("casadei.providers.gemini_flash.genai"):
            provider = GeminiFlash()
            provider._client = mock_client

            provider._generate_text(images=[img], prompt="Describe this image")

        call_kwargs = mock_client.models.generate_content.call_args
        contents = call_kwargs.kwargs["contents"]
        assert "Describe this image" in contents
        assert img in contents

    def test_generate_text_works_without_images(self):
        from casadei.providers.gemini_flash import GeminiFlash

        mock_response = MagicMock()
        mock_response.text = "Text only answer."

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch("casadei.providers.gemini_flash.genai"):
            provider = GeminiFlash()
            provider._client = mock_client

            result = provider._generate_text(images=[], prompt="What is 2+2?")

        assert result == "Text only answer."
        call_kwargs = mock_client.models.generate_content.call_args
        contents = call_kwargs.kwargs["contents"]
        assert contents == ["What is 2+2?"]

    def test_generate_text_handles_none_response_text(self):
        from casadei.providers.gemini_flash import GeminiFlash

        mock_response = MagicMock()
        mock_response.text = None

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch("casadei.providers.gemini_flash.genai"):
            provider = GeminiFlash()
            provider._client = mock_client

            result = provider._generate_text(images=[], prompt="test")

        assert result == ""


class TestGeminiFlashStreaming:
    def test_streaming_yields_chunks(self):
        from casadei.providers.gemini_flash import GeminiFlash

        chunk1 = MagicMock()
        chunk1.text = "Hello"
        chunk2 = MagicMock()
        chunk2.text = " world"
        chunk3 = MagicMock()
        chunk3.text = None  # empty chunk should be skipped

        mock_client = MagicMock()
        mock_client.models.generate_content_stream.return_value = iter(
            [chunk1, chunk2, chunk3]
        )

        with patch("casadei.providers.gemini_flash.genai"):
            provider = GeminiFlash()
            provider._client = mock_client

            chunks = list(
                provider._generate_text_streaming(images=[], prompt="hello")
            )

        assert chunks == ["Hello", " world"]

    def test_streaming_passes_images_and_prompt(self):
        from casadei.providers.gemini_flash import GeminiFlash

        chunk = MagicMock()
        chunk.text = "ok"

        img = PILImage.new("RGB", (32, 32))
        mock_client = MagicMock()
        mock_client.models.generate_content_stream.return_value = iter([chunk])

        with patch("casadei.providers.gemini_flash.genai"):
            provider = GeminiFlash()
            provider._client = mock_client

            list(provider._generate_text_streaming(images=[img], prompt="describe"))

        call_kwargs = mock_client.models.generate_content_stream.call_args
        contents = call_kwargs.kwargs["contents"]
        assert "describe" in contents
        assert img in contents


class TestGeminiFlashRunIntegration:
    def test_run_returns_text_media_bundle(self):
        from casadei.providers.gemini_flash import GeminiFlash
        from casadei.media import TextMedia

        mock_response = MagicMock()
        mock_response.text = "A red square."

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        with patch("casadei.providers.gemini_flash.genai"):
            provider = GeminiFlash()
            provider._client = mock_client

            bundle = MediaBundle(items={
                "image": ImageMedia(image=PILImage.new("RGB", (64, 64), "red")),
                "prompt": TextMedia(text="What color is this?"),
            })
            result = provider.run(bundle)

        assert "text" in result.items
        assert isinstance(result.items["text"], TextMedia)
        assert result.items["text"].text == "A red square."
