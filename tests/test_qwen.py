# tests/test_qwen.py
import pytest
import torch
from unittest.mock import MagicMock, patch
from PIL import Image as PILImage

from casadei.media import ImageMedia, TextMedia, MediaBundle
from casadei.models.base import ImageConstraint, TextConstraint
from casadei.providers.qwen_image_edit import QwenImageEdit


class TestQwenImageEditCapability:
    def test_accepts_up_to_2_images(self):
        img_constraints = [
            c for c in QwenImageEdit.capability.inputs
            if isinstance(c, ImageConstraint)
        ]
        assert len(img_constraints) == 1
        assert img_constraints[0].max_count == 2

    def test_requires_text_prompt(self):
        txt_constraints = [
            c for c in QwenImageEdit.capability.inputs
            if isinstance(c, TextConstraint)
        ]
        assert len(txt_constraints) >= 1
        assert txt_constraints[0].required is True

    def test_outputs_single_image(self):
        img_constraints = [
            c for c in QwenImageEdit.capability.outputs
            if isinstance(c, ImageConstraint)
        ]
        assert len(img_constraints) == 1
        assert img_constraints[0].max_count == 1

    def test_is_image_edit_model(self):
        from casadei.models.image_edit import ImageEditModel
        assert issubclass(QwenImageEdit, ImageEditModel)


class TestQwenImageEditInference:
    @patch("casadei.providers.qwen_image_edit.QwenImageEditPlusPipeline")
    def test_load_model(self, mock_pipeline_cls):
        mock_pipe = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe

        model = QwenImageEdit()
        model.load_model()

        mock_pipeline_cls.from_pretrained.assert_called_once()
        if torch.cuda.is_available():
            mock_pipe.to.assert_called_once_with("cuda")
        else:
            mock_pipe.to.assert_not_called()

    @patch("casadei.providers.qwen_image_edit.QwenImageEditPlusPipeline")
    def test_edit_calls_pipeline(self, mock_pipeline_cls):
        fake_output_img = PILImage.new("RGB", (512, 512), color="green")
        mock_pipe = MagicMock()
        mock_pipe.return_value.images = [fake_output_img]
        # .to("cuda") returns the same pipe object
        mock_pipe.to.return_value = mock_pipe
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe

        model = QwenImageEdit()
        model.load_model()

        input_img = PILImage.new("RGB", (512, 512), color="red")
        result = model._edit(
            images=[input_img],
            prompt="make it green",
            negative_prompt=" ",
        )
        assert result.size == (512, 512)
        mock_pipe.assert_called_once()

    @patch("casadei.providers.qwen_image_edit.QwenImageEditPlusPipeline")
    def test_run_end_to_end(self, mock_pipeline_cls):
        fake_output_img = PILImage.new("RGB", (512, 512), color="blue")
        mock_pipe = MagicMock()
        mock_pipe.return_value.images = [fake_output_img]
        mock_pipe.to.return_value = mock_pipe
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe

        model = QwenImageEdit()
        model.load_model()

        bundle = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (512, 512))),
            "prompt": TextMedia(text="make it blue"),
        })
        result = model.run(bundle)
        assert "image" in result.items
        assert isinstance(result["image"], ImageMedia)

    @patch("casadei.providers.qwen_image_edit.QwenImageEditPlusPipeline")
    def test_unload_model(self, mock_pipeline_cls):
        mock_pipe = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe


        model = QwenImageEdit()
        model.load_model()
        model.unload_model()
        assert model._pipeline is None

    def test_edit_without_load_raises(self):
        model = QwenImageEdit()
        with pytest.raises(RuntimeError, match="[Nn]ot loaded"):
            model._edit(
                images=[PILImage.new("RGB", (100, 100))],
                prompt="test",
                negative_prompt="",
            )
