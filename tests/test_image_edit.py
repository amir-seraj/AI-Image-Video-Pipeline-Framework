# tests/test_image_edit.py
import pytest
from PIL import Image as PILImage

from casadei.media import ImageMedia, TextMedia, MediaBundle
from casadei.models.base import ModelCapability, ImageConstraint, TextConstraint
from casadei.models.image_edit import ImageEditModel


class TestImageEditModel:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            ImageEditModel()

    def test_subclass_inherits_defaults(self):
        class MockEditor(ImageEditModel):
            capability = ModelCapability(
                inputs=[
                    ImageConstraint(required=True, max_count=1),
                    TextConstraint(required=True),
                ],
                outputs=[ImageConstraint(required=True, max_count=1)],
            )

            def load_model(self):
                pass

            def unload_model(self):
                pass

            def _edit(self, images, prompt, negative_prompt, **kwargs):
                return images[0]

        editor = MockEditor()
        assert isinstance(editor, ImageEditModel)

    def test_run_delegates_to_edit(self):
        class MockEditor(ImageEditModel):
            capability = ModelCapability(
                inputs=[
                    ImageConstraint(required=True, max_count=1),
                    TextConstraint(required=True),
                ],
                outputs=[ImageConstraint(required=True, max_count=1)],
            )

            def load_model(self):
                pass

            def unload_model(self):
                pass

            def _edit(self, images, prompt, negative_prompt, **kwargs):
                return PILImage.new("RGB", (100, 100), color="green")

        editor = MockEditor()
        bundle = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100), color="red")),
            "prompt": TextMedia(text="make it green"),
        })
        result = editor.run(bundle)
        assert "image" in result.items
        output_img = result["image"]
        assert isinstance(output_img, ImageMedia)
        assert output_img.image.getpixel((50, 50)) == (0, 128, 0)

    def test_run_validates_inputs(self):
        class MockEditor(ImageEditModel):
            capability = ModelCapability(
                inputs=[
                    ImageConstraint(required=True, max_count=1),
                    TextConstraint(required=True),
                ],
                outputs=[ImageConstraint(required=True, max_count=1)],
            )

            def load_model(self):
                pass

            def unload_model(self):
                pass

            def _edit(self, images, prompt, negative_prompt, **kwargs):
                return images[0]

        editor = MockEditor()
        bundle = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100))),
        })
        with pytest.raises(ValueError, match="[Rr]equired.*[Tt]ext"):
            editor.run(bundle)

    def test_run_with_multiple_text_inputs(self):
        """prompt and negative_prompt are both TextMedia in the bundle."""
        class MockEditor(ImageEditModel):
            capability = ModelCapability(
                inputs=[
                    ImageConstraint(required=True, max_count=1),
                    TextConstraint(required=True, max_count=2),
                ],
                outputs=[ImageConstraint(required=True, max_count=1)],
            )

            def load_model(self):
                pass

            def unload_model(self):
                pass

            def _edit(self, images, prompt, negative_prompt, **kwargs):
                assert prompt == "make it green"
                assert negative_prompt == "blurry"
                return PILImage.new("RGB", (100, 100), color="green")

        editor = MockEditor()
        bundle = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100))),
            "prompt": TextMedia(text="make it green"),
            "negative_prompt": TextMedia(text="blurry"),
        })
        result = editor.run(bundle)
        assert "image" in result.items
