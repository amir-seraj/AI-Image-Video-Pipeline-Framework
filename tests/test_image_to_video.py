import pytest
import numpy as np
from PIL import Image as PILImage

from casadei.media import ImageMedia, TextMedia, VideoMedia, MediaBundle
from casadei.models.base import ModelCapability, ImageConstraint, TextConstraint, VideoConstraint
from casadei.models.image_to_video import ImageToVideoModel


class TestImageToVideoModel:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            ImageToVideoModel()

    def test_subclass_inherits_defaults(self):
        class MockGenerator(ImageToVideoModel):
            capability = ModelCapability(
                inputs=[
                    ImageConstraint(required=True, max_count=1),
                    TextConstraint(required=True),
                ],
                outputs=[VideoConstraint(required=True, max_count=1)],
            )

            def load_model(self):
                pass

            def unload_model(self):
                pass

            def _generate(self, image, prompt, negative_prompt, **kwargs):
                return [np.zeros((100, 100, 3), dtype=np.uint8)]

        gen = MockGenerator()
        assert isinstance(gen, ImageToVideoModel)

    def test_run_delegates_to_generate(self):
        class MockGenerator(ImageToVideoModel):
            capability = ModelCapability(
                inputs=[
                    ImageConstraint(required=True, max_count=1),
                    TextConstraint(required=True),
                ],
                outputs=[VideoConstraint(required=True, max_count=1)],
            )

            def load_model(self):
                pass

            def unload_model(self):
                pass

            def _generate(self, image, prompt, negative_prompt, **kwargs):
                return [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(8)]

        gen = MockGenerator()
        bundle = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100), color="red")),
            "prompt": TextMedia(text="animate this image"),
        })
        result = gen.run(bundle)
        assert "video" in result.items
        output_video = result["video"]
        assert isinstance(output_video, VideoMedia)
        assert output_video.frame_count == 8

    def test_run_validates_inputs(self):
        class MockGenerator(ImageToVideoModel):
            capability = ModelCapability(
                inputs=[
                    ImageConstraint(required=True, max_count=1),
                    TextConstraint(required=True),
                ],
                outputs=[VideoConstraint(required=True, max_count=1)],
            )

            def load_model(self):
                pass

            def unload_model(self):
                pass

            def _generate(self, image, prompt, negative_prompt, **kwargs):
                return [np.zeros((100, 100, 3), dtype=np.uint8)]

        gen = MockGenerator()
        bundle = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100))),
        })
        with pytest.raises(ValueError, match="[Rr]equired.*[Tt]ext"):
            gen.run(bundle)

    def test_run_with_prompt_and_negative_prompt(self):
        class MockGenerator(ImageToVideoModel):
            capability = ModelCapability(
                inputs=[
                    ImageConstraint(required=True, max_count=1),
                    TextConstraint(required=True, max_count=2),
                ],
                outputs=[VideoConstraint(required=True, max_count=1)],
            )

            def load_model(self):
                pass

            def unload_model(self):
                pass

            def _generate(self, image, prompt, negative_prompt, **kwargs):
                assert prompt == "make it dance"
                assert negative_prompt == "static, blurry"
                return [np.zeros((100, 100, 3), dtype=np.uint8)]

        gen = MockGenerator()
        bundle = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100))),
            "prompt": TextMedia(text="make it dance"),
            "negative_prompt": TextMedia(text="static, blurry"),
        })
        result = gen.run(bundle)
        assert "video" in result.items
