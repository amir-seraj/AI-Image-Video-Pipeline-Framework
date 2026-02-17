"""Tests for model base classes and capability system."""

import pytest
from PIL import Image as PILImage

from casadei.media import ImageMedia, TextMedia, MediaBundle
from casadei.models.base import (
    AIModel,
    ModelCapability,
    ImageConstraint,
    TextConstraint,
    VideoConstraint,
)


class TestModelCapability:
    def test_create_capability(self):
        cap = ModelCapability(
            inputs=[
                ImageConstraint(required=True, max_count=2),
                TextConstraint(required=True),
            ],
            outputs=[
                ImageConstraint(required=True, max_count=1),
            ],
        )
        assert len(cap.inputs) == 2
        assert len(cap.outputs) == 1

    def test_image_constraint_defaults(self):
        c = ImageConstraint()
        assert c.required is True
        assert c.max_count == 1
        assert c.max_width is None
        assert c.max_height is None
        assert "png" in c.supported_formats

    def test_text_constraint_defaults(self):
        c = TextConstraint()
        assert c.required is True
        assert c.max_length is None

    def test_validate_inputs_valid(self):
        cap = ModelCapability(
            inputs=[
                ImageConstraint(required=True, max_count=1),
                TextConstraint(required=True),
            ],
            outputs=[ImageConstraint()],
        )
        bundle = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100))),
            "prompt": TextMedia(text="edit this"),
        })
        errors = cap.validate_inputs(bundle)
        assert errors == []

    def test_validate_inputs_missing_required(self):
        cap = ModelCapability(
            inputs=[
                ImageConstraint(required=True, max_count=1),
                TextConstraint(required=True),
            ],
            outputs=[ImageConstraint()],
        )
        bundle = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100))),
        })
        errors = cap.validate_inputs(bundle)
        assert len(errors) > 0
        assert "TextMedia" in errors[0] or "text" in errors[0].lower()

    def test_validate_inputs_too_many(self):
        cap = ModelCapability(
            inputs=[ImageConstraint(required=True, max_count=1)],
            outputs=[ImageConstraint()],
        )
        bundle = MediaBundle(items={
            "img1": ImageMedia(image=PILImage.new("RGB", (100, 100))),
            "img2": ImageMedia(image=PILImage.new("RGB", (100, 100))),
        })
        errors = cap.validate_inputs(bundle)
        assert len(errors) > 0
        assert "too many" in errors[0].lower() or "max" in errors[0].lower()

    def test_validate_multiple_text_inputs_allowed(self):
        cap = ModelCapability(
            inputs=[TextConstraint(required=True, max_count=3)],
            outputs=[TextConstraint()],
        )
        bundle = MediaBundle(items={
            "prompt": TextMedia(text="do this"),
            "negative": TextMedia(text="not this"),
            "style": TextMedia(text="like this"),
        })
        errors = cap.validate_inputs(bundle)
        assert errors == []


class TestAIModel:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            AIModel()

    def test_subclass_must_define_capability(self):
        class BadModel(AIModel):
            capability = None

            def load_model(self):
                pass

            def unload_model(self):
                pass

            def run(self, inputs):
                return inputs

        with pytest.raises(TypeError):
            BadModel()

    def test_valid_subclass(self):
        class GoodModel(AIModel):
            capability = ModelCapability(
                inputs=[TextConstraint(required=True)],
                outputs=[TextConstraint()],
            )

            def load_model(self):
                self._loaded = True

            def unload_model(self):
                self._loaded = False

            def run(self, inputs: MediaBundle) -> MediaBundle:
                return inputs

        model = GoodModel()
        assert model.capability is not None
