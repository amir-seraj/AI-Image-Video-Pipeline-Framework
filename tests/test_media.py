import pytest
from pathlib import Path
from PIL import Image as PILImage

from casadei.media import ImageMedia, TextMedia, VideoMedia, MediaBundle


class TestImageMedia:
    def test_create_from_pil(self):
        img = PILImage.new("RGB", (512, 512), color="red")
        media = ImageMedia(image=img)
        assert media.image.size == (512, 512)
        assert media.format == "png"

    def test_create_with_format(self):
        img = PILImage.new("RGB", (256, 256))
        media = ImageMedia(image=img, format="jpg")
        assert media.format == "jpg"

    def test_save_and_load(self, tmp_path):
        img = PILImage.new("RGB", (100, 100), color="blue")
        media = ImageMedia(image=img)
        path = tmp_path / "test.png"
        media.save(path)
        loaded = ImageMedia.load(path)
        assert loaded.image.size == (100, 100)

    def test_width_height(self):
        img = PILImage.new("RGB", (640, 480))
        media = ImageMedia(image=img)
        assert media.width == 640
        assert media.height == 480


class TestTextMedia:
    def test_create(self):
        media = TextMedia(text="hello world")
        assert media.text == "hello world"

    def test_empty_text(self):
        media = TextMedia(text="")
        assert media.text == ""

    def test_is_complete_plain_text(self):
        media = TextMedia(text="no variables here")
        assert media.is_complete is True

    def test_is_complete_with_unfilled_variables(self):
        media = TextMedia(text="I want to add $item in the image")
        assert media.is_complete is False

    def test_variables_returns_variable_names(self):
        media = TextMedia(text="Add $item near the $location")
        assert media.variables == {"item", "location"}

    def test_variables_empty_for_plain_text(self):
        media = TextMedia(text="plain text")
        assert media.variables == set()

    def test_fill_replaces_variables(self):
        media = TextMedia(text="I want to add $item in the image")
        filled = media.fill(item="a red car")
        assert filled.text == "I want to add a red car in the image"
        assert filled.is_complete is True

    def test_fill_multiple_variables(self):
        media = TextMedia(text="Replace $source with $target")
        filled = media.fill(source="the cat", target="a dog")
        assert filled.text == "Replace the cat with a dog"

    def test_fill_partial_leaves_remaining(self):
        media = TextMedia(text="Add $item near the $location")
        filled = media.fill(item="tree")
        assert filled.text == "Add tree near the $location"
        assert filled.is_complete is False
        assert filled.variables == {"location"}

    def test_fill_returns_new_instance(self):
        media = TextMedia(text="Add $item")
        filled = media.fill(item="tree")
        assert media.text == "Add $item"  # original unchanged
        assert filled.text == "Add tree"

    def test_fill_unknown_variable_ignored(self):
        media = TextMedia(text="Hello $name")
        filled = media.fill(name="world", extra="ignored")
        assert filled.text == "Hello world"


class TestVideoMedia:
    def test_create(self, tmp_path):
        path = tmp_path / "video.mp4"
        path.touch()
        media = VideoMedia(path=path)
        assert media.path == path

    def test_nonexistent_path_raises(self):
        with pytest.raises(ValueError):
            VideoMedia(path=Path("/nonexistent/video.mp4"))


class TestMediaBundle:
    def test_create_bundle(self):
        img = PILImage.new("RGB", (100, 100))
        bundle = MediaBundle(items={
            "image": ImageMedia(image=img),
            "prompt": TextMedia(text="edit this"),
        })
        assert "image" in bundle.items
        assert "prompt" in bundle.items

    def test_getitem(self):
        text = TextMedia(text="hello")
        bundle = MediaBundle(items={"greeting": text})
        assert bundle["greeting"].text == "hello"

    def test_missing_key_raises(self):
        bundle = MediaBundle(items={})
        with pytest.raises(KeyError):
            bundle["missing"]

    def test_bundle_with_multiple_texts(self):
        bundle = MediaBundle(items={
            "prompt": TextMedia(text="make it blue"),
            "negative_prompt": TextMedia(text="blurry, ugly"),
            "caption": TextMedia(text="A beautiful landscape"),
        })
        assert len([v for v in bundle.items.values() if isinstance(v, TextMedia)]) == 3

    def test_bundle_with_multiple_images(self):
        bundle = MediaBundle(items={
            "source": ImageMedia(image=PILImage.new("RGB", (100, 100))),
            "reference": ImageMedia(image=PILImage.new("RGB", (200, 200))),
            "mask": ImageMedia(image=PILImage.new("L", (100, 100))),
        })
        assert len([v for v in bundle.items.values() if isinstance(v, ImageMedia)]) == 3

    def test_bundle_with_template_text(self):
        bundle = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (100, 100))),
            "prompt": TextMedia(text="Add $item to the scene"),
        })
        prompt = bundle["prompt"]
        assert isinstance(prompt, TextMedia)
        assert prompt.is_complete is False
        filled = prompt.fill(item="a sunset")
        assert filled.is_complete is True
