import pytest
import numpy as np

from casadei.media import VideoMedia, TextMedia, MediaBundle
from casadei.models.base import ModelCapability, TextConstraint, VideoConstraint
from casadei.models.video_edit import VideoEditModel


class TestVideoEditModel:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            VideoEditModel()

    def test_subclass_inherits_defaults(self):
        class MockEditor(VideoEditModel):
            capability = ModelCapability(
                inputs=[
                    VideoConstraint(required=True, max_count=1),
                    TextConstraint(required=True),
                ],
                outputs=[VideoConstraint(required=True, max_count=1)],
            )

            def load_model(self):
                pass

            def unload_model(self):
                pass

            def _edit(self, video_frames, prompt, negative_prompt, **kwargs):
                return video_frames

        editor = MockEditor()
        assert isinstance(editor, VideoEditModel)

    def test_run_delegates_to_edit(self):
        class MockEditor(VideoEditModel):
            capability = ModelCapability(
                inputs=[
                    VideoConstraint(required=True, max_count=1),
                    TextConstraint(required=True),
                ],
                outputs=[VideoConstraint(required=True, max_count=1)],
            )

            def load_model(self):
                pass

            def unload_model(self):
                pass

            def _edit(self, video_frames, prompt, negative_prompt, **kwargs):
                # Return frames with all pixels set to green
                return [np.full_like(f, [0, 128, 0]) for f in video_frames]

        editor = MockEditor()
        input_frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(4)]
        bundle = MediaBundle(items={
            "video": VideoMedia.from_frames(input_frames, fps=16),
            "prompt": TextMedia(text="make it green"),
        })
        result = editor.run(bundle)
        assert "video" in result.items
        output_video = result["video"]
        assert isinstance(output_video, VideoMedia)
        assert output_video.frame_count == 4
        assert output_video.fps == 16

    def test_run_validates_inputs(self):
        class MockEditor(VideoEditModel):
            capability = ModelCapability(
                inputs=[
                    VideoConstraint(required=True, max_count=1),
                    TextConstraint(required=True),
                ],
                outputs=[VideoConstraint(required=True, max_count=1)],
            )

            def load_model(self):
                pass

            def unload_model(self):
                pass

            def _edit(self, video_frames, prompt, negative_prompt, **kwargs):
                return video_frames

        editor = MockEditor()
        input_frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        bundle = MediaBundle(items={
            "video": VideoMedia.from_frames(input_frames),
        })
        with pytest.raises(ValueError, match="[Rr]equired.*[Tt]ext"):
            editor.run(bundle)

    def test_run_requires_frames(self, tmp_path):
        class MockEditor(VideoEditModel):
            capability = ModelCapability(
                inputs=[
                    VideoConstraint(required=True, max_count=1),
                    TextConstraint(required=True),
                ],
                outputs=[VideoConstraint(required=True, max_count=1)],
            )

            def load_model(self):
                pass

            def unload_model(self):
                pass

            def _edit(self, video_frames, prompt, negative_prompt, **kwargs):
                return video_frames

        editor = MockEditor()
        path = tmp_path / "video.mp4"
        path.touch()
        bundle = MediaBundle(items={
            "video": VideoMedia(path=path),
            "prompt": TextMedia(text="edit this"),
        })
        with pytest.raises(ValueError, match="in-memory frames"):
            editor.run(bundle)

    def test_run_with_prompt_and_negative_prompt(self):
        class MockEditor(VideoEditModel):
            capability = ModelCapability(
                inputs=[
                    VideoConstraint(required=True, max_count=1),
                    TextConstraint(required=True, max_count=2),
                ],
                outputs=[VideoConstraint(required=True, max_count=1)],
            )

            def load_model(self):
                pass

            def unload_model(self):
                pass

            def _edit(self, video_frames, prompt, negative_prompt, **kwargs):
                assert prompt == "make it cinematic"
                assert negative_prompt == "blurry"
                return video_frames

        editor = MockEditor()
        input_frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        bundle = MediaBundle(items={
            "video": VideoMedia.from_frames(input_frames),
            "prompt": TextMedia(text="make it cinematic"),
            "negative_prompt": TextMedia(text="blurry"),
        })
        result = editor.run(bundle)
        assert "video" in result.items
