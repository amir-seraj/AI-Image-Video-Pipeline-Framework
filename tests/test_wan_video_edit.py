import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from casadei.media import VideoMedia, TextMedia, MediaBundle
from casadei.models.base import TextConstraint, VideoConstraint
from casadei.providers.wan_video_edit import WanVideoEdit


class TestWanVideoEditCapability:
    def test_accepts_single_video(self):
        vid_constraints = [
            c for c in WanVideoEdit.capability.inputs
            if isinstance(c, VideoConstraint)
        ]
        assert len(vid_constraints) == 1
        assert vid_constraints[0].max_count == 1

    def test_requires_text_prompt(self):
        txt_constraints = [
            c for c in WanVideoEdit.capability.inputs
            if isinstance(c, TextConstraint)
        ]
        assert len(txt_constraints) >= 1
        assert txt_constraints[0].required is True

    def test_outputs_single_video(self):
        vid_constraints = [
            c for c in WanVideoEdit.capability.outputs
            if isinstance(c, VideoConstraint)
        ]
        assert len(vid_constraints) == 1
        assert vid_constraints[0].max_count == 1

    def test_is_video_edit_model(self):
        from casadei.models.video_edit import VideoEditModel
        assert issubclass(WanVideoEdit, VideoEditModel)


class TestWanVideoEditInference:
    @patch("casadei.providers.wan_video_edit.UniPCMultistepScheduler")
    @patch("casadei.providers.wan_video_edit.AutoencoderKLWan")
    @patch("casadei.providers.wan_video_edit.WanVideoToVideoPipeline")
    def test_load_model(self, mock_pipeline_cls, mock_vae_cls, mock_sched_cls):
        mock_pipe = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe
        mock_vae_cls.from_pretrained.return_value = MagicMock()
        mock_sched_cls.from_config.return_value = MagicMock()

        model = WanVideoEdit()
        model.load_model()

        mock_pipeline_cls.from_pretrained.assert_called_once()
        mock_vae_cls.from_pretrained.assert_called_once()

    @patch("casadei.providers.wan_video_edit.UniPCMultistepScheduler")
    @patch("casadei.providers.wan_video_edit.AutoencoderKLWan")
    @patch("casadei.providers.wan_video_edit.WanVideoToVideoPipeline")
    def test_edit_calls_pipeline(self, mock_pipeline_cls, mock_vae_cls, mock_sched_cls):
        fake_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(4)]
        mock_pipe = MagicMock()
        mock_pipe.return_value.frames = [fake_frames]
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe
        mock_vae_cls.from_pretrained.return_value = MagicMock()
        mock_sched_cls.from_config.return_value = MagicMock()

        model = WanVideoEdit()
        model.load_model()

        input_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(4)]
        result = model._edit(
            video_frames=input_frames,
            prompt="make it cinematic",
            negative_prompt="",
        )
        assert len(result) == 4
        mock_pipe.assert_called_once()

    @patch("casadei.providers.wan_video_edit.UniPCMultistepScheduler")
    @patch("casadei.providers.wan_video_edit.AutoencoderKLWan")
    @patch("casadei.providers.wan_video_edit.WanVideoToVideoPipeline")
    def test_run_end_to_end(self, mock_pipeline_cls, mock_vae_cls, mock_sched_cls):
        fake_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(4)]
        mock_pipe = MagicMock()
        mock_pipe.return_value.frames = [fake_frames]
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe
        mock_vae_cls.from_pretrained.return_value = MagicMock()
        mock_sched_cls.from_config.return_value = MagicMock()

        model = WanVideoEdit()
        model.load_model()

        input_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(4)]
        bundle = MediaBundle(items={
            "video": VideoMedia.from_frames(input_frames, fps=16),
            "prompt": TextMedia(text="make it look like watercolor"),
        })
        result = model.run(bundle)
        assert "video" in result.items
        assert isinstance(result["video"], VideoMedia)
        assert result["video"].frame_count == 4

    @patch("casadei.providers.wan_video_edit.UniPCMultistepScheduler")
    @patch("casadei.providers.wan_video_edit.AutoencoderKLWan")
    @patch("casadei.providers.wan_video_edit.WanVideoToVideoPipeline")
    def test_unload_model(self, mock_pipeline_cls, mock_vae_cls, mock_sched_cls):
        mock_pipe = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe
        mock_vae_cls.from_pretrained.return_value = MagicMock()
        mock_sched_cls.from_config.return_value = MagicMock()

        model = WanVideoEdit()
        model.load_model()
        model.unload_model()
        assert model._pipeline is None

    def test_edit_without_load_raises(self):
        model = WanVideoEdit()
        input_frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        with pytest.raises(RuntimeError, match="[Nn]ot loaded"):
            model._edit(
                video_frames=input_frames,
                prompt="test",
                negative_prompt="",
            )
