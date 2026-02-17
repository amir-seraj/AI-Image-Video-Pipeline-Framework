import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from PIL import Image as PILImage

from casadei.media import ImageMedia, TextMedia, MediaBundle
from casadei.models.base import ImageConstraint, TextConstraint, VideoConstraint
from casadei.providers.wan_i2v import WanImageToVideo


class TestWanImageToVideoCapability:
    def test_accepts_single_image(self):
        img_constraints = [
            c for c in WanImageToVideo.capability.inputs
            if isinstance(c, ImageConstraint)
        ]
        assert len(img_constraints) == 1
        assert img_constraints[0].max_count == 1

    def test_requires_text_prompt(self):
        txt_constraints = [
            c for c in WanImageToVideo.capability.inputs
            if isinstance(c, TextConstraint)
        ]
        assert len(txt_constraints) >= 1
        assert txt_constraints[0].required is True

    def test_outputs_single_video(self):
        vid_constraints = [
            c for c in WanImageToVideo.capability.outputs
            if isinstance(c, VideoConstraint)
        ]
        assert len(vid_constraints) == 1
        assert vid_constraints[0].max_count == 1

    def test_is_image_to_video_model(self):
        from casadei.models.image_to_video import ImageToVideoModel
        assert issubclass(WanImageToVideo, ImageToVideoModel)


class TestWanImageToVideoInference:
    @patch("casadei.providers.wan_i2v.CLIPVisionModel")
    @patch("casadei.providers.wan_i2v.AutoencoderKLWan")
    @patch("casadei.providers.wan_i2v.WanImageToVideoPipeline")
    def test_load_model(self, mock_pipeline_cls, mock_vae_cls, mock_clip_cls):
        mock_pipe = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe
        mock_vae_cls.from_pretrained.return_value = MagicMock()
        mock_clip_cls.from_pretrained.return_value = MagicMock()

        model = WanImageToVideo()
        model.load_model()

        mock_pipeline_cls.from_pretrained.assert_called_once()
        mock_vae_cls.from_pretrained.assert_called_once()
        mock_clip_cls.from_pretrained.assert_called_once()

    @patch("casadei.providers.wan_i2v.CLIPVisionModel")
    @patch("casadei.providers.wan_i2v.AutoencoderKLWan")
    @patch("casadei.providers.wan_i2v.WanImageToVideoPipeline")
    def test_generate_calls_pipeline(self, mock_pipeline_cls, mock_vae_cls, mock_clip_cls):
        fake_frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(4)]
        mock_pipe = MagicMock()
        mock_pipe.return_value.frames = [fake_frames]
        mock_pipe.vae_scale_factor_spatial = 8
        mock_pipe.transformer.config.patch_size = [1, 2]
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe
        mock_vae_cls.from_pretrained.return_value = MagicMock()
        mock_clip_cls.from_pretrained.return_value = MagicMock()

        model = WanImageToVideo()
        model.load_model()

        input_img = PILImage.new("RGB", (1280, 720), color="red")
        result = model._generate(
            image=input_img,
            prompt="animate this",
            negative_prompt="",
        )
        assert len(result) == 4
        mock_pipe.assert_called_once()

    @patch("casadei.providers.wan_i2v.CLIPVisionModel")
    @patch("casadei.providers.wan_i2v.AutoencoderKLWan")
    @patch("casadei.providers.wan_i2v.WanImageToVideoPipeline")
    def test_run_end_to_end(self, mock_pipeline_cls, mock_vae_cls, mock_clip_cls):
        fake_frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(4)]
        mock_pipe = MagicMock()
        mock_pipe.return_value.frames = [fake_frames]
        mock_pipe.vae_scale_factor_spatial = 8
        mock_pipe.transformer.config.patch_size = [1, 2]
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe
        mock_vae_cls.from_pretrained.return_value = MagicMock()
        mock_clip_cls.from_pretrained.return_value = MagicMock()

        model = WanImageToVideo()
        model.load_model()

        bundle = MediaBundle(items={
            "image": ImageMedia(image=PILImage.new("RGB", (1280, 720))),
            "prompt": TextMedia(text="animate this scene"),
        })
        result = model.run(bundle)
        assert "video" in result.items
        from casadei.media import VideoMedia
        assert isinstance(result["video"], VideoMedia)
        assert result["video"].frame_count == 4

    @patch("casadei.providers.wan_i2v.CLIPVisionModel")
    @patch("casadei.providers.wan_i2v.AutoencoderKLWan")
    @patch("casadei.providers.wan_i2v.WanImageToVideoPipeline")
    def test_unload_model(self, mock_pipeline_cls, mock_vae_cls, mock_clip_cls):
        mock_pipe = MagicMock()
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe
        mock_vae_cls.from_pretrained.return_value = MagicMock()
        mock_clip_cls.from_pretrained.return_value = MagicMock()

        model = WanImageToVideo()
        model.load_model()
        model.unload_model()
        assert model._pipeline is None

    def test_generate_without_load_raises(self):
        model = WanImageToVideo()
        with pytest.raises(RuntimeError, match="[Nn]ot loaded"):
            model._generate(
                image=PILImage.new("RGB", (100, 100)),
                prompt="test",
                negative_prompt="",
            )
