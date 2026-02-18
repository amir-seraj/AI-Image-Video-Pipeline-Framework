import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call
from PIL import Image as PILImage

from casadei.media import ImageMedia, TextMedia, MediaBundle
from casadei.models.base import ImageConstraint, TextConstraint, VideoConstraint
from casadei.providers.wan_i2v_fp8 import WanImageToVideoFP8


class TestWanImageToVideoFP8Capability:
    def test_is_image_to_video_model(self):
        from casadei.models.image_to_video import ImageToVideoModel
        assert issubclass(WanImageToVideoFP8, ImageToVideoModel)

    def test_has_same_capability_as_base(self):
        from casadei.providers.wan_i2v import WanImageToVideo
        fp8 = WanImageToVideoFP8
        base = WanImageToVideo
        # Same input constraints
        assert len(fp8.capability.inputs) == len(base.capability.inputs)
        assert len(fp8.capability.outputs) == len(base.capability.outputs)
        # Check image constraint
        fp8_img = [c for c in fp8.capability.inputs if isinstance(c, ImageConstraint)]
        base_img = [c for c in base.capability.inputs if isinstance(c, ImageConstraint)]
        assert len(fp8_img) == len(base_img)
        assert fp8_img[0].max_count == base_img[0].max_count
        # Check text constraint
        fp8_txt = [c for c in fp8.capability.inputs if isinstance(c, TextConstraint)]
        base_txt = [c for c in base.capability.inputs if isinstance(c, TextConstraint)]
        assert len(fp8_txt) == len(base_txt)
        assert fp8_txt[0].required == base_txt[0].required
        # Check video constraint
        fp8_vid = [c for c in fp8.capability.outputs if isinstance(c, VideoConstraint)]
        base_vid = [c for c in base.capability.outputs if isinstance(c, VideoConstraint)]
        assert len(fp8_vid) == len(base_vid)
        assert fp8_vid[0].max_count == base_vid[0].max_count

    def test_uses_same_model_id(self):
        from casadei.providers.wan_i2v import WanImageToVideo
        assert WanImageToVideoFP8.MODEL_ID == WanImageToVideo.MODEL_ID


class TestWanImageToVideoFP8Inference:
    @patch("casadei.providers.wan_i2v_fp8.torch")
    @patch("casadei.providers.wan_i2v_fp8.CLIPVisionModel")
    @patch("casadei.providers.wan_i2v_fp8.AutoencoderKLWan")
    @patch("casadei.providers.wan_i2v_fp8.WanImageToVideoPipeline")
    def test_load_model_applies_compile(
        self, mock_pipeline_cls, mock_vae_cls, mock_clip_cls, mock_torch
    ):
        mock_torch.cuda.is_available.return_value = True
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.float32 = "float32"

        mock_pipe = MagicMock()
        # Capture the original transformer mock before it gets reassigned
        original_transformer = mock_pipe.transformer
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe
        mock_vae_cls.from_pretrained.return_value = MagicMock()
        mock_clip_cls.from_pretrained.return_value = MagicMock()

        # Mock the quantize_ function
        with patch("casadei.providers.wan_i2v_fp8.quantize_") as mock_quantize, \
             patch("casadei.providers.wan_i2v_fp8.Float8WeightOnlyConfig") as mock_fp8_config:
            mock_fp8_config_instance = MagicMock()
            mock_fp8_config.return_value = mock_fp8_config_instance
            compiled_transformer = MagicMock()
            mock_torch.compile.return_value = compiled_transformer

            model = WanImageToVideoFP8()
            model.load_model()

            # Verify FP8 quantization was applied to original transformer
            mock_quantize.assert_called_once_with(
                original_transformer, mock_fp8_config_instance
            )
            # Verify torch.compile was called with the original transformer
            mock_torch.compile.assert_called_once_with(
                original_transformer,
                mode="max-autotune",
                fullgraph=True,
            )
            # Verify the compiled transformer was assigned back
            assert mock_pipe.transformer == compiled_transformer

    @patch("casadei.providers.wan_i2v_fp8.torch")
    @patch("casadei.providers.wan_i2v_fp8.CLIPVisionModel")
    @patch("casadei.providers.wan_i2v_fp8.AutoencoderKLWan")
    @patch("casadei.providers.wan_i2v_fp8.WanImageToVideoPipeline")
    def test_generate_calls_pipeline(
        self, mock_pipeline_cls, mock_vae_cls, mock_clip_cls, mock_torch
    ):
        mock_torch.cuda.is_available.return_value = False
        mock_torch.float32 = "float32"

        fake_frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(4)]
        mock_pipe = MagicMock()
        mock_pipe.return_value.frames = [fake_frames]
        mock_pipe.vae_scale_factor_spatial = 8
        mock_pipe.transformer.config.patch_size = [1, 2]
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe
        mock_vae_cls.from_pretrained.return_value = MagicMock()
        mock_clip_cls.from_pretrained.return_value = MagicMock()

        model = WanImageToVideoFP8()
        model.load_model()

        input_img = PILImage.new("RGB", (1280, 720), color="red")
        result = model._generate(
            image=input_img,
            prompt="animate this",
            negative_prompt="",
        )
        assert len(result) == 4
        mock_pipe.assert_called_once()

    def test_generate_without_load_raises(self):
        model = WanImageToVideoFP8()
        with pytest.raises(RuntimeError, match="[Nn]ot loaded"):
            model._generate(
                image=PILImage.new("RGB", (100, 100)),
                prompt="test",
                negative_prompt="",
            )
