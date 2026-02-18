import pytest
import numpy as np
from unittest.mock import MagicMock, patch, call

from casadei.models.base import TextConstraint, VideoConstraint
from casadei.providers.wan_video_edit_fp8 import WanVideoEditFP8
from casadei.providers.wan_video_edit import WanVideoEdit


class TestWanVideoEditFP8Capability:
    def test_is_video_edit_model(self):
        from casadei.models.video_edit import VideoEditModel
        assert issubclass(WanVideoEditFP8, VideoEditModel)

    def test_has_same_capability_as_base(self):
        """FP8 variant should advertise the same inputs/outputs as the base."""
        base_cap = WanVideoEdit.capability
        fp8_cap = WanVideoEditFP8.capability

        assert len(fp8_cap.inputs) == len(base_cap.inputs)
        assert len(fp8_cap.outputs) == len(base_cap.outputs)

        for fp8_in, base_in in zip(fp8_cap.inputs, base_cap.inputs):
            assert type(fp8_in) is type(base_in)
            assert fp8_in.required == base_in.required

        for fp8_out, base_out in zip(fp8_cap.outputs, base_cap.outputs):
            assert type(fp8_out) is type(base_out)
            assert fp8_out.required == base_out.required

    def test_uses_same_model_id(self):
        assert WanVideoEditFP8.MODEL_ID == WanVideoEdit.MODEL_ID


class TestWanVideoEditFP8Inference:
    @patch("casadei.providers.wan_video_edit_fp8.torch")
    @patch("casadei.providers.wan_video_edit_fp8.quantize_")
    @patch("casadei.providers.wan_video_edit_fp8.Float8WeightOnlyConfig")
    @patch("casadei.providers.wan_video_edit_fp8.UniPCMultistepScheduler")
    @patch("casadei.providers.wan_video_edit_fp8.AutoencoderKLWan")
    @patch("casadei.providers.wan_video_edit_fp8.WanVideoToVideoPipeline")
    def test_load_model_applies_compile(
        self,
        mock_pipeline_cls,
        mock_vae_cls,
        mock_sched_cls,
        mock_fp8_config_cls,
        mock_quantize,
        mock_torch,
    ):
        mock_pipe = MagicMock()
        # Capture the original transformer mock before load_model reassigns it
        original_transformer = mock_pipe.transformer
        mock_pipeline_cls.from_pretrained.return_value = mock_pipe
        mock_vae_cls.from_pretrained.return_value = MagicMock()
        mock_sched_cls.from_config.return_value = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.bfloat16 = MagicMock()
        mock_torch.float32 = MagicMock()
        mock_fp8_config_instance = MagicMock()
        mock_fp8_config_cls.return_value = mock_fp8_config_instance
        compiled_transformer = MagicMock()
        mock_torch.compile.return_value = compiled_transformer

        model = WanVideoEditFP8()
        model.load_model()

        # quantize_ should be called with the original transformer
        mock_quantize.assert_called_once_with(
            original_transformer, mock_fp8_config_instance
        )

        # torch.compile should be called on the (quantized) transformer
        mock_torch.compile.assert_called_once_with(
            original_transformer, mode="max-autotune", fullgraph=True
        )

        # The pipeline's transformer should now be the compiled version
        assert mock_pipe.transformer == compiled_transformer

    def test_edit_without_load_raises(self):
        model = WanVideoEditFP8()
        input_frames = [np.zeros((100, 100, 3), dtype=np.uint8)]
        with pytest.raises(RuntimeError, match="[Nn]ot loaded"):
            model._edit(
                video_frames=input_frames,
                prompt="test",
                negative_prompt="",
            )
