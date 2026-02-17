"""End-to-end test: real Wan2.1 video generation/editing through a pipeline.

Requires:
- CUDA GPU with sufficient memory
- Internet access on first run (to download Wan model weights)

Run with:
    python -m pytest tests/test_e2e_wan.py -v -s
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from PIL import Image as PILImage

from casadei import (
    Agent,
    AgentConfig,
    AgentStep,
    ImageMedia,
    LoggedPipeline,
    Pipeline,
    VideoMedia,
)
from casadei.providers.wan_i2v import WanImageToVideo
from casadei.providers.wan_video_edit import WanVideoEdit

OUTPUT_DIR = Path(__file__).parent / "output"
IMAGE_DIR = Path(__file__).parent / "Image"


@pytest.fixture(scope="module", autouse=True)
def setup_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)


@pytest.fixture(scope="module")
def i2v_agent():
    """Load the real Wan I2V model once for all tests in this module."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available; skipping Wan E2E test")

    original_params = WanImageToVideo.DEFAULT_PARAMS.copy()
    WanImageToVideo.DEFAULT_PARAMS["num_inference_steps"] = 10
    WanImageToVideo.DEFAULT_PARAMS["num_frames"] = 17

    agent = Agent(
        config=AgentConfig(
            name="wan_i2v",
            model="wan_i2v",
            prompt_template="$prompt",
        )
    )
    agent.load()
    yield agent
    agent.unload()
    WanImageToVideo.DEFAULT_PARAMS = original_params


@pytest.fixture(scope="module")
def video_edit_agent():
    """Load the real Wan Video Edit model once for all tests in this module."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available; skipping Wan E2E test")

    original_params = WanVideoEdit.DEFAULT_PARAMS.copy()
    WanVideoEdit.DEFAULT_PARAMS["num_inference_steps"] = 10

    agent = Agent(
        config=AgentConfig(
            name="wan_video_edit",
            model="wan_video_edit",
            prompt_template="$prompt",
        )
    )
    agent.load()
    yield agent
    agent.unload()
    WanVideoEdit.DEFAULT_PARAMS = original_params


class TestWanI2VE2E:
    def test_image_to_video_generation(self, i2v_agent):
        """Generate a short video from a test image."""
        input_img = PILImage.new("RGB", (1280, 720), color="blue")

        pipeline = Pipeline(
            name="i2v_pipeline",
            steps=[
                AgentStep(
                    name="wan_generate",
                    agent=i2v_agent,
                    input_map={"image": "source_image"},
                    output_map={"video": "result_video"},
                    template_kwargs={
                        "prompt": "A calm ocean scene with gentle waves",
                    },
                ),
            ],
        )

        logged = LoggedPipeline(pipeline)
        result, log = logged.run({
            "source_image": ImageMedia(image=input_img),
        })

        assert "result_video" in result
        assert isinstance(result["result_video"], VideoMedia)
        assert result["result_video"].frame_count > 0

        # Save for visual inspection
        output_video = result["result_video"]
        output_video.save(OUTPUT_DIR / "i2v_output.mp4")

        print(f"\nOutput saved to: {OUTPUT_DIR / 'i2v_output.mp4'}")
        print(f"Frames: {output_video.frame_count}, FPS: {output_video.fps}")
        print(f"\n{log.summary()}")


class TestWanVideoEditE2E:
    def test_video_editing(self, video_edit_agent):
        """Edit a synthetic video with text guidance."""
        # Create a simple test video (solid color frames)
        input_frames = [
            np.full((480, 640, 3), [0, 0, 255], dtype=np.uint8)
            for _ in range(17)
        ]

        pipeline = Pipeline(
            name="video_edit_pipeline",
            steps=[
                AgentStep(
                    name="wan_edit",
                    agent=video_edit_agent,
                    input_map={"video": "source_video"},
                    output_map={"video": "result_video"},
                    template_kwargs={
                        "prompt": "Transform into a sunset scene with warm orange tones",
                    },
                ),
            ],
        )

        logged = LoggedPipeline(pipeline)
        result, log = logged.run({
            "source_video": VideoMedia.from_frames(input_frames, fps=16),
        })

        assert "result_video" in result
        assert isinstance(result["result_video"], VideoMedia)
        assert result["result_video"].frame_count > 0

        output_video = result["result_video"]
        output_video.save(OUTPUT_DIR / "video_edit_output.mp4")

        print(f"\nOutput saved to: {OUTPUT_DIR / 'video_edit_output.mp4'}")
        print(f"Frames: {output_video.frame_count}, FPS: {output_video.fps}")
        print(f"\n{log.summary()}")
