"""End-to-end test: real Qwen image editing through a pipeline.

Requires:
- CUDA GPU (uses sequential CPU offloading for 8GB cards)
- Internet access on first run (to download Qwen/Qwen-Image-Edit-2511)

Run with:
    python -m pytest tests/test_e2e_qwen.py -v -s
"""

import pytest
from pathlib import Path
from PIL import Image as PILImage

from casadei import (
    Agent,
    AgentConfig,
    AgentStep,
    CodeStep,
    ImageMedia,
    LoggedPipeline,
    Pipeline,
)
from casadei.providers.qwen_image_edit import QwenImageEdit

# Where to save results for visual inspection
OUTPUT_DIR = Path(__file__).parent.parent / "output"


@pytest.fixture(scope="module", autouse=True)
def setup_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)


@pytest.fixture(scope="module")
def qwen_agent():
    """Load the real Qwen model once for all tests in this module."""
    # Use fewer inference steps to keep runtime practical on 8GB GPUs
    original_params = QwenImageEdit.DEFAULT_PARAMS.copy()
    QwenImageEdit.DEFAULT_PARAMS["num_inference_steps"] = 10

    agent = Agent(
        config=AgentConfig(
            name="qwen_editor",
            model="qwen_image_edit",
            prompt_template="$prompt",
        )
    )
    agent.load()
    yield agent
    agent.unload()
    QwenImageEdit.DEFAULT_PARAMS = original_params


IMAGE_DIR = Path(__file__).parent / "Image"


class TestQwenE2E:
    def test_shoe_replacement(self, qwen_agent):
        """Load real photos of model legs and shoes, ask Qwen to swap the footwear."""
        legs_img = PILImage.open(IMAGE_DIR / "legs001.jpeg").convert("RGB")
        shoes_img = PILImage.open(IMAGE_DIR / "shoes001.jpeg").convert("RGB")

        pipeline = Pipeline(
            name="shoe_replace_pipeline",
            steps=[
                AgentStep(
                    name="qwen_edit",
                    agent=qwen_agent,
                    input_map={"legs": "legs", "shoes": "shoes"},
                    output_map={"image": "result"},
                    template_kwargs={
                        "prompt": "Replace the shoes on the model's feet with the shoes shown in the second image",
                    },
                ),
            ],
        )

        logged = LoggedPipeline(pipeline)
        result, log = logged.run({
            "legs": ImageMedia(image=legs_img),
            "shoes": ImageMedia(image=shoes_img),
        })

        # Verify we got a result image
        assert "result" in result
        assert isinstance(result["result"], ImageMedia)

        result_image = result["result"].image
        assert isinstance(result_image, PILImage.Image)
        assert result_image.size[0] > 0
        assert result_image.size[1] > 0

        # Save all images for visual inspection
        legs_img.save(OUTPUT_DIR / "legs_input.png")
        shoes_img.save(OUTPUT_DIR / "shoes_input.png")
        result_image.save(OUTPUT_DIR / "shoe_replace_output.png")

        print(f"\nLegs input saved to:  {OUTPUT_DIR / 'legs_input.png'}")
        print(f"Shoes input saved to: {OUTPUT_DIR / 'shoes_input.png'}")
        print(f"Output saved to:      {OUTPUT_DIR / 'shoe_replace_output.png'}")
        print(f"Output size: {result_image.size}")

        # Verify logging
        assert log.pipeline_name == "shoe_replace_pipeline"
        assert len(log.step_logs) == 1
        assert log.step_logs[0].step_name == "qwen_edit"
        assert log.total_duration_ms > 0

        print(f"\n{log.summary()}")
