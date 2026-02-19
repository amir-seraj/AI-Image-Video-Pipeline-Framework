"""End-to-end test: Qwen image editing through a pipeline.

Supports both the full-precision and quantized models via --model arg.

Requires:
- CUDA GPU with bfloat16 support
- Internet access on first run (to download model weights)

Run with:
    python -m pytest tests/test_e2e_qwen.py -v -s
    python -m pytest tests/test_e2e_qwen.py -v -s --model qwen_image_edit_fp8
"""

import pytest
import torch
from datetime import datetime
from pathlib import Path
from PIL import Image as PILImage

from casadei import (
    Agent,
    AgentConfig,
    AgentStep,
    ImageMedia,
    LoggedPipeline,
    Pipeline,
)

# Where to save results for visual inspection
OUTPUT_DIR = Path(__file__).parent / "output"

VALID_MODELS = {
    "qwen_image_edit": "full",
    "qwen_image_edit_fp8": "fp8",
    "qwen_image_edit_gguf": "gguf",
}


def pad_to_square(img: PILImage.Image) -> PILImage.Image:
    """Pad an image with white to make it 1:1 aspect ratio."""
    w, h = img.size
    if w == h:
        return img
    size = max(w, h)
    square = PILImage.new("RGB", (size, size), (255, 255, 255))
    offset_x = (size - w) // 2
    offset_y = (size - h) // 2
    square.paste(img, (offset_x, offset_y))
    return square


@pytest.fixture(scope="module")
def model_name(request):
    return request.config.getoption("--model")


@pytest.fixture(scope="module")
def model_label(model_name):
    return VALID_MODELS[model_name]


@pytest.fixture(scope="module", autouse=True)
def setup_output_dir():
    OUTPUT_DIR.mkdir(exist_ok=True)


@pytest.fixture(scope="module")
def qwen_agent(model_name):
    """Load the Qwen model once for all tests in this module."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available; skipping Qwen E2E test")

    agent = Agent(
        config=AgentConfig(
            name=f"qwen_editor_{model_name}",
            model=model_name,
            prompt_template="$prompt",
        )
    )
    agent.load()
    yield agent
    agent.unload()


IMAGE_DIR = Path(__file__).parent / "Image"


class TestQwenE2E:
    def test_shoe_replacement(self, qwen_agent, model_label):
        """Load real photos of model legs and shoes, ask Qwen to swap the footwear."""
        legs_img = PILImage.open(IMAGE_DIR / "legs001.jpeg").convert("RGB")
        shoes_img = PILImage.open(IMAGE_DIR / "shoes001.jpeg").convert("RGB")

        # Pad legs image to 1:1 aspect ratio with white padding
        legs_img = pad_to_square(legs_img)

        # Create output dir early so steps can be saved during inference
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = OUTPUT_DIR / f"{timestamp}_{model_label}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Enable step visualization — save every 5th step to reduce VAE decode overhead
        qwen_agent._model.save_steps_dir = run_dir / "steps"
        qwen_agent._model.save_steps_interval = 5

        pipeline = Pipeline(
            name="shoe_replace_pipeline",
            steps=[
                AgentStep(
                    name="qwen_edit",
                    agent=qwen_agent,
                    input_map={"legs": "legs", "shoes": "shoes"},
                    output_map={"image": "result"},
                    template_kwargs={
                        "prompt": "In the first image there is a person wearing shoes. Replace the shoes on the person's feet with the shoes shown in the second image. Keep the person's pose, legs, clothing, and the background exactly the same.",
                    },
                ),
            ],
        )

        logged = LoggedPipeline(pipeline)
        result, log = logged.run({
            "legs": ImageMedia(image=legs_img),
            "shoes": ImageMedia(image=shoes_img),
        })

        qwen_agent._model.save_steps_dir = None
        qwen_agent._model.save_steps_interval = 1

        # Verify we got a result image
        assert "result" in result
        assert isinstance(result["result"], ImageMedia)

        result_image = result["result"].image
        assert isinstance(result_image, PILImage.Image)
        assert result_image.size == legs_img.size

        # Save inputs and final output alongside the steps
        legs_img.save(run_dir / "legs_input.png")
        shoes_img.save(run_dir / "shoes_input.png")
        result_image.save(run_dir / "shoe_replace_output.png")

        print(f"\nModel: {model_label}")
        print(f"Legs input saved to:  {run_dir / 'legs_input.png'}")
        print(f"Shoes input saved to: {run_dir / 'shoes_input.png'}")
        print(f"Output saved to:      {run_dir / 'shoe_replace_output.png'}")
        print(f"Output size: {result_image.size}")
        steps_count = len(list((run_dir / "steps").glob("step_*.png")))
        print(f"Step images:  {steps_count} saved to {run_dir / 'steps'}")

        # Verify logging
        assert log.pipeline_name == "shoe_replace_pipeline"
        assert len(log.step_logs) == 1
        assert log.step_logs[0].step_name == "qwen_edit"
        assert log.total_duration_ms > 0

        print(f"\n{log.summary()}")
