"""I2V test with landscape padding — fixes portrait aspect ratio issue.

The 720P model expects landscape (1280x720). Our test images are portrait.
This script pads the image to landscape before feeding it to the model.
"""

import time
import torch
from pathlib import Path
from PIL import Image as PILImage

from casadei import Agent, AgentConfig, AgentStep, ImageMedia, LoggedPipeline, Pipeline, VideoMedia
from casadei.providers.wan_i2v_fp8 import WanImageToVideoFP8

IMAGE_PATH = Path(__file__).parent / "Image" / "1model.jpg"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

WanImageToVideoFP8.DEFAULT_PARAMS["num_frames"] = 17
WanImageToVideoFP8.DEFAULT_PARAMS["num_inference_steps"] = 30


def pad_to_landscape(img: PILImage.Image) -> PILImage.Image:
    """Pad a portrait image to 16:9 landscape with black bars."""
    w, h = img.size
    target_ratio = 16 / 9  # 1280/720
    current_ratio = w / h

    if current_ratio >= target_ratio:
        return img  # already landscape enough

    # Pad width to achieve 16:9
    new_w = int(h * target_ratio)
    padded = PILImage.new("RGB", (new_w, h), (0, 0, 0))
    padded.paste(img, ((new_w - w) // 2, 0))
    return padded


# Pad input image to landscape
source = PILImage.open(IMAGE_PATH).convert("RGB")
print(f"Original: {source.size[0]}x{source.size[1]}")
source = pad_to_landscape(source)
print(f"Padded:   {source.size[0]}x{source.size[1]}")
padded_path = OUTPUT_DIR / "padded_input.png"
source.save(padded_path)

print("\nLoading FP8 model...")
t0 = time.time()
agent = Agent(
    config=AgentConfig(name="wan_i2v_fp8", model="wan_i2v_fp8", prompt_template="$prompt")
)
agent.load()
print(f"Model loaded in {time.time() - t0:.1f}s")

if torch.cuda.is_available():
    print(f"GPU memory after load: {torch.cuda.memory_allocated() / 1024**3:.1f} GB")

print(f"Running inference (17 frames, 30 steps, FP8 + TaylorSeer)...")
t1 = time.time()
pipeline = Pipeline(
    name="landscape_test",
    steps=[
        AgentStep(
            name="generate",
            agent=agent,
            input_map={"image": "source_image"},
            output_map={"video": "result_video"},
            template_kwargs={
                "prompt": "A woman gracefully takes off her black high heel shoe, slowly unbuckling the ankle strap and sliding the shoe off her foot, smooth natural motion",
            },
        ),
    ],
)

logged = LoggedPipeline(pipeline)
result, log = logged.run({
    "source_image": ImageMedia(image=source),
})

t_total = time.time() - t1
video = result["result_video"]
out_path = OUTPUT_DIR / "landscape_fp8_taylorseer.mp4"
video.save(out_path)

print(f"\nOutput: {out_path}")
print(f"Frames: {video.frame_count}, FPS: {video.fps}, Duration: {video.duration_seconds:.1f}s")
print(f"Total inference: {t_total:.1f}s")
if torch.cuda.is_available():
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak GPU memory: {peak_gb:.1f} GB")
print(f"\n{log.summary()}")

agent.unload()
