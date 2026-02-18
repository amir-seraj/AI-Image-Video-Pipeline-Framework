"""Quick I2V test: woman taking off a shoe — FP8 optimized."""

import time
import torch
from pathlib import Path
from PIL import Image as PILImage

from casadei import Agent, AgentConfig, AgentStep, ImageMedia, LoggedPipeline, Pipeline, VideoMedia
from casadei.providers.wan_i2v_fp8 import WanImageToVideoFP8

IMAGE_PATH = Path(__file__).parent / "Image" / "legs001.jpeg"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

WanImageToVideoFP8.DEFAULT_PARAMS["num_frames"] = 81
WanImageToVideoFP8.DEFAULT_PARAMS["num_inference_steps"] = 30

print("Loading FP8 model...")
t0 = time.time()
agent = Agent(
    config=AgentConfig(
        name="wan_i2v_fp8",
        model="wan_i2v_fp8",
        prompt_template="$prompt",
    )
)
agent.load()
print(f"Model loaded in {time.time() - t0:.1f}s")

if torch.cuda.is_available():
    mem_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"GPU memory after load: {mem_gb:.1f} GB")

print("Running inference (first run includes torch.compile warmup)...")
t1 = time.time()
pipeline = Pipeline(
    name="shoe_removal_fp8",
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
    "source_image": ImageMedia(image=PILImage.open(IMAGE_PATH)),
})

t_total = time.time() - t1
video = result["result_video"]
out_path = OUTPUT_DIR / "shoe_removal_fp8.mp4"
video.save(out_path)

print(f"\nOutput: {out_path}")
print(f"Frames: {video.frame_count}, FPS: {video.fps}, Duration: {video.duration_seconds:.1f}s")
print(f"Total inference: {t_total:.1f}s")
if torch.cuda.is_available():
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak GPU memory: {peak_gb:.1f} GB")
print(f"\n{log.summary()}")

agent.unload()
