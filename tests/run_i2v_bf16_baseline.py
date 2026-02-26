"""BF16 quality baseline: 17 frames, 50 steps, no optimizations.

Establishes ground-truth video quality before adding FP8/caching.
Expected runtime: ~30 minutes on Jetson AGX Thor.
"""

import time
import torch
from pathlib import Path
from PIL import Image as PILImage

from casadei import Agent, AgentConfig, AgentStep, ImageMedia, LoggedPipeline, Pipeline, VideoMedia
from casadei.providers.wan_i2v import WanImageToVideo

IMAGE_PATH = Path(__file__).parent / "Image" / "legs001.jpeg"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

WanImageToVideo.DEFAULT_PARAMS["num_frames"] = 17
WanImageToVideo.DEFAULT_PARAMS["num_inference_steps"] = 50

print("Loading BF16 model (no FP8, no compile, no caching)...")
t0 = time.time()
agent = Agent(
    config=AgentConfig(
        name="wan_i2v",
        model="wan_i2v",
        prompt_template="$prompt",
    )
)
agent.load()
load_time = time.time() - t0
print(f"Model loaded in {load_time:.1f}s")

if torch.cuda.is_available():
    mem_gb = torch.cuda.memory_allocated() / 1024**3
    print(f"GPU memory after load: {mem_gb:.1f} GB")

print("Running inference (50 steps, pure BF16)...")
t1 = time.time()
pipeline = Pipeline(
    name="bf16_baseline",
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
out_path = OUTPUT_DIR / "bf16_baseline_50steps.mp4"
video.save(out_path)

print(f"\nOutput: {out_path}")
print(f"Frames: {video.frame_count}, FPS: {video.fps}, Duration: {video.duration_seconds:.1f}s")
print(f"Total inference: {t_total:.1f}s")
if torch.cuda.is_available():
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"Peak GPU memory: {peak_gb:.1f} GB")
print(f"\n{log.summary()}")

agent.unload()
