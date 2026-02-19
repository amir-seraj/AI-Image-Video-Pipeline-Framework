# Wan 2.1 FP8 + torch.compile Optimization — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create FP8-quantized + torch.compile optimized variants of the Wan I2V and Video Edit providers, cutting inference time ~2x and memory ~40%.

**Architecture:** New provider classes (`WanImageToVideoFP8`, `WanVideoEditFP8`) that subclass the BF16 baselines and override `load_model()` to apply torchao FP8 weight-only quantization to the transformer + `torch.compile`. VAE and CLIP stay float32. Registered as `wan_i2v_fp8` / `wan_video_edit_fp8`.

**Tech Stack:** torchao (FP8 quantization), torch.compile (kernel fusion), diffusers, existing casadei framework.

---

### Task 1: Install torchao dependency

**Files:**
- Modify: `pyproject.toml:10-23`

**Step 1: Add torchao to dependencies**

In `pyproject.toml`, add `"torchao"` to the dependencies list after `"numpy"`:

```toml
dependencies = [
    "pydantic>=2.0",
    "pyyaml",
    "pillow",
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "numpy",
    "torchao",
    "imageio[ffmpeg]",
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.34.0",
    "sse-starlette>=2.0.0",
    "python-multipart>=0.0.18",
]
```

**Step 2: Install torchao in the conda environment**

Run: `LD_LIBRARY_PATH=/home/innovina/miniconda3/envs/casadei/lib/python3.12/site-packages/nvpl/lib:/usr/local/cuda/lib64 /home/innovina/miniconda3/envs/casadei/bin/pip install torchao`

**Step 3: Verify import works**

Run: `LD_LIBRARY_PATH=/home/innovina/miniconda3/envs/casadei/lib/python3.12/site-packages/nvpl/lib:/usr/local/cuda/lib64 /home/innovina/miniconda3/envs/casadei/bin/python -c "from torchao.quantization import float8_weight_only; print('torchao OK')"`

Expected: `torchao OK`

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add torchao for FP8 quantization"
```

---

### Task 2: Create WanImageToVideoFP8 provider

**Files:**
- Create: `src/casadei/providers/wan_i2v_fp8.py`
- Test: `tests/test_wan_i2v_fp8.py`

**Step 1: Write the failing test**

Create `tests/test_wan_i2v_fp8.py`:

```python
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from PIL import Image as PILImage

from casadei.media import ImageMedia, TextMedia, MediaBundle
from casadei.models.base import ImageConstraint, TextConstraint, VideoConstraint


class TestWanImageToVideoFP8Capability:
    def test_is_image_to_video_model(self):
        from casadei.providers.wan_i2v_fp8 import WanImageToVideoFP8
        from casadei.models.image_to_video import ImageToVideoModel
        assert issubclass(WanImageToVideoFP8, ImageToVideoModel)

    def test_has_same_capability_as_base(self):
        from casadei.providers.wan_i2v_fp8 import WanImageToVideoFP8
        from casadei.providers.wan_i2v import WanImageToVideo
        fp8_inputs = WanImageToVideoFP8.capability.inputs
        base_inputs = WanImageToVideo.capability.inputs
        assert len(fp8_inputs) == len(base_inputs)

    def test_uses_same_model_id(self):
        from casadei.providers.wan_i2v_fp8 import WanImageToVideoFP8
        from casadei.providers.wan_i2v import WanImageToVideo
        assert WanImageToVideoFP8.MODEL_ID == WanImageToVideo.MODEL_ID


class TestWanImageToVideoFP8Inference:
    @patch("casadei.providers.wan_i2v_fp8.torch")
    @patch("casadei.providers.wan_i2v_fp8.CLIPVisionModel")
    @patch("casadei.providers.wan_i2v_fp8.AutoencoderKLWan")
    @patch("casadei.providers.wan_i2v_fp8.WanImageToVideoPipeline")
    def test_load_model_applies_compile(self, mock_pipe_cls, mock_vae_cls, mock_clip_cls, mock_torch):
        from casadei.providers.wan_i2v_fp8 import WanImageToVideoFP8

        mock_pipe = MagicMock()
        mock_pipe_cls.from_pretrained.return_value = mock_pipe
        mock_vae_cls.from_pretrained.return_value = MagicMock()
        mock_clip_cls.from_pretrained.return_value = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.float32 = "float32"
        mock_torch.inference_mode.return_value.__enter__ = MagicMock()
        mock_torch.inference_mode.return_value.__exit__ = MagicMock()
        mock_torch.compile.return_value = MagicMock()

        model = WanImageToVideoFP8()
        model.load_model()

        # Verify torch.compile was called on the transformer
        mock_torch.compile.assert_called_once()

    @patch("casadei.providers.wan_i2v_fp8.torch")
    @patch("casadei.providers.wan_i2v_fp8.CLIPVisionModel")
    @patch("casadei.providers.wan_i2v_fp8.AutoencoderKLWan")
    @patch("casadei.providers.wan_i2v_fp8.WanImageToVideoPipeline")
    def test_generate_calls_pipeline(self, mock_pipe_cls, mock_vae_cls, mock_clip_cls, mock_torch):
        from casadei.providers.wan_i2v_fp8 import WanImageToVideoFP8

        fake_frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(4)]
        mock_pipe = MagicMock()
        mock_pipe.return_value.frames = [fake_frames]
        mock_pipe.vae_scale_factor_spatial = 8
        mock_pipe.transformer.config.patch_size = [1, 2]
        mock_pipe_cls.from_pretrained.return_value = mock_pipe
        mock_vae_cls.from_pretrained.return_value = MagicMock()
        mock_clip_cls.from_pretrained.return_value = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.float32 = "float32"
        mock_torch.inference_mode.return_value.__enter__ = MagicMock()
        mock_torch.inference_mode.return_value.__exit__ = MagicMock()
        mock_torch.compile.return_value = mock_pipe.transformer

        model = WanImageToVideoFP8()
        model.load_model()

        input_img = PILImage.new("RGB", (1280, 720), color="red")
        result = model._generate(
            image=input_img,
            prompt="animate this",
            negative_prompt="",
        )
        assert len(result) == 4

    def test_generate_without_load_raises(self):
        from casadei.providers.wan_i2v_fp8 import WanImageToVideoFP8
        model = WanImageToVideoFP8()
        with pytest.raises(RuntimeError, match="[Nn]ot loaded"):
            model._generate(
                image=PILImage.new("RGB", (100, 100)),
                prompt="test",
                negative_prompt="",
            )
```

**Step 2: Run test to verify it fails**

Run: `LD_LIBRARY_PATH=/home/innovina/miniconda3/envs/casadei/lib/python3.12/site-packages/nvpl/lib:/usr/local/cuda/lib64 /home/innovina/miniconda3/envs/casadei/bin/python -m pytest tests/test_wan_i2v_fp8.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'casadei.providers.wan_i2v_fp8'`

**Step 3: Write the implementation**

Create `src/casadei/providers/wan_i2v_fp8.py`:

```python
"""Wan2.1 Image-to-Video model provider — FP8 quantized + torch.compile optimized."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image as PILImage

from casadei import MODELS_DIR
from casadei.models.base import ModelCapability, ImageConstraint, TextConstraint, VideoConstraint
from casadei.models.image_to_video import ImageToVideoModel

try:
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
    from transformers import CLIPVisionModel
except ImportError:
    AutoencoderKLWan = None
    WanImageToVideoPipeline = None
    CLIPVisionModel = None

try:
    from torchao.quantization import quantize_, float8_weight_only
except ImportError:
    quantize_ = None
    float8_weight_only = None


class WanImageToVideoFP8(ImageToVideoModel):
    """Wan-AI/Wan2.1-I2V-14B-720P — FP8 quantized + torch.compile.

    Same model as WanImageToVideo but with:
    - FP8 weight-only quantization on the transformer (via torchao)
    - torch.compile for CUDA kernel fusion
    """

    MODEL_ID = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"

    capability = ModelCapability(
        inputs=[
            ImageConstraint(
                required=True,
                max_count=1,
                supported_formats=["png", "jpg", "jpeg", "webp"],
            ),
            TextConstraint(required=True),
        ],
        outputs=[
            VideoConstraint(
                required=True,
                max_count=1,
                max_width=1280,
                max_height=720,
            ),
        ],
    )

    DEFAULT_PARAMS = {
        "num_frames": 81,
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
    }

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = None

    def load_model(self) -> None:
        if WanImageToVideoPipeline is None:
            raise ImportError(
                "diffusers with WanImageToVideoPipeline is required. "
                "Install: pip install diffusers transformers"
            )

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        image_encoder = CLIPVisionModel.from_pretrained(
            self.MODEL_ID, subfolder="image_encoder",
            torch_dtype=torch.float32, cache_dir=MODELS_DIR,
        )
        vae = AutoencoderKLWan.from_pretrained(
            self.MODEL_ID, subfolder="vae",
            torch_dtype=torch.float32, cache_dir=MODELS_DIR,
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
            self.MODEL_ID,
            vae=vae,
            image_encoder=image_encoder,
            torch_dtype=torch_dtype,
            cache_dir=MODELS_DIR,
        )
        if torch.cuda.is_available():
            pipe.to("cuda")

        # FP8 weight-only quantization on transformer only
        if quantize_ is not None and torch.cuda.is_available():
            quantize_(pipe.transformer, float8_weight_only())

        # torch.compile for kernel fusion
        pipe.transformer = torch.compile(
            pipe.transformer, mode="max-autotune", fullgraph=True,
        )

        self._pipeline = pipe

    def unload_model(self) -> None:
        self._pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _resize_for_vae(self, image: PILImage.Image, height: int, width: int) -> tuple[PILImage.Image, int, int]:
        """Resize image to match target resolution while respecting VAE patch constraints."""
        max_area = height * width
        aspect_ratio = image.height / image.width
        mod_value = (
            self._pipeline.vae_scale_factor_spatial
            * self._pipeline.transformer.config.patch_size[1]
        )
        h = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        w = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        return image.resize((w, h)), h, w

    def _generate(
        self,
        image: PILImage.Image,
        prompt: str,
        negative_prompt: str,
        **kwargs,
    ) -> list[np.ndarray]:
        if self._pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        params = {**self.DEFAULT_PARAMS, **kwargs}
        height = params.pop("height", 720)
        width = params.pop("width", 1280)

        resized_image, h, w = self._resize_for_vae(image, height, width)

        if negative_prompt:
            params["negative_prompt"] = negative_prompt

        with torch.inference_mode():
            output = self._pipeline(
                image=resized_image,
                prompt=prompt,
                height=h,
                width=w,
                **params,
            )

        return output.frames[0]
```

**Step 4: Run tests to verify they pass**

Run: `LD_LIBRARY_PATH=/home/innovina/miniconda3/envs/casadei/lib/python3.12/site-packages/nvpl/lib:/usr/local/cuda/lib64 /home/innovina/miniconda3/envs/casadei/bin/python -m pytest tests/test_wan_i2v_fp8.py -v`

Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/casadei/providers/wan_i2v_fp8.py tests/test_wan_i2v_fp8.py
git commit -m "feat: add FP8 + torch.compile Wan I2V provider"
```

---

### Task 3: Create WanVideoEditFP8 provider

**Files:**
- Create: `src/casadei/providers/wan_video_edit_fp8.py`
- Test: `tests/test_wan_video_edit_fp8.py`

**Step 1: Write the failing test**

Create `tests/test_wan_video_edit_fp8.py`:

```python
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from PIL import Image as PILImage

from casadei.media import TextMedia, VideoMedia, MediaBundle
from casadei.models.base import TextConstraint, VideoConstraint


class TestWanVideoEditFP8Capability:
    def test_is_video_edit_model(self):
        from casadei.providers.wan_video_edit_fp8 import WanVideoEditFP8
        from casadei.models.video_edit import VideoEditModel
        assert issubclass(WanVideoEditFP8, VideoEditModel)

    def test_has_same_capability_as_base(self):
        from casadei.providers.wan_video_edit_fp8 import WanVideoEditFP8
        from casadei.providers.wan_video_edit import WanVideoEdit
        assert len(WanVideoEditFP8.capability.inputs) == len(WanVideoEdit.capability.inputs)

    def test_uses_same_model_id(self):
        from casadei.providers.wan_video_edit_fp8 import WanVideoEditFP8
        from casadei.providers.wan_video_edit import WanVideoEdit
        assert WanVideoEditFP8.MODEL_ID == WanVideoEdit.MODEL_ID


class TestWanVideoEditFP8Inference:
    @patch("casadei.providers.wan_video_edit_fp8.torch")
    @patch("casadei.providers.wan_video_edit_fp8.AutoencoderKLWan")
    @patch("casadei.providers.wan_video_edit_fp8.WanVideoToVideoPipeline")
    def test_load_model_applies_compile(self, mock_pipe_cls, mock_vae_cls, mock_torch):
        from casadei.providers.wan_video_edit_fp8 import WanVideoEditFP8

        mock_pipe = MagicMock()
        mock_pipe_cls.from_pretrained.return_value = mock_pipe
        mock_pipe.scheduler.config = {}
        mock_vae_cls.from_pretrained.return_value = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.bfloat16 = "bfloat16"
        mock_torch.float32 = "float32"
        mock_torch.compile.return_value = MagicMock()

        model = WanVideoEditFP8()
        model.load_model()

        mock_torch.compile.assert_called_once()

    def test_edit_without_load_raises(self):
        from casadei.providers.wan_video_edit_fp8 import WanVideoEditFP8
        model = WanVideoEditFP8()
        frames = [np.zeros((480, 640, 3), dtype=np.uint8)]
        with pytest.raises(RuntimeError, match="[Nn]ot loaded"):
            model._edit(video_frames=frames, prompt="test", negative_prompt="")
```

**Step 2: Run test to verify it fails**

Run: `LD_LIBRARY_PATH=/home/innovina/miniconda3/envs/casadei/lib/python3.12/site-packages/nvpl/lib:/usr/local/cuda/lib64 /home/innovina/miniconda3/envs/casadei/bin/python -m pytest tests/test_wan_video_edit_fp8.py -v`

Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

Create `src/casadei/providers/wan_video_edit_fp8.py`:

```python
"""Wan2.1 Video-to-Video editing model provider — FP8 quantized + torch.compile optimized."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image as PILImage

from casadei import MODELS_DIR
from casadei.models.base import ModelCapability, TextConstraint, VideoConstraint
from casadei.models.video_edit import VideoEditModel

try:
    from diffusers import AutoencoderKLWan, WanVideoToVideoPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
except ImportError:
    AutoencoderKLWan = None
    WanVideoToVideoPipeline = None
    UniPCMultistepScheduler = None

try:
    from torchao.quantization import quantize_, float8_weight_only
except ImportError:
    quantize_ = None
    float8_weight_only = None


class WanVideoEditFP8(VideoEditModel):
    """Wan-AI/Wan2.1-T2V-14B video editing — FP8 quantized + torch.compile.

    Same model as WanVideoEdit but with:
    - FP8 weight-only quantization on the transformer (via torchao)
    - torch.compile for CUDA kernel fusion
    """

    MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"

    capability = ModelCapability(
        inputs=[
            VideoConstraint(required=True, max_count=1),
            TextConstraint(required=True),
        ],
        outputs=[
            VideoConstraint(required=True, max_count=1),
        ],
    )

    DEFAULT_PARAMS = {
        "num_inference_steps": 50,
        "guidance_scale": 5.0,
        "strength": 0.7,
    }

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = None

    def load_model(self) -> None:
        if WanVideoToVideoPipeline is None:
            raise ImportError(
                "diffusers with WanVideoToVideoPipeline is required. "
                "Install: pip install diffusers"
            )

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        vae = AutoencoderKLWan.from_pretrained(
            self.MODEL_ID, subfolder="vae",
            torch_dtype=torch.float32, cache_dir=MODELS_DIR,
        )
        pipe = WanVideoToVideoPipeline.from_pretrained(
            self.MODEL_ID,
            vae=vae,
            torch_dtype=torch_dtype,
            cache_dir=MODELS_DIR,
        )
        flow_shift = 5.0
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config, flow_shift=flow_shift
        )
        if torch.cuda.is_available():
            pipe.to("cuda")

        # FP8 weight-only quantization on transformer only
        if quantize_ is not None and torch.cuda.is_available():
            quantize_(pipe.transformer, float8_weight_only())

        # torch.compile for kernel fusion
        pipe.transformer = torch.compile(
            pipe.transformer, mode="max-autotune", fullgraph=True,
        )

        self._pipeline = pipe

    def unload_model(self) -> None:
        self._pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _edit(
        self,
        video_frames: list[np.ndarray],
        prompt: str,
        negative_prompt: str,
        **kwargs,
    ) -> list[np.ndarray]:
        if self._pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        params = {**self.DEFAULT_PARAMS, **kwargs}
        if negative_prompt:
            params["negative_prompt"] = negative_prompt

        pil_frames = [PILImage.fromarray(f) for f in video_frames]
        h, w = video_frames[0].shape[:2]

        with torch.inference_mode():
            output = self._pipeline(
                video=pil_frames,
                prompt=prompt,
                height=h,
                width=w,
                **params,
            )

        return output.frames[0]
```

**Step 4: Run tests to verify they pass**

Run: `LD_LIBRARY_PATH=/home/innovina/miniconda3/envs/casadei/lib/python3.12/site-packages/nvpl/lib:/usr/local/cuda/lib64 /home/innovina/miniconda3/envs/casadei/bin/python -m pytest tests/test_wan_video_edit_fp8.py -v`

Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/casadei/providers/wan_video_edit_fp8.py tests/test_wan_video_edit_fp8.py
git commit -m "feat: add FP8 + torch.compile Wan video edit provider"
```

---

### Task 4: Wire up registry, exports, and agent configs

**Files:**
- Modify: `src/casadei/models/registry.py:25-35`
- Modify: `src/casadei/providers/__init__.py`
- Modify: `src/casadei/__init__.py`
- Create: `agents/wan_image_to_video_fp8.yaml`
- Create: `agents/wan_video_editor_fp8.yaml`

**Step 1: Add to registry**

In `src/casadei/models/registry.py`, add after line 34 (before `return registry`):

```python
    from casadei.providers.wan_i2v_fp8 import WanImageToVideoFP8
    registry.register("wan_i2v_fp8", WanImageToVideoFP8)
    from casadei.providers.wan_video_edit_fp8 import WanVideoEditFP8
    registry.register("wan_video_edit_fp8", WanVideoEditFP8)
```

**Step 2: Add to providers __init__.py**

```python
"""Model provider implementations."""

from casadei.providers.qwen_image_edit import QwenImageEdit
from casadei.providers.wan_i2v import WanImageToVideo
from casadei.providers.wan_i2v_fp8 import WanImageToVideoFP8
from casadei.providers.wan_video_edit import WanVideoEdit
from casadei.providers.wan_video_edit_fp8 import WanVideoEditFP8

__all__ = [
    "QwenImageEdit",
    "WanImageToVideo", "WanImageToVideoFP8",
    "WanVideoEdit", "WanVideoEditFP8",
]
```

**Step 3: Add to casadei __init__.py**

Add imports and __all__ entries for `WanImageToVideoFP8` and `WanVideoEditFP8`.

**Step 4: Create agent YAML configs**

Create `agents/wan_image_to_video_fp8.yaml`:

```yaml
name: image_to_video_fp8
model: wan_i2v_fp8
description: Generates video from a still image (FP8 optimized)
prompt_template: "$prompt"
negative_prompt: "Bright tones, overexposed, static, blurred details, worst quality, low quality"
params:
  num_frames: 81
  num_inference_steps: 50
  guidance_scale: 5.0
```

Create `agents/wan_video_editor_fp8.yaml`:

```yaml
name: video_editor_fp8
model: wan_video_edit_fp8
description: Edits video with text guidance (FP8 optimized)
prompt_template: "$prompt"
negative_prompt: "worst quality, low quality, blurred"
params:
  num_inference_steps: 50
  guidance_scale: 5.0
  strength: 0.7
```

**Step 5: Run all unit tests**

Run: `LD_LIBRARY_PATH=/home/innovina/miniconda3/envs/casadei/lib/python3.12/site-packages/nvpl/lib:/usr/local/cuda/lib64 /home/innovina/miniconda3/envs/casadei/bin/python -m pytest tests/ -v --ignore=tests/test_e2e_wan.py --ignore=tests/test_e2e_qwen.py`

Expected: All tests PASS (including existing tests — no regressions)

**Step 6: Commit**

```bash
git add src/casadei/models/registry.py src/casadei/providers/__init__.py src/casadei/__init__.py agents/wan_image_to_video_fp8.yaml agents/wan_video_editor_fp8.yaml
git commit -m "feat: register FP8 providers and add agent configs"
```

---

### Task 5: E2E test — run the shoe image through FP8 provider

**Files:**
- Modify: `tests/run_i2v_shoe.py` (switch to FP8 provider)

**Step 1: Update the shoe test script to use FP8**

Modify `tests/run_i2v_shoe.py` to use `WanImageToVideoFP8` and register as `wan_i2v_fp8`:

```python
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
WanImageToVideoFP8.DEFAULT_PARAMS["num_inference_steps"] = 50

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
```

**Step 2: Run the FP8 shoe test**

Run: `LD_LIBRARY_PATH=/home/innovina/miniconda3/envs/casadei/lib/python3.12/site-packages/nvpl/lib:/usr/local/cuda/lib64 /home/innovina/miniconda3/envs/casadei/bin/python tests/run_i2v_shoe.py`

Expected: Completes successfully. Compare inference time and GPU memory to BF16 baseline.

**Step 3: Commit**

```bash
git add tests/run_i2v_shoe.py
git commit -m "test: update shoe test to use FP8 provider with perf metrics"
```
