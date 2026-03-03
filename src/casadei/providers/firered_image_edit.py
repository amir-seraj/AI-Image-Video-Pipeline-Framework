"""FireRed-Image-Edit-1.0 provider.

FireRedTeam's 20B MMDiT image editing model built on Qwen-Image foundation.
Uses QwenImageEditPlusPipeline (same as Qwen-Image-Edit). ~58 GB in BF16,
fits on Jetson Thor (128 GB unified) without quantization.

Download + first run:  python src/casadei/providers/firered_image_edit.py
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

import torch
from PIL import Image as PILImage

from casadei import MODELS_DIR
from casadei.models.base import ModelCapability, ImageConstraint, TextConstraint
from casadei.models.image_edit import ImageEditModel
from casadei.providers._base import (
    clamp_steps,
    decode_and_save_step_latents,
    make_step_callback,
)

logger = logging.getLogger(__name__)

MODEL_ID = "FireRedTeam/FireRed-Image-Edit-1.0"

try:
    from diffusers import QwenImageEditPlusPipeline
except ImportError:
    QwenImageEditPlusPipeline = None


def download_model() -> None:
    """Download the model to HF cache if not already present."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {MODEL_ID} (~58 GB) ...")
    snapshot_download(MODEL_ID, cache_dir=MODELS_DIR)
    print("Download complete.")


class FireRedImageEdit(ImageEditModel):
    """FireRedTeam/FireRed-Image-Edit-1.0 — 20B MMDiT image editor.

    Built on the Qwen-Image foundation. Accepts up to 2 images and a text
    prompt, produces 1 edited image. Uses QwenImageEditPlusPipeline.
    ~58 GB in BF16.

    Text prompt token limits (T5/CLIP text encoder):
      - Max position embeddings: 128,000
      - Tokenizer model_max_length: 131,072
    """

    MODEL_ID = MODEL_ID

    capability = ModelCapability(
        inputs=[
            ImageConstraint(
                required=True,
                max_count=2,
                supported_formats=["png", "jpg", "jpeg", "webp"],
            ),
            TextConstraint(required=True, max_count=2),
        ],
        outputs=[
            ImageConstraint(required=True, max_count=1),
        ],
    )

    PIPELINE_CLS = QwenImageEditPlusPipeline

    DEFAULT_PARAMS = {
        "num_inference_steps": 40,
        "true_cfg_scale": 2.0,
        "negative_prompt": " ",
        "num_images_per_prompt": 1,
    }

    MIN_STEPS = 1
    MAX_STEPS = 50

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = None
        self.save_steps_dir: Path | None = None
        self.save_steps_interval: int = 1

    def load_model(self) -> None:
        if QwenImageEditPlusPipeline is None:
            raise ImportError(
                "diffusers with QwenImageEditPlusPipeline is required. "
                "Install: pip install git+https://github.com/huggingface/diffusers"
            )

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            self.MODEL_ID, torch_dtype=torch_dtype, cache_dir=MODELS_DIR
        )
        if torch.cuda.is_available():
            pipe.to("cuda")

        self._pipeline = pipe

    def unload_model(self) -> None:
        self._pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _edit(
        self,
        images: list[PILImage.Image],
        prompt: str,
        negative_prompt: str,
        **kwargs,
    ) -> PILImage.Image:
        if self._pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        params = {**self.DEFAULT_PARAMS, **kwargs}
        params["negative_prompt"] = negative_prompt or params["negative_prompt"]
        clamp_steps(params, "num_inference_steps", self.MIN_STEPS, self.MAX_STEPS)

        target_size = images[-1].size
        saved_latents: list[tuple[int, torch.Tensor]] = []

        if self.save_steps_dir is not None:
            steps_dir = Path(self.save_steps_dir)
            steps_dir.mkdir(parents=True, exist_ok=True)
            params["callback_on_step_end"] = make_step_callback(
                saved_latents, self.save_steps_interval
            )
            params["callback_on_step_end_tensor_inputs"] = ["latents"]

        with torch.inference_mode():
            output = self._pipeline(
                image=images,
                prompt=prompt,
                **params,
            )

        if saved_latents and self.save_steps_dir is not None:
            decode_and_save_step_latents(
                saved_latents, self._pipeline,
                output.images[0], Path(self.save_steps_dir),
            )

        result = output.images[0]
        if result.size != target_size:
            result = result.resize(target_size, PILImage.LANCZOS)
        return result


if __name__ == "__main__":
    download_model()
