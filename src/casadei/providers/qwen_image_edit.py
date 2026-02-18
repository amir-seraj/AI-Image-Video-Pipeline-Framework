"""Qwen Image Edit model provider."""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from PIL import Image as PILImage

from casadei import MODELS_DIR
from casadei.models.base import ModelCapability, ImageConstraint, TextConstraint
from casadei.models.image_edit import ImageEditModel

try:
    from diffusers import QwenImageEditPlusPipeline
except ImportError:
    QwenImageEditPlusPipeline = None


class QwenImageEdit(ImageEditModel):
    """Qwen/Qwen-Image-Edit-2511 model.

    Accepts up to 2 images and a text prompt, produces 1 edited image.
    Requires CUDA GPU with bfloat16 support.
    """

    MODEL_ID = "Qwen/Qwen-Image-Edit-2511"

    capability = ModelCapability(
        inputs=[
            ImageConstraint(
                required=True,
                max_count=2,
                supported_formats=["png", "jpg", "jpeg", "webp"],
            ),
            TextConstraint(required=True),
        ],
        outputs=[
            ImageConstraint(required=True, max_count=1),
        ],
    )

    DEFAULT_PARAMS = {
        "num_inference_steps": 40,
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "num_images_per_prompt": 1,
    }

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
            pipe.enable_model_cpu_offload()

        self._pipeline = pipe

    def unload_model(self) -> None:
        self._pipeline = None
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

        target_size = images[0].size

        # Collect latents during inference for deferred VAE decode (avoids OOM
        # from having transformer + VAE on GPU simultaneously with cpu_offload).
        saved_latents: list[tuple[int, torch.Tensor]] = []

        if self.save_steps_dir is not None:
            steps_dir = Path(self.save_steps_dir)
            steps_dir.mkdir(parents=True, exist_ok=True)
            interval = self.save_steps_interval

            def callback_on_step_end(_pipe, step_index, timestep, callback_kwargs):
                if step_index % interval != 0:
                    return callback_kwargs
                # Save a CPU copy of the latents — decoded after inference finishes
                saved_latents.append(
                    (step_index, callback_kwargs["latents"].detach().cpu())
                )
                print(f"  step {step_index} latents captured")
                return callback_kwargs

            params["callback_on_step_end"] = callback_on_step_end
            params["callback_on_step_end_tensor_inputs"] = ["latents"]

        with torch.inference_mode():
            output = self._pipeline(
                image=images,
                prompt=prompt,
                **params,
            )

        # Decode saved latents through VAE now that the transformer is offloaded
        if saved_latents and self.save_steps_dir is not None:
            steps_dir = Path(self.save_steps_dir)
            pipe = self._pipeline
            vae_config = pipe.vae.config
            _mean = vae_config["latents_mean"]
            _std = vae_config["latents_std"]
            print(f"  Decoding {len(saved_latents)} step latents via VAE...")
            for step_index, lat in saved_latents:
                with torch.no_grad():
                    device = next(pipe.vae.parameters()).device
                    lat = lat.to(device=device)
                    latents_mean = torch.tensor(
                        _mean, device=device, dtype=lat.dtype
                    ).view(1, -1, 1, 1)
                    latents_std = torch.tensor(
                        _std, device=device, dtype=lat.dtype
                    ).view(1, -1, 1, 1)
                    denormed = lat * latents_std + latents_mean
                    denormed = denormed.unsqueeze(2)
                    decoded = pipe.vae.decode(denormed, return_dict=False)[0]
                pixels = (decoded[0, :, 0].permute(1, 2, 0).float().cpu().numpy() / 2 + 0.5)
                pixels = (np.clip(pixels, 0, 1) * 255).astype(np.uint8)
                step_img = PILImage.fromarray(pixels)
                step_img.save(steps_dir / f"step_{step_index:03d}.png")
                print(f"  step {step_index} decoded and saved")
            del saved_latents
            torch.cuda.empty_cache()

        result = output.images[0]
        if result.size != target_size:
            result = result.resize(target_size, PILImage.LANCZOS)
        return result
