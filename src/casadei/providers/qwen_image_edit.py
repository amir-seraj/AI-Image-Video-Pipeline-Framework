"""Qwen Image Edit model provider."""

from __future__ import annotations

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

    PIPELINE_CLS = QwenImageEditPlusPipeline

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
            device = pipe._execution_device
            dtype = pipe.vae.dtype
            z_dim = pipe.vae.config.get("z_dim", 16)
            _mean = pipe.vae.config["latents_mean"]
            _std = pipe.vae.config["latents_std"]

            # 5D mean/std for the Qwen causal-3D VAE: (1, C, 1, 1, 1)
            latents_mean = torch.tensor(
                _mean, device=device, dtype=dtype
            ).view(1, z_dim, 1, 1, 1)
            latents_std = torch.tensor(
                _std, device=device, dtype=dtype
            ).view(1, z_dim, 1, 1, 1)

            result_img = output.images[0]
            img_h, img_w = result_img.size[1], result_img.size[0]

            # Unpack packed transformer latents → (B, C, 1, H_lat, W_lat)
            unpacked = []
            step_indices = []
            for step_index, lat in saved_latents:
                lat = lat.to(device=device, dtype=dtype)
                lat_5d = pipe._unpack_latents(lat, img_h, img_w, pipe.vae_scale_factor)
                unpacked.append(lat_5d)
                step_indices.append(step_index)

            batch = torch.cat(unpacked, dim=0)
            denormed = batch * latents_std + latents_mean
            del unpacked, batch

            print(f"  Decoding {len(step_indices)} step latents via VAE...")
            with torch.no_grad():
                decoded = pipe.vae.decode(denormed, return_dict=False)[0]
            del denormed

            images_4d = decoded[:, :, 0]
            for i, step_index in enumerate(step_indices):
                img = pipe.image_processor.postprocess(
                    images_4d[i : i + 1], output_type="pil"
                )[0]
                img.save(steps_dir / f"step_{step_index:03d}.png")
                print(f"  step {step_index} decoded and saved")

            del saved_latents, decoded, images_4d
            torch.cuda.empty_cache()

        result = output.images[0]
        if result.size != target_size:
            result = result.resize(target_size, PILImage.LANCZOS)
        return result
