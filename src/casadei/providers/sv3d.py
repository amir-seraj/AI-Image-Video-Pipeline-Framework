"""SV3D (Stable Video 3D) multi-view generation provider.

Generates 21 views from a single image using stabilityai/sv3d (SV3D_p variant).
Uses pose-conditioned generation for controllable azimuth/elevation angles.

On Blackwell GPUs (SM 110) / CUDA 13.0, CUBLAS crashes in linear layers
inside CLIP and VAE. We offload those to CPU and keep only the UNet on GPU.
"""

from __future__ import annotations

import inspect
import logging
import types

import numpy as np
import torch
from PIL import Image as PILImage

from casadei import MODELS_DIR
from casadei.models.base import ModelCapability, ImageConstraint
from casadei.models.image_to_multiview import ImageToMultiViewModel

try:
    from diffusers import AutoencoderKL, EulerDiscreteScheduler
    from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
    from diffusers_sv3d import (
        SV3DUNetSpatioTemporalConditionModel,
        StableVideo3DDiffusionPipeline,
    )
except ImportError:
    AutoencoderKL = None
    StableVideo3DDiffusionPipeline = None

logger = logging.getLogger(__name__)

SV3D_DIFFUSERS = "chenguolin/sv3d-diffusers"
SV3D_RES = 576
SV3D_NUM_FRAMES = 21


def _patch_cpu_offload(pipe, cuda_dev: torch.device):
    """Move CLIP image_encoder and VAE to CPU, patch pipeline methods.

    CUBLAS on Blackwell (SM 110) crashes in linear layers of CLIP and VAE.
    The UNet stays on CUDA for denoising. We override _encode_image,
    _encode_vae_image, and decode_latents to run on CPU.
    """
    unet_dtype = pipe.unet.dtype
    pipe.image_encoder = pipe.image_encoder.to("cpu", dtype=torch.float32)
    pipe.vae = pipe.vae.to("cpu", dtype=torch.float32)

    # Override _execution_device so the pipeline creates latents/noise on CUDA
    # (default implementation checks the first module, which is now image_encoder on CPU)
    pipe.__class__._execution_device = property(lambda self: cuda_dev)

    logger.info("SV3D: CLIP image_encoder and VAE moved to CPU (Blackwell workaround)")

    # --- Patch _encode_image to run CLIP on CPU ---
    original_encode_image = pipe._encode_image.__func__

    def _encode_image_cpu(self, image, device, num_videos_per_prompt, do_classifier_free_guidance):
        """Run CLIP on CPU, then move embeddings to CUDA."""
        embeddings = original_encode_image(self, image, "cpu", num_videos_per_prompt, do_classifier_free_guidance)
        return embeddings.to(device=cuda_dev, dtype=unet_dtype)

    pipe._encode_image = types.MethodType(_encode_image_cpu, pipe)

    # --- Patch _encode_vae_image to run VAE on CPU ---
    def _encode_vae_image_cpu(self, image, device, num_videos_per_prompt, do_classifier_free_guidance):
        """Run VAE encode on CPU, return latents on CUDA."""
        image_cpu = image.to("cpu", dtype=torch.float32)
        image_latents = self.vae.encode(image_cpu).latent_dist.mode()
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1)
        if do_classifier_free_guidance:
            negative = torch.zeros_like(image_latents)
            image_latents = torch.cat([negative, image_latents])
        return image_latents.to(device=cuda_dev, dtype=unet_dtype)

    pipe._encode_vae_image = types.MethodType(_encode_vae_image_cpu, pipe)

    # --- Patch decode_latents to run VAE on CPU ---
    from diffusers.utils.torch_utils import is_compiled_module

    def decode_latents_cpu(self, latents, num_frames, decode_chunk_size=14):
        """Decode on CPU, return frames."""
        latents = latents.flatten(0, 1).to("cpu", dtype=torch.float32)
        latents = latents / self.vae.config.scaling_factor

        forward_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        accepts_num_frames = "num_frames" in set(inspect.signature(forward_fn).parameters.keys())

        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            chunk = latents[i : i + decode_chunk_size]
            decode_kwargs = {}
            if accepts_num_frames:
                decode_kwargs["num_frames"] = chunk.shape[0]
            frame = self.vae.decode(chunk, **decode_kwargs).sample
            frames.append(frame)

        frames = torch.cat(frames, dim=0)
        frames = frames.reshape(-1, num_frames, *frames.shape[1:]).float()
        return frames

    pipe.decode_latents = types.MethodType(decode_latents_cpu, pipe)


class SV3D(ImageToMultiViewModel):
    """Stable Video 3D (SV3D_p) multi-view generation.

    Takes a single image and generates 21 views at evenly-spaced azimuth
    angles with controllable elevation. Output is 21 individual 576x576 frames.
    """

    capability = ModelCapability(
        inputs=[
            ImageConstraint(
                required=True,
                max_count=1,
                supported_formats=["png", "jpg", "jpeg", "webp"],
            ),
        ],
        outputs=[
            ImageConstraint(required=True, max_count=21),
        ],
    )

    DEFAULT_PARAMS = {
        "elevation": 10.0,
        "decode_chunk_size": 8,
    }

    def __init__(self) -> None:
        super().__init__()
        self._pipeline = None

    def load_model(self) -> None:
        if StableVideo3DDiffusionPipeline is None:
            raise ImportError(
                "diffusers_sv3d is required. Install from: "
                "https://github.com/chenguolin/sv3d-diffusers"
            )

        logger.info("Loading SV3D components from %s ...", SV3D_DIFFUSERS)

        unet = SV3DUNetSpatioTemporalConditionModel.from_pretrained(
            SV3D_DIFFUSERS, subfolder="unet", cache_dir=MODELS_DIR,
        )
        vae = AutoencoderKL.from_pretrained(
            SV3D_DIFFUSERS, subfolder="vae", cache_dir=MODELS_DIR,
        )
        scheduler = EulerDiscreteScheduler.from_pretrained(
            SV3D_DIFFUSERS, subfolder="scheduler", cache_dir=MODELS_DIR,
        )
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            SV3D_DIFFUSERS, subfolder="image_encoder", cache_dir=MODELS_DIR,
        )
        feature_extractor = CLIPImageProcessor.from_pretrained(
            SV3D_DIFFUSERS, subfolder="feature_extractor", cache_dir=MODELS_DIR,
        )

        pipe = StableVideo3DDiffusionPipeline(
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            unet=unet,
            vae=vae,
            scheduler=scheduler,
        )

        if torch.cuda.is_available():
            # Move entire pipeline to CUDA first, then offload CLIP+VAE to CPU
            pipe = pipe.to("cuda")
            _patch_cpu_offload(pipe, torch.device("cuda"))
            logger.info("SV3D loaded (UNet on CUDA, CLIP+VAE on CPU)")
        else:
            logger.info("SV3D loaded on CPU")

        self._pipeline = pipe

    def unload_model(self) -> None:
        self._pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("SV3D unloaded")

    def _generate_views(
        self,
        image: PILImage.Image,
        num_views: int = 21,
        **kwargs,
    ) -> list[PILImage.Image]:
        if self._pipeline is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        elevation = kwargs.pop("elevation", self.DEFAULT_PARAMS["elevation"])
        decode_chunk_size = kwargs.pop(
            "decode_chunk_size", self.DEFAULT_PARAMS["decode_chunk_size"]
        )

        # Prepare input: RGBA -> RGB on white background, resize to 576x576
        image.load()
        if image.mode == "RGBA":
            rgb = PILImage.new("RGB", image.size, (255, 255, 255))
            rgb.paste(image, mask=image.split()[3])
            image = rgb
        elif image.mode != "RGB":
            image = image.convert("RGB")

        image = image.resize((SV3D_RES, SV3D_RES), PILImage.LANCZOS)

        # Compute camera angles for SV3D_p
        num_frames = SV3D_NUM_FRAMES
        elevations_deg = [elevation] * num_frames
        polars_rad = [np.deg2rad(90 - e) for e in elevations_deg]
        azimuths_deg = np.linspace(0, 360, num_frames + 1)[1:] % 360
        azimuths_rad = [
            np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg
        ]
        azimuths_rad[:-1].sort()

        with torch.no_grad():
            result = self._pipeline(
                image,
                height=SV3D_RES,
                width=SV3D_RES,
                num_frames=num_frames,
                decode_chunk_size=decode_chunk_size,
                polars_rad=polars_rad,
                azimuths_rad=azimuths_rad,
            )

        views = result.frames[0]  # list of PIL images
        return views[:num_views]
