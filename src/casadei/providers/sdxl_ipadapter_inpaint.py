"""SDXL Inpainting with IP-Adapter reference conditioning.

Optimised for AGX Thor unified memory — all components stay on GPU
without CPU offloading. Uses crop-inpaint-composite for high-detail
results on small inpaint regions.
"""

from __future__ import annotations

import gc

import numpy as np
import torch
from PIL import Image as PILImage, ImageFilter

from casadei.models.base import ModelCapability, ImageConstraint, TextConstraint
from casadei.models.reference_inpaint import ReferenceInpaintModel

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_INPAINT_SIZE = 1024  # SDXL optimal resolution


class SDXLIPAdapterInpaint(ReferenceInpaintModel):
    """Reference-conditioned inpainting using SDXL Inpainting + IP-Adapter.

    Uses crop-inpaint-composite: each masked region is cropped, upscaled to
    1024x1024 for inpainting, then composited back at original resolution.
    This gives small regions (like shoes) full SDXL detail.
    """

    capability = ModelCapability(
        inputs=[
            ImageConstraint(required=True, max_count=3),
            TextConstraint(required=False, max_count=2),
        ],
        outputs=[
            ImageConstraint(max_count=1),
        ],
    )

    DEFAULT_PARAMS = {
        "num_inference_steps": 40,
        "guidance_scale": 7.5,
        "ip_adapter_scale": 0.95,
        "strength": 0.99,
        "crop_padding": 0.4,  # 40% padding around mask bbox
    }

    def __init__(self) -> None:
        super().__init__()
        self._pipe = None
        self._image_encoder = None

    def load_model(self) -> None:
        from diffusers import StableDiffusionXLInpaintPipeline, AutoencoderKL
        from transformers import CLIPVisionModelWithProjection

        vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix",
            torch_dtype=torch.float16,
        )

        self._image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
        )

        self._pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            vae=vae,
            image_encoder=self._image_encoder,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        self._pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
        )

        self._pipe.to(_DEVICE)

    def unload_model(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
        if self._image_encoder is not None:
            del self._image_encoder
            self._image_encoder = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _inpaint(
        self,
        image: PILImage.Image,
        mask: PILImage.Image,
        reference: PILImage.Image,
        prompt: str,
        negative_prompt: str,
        **kwargs,
    ) -> PILImage.Image:
        if self._pipe is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        ip_adapter_scale = kwargs.get(
            "ip_adapter_scale", self.DEFAULT_PARAMS["ip_adapter_scale"]
        )
        self._pipe.set_ip_adapter_scale(ip_adapter_scale)

        image = image.convert("RGB")
        mask = mask.convert("L")
        reference = reference.convert("RGB")

        if mask.size != image.size:
            mask = mask.resize(image.size, PILImage.NEAREST)

        strength = kwargs.get("strength", self.DEFAULT_PARAMS["strength"])
        steps = kwargs.get("num_inference_steps", self.DEFAULT_PARAMS["num_inference_steps"])
        guidance = kwargs.get("guidance_scale", self.DEFAULT_PARAMS["guidance_scale"])
        padding = kwargs.get("crop_padding", self.DEFAULT_PARAMS["crop_padding"])

        prompt = prompt or "a high quality photo of a shoe on a foot, photorealistic, detailed"
        negative_prompt = negative_prompt or "blurry, distorted, low quality, cartoon, deformed"

        # Find connected mask regions (individual shoes)
        regions = _find_mask_regions(mask)

        if not regions:
            # Fallback: inpaint the whole image
            return self._inpaint_region(
                image, mask, reference, prompt, negative_prompt,
                strength, steps, guidance,
            )

        # Crop-inpaint-composite each region
        result = image.copy()
        for bbox in regions:
            crop_box = _padded_square_crop(bbox, image.size, padding)
            result = self._inpaint_crop_composite(
                result, mask, reference, crop_box,
                prompt, negative_prompt, strength, steps, guidance,
            )

        return result

    def _inpaint_region(
        self, image, mask, reference, prompt, neg_prompt, strength, steps, guidance,
    ) -> PILImage.Image:
        """Inpaint at the given image size (no cropping)."""
        result = self._pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            image=image,
            mask_image=mask,
            ip_adapter_image=reference,
            num_inference_steps=steps,
            guidance_scale=guidance,
            strength=strength,
        )
        return result.images[0]

    def _inpaint_crop_composite(
        self, image, mask, reference, crop_box,
        prompt, neg_prompt, strength, steps, guidance,
    ) -> PILImage.Image:
        """Crop region, inpaint at 1024x1024, composite back."""
        x1, y1, x2, y2 = crop_box
        crop_w, crop_h = x2 - x1, y2 - y1

        # Crop image and mask
        img_crop = image.crop(crop_box)
        mask_crop = mask.crop(crop_box)

        # Scale to SDXL optimal size
        img_scaled = img_crop.resize((_INPAINT_SIZE, _INPAINT_SIZE), PILImage.LANCZOS)
        mask_scaled = mask_crop.resize((_INPAINT_SIZE, _INPAINT_SIZE), PILImage.NEAREST)

        # Light feather on mask edges
        mask_scaled = _feather_mask(mask_scaled, radius=6)

        # Run inpainting at full resolution
        result_scaled = self._pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            image=img_scaled,
            mask_image=mask_scaled,
            ip_adapter_image=reference,
            num_inference_steps=steps,
            guidance_scale=guidance,
            strength=strength,
            height=_INPAINT_SIZE,
            width=_INPAINT_SIZE,
        ).images[0]

        # Scale back to crop size and composite
        result_crop = result_scaled.resize((crop_w, crop_h), PILImage.LANCZOS)

        # Use the original mask (not feathered) for compositing
        # to keep sharp boundary with original image
        composite_mask = mask_crop.resize((crop_w, crop_h), PILImage.NEAREST)
        # Slight blur on composite mask for seamless blending
        composite_mask = composite_mask.filter(ImageFilter.GaussianBlur(3))

        # Paste inpainted crop back
        output = image.copy()
        region_bg = image.crop(crop_box)
        blended = PILImage.composite(result_crop, region_bg, composite_mask)
        output.paste(blended, (x1, y1))

        return output


def _find_mask_regions(mask: PILImage.Image) -> list[tuple[int, int, int, int]]:
    """Find bounding boxes of connected white regions in the mask.

    Uses a simple row-scan flood fill — no scipy dependency.
    """
    mask_arr = np.array(mask)
    binary = (mask_arr > 128).astype(np.uint8)

    if binary.sum() == 0:
        return []

    h, w = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    regions = []

    for y in range(h):
        for x in range(w):
            if binary[y, x] and not visited[y, x]:
                # BFS flood fill
                stack = [(y, x)]
                visited[y, x] = True
                min_x, max_x, min_y, max_y = x, x, y, y

                while stack:
                    cy, cx = stack.pop()
                    min_x = min(min_x, cx)
                    max_x = max(max_x, cx)
                    min_y = min(min_y, cy)
                    max_y = max(max_y, cy)

                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

                area = (max_x - min_x) * (max_y - min_y)
                if area >= 50:  # skip tiny noise
                    regions.append((min_x, min_y, max_x, max_y))

    return regions


def _padded_square_crop(
    bbox: tuple[int, int, int, int],
    image_size: tuple[int, int],
    padding: float = 0.4,
) -> tuple[int, int, int, int]:
    """Expand bbox with padding and make it square, clamped to image bounds."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    img_w, img_h = image_size

    # Add padding
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x1 -= pad_x
    y1 -= pad_y
    x2 += pad_x
    y2 += pad_y

    # Make square (use the longer side)
    side = max(x2 - x1, y2 - y1)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    x1 = cx - side // 2
    y1 = cy - side // 2
    x2 = x1 + side
    y2 = y1 + side

    # Clamp to image bounds
    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > img_w:
        x1 -= (x2 - img_w)
        x2 = img_w
    if y2 > img_h:
        y1 -= (y2 - img_h)
        y2 = img_h

    x1 = max(0, x1)
    y1 = max(0, y1)

    return (x1, y1, x2, y2)


def _feather_mask(mask: PILImage.Image, radius: int = 16) -> PILImage.Image:
    """Gaussian-blur the mask edges for smoother inpainting transitions."""
    return mask.filter(ImageFilter.GaussianBlur(radius))
