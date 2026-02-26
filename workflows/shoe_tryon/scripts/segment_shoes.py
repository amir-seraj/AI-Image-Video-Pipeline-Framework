"""Pipeline code step: segment shoes using GroundingDINO + SAM2.

Detects shoe bounding boxes with GroundingDINO, then segments them
with SAM2 to produce a binary mask. Optimised for AGX Thor unified
memory — both models stay resident and run on GPU without offloading.
"""

import gc

import numpy as np
import torch
from PIL import Image as PILImage

from casadei.media import ImageMedia, Media

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Persistent model cache ────────────────────────────────────────────
_gdino_model = None
_gdino_processor = None
_sam_model = None
_sam_processor = None


def _get_gdino():
    global _gdino_model, _gdino_processor
    if _gdino_model is None:
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        _gdino_processor = AutoProcessor.from_pretrained(
            "IDEA-Research/grounding-dino-base"
        )
        _gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            "IDEA-Research/grounding-dino-base",
        ).to(_DEVICE)
    return _gdino_model, _gdino_processor


def _get_sam():
    global _sam_model, _sam_processor
    if _sam_model is None:
        from transformers import AutoProcessor, AutoModelForMaskGeneration

        _sam_processor = AutoProcessor.from_pretrained(
            "facebook/sam2.1-hiera-large"
        )
        _sam_model = AutoModelForMaskGeneration.from_pretrained(
            "facebook/sam2.1-hiera-large",
        ).to(_DEVICE)
    return _sam_model, _sam_processor


# ── Public entry point ────────────────────────────────────────────────
def process(context: dict[str, Media]) -> dict[str, Media]:
    """Segment shoes in the person image and return a binary mask.

    Reads: context["person"] (ImageMedia)
    Returns: {"mask": ImageMedia} — white where shoes are
    """
    person_media = context["person"]
    if not isinstance(person_media, ImageMedia):
        raise ValueError("Expected 'person' to be an ImageMedia")

    person_image = person_media.image.convert("RGB")
    mask = _detect_and_segment(person_image)

    return {"mask": ImageMedia(image=mask)}


def _detect_and_segment(image: PILImage.Image) -> PILImage.Image:
    """Run GroundingDINO detection then SAM2 segmentation."""
    boxes = _detect_shoes(image)

    if len(boxes) == 0:
        raise RuntimeError(
            "No shoes detected in the image. Make sure the person's "
            "shoes are visible."
        )

    mask = _segment_with_sam2(image, boxes)
    mask = _cleanup_mask(mask)

    return mask


def _detect_shoes(image: PILImage.Image) -> list[list[float]]:
    """Detect shoe bounding boxes using GroundingDINO."""
    model, processor = _get_gdino()

    inputs = processor(images=image, text="shoe.", return_tensors="pt").to(_DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.25,
        text_threshold=0.25,
        target_sizes=[image.size[::-1]],  # (H, W)
    )[0]

    return results["boxes"].cpu().tolist()


def _segment_with_sam2(
    image: PILImage.Image, boxes: list[list[float]]
) -> PILImage.Image:
    """Segment detected regions using SAM2."""
    model, processor = _get_sam()

    input_boxes = [boxes]
    inputs = processor(
        images=image,
        input_boxes=input_boxes,
        return_tensors="pt",
    ).to(_DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.post_process_masks(
        outputs.pred_masks,
        inputs["original_sizes"],
    )

    combined = None
    for mask_set in masks:
        # mask_set shape: (num_boxes, num_predictions, H, W)
        for mask_tensor in mask_set:
            binary = mask_tensor[0].cpu().numpy().astype(bool)
            if combined is None:
                combined = binary
            else:
                combined = combined | binary

    if combined is None:
        raise RuntimeError("SAM2 produced no masks")

    mask_array = (combined * 255).astype(np.uint8)
    return PILImage.fromarray(mask_array, mode="L")


def _cleanup_mask(mask: PILImage.Image) -> PILImage.Image:
    """Apply morphological operations to clean up mask edges."""
    from PIL import ImageFilter

    # Dilate slightly to ensure full shoe coverage
    mask = mask.filter(ImageFilter.MaxFilter(5))
    # Erode slightly to clean edges
    mask = mask.filter(ImageFilter.MinFilter(3))
    # Dilate again for a small safety margin
    mask = mask.filter(ImageFilter.MaxFilter(3))

    return mask
