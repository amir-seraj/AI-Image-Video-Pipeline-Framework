#!/usr/bin/env python3
"""Run VLM analysis directly on all seeded products (no API server needed).

Loads the model once, analyzes all sketches, updates store.json.

Usage:
    python run_vlm_direct.py
"""

import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
STORE_PATH = DATA_DIR / "store.json"
UPLOADS_DIR = DATA_DIR / "uploads"

ANALYZE_PROMPT = (
    "Analyze this shoe design sketch. Provide:\n"
    "1. A comma-separated list of visual feature tags (e.g. pointed toe, stiletto heel, ankle strap)\n"
    "2. A brief description of the design.\n\n"
    "Format your response exactly as:\n"
    "LABELS: tag1, tag2, tag3\n"
    "DESCRIPTION: Your description here"
)


def main():
    import torch
    from PIL import Image

    # Quick CUDA test
    logger.info("Testing CUDA...")
    t = torch.randn(2, 2, device="cuda")
    logger.info("CUDA OK: %s", t @ t)

    # Load store
    store = json.loads(STORE_PATH.read_text())
    products = store["products"]
    logger.info("Found %d products", len(products))

    # Load VLM once
    logger.info("Loading Qwen3-VL-8B...")
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    from casadei import MODELS_DIR

    model_dir = MODELS_DIR / "Qwen3-VL-8B-Instruct"
    processor = Qwen3VLProcessor.from_pretrained(str(model_dir))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    logger.info("Model loaded.")

    updated = 0
    for i, (pid, product) in enumerate(products.items(), 1):
        name = product["name"]
        sketches = product.get("sketches", [])
        if not sketches:
            logger.info("[%d/%d] %s — no sketches, skipping", i, len(products), name)
            continue

        # Find first sketch image
        sketch_dir = UPLOADS_DIR / pid
        image_path = None
        for sketch in sketches:
            sid = sketch["id"]
            for f in sketch_dir.glob(f"{sid}_*"):
                image_path = f
                break
            if image_path:
                break

        if not image_path:
            logger.warning("[%d/%d] %s — sketch file not found, skipping", i, len(products), name)
            continue

        logger.info("[%d/%d] Analyzing: %s", i, len(products), name)

        # Prepare input
        pil_image = Image.open(image_path).convert("RGB")
        content = [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": ANALYZE_PROMPT},
        ]
        messages = [{"role": "user", "content": content}]

        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text_input],
            images=[pil_image],
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # Generate
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=1024)

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # Parse
        label = ""
        description = ""
        for line in response_text.split("\n"):
            line = line.strip()
            if line.upper().startswith("LABELS:"):
                label = line[len("LABELS:"):].strip()
            elif line.upper().startswith("DESCRIPTION:"):
                description = line[len("DESCRIPTION:"):].strip()

        if not label and not description and response_text:
            description = response_text.strip()

        product["label"] = label
        product["description"] = description
        updated += 1

        logger.info("  Label: %s", label)
        logger.info("  Desc:  %s", description[:100])

    # Save
    STORE_PATH.write_text(json.dumps(store, indent=2))
    logger.info("Updated %d products. Saved to %s", updated, STORE_PATH)

    # Cleanup
    del model, processor
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
