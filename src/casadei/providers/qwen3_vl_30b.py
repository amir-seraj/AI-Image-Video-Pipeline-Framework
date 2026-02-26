"""Qwen3-VL-30B-A3B-Instruct vision-language model provider."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from threading import Thread

import torch
from PIL import Image as PILImage

from casadei import MODELS_DIR
from casadei.models.base import ModelCapability, ImageConstraint, TextConstraint
from casadei.models.vision_language import VisionLanguageModel

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
LOCAL_DIR = MODELS_DIR / "Qwen3-VL-30B-A3B-Instruct"


def download_model() -> None:
    """Download the model to local dir if not already present."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {MODEL_ID} (~58 GB) ...")
    snapshot_download(MODEL_ID, local_dir=str(LOCAL_DIR))
    print("Download complete.")

# Config files that must be valid JSON/text (not HTML from a bad download).
_CONFIG_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "generation_config.json",
    "chat_template.json",
    "model.safetensors.index.json",
    "vocab.json",
    "merges.txt",
    "video_preprocessor_config.json",
]


def _fix_corrupt_configs(model_dir: Path) -> None:
    """Re-download config files that are corrupt HTML pages."""
    from huggingface_hub import hf_hub_download

    to_fix: list[str] = []
    for name in _CONFIG_FILES:
        fpath = model_dir / name
        if not fpath.exists():
            to_fix.append(name)
            continue
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
            if content.lstrip().startswith("<!doctype") or content.lstrip().startswith("<html"):
                to_fix.append(name)
        except Exception:
            to_fix.append(name)

    if not to_fix:
        logger.info("All config files OK.")
        return

    logger.info("Re-downloading %d corrupt config files: %s", len(to_fix), to_fix)
    for name in to_fix:
        fpath = model_dir / name
        if fpath.exists():
            fpath.unlink()
        try:
            hf_hub_download(
                MODEL_ID,
                filename=name,
                local_dir=str(model_dir),
                force_download=True,
            )
            logger.info("  Downloaded %s", name)
        except Exception as exc:
            logger.warning("  Failed to download %s: %s", name, exc)


class Qwen3VL30B(VisionLanguageModel):
    """Qwen3-VL-30B-A3B-Instruct vision-language model.

    30B-parameter MoE model (~3B active per token).
    Accepts images and a text prompt, produces a text response.
    """

    capability = ModelCapability(
        inputs=[
            ImageConstraint(
                required=True,
                max_count=4,
                supported_formats=["png", "jpg", "jpeg", "webp"],
            ),
            TextConstraint(required=True),
        ],
        outputs=[
            TextConstraint(required=True, max_count=1),
        ],
    )

    DEFAULT_PARAMS = {
        "max_new_tokens": 1024,
    }

    def __init__(self) -> None:
        super().__init__()
        self._model = None
        self._processor = None

    def load_model(self) -> None:
        from transformers import Qwen3VLMoeForConditionalGeneration, Qwen3VLProcessor

        _fix_corrupt_configs(LOCAL_DIR)

        logger.info("Loading Qwen3-VL-30B-A3B-Instruct from %s", LOCAL_DIR)
        self._processor = Qwen3VLProcessor.from_pretrained(str(LOCAL_DIR))
        self._model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            str(LOCAL_DIR),
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            attn_implementation="sdpa",
        )
        logger.info("Qwen3-VL-30B-A3B-Instruct loaded.")

    def unload_model(self) -> None:
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prepare_inputs(
        self, images: list[PILImage.Image], prompt: str
    ) -> dict:
        """Build chat messages, apply template, and tokenize."""
        content: list[dict] = []
        for img in images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        text_input = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self._processor(
            text=[text_input],
            images=images if images else None,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

    def _generate_text(
        self,
        images: list[PILImage.Image],
        prompt: str,
        **kwargs,
    ) -> str:
        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        inputs = self._prepare_inputs(images, prompt)

        params = {**self.DEFAULT_PARAMS, **kwargs}
        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **params)

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        result = self._processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return result.strip()

    def _generate_text_streaming(
        self,
        images: list[PILImage.Image],
        prompt: str,
        **kwargs,
    ) -> Iterator[str]:
        """Yield token chunks as they are generated."""
        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        from transformers import TextIteratorStreamer

        inputs = self._prepare_inputs(images, prompt)

        streamer = TextIteratorStreamer(
            self._processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        params = {**self.DEFAULT_PARAMS, **kwargs, "streamer": streamer}
        generation_kwargs = {**inputs, **params}

        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        for chunk in streamer:
            if chunk:
                yield chunk

        thread.join()


if __name__ == "__main__":
    download_model()
