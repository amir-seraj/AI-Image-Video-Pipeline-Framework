"""HunyuanImage-3.0-Instruct-Distil INT8 provider.

Loads EricRollei/HunyuanImage-3.0-Instruct-Distil-INT8 — an INT8 quantized
version of Tencent's 83B MoE distilled image generation model. Uses ~85 GB
memory, fits on Jetson Thor (128 GB unified).

Distil variant: only 8 diffusion steps (vs 50 for Instruct), CFG-distilled
(no classifier-free guidance needed). ~6x faster than the full Instruct model.

Uses bitsandbytes INT8 quantization with standard transformers API.
No FlashInfer dependency — uses eager MoE + PyTorch SDPA attention.

Download + first run:  python src/casadei/providers/hunyuan_image3_distil_int8.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
from PIL import Image as PILImage

from casadei import MODELS_DIR
from casadei.models.base import ModelCapability, ImageConstraint, TextConstraint
from casadei.models.image_edit import ImageEditModel
from casadei.providers._base import verify_safetensors, clamp_steps

logger = logging.getLogger(__name__)

MODEL_ID = "EricRollei/HunyuanImage-3.0-Instruct-Distil-INT8"


def download_model() -> None:
    """Download the Distil-INT8 model to HF cache if not already present."""
    from huggingface_hub import snapshot_download

    print(f"Downloading {MODEL_ID} (~80 GB) ...")
    snapshot_download(MODEL_ID, cache_dir=MODELS_DIR)
    verify_safetensors(MODEL_ID, MODELS_DIR)
    print("Download complete.")


class HunyuanImage3DistilINT8(ImageEditModel):
    """HunyuanImage-3.0-Instruct-Distil — INT8 quantized (bitsandbytes 8-bit).

    83B MoE distilled model with INT8 quantization on FFN/expert layers,
    attention projections and VAE kept in BF16. Uses ~85 GB memory.
    CFG-distilled: only 8 diffusion steps, no guidance needed.
    Supports text-to-image and image editing via generate_image().
    """

    capability = ModelCapability(
        inputs=[
            ImageConstraint(
                required=False,
                max_count=4,
                supported_formats=["png", "jpg", "jpeg", "webp"],
            ),
            TextConstraint(required=True),
        ],
        outputs=[
            ImageConstraint(required=True, max_count=1),
        ],
    )

    DEFAULT_PARAMS = {
        "diff_infer_steps": 8,
        "seed": 42,
        "image_size": "auto",
        "use_system_prompt": "en_unified",
        "bot_task": "think_recaption",
    }

    MIN_STEPS = 1
    MAX_STEPS = 20

    def __init__(self) -> None:
        super().__init__()
        self._model = None

    _CUDA_MEM_FRACTION = 0.90

    def load_model(self) -> None:
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        verify_safetensors(MODEL_ID, MODELS_DIR)

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )

        max_memory = None
        if torch.cuda.is_available():
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            max_memory = {0: f"{int(total_gb * 0.90)}GiB"}

        logger.info("Loading %s with INT8 quantization...", MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory=max_memory,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            cache_dir=MODELS_DIR,
        )
        model.load_tokenizer(MODEL_ID)

        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(
                self._CUDA_MEM_FRACTION, device=0
            )

        # --- Safety-net monkey-patches for upstream model bugs ----------
        import importlib
        mod = importlib.import_module(type(model).__module__)

        # Bug 1: to_device() must recurse into dicts (cond_vit_image_kwargs).
        _orig_to_device = mod.to_device

        def _patched_to_device(data, device):
            if isinstance(data, dict):
                return {k: _patched_to_device(v, device) for k, v in data.items()}
            return _orig_to_device(data, device)

        mod.to_device = _patched_to_device

        # Bug 2: lazy_initialization(key_states) needs value_states too
        # (transformers 5.x API change).
        from transformers.cache_utils import StaticLayer
        _orig_lazy_init = StaticLayer.lazy_initialization

        def _patched_lazy_init(self, key_states, value_states=None):
            if value_states is None:
                value_states = key_states
            return _orig_lazy_init(self, key_states, value_states)

        StaticLayer.lazy_initialization = _patched_lazy_init

        self._model = model
        logger.info("Model loaded. GPU mem: %.2f GB", torch.cuda.memory_allocated() / 1024**3)

    def unload_model(self) -> None:
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(1.0, device=0)

    def _edit(
        self,
        images: list[PILImage.Image],
        prompt: str,
        negative_prompt: str,
        **kwargs,
    ) -> PILImage.Image:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        params = {**self.DEFAULT_PARAMS, **kwargs}

        gen_kwargs = {
            "prompt": prompt,
            "seed": params.get("seed", 42),
            "image_size": params.get("image_size", "auto"),
            "use_system_prompt": params.get("use_system_prompt", "en_unified"),
            "bot_task": params.get("bot_task", "think_recaption"),
            "diff_infer_steps": params.get("diff_infer_steps", 8),
        }

        clamp_steps(gen_kwargs, "diff_infer_steps", self.MIN_STEPS, self.MAX_STEPS)

        if images:
            gen_kwargs["image"] = images
            gen_kwargs["infer_align_image_size"] = True

        cot_text, samples = self._model.generate_image(**gen_kwargs)
        return samples[0]


if __name__ == "__main__":
    download_model()
