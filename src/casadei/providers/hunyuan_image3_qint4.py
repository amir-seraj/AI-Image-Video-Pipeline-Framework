"""HunyuanImage-3.0 QInt4 provider.

Loads wikeeyang/Hunyuan-Image-30-Qint4 — an optimum-quanto QInt4 (4-bit integer)
quantized version of Tencent's 80B MoE base image generation model.
Uses ~50-60 GB GPU memory, fits on Jetson Thor (128 GB unified).

Uses the model's bundled load_quantized_model.py (unofficial loading code)
with Method 2 (~75 GB CPU peak, vs ~160 GB for Method 1).

Download + first run:  python src/casadei/providers/hunyuan_image3_qint4.py
"""

from __future__ import annotations

import importlib.util
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

MODEL_ID = "wikeeyang/Hunyuan-Image-30-Qint4"
LOCAL_DIR = MODELS_DIR / "Hunyuan-Image-30-Qint4"


def download_model() -> Path:
    """Download the QInt4 model to LOCAL_DIR if not already present."""
    from huggingface_hub import snapshot_download

    existing = sorted(LOCAL_DIR.glob("model-*-of-*.safetensors"))
    if existing:
        print(f"Found {len(existing)} shard(s) at {LOCAL_DIR}")
        verify_safetensors(MODEL_ID, LOCAL_DIR)
        return LOCAL_DIR

    print(f"Downloading {MODEL_ID} to {LOCAL_DIR} ...")
    print("This is ~40-50 GB (QInt4 pre-quantized) and will take a while.")
    snapshot_download(MODEL_ID, local_dir=str(LOCAL_DIR))
    print(f"Download complete: {LOCAL_DIR}")
    verify_safetensors(MODEL_ID, LOCAL_DIR)
    return LOCAL_DIR


def _import_load_quantized(model_path: Path):
    """Import load_quantized_model.py from the model's local directory.

    NOTE: does NOT touch sys.path — caller must add model_path to sys.path
    before calling this, and keep it there until load_quantized_hi3_m* returns,
    because the loader imports hunyuan_image_3 lazily inside those functions.
    """
    loader_path = model_path / "load_quantized_model.py"
    if not loader_path.exists():
        raise FileNotFoundError(
            f"load_quantized_model.py not found at {loader_path}. "
            "Make sure the model was downloaded with download_model()."
        )
    spec = importlib.util.spec_from_file_location("load_quantized_model", loader_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class HunyuanImage3QInt4(ImageEditModel):
    """HunyuanImage-3.0 — QInt4 quantized (optimum-quanto 4-bit integer).

    80B MoE base image generation model with QInt4 quantization via
    optimum-quanto. Uses ~50-60 GB GPU memory. Loaded with Method 2
    (meta-device + requantize on CPU) to keep CPU peak at ~75 GB.
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
        "diff_infer_steps": 50,
        "seed": 42,
        "image_size": "auto",
        "use_system_prompt": "en_unified",
        "bot_task": "think_recaption",
    }

    MIN_STEPS = 1
    MAX_STEPS = 100

    def __init__(self) -> None:
        super().__init__()
        self._model = None

    _CUDA_MEM_FRACTION = 0.90

    @staticmethod
    def _load_quantized_to_gpu(model_path: Path, model_path_str: str):
        """Load QInt4 model to CUDA, skipping slow CPU TinyGemm reformat.

        The bundled load_quantized_hi3_m2 requantizes to CPU (packing in
        TinyGemm format), then .to('cuda') must unpack/repack every weight
        — 15-20+ min on 80B (optimum-quanto issue #270 / #367).

        We use requantize(device='cuda') which moves the empty model
        structure to CUDA first, then loads weights directly in CUDA format.
        No CPU-side TinyGemm reformatting ever happens.

        On Jetson unified memory (CPU+GPU share same RAM pool), we limit
        CUDA to 50% during loading so the state_dict can stay in physical
        RAM alongside the CUDA model.  Fraction is restored after loading.
        """
        import gc
        import json
        from safetensors.torch import load_file
        from optimum.quanto import requantize
        from transformers import AutoConfig
        from transformers.generation.utils import GenerationConfig

        from hunyuan_image_3.hunyuan import HunyuanImage3ForCausalMM

        total_gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

        # On Jetson unified memory, limit CUDA allocation during loading
        # so the state_dict swap-in has room in physical memory.
        # Model needs ~68 GB CUDA; state_dict needs ~47 GB CPU (mmap'd).
        # Limiting CUDA to 60% (~74 GB) leaves ~49 GB for CPU/OS.
        load_fraction = 0.60
        torch.cuda.set_per_process_memory_fraction(load_fraction, device=0)
        print(f"CUDA limit set to {load_fraction*100:.0f}% "
              f"({total_gpu_gb * load_fraction:.0f} GB) during loading")

        config = AutoConfig.from_pretrained(model_path_str, trust_remote_code=True)

        print("Loading safetensors state_dict to CPU...")
        state_dict = load_file(f"{model_path_str}/model.safetensors", device="cpu")
        with open(f"{model_path_str}/quantization_map.json") as f:
            quantization_map = json.load(f)
        print(f"State dict loaded: {len(state_dict)} tensors")

        print("Creating meta model...")
        with torch.device("meta"):
            qmodel = HunyuanImage3ForCausalMM(config)
        qmodel = qmodel.to(torch.bfloat16)

        print("Requantizing with device=cuda (bypasses CPU TinyGemm reformat)...")
        requantize(qmodel, state_dict, quantization_map, device=torch.device("cuda"))

        del state_dict, quantization_map
        gc.collect()
        torch.cuda.empty_cache()

        generation_config = GenerationConfig.from_pretrained(model_path_str)
        qmodel.generation_config = generation_config

        final_gb = torch.cuda.memory_allocated() / 1024**3
        print(f"Model loaded to GPU. CUDA usage: {final_gb:.1f} GB")
        return qmodel

    def load_model(self) -> None:
        model_path = LOCAL_DIR if LOCAL_DIR.exists() else None
        if model_path is None:
            raise RuntimeError(
                f"{LOCAL_DIR} not found. Run download_model() or "
                "execute: python src/casadei/providers/hunyuan_image3_qint4.py"
            )

        verify_safetensors(MODEL_ID, model_path)

        logger.info("Loading %s with QInt4 (optimum-quanto, method 2)...", MODEL_ID)
        model_path_str = str(model_path)

        # load_quantized_model.py imports `from hunyuan_image_3.hunyuan import ...`
        # but the model ships with flat .py files (hunyuan.py, etc.), not a package.
        # Create a virtual hunyuan_image_3 namespace package whose __path__ points
        # at the model dir so submodule lookups find the flat files automatically.
        import types as _types
        _pkg_name = "hunyuan_image_3"
        if _pkg_name not in sys.modules:
            _pkg = _types.ModuleType(_pkg_name)
            _pkg.__path__ = [model_path_str]
            _pkg.__package__ = _pkg_name
            sys.modules[_pkg_name] = _pkg

        sys.path.insert(0, model_path_str)
        try:
            model = self._load_quantized_to_gpu(model_path, model_path_str)
            model.load_tokenizer(model_path_str)
        finally:
            try:
                sys.path.remove(model_path_str)
            except ValueError:
                pass

        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(
                self._CUDA_MEM_FRACTION, device=0
            )

        # --- Safety-net monkey-patches for upstream model bugs ----------
        import importlib as _importlib
        mod = _importlib.import_module(type(model).__module__)

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
            "diff_infer_steps": params.get("diff_infer_steps", 50),
        }

        clamp_steps(gen_kwargs, "diff_infer_steps", self.MIN_STEPS, self.MAX_STEPS)

        if images:
            gen_kwargs["image"] = images
            gen_kwargs["infer_align_image_size"] = True

        cot_text, samples = self._model.generate_image(**gen_kwargs)
        return samples[0]


if __name__ == "__main__":
    download_model()
