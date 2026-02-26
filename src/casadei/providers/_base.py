"""Shared utilities for image-edit providers.

Functions extracted from individual providers to eliminate code duplication:
- safetensors verification and repair
- inference step clamping
- step-saving callback for deferred VAE decode
- batch VAE decode of captured latents
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
from PIL import Image as PILImage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Safetensors verification / repair
# ---------------------------------------------------------------------------

def verify_safetensors(
    model_id: str,
    path: Path,
    *,
    shard_pattern: str = "model-*-of-*.safetensors",
    max_retries: int = 5,
) -> None:
    """Validate safetensors shards and re-download any that are corrupt.

    Supports two directory layouts:

    - **HF cache**: *path* is the cache root (e.g. ``MODELS_DIR``).
      Shards live under ``models--{org}--{name}/snapshots/{hash}/``.
    - **Local directory**: *path* contains safetensors files directly.

    If no shards matching *shard_pattern* are found the function returns
    silently (the model hasn't been downloaded yet).
    """
    from safetensors import safe_open
    from huggingface_hub import hf_hub_download

    path = Path(path)

    # Auto-detect HF cache vs local directory layout.
    repo_dir = path / f"models--{model_id.replace('/', '--')}"
    snap_dir = repo_dir / "snapshots"

    if snap_dir.is_dir():
        snap_dirs = list(snap_dir.iterdir())
        if not snap_dirs:
            return
        shard_root = snap_dirs[0]
        is_hf_cache = True
    elif list(path.glob(shard_pattern)):
        shard_root = path
        is_hf_cache = False
    else:
        return  # nothing cached yet

    shards = sorted(shard_root.glob(shard_pattern))
    if not shards:
        return

    # First pass — validate every shard
    corrupt: list[str] = []
    for shard in shards:
        try:
            with safe_open(str(shard), framework="pt") as sf:
                _ = sf.keys()
        except Exception:
            corrupt.append(shard.name)

    if not corrupt:
        print(f"All {len(shards)} shards OK.")
        return

    print(f"Found {len(corrupt)}/{len(shards)} corrupt shards — re-downloading...")

    for idx, name in enumerate(corrupt, 1):
        # Remove corrupt file (handle HF cache symlinks)
        fpath = shard_root / name
        if fpath.is_symlink():
            target = fpath.resolve()
            fpath.unlink()
            if target.exists():
                target.unlink()
        elif fpath.exists():
            fpath.unlink()

        for attempt in range(1, max_retries + 1):
            try:
                print(
                    f"  [{idx}/{len(corrupt)}] Downloading {name} "
                    f"(attempt {attempt}/{max_retries}) ...",
                    flush=True,
                )
                dl_kwargs: dict = {"force_download": True}
                if is_hf_cache:
                    dl_kwargs["cache_dir"] = str(path)
                else:
                    dl_kwargs["local_dir"] = str(path)
                hf_hub_download(model_id, filename=name, **dl_kwargs)
                break
            except Exception as e:
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Failed to download {name} after "
                        f"{max_retries} attempts: {e}"
                    ) from e
                wait = 10 * attempt
                print(f"    Error: {e}. Retrying in {wait}s...", flush=True)
                time.sleep(wait)

    # Second pass — verify repairs
    still_bad = []
    for name in corrupt:
        shard = shard_root / name
        try:
            with safe_open(str(shard), framework="pt") as sf:
                _ = sf.keys()
        except Exception:
            still_bad.append(name)

    if still_bad:
        raise RuntimeError(
            f"Shards still corrupt after re-download: {still_bad}"
        )
    print(f"Re-downloaded {len(corrupt)} shards. All verified OK.")


# ---------------------------------------------------------------------------
# Inference step clamping
# ---------------------------------------------------------------------------

def clamp_steps(
    params: dict,
    key: str,
    min_steps: int,
    max_steps: int,
) -> None:
    """Clamp inference steps in *params* to ``[min_steps, max_steps]``.

    Prints a console warning when clamping occurs.  Modifies *params*
    in place.
    """
    steps = params.get(key)
    if steps is None:
        return
    if steps < min_steps or steps > max_steps:
        clamped = max(min_steps, min(max_steps, steps))
        print(
            f"  WARNING: {key}={steps} out of range "
            f"[{min_steps}, {max_steps}], clamped to {clamped}"
        )
        params[key] = clamped


# ---------------------------------------------------------------------------
# Step-saving callback + deferred VAE decode
# ---------------------------------------------------------------------------

def make_step_callback(
    saved_latents: list[tuple[int, torch.Tensor]],
    interval: int,
):
    """Return a ``callback_on_step_end`` closure that captures latents.

    Appends ``(step_index, cpu_latents)`` to *saved_latents* every
    *interval* steps.  Pair with :func:`decode_and_save_step_latents`
    for deferred VAE decode after inference completes.
    """

    def callback_on_step_end(_pipe, step_index, timestep, callback_kwargs):
        if step_index % interval != 0:
            return callback_kwargs
        saved_latents.append(
            (step_index, callback_kwargs["latents"].detach().cpu())
        )
        print(f"  step {step_index} latents captured")
        return callback_kwargs

    return callback_on_step_end


def decode_and_save_step_latents(
    saved_latents: list[tuple[int, torch.Tensor]],
    pipeline,
    output_image: PILImage.Image,
    steps_dir: Path,
) -> None:
    """Decode captured latents through the pipeline VAE and save as PNGs.

    *saved_latents* is a list of ``(step_index, packed_cpu_latent)`` tuples
    produced by :func:`make_step_callback`.  Latents are un-packed,
    de-normalised, batch-decoded through the VAE, and saved to *steps_dir*.
    """
    pipe = pipeline
    device = pipe._execution_device
    dtype = pipe.vae.dtype
    z_dim = pipe.vae.config.get("z_dim", 16)
    _mean = pipe.vae.config["latents_mean"]
    _std = pipe.vae.config["latents_std"]

    latents_mean = torch.tensor(
        _mean, device=device, dtype=dtype
    ).view(1, z_dim, 1, 1, 1)
    latents_std = torch.tensor(
        _std, device=device, dtype=dtype
    ).view(1, z_dim, 1, 1, 1)

    img_h, img_w = output_image.size[1], output_image.size[0]

    unpacked = []
    step_indices = []
    for step_index, lat in saved_latents:
        lat = lat.to(device=device, dtype=dtype)
        lat_5d = pipe._unpack_latents(
            lat, img_h, img_w, pipe.vae_scale_factor
        )
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

    del decoded, images_4d
    torch.cuda.empty_cache()
