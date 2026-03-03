# 360° Spin Frame Generation — Design

**Date:** 2026-03-03
**Status:** Approved

## Goal

Replace the placeholder `_run_generate_360()` endpoint with real frame generation. Implement two providers — Zero123++ (local GPU) and Gemini Flash (API) — so both can be tested and compared.

## Architecture

### New Model Base Class

`ImageToMultiViewModel(AIModel)` in `src/casadei/models/image_to_multiview.py`:
- Capability: 1 input image → N output images at different viewing angles
- Abstract method: `_generate_views(image: ImageMedia, num_views: int, **kwargs) -> list[ImageMedia]`
- Public method: `run(inputs: MediaBundle, **kwargs) -> MediaBundle` (validates inputs, calls `_generate_views`, wraps output)

### Provider 1: Zero123++ (Local GPU)

`src/casadei/providers/zero123pp.py`:
- Model ID: `sudo-ai/zero123plus-v1.2` (HuggingFace diffusers)
- Registered as `"zero123pp"` in model registry
- Generates 6 views (0°, 60°, 120°, 180°, 240°, 300°) in a single forward pass
- The model outputs a 3x2 grid image; split into 6 individual frames
- dtype: bfloat16 on CUDA, float32 on CPU
- VRAM: ~4GB
- Load/unload pattern matches existing providers (eager unload, `torch.cuda.empty_cache()`)

### Provider 2: Gemini Flash Multi-View (API)

`src/casadei/providers/gemini_flash_multiview.py`:
- Uses `google-genai` SDK (same as existing `GeminiFlashImageEdit`)
- Registered as `"gemini_multiview"` in model registry
- Sends 6 sequential API calls with angle-specific prompts
- Each call: "Generate this product from a {angle}° viewing angle, maintaining exact appearance"
- No local GPU load

### Updated Endpoint

`POST /api/products/{product_id}/variations/{variation_id}/generate-360`:
- Accepts optional query param `provider` (`"zero123pp"` | `"gemini_multiview"`, default: `"zero123pp"`)
- Background thread `_run_generate_360()`:
  1. Load source image from `variation.results[0]`
  2. Instantiate chosen provider from registry
  3. Load model
  4. Generate 6 views with progress updates
  5. Save frames as `spin_frame_0.png` ... `spin_frame_5.png` in `data/results/{product_id}/{variation_id}/`
  6. Set `variation.spin_frames = [ResultFile(filename=f"spin_frame_{i}.png") for i in range(6)]`
  7. Set `variation.status = JobStatus.completed`
  8. Persist product via `store.save_product()`
  9. Unload model
  10. `job_manager.complete(job_id)`
- On failure: set `variation.status = JobStatus.failed`, `job_manager.fail(job_id, error)`

## File Changes

| File | Action |
|------|--------|
| `src/casadei/models/image_to_multiview.py` | New — base class |
| `src/casadei/providers/zero123pp.py` | New — Zero123++ provider |
| `src/casadei/providers/gemini_flash_multiview.py` | New — Gemini multi-view provider |
| `src/casadei/models/__init__.py` | Export `ImageToMultiViewModel` |
| `src/casadei/providers/__init__.py` | Export + register new providers |
| `src/casadei/api/app.py` | Update `_run_generate_360()` + endpoint signature |

## Not In Scope

- Agent YAML configs (no prompt template system)
- Pipeline integration
- Frame interpolation beyond 6 frames
- 3D mesh generation
- Frontend changes (SpinViewer already handles frame arrays)
