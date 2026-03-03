# Design: GeminiFlashImageEdit Provider

**Date:** 2026-03-03
**Model:** `gemini-3.1-flash-image-preview` (Google "Nano Banana 2")
**Approach:** google-genai SDK, API-based (no local weights)

## Overview

Add an `ImageEditModel` provider wrapping the Gemini 3.1 Flash Image Preview API.
The model accepts up to 10 images and a text prompt, produces 1 edited image.
It is purely API-based — no GPU, no local model weights.

## Files

| File | Change |
|------|--------|
| `src/casadei/providers/gemini_flash_image_edit.py` | New provider class |
| `src/casadei/models/registry.py` | Register as `gemini_flash_image_edit` |
| `src/casadei/providers/__init__.py` | Export `GeminiFlashImageEdit` |
| `pyproject.toml` | Add `google-genai>=1.0` dependency |

## Provider Class

**Class:** `GeminiFlashImageEdit(ImageEditModel)`
**Registry key:** `gemini_flash_image_edit`

### Capability
- Inputs: 1–10 images (required), 1 text prompt (required)
- Outputs: 1 image

### Methods

- `load_model()`: instantiate `genai.Client()` — reads `GEMINI_API_KEY` from env automatically
- `unload_model()`: set `self._client = None` (no GPU/memory to free)
- `_edit(images, prompt, negative_prompt, **kwargs)`: pass `[prompt] + images` as `contents`, call `client.models.generate_content`, extract first image part via `part.as_image()`

### API call pattern (from official docs)
```python
response = client.models.generate_content(
    model="gemini-3.1-flash-image-preview",
    contents=[prompt, *images],  # PIL Images passed directly
)
for part in response.parts:
    if part.inline_data is not None:
        return part.as_image()
```

### DEFAULT_PARAMS
```python
DEFAULT_PARAMS = {}  # No diffusers pipeline, no introspectable params
```

### Error handling
- Raise `RuntimeError` if model not loaded
- Raise `RuntimeError` if API returns no image part in response
- Resize output to match first input image dimensions (same pattern as FireRedImageEdit)

## Dependencies
- `google-genai>=1.0` — Google's current GenAI SDK
- `GEMINI_API_KEY` env var — must be set before calling `load_model()`
