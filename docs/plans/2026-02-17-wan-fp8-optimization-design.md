# Wan 2.1 FP8 + torch.compile Optimization

## Problem
Wan 2.1 I2V-14B in BF16 takes ~1 hour for 50 inference steps / 81 frames on Jetson Thor. Too slow for production use.

## Approach
torchao FP8 weight-only quantization + torch.compile kernel fusion. Chosen over TensorRT due to a known SM 110 FP8 fallback bug (github.com/NVIDIA/TensorRT/issues/4590).

## Architecture

New FP8 provider variants alongside existing BF16 baselines:

```
providers/wan_i2v.py              # BF16 baseline (unchanged)
providers/wan_i2v_fp8.py          # FP8 + torch.compile
providers/wan_video_edit.py       # BF16 baseline (unchanged)
providers/wan_video_edit_fp8.py   # FP8 + torch.compile
```

Registered as `wan_i2v_fp8` and `wan_video_edit_fp8` in the model registry. Agent YAML selects which variant to use.

## Implementation

1. Load transformer with `TorchAoConfig("float8wo")` FP8 weight-only quantization
2. VAE and CLIP stay float32 (small, quantization-sensitive)
3. Apply `torch.compile(mode="max-autotune", fullgraph=True)` to transformer
4. Optional: save/load pre-quantized weights to skip re-quantization

## Expected Results

| Metric             | BF16 (current) | FP8 + compile  |
|--------------------|----------------|----------------|
| GPU memory         | ~65 GB         | ~35-40 GB      |
| 50 steps / 81 frames | ~1 hour     | ~25-30 min     |
| First-run overhead | none           | ~3 min (cached)|

## Dependencies
- `torchao` (pip)

## Testing
- New unit tests: test_wan_i2v_fp8.py, test_wan_video_edit_fp8.py
- E2e validation: same shoe image through BF16 and FP8 for visual comparison
- Performance logging: time per step, GPU memory usage

## Files to Create/Modify
- CREATE: src/casadei/providers/wan_i2v_fp8.py
- CREATE: src/casadei/providers/wan_video_edit_fp8.py
- CREATE: agents/wan_image_to_video_fp8.yaml
- CREATE: agents/wan_video_editor_fp8.yaml
- CREATE: tests/test_wan_i2v_fp8.py
- CREATE: tests/test_wan_video_edit_fp8.py
- MODIFY: src/casadei/models/registry.py (register fp8 variants)
- MODIFY: src/casadei/providers/__init__.py (exports)
- MODIFY: src/casadei/__init__.py (exports)
- MODIFY: pyproject.toml (add torchao dependency)
