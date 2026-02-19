"""Benchmark: FP8+compiled vs BF16 eager on the actual Qwen transformer."""
import os
os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"

import torch
import time

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
print(f"GPU: {torch.cuda.get_device_name()}")

# --- BF16 EAGER BASELINE ---
print("\n=== BF16 EAGER ===")
print("Loading transformer...")
from diffusers import QwenImageTransformer2DModel
transformer_bf16 = QwenImageTransformer2DModel.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    cache_dir=MODELS_DIR,
).to("cuda")

in_channels = transformer_bf16.config.in_channels
joint_attn_dim = transformer_bf16.config.joint_attention_dim
hidden_states = torch.randn(1, 256, in_channels, device="cuda", dtype=torch.bfloat16)
encoder_hidden_states = torch.randn(1, 64, joint_attn_dim, device="cuda", dtype=torch.bfloat16)
timestep = torch.tensor([500.0], device="cuda", dtype=torch.bfloat16)
img_shapes = [[(1, 16, 16)]]

# Warmup
with torch.inference_mode():
    for _ in range(3):
        transformer_bf16(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                         timestep=timestep, img_shapes=img_shapes)
    torch.cuda.synchronize()

# Benchmark
times = []
with torch.inference_mode():
    for _ in range(10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        transformer_bf16(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                         timestep=timestep, img_shapes=img_shapes)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

bf16_avg = sum(times) / len(times)
print(f"BF16 eager: {bf16_avg*1000:.1f} ms/forward")

del transformer_bf16
torch.cuda.empty_cache()

# --- FP8 + COMPILED ---
print("\n=== FP8 + COMPILED ===")
print("Loading transformer...")
transformer_fp8 = QwenImageTransformer2DModel.from_pretrained(
    "Qwen/Qwen-Image-Edit-2511",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    cache_dir=MODELS_DIR,
).to("cuda")

print("FP8 quantizing blocks...")
from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
for block in transformer_fp8.transformer_blocks:
    quantize_(block, Float8DynamicActivationFloat8WeightConfig())

print("Compiling blocks...")
torch._inductor.config.coordinate_descent_tuning = False
torch._inductor.config.max_autotune = False
for i in range(len(transformer_fp8.transformer_blocks)):
    transformer_fp8.transformer_blocks[i] = torch.compile(
        transformer_fp8.transformer_blocks[i], backend="inductor"
    )

# Warmup (triggers compilation)
print("Warmup (compiling Triton kernels)...")
t0 = time.time()
with torch.inference_mode():
    for _ in range(3):
        transformer_fp8(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                        timestep=timestep, img_shapes=img_shapes)
    torch.cuda.synchronize()
print(f"Warmup done in {time.time()-t0:.1f}s")

# Benchmark
times = []
with torch.inference_mode():
    for _ in range(10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        transformer_fp8(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                        timestep=timestep, img_shapes=img_shapes)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

fp8_avg = sum(times) / len(times)
print(f"FP8 compiled: {fp8_avg*1000:.1f} ms/forward")
print(f"\nSpeedup: {bf16_avg/fp8_avg:.2f}x")
