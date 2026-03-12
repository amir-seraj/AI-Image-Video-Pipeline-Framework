"""Blackwell (SM 110) cuBLAS workaround: replace torch.nn.functional.linear with Triton.

cuBLAS is broken on Jetson Thor (SM 11.0, CUDA 13.0, Driver 580) — all matmul
operations (gemm, addmm, F.linear) crash with CUBLAS_STATUS_INVALID_VALUE.
Convolutions work fine (they use cuDNN, not cuBLAS).

This module provides a Triton-based F.linear replacement that bypasses cuBLAS
entirely. Call `patch_linear()` once at startup before loading any model.

Also patches torch.mm, torch.matmul, torch.bmm, torch.addmm for full coverage.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

_patched = False


@triton.jit
def _linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_n[:, None] < N) & (offs_k[None, :] + k < K), other=0.0)
        acc += tl.dot(x, w.trans())
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk
    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc += bias[None, :]
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _triton_linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Drop-in replacement for torch.nn.functional.linear using Triton."""
    if not input.is_cuda:
        # Fall back to CPU implementation
        return torch._C._nn.linear(input, weight, bias)

    orig_shape = input.shape
    orig_dtype = input.dtype
    x_2d = input.reshape(-1, input.shape[-1]).contiguous().float()
    M, K = x_2d.shape
    N = weight.shape[0]
    w_f32 = weight.float().contiguous()
    out = torch.empty(M, N, device=input.device, dtype=torch.float32)

    BLOCK_M = min(128, triton.next_power_of_2(M))
    BLOCK_N = min(128, triton.next_power_of_2(N))
    BLOCK_K = min(64, triton.next_power_of_2(K))
    # Ensure block sizes are at least 16 for tl.dot
    BLOCK_M = max(16, BLOCK_M)
    BLOCK_N = max(16, BLOCK_N)
    BLOCK_K = max(16, BLOCK_K)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    _linear_kernel[grid](
        x_2d, w_f32,
        bias.float().contiguous() if bias is not None else x_2d,
        out, M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        w_f32.stride(0), w_f32.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return out.to(orig_dtype).reshape(*orig_shape[:-1], N)


def _triton_mm(input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for torch.mm using Triton."""
    if not input.is_cuda:
        return torch._orig_mm(input, mat2)
    # mm is just linear without bias, with weight transposed
    # input: (M, K), mat2: (K, N) -> (M, N)
    M, K = input.shape
    N = mat2.shape[1]
    return _triton_linear(input, mat2.T, None)


def _triton_matmul(input: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for torch.matmul using Triton."""
    if not input.is_cuda:
        return torch._orig_matmul(input, other)
    if input.dim() == 2 and other.dim() == 2:
        return _triton_mm(input, other)
    if input.dim() >= 2 and other.dim() == 2:
        # Batch matmul: reshape to 2D, compute, reshape back
        orig_shape = input.shape
        x_2d = input.reshape(-1, input.shape[-1])
        result = _triton_mm(x_2d, other)
        return result.reshape(*orig_shape[:-1], other.shape[1])
    if input.dim() >= 2 and other.dim() >= 2:
        # General batched matmul — reshape both
        *batch_a, M, K = input.shape
        *batch_b, K2, N = other.shape
        x_2d = input.reshape(-1, K)
        # For batched case, iterate (slower but correct)
        o_2d = other.reshape(-1, K2, N)
        if len(batch_a) == len(batch_b):
            results = []
            batch_size = x_2d.shape[0] // M
            for i in range(batch_size):
                chunk_x = x_2d[i * M:(i + 1) * M]
                chunk_o = o_2d[i]
                results.append(_triton_mm(chunk_x, chunk_o))
            return torch.stack(results).reshape(*batch_a, M, N)
    # Fallback for edge cases: compute on CPU
    return (input.cpu() @ other.cpu()).to(input.device)


def _triton_addmm(bias: torch.Tensor, input: torch.Tensor, mat2: torch.Tensor, *, beta=1, alpha=1) -> torch.Tensor:
    """Drop-in replacement for torch.addmm."""
    if not input.is_cuda:
        return torch._orig_addmm(bias, input, mat2, beta=beta, alpha=alpha)
    result = _triton_mm(input, mat2)
    if alpha != 1:
        result = result * alpha
    if beta != 0:
        result = result + bias * beta
    return result


def _triton_bmm(input: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for torch.bmm."""
    if not input.is_cuda:
        return torch._orig_bmm(input, mat2)
    B, M, K = input.shape
    _, _, N = mat2.shape
    results = []
    for i in range(B):
        results.append(_triton_mm(input[i], mat2[i]))
    return torch.stack(results)


def patch_linear():
    """Monkey-patch torch to use Triton matmul instead of cuBLAS.

    Call this once at the start of your program, before loading models.
    Safe to call multiple times (idempotent).
    """
    global _patched
    if _patched:
        return

    import torch.nn.functional as F

    # Save originals
    torch._orig_mm = torch.mm
    torch._orig_matmul = torch.matmul
    torch._orig_addmm = torch.addmm
    torch._orig_bmm = torch.bmm

    # Patch F.linear
    F.linear = _triton_linear

    # Patch torch ops
    torch.mm = _triton_mm
    torch.matmul = _triton_matmul
    torch.addmm = _triton_addmm
    torch.bmm = _triton_bmm

    # Also patch the Tensor methods
    torch.Tensor.matmul = lambda self, other: _triton_matmul(self, other)
    torch.Tensor.mm = lambda self, other: _triton_mm(self, other)
    torch.Tensor.bmm = lambda self, other: _triton_bmm(self, other)

    # Patch the @ operator (Tensor.__matmul__)
    torch.Tensor.__matmul__ = lambda self, other: _triton_matmul(self, other)

    _patched = True
    logger.info("Triton linear patch applied — all matmul ops bypass cuBLAS")
    print("[triton_linear_patch] All CUDA matmul ops now use Triton (Blackwell cuBLAS workaround)")
