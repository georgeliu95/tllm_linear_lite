"""
nvfp4_gemm.py - Unified NVFP4 GEMM dispatch for tllm_linear_lite.

Dispatches FP4 GEMM to the best available backend based on problem shape:
  - cutlass:   Best performance, CUTLASS 3.x templates. Requires K%32==0, N%32==0, SWIZZLED scales.
  - cuda_core: Small M (M <= 16), optimized for decode phase. Requires LINEAR scale_a.
  - cublaslt:  All shapes fallback, uses cuBLASLt BlockScaleGemm API. Requires SWIZZLED scales.
  - cutedsl:   (TODO) Cute DSL kernels, Blackwell only.

Usage:
    from tllm_linear_lite.nvfp4_gemm import nvfp4_gemm

    output = nvfp4_gemm(
        act_fp4, weight_fp4,
        act_sf, weight_sf,
        alpha,
        output_dtype=torch.bfloat16,
        backend="auto",
    )
"""

import torch
from typing import Optional

import tllm_linear_lite  # noqa: F401 - loads _C.so, registers ops


# Maximum M for cuda_core backend
_CUDA_CORE_MAX_M = 16

# Available backends (order = preference for "auto" mode)
_BACKENDS = ["cutlass", "cublaslt", "cuda_core"]


def nvfp4_gemm(
    act_fp4: torch.Tensor,
    weight_fp4: torch.Tensor,
    act_sf: torch.Tensor,
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: Optional[torch.dtype] = None,
    backend: str = "auto",
) -> torch.Tensor:
    """Unified NVFP4 GEMM dispatch.

    Computes: output[M, N] = alpha * (act_fp4[M, K] @ weight_fp4[N, K]^T)
    with per-16-element FP8 E4M3 block scale factors.

    Args:
        act_fp4:      Activation FP4 packed [M, K/2] (uint8)
        weight_fp4:   Weight FP4 packed [N, K/2] (uint8)
        act_sf:       Activation scale factors (FP8 E4M3 as uint8)
                      - SWIZZLED layout for cublaslt/cutlass
                      - LINEAR layout for cuda_core
        weight_sf:    Weight scale factors (FP8 E4M3 as uint8, SWIZZLED layout)
        alpha:        Global scale [1] float32 tensor (device)
        output_dtype: Output dtype (default: bfloat16). Supports fp16, bf16, fp32.
        backend:      Backend selection:
                      - "auto": auto-select (cuda_core for M<=16, cutlass for K%32==0 & N%32==0, else cublaslt)
                      - "cutlass": force CUTLASS (requires K%32==0, N%32==0)
                      - "cublaslt": force cuBLASLt
                      - "cuda_core": force CUDA core (M must be <= 16)

    Returns:
        Output tensor [M, N] in the specified dtype.

    Raises:
        ValueError: If backend is invalid or constraints not met.
    """
    if output_dtype is None:
        output_dtype = torch.bfloat16

    out_scalar_type = {
        torch.float16: torch.float16,
        torch.bfloat16: torch.bfloat16,
        torch.float32: torch.float32,
    }.get(output_dtype)
    if out_scalar_type is None:
        raise ValueError(f"Unsupported output_dtype: {output_dtype}")

    M = act_fp4.shape[0]
    N = weight_fp4.shape[0]
    K = act_fp4.shape[1] * 2

    # Backend selection
    if backend == "auto":
        if M <= _CUDA_CORE_MAX_M and N % 2 == 0 and K % 16 == 0:
            backend = "cuda_core"
        elif K % 32 == 0 and N % 32 == 0:
            backend = "cutlass"
        else:
            backend = "cublaslt"

    if backend == "cutlass":
        if K % 32 != 0 or N % 32 != 0:
            raise ValueError(
                f"cutlass backend requires K%32==0 and N%32==0, got K={K}, N={N}"
            )
        # CUTLASS expects SWIZZLED layout for both scale factors
        return torch.ops.tllm_linear_lite.cutlass_fp4_gemm(
            act_fp4, weight_fp4, act_sf, weight_sf, alpha,
            out_scalar_type,
        )

    elif backend == "cuda_core":
        if M > _CUDA_CORE_MAX_M:
            raise ValueError(
                f"cuda_core backend requires M <= {_CUDA_CORE_MAX_M}, got M={M}"
            )
        # cuda_core expects LINEAR layout for act_sf
        return torch.ops.tllm_linear_lite.cuda_core_nvfp4_gemm(
            act_fp4, weight_fp4, act_sf, weight_sf, alpha,
            out_scalar_type,
        )

    elif backend == "cublaslt":
        # cuBLASLt expects SWIZZLED layout for both scale factors
        return torch.ops.tllm_linear_lite.cublaslt_fp4_gemm(
            act_fp4, weight_fp4, act_sf, weight_sf, alpha,
            out_scalar_type,
        )

    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Available: {_BACKENDS + ['auto']}"
        )
