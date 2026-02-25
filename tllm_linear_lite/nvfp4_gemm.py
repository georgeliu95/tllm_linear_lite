# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
nvfp4_gemm.py - Unified NVFP4 GEMM dispatch for tllm_linear_lite.

Dispatches FP4 GEMM to the best available backend based on problem shape:
  - cutlass:   Best performance, CUTLASS 3.x templates. Requires K%32==0, N%32==0, SWIZZLED scales.
  - cuda_core: Small M (M <= 16), optimized for decode phase. Requires LINEAR scale_a.
  - cublaslt:  All shapes fallback, uses cuBLASLt BlockScaleGemm API. Requires SWIZZLED scales.
  - cutedsl:   Cute DSL JIT kernels, Blackwell SM100/SM103 only, bfloat16 output.

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
from tllm_linear_lite.cutedsl import IS_CUTLASS_DSL_AVAILABLE


# Maximum M for cuda_core backend
_CUDA_CORE_MAX_M = 16

# Available backends (order = preference for "auto" mode)
_BACKENDS = ["cutedsl", "cutlass", "cublaslt", "cuda_core"]

# Lazy-initialized cutedsl runner + tuner (created on first use)
_cutedsl_runner = None
_cutedsl_tuner = None


def nvfp4_gemm(
    act_fp4: torch.Tensor,
    weight_fp4: torch.Tensor,
    act_sf: torch.Tensor,
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: Optional[torch.dtype] = None,
    bias: Optional[torch.Tensor] = None,
    backend: str = "auto",
) -> torch.Tensor:
    """Unified NVFP4 GEMM dispatch.

    Computes: output[M, N] = alpha * (act_fp4[M, K] @ weight_fp4[N, K]^T) + bias
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
        bias:         Optional bias [N] tensor. Fused in cuBLASLt epilogue; for other
                      backends, added post-GEMM (output += bias).
        backend:      Backend selection:
                      - "auto": auto-select (cutedsl if available on Blackwell, cuda_core for M<=16,
                        cutlass for K%32==0 & N%32==0, else cublaslt)
                      - "cutedsl": force Cute DSL (SM100/103, bf16 only, K%32==0)
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

    # Check cutedsl eligibility
    _cutedsl_eligible = (
        IS_CUTLASS_DSL_AVAILABLE
        and K % 32 == 0
        and out_scalar_type == torch.bfloat16
        and _is_blackwell_sm()
    )

    # Backend selection
    if backend == "auto":
        if M <= _CUDA_CORE_MAX_M and N % 2 == 0 and K % 16 == 0:
            backend = "cuda_core"
        elif _cutedsl_eligible:
            backend = "cutedsl"
        elif K % 32 == 0 and N % 32 == 0:
            backend = "cutlass"
        else:
            backend = "cublaslt"

    if backend == "cutedsl":
        if not IS_CUTLASS_DSL_AVAILABLE:
            raise ValueError(
                "cutedsl backend requires nvidia-cutlass-dsl. "
                "Install with: pip install nvidia-cutlass-dsl>=4.3.4 cuda-core apache-tvm-ffi==0.1.6"
            )
        out = _run_cutedsl(act_fp4, weight_fp4, act_sf, weight_sf, alpha, M, N, K)
        if bias is not None:
            out = out + bias  # post-GEMM bias (no epilogue fusion for cutedsl yet)
        return out

    elif backend == "cutlass":
        if K % 32 != 0 or N % 32 != 0:
            raise ValueError(
                f"cutlass backend requires K%32==0 and N%32==0, got K={K}, N={N}"
            )
        # CUTLASS expects SWIZZLED layout for both scale factors
        out = torch.ops.tllm_linear_lite.cutlass_fp4_gemm(
            act_fp4, weight_fp4, act_sf, weight_sf, alpha,
            out_scalar_type,
        )
        if bias is not None:
            out = out + bias  # post-GEMM bias (no epilogue fusion for cutlass yet)
        return out

    elif backend == "cuda_core":
        if M > _CUDA_CORE_MAX_M:
            raise ValueError(
                f"cuda_core backend requires M <= {_CUDA_CORE_MAX_M}, got M={M}"
            )
        # cuda_core expects LINEAR layout for act_sf
        out = torch.ops.tllm_linear_lite.cuda_core_nvfp4_gemm(
            act_fp4, weight_fp4, act_sf, weight_sf, alpha,
            out_scalar_type,
        )
        if bias is not None:
            out = out + bias  # post-GEMM bias (no epilogue fusion for cuda_core)
        return out

    elif backend == "cublaslt":
        # cuBLASLt expects SWIZZLED layout for both scale factors
        # Bias is fused in cuBLASLt epilogue when provided
        return torch.ops.tllm_linear_lite.cublaslt_fp4_gemm(
            act_fp4, weight_fp4, act_sf, weight_sf, alpha,
            bias,  # fused in epilogue (None if not provided)
            out_scalar_type,
        )

    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Available: {_BACKENDS + ['auto']}"
        )


# ============================================================================
# Cute DSL helpers
# ============================================================================

def _is_blackwell_sm() -> bool:
    """Check if current GPU is Blackwell (SM 100 or 103)."""
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    sm = props.major * 10 + props.minor
    return sm in (100, 103)


def _run_cutedsl(
    act_fp4: torch.Tensor,
    weight_fp4: torch.Tensor,
    act_sf: torch.Tensor,
    weight_sf: torch.Tensor,
    alpha: torch.Tensor,
    M: int, N: int, K: int,
) -> torch.Tensor:
    """Run NVFP4 GEMM via Cute DSL with simple auto-tuning."""
    global _cutedsl_runner, _cutedsl_tuner

    if _cutedsl_runner is None:
        from tllm_linear_lite.cutedsl.runner import CuteDSLNVFP4Runner
        from tllm_linear_lite.cutedsl.tuner import SimpleTuner
        _cutedsl_runner = CuteDSLNVFP4Runner(output_dtype=torch.bfloat16)
        _cutedsl_tuner = SimpleTuner(warmup=3, repeat=5)

    # Get valid tactics for this shape
    tactics = _cutedsl_runner.get_valid_tactics(M, N, K)

    # Auto-tune: pick the best tactic for this shape
    def run_fn(tactic):
        return _cutedsl_runner.run(act_fp4, weight_fp4, act_sf, weight_sf, alpha, tactic)

    best_tactic = _cutedsl_tuner.choose_best((M, N, K), tactics, run_fn)

    return _cutedsl_runner.run(act_fp4, weight_fp4, act_sf, weight_sf, alpha, best_tactic)
