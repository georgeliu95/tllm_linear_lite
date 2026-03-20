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
CUDA-based global amax + global_scale kernel (TRUE single kernel).

Last-block reduction pattern: each CTA writes per-block max to a temp
buffer, the last CTA reduces all partials and writes both amax and
global_scale.  No zero-init, no extra kernels, no host-side allocation
on the hot path.

Usage:
    from tllm_linear_lite.amax.cuda_amax import cuda_amax, cuda_prologue

    x = torch.randn(128, 6144, device="cuda", dtype=torch.bfloat16)
    amax = cuda_amax(x)                         # [2] tensor, amax only
    amax, gs = cuda_prologue(x, quant_range=2688.0)  # fused amax + scale
"""

from __future__ import annotations

import torch


def cuda_amax(x: torch.Tensor) -> torch.Tensor:
    """Compute global absolute maximum using a single CUDA kernel.

    Args:
        x: Input tensor (bf16/fp16, >= 2D, CUDA, contiguous).

    Returns:
        [2] float32 tensor: [amax, amax] (quant_range=0 → both are amax).
    """
    if not x.is_contiguous():
        x = x.contiguous()
    return torch.ops.tllm_linear_lite.calculate_global_amax(x, 0.0, 1e-12)


def cuda_prologue(
    x: torch.Tensor,
    quant_range: float = 2688.0,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute amax and global_scale in a single CUDA kernel.

    Equivalent to::

        amax = x.abs().max().float().clamp_min(eps)
        global_scale = quant_range / amax

    but fused into one kernel launch with zero extra overhead.

    Args:
        x: Input tensor (bf16/fp16, >= 2D, CUDA, contiguous).
        quant_range: Numerator for scale computation (e.g. 2688 for NVFP4).
        eps: Floor for amax to prevent division by zero.

    Returns:
        (amax, global_scale) — both are scalar float32 tensors (views into
        the same [2]-element buffer, no allocation).
    """
    if not x.is_contiguous():
        x = x.contiguous()
    buf = torch.ops.tllm_linear_lite.calculate_global_amax(x, quant_range, eps)
    return buf[0], buf[1]
