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
CuteDSLNVFP4Runner -- Simplified Cute DSL NVFP4 GEMM runner.

Extracted from TRT-LLM's CuteDSLNVFP4BlackwellLinear, with the
TRT-LLM autotuner replaced by a built-in SimpleTuner.

Requirements:
    pip install nvidia-cutlass-dsl>=4.3.4 cuda-core apache-tvm-ffi==0.1.6
"""

import logging
from typing import List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


def _pad_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


def _get_sm_version() -> int:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.major * 10 + props.minor


class CuteDSLNVFP4Runner:
    """Runs NVFP4 GEMM using Cute DSL kernels on Blackwell (SM100/SM103).

    This runner JIT-compiles the Sm100BlockScaledPersistentDenseGemmKernel
    from nvidia-cutlass-dsl with different tiling tactics, profiles them,
    and caches the best one per shape bucket.

    Args:
        output_dtype: Must be torch.bfloat16 (only supported output).
        use_tvm_ffi: Use TVM-FFI for lower host launch overhead (default True).
    """

    # Class-level kernel cache (shared across instances)
    _kernel_cache = {}

    # Default tactics: (mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch)
    DEFAULT_TACTICS = [
        ((128, 128), (2, 1), False, True),
        ((128, 128), (1, 2), False, True),
        ((128, 256), (2, 1), False, True),
        ((256, 128), (2, 1), False, True),
        ((128, 128), (1, 1), False, True),
    ]

    def __init__(
        self,
        output_dtype: torch.dtype = torch.bfloat16,
        use_tvm_ffi: bool = True,
    ):
        if output_dtype != torch.bfloat16:
            raise ValueError("CuteDSL NVFP4 only supports bfloat16 output")

        sm = _get_sm_version()
        if sm not in (100, 103):
            raise ValueError(
                f"CuteDSL NVFP4 requires SM 100 or 103, got SM {sm}"
            )

        self.output_dtype = output_dtype
        self.use_tvm_ffi = use_tvm_ffi
        self.sf_vec_size = 16

        # Lazy imports
        import cutlass
        import cutlass.cute as cute
        self._cutlass = cutlass
        self._cute = cute

        from .kernels.dense_blockscaled_gemm_persistent import (
            Sm100BlockScaledPersistentDenseGemmKernel,
        )
        from .kernels.utils import make_ptr
        self._kernel_class = Sm100BlockScaledPersistentDenseGemmKernel
        self._make_ptr = make_ptr

    def get_valid_tactics(
        self, m: int, n: int, k: int
    ) -> List[Tuple]:
        """Return tactics that are valid for the given shape."""
        valid = []
        for tactic in self.DEFAULT_TACTICS:
            mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch = tactic
            kernel_m = n if swap_ab else m
            kernel_n = m if swap_ab else n
            try:
                if self._kernel_class.can_implement(
                    self._cutlass.Float4E2M1FN,
                    self._cutlass.Float8E4M3FN,
                    self.sf_vec_size,
                    kernel_m, kernel_n, k,
                    mma_tiler_mn,
                    cluster_shape_mn,
                    use_prefetch,
                ):
                    valid.append(tactic)
            except Exception:
                continue
        return valid if valid else [self.DEFAULT_TACTICS[0]]

    def _make_global_pointer(self, tensor: torch.Tensor, dtype, assumed_align: int):
        return self._make_ptr(
            dtype,
            tensor.data_ptr(),
            assumed_align=assumed_align,
        )

    def run(
        self,
        act_fp4: torch.Tensor,
        weight_fp4: torch.Tensor,
        act_sf: torch.Tensor,
        weight_sf: torch.Tensor,
        alpha: torch.Tensor,
        tactic: Optional[Tuple] = None,
    ) -> torch.Tensor:
        """Run NVFP4 GEMM with the given tactic.

        Args:
            act_fp4:    [m, k/2] FP4 packed activation (uint8)
            weight_fp4: [n, k/2] FP4 packed weight (uint8)
            act_sf:     Activation scale factors (FP8 E4M3, swizzled layout)
            weight_sf:  Weight scale factors (FP8 E4M3, swizzled layout)
            alpha:      [1] float32 global scale
            tactic:     (mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch)
                        or None for default.

        Returns:
            [m, n] bfloat16 output tensor.
        """
        cutlass = self._cutlass
        cute = self._cute

        if tactic is None:
            tactic = self.DEFAULT_TACTICS[0]

        mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch = tactic

        m = act_fp4.shape[0]
        k_compressed = act_fp4.shape[1]
        n = weight_fp4.shape[0]
        real_k = k_compressed * 2

        # Allocate output
        c_tensor = torch.empty(m, n, dtype=self.output_dtype, device="cuda")
        if swap_ab:
            c_tensor = c_tensor.permute(1, 0)

        # Scale factor dimensions (padded to 128x4 tile)
        sf_m = _pad_up(m, 128)
        sf_k = _pad_up(real_k // self.sf_vec_size, 4)
        sf_n = _pad_up(n, 128)

        # Reshape scale factors
        a_sf = act_sf.reshape(sf_m * sf_k)
        b_sf = weight_sf.reshape(sf_n * sf_k)

        # Determine kernel A/B (swap_ab swaps the operands)
        if swap_ab:
            kernel_m, kernel_n = n, m
            kernel_sf_m, kernel_sf_n = sf_n, sf_m
            kernel_a, kernel_b = weight_fp4, act_fp4
            kernel_a_sf, kernel_b_sf = b_sf, a_sf
        else:
            kernel_m, kernel_n = m, n
            kernel_sf_m, kernel_sf_n = sf_m, sf_n
            kernel_a, kernel_b = act_fp4, weight_fp4
            kernel_a_sf, kernel_b_sf = a_sf, b_sf

        # Build or retrieve compiled kernel
        cache_key = (self.sf_vec_size, mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch)

        if cache_key not in self._kernel_cache:
            # Create pointers for JIT compilation
            a_ptr = self._make_global_pointer(kernel_a, cutlass.Float4E2M1FN, 32)
            b_ptr = self._make_global_pointer(kernel_b, cutlass.Float4E2M1FN, 32)
            a_sf_ptr = self._make_global_pointer(kernel_a_sf, cutlass.Float8E4M3FN, 16)
            b_sf_ptr = self._make_global_pointer(kernel_b_sf, cutlass.Float8E4M3FN, 16)
            c_ptr = self._make_global_pointer(c_tensor, cutlass.BFloat16, 16)
            alpha_cute = cute.runtime.from_dlpack(alpha)

            if self.use_tvm_ffi:
                stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
            else:
                import cuda.bindings.driver as cuda_drv
                torch_stream = torch.cuda.current_stream()
                stream = cuda_drv.CUstream(torch_stream.cuda_stream)

            gemm = self._kernel_class(
                self.sf_vec_size, mma_tiler_mn, cluster_shape_mn, use_prefetch,
            )

            hardware_info = cutlass.utils.HardwareInfo()
            max_active_clusters = hardware_info.get_max_active_clusters(
                cluster_shape_mn[0] * cluster_shape_mn[1]
            )

            compiled_gemm = cute.compile(
                gemm.wrapper,
                kernel_m, kernel_n, real_k,
                kernel_sf_m // 128, kernel_sf_n // 128, sf_k // 4,
                1,  # batch
                a_ptr, b_ptr, a_sf_ptr, b_sf_ptr, c_ptr,
                alpha_cute, max_active_clusters, stream, swap_ab,
                options="--opt-level 2 --enable-tvm-ffi" if self.use_tvm_ffi else "--opt-level 2",
            )
            self._kernel_cache[cache_key] = compiled_gemm
        else:
            compiled_gemm = self._kernel_cache[cache_key]

        # Launch
        if self.use_tvm_ffi:
            compiled_gemm(
                kernel_m, kernel_n, real_k,
                kernel_sf_m // 128, kernel_sf_n // 128, sf_k // 4,
                kernel_a.data_ptr(), kernel_b.data_ptr(),
                kernel_a_sf.data_ptr(), kernel_b_sf.data_ptr(),
                c_tensor.data_ptr(), alpha,
            )
        else:
            import cuda.bindings.driver as cuda_drv
            a_ptr = self._make_global_pointer(kernel_a, cutlass.Float4E2M1FN, 32)
            b_ptr = self._make_global_pointer(kernel_b, cutlass.Float4E2M1FN, 32)
            a_sf_ptr = self._make_global_pointer(kernel_a_sf, cutlass.Float8E4M3FN, 16)
            b_sf_ptr = self._make_global_pointer(kernel_b_sf, cutlass.Float8E4M3FN, 16)
            c_ptr = self._make_global_pointer(c_tensor, cutlass.BFloat16, 16)
            alpha_cute = cute.runtime.from_dlpack(alpha)
            torch_stream = torch.cuda.current_stream()
            stream = cuda_drv.CUstream(torch_stream.cuda_stream)

            compiled_gemm(
                kernel_m, kernel_n, real_k,
                kernel_sf_m // 128, kernel_sf_n // 128, sf_k // 4,
                a_ptr, b_ptr, a_sf_ptr, b_sf_ptr, c_ptr,
                alpha_cute, stream,
            )

        if swap_ab:
            c_tensor = c_tensor.permute(1, 0)

        return c_tensor
