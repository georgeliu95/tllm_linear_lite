/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * [Standalone -- minimal rewrite, based on TRT-LLM]
 *
 * Minimal type conversion adapters for CUTLASS FP4 GEMM.
 * Derived from TensorRT-LLM's kernels/cutlass_kernels/cutlass_type_conversion.h (Apache 2.0).
 *
 * Changes from the original:
 *   - Removed all NvInferRuntime.h / nvinfer1::DataType dependencies
 *   - Kept only TllmToCutlassTypeAdapter and CutlassToTllmTypeAdapter
 *     for half, __nv_bfloat16, __nv_fp8_e4m3/e5m2, __nv_fp4_e2m1
 */

#pragma once

#include <cuda_fp16.h>
#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif
#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif

#include <cutlass/half.h>
#include <cutlass/bfloat16.h>
#include <cutlass/float8.h>
#include <cutlass/float_subbyte.h>

#include "trtllm_cutlass_compat.h"

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace cutlass_kernels
{

template <typename T>
struct TllmToCutlassTypeAdapter { using type = T; };

template <> struct TllmToCutlassTypeAdapter<half> { using type = cutlass::half_t; };

#ifdef ENABLE_BF16
template <> struct TllmToCutlassTypeAdapter<__nv_bfloat16> { using type = cutlass::bfloat16_t; };
#endif

#ifdef ENABLE_FP8
template <> struct TllmToCutlassTypeAdapter<__nv_fp8_e4m3> { using type = cutlass::float_e4m3_t; };
template <> struct TllmToCutlassTypeAdapter<__nv_fp8_e5m2> { using type = cutlass::float_e5m2_t; };
#endif

#if defined(ENABLE_FP4)
template <> struct TllmToCutlassTypeAdapter<__nv_fp4_e2m1> { using type = cutlass::float_e2m1_t; };
#endif

template <typename T>
struct CutlassToTllmTypeAdapter { using type = T; };

template <> struct CutlassToTllmTypeAdapter<cutlass::half_t> { using type = half; };

#ifdef ENABLE_BF16
template <> struct CutlassToTllmTypeAdapter<cutlass::bfloat16_t> { using type = __nv_bfloat16; };
#endif

#ifdef ENABLE_FP8
template <> struct CutlassToTllmTypeAdapter<cutlass::float_e4m3_t> { using type = __nv_fp8_e4m3; };
template <> struct CutlassToTllmTypeAdapter<cutlass::float_e5m2_t> { using type = __nv_fp8_e5m2; };
#endif

#if defined(ENABLE_FP4)
template <> struct CutlassToTllmTypeAdapter<cutlass::float_e2m1_t> { using type = __nv_fp4_e2m1; };
#endif

} // namespace cutlass_kernels
} // namespace kernels

TRTLLM_NAMESPACE_END
