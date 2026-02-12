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
 * Standalone CUDA Core NVFP4 GEMM for tllm_linear_lite.
 *
 * Derived from TensorRT-LLM's cudaCoreGemmNVFP4.cu and cudaNvfp4MM.cpp
 * (Apache 2.0 License). Optimized for small M (decode phase, M <= 16).
 *
 * Provides:
 *   torch.ops.tllm_linear_lite.cuda_core_nvfp4_gemm(a, b, scale_a, scale_b, alpha, out_dtype?) -> Tensor
 *
 * Dependencies:
 *   - CUTLASS (header-only): cutlass::NumericArrayConverter for FP4->float conversion
 *   - CUB: cub::WarpReduce for intra-warp reduction
 *   - CUDA 12.x+ (sm_100a for __nv_fp4_e2m1, cudaGridDependencySynchronize)
 */

#include "tllm_compat.cuh"

#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_size.h>

#include <cub/cub.cuh>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

using SizeType32 = int32_t;

// ============================================================================
// Type adapter: map CUDA __nv_fp4_e2m1 to CUTLASS float_e2m1_t
// ============================================================================

template <typename T>
struct ToCutlassType
{
    using type = T;
};

#if defined(ENABLE_FP8)
template <>
struct ToCutlassType<__nv_fp4_e2m1>
{
    using type = cutlass::float_e2m1_t;
};
#endif

// ============================================================================
// Convert PyTorch dtype to CUDA data type
// ============================================================================

namespace
{

inline cudaDataType_t torchDtypeToCuda(at::ScalarType dtype)
{
    switch (dtype)
    {
    case at::ScalarType::Float: return CUDA_R_32F;
    case at::ScalarType::Half: return CUDA_R_16F;
    case at::ScalarType::BFloat16: return CUDA_R_16BF;
    case at::ScalarType::Byte: return CUDA_R_8U;
    default:
        throw std::runtime_error("Unsupported dtype for cuda_core_nvfp4_gemm");
    }
}

} // anonymous namespace

// ============================================================================
// CUDA Core NVFP4 GEMM kernel
// ============================================================================

namespace cuda_core_gemm_nvfp4
{

template <typename InputType, typename OutputType, typename ScaleType,
    SizeType32 TILE_M, SizeType32 TILE_N, SizeType32 BLOCK_SIZE>
__device__ void cudaCoreGemmImpl(
    InputType const* __restrict__ act,
    InputType const* __restrict__ weight,
    ScaleType const* __restrict__ scale_a,
    ScaleType const* __restrict__ scale_w,
    float const alpha,
    OutputType* __restrict__ output,
    SizeType32 m, SizeType32 n, SizeType32 k)
{
    using VecType = int4;
    using ScaleVecType = __nv_fp8x2_e4m3;
    using CvtInputType = typename ToCutlassType<InputType>::type;
    static constexpr SizeType32 step_k
        = static_cast<SizeType32>(128 / cutlass::sizeof_bits<CvtInputType>::value);
    static constexpr SizeType32 nvfp4_scale_granularity = 16;
    static constexpr SizeType32 step_k_scale = step_k / nvfp4_scale_granularity;
    static constexpr SizeType32 tile_k = step_k * BLOCK_SIZE;

    auto tile_id_m = static_cast<SizeType32>(blockIdx.x * TILE_M);
    auto tile_id_n = static_cast<SizeType32>(blockIdx.y * TILE_N);
    auto tid = static_cast<SizeType32>(threadIdx.x);

    float tile_a[step_k];
    float tile_w[TILE_N * step_k];
    float tile_a_scale[step_k_scale];
    float tile_w_scale[TILE_N * step_k_scale];
    float acc[TILE_M * TILE_N];

    static_assert(step_k % 4 == 0);
    using Converter = cutlass::NumericArrayConverter<float, CvtInputType, 8>;
    using CvtSrcType = typename Converter::source_type;
    using CvtResType = typename Converter::result_type;
    static constexpr SizeType32 k_cvt_count
        = static_cast<SizeType32>(sizeof(VecType) / sizeof(CvtSrcType));

#pragma unroll
    for (SizeType32 i = 0; i < TILE_M * TILE_N; ++i)
    {
        acc[i] = 0;
    }

    act += tile_id_m * k / 2;
    weight += tile_id_n * k / 2;
    output += tile_id_m * n + tile_id_n;
    scale_a += tile_id_m * k / nvfp4_scale_granularity;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
#endif

    int const num_cols_sf = k / nvfp4_scale_granularity;
    int const num_sf_tiles_k = (num_cols_sf + 4 - 1) / 4;

    for (SizeType32 idx_k = tid * step_k; idx_k < k; idx_k += tile_k)
    {
        // Load weight tiles
        for (SizeType32 j = 0; j < TILE_N; ++j)
        {
            auto tile_w_quantized
                = reinterpret_cast<VecType const*>(weight + (j * k + idx_k) / 2)[0];
#pragma unroll
            for (SizeType32 cvt_idx = 0; cvt_idx < k_cvt_count; ++cvt_idx)
            {
                reinterpret_cast<CvtResType*>(tile_w)[j * k_cvt_count + cvt_idx]
                    = Converter::convert(
                        reinterpret_cast<CvtSrcType*>(&tile_w_quantized)[cvt_idx]);
            }
        }

        // Load weight scales (swizzled layout)
        for (SizeType32 j = 0; j < TILE_N; ++j)
        {
            int const row_idx = tile_id_n + j;
            int const col_idx = idx_k / nvfp4_scale_granularity;
            int const tile_offset
                = ((row_idx / 128) * num_sf_tiles_k + col_idx / 4) * 512;
            int const dst_idx
                = tile_offset + (row_idx % 32) * 16
                + ((row_idx % 128) / 32) * 4 + col_idx % 4;
            auto tile_w_scale_fp8x2
                = reinterpret_cast<ScaleVecType const*>(scale_w + dst_idx)[0];
            const char2 tmp = reinterpret_cast<char2 const&>(tile_w_scale_fp8x2);
            tile_w_scale[j * step_k_scale + 0]
                = static_cast<float>(reinterpret_cast<__nv_fp8_e4m3 const&>(tmp.x));
            tile_w_scale[j * step_k_scale + 1]
                = static_cast<float>(reinterpret_cast<__nv_fp8_e4m3 const&>(tmp.y));
        }

        // Load activation tiles and accumulate
#pragma unroll
        for (SizeType32 i = 0; i < TILE_M; ++i)
        {
            auto tile_a_quantized
                = reinterpret_cast<VecType const*>(act + (i * k + idx_k) / 2)[0];
#pragma unroll
            for (SizeType32 cvt_idx = 0; cvt_idx < k_cvt_count; ++cvt_idx)
            {
                reinterpret_cast<CvtResType*>(tile_a)[cvt_idx]
                    = Converter::convert(
                        reinterpret_cast<CvtSrcType*>(&tile_a_quantized)[cvt_idx]);
            }
            auto tile_a_scale_fp8x2
                = reinterpret_cast<ScaleVecType const*>(
                    scale_a + (i * k + idx_k) / nvfp4_scale_granularity)[0];
            const char2 tmp = reinterpret_cast<char2 const&>(tile_a_scale_fp8x2);
            tile_a_scale[0]
                = static_cast<float>(reinterpret_cast<__nv_fp8_e4m3 const&>(tmp.x));
            tile_a_scale[1]
                = static_cast<float>(reinterpret_cast<__nv_fp8_e4m3 const&>(tmp.y));

#pragma unroll
            for (SizeType32 j = 0; j < TILE_N; ++j)
            {
#pragma unroll
                for (SizeType32 l = 0; l < step_k; ++l)
                {
                    acc[i * TILE_N + j] = fma(
                        alpha * tile_a[l]
                            * tile_a_scale[l / nvfp4_scale_granularity],
                        tile_w[j * step_k + l]
                            * tile_w_scale[j * step_k_scale
                                + l / nvfp4_scale_granularity],
                        acc[i * TILE_N + j]);
                }
            }
        }
    }

    // Warp reduction
    typedef cub::WarpReduce<float> WarpReduce;
    static constexpr SizeType32 warp_size = 32;
    static constexpr SizeType32 warp_num = BLOCK_SIZE / warp_size;
    SizeType32 warp_id = tid / warp_size, lane_id = tid % warp_size;

    __shared__ float shmem[TILE_M * TILE_N * warp_num];
    __shared__ typename WarpReduce::TempStorage temp_storage[warp_num];

#pragma unroll
    for (SizeType32 mi = 0; mi < TILE_M; ++mi)
    {
#pragma unroll
        for (SizeType32 ni = 0; ni < TILE_N; ++ni)
        {
            float val = WarpReduce(temp_storage[warp_id]).Sum(acc[mi * TILE_N + ni]);
            if (lane_id == 0)
            {
                shmem[mi * TILE_N + ni + warp_id * TILE_M * TILE_N] = val;
            }
        }
    }
    __syncthreads();

    for (SizeType32 ii = tid; ii < TILE_M * TILE_N; ii += BLOCK_SIZE)
    {
        SizeType32 mid = ii / TILE_N, nid = ii % TILE_N;
        float val = 0;
#pragma unroll
        for (SizeType32 jj = 0; jj < warp_num; ++jj)
        {
            val += shmem[jj * TILE_M * TILE_N + ii];
        }
        output[mid * n + nid] = static_cast<OutputType>(val);
    }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// Kernel entry point
template <typename InputType, typename OutputType, typename ScaleType,
    SizeType32 TILE_M, SizeType32 TILE_N, SizeType32 BLOCK_SIZE>
__global__ void cudaCoreGemmFp4(
    InputType const* __restrict__ act,
    InputType const* __restrict__ weight,
    ScaleType const* __restrict__ scale_a,
    ScaleType const* __restrict__ scale_w,
    float const* alpha_ptr,
    OutputType* __restrict__ output,
    SizeType32 m, SizeType32 n, SizeType32 k)
{
    float alpha = alpha_ptr[0];
    cudaCoreGemmImpl<InputType, OutputType, ScaleType, TILE_M, TILE_N, BLOCK_SIZE>(
        act, weight, scale_a, scale_w, alpha, output, m, n, k);
}

// Launch helper with PDL support
template <typename InputType, typename OutputType, typename ScaleType,
    SizeType32 TILE_M, SizeType32 TILE_N, SizeType32 BLOCK_SIZE>
void cudaCoreGemmKernel(
    void const* act, void const* weight, void* output,
    SizeType32 m, SizeType32 n, SizeType32 k,
    __nv_fp8_e4m3 const* scale_a, __nv_fp8_e4m3 const* scale_b,
    float const* alpha_ptr, cudaStream_t stream)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid(m / TILE_M, n / TILE_N);

    if (tensorrt_llm::common::getEnvEnablePDL())
    {
        cudaLaunchConfig_t config = {0};
        config.gridDim = grid;
        config.blockDim = block;
        config.dynamicSmemBytes = 0;
        config.stream = stream;

        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute[0].val.programmaticStreamSerializationAllowed = 1;
        config.attrs = attribute;
        config.numAttrs = 1;

        cudaLaunchKernelEx(&config,
            cudaCoreGemmFp4<InputType, OutputType, ScaleType, TILE_M, TILE_N, BLOCK_SIZE>,
            reinterpret_cast<InputType const*>(act),
            reinterpret_cast<InputType const*>(weight),
            reinterpret_cast<ScaleType const*>(scale_a),
            reinterpret_cast<ScaleType const*>(scale_b),
            alpha_ptr,
            reinterpret_cast<OutputType*>(output),
            m, n, k);
    }
    else
    {
        cudaCoreGemmFp4<InputType, OutputType, ScaleType, TILE_M, TILE_N, BLOCK_SIZE>
            <<<grid, block, 0, stream>>>(
                reinterpret_cast<InputType const*>(act),
                reinterpret_cast<InputType const*>(weight),
                reinterpret_cast<ScaleType const*>(scale_a),
                reinterpret_cast<ScaleType const*>(scale_b),
                alpha_ptr,
                reinterpret_cast<OutputType*>(output),
                m, n, k);
    }
}

// Recursive template to instantiate TILE_M from 1 to 16
template <typename InputType, typename OutputType, typename ScaleType,
    int TILE_M, int TILE_N, int BLOCK_SIZE>
bool cudaCoreGemmTemplateCaller(
    void const* act, void const* weight, void* output,
    SizeType32 m, SizeType32 n, SizeType32 k,
    __nv_fp8_e4m3 const* scale_a, __nv_fp8_e4m3 const* scale_b,
    float const* alpha_ptr, cudaStream_t stream)
{
    constexpr int maxM = 16;
    if (m == TILE_M)
    {
        cudaCoreGemmKernel<InputType, OutputType, ScaleType, TILE_M, TILE_N, BLOCK_SIZE>(
            act, weight, output, m, n, k, scale_a, scale_b, alpha_ptr, stream);
        return true;
    }
    if constexpr (TILE_M < maxM)
    {
        return cudaCoreGemmTemplateCaller<InputType, OutputType, ScaleType,
            TILE_M + 1, TILE_N, BLOCK_SIZE>(
            act, weight, output, m, n, k, scale_a, scale_b, alpha_ptr, stream);
    }
    return false;
}

template <typename InputType, typename OutputType, typename ScaleType = __nv_fp8_e4m3>
bool cudaCoreGemmLauncher(
    void const* act, void const* weight, void* output,
    SizeType32 m, SizeType32 n, SizeType32 k,
    __nv_fp8_e4m3 const* scale_a, __nv_fp8_e4m3 const* scale_b,
    float const* alpha_ptr, cudaStream_t stream)
{
    return cudaCoreGemmTemplateCaller<InputType, OutputType, ScaleType, 1, 2, 128>(
        act, weight, output, m, n, k, scale_a, scale_b, alpha_ptr, stream);
}

bool dispatch(
    void const* act, void const* weight, void* output,
    SizeType32 m, SizeType32 n, SizeType32 k,
    cudaDataType_t inputType, cudaDataType_t outputType,
    __nv_fp8_e4m3 const* scale_a, __nv_fp8_e4m3 const* scale_b,
    float const* alpha_ptr, cudaStream_t stream)
{
    if (n % 2 != 0 || k % 16 != 0)
        return false;

    if (inputType != CUDA_R_8U)
        return false;

    if (outputType == CUDA_R_16F)
        return cudaCoreGemmLauncher<__nv_fp4_e2m1, half>(
            act, weight, output, m, n, k, scale_a, scale_b, alpha_ptr, stream);
    else if (outputType == CUDA_R_16BF)
        return cudaCoreGemmLauncher<__nv_fp4_e2m1, __nv_bfloat16>(
            act, weight, output, m, n, k, scale_a, scale_b, alpha_ptr, stream);
    else if (outputType == CUDA_R_32F)
        return cudaCoreGemmLauncher<__nv_fp4_e2m1, float>(
            act, weight, output, m, n, k, scale_a, scale_b, alpha_ptr, stream);

    return false;
}

} // namespace cuda_core_gemm_nvfp4

// ============================================================================
// PyTorch Op
// ============================================================================

/// CUDA Core NVFP4 GEMM for small M (decode phase, M <= 16).
///
/// NOTE: scale_a must be in LINEAR layout (not swizzled).
///       scale_b must be in SWIZZLED layout.
///       If your activation scales are swizzled, call
///       block_scale_interleave_reverse first.
///
/// @param a          Activation FP4 packed [m, k/2] (uint8)
/// @param b          Weight FP4 packed [n, k/2] (uint8)
/// @param scale_a    Activation scale factors (FP8 E4M3, LINEAR layout)
/// @param scale_b    Weight scale factors (FP8 E4M3, SWIZZLED layout)
/// @param alpha      Global alpha [1] float32 tensor (device)
/// @param out_dtype  Optional output dtype (default: bfloat16)
/// @return           Output tensor [m, n]
at::Tensor cuda_core_nvfp4_gemm(
    at::Tensor const& a,
    at::Tensor const& b,
    at::Tensor const& scale_a,
    at::Tensor const& scale_b,
    at::Tensor const& alpha,
    std::optional<c10::ScalarType> out_dtype)
{
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "Inputs must be contiguous");
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "Inputs must be 2D");
    TORCH_CHECK(a.sizes()[1] == b.sizes()[1], "K dimension mismatch");
    TORCH_CHECK(alpha.numel() > 0 && alpha.dtype() == torch::kFloat32, "alpha must be float32");

    int32_t m = a.sizes()[0];
    int32_t n = b.sizes()[0];
    int32_t k = a.sizes()[1] * 2;

    TORCH_CHECK(m <= 16, "cuda_core_nvfp4_gemm only supports M <= 16 (decode phase)");

    at::ScalarType dtype = out_dtype.value_or(at::ScalarType::BFloat16);
    at::Tensor out = at::empty({m, n}, a.options().dtype(dtype));

    cudaDataType_t inputCudaType = torchDtypeToCuda(a.scalar_type());
    cudaDataType_t outputCudaType = torchDtypeToCuda(dtype);

    auto stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();

    bool ok = cuda_core_gemm_nvfp4::dispatch(
        a.data_ptr(), b.data_ptr(), out.data_ptr(),
        m, n, k,
        inputCudaType, outputCudaType,
        reinterpret_cast<__nv_fp8_e4m3 const*>(scale_a.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3 const*>(scale_b.data_ptr()),
        alpha.data_ptr<float>(),
        stream);

    TORCH_CHECK(ok, "cuda_core_nvfp4_gemm dispatch failed (check M<=16, N%2==0, K%16==0)");

    return out;
}

// ============================================================================
// Op registration
// ============================================================================

TORCH_LIBRARY_FRAGMENT(tllm_linear_lite, m)
{
    m.def(
        "cuda_core_nvfp4_gemm(Tensor a, Tensor b, Tensor scale_a, Tensor scale_b, "
        "Tensor alpha, ScalarType? out_dtype=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(tllm_linear_lite, CUDA, m)
{
    m.impl("cuda_core_nvfp4_gemm", &cuda_core_nvfp4_gemm);
}
