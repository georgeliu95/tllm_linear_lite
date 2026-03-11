/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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
 * [Standalone FP4-only modification]
 *
 * This file is derived from TensorRT-LLM's tensorrt_llm/kernels/quantization.cu.
 *
 * Changes from the original:
 *   - Replaced includes: tensorrt_llm/common/{assert,config,cudaTypeUtils,cudaUtils,
 *     envUtils,quantTypeUtils,reduceKernelUtils}.cuh/h,
 *     tensorrt_llm/kernels/quantization.{cuh,h}
 *     -> tllm_compat.cuh + quantization.cuh + quantization.h (local headers)
 *   - Removed: invokeQuantization<T>() impl + instantiations (float/half/bf16 -> INT8)
 *   - Removed: invokePerTokenQuantization<T>() impl + INSTANTIATE_INVOKE_PER_TOKEN_QUANTIZATION
 *              macro + all instantiations (float/half/bf16 -> int8/fp8_e4m3)
 *   - Removed: invokeMxFP8Quantization<T>() impl (FP16/BF16 -> MXFP8)
 *   - Removed: invokeMxFP8Quantization<half> and <__nv_bfloat16> template instantiations
 *   - Kept:    invokeFP4Quantization<T>() impl + instantiations (half/bf16/fp8 x SF_VEC_SIZE 16/32)
 *   - Kept:    block_scale_interleave_kernel, block_scale_interleave_reverse_kernel,
 *              invokeBlockScaleInterleave, invokeBlockScaleInterleaveReverse
 *   - Kept:    VecTypeImpl, VecType, getMaxAbs,
 *              computePerTokenGlobalScaleForFP4QuantizationKernel,
 *              computePerTokenGlobalScaleForFP4Quantization + instantiations (half/bf16)
 */

#include "tllm_compat.cuh"
#include "quantization.cuh"
#include "quantization.h"
#include <float.h>
 
 using namespace tensorrt_llm::common;
 
 TRTLLM_NAMESPACE_BEGIN
 
namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////
// FP4 Quantization
 
 template <typename T, int SF_VEC_SIZE>
 void invokeFP4Quantization(int b, int m, int n, T const* input, float const* SFScale, int64_t* output, int32_t* SFOuput,
     bool useUE8M0, QuantizationSFLayout layout, int multiProcessorCount, cudaStream_t stream, int kernelVersion)
 {
 #ifdef ENABLE_FP8
     if constexpr (std::is_same_v<T, __nv_fp8_e4m3>)
     {
         // FP8→FP4 only supports v0; v1 is FP16/BF16 only.
         // Grid, Block size. Each thread converts 16 values.
         dim3 block(std::min(int(n / CVT_FP8_TO_FP4_ELTS_PER_THREAD), 512));
         int const numBlocksPerSM = std::max(1u, 2048u / block.x);
         int numBlocksForM = layout == QuantizationSFLayout::SWIZZLED ? PadUpFn(m, 128) : m;
         dim3 grid(std::min(numBlocksForM, multiProcessorCount * numBlocksPerSM));

         auto* kernel_instance = useUE8M0
             ? &quantize_with_block_size<BlockScaleQuantizationType::FP8_TO_FP4, T, SF_VEC_SIZE, true>
             : &quantize_with_block_size<BlockScaleQuantizationType::FP8_TO_FP4, T, SF_VEC_SIZE, false>;
         kernel_instance<<<grid, block, 0, stream>>>(b, m, n, n, input, SFScale, reinterpret_cast<uint32_t*>(output),
             reinterpret_cast<uint32_t*>(SFOuput), layout);
     }
     else
 #endif
     {
         // FP16/BF16→FP4: dispatch to v0 (8 elems/thread) or v1 (16 elems/thread).
         int numBlocksForM = layout == QuantizationSFLayout::SWIZZLED ? PadUpFn(m, 128) : m;

         // PDL launch config shared by both versions.
         cudaLaunchConfig_t config;
         config.dynamicSmemBytes = 0;
         config.stream = stream;
         cudaLaunchAttribute attrs[1];
         attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
         attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
         config.numAttrs = 1;
         config.attrs = attrs;

         if (kernelVersion == 0)
         {
             // v0: 8 elements per thread, 16-byte vectorized load (LDG.E.128×1).
             dim3 block(std::min(int(n / CVT_ELTS_PER_THREAD), 512));
             int const numBlocksPerSM = std::max(1u, 2048u / block.x);
             dim3 grid(std::min(numBlocksForM, multiProcessorCount * numBlocksPerSM));
             config.gridDim = grid;
             config.blockDim = block;

             auto* kernel_instance = useUE8M0
                 ? &quantize_with_block_size<BlockScaleQuantizationType::FP16_TO_FP4, T, SF_VEC_SIZE, true>
                 : &quantize_with_block_size<BlockScaleQuantizationType::FP16_TO_FP4, T, SF_VEC_SIZE, false>;
             cudaLaunchKernelEx(&config, kernel_instance, b, m, n, n, input, SFScale,
                 reinterpret_cast<uint32_t*>(output), reinterpret_cast<uint32_t*>(SFOuput), layout);
         }
         else
         {
             // v1: 16 elements per thread, 256-bit vectorized load (LDG.E.128×2), better ILP.
             dim3 block(std::min(int(n / CVT_OPT_ELTS_PER_THREAD), 512));
             int const numBlocksPerSM = std::max(1u, 2048u / block.x);
             dim3 grid(std::min(numBlocksForM, multiProcessorCount * numBlocksPerSM));
             config.gridDim = grid;
             config.blockDim = block;

             auto* kernel_instance = useUE8M0
                 ? &opt_quantize_with_block_size_v1<BlockScaleQuantizationType::FP16_TO_FP4, T, SF_VEC_SIZE, true>
                 : &opt_quantize_with_block_size_v1<BlockScaleQuantizationType::FP16_TO_FP4, T, SF_VEC_SIZE, false>;
             cudaLaunchKernelEx(&config, kernel_instance, b, m, n, n, input, SFScale,
                 reinterpret_cast<uint32_t*>(output), reinterpret_cast<uint32_t*>(SFOuput), layout);
         }
     }
 }
 
////////////////////////////////////////////////////////////////////////////////////////////////////
 
 __global__ void block_scale_interleave_kernel(int numBatches, int numRows, int numRowsPadded, int numCols,
     int numColsPadded, uint8_t const* SFIn, uint8_t* SFOutput)
 {
     for (int rowIdx = blockIdx.x; rowIdx < numRowsPadded; rowIdx += gridDim.x)
     {
         for (int batchIdx = 0; batchIdx < numBatches; batchIdx++)
         {
             for (int colIdx = threadIdx.x; colIdx < numColsPadded; colIdx += blockDim.x)
             {
                 uint8_t sf = 0;
                 if (rowIdx < numRows && colIdx < numCols)
                 {
                     int64_t inOffset = batchIdx * numRows * numCols + rowIdx * numCols + colIdx;
                     sf = SFIn[inOffset];
                 }
 
                 std::optional<int> batchIdxOpt = batchIdx;
                 std::optional<int> numRowsOpt = numRows;
 
                 // Without batching, the math in get_sf_out_offset is the same as
                 // int const numSfTilesK = (numCols + 4 - 1) / 4;
                 // int const tileOffset = ((mi / 128) * numSfTilesK + ki / 4) * 512;
                 // int const dstIdx = tileOffset + (mi % 32) * 16 + ((mi % 128) / 32) * 4 + ki % 4;
                 auto dstIdx = get_sf_out_offset_128x4(batchIdxOpt, rowIdx, colIdx, numRowsOpt, numCols);
                 SFOutput[dstIdx] = sf;
             }
         }
     }
 }
 
 __global__ void block_scale_interleave_reverse_kernel(
     int numBatches, int numRows, int numCols, uint8_t const* SFIn, uint8_t* SFOutput)
 {
     for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x)
     {
         for (int batchIdx = 0; batchIdx < numBatches; batchIdx++)
         {
             for (int colIdx = threadIdx.x; colIdx < numCols; colIdx += blockDim.x)
             {
                 std::optional<int> batchIdxOpt = batchIdx;
                 std::optional<int> numRowsOpt = numRows;
 
                 // Get the swizzled input index using the same swizzling pattern
                 auto srcIdx = get_sf_out_offset_128x4(batchIdxOpt, rowIdx, colIdx, numRowsOpt, numCols);
                 auto sf = SFIn[srcIdx];
 
                 // Output goes to linear layout
                 int64_t outOffset = batchIdx * numRows * numCols + rowIdx * numCols + colIdx;
                 SFOutput[outOffset] = sf;
             }
         }
     }
 }
 
 // This is intended for weight loading, so m and n are large, b <= 256
 void invokeBlockScaleInterleave(int b, int m, int m_padded, int n, int n_padded, uint8_t const* SFIn, uint8_t* SFOutput,
     int multiProcessorCount, cudaStream_t stream)
 {
     // Each thread reads 1 int8 value
     dim3 block(std::min(n_padded, 1024));
     // Get number of blocks per SM (assume we can fully utilize the SM).
     int const numBlocksPerSM = std::max(1u, 4096u / block.x);
     dim3 grid(std::min(m_padded, multiProcessorCount * numBlocksPerSM));
 
     block_scale_interleave_kernel<<<grid, block, 0, stream>>>(b, m, m_padded, n, n_padded, SFIn, SFOutput);
 }
 
 // This is intended for weight loading, so m and n are large, b <= 256
 void invokeBlockScaleInterleaveReverse(
     int b, int m, int n, uint8_t const* SFIn, uint8_t* SFOutput, int multiProcessorCount, cudaStream_t stream)
 {
     // Each thread reads 1 int8 value
     dim3 block(std::min(n, 1024));
     // Get number of blocks per SM (assume we can fully utilize the SM).
     int const numBlocksPerSM = std::max(1u, 4096u / block.x);
     dim3 grid(std::min(m, multiProcessorCount * numBlocksPerSM));
 
     block_scale_interleave_reverse_kernel<<<grid, block, 0, stream>>>(b, m, n, SFIn, SFOutput);
 }
 
 template <typename T>
 struct VecTypeImpl
 {
     using type = T;
 };
 
 template <>
 struct VecTypeImpl<half>
 {
     using type = half2;
 };
 
 template <>
 struct VecTypeImpl<__nv_bfloat16>
 {
     using type = __nv_bfloat162;
 };
 
 template <typename T>
 using VecType = typename VecTypeImpl<T>::type;
 
 template <typename T>
 __device__ float getMaxAbs(float4& vec)
 {
     auto absMaxVec = cuda_abs(reinterpret_cast<VecType<T>*>(&vec)[0]);
     for (int i = 1; i < 4; ++i)
     {
         absMaxVec = cuda_max(absMaxVec, cuda_abs(reinterpret_cast<VecType<T>*>(&vec)[i]));
     }
     float absMaxVal;
     if constexpr (sizeof(T) == 4)
     {
         absMaxVal = static_cast<float>(absMaxVec);
     }
     else
     {
         absMaxVal = static_cast<float>(cuda_max(absMaxVec.x, absMaxVec.y));
     }
     tensorrt_llm::common::blockReduceMaxV2<float, 1>(&absMaxVal);
     return absMaxVal;
 }
 
 template <typename T>
 __global__ void computePerTokenGlobalScaleForFP4QuantizationKernel(
     int b, int m, int n, T const* input, int const* tokensPerBatch, float* globalScale)
 {
     static constexpr int ElemsPerVec = 16 / sizeof(T);
     int batchIdx = blockIdx.x;
     int realTokensNum = (tokensPerBatch == nullptr) ? m : tokensPerBatch[batchIdx];
     input += batchIdx * m * n;
     globalScale += batchIdx * m;
     for (int tokenIdx = blockIdx.y; tokenIdx < realTokensNum; tokenIdx += gridDim.y)
     {
         float perTokenMaxAbsVal = 0.f;
         for (int vecIdx = threadIdx.x; vecIdx < n / ElemsPerVec; vecIdx += blockDim.x)
         {
             float4 vec = reinterpret_cast<float4 const*>(input + tokenIdx * n)[vecIdx];
             float maxAbsVal = getMaxAbs<T>(vec);
             perTokenMaxAbsVal = cuda_max(perTokenMaxAbsVal, maxAbsVal);
         }
         float globalScaleVal = 448.f * 6.f / perTokenMaxAbsVal;
         if (threadIdx.x == 0)
         {
             globalScale[tokenIdx] = globalScaleVal;
         }
     }
 }
 
 template <typename T>
 void computePerTokenGlobalScaleForFP4Quantization(int b, int m, int n, T const* input, int const* tokensPerBatch,
     float* globalScale, int multiProcessorCount, cudaStream_t stream)
 {
 
     static constexpr int ElemsPerVec = 16 / sizeof(T);
     TLLM_CHECK(n % (ElemsPerVec * 32) == 0 and b > 0);
     dim3 block(std::min(n / ElemsPerVec, 1024));
     dim3 grid(b, std::max(1, std::min(m, multiProcessorCount / b)));
 
     cudaLaunchConfig_t config;
     config.gridDim = grid;
     config.blockDim = block;
     config.dynamicSmemBytes = 0;
     config.stream = stream;
     cudaLaunchAttribute attrs[1];
     attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
     attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
     config.numAttrs = 1;
     config.attrs = attrs;
     TLLM_CUDA_CHECK(cudaLaunchKernelEx(
         &config, computePerTokenGlobalScaleForFP4QuantizationKernel<T>, b, m, n, input, tokensPerBatch, globalScale));
 }
 
 // Instantiate the function.
 template void invokeFP4Quantization<half, 16>(int b, int m, int n, half const* input, float const* SFScale,
     int64_t* output, int32_t* SFOuput, bool useUE8M0, QuantizationSFLayout layout, int multiProcessorCount,
     cudaStream_t stream, int kernelVersion);
template void invokeFP4Quantization<half, 32>(int b, int m, int n, half const* input, float const* SFScale,
    int64_t* output, int32_t* SFOuput, bool useUE8M0, QuantizationSFLayout layout, int multiProcessorCount,
    cudaStream_t stream, int kernelVersion);
template void computePerTokenGlobalScaleForFP4Quantization<half>(int b, int m, int n, half const* input,
     int const* tokensPerBatch, float* globalScale, int multiProcessorCount, cudaStream_t stream);
 #ifdef ENABLE_BF16
 template void invokeFP4Quantization<__nv_bfloat16, 16>(int b, int m, int n, __nv_bfloat16 const* input,
     float const* SFScale, int64_t* output, int32_t* SFOuput, bool useUE8M0, QuantizationSFLayout layout,
     int multiProcessorCount, cudaStream_t stream, int kernelVersion);
template void invokeFP4Quantization<__nv_bfloat16, 32>(int b, int m, int n, __nv_bfloat16 const* input,
    float const* SFScale, int64_t* output, int32_t* SFOuput, bool useUE8M0, QuantizationSFLayout layout,
    int multiProcessorCount, cudaStream_t stream, int kernelVersion);
template void computePerTokenGlobalScaleForFP4Quantization<__nv_bfloat16>(int b, int m, int n,
     __nv_bfloat16 const* input, int const* tokensPerBatch, float* globalScale, int multiProcessorCount,
     cudaStream_t stream);
 #endif
 
 #ifdef ENABLE_FP8
 template void invokeFP4Quantization<__nv_fp8_e4m3, 16>(int b, int m, int n, __nv_fp8_e4m3 const* input,
     float const* SFScale, int64_t* output, int32_t* SFOuput, bool useUE8M0, QuantizationSFLayout layout,
     int multiProcessorCount, cudaStream_t stream, int kernelVersion);
 template void invokeFP4Quantization<__nv_fp8_e4m3, 32>(int b, int m, int n, __nv_fp8_e4m3 const* input,
     float const* SFScale, int64_t* output, int32_t* SFOuput, bool useUE8M0, QuantizationSFLayout layout,
     int multiProcessorCount, cudaStream_t stream, int kernelVersion);
 #endif
 
 } // namespace kernels
 
 TRTLLM_NAMESPACE_END