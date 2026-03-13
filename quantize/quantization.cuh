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
 * This file is derived from TensorRT-LLM's tensorrt_llm/kernels/quantization.cuh.
 *
 * Changes from the original:
 *   - Replaced includes: tensorrt_llm/common/{assert,config,cudaTypeUtils,cudaUtils,
 *     quantTypeUtils,reduceKernelUtils}.cuh/h, tensorrt_llm/kernels/quantization.h
 *     -> tllm_compat.cuh + quantization.h (local headers)
 *   - Removed: quantizedKernel() x3 overloads   (INT8 scalar quantization kernels for float4/half2/bf162)
 *   - Removed: DstVec<T,N> template              (output vector packing for per-token quant)
 *   - Removed: clampAndAbsMax()                   (clamping helper for per-token quant)
 *   - Removed: quantizeAndStore()                 (quantize + store helper for per-token quant)
 *   - Removed: perTokenQuantization<T> kernel     (per-token INT8/FP8 quantization kernel,
 *              depends on QuantMode, QuantTypeStaticVals, packed_as, blockAllReduceMax,
 *              blockReduceSumV2, cuda_cast, cuda_clamp, cuda_sum)
 *   - Kept:    FP4/MXFP8 quantization kernels and helpers:
 *              fp32_vec_to_e2m1, fp32_vec_to_e4m3, reciprocal_approximate_ftz, exp2f_rcp,
 *              PackedVec, cvt_warp_fp16_to_fp4, cvt_warp_fp8_to_fp4, cvt_warp_fp16_to_mxfp8,
 *              get_sf_out_offset_128x4, cvt_quant_get_sf_out_offset, quantize_with_block_size
 *   - Note:    cvt_warp_fp16_to_mxfp8 is kept because quantize_with_block_size template
 *              references BlockScaleQuantizationType::FP16_TO_MXFP8; removing it would
 *              require modifying the kernel template structure.
 *   - Added:   opt_quantize_with_block_size_v1 — optimized kernel (16 elems/thread,
 *              256-bit vectorized load via LDG.E.128×2, better ILP)
 *   - Added:   FourOverSix adaptive 4/6 block scaling:
 *              AdaptiveScaleRule enum (NONE/MSE/MAE/ABS_MAX),
 *              fake_quant_e2m1_8<Rule> (register-efficient fake-quant + error),
 *              cvt_warp_fp16_to_fp4_adaptive (per-block r=6 vs r=4 selection),
 *              opt_quantize_with_block_size_adaptive kernel (streaming, ~21 R32 peak)
 */

#include "tllm_compat.cuh"
#include "quantization.h"
#include <float.h>
 
 using namespace tensorrt_llm::common;
 
 TRTLLM_NAMESPACE_BEGIN
 
namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////
// FP4/MXFP8 Quantization
 
 constexpr int CVT_ELTS_PER_THREAD = 8;
 constexpr int CVT_FP4_THREADS_PER_WARP = 32;
 constexpr int CVT_FP8_TO_FP4_ELTS_PER_THREAD = 16;
 
 // Convert 8 float32 values into 8 e2m1 values (represented as one uint32_t).
 inline __device__ uint32_t fp32_vec_to_e2m1(float (&array)[8])
 {
 #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
     uint32_t val;
     asm volatile(
         "{\n"
         ".reg .b8 byte0;\n"
         ".reg .b8 byte1;\n"
         ".reg .b8 byte2;\n"
         ".reg .b8 byte3;\n"
         "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
         "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
         "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
         "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
         "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
         "}"
         : "=r"(val)
         : "f"(array[0]), "f"(array[1]), "f"(array[2]), "f"(array[3]), "f"(array[4]), "f"(array[5]), "f"(array[6]),
         "f"(array[7]));
     return val;
 #else
     // static_assert(false, "not supported.");
     return 0;
 #endif
 }
 
 // Convert 4 float2 values into 8 e2m1 values (represented as one uint32_t).
 inline __device__ uint32_t fp32_vec_to_e2m1(float2 (&array)[4])
 {
 #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
     uint32_t val;
     asm volatile(
         "{\n"
         ".reg .b8 byte0;\n"
         ".reg .b8 byte1;\n"
         ".reg .b8 byte2;\n"
         ".reg .b8 byte3;\n"
         "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
         "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
         "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
         "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
         "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
         "}"
         : "=r"(val)
         : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y), "f"(array[2].x), "f"(array[2].y),
         "f"(array[3].x), "f"(array[3].y));
     return val;
 #else
     // static_assert(false, "not supported.");
     return 0;
 #endif
 }
 
 // Convert 8 float2 values into 16 e2m1 values (represented as one uint64_t).
 inline __device__ uint64_t fp32_vec_to_e2m1(float2 (&array)[8])
 {
 #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
     uint64_t val;
     asm volatile(
         "{\n"
         ".reg .b8 byte0;\n"
         ".reg .b8 byte1;\n"
         ".reg .b8 byte2;\n"
         ".reg .b8 byte3;\n"
         ".reg .b8 byte4;\n"
         ".reg .b8 byte5;\n"
         ".reg .b8 byte6;\n"
         ".reg .b8 byte7;\n"
         ".reg .b32 val0;\n"
         ".reg .b32 val1;\n"
         "cvt.rn.satfinite.e2m1x2.f32   byte0,  %2,  %1;\n"
         "cvt.rn.satfinite.e2m1x2.f32   byte1,  %4,  %3;\n"
         "cvt.rn.satfinite.e2m1x2.f32   byte2,  %6,  %5;\n"
         "cvt.rn.satfinite.e2m1x2.f32   byte3,  %8,  %7;\n"
         "cvt.rn.satfinite.e2m1x2.f32   byte4, %10,  %9;\n"
         "cvt.rn.satfinite.e2m1x2.f32   byte5, %12, %11;\n"
         "cvt.rn.satfinite.e2m1x2.f32   byte6, %14, %13;\n"
         "cvt.rn.satfinite.e2m1x2.f32   byte7, %16, %15;\n"
         "mov.b32 val0, {byte0, byte1, byte2, byte3};\n"
         "mov.b32 val1, {byte4, byte5, byte6, byte7};\n"
         "mov.b64 %0, {val0, val1};\n"
         "}"
         : "=l"(val)
         : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y), "f"(array[2].x), "f"(array[2].y),
         "f"(array[3].x), "f"(array[3].y), "f"(array[4].x), "f"(array[4].y), "f"(array[5].x), "f"(array[5].y),
         "f"(array[6].x), "f"(array[6].y), "f"(array[7].x), "f"(array[7].y));
     return val;
 #else
     // static_assert(false, "not supported.");
     return 0;
 #endif
 }
 
 // Convert 4 float2 values into 8 e4m3 values (represented as one uint64_t).
 inline __device__ uint64_t fp32_vec_to_e4m3(float2 (&array)[4])
 {
     union
     {
         uint64_t val;
         __nv_fp8x2_e4m3 elts[4];
     } u;
 
     static_assert(sizeof(u.val) == sizeof(u.elts), "Expected to alias uint64_t and __nv_fp8x2_e4m3[4]");
 
     u.elts[0] = __nv_fp8x2_e4m3(array[0]);
     u.elts[1] = __nv_fp8x2_e4m3(array[1]);
     u.elts[2] = __nv_fp8x2_e4m3(array[2]);
     u.elts[3] = __nv_fp8x2_e4m3(array[3]);
     return u.val;
 }
 
 // Fast reciprocal.
 inline __device__ float reciprocal_approximate_ftz(float a)
 {
     float b;
     asm volatile("rcp.approx.ftz.f32 %0, %1;\n" : "=f"(b) : "f"(a));
     return b;
 }
 
 __device__ __forceinline__ float exp2f_rcp(uint8_t exp)
 {
     constexpr uint32_t FP32_EXPONENT_BIAS = 127;
     return (exp == 0) ? 1 : exp2f(FP32_EXPONENT_BIAS - static_cast<float>(exp));
 }
 
 // Define a 16 bytes packed data type.
 template <class Type>
 struct PackedVec
 {
     typename TypeConverter<Type>::Type elts[4];
     static_assert(sizeof(elts) == sizeof(Type) * CVT_ELTS_PER_THREAD,
         "Vector size should match the number of elements per thread.");
 };
 
 template <>
 struct PackedVec<__nv_fp8_e4m3>
 {
     __nv_fp8x2_e4m3 elts[8];
     static_assert(sizeof(elts) == sizeof(__nv_fp8_e4m3) * CVT_FP8_TO_FP4_ELTS_PER_THREAD,
         "Vector size should match the number of elements per thread.");
 };
 
 // Quantizes the provided PackedVec into the uint32_t output
 template <class Type, int SF_VEC_SIZE, bool UE8M0_SF>
 __device__ uint32_t cvt_warp_fp16_to_fp4(PackedVec<Type>& vec, float SFScaleVal, uint8_t* SFout)
 {
 #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
     // Get absolute maximum values among the local 8 values.
     auto localMax = cuda_abs(vec.elts[0]);
 
 // Local maximum value.
 #pragma unroll
     for (int i = 1; i < CVT_ELTS_PER_THREAD / 2; i++)
     {
         localMax = cuda_max(localMax, cuda_abs(vec.elts[i]));
     }
 
     constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / CVT_ELTS_PER_THREAD;
     // Get the absolute maximum among all 16 values (two threads for 16, four threads for 32).
     localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
     if constexpr (CVT_NUM_THREADS_PER_SF == 4)
     {
         localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 2), localMax);
     }
     // Get the final absolute maximum values.
     float vecMax = float(cuda_max(localMax.x, localMax.y));
 
     // 8 bits representation of the SF.
     uint8_t fp8SFVal;
     float outputScale;
     // Write the SF to global memory (STG.8).
     if constexpr (UE8M0_SF)
     {
         __nv_fp8_e8m0 tmp;
         // Scale the max value to the range of E2m1.
         vecMax *= reciprocal_approximate_ftz(6.0f);
         tmp.__x = __nv_cvt_float_to_e8m0(vecMax, __NV_SATFINITE, cudaRoundPosInf);
         fp8SFVal = tmp.__x;
         outputScale = vecMax != 0 ? exp2f_rcp(fp8SFVal) : 0.0f;
     }
     else
     {
         // Get the SF (max value of the vector / max value of e2m1).
         // maximum value of e2m1 = 6.0.
         // TODO: use half as compute data type.
         auto SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
         // Here SFValue is always positive, so E4M3 is the same as UE4M3.
         __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
         fp8SFVal = tmp.__x;
         SFValue = static_cast<float>(tmp);
         // Get the output scale.
         // Recipe: final_scale = reciprocal(fp32(fp8(SFValue * SFScaleVal)) * reciprocal(SFScaleVal))
         outputScale = vecMax != 0 ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal)) : 0.0f;
     }
 
     if (SFout)
     {
         // Write the SF to global memory (STG.8).
         *SFout = fp8SFVal;
     }
 
     // Convert the input to float.
     float2 fp2Vals[CVT_ELTS_PER_THREAD / 2];
 
 #pragma unroll
     for (int i = 0; i < CVT_ELTS_PER_THREAD / 2; i++)
     {
         if constexpr (std::is_same_v<Type, half>)
         {
             fp2Vals[i] = __half22float2(vec.elts[i]);
         }
         else
         {
             fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
         }
         fp2Vals[i].x *= outputScale;
         fp2Vals[i].y *= outputScale;
     }
 
     // Convert to e2m1 values.
     uint32_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);
 
     // Write the e2m1 values to global memory.
     return e2m1Vec;
 #else
     return 0;
 #endif
 }
 
 template <class Type, int SF_VEC_SIZE, bool UE8M0_SF>
 __device__ uint64_t cvt_warp_fp8_to_fp4(PackedVec<Type>& vec, float SFScaleVal, uint8_t* SFout)
 {
 #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
 
     float const dequant_to_fp16_scale = 6.f * reciprocal_approximate_ftz(SFScaleVal);
 
     // Dequant fp8 to fp16
     __half2 vec_half2[8];
 #pragma unroll
     for (int i = 0; i < CVT_FP8_TO_FP4_ELTS_PER_THREAD / 2; i++)
     {
         float2 tmp = static_cast<float2>(vec.elts[i]);
         tmp.x *= dequant_to_fp16_scale;
         tmp.y *= dequant_to_fp16_scale;
         vec_half2[i] = __float22half2_rn(tmp);
     }
 
     // Get absolute maximum values among the local 8 values.
     auto localMax = __habs2(vec_half2[0]);
     // Local maximum value.
 #pragma unroll
     for (int i = 1; i < CVT_FP8_TO_FP4_ELTS_PER_THREAD / 2; i++)
     {
         localMax = __hmax2(localMax, __habs2(vec_half2[i]));
     }
 
     constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / CVT_FP8_TO_FP4_ELTS_PER_THREAD;
     if constexpr (CVT_NUM_THREADS_PER_SF == 2)
     {
         // For block 32, we need to reduce the local max across two threads.
         localMax = __hmax2(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
     }
 
     // Get the final absolute maximum values.
     float vecMax = float(__hmax(localMax.x, localMax.y));
 
     // Get the SF (max value of the vector / max value of e2m1).
     // maximum value of e2m1 = 6.0.
     // TODO: use half as compute data type.
     float SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
     float SFValueNarrow;
     // 8 bits representation of the SF.
     uint8_t fp8SFVal;
     // Write the SF to global memory (STG.8).
     if constexpr (UE8M0_SF)
     {
         __nv_fp8_e8m0 tmp;
         tmp.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);
         SFValueNarrow = static_cast<float>(tmp);
         fp8SFVal = tmp.__x;
     }
     else
     {
         // Here SFValue is always positive, so E4M3 is the same as UE4M3.
         __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
         fp8SFVal = tmp.__x;
         SFValueNarrow = static_cast<float>(tmp);
     }
     // Get the output scale.
     // Recipe: final_scale = reciprocal(fp32(fp8(SFValue * SFScaleVal))) * reciprocal(SFScaleVal))
     float outputScale = SFValue != 0 ? SFScaleVal * reciprocal_approximate_ftz(SFValueNarrow) : 0.0f;
 
     if (SFout)
     {
         // Write the SF to global memory (STG.8).
         *SFout = fp8SFVal;
     }
 
     // Convert the input to float.
     float2 fp2Vals[CVT_FP8_TO_FP4_ELTS_PER_THREAD / 2];
 
 #pragma unroll
     for (int i = 0; i < CVT_FP8_TO_FP4_ELTS_PER_THREAD / 2; i++)
     {
         fp2Vals[i] = __half22float2(vec_half2[i]);
         fp2Vals[i].x *= outputScale;
         fp2Vals[i].y *= outputScale;
     }
 
     // Convert to e2m1 values.
     uint64_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);
 
     // Write the e2m1 values to global memory.
     return e2m1Vec;
 #else
     return 0;
 #endif
 }
 
 // Quantizes the provided PackedVec into the uint64_t output
 template <class Type, int SF_VEC_SIZE>
 __device__ uint64_t cvt_warp_fp16_to_mxfp8(PackedVec<Type>& vec, uint8_t* SFout)
 {
 #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
     // Get absolute maximum values among the local 8 values.
     auto localMax = cuda_abs(vec.elts[0]);
 
 // Local maximum value.
 #pragma unroll
     for (int i = 1; i < CVT_ELTS_PER_THREAD / 2; i++)
     {
         localMax = cuda_max(localMax, cuda_abs(vec.elts[i]));
     }
 
     constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / CVT_ELTS_PER_THREAD;
     // Get the absolute maximum among all 16 values (two threads for 16, four threads for 32).
     localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
     if constexpr (CVT_NUM_THREADS_PER_SF == 4)
     {
         localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 2), localMax);
     }
     // Get the final absolute maximum values.
     float vecMax = float(cuda_max(localMax.x, localMax.y));
 
     // Get the SF (max value of the vector / max value of mxfp8).
     float SFValue = vecMax * reciprocal_approximate_ftz(448.0f);
     // 8 bits representation of the SF.
     uint8_t fp8SFVal;
     // Write the SF to global memory (STG.8).
     __nv_fp8_e8m0 tmpSFVal;
     tmpSFVal.__x = __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf);
     SFValue = static_cast<float>(tmpSFVal);
     fp8SFVal = tmpSFVal.__x;
     // Get the output scale (reciprocal of the SFValue).
     float outputScale = vecMax != 0.f ? reciprocal_approximate_ftz(SFValue) : 0.0f;
 
     if (SFout)
     {
         // Write the SF to global memory (STG.8).
         *SFout = fp8SFVal;
     }
 
     // Convert the input to float.
     float2 fp2Vals[CVT_ELTS_PER_THREAD / 2];
 
 #pragma unroll
     for (int i = 0; i < CVT_ELTS_PER_THREAD / 2; i++)
     {
         if constexpr (std::is_same_v<Type, half>)
         {
             fp2Vals[i] = __half22float2(vec.elts[i]);
         }
         else
         {
             fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
         }
         fp2Vals[i].x *= outputScale;
         fp2Vals[i].y *= outputScale;
     }
 
     // Convert to e4m3 values.
     uint64_t e4m3Vec = fp32_vec_to_e4m3(fp2Vals);
 
     // Write the e4m3 values to global memory.
     return e4m3Vec;
 #else
     return 0;
 #endif
 }
 
 inline __host__ __device__ int64_t get_sf_out_offset_128x4(
     std::optional<int> batchIdx, int mIdx, int kIdx, std::optional<int> numRows, int numColVecs)
 {
     // SF layout [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
     // --> index [mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]
 
     // batched tensor
     // SF layout [numBTiles, numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
     // --> index [bTileIdx, mTileIdx, kTileIdx, outerMIdx, innerMIdx, innerKIdx]
 
     int32_t innerKIdx = (kIdx % 4);
     int64_t innerKStride = 1;
 
     int32_t innerMIdx = (mIdx % (32 * 4)) / 32;
     int64_t innerMStride = 4 * innerKStride; // 4
 
     // M tile layout [32, 4] is column-major.
     int32_t outerMIdx = (mIdx % 32);
     int64_t outerMStride = 4 * innerMStride; // 16
 
     int32_t kTileIdx = (kIdx / 4);
     int64_t kTileStride = 32 * outerMStride; // 512
 
     // SF vector size 16 or 32. We round the "numCols" up to a multiple of 64 or 128.
     // It is the same as rounding the "numColVecs" up to a multiple of 4.
     int32_t numKTiles = (numColVecs + 4 - 1) / 4;
     int32_t mTileIdx = mIdx / (32 * 4);
     int64_t mTileStride = numKTiles * kTileStride;
 
     // Each SF block has 128 rows so pad rows to the multiple of 128.
     int32_t numMTiles = (numRows.value_or(0) + 128 - 1) / 128;
     int64_t bTileStride = numMTiles * mTileStride;
 
     // Compute the global offset.
     int64_t SFOffset = batchIdx.value_or(0) * bTileStride + mTileIdx * mTileStride + kTileIdx * kTileStride
         + outerMIdx * outerMStride + innerMIdx * innerMStride + innerKIdx * innerKStride;
 
     return SFOffset;
 }
 
 template <class SFType, int CVT_NUM_THREADS_PER_SF>
 __device__ uint8_t* cvt_quant_get_sf_out_offset(std::optional<int> batchIdx, int rowIdx, int colVecIdx,
     std::optional<int> numRows, int numColVecs, SFType* SFout, QuantizationSFLayout layout)
 {
 #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
     // Each thread holds one vector.
     static_assert(CVT_NUM_THREADS_PER_SF == 1 || CVT_NUM_THREADS_PER_SF == 2 || CVT_NUM_THREADS_PER_SF == 4);
 
     // One pair of threads write one SF to global memory.
     // TODO: stage through smem for packed STG.32
     // is it better than STG.8 from 4 threads ?
     if (threadIdx.x % CVT_NUM_THREADS_PER_SF == 0)
     {
         if (layout == QuantizationSFLayout::SWIZZLED)
         {
             // SF vector index (16 elements share one SF in the K dimension).
             // numRows and numCols are unpadded.
             int32_t kIdx = colVecIdx / CVT_NUM_THREADS_PER_SF;
             int32_t mIdx = rowIdx;
 
             auto SFOffset = get_sf_out_offset_128x4(batchIdx, mIdx, kIdx, numRows, numColVecs);
             return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
         }
         else if (layout == QuantizationSFLayout::LINEAR)
         {
             // Linear row-major layout, no padding required.
             int32_t KTileIdx = colVecIdx / CVT_NUM_THREADS_PER_SF;
 
             int32_t numKTiles = numColVecs;
             int64_t mTileStride = numKTiles;
 
             int64_t BTileStride = numRows.value_or(0) * mTileStride;
 
             int64_t SFOffset = batchIdx.value_or(0) * BTileStride + rowIdx * mTileStride + KTileIdx;
             return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
         }
         else
         {
             return nullptr;
         }
     }
 #endif
     return nullptr;
 }
 
 template <BlockScaleQuantizationType quantization_type, class Type, int SF_VEC_SIZE, bool UE8M0_SF>
 __global__ void
 #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
     __launch_bounds__(512, 4) quantize_with_block_size(
 #else
 quantize_with_block_size(
 #endif
         int32_t numbatches, int32_t numRows, int32_t numCols, int32_t numPaddedCols, Type const* in,
         float const* SFScale, uint32_t* out, uint32_t* SFout, QuantizationSFLayout layout)
 {
 #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
 
     // The elements per thread.
     static constexpr int ELTS_PER_THREAD = quantization_type == BlockScaleQuantizationType::FP8_TO_FP4
         ? CVT_FP8_TO_FP4_ELTS_PER_THREAD
         : CVT_ELTS_PER_THREAD;
 
     using PackedVec = PackedVec<Type>;
     static constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / ELTS_PER_THREAD; // 2 or 4
     static_assert(sizeof(PackedVec) == sizeof(Type) * ELTS_PER_THREAD, "Vec size is not matched.");
 
     // Get the global scaling factor, which will be applied to the SF.
     // Note SFScale is the same as next GEMM's alpha, which is (448.f / (Alpha_A / 6.f)).
     // This value is prepared by model, no need to be protected by ACKBULK
     float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];
 
     // Is it swizzled layout?
     bool isSfSwizzledLayout = layout == QuantizationSFLayout::SWIZZLED;
 
     // The number of padded rows considering 128x4 SF layout.
     int numPaddedRowsForSf = isSfSwizzledLayout ? PadUpFn(numRows, 128) : numRows;
     int numColsForSf = isSfSwizzledLayout ? PadUpFn(numPaddedCols, 4 * SF_VEC_SIZE) : numPaddedCols;
 
     // The number of threads in the column dimension。
     // Note that numCols/numPaddedCols/numColsForSf are guaranteed to be multiples of ELTS_PER_THREAD.
     int numColThreads = numCols / ELTS_PER_THREAD;
     int numPaddedColThreads = numPaddedCols / ELTS_PER_THREAD;
     int numColThreadsForSf = numColsForSf / ELTS_PER_THREAD;
 
     cudaGridDependencySynchronize();
     // Input tensor batch/row/col loops.
     // Optimization: Iterate over actual rows first (hot path), then padding rows (cold path)
     // This improves performance for small batch sizes with swizzled layout
     for (int rowIdx = blockIdx.x; rowIdx < numPaddedRowsForSf; rowIdx += gridDim.x)
     {
         // Early exit for padding-only blocks: if this block only processes padding rows,
         // we can skip the batch loop and just zero out the scale factors
         bool isRowPadding = (rowIdx >= numRows);
 
         if (isRowPadding)
         {
             // Fast path: This row is entirely padding, only zero out scale factors.
             // Note: Padding rows do NOT exist in the output tensor (which is sized [numRows, K]),
             // they only exist in the swizzled scale factor layout. Do NOT write to output buffer here.
             for (int batchIdx = 0; batchIdx < numbatches; batchIdx++)
             {
                 for (int colIdx = threadIdx.x; colIdx < numColThreadsForSf; colIdx += blockDim.x)
                 {
                     std::optional<int> optionalBatchIdx = batchIdx;
                     std::optional<int> optionalNumRows = numRows;
 
                     // The SF output pointer.
                     auto sf_out = cvt_quant_get_sf_out_offset<uint32_t, CVT_NUM_THREADS_PER_SF>(
                         optionalBatchIdx, rowIdx, colIdx, optionalNumRows, numPaddedCols / SF_VEC_SIZE, SFout, layout);
 
                     // Set the SF padding to 0.
                     if (sf_out != nullptr)
                     {
                         sf_out[0] = 0x00;
                     }
                 }
             }
         }
         else
         {
             // Normal path: This row contains actual data
             for (int batchIdx = 0; batchIdx < numbatches; batchIdx++)
             {
                 for (int colIdx = threadIdx.x; colIdx < numColThreadsForSf; colIdx += blockDim.x)
                 {
                     std::optional<int> optionalBatchIdx = batchIdx;
                     std::optional<int> optionalNumRows = numRows;
 
                     // The SF output pointer.
                     auto sf_out = cvt_quant_get_sf_out_offset<uint32_t, CVT_NUM_THREADS_PER_SF>(
                         optionalBatchIdx, rowIdx, colIdx, optionalNumRows, numPaddedCols / SF_VEC_SIZE, SFout, layout);
 
                     // The input tensor offset.
                     int64_t inOffset = static_cast<int64_t>(batchIdx * numRows + rowIdx) * numColThreads + colIdx;
                     int64_t outOffset
                         = static_cast<int64_t>(batchIdx * numRows + rowIdx) * numPaddedColThreads + colIdx;
 
                     // Set the values to 0 of those are padded columns.
                     if (colIdx >= numColThreads && colIdx < numPaddedColThreads)
                     {
                         // Dispatch the quantization kernel.
                         if constexpr (quantization_type == BlockScaleQuantizationType::FP16_TO_FP4)
                         {
                             reinterpret_cast<uint32_t*>(out)[outOffset] = 0u;
                         }
                         else if constexpr (quantization_type == BlockScaleQuantizationType::FP8_TO_FP4
                             || quantization_type == BlockScaleQuantizationType::FP16_TO_MXFP8)
                         {
                             reinterpret_cast<uint64_t*>(out)[outOffset] = 0ull;
                         }
                     }
 
                     // Set the SF padding to 0.
                     if (colIdx >= numColThreads)
                     {
                         // Set the SF padding to 0.
                         if (sf_out != nullptr)
                         {
                             sf_out[0] = 0x00;
                         }
                     }
                     else
                     {
                         // Load the input vector.
                         PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
 
                         // Dispatch the quantization kernel.
                         if constexpr (quantization_type == BlockScaleQuantizationType::FP16_TO_FP4)
                         {
                             reinterpret_cast<uint32_t*>(out)[outOffset]
                                 = cvt_warp_fp16_to_fp4<Type, SF_VEC_SIZE, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
                         }
                         else if constexpr (quantization_type == BlockScaleQuantizationType::FP8_TO_FP4)
                         {
                             reinterpret_cast<uint64_t*>(out)[outOffset]
                                 = cvt_warp_fp8_to_fp4<__nv_fp8_e4m3, SF_VEC_SIZE, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
                         }
                         else if constexpr (quantization_type == BlockScaleQuantizationType::FP16_TO_MXFP8)
                         {
                             reinterpret_cast<uint64_t*>(out)[outOffset]
                                 = cvt_warp_fp16_to_mxfp8<Type, SF_VEC_SIZE>(in_vec, sf_out);
                         }
                     }
                 }
             }
         }
     }
     cudaTriggerProgrammaticLaunchCompletion();
 #endif
 }
 
////////////////////////////////////////////////////////////////////////////////////////////////////
// v1 Optimization: 16 elements per thread, 256-bit vectorized load, increased ILP

// v1 processes 16 elements per thread (vs 8 in v0).
constexpr int CVT_OPT_ELTS_PER_THREAD = 16;

/*
 * PackedVec_Opt: 32-byte aligned vector holding 16 FP16/BF16 elements (8 vec2 values).
 * Used exclusively by opt_quantize_with_block_size_v1.
 */
template <class Type>
struct __align__(32) PackedVec_Opt
{
    typename TypeConverter<Type>::Type elts[8];
    static_assert(sizeof(elts) == sizeof(Type) * CVT_OPT_ELTS_PER_THREAD,
        "Vector size should match the number of elements per thread.");
};

/*
 * Load 32 bytes (256 bits) from global memory using two consecutive 128-bit loads.
 * The two-instruction sequence guarantees LDG.E.128 in SASS, which is optimal for
 * 32-byte aligned global loads. A single 256-bit PTX load is not available on sm_100.
 */
template <class T>
__device__ __forceinline__ void load_256bit(T* dst, void const* src)
{
    static_assert(sizeof(T) == 32, "load_256bit requires T to be exactly 32 bytes");
    uint32_t* dst_u32 = reinterpret_cast<uint32_t*>(dst);
    char const* src_char = reinterpret_cast<char const*>(src);

    // First 128-bit load (bytes 0-15)
    asm volatile(
        "ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(dst_u32[0]), "=r"(dst_u32[1]), "=r"(dst_u32[2]), "=r"(dst_u32[3])
        : "l"(src_char)
    );

    // Second 128-bit load (bytes 16-31)
    asm volatile(
        "ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
        : "=r"(dst_u32[4]), "=r"(dst_u32[5]), "=r"(dst_u32[6]), "=r"(dst_u32[7])
        : "l"(src_char + 16)
    );
}

/*
 * Optimized (v1) inner implementation: process 16 elements per thread for increased ILP.
 *
 * Key differences from cvt_warp_fp16_to_fp4 (v0, 8 elements/thread):
 * - Input: PackedVec_Opt with 8 half2/bfloat162 (16 elements, 32 bytes)
 * - Output: uint64_t (16 e2m1 values)
 * - SF thread cooperation:
 *   - SF_VEC_SIZE=16: CVT_NUM_THREADS_PER_SF=1 (single thread, no shuffle)
 *   - SF_VEC_SIZE=32: CVT_NUM_THREADS_PER_SF=2 (two threads, one shuffle)
 *
 * More compute per memory load hides L1TEX scoreboard stall (~7 cycles, ~40% of CPI).
 */
template <class Type, int SF_VEC_SIZE, bool UE8M0_SF, typename VecType>
__device__ uint64_t cvt_warp_fp16_to_fp4_impl_opt(VecType& vec, float SFScaleVal, uint8_t* SFout)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // Get absolute maximum values among the local 16 values (8 vec2 elements).
    auto localMax = cuda_abs(vec.elts[0]);

#pragma unroll
    for (int i = 1; i < CVT_OPT_ELTS_PER_THREAD / 2; i++)
    {
        localMax = cuda_max(localMax, cuda_abs(vec.elts[i]));
    }

    // SF thread cooperation:
    // - SF_VEC_SIZE=16, CVT_OPT_ELTS_PER_THREAD=16: CVT_NUM_THREADS_PER_SF=1 (single thread)
    // - SF_VEC_SIZE=32, CVT_OPT_ELTS_PER_THREAD=16: CVT_NUM_THREADS_PER_SF=2 (two threads)
    constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / CVT_OPT_ELTS_PER_THREAD;
    static_assert(CVT_NUM_THREADS_PER_SF == 1 || CVT_NUM_THREADS_PER_SF == 2,
        "v1 only supports SF_VEC_SIZE of 16 (1 thread) or 32 (2 threads)");

    // Cross-thread max reduction only needed when two threads share one SF slot.
    if constexpr (CVT_NUM_THREADS_PER_SF == 2)
    {
        localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
    }

    float vecMax = float(cuda_max(localMax.x, localMax.y));

    uint8_t fp8SFVal;
    float outputScale;
    if constexpr (UE8M0_SF)
    {
        __nv_fp8_e8m0 tmp;
        vecMax *= reciprocal_approximate_ftz(6.0f);
        tmp.__x = __nv_cvt_float_to_e8m0(vecMax, __NV_SATFINITE, cudaRoundPosInf);
        fp8SFVal = tmp.__x;
        outputScale = vecMax != 0 ? exp2f_rcp(fp8SFVal) : 0.0f;
    }
    else
    {
        // maximum value of e2m1 = 6.0
        auto SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
        __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
        fp8SFVal = tmp.__x;
        SFValue = static_cast<float>(tmp);
        outputScale = vecMax != 0 ? reciprocal_approximate_ftz(SFValue * reciprocal_approximate_ftz(SFScaleVal)) : 0.0f;
    }

    if (SFout)
    {
        *SFout = fp8SFVal;
    }

    // Convert the input to float: 8 vec2 elements = 16 floats.
    float2 fp2Vals[CVT_OPT_ELTS_PER_THREAD / 2];

#pragma unroll
    for (int i = 0; i < CVT_OPT_ELTS_PER_THREAD / 2; i++)
    {
        if constexpr (std::is_same_v<Type, half>)
        {
            fp2Vals[i] = __half22float2(vec.elts[i]);
        }
        else
        {
            fp2Vals[i] = __bfloat1622float2(vec.elts[i]);
        }
        fp2Vals[i].x *= outputScale;
        fp2Vals[i].y *= outputScale;
    }

    // Convert 16 floats to 16 e2m1 values packed in uint64_t.
    uint64_t e2m1Vec = fp32_vec_to_e2m1(fp2Vals);
    return e2m1Vec;
#else
    return 0;
#endif
}

/*
 * v1 Kernel: Optimized 16 elements per thread for increased ILP (FP16/BF16 to FP4 only).
 *
 * This kernel processes 16 elements per thread instead of 8, doubling the
 * compute work per memory load. This helps hide L1TEX scoreboard stalls
 * by providing more independent operations for the scheduler.
 *
 * Key differences from v0 (quantize_with_block_size):
 * - ELTS_PER_THREAD: 16 instead of 8
 * - PackedVec: 32 bytes instead of 16 bytes (32-byte aligned vectorized load via load_256bit)
 * - Output: uint64_t (16 e2m1 values) instead of uint32_t (8 e2m1 values)
 * - numColThreads: numCols / 16 instead of numCols / 8
 *
 * NOTE: This v1 kernel only supports FP16/BF16 to FP4 quantization.
 * FP8 to FP4 and FP16 to MXFP8 are not supported in v1.
 */
template <BlockScaleQuantizationType quantization_type, class Type, int SF_VEC_SIZE, bool UE8M0_SF>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    __launch_bounds__(512, 4) opt_quantize_with_block_size_v1(
#else
opt_quantize_with_block_size_v1(
#endif
        int32_t numbatches, int32_t numRows, int32_t numCols, int32_t numPaddedCols, Type const* in,
        float const* SFScale, uint32_t* out, uint32_t* SFout, QuantizationSFLayout layout)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

    // v1 only supports FP16/BF16 to FP4 quantization.
    static_assert(quantization_type == BlockScaleQuantizationType::FP16_TO_FP4,
        "opt_quantize_with_block_size_v1 only supports FP16_TO_FP4 quantization type");

    static constexpr int ELTS_PER_THREAD = CVT_OPT_ELTS_PER_THREAD;
    using PackedVec = PackedVec_Opt<Type>;
    static constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / ELTS_PER_THREAD;
    static_assert(sizeof(PackedVec) == sizeof(Type) * ELTS_PER_THREAD, "Vec size is not matched.");
    static_assert(CVT_NUM_THREADS_PER_SF == 1 || CVT_NUM_THREADS_PER_SF == 2,
        "v1 only supports SF_VEC_SIZE of 16 (1 thread) or 32 (2 threads)");

    float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];

    bool isSfSwizzledLayout = layout == QuantizationSFLayout::SWIZZLED;
    int numPaddedRowsForSf = isSfSwizzledLayout ? PadUpFn(numRows, 128) : numRows;
    int numColsForSf = isSfSwizzledLayout ? PadUpFn(numPaddedCols, 4 * SF_VEC_SIZE) : numPaddedCols;

    // v1 processes 16 elements per thread, so numColThreads = numCols / 16.
    int numColThreads = numCols / ELTS_PER_THREAD;
    int numPaddedColThreads = numPaddedCols / ELTS_PER_THREAD;
    int numColThreadsForSf = numColsForSf / ELTS_PER_THREAD;

    asm volatile("griddepcontrol.wait;");
    for (int rowIdx = blockIdx.x; rowIdx < numPaddedRowsForSf; rowIdx += gridDim.x)
    {
        for (int batchIdx = 0; batchIdx < numbatches; batchIdx++)
        {
            for (int colIdx = threadIdx.x; colIdx < numColThreadsForSf; colIdx += blockDim.x)
            {
                std::optional<int> optionalBatchIdx = batchIdx;
                std::optional<int> optionalNumRows = numRows;

                // Each thread covers SF_VEC_SIZE/16 = 1 or 2 SF slots.
                auto sf_out = cvt_quant_get_sf_out_offset<uint32_t, CVT_NUM_THREADS_PER_SF>(
                    optionalBatchIdx, rowIdx, colIdx, optionalNumRows, numPaddedCols / SF_VEC_SIZE, SFout, layout);

                int64_t inOffset = static_cast<int64_t>(batchIdx * numRows + rowIdx) * numColThreads + colIdx;
                int64_t outOffset = static_cast<int64_t>(batchIdx * numRows + rowIdx) * numPaddedColThreads + colIdx;

                // Zero out padded columns in the output tensor.
                if (rowIdx < numRows && colIdx >= numColThreads && colIdx < numPaddedColThreads)
                {
                    // v1 outputs uint64_t (16 e2m1 values).
                    reinterpret_cast<uint64_t*>(out)[outOffset] = 0ull;
                }

                // Zero out SF for padding rows or padding columns.
                if (rowIdx >= numRows || colIdx >= numColThreads)
                {
                    if (sf_out != nullptr)
                    {
                        sf_out[0] = 0x00;
                    }
                }
                else
                {
                    // Load the input vector using 256-bit vectorized load (two 128-bit loads).
                    // This guarantees LDG.E.128 instructions in SASS. See load_256bit() for details.
                    PackedVec in_vec;
                    load_256bit(&in_vec, reinterpret_cast<char const*>(in) + inOffset * sizeof(PackedVec));

                    // Dispatch the v1 quantization implementation (16 elements, uint64_t output).
                    reinterpret_cast<uint64_t*>(out)[outOffset]
                        = cvt_warp_fp16_to_fp4_impl_opt<Type, SF_VEC_SIZE, UE8M0_SF>(in_vec, SFScaleVal, sf_out);
                }
            }
        }
    }
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// FourOverSix Adaptive 4/6 Block Scaling

enum class AdaptiveScaleRule { NONE = 0, MSE = 1, MAE = 2, ABS_MAX = 3 };

/*
 * Fake-quantize 4 half2 values (8 elements) to E2M1, dequantize back, and
 * accumulate error — all in a single register-efficient pass.
 *
 * Accepts half2 input directly (no float array materialization). Float
 * conversion happens just-in-time inside the loop body, and the converted
 * values are immediately consumed for scaling, error computation, then
 * discarded — keeping register pressure minimal.
 *
 * Returns the packed E2M1 output (uint32_t = 8 nibbles).
 *
 * Processes 8 elements (4 half2) rather than the full 16-element block to cap
 * register pressure. The caller invokes this twice (lo + hi halves) so the
 * second call reuses the same physical regs after the first's die.
 * PTX `mov.b32 {b0,b1,b2,b3}` naturally packs 4 bytes = 8 nibbles = 1 uint32.
 *
 * Quantization uses the f16x2/bf16x2 → e2m1x2 path (PTX ISA 9.1, CUDA 13.1+):
 * scale multiplication stays in half precision (HMUL2, 1 inst per pair) and
 * cvt.rn.satfinite.e2m1x2.f16x2 takes the half2 directly — avoiding the
 * half→float conversion + f32 scale multiply of the original f32 path.
 * Trade-off: fp16 scale has 10-bit mantissa vs fp32's 23-bit; acceptable for
 * the adaptive error-comparison use case (both candidates lose similar precision).
 * Requires: CUDA 13.1+ (PTX ISA 9.1). CUDA 13.0 ptxas will reject this.
 */
template <AdaptiveScaleRule Rule, class HalfType>
__device__ __forceinline__ uint32_t fake_quant_e2m1_8(
    HalfType const (&h2)[4], float outputScale, float decodeScale, float& err)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // --- Quant: scale in half precision + f16x2/bf16x2 → e2m1x2 (PTX ISA 9.1) ---
    // 4 × HMUL2 + 4 × cvt.e2m1x2.f16x2 = 8 instructions
    // (was: 4 × F2FP + 8 × FMUL + 4 × cvt.e2m1x2.f32 = 16 instructions)
    uint32_t e2m1_packed;
    {
        HalfType scale_vec;
        if constexpr (std::is_same_v<HalfType, __half2>)
            scale_vec = __float2half2_rn(outputScale);
        else
            scale_vec = __float2bfloat162_rn(outputScale);

        HalfType sq0 = __hmul2(h2[0], scale_vec);
        HalfType sq1 = __hmul2(h2[1], scale_vec);
        HalfType sq2 = __hmul2(h2[2], scale_vec);
        HalfType sq3 = __hmul2(h2[3], scale_vec);

        uint32_t u0 = reinterpret_cast<uint32_t&>(sq0);
        uint32_t u1 = reinterpret_cast<uint32_t&>(sq1);
        uint32_t u2 = reinterpret_cast<uint32_t&>(sq2);
        uint32_t u3 = reinterpret_cast<uint32_t&>(sq3);

        if constexpr (std::is_same_v<HalfType, __half2>)
        {
            asm(
                "{\n"
                ".reg .b8 byte0, byte1, byte2, byte3;\n"
                "cvt.rn.satfinite.e2m1x2.f16x2 byte0, %1;\n"
                "cvt.rn.satfinite.e2m1x2.f16x2 byte1, %2;\n"
                "cvt.rn.satfinite.e2m1x2.f16x2 byte2, %3;\n"
                "cvt.rn.satfinite.e2m1x2.f16x2 byte3, %4;\n"
                "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
                "}"
                : "=r"(e2m1_packed)
                : "r"(u0), "r"(u1), "r"(u2), "r"(u3)
            );
        }
        else
        {
            asm(
                "{\n"
                ".reg .b8 byte0, byte1, byte2, byte3;\n"
                "cvt.rn.satfinite.e2m1x2.bf16x2 byte0, %1;\n"
                "cvt.rn.satfinite.e2m1x2.bf16x2 byte1, %2;\n"
                "cvt.rn.satfinite.e2m1x2.bf16x2 byte2, %3;\n"
                "cvt.rn.satfinite.e2m1x2.bf16x2 byte3, %4;\n"
                "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
                "}"
                : "=r"(e2m1_packed)
                : "r"(u0), "r"(u1), "r"(u2), "r"(u3)
            );
        }
    }

    // Per-byte dequant: each DEQUANT_BYTE extracts one .b8 from e2m1_packed and
    // converts back to f16x2. Separate asm blocks (non-volatile) so the compiler
    // can interleave dequant with error computation if profitable.
#define DEQUANT_BYTE(IDX, BSEL)                                       \
    uint32_t dq##IDX;                                                  \
    asm(                                                               \
        "{\n"                                                          \
        ".reg .b8 b0, b1, b2, b3;\n"                                  \
        "mov.b32 {b0, b1, b2, b3}, %1;\n"                             \
        "cvt.rn.f16x2.e2m1x2 %0, " BSEL ";\n"                        \
        "}\n"                                                          \
        : "=r"(dq##IDX) : "r"(e2m1_packed)                            \
    )

#define ERROR_PAIR(IDX)                                                \
    {                                                                  \
        float2 orig;                                                   \
        if constexpr (std::is_same_v<HalfType, __half2>)               \
            orig = __half22float2(h2[IDX]);                            \
        else                                                           \
            orig = __bfloat1622float2(h2[IDX]);                        \
        half2 dq_h = reinterpret_cast<half2&>(dq##IDX);               \
        float2 dq_f = __half22float2(dq_h);                           \
        dq_f.x *= decodeScale;                                        \
        dq_f.y *= decodeScale;                                        \
        float dx = dq_f.x - orig.x;                                   \
        float dy = dq_f.y - orig.y;                                   \
        if constexpr (Rule == AdaptiveScaleRule::MSE)                  \
            err += dx * dx + dy * dy;                                  \
        else if constexpr (Rule == AdaptiveScaleRule::MAE)             \
            err += fabsf(dx) + fabsf(dy);                             \
        else if constexpr (Rule == AdaptiveScaleRule::ABS_MAX)         \
            err = fmaxf(err, fmaxf(fabsf(dx), fabsf(dy)));           \
    }

    DEQUANT_BYTE(0, "b0"); ERROR_PAIR(0);
    DEQUANT_BYTE(1, "b1"); ERROR_PAIR(1);
    DEQUANT_BYTE(2, "b2"); ERROR_PAIR(2);
    DEQUANT_BYTE(3, "b3"); ERROR_PAIR(3);

#undef DEQUANT_BYTE
#undef ERROR_PAIR

    return e2m1_packed;
#else
    return 0;
#endif
}

/*
 * Adaptive 4/6 inner implementation (streaming, 16 elements per thread).
 *
 * Sequentially computes fake-quant + error for r=6 then r=4, selecting the
 * candidate with lower error. Only ~21 R32 registers at peak — no spilling.
 *
 * The r=4 candidate uses scale_expansion_factor=1.5 (sf_4 = sf_6_hp * 1.5),
 * mapping input to E2M1 range [-4, 4] instead of [-6, 6]. This avoids the
 * coarse 4→6 interval (gap=2) at the cost of one fewer quantization level.
 *
 * Both candidates' SF values fit in E4M3 (max 256 for r=6, max 384 for r=4,
 * both < 448). The decoder is agnostic to the r choice — it simply uses
 * x_reconstructed = e2m1_val * sf_e4m3 / SFScaleVal.
 */
template <class Type, int SF_VEC_SIZE, AdaptiveScaleRule Rule, typename VecType>
__device__ uint64_t cvt_warp_fp16_to_fp4_adaptive(VecType& vec, float SFScaleVal, uint8_t* SFout)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    static_assert(Rule != AdaptiveScaleRule::NONE, "Use cvt_warp_fp16_to_fp4_impl_opt for non-adaptive");

    // --- Step 1: Compute vecMax (same as v1) ---
    auto localMax = cuda_abs(vec.elts[0]);
#pragma unroll
    for (int i = 1; i < CVT_OPT_ELTS_PER_THREAD / 2; i++)
    {
        localMax = cuda_max(localMax, cuda_abs(vec.elts[i]));
    }

    constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / CVT_OPT_ELTS_PER_THREAD;
    static_assert(CVT_NUM_THREADS_PER_SF == 1 || CVT_NUM_THREADS_PER_SF == 2,
        "adaptive only supports SF_VEC_SIZE of 16 (1 thread) or 32 (2 threads)");

    if constexpr (CVT_NUM_THREADS_PER_SF == 2)
    {
        localMax = cuda_max(__shfl_xor_sync(uint32_t(-1), localMax, 1), localMax);
    }

    float vecMax = float(cuda_max(localMax.x, localMax.y));

    // --- Step 2: Compute E4M3 SF candidates for r=6 and r=4 ---
    float sf_hp = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
    __nv_fp8_e4m3 sf_6_e4m3 = __nv_fp8_e4m3(sf_hp);
    __nv_fp8_e4m3 sf_4_e4m3 = __nv_fp8_e4m3(sf_hp * 1.5f);

    float rcp_SFScale = reciprocal_approximate_ftz(SFScaleVal);

    // Zero-copy references: reinterpret vec.elts[0..3] and vec.elts[4..7] as
    // const-ref arrays without copying values into separate registers.
    // Saves ~8 regs vs naive `lo_6[4] = {vec.elts[0], ...}` value copy.
    using HalfVecT = typename TypeConverter<Type>::Type;
    using Arr4Ref = HalfVecT const (&)[4];
    Arr4Ref lo = reinterpret_cast<Arr4Ref>(vec.elts[0]);
    Arr4Ref hi = reinterpret_cast<Arr4Ref>(vec.elts[4]);

    // --- Step 3+4: r=6 fake-quant + error (stored as initial winner) ---
    // Scoped block: sf_f / outputScale / decodeScale die at '}', freeing ~3 regs
    // before r=4 scope allocates its own set — prevents both sets coexisting.
    float best_err = 0.0f;
    uint32_t e2m1_lo, e2m1_hi;
    __nv_fp8_e4m3 sf_best = sf_6_e4m3;
    {
        float sf_f = static_cast<float>(sf_6_e4m3);
        float decodeScale = sf_f * rcp_SFScale;
        float outputScale = vecMax != 0 ? reciprocal_approximate_ftz(decodeScale) : 0.0f;
        // Process 16 elements as two batches of 8 to keep register pressure low.
        // fake_quant_e2m1_8 peaks at 8 regs (s0-s7 scaled floats for quant asm input);
        // a hypothetical 16-at-once would need 16 simultaneous regs, blowing the budget.
        // The second call reuses the same physical regs after the first call's s0-s7 die.
        // best_err accumulates across both calls (passed by reference).
        e2m1_lo = fake_quant_e2m1_8<Rule>(lo, outputScale, decodeScale, best_err);
        e2m1_hi = fake_quant_e2m1_8<Rule>(hi, outputScale, decodeScale, best_err);
    }

    // --- Step 5: r=4 fake-quant + error ---
    // Scoped block reuses the same ~3 reg slots freed from r=6 scope above.
    float err_4 = 0.0f;
    uint32_t e2m1_4_lo, e2m1_4_hi;
    {
        float sf_f = static_cast<float>(sf_4_e4m3);
        float decodeScale = sf_f * rcp_SFScale;
        float outputScale = vecMax != 0 ? reciprocal_approximate_ftz(decodeScale) : 0.0f;
        e2m1_4_lo = fake_quant_e2m1_8<Rule>(lo, outputScale, decodeScale, err_4);
        e2m1_4_hi = fake_quant_e2m1_8<Rule>(hi, outputScale, decodeScale, err_4);
    }

    // --- Step 6: For SF_VEC_SIZE=32, reduce errors across 2 threads ---
    if constexpr (CVT_NUM_THREADS_PER_SF == 2)
    {
        if constexpr (Rule == AdaptiveScaleRule::ABS_MAX)
        {
            best_err = fmaxf(best_err, __shfl_xor_sync(uint32_t(-1), best_err, 1));
            err_4 = fmaxf(err_4, __shfl_xor_sync(uint32_t(-1), err_4, 1));
        }
        else
        {
            best_err += __shfl_xor_sync(uint32_t(-1), best_err, 1);
            err_4 += __shfl_xor_sync(uint32_t(-1), err_4, 1);
        }
    }

    // --- Step 7: Overwrite with r=4 if it produces lower error ---
    // Serialized winner selection: r=6 result already in e2m1_lo/hi, conditionally
    // overwrite with r=4. Avoids keeping separate winner_lo/hi/sf variables.
    if (err_4 < best_err)
    {
        e2m1_lo = e2m1_4_lo;
        e2m1_hi = e2m1_4_hi;
        sf_best = sf_4_e4m3;
    }

    if (SFout)
    {
        *SFout = sf_best.__x;
    }

    // Pack two uint32_t halves into one uint64_t.
    uint64_t e2m1_out;
    asm volatile("mov.b64 %0, {%1, %2};" : "=l"(e2m1_out) : "r"(e2m1_lo), "r"(e2m1_hi));
    return e2m1_out;
#else
    return 0;
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// FourOverSix Adaptive Kernel: streaming 4/6 selection with per-thread sequential error comparison

/*
 * Adaptive kernel: same structure as opt_quantize_with_block_size_v1 but calls
 * cvt_warp_fp16_to_fp4_adaptive for per-block 4/6 scale selection.
 *
 * Only supports FP16/BF16→FP4 with E4M3 scale factors (not UE8M0).
 * The caller must pass globalScale = 1536/amax (not 2688/amax).
 */
template <BlockScaleQuantizationType quantization_type, class Type, int SF_VEC_SIZE, AdaptiveScaleRule Rule>
__global__ void
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    // (512, 3): register budget = 65536 / (512*3) = 42 regs/thread.
    // Adaptive kernel compiles to ~40 regs — fits within budget with no spill.
    // (512, 4) would force 32-reg budget → spill to local memory (L2 traffic 4.5x).
    // (512, 2) eliminates spill but compiler inflates to 63 regs → occupancy 33%.
    // (512, 3) at 40 regs → 4 blocks/SM → 56% occupancy — best trade-off.
    __launch_bounds__(512, 3) opt_quantize_with_block_size_adaptive(
#else
opt_quantize_with_block_size_adaptive(
#endif
        int32_t numbatches, int32_t numRows, int32_t numCols, int32_t numPaddedCols, Type const* in,
        float const* SFScale, uint32_t* out, uint32_t* SFout, QuantizationSFLayout layout)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)

    static_assert(quantization_type == BlockScaleQuantizationType::FP16_TO_FP4,
        "adaptive kernel only supports FP16_TO_FP4");
    static_assert(Rule != AdaptiveScaleRule::NONE,
        "use opt_quantize_with_block_size_v1 for non-adaptive quantization");

    static constexpr int ELTS_PER_THREAD = CVT_OPT_ELTS_PER_THREAD;
    using PackedVec = PackedVec_Opt<Type>;
    static constexpr int CVT_NUM_THREADS_PER_SF = SF_VEC_SIZE / ELTS_PER_THREAD;
    static_assert(sizeof(PackedVec) == sizeof(Type) * ELTS_PER_THREAD, "Vec size mismatch.");
    static_assert(CVT_NUM_THREADS_PER_SF == 1 || CVT_NUM_THREADS_PER_SF == 2,
        "adaptive only supports SF_VEC_SIZE of 16 or 32");

    float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[0];

    bool isSfSwizzledLayout = layout == QuantizationSFLayout::SWIZZLED;
    int numPaddedRowsForSf = isSfSwizzledLayout ? PadUpFn(numRows, 128) : numRows;
    int numColsForSf = isSfSwizzledLayout ? PadUpFn(numPaddedCols, 4 * SF_VEC_SIZE) : numPaddedCols;

    int numColThreads = numCols / ELTS_PER_THREAD;
    int numPaddedColThreads = numPaddedCols / ELTS_PER_THREAD;
    int numColThreadsForSf = numColsForSf / ELTS_PER_THREAD;

    asm volatile("griddepcontrol.wait;");
    for (int rowIdx = blockIdx.x; rowIdx < numPaddedRowsForSf; rowIdx += gridDim.x)
    {
        for (int batchIdx = 0; batchIdx < numbatches; batchIdx++)
        {
            for (int colIdx = threadIdx.x; colIdx < numColThreadsForSf; colIdx += blockDim.x)
            {
                std::optional<int> optionalBatchIdx = batchIdx;
                std::optional<int> optionalNumRows = numRows;

                auto sf_out = cvt_quant_get_sf_out_offset<uint32_t, CVT_NUM_THREADS_PER_SF>(
                    optionalBatchIdx, rowIdx, colIdx, optionalNumRows, numPaddedCols / SF_VEC_SIZE, SFout, layout);

                int64_t inOffset = static_cast<int64_t>(batchIdx * numRows + rowIdx) * numColThreads + colIdx;
                int64_t outOffset = static_cast<int64_t>(batchIdx * numRows + rowIdx) * numPaddedColThreads + colIdx;

                if (rowIdx < numRows && colIdx >= numColThreads && colIdx < numPaddedColThreads)
                {
                    reinterpret_cast<uint64_t*>(out)[outOffset] = 0ull;
                }

                if (rowIdx >= numRows || colIdx >= numColThreads)
                {
                    if (sf_out != nullptr)
                    {
                        sf_out[0] = 0x00;
                    }
                }
                else
                {
                    PackedVec in_vec;
                    load_256bit(&in_vec, reinterpret_cast<char const*>(in) + inOffset * sizeof(PackedVec));

                    reinterpret_cast<uint64_t*>(out)[outOffset]
                        = cvt_warp_fp16_to_fp4_adaptive<Type, SF_VEC_SIZE, Rule>(in_vec, SFScaleVal, sf_out);
                }
            }
        }
    }
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void block_scale_interleave_kernel(
    int numbatches, int numRows, int numCols, uint8_t const* SFIn, uint8_t* SFOutput);
} // namespace kernels

TRTLLM_NAMESPACE_END