/*
 * Standalone compatibility header for TensorRT-LLM quantization kernels.
 *
 * This header replaces all TensorRT-LLM common/ headers with minimal,
 * self-contained implementations of only the utilities used by the
 * FP4 quantization kernels. No TensorRT-LLM dependency required.
 *
 * Extracted from TensorRT-LLM (Apache 2.0 License).
 */

#pragma once

// PyTorch's CUDAExtension adds -D__CUDA_NO_HALF_OPERATORS__ etc. by default,
// which disables implicit conversions and operators for half/bfloat16 types.
// The TRT-LLM quantization kernels rely on these operators, so we must
// re-enable them before including the CUDA headers.
#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_HALF_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_BFLOAT16_OPERATORS__

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>

#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif

// ============================================================================
// Namespace macros (replaces tensorrt_llm/common/config.h)
// ============================================================================

#define TRTLLM_NAMESPACE_BEGIN                                                 \
    namespace tensorrt_llm                                                     \
    {                                                                          \
    inline namespace _v1                                                       \
    {

#define TRTLLM_NAMESPACE_END                                                   \
    }                                                                          \
    }

// ============================================================================
// Assert / check macros (replaces tensorrt_llm/common/assert.h)
// ============================================================================

#define TLLM_CHECK(val)                                                        \
    do                                                                         \
    {                                                                          \
        if (!(val))                                                            \
        {                                                                      \
            throw std::runtime_error(                                          \
                std::string(__FILE__) + ":" + std::to_string(__LINE__) +       \
                " check failed: " #val);                                       \
        }                                                                      \
    } while (0)

#define TLLM_CHECK_WITH_INFO(val, info, ...)                                   \
    do                                                                         \
    {                                                                          \
        if (!(val))                                                            \
        {                                                                      \
            throw std::runtime_error(                                          \
                std::string(__FILE__) + ":" + std::to_string(__LINE__) +       \
                " " + std::string(info));                                      \
        }                                                                      \
    } while (0)

#define TLLM_CUDA_CHECK(call)                                                  \
    do                                                                         \
    {                                                                          \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess)                                                \
        {                                                                      \
            throw std::runtime_error(                                          \
                std::string(__FILE__) + ":" + std::to_string(__LINE__) +       \
                " CUDA error: " + cudaGetErrorString(err));                    \
        }                                                                      \
    } while (0)

// ============================================================================
// TypeConverter (replaces tensorrt_llm/common/cudaTypeUtils.cuh partial)
// ============================================================================

TRTLLM_NAMESPACE_BEGIN

namespace common
{

template <typename T>
struct TypeConverter
{
    using Type = half2;
}; // default

template <>
struct TypeConverter<half2>
{
    using Type = half;
};

template <>
struct TypeConverter<half>
{
    using Type = half2;
};

#ifdef ENABLE_BF16
template <>
struct TypeConverter<__nv_bfloat162>
{
    using Type = __nv_bfloat16;
};

template <>
struct TypeConverter<__nv_bfloat16>
{
    using Type = __nv_bfloat162;
};
#endif // ENABLE_BF16

// ============================================================================
// cuda_abs (replaces tensorrt_llm/common/cudaTypeUtils.cuh partial)
// ============================================================================

template <typename T>
__device__ inline T cuda_abs(T val)
{
    return val < T(0) ? -val : val;
}

template <>
__device__ inline float cuda_abs(float val)
{
    return fabsf(val);
}

template <>
__device__ inline float2 cuda_abs(float2 val)
{
    return make_float2(fabsf(val.x), fabsf(val.y));
}

template <>
__device__ inline half cuda_abs(half val)
{
    return __habs(val);
}

template <>
__device__ inline half2 cuda_abs(half2 val)
{
    return __habs2(val);
}

#ifdef ENABLE_BF16
#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
template <>
__device__ inline __nv_bfloat16 cuda_abs(__nv_bfloat16 val)
{
    return __habs(val);
}

template <>
__device__ inline __nv_bfloat162 cuda_abs(__nv_bfloat162 val)
{
    return __habs2(val);
}
#endif
#endif // ENABLE_BF16

// ============================================================================
// cuda_max (replaces tensorrt_llm/common/cudaTypeUtils.cuh partial)
// ============================================================================

// Unary max: extract scalar max from a vector type
template <typename To, typename Ti>
__device__ inline To cuda_max(Ti val)
{
    return static_cast<To>(val);
}

template <>
__device__ inline float cuda_max(float2 val)
{
    return fmaxf(val.x, val.y);
}

template <>
__device__ inline half cuda_max(half2 val)
{
    return __hmax(val.x, val.y);
}

#ifdef ENABLE_BF16
template <>
__device__ inline __nv_bfloat16 cuda_max(__nv_bfloat162 val)
{
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
    return __hmax(val.x, val.y);
#else
    return __float2bfloat16(0.0f);
#endif
}
#endif

// Binary max: max of two values
template <typename T>
__device__ inline T cuda_max(T val1, T val2)
{
    return (val1 > val2) ? val1 : val2;
}

template <>
__device__ inline float cuda_max(float val1, float val2)
{
    return fmaxf(val1, val2);
}

template <>
__device__ inline float2 cuda_max(float2 val1, float2 val2)
{
    float2 out;
    out.x = fmaxf(val1.x, val2.x);
    out.y = fmaxf(val1.y, val2.y);
    return out;
}

template <>
__device__ inline half2 cuda_max(half2 val1, half2 val2)
{
    return __hmax2(val1, val2);
}

#ifdef ENABLE_BF16
template <>
__device__ inline __nv_bfloat162 cuda_max(__nv_bfloat162 val1, __nv_bfloat162 val2)
{
    return __hmax2(val1, val2);
}
#endif // ENABLE_BF16

// ============================================================================
// Warp / block reduction utilities
// (replaces tensorrt_llm/common/reduceKernelUtils.cuh partial)
// ============================================================================

#define TLLM_FINAL_MASK 0xffffffff

template <typename T, int NUM>
__inline__ __device__ T warpReduceMaxV2(T* val)
{
#pragma unroll
    for (int i = 0; i < NUM; i++)
    {
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            val[i] = max(val[i], __shfl_xor_sync(TLLM_FINAL_MASK, val[i], mask, 32));
    }
    return (T)(0.0f);
}

template <typename T, int NUM>
__inline__ __device__ T blockReduceMaxV2(T* val)
{
    static __shared__ T shared[32][NUM];
    int lane = threadIdx.x & 0x1f; // in-warp idx
    int wid = threadIdx.x >> 5;    // warp idx

    warpReduceMaxV2<T, NUM>(val);  // get max in each warp

    if (lane == 0)                 // record in-warp max by warp idx
    {
#pragma unroll
        for (int i = 0; i < NUM; i++)
        {
            shared[wid][i] = val[i];
        }
    }

    __syncthreads();

    bool is_mask = threadIdx.x < (blockDim.x / 32.f);
#pragma unroll
    for (int i = 0; i < NUM; i++)
    {
        val[i] = is_mask ? shared[lane][i] : (T)-1e20f;
    }
    warpReduceMaxV2<T, NUM>(val);

    return (T)0.0f;
}

// ============================================================================
// Environment utilities (replaces tensorrt_llm/common/envUtils.h)
// ============================================================================

inline int getSMVersion()
{
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    return prop.major * 10 + prop.minor;
}

inline bool getEnvEnablePDL()
{
    static std::once_flag flag;
    static bool enablePDL = true;

    std::call_once(flag,
        [&]()
        {
            if (getSMVersion() >= 90)
            {
                char const* env = std::getenv("TRTLLM_ENABLE_PDL");
                if (env)
                {
                    if (env[0] == '1' && env[1] == '\0')
                    {
                        enablePDL = true;
                    }
                    else if (env[0] == '0' && env[1] == '\0')
                    {
                        enablePDL = false;
                    }
                }
            }
        });
    return enablePDL;
}

// ============================================================================
// Device query (replaces tensorrt_llm/common/cudaUtils.h partial)
// ============================================================================

inline int getMultiProcessorCount()
{
    int nSM = 0;
    int deviceID = 0;
    cudaGetDevice(&deviceID);
    cudaDeviceGetAttribute(&nSM, cudaDevAttrMultiProcessorCount, deviceID);
    return nSM;
}

} // namespace common

TRTLLM_NAMESPACE_END
