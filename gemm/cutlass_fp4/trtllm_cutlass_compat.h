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
 * [Standalone -- new file]
 *
 * Compatibility shim for CUTLASS FP4 GEMM kernels extracted from TensorRT-LLM.
 * Provides all TRT-LLM macros, types, and utility functions used by the
 * FP4 GEMM template headers, without depending on TRT-LLM.
 *
 * Replaces: tensorrt_llm/common/{config,assert,cudaUtils,envUtils,logger}.h
 * Provides: TRTLLM_NAMESPACE_BEGIN/END, TLLM_CHECK, TLLM_LOG_*, check_cuda_error,
 *           getSMVersion, getEnvEnablePDL, getMaxSharedMemoryPerBlockOptin
 */

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <mutex>
#include <stdexcept>
#include <string>

// ============================================================================
// Namespace macros (tensorrt_llm/common/config.h)
// ============================================================================

#ifndef TRTLLM_NAMESPACE_BEGIN
#define TRTLLM_NAMESPACE_BEGIN \
    namespace tensorrt_llm    \
    {                         \
    inline namespace _v1      \
    {
#define TRTLLM_NAMESPACE_END \
    }                        \
    }
#endif

// ============================================================================
// Assert / check macros (tensorrt_llm/common/assert.h)
// ============================================================================

#ifndef TLLM_CHECK_WITH_INFO
#define TLLM_CHECK_WITH_INFO(val, info, ...)                              \
    do                                                                    \
    {                                                                     \
        if (!(val))                                                       \
        {                                                                 \
            throw std::runtime_error(                                     \
                std::string(__FILE__) + ":" + std::to_string(__LINE__) +  \
                " " + std::string(info));                                 \
        }                                                                 \
    } while (0)
#endif

#ifndef TLLM_CHECK
#define TLLM_CHECK(val) TLLM_CHECK_WITH_INFO(val, "check failed: " #val)
#endif

// ============================================================================
// Logging (tensorrt_llm/common/logger.h) -- no-ops for standalone
// ============================================================================

#ifndef TLLM_LOG_DEBUG
#define TLLM_LOG_DEBUG(...) ((void)0)
#endif

#ifndef TLLM_LOG_WARNING
#define TLLM_LOG_WARNING(...) ((void)0)
#endif

#ifndef TLLM_LOG_INFO
#define TLLM_LOG_INFO(...) ((void)0)
#endif

// ============================================================================
// CUDA utilities (tensorrt_llm/common/cudaUtils.h subset)
// ============================================================================

TRTLLM_NAMESPACE_BEGIN

namespace common
{

// check_cuda_error: a real inline function (not a macro!) so that
// `tk::check_cuda_error(...)` works when tk = tensorrt_llm::common.
// TRT-LLM defines it as a macro that calls `check(...)`, but macro expansion
// with namespace prefix `tk::` causes double-qualification. Using a real
// function avoids this issue entirely.
template <typename T>
inline void check_cuda_error(T result)
{
    if (result)
    {
        throw std::runtime_error(
            std::string("CUDA error: ") + std::to_string(static_cast<int>(result)));
    }
}

inline void sync_check_cuda_error(cudaStream_t stream)
{
    // In debug mode you might want cudaStreamSynchronize + check here.
    // For release, just check last error.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        throw std::runtime_error(
            std::string("CUDA sync error: ") + cudaGetErrorString(err));
    }
}

inline int getSMVersion()
{
    int device = 0;
    cudaGetDevice(&device);
    int sm_major = 0, sm_minor = 0;
    cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&sm_minor, cudaDevAttrComputeCapabilityMinor, device);
    return sm_major * 10 + sm_minor;
}

inline int getMaxSharedMemoryPerBlockOptin()
{
    int device = 0;
    cudaGetDevice(&device);
    int smem = 0;
    cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    return smem;
}

inline bool getEnvEnablePDL()
{
    static std::once_flag flag;
    static bool enablePDL = true;
    std::call_once(flag, [&]()
    {
        char const* env = std::getenv("TRTLLM_ENABLE_PDL");
        if (env)
        {
            enablePDL = (env[0] == '1' && env[1] == '\0');
        }
    });
    return enablePDL;
}

} // namespace common

TRTLLM_NAMESPACE_END

// Convenience alias used throughout TRT-LLM kernel code
namespace tk = tensorrt_llm::common;
