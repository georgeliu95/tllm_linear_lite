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
 * Standalone cuBLASLt FP4 GEMM for tllm_linear_lite.
 *
 * Derived from TensorRT-LLM's cublasFp4ScaledMM.cpp and cublasMMWrapper.cpp
 * (Apache 2.0 License). All TRT-LLM internal dependencies replaced with
 * self-contained implementations.
 *
 * Provides:
 *   torch.ops.tllm_linear_lite.cublaslt_fp4_gemm(a, b, scale_a, scale_b, alpha, out_dtype?) -> Tensor
 *
 * This performs: C[m,n] = alpha * (A_fp4[m,k] @ B_fp4[n,k]^T) with block-wise FP8 E4M3 scaling.
 * Uses cuBLASLt's BlockScaleGemm API (CUDA_R_4F_E2M1 + CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3).
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// ============================================================================
// Error checking macros
// ============================================================================

#define CUBLAS_CHECK(call)                                                     \
    do                                                                         \
    {                                                                          \
        cublasStatus_t status = (call);                                        \
        if (status != CUBLAS_STATUS_SUCCESS)                                   \
        {                                                                      \
            throw std::runtime_error(                                          \
                std::string(__FILE__) + ":" + std::to_string(__LINE__) +       \
                " cuBLAS error: " + std::to_string(static_cast<int>(status))); \
        }                                                                      \
    } while (0)

#define CUDA_CHECK(call)                                                       \
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

// cuBLAS workspace: 32 MB (matches TRT-LLM's CUBLAS_WORKSPACE_SIZE)
static constexpr size_t WORKSPACE_SIZE = 33554432;

// ============================================================================
// Thread-local cuBLAS handle management
// ============================================================================

namespace
{

// Simple thread-local singleton for cuBLAS handles (per device)
std::shared_ptr<cublasHandle_t> getCublasHandle()
{
    thread_local std::unordered_map<int, std::shared_ptr<cublasHandle_t>> handles;
    int device;
    cudaGetDevice(&device);

    auto it = handles.find(device);
    if (it != handles.end())
        return it->second;

    auto handle = std::shared_ptr<cublasHandle_t>(new cublasHandle_t, [](cublasHandle_t* h)
        {
            cublasDestroy(*h);
            delete h;
        });
    CUBLAS_CHECK(cublasCreate(handle.get()));
    handles[device] = handle;
    return handle;
}

std::shared_ptr<cublasLtHandle_t> getCublasLtHandle()
{
    thread_local std::unordered_map<int, std::shared_ptr<cublasLtHandle_t>> handles;
    int device;
    cudaGetDevice(&device);

    auto it = handles.find(device);
    if (it != handles.end())
        return it->second;

    auto handle = std::shared_ptr<cublasLtHandle_t>(new cublasLtHandle_t, [](cublasLtHandle_t* h)
        {
            cublasLtDestroy(*h);
            delete h;
        });
    CUBLAS_CHECK(cublasLtCreate(handle.get()));
    handles[device] = handle;
    return handle;
}

// Thread-local workspace tensor (per device)
at::Tensor const& getWorkspaceTensor(c10::Device device)
{
    thread_local std::unordered_map<int, at::Tensor> workspaces;
    int device_id = device.index();

    if (workspaces.find(device_id) == workspaces.end())
    {
        workspaces[device_id] = torch::empty(
            WORKSPACE_SIZE, torch::TensorOptions().dtype(torch::kUInt8).device(device));
    }
    return workspaces[device_id];
}

// Thread-local beta=0.0 device pointer (per device)
float const* getBetaDevicePointer()
{
    thread_local std::unordered_map<int, float*> betas;
    int device;
    cudaGetDevice(&device);

    auto it = betas.find(device);
    if (it != betas.end())
        return it->second;

    float* d_beta;
    CUDA_CHECK(cudaMalloc(&d_beta, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_beta, 0, sizeof(float)));
    betas[device] = d_beta;
    return d_beta;
}

// Convert PyTorch dtype to CUDA data type
cudaDataType_t toCudaDataType(at::ScalarType dtype)
{
    switch (dtype)
    {
    case at::ScalarType::Half: return CUDA_R_16F;
    case at::ScalarType::BFloat16: return CUDA_R_16BF;
    case at::ScalarType::Float: return CUDA_R_32F;
    default:
        throw std::runtime_error(
            "Unsupported output dtype for FP4 GEMM. Supported: Float16, BFloat16, Float32");
    }
}

} // anonymous namespace

// ============================================================================
// cuBLASLt FP4 Block-Scaled GEMM
// ============================================================================

namespace
{

/// Run a single FP4 block-scaled GEMM using cuBLASLt.
///
/// Computes: C[m,n] = alpha * (A_fp4[m,k] @ B_fp4[n,k]^T)
/// with per-16-element FP8 E4M3 block scale factors.
///
/// @param out       Output tensor [m, n] (pre-allocated)
/// @param a         Activation FP4 packed [m, k/2] (uint8)
/// @param b         Weight FP4 packed [n, k/2] (uint8)
/// @param scale_a   Activation scale factors (FP8 E4M3 as uint8, swizzled)
/// @param scale_b   Weight scale factors (FP8 E4M3 as uint8, swizzled)
/// @param alpha     Global scale tensor [1] (float32, device)
/// @param bias      Optional bias tensor [n] (same dtype as output, or empty)
/// @param algo      Optional algorithm (nullptr for heuristic default)
void cublaslt_fp4_gemm_impl(
    torch::Tensor& out,
    torch::Tensor const& a,
    torch::Tensor const& b,
    torch::Tensor const& scale_a,
    torch::Tensor const& scale_b,
    torch::Tensor const& alpha,
    std::optional<at::Tensor> const& bias = std::nullopt,
    cublasLtMatmulAlgo_t const* algo = nullptr)
{
    int32_t m = a.sizes()[0];
    int32_t n = b.sizes()[0];
    int32_t k = a.sizes()[1] * 2; // FP4: 2 values per byte

    at::cuda::CUDAGuard deviceGuard(a.device());
    auto ltHandle = getCublasLtHandle();
    auto stream = at::cuda::getCurrentCUDAStream(a.get_device()).stream();

    cudaDataType_t outType = toCudaDataType(out.scalar_type());

    // --- Create descriptors ---
    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;

    // FP4 GEMM: A/B are CUDA_R_4F_E2M1, compute in FP32
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));

    // Layout: we swap A and B for row-major -> column-major conversion
    // cuBLASLt sees: B^T[k,n] x A[k,m] = C[n,m] which is C[m,n] in row-major
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // FP4 requires device pointer mode
    cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode)));

    // Fast accumulation disabled
    int8_t fastAcc = 0;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fastAcc, sizeof(fastAcc)));

    // Matrix layouts (after swap: first=B[n,k], second=A[m,k])
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_4F_E2M1, k, n, k)); // B transposed
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_4F_E2M1, k, m, k)); // A not transposed
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, outType, n, m, n));
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&Ddesc, outType, n, m, n));

    // --- Set scale descriptors (FP4 block scaling) ---
    void* a_sf_ptr = const_cast<void*>(reinterpret_cast<void const*>(scale_b.data_ptr())); // swapped
    void* b_sf_ptr = const_cast<void*>(reinterpret_cast<void const*>(scale_a.data_ptr())); // swapped
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_sf_ptr, sizeof(void*)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_sf_ptr, sizeof(void*)));

    // Block scale modes: 16-element blocks with UE4M3 scale factors
    cublasLtMatmulMatrixScale_t scaleMode16 = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulMatrixScale_t scaleMode32F = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode16, sizeof(scaleMode16)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode16, sizeof(scaleMode16)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_C_SCALE_MODE, &scaleMode32F, sizeof(scaleMode32F)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_MODE, &scaleMode32F, sizeof(scaleMode32F)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, &scaleMode32F, sizeof(scaleMode32F)));

    // --- Set bias epilogue if bias is provided ---
    if (bias.has_value() && bias->defined())
    {
        void* bias_ptr = const_cast<void*>(bias->data_ptr());
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_ptr, sizeof(void*)));

        cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
            operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)));
    }

    // C/D scale pointers = nullptr
    void const* null_ptr = nullptr;
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, &null_ptr, sizeof(null_ptr)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &null_ptr, sizeof(null_ptr)));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, &null_ptr, sizeof(null_ptr)));

    // --- Select algorithm ---
    cublasLtMatmulAlgo_t const* selected_algo = algo;
    cublasLtMatmulAlgo_t default_algo;

    if (algo == nullptr)
    {
        // Use heuristic
        cublasLtMatmulPreference_t preference;
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&preference));
        CUBLAS_CHECK(cublasLtMatmulPreferenceInit(preference));
        uint64_t ws_size = WORKSPACE_SIZE;
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, sizeof(ws_size)));

        std::vector<cublasLtMatmulHeuristicResult_t> heuristics(200);
        int return_count = 0;
        CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
            *ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc,
            preference, heuristics.size(), heuristics.data(), &return_count));
        cublasLtMatmulPreferenceDestroy(preference);

        if (return_count > 0 && heuristics[0].state == CUBLAS_STATUS_SUCCESS
            && heuristics[0].workspaceSize <= WORKSPACE_SIZE)
        {
            default_algo = heuristics[0].algo;
            selected_algo = &default_algo;
        }
    }

    // --- Execute GEMM ---
    float const* alpha_ptr = alpha.data_ptr<float>();
    float const* beta_ptr = getBetaDevicePointer();
    auto const& workspace = getWorkspaceTensor(a.device());

    // After swap: A=b_ptr, B=a_ptr
    CUBLAS_CHECK(cublasLtMatmul(
        *ltHandle, operationDesc,
        alpha_ptr,
        b.data_ptr(), Adesc,   // B (swapped to A position)
        a.data_ptr(), Bdesc,   // A (swapped to B position)
        beta_ptr,
        out.data_ptr(), Cdesc,
        out.data_ptr(), Ddesc,
        selected_algo,
        workspace.data_ptr(), WORKSPACE_SIZE,
        stream));

    // --- Cleanup ---
    cublasLtMatrixLayoutDestroy(Ddesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatmulDescDestroy(operationDesc);
}

} // anonymous namespace

// ============================================================================
// PyTorch Op
// ============================================================================

/// cuBLASLt FP4 GEMM PyTorch op with optional bias epilogue fusion.
///
/// @param a          Activation FP4 packed [m, k/2] (uint8)
/// @param b          Weight FP4 packed [n, k/2] (uint8)
/// @param scale_a    Activation scale factors (FP8 E4M3, swizzled layout)
/// @param scale_b    Weight scale factors (FP8 E4M3, swizzled layout)
/// @param alpha      Global alpha [1] float32 tensor (device)
/// @param bias       Optional bias [n] tensor (same dtype as output). Fused in cuBLASLt epilogue.
/// @param out_dtype  Optional output dtype (default: bfloat16)
/// @return           Output tensor [m, n]
at::Tensor cublaslt_fp4_gemm(
    at::Tensor const& a,
    at::Tensor const& b,
    at::Tensor const& scale_a,
    at::Tensor const& scale_b,
    at::Tensor const& alpha,
    std::optional<at::Tensor> const& bias,
    std::optional<c10::ScalarType> out_dtype)
{
    TORCH_CHECK(a.is_cuda(), "a must be a CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be a CUDA tensor");
    TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "b must be contiguous");
    TORCH_CHECK(alpha.numel() > 0 && alpha.dtype() == torch::kFloat32, "alpha must be non-empty float32");

    int64_t m = a.sizes()[0];
    int64_t n = b.sizes()[0];

    if (bias.has_value() && bias->defined())
    {
        TORCH_CHECK(bias->is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias->numel() == n, "bias must have size [n]");
    }

    at::ScalarType dtype = out_dtype.value_or(at::ScalarType::BFloat16);
    at::Tensor out = at::empty({m, n}, a.options().dtype(dtype));

    cublaslt_fp4_gemm_impl(out, a, b, scale_a, scale_b, alpha, bias);

    return out;
}

// ============================================================================
// Op registration
// ============================================================================

TORCH_LIBRARY_FRAGMENT(tllm_linear_lite, m)
{
    m.def(
        "cublaslt_fp4_gemm(Tensor a, Tensor b, Tensor scale_a, Tensor scale_b, "
        "Tensor alpha, Tensor? bias=None, ScalarType? out_dtype=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(tllm_linear_lite, CUDA, m)
{
    m.impl("cublaslt_fp4_gemm", &cublaslt_fp4_gemm);
}
