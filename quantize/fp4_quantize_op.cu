/*
 * PyTorch Op wrapper for FP4 quantization kernels.
 *
 * Registers the following ops under the "tllm_linear_lite" namespace:
 *   - torch.ops.tllm_linear_lite.fp4_quantize
 *   - torch.ops.tllm_linear_lite.calculate_nvfp4_global_scale
 *
 * Based on TensorRT-LLM fp4Quantize.cpp (Apache 2.0 License).
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <optional>

#include "tllm_compat.cuh"
#include "quantization.h"

// ---------------------------------------------------------------------------
// Convenience macros (replaces tensorrt_llm/thop/thUtils.h)
// ---------------------------------------------------------------------------

#define CHECK_TH_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_TYPE(x, st) \
    TORCH_CHECK(x.scalar_type() == st, #x " dtype is ", x.scalar_type(), ", while ", st, " is expected")
#define CHECK_INPUT(x, st) \
    CHECK_TH_CUDA(x);      \
    CHECK_CONTIGUOUS(x);    \
    CHECK_TYPE(x, st)

// FP4 E2M1 packed as uint8 (2 values per byte)
constexpr auto FLOAT4_E2M1X2 = torch::ScalarType::Byte;
// Scale factor dtype: FP8 E4M3 stored as uint8
constexpr auto SF_DTYPE = torch::ScalarType::Byte;

// ---------------------------------------------------------------------------
// fp4_quantize
// ---------------------------------------------------------------------------

/// Quantize a 2D+ tensor from fp16/bf16/fp8 to FP4 (E2M1) with block scaling.
///
/// @param self          Input tensor [*, K], fp16/bf16/fp8_e4m3
/// @param globalScale   Optional [1] float tensor = (448 * 6) / self.abs().max()
/// @param sfVecSize     Scale factor vector size: 16 (NVFP4) or 32 (MXFP4)
/// @param sfUseUE8M0    If true, use UE8M0 scale factors (MXFP4 style)
/// @param isSfSwizzledLayout If true, scale factors in swizzled layout for CUTLASS
/// @return (fp4_values, scale_factors) tuple
std::tuple<at::Tensor, at::Tensor> fp4_quantize(
    at::Tensor const& self,
    std::optional<at::Tensor> const& globalScale,
    int64_t sfVecSize,
    bool sfUseUE8M0,
    bool isSfSwizzledLayout)
{
    CHECK_TH_CUDA(self);
    CHECK_CONTIGUOUS(self);
    if (sfUseUE8M0)
    {
        TORCH_CHECK(sfVecSize == 32, "sfVecSize can only be 32, when sfUseUE8M0 is true");
    }
    else
    {
        TORCH_CHECK(globalScale.has_value(), "globalScale is required when sfUseUE8M0 is false");
        CHECK_INPUT(globalScale.value(), torch::kFloat32);
        TORCH_CHECK(sfVecSize == 16, "sfVecSize can only be 16, when sfUseUE8M0 is false");
    }

    float* globalScalePtr = nullptr;
    if (globalScale.has_value())
    {
        globalScalePtr = globalScale->data_ptr<float>();
    }

    auto const& inputShape = self.sizes();
    auto const& rank = inputShape.size();

    TORCH_CHECK(rank >= 2, "Input should be >=2D tensor.");
    int64_t m = 1;
    for (size_t i = 0; i < rank - 1; i++)
    {
        m *= inputShape[i];
    }
    auto const k = inputShape[rank - 1];
    TORCH_CHECK(k % sfVecSize == 0, "Last dimension must be divisible by sfVecSize");

    // Output shape: same as input but last dim halved (2 FP4 values per byte)
    std::vector<int64_t> outputShape(inputShape.begin(), inputShape.end());
    outputShape[rank - 1] = k / 2;

    at::Tensor valueE2M1 = at::empty(outputShape, self.options().dtype(FLOAT4_E2M1X2));

    // Scale factor size depends on layout
    int64_t SFSize = isSfSwizzledLayout
        ? tensorrt_llm::computeSwizzledLayoutSFSize(m, k / sfVecSize)
        : tensorrt_llm::computeLinearLayoutSFSize(m, k / sfVecSize);

    at::Tensor scaleFP8SF = at::empty({SFSize}, self.options().dtype(SF_DTYPE));

    thread_local int const mMultiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();

    auto const layout = isSfSwizzledLayout
        ? tensorrt_llm::QuantizationSFLayout::SWIZZLED
        : tensorrt_llm::QuantizationSFLayout::LINEAR;

    auto stream = at::cuda::getCurrentCUDAStream(self.get_device()).stream();

#define LAUNCH_FP4_QUANTIZE_KERNEL(T, SF_VEC_SIZE)                                          \
    tensorrt_llm::kernels::invokeFP4Quantization<T, SF_VEC_SIZE>(                           \
        1, m, k,                                                                            \
        reinterpret_cast<T*>(self.data_ptr()),                                              \
        globalScalePtr,                                                                     \
        reinterpret_cast<int64_t*>(valueE2M1.data_ptr()),                                   \
        reinterpret_cast<int32_t*>(scaleFP8SF.data_ptr()),                                  \
        sfUseUE8M0, layout, mMultiProcessorCount, stream)

    if (sfUseUE8M0)
    {
        if (self.scalar_type() == at::ScalarType::Half)
        {
            LAUNCH_FP4_QUANTIZE_KERNEL(half, 32);
        }
        else if (self.scalar_type() == at::ScalarType::BFloat16)
        {
#ifdef ENABLE_BF16
            LAUNCH_FP4_QUANTIZE_KERNEL(__nv_bfloat16, 32);
#else
            TORCH_CHECK(false, "BFloat16 support not enabled at build time.");
#endif
        }
        else if (self.scalar_type() == at::ScalarType::Float8_e4m3fn)
        {
#ifdef ENABLE_FP8
            LAUNCH_FP4_QUANTIZE_KERNEL(__nv_fp8_e4m3, 32);
#else
            TORCH_CHECK(false, "FP8 support not enabled at build time.");
#endif
        }
        else
        {
            TORCH_CHECK(false, "fp4_quantize only supports fp16/bf16/fp8_e4m3 input.");
        }
    }
    else
    {
        if (self.scalar_type() == at::ScalarType::Half)
        {
            LAUNCH_FP4_QUANTIZE_KERNEL(half, 16);
        }
        else if (self.scalar_type() == at::ScalarType::BFloat16)
        {
#ifdef ENABLE_BF16
            LAUNCH_FP4_QUANTIZE_KERNEL(__nv_bfloat16, 16);
#else
            TORCH_CHECK(false, "BFloat16 support not enabled at build time.");
#endif
        }
        else if (self.scalar_type() == at::ScalarType::Float8_e4m3fn)
        {
#ifdef ENABLE_FP8
            LAUNCH_FP4_QUANTIZE_KERNEL(__nv_fp8_e4m3, 16);
#else
            TORCH_CHECK(false, "FP8 support not enabled at build time.");
#endif
        }
        else
        {
            TORCH_CHECK(false, "fp4_quantize only supports fp16/bf16/fp8_e4m3 input.");
        }
    }

#undef LAUNCH_FP4_QUANTIZE_KERNEL

    return {valueE2M1, scaleFP8SF};
}

// ---------------------------------------------------------------------------
// calculate_nvfp4_global_scale
// ---------------------------------------------------------------------------

/// Compute per-token global scale for NVFP4 quantization.
///
/// @param input          Input tensor [token_num, hidden_size] or [batch, token_num, hidden_size]
/// @param tokensPerBatch Optional [batch] int tensor with actual token counts
/// @return globalScale tensor with same shape as input but last dim = 1
at::Tensor calculate_nvfp4_global_scale(
    at::Tensor const& input,
    std::optional<at::Tensor> const& tokensPerBatch)
{
    CHECK_TH_CUDA(input);
    CHECK_CONTIGUOUS(input);

    auto const& inputShape = input.sizes();
    auto const& rank = inputShape.size();

    TORCH_CHECK(rank >= 2 && rank <= 3, "Input must be 2D or 3D tensor.");

    int64_t batch_size = 1;
    int64_t token_num = 1;
    int64_t hidden_size = inputShape[rank - 1];

    if (rank == 2)
    {
        token_num = inputShape[0];
        batch_size = 1;
    }
    else if (rank == 3)
    {
        batch_size = inputShape[0];
        token_num = inputShape[1];
    }

    // Output: same shape as input, last dim = 1
    std::vector<int64_t> outputShape(inputShape.begin(), inputShape.end());
    outputShape[rank - 1] = 1;

    at::Tensor globalScale = at::empty(outputShape, input.options().dtype(torch::kFloat32));

    static int multiProcessorCount = tensorrt_llm::common::getMultiProcessorCount();

    int const* tokensPerBatchPtr = nullptr;
    if (tokensPerBatch.has_value())
    {
        CHECK_TH_CUDA(tokensPerBatch.value());
        CHECK_CONTIGUOUS(tokensPerBatch.value());
        auto const& tokensShape = tokensPerBatch.value().sizes();
        TORCH_CHECK(tokensShape.size() == 1, "tokensPerBatch should have exactly 1 dimension");
        TORCH_CHECK(tokensShape[0] == batch_size, "tokensPerBatch first dimension must match batch size");
        tokensPerBatchPtr = tokensPerBatch.value().data_ptr<int>();
    }

    auto stream = at::cuda::getCurrentCUDAStream(input.get_device()).stream();

    if (input.scalar_type() == at::ScalarType::Half)
    {
        tensorrt_llm::kernels::computePerTokenGlobalScaleForFP4Quantization<half>(
            batch_size, token_num, hidden_size,
            reinterpret_cast<half const*>(input.data_ptr()),
            tokensPerBatchPtr,
            globalScale.data_ptr<float>(),
            multiProcessorCount, stream);
    }
    else if (input.scalar_type() == at::ScalarType::BFloat16)
    {
#ifdef ENABLE_BF16
        tensorrt_llm::kernels::computePerTokenGlobalScaleForFP4Quantization<__nv_bfloat16>(
            batch_size, token_num, hidden_size,
            reinterpret_cast<__nv_bfloat16 const*>(input.data_ptr()),
            tokensPerBatchPtr,
            globalScale.data_ptr<float>(),
            multiProcessorCount, stream);
#else
        TORCH_CHECK(false, "BFloat16 support not enabled at build time.");
#endif
    }
    else
    {
        TORCH_CHECK(false, "calculate_nvfp4_global_scale only supports fp16/bf16 input.");
    }

    return globalScale;
}

// ---------------------------------------------------------------------------
// Op registration
// ---------------------------------------------------------------------------

TORCH_LIBRARY_FRAGMENT(tllm_linear_lite, m)
{
    m.def(
        "fp4_quantize(Tensor input, Tensor? globalScale, int sfVecSize, "
        "bool sfUseUE8M0=False, bool isSfSwizzledLayout=True) -> (Tensor, Tensor)");
    m.def("calculate_nvfp4_global_scale(Tensor input, Tensor? tokensPerBatch) -> Tensor");
}

TORCH_LIBRARY_IMPL(tllm_linear_lite, CUDA, m)
{
    m.impl("fp4_quantize", &fp4_quantize);
    m.impl("calculate_nvfp4_global_scale", &calculate_nvfp4_global_scale);
}
