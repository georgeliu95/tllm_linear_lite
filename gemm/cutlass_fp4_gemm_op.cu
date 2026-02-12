/*
 * PyTorch Op wrapper for CUTLASS NVFP4 GEMM.
 *
 * Provides:
 *   torch.ops.tllm_linear_lite.cutlass_fp4_gemm(a, b, scale_a, scale_b, alpha, out_dtype?) -> Tensor
 *
 * Uses CutlassFp4GemmRunner<T, W4A4_NVFP4_NVFP4> from the extracted CUTLASS templates.
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass_fp4/trtllm_cutlass_compat.h"
#include "cutlass_fp4/gemm_configs.h"
#include "cutlass_fp4/fp4_gemm.h"

namespace tkc = tensorrt_llm::cutlass_extensions;
using tensorrt_llm::kernels::cutlass_kernels::CutlassFp4GemmRunner;
using tensorrt_llm::kernels::cutlass_kernels::FP4GemmType;

namespace
{

// Default GEMM config heuristic (from TRT-LLM fp4Gemm.cpp)
tkc::CutlassGemmConfig getDefaultGemmConfig(int sm)
{
    if (sm >= 120)
    {
        return tkc::CutlassGemmConfig(tkc::CutlassTileConfigSM120::CtaShape128x128x256B,
            tkc::MainloopScheduleType::AUTO, tkc::EpilogueScheduleType::AUTO,
            tkc::ClusterShape::ClusterShape_1x1x1);
    }
    else if (sm == 103)
    {
        return tkc::CutlassGemmConfig(tkc::CutlassTileConfigSM100::CtaShape128x256x256B,
            tkc::MainloopScheduleType::AUTO, tkc::EpilogueScheduleType::AUTO,
            tkc::ClusterShape::ClusterShape_1x1x1);
    }
    else
    {
        return tkc::CutlassGemmConfig(tkc::CutlassTileConfigSM100::CtaShape128x256x128B,
            tkc::MainloopScheduleType::AUTO, tkc::EpilogueScheduleType::AUTO,
            tkc::ClusterShape::ClusterShape_1x1x1);
    }
}

template <typename T>
void runGemm(at::Tensor& out, at::Tensor const& mat1, at::Tensor const& mat2,
    at::Tensor const& mat1Scale, at::Tensor const& mat2Scale, at::Tensor const& globalScale,
    int64_t m, int64_t n, int64_t k, tkc::CutlassGemmConfig const& gemmConfig)
{
    CutlassFp4GemmRunner<T, FP4GemmType::W4A4_NVFP4_NVFP4> gemmRunner;
    int64_t const wsBytes = gemmRunner.getWorkspaceSize(m, n, k, 1);

    at::Tensor workspace = at::empty({wsBytes}, mat1.options().dtype(at::kByte));

    gemmRunner.gemm(out.data_ptr(), mat1.const_data_ptr(), mat2.const_data_ptr(),
        mat1Scale.const_data_ptr(), mat2Scale.const_data_ptr(),
        globalScale.data_ptr<float>(),
        m, n, k, /*batch_count=*/1, gemmConfig,
        reinterpret_cast<char*>(workspace.data_ptr()), wsBytes,
        at::cuda::getCurrentCUDAStream(mat1.get_device()));
}

} // anonymous namespace

at::Tensor cutlass_fp4_gemm(
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

    int64_t m = a.sizes()[0];
    int64_t n = b.sizes()[0];
    int64_t k = a.sizes()[1] * 2; // FP4: 2 values per byte

    TORCH_CHECK(k % 32 == 0, "K must be divisible by 32 for CUTLASS FP4 GEMM");
    TORCH_CHECK(n % 32 == 0, "N must be divisible by 32 for CUTLASS FP4 GEMM");

    at::ScalarType dtype = out_dtype.value_or(at::ScalarType::BFloat16);
    at::Tensor out = at::empty({m, n}, a.options().dtype(dtype));

    int sm = tensorrt_llm::common::getSMVersion();
    auto gemmConfig = getDefaultGemmConfig(sm);

    if (dtype == at::ScalarType::Half)
    {
        runGemm<half>(out, a, b, scale_a, scale_b, alpha, m, n, k, gemmConfig);
    }
    else if (dtype == at::ScalarType::BFloat16)
    {
#ifdef ENABLE_BF16
        runGemm<__nv_bfloat16>(out, a, b, scale_a, scale_b, alpha, m, n, k, gemmConfig);
#else
        TORCH_CHECK(false, "BFloat16 not enabled at build time");
#endif
    }
    else if (dtype == at::ScalarType::Float)
    {
        TORCH_CHECK(false, "FP32 output not supported in this build (fp4_gemm_fp32.cu not included)");
    }
    else
    {
        TORCH_CHECK(false, "Unsupported output dtype for CUTLASS FP4 GEMM");
    }

    return out;
}

TORCH_LIBRARY_FRAGMENT(tllm_linear_lite, m)
{
    m.def(
        "cutlass_fp4_gemm(Tensor a, Tensor b, Tensor scale_a, Tensor scale_b, "
        "Tensor alpha, ScalarType? out_dtype=None) -> Tensor");
}

TORCH_LIBRARY_IMPL(tllm_linear_lite, CUDA, m)
{
    m.impl("cutlass_fp4_gemm", &cutlass_fp4_gemm);
}
