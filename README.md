# tllm_linear_lite

Standalone modules for the NVFP4 GEMM pipeline, extracted from [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) with **no TensorRT-LLM dependency**. Includes FP4 quantization, NVFP4 GEMM (CUTLASS + cuBLASLt + CUDA core), and Triton-based amax.

## Provided Ops

### Quantize (CUDA extension)

| Op | Signature | Description |
|----|-----------|-------------|
| `torch.ops.tllm_linear_lite.fp4_quantize` | `(input, globalScale?, sfVecSize, sfUseUE8M0, isSfSwizzledLayout) -> (Tensor, Tensor)` | Block-wise FP4 (E2M1) quantization with FP8 E4M3 scale factors |
| `torch.ops.tllm_linear_lite.calculate_nvfp4_global_scale` | `(input, tokensPerBatch?) -> Tensor` | Per-token global scale computation for NVFP4 |

### GEMM (CUDA extension)

| Op | Signature | Description |
|----|-----------|-------------|
| `torch.ops.tllm_linear_lite.cutlass_fp4_gemm` | `(a, b, scale_a, scale_b, alpha, out_dtype?) -> Tensor` | CUTLASS 3.x FP4 GEMM, best perf (K%32==0, N%32==0) |
| `torch.ops.tllm_linear_lite.cublaslt_fp4_gemm` | `(a, b, scale_a, scale_b, alpha, out_dtype?) -> Tensor` | cuBLASLt FP4 BlockScaleGemm, all shapes |
| `torch.ops.tllm_linear_lite.cuda_core_nvfp4_gemm` | `(a, b, scale_a, scale_b, alpha, out_dtype?) -> Tensor` | CUDA core FP4 GEMM, M <= 16 (decode phase) |

### Python API

| Function | Module | Description |
|----------|--------|-------------|
| `nvfp4_gemm(...)` | `tllm_linear_lite.nvfp4_gemm` | Unified dispatch: auto-selects backend (cuda_core M<=16, cutlass K%32==0, else cuBLASLt) |
| `triton_amax(...)` | `tllm_linear_lite.amax` | Global absolute maximum with autotuning |

## Project Structure

```
tllm_linear_lite/
├── setup.py                        # Build script (torch CUDAExtension)
├── 3rdparty/
│   └── cutlass/                    # CUTLASS v4.3.0 (git submodule, header-only)
├── tllm_linear_lite/               # Python package
│   ├── __init__.py                 # Loads .so, registers ops
│   ├── nvfp4_gemm.py              # Unified GEMM dispatch (auto/cutlass/cublaslt/cuda_core)
│   └── amax/
│       └── triton_amax.py          # Triton-based amax kernel (autotuned)
├── quantize/                       # FP4 quantization CUDA sources
│   ├── tllm_compat.cuh             # Standalone compat header (replaces TRT-LLM common/)
│   ├── quantization.{h,cuh,cu}     # FP4 quantization kernels
│   └── fp4_quantize_op.cu          # PyTorch op registration
├── gemm/                           # NVFP4 GEMM CUDA sources
│   ├── cublaslt_fp4_gemm.cpp       # cuBLASLt BlockScaleGemm backend
│   ├── cuda_core_nvfp4_gemm.cu     # CUDA core backend (M<=16, uses CUTLASS + CUB)
│   ├── cutlass_fp4_gemm_op.cu      # CUTLASS backend PyTorch op wrapper
│   └── cutlass_fp4/                # CUTLASS 3.x FP4 GEMM templates (SM100/103/120)
│       ├── trtllm_cutlass_compat.h # Standalone compat for CUTLASS kernel headers
│       ├── fp4_gemm.h              # CutlassFp4GemmRunner interface
│       ├── fp4_gemm_template.h     # GEMM dispatch logic (NVFP4 only, MXFP8 removed)
│       ├── nvfp4_nvfp4_gemm_template_sm100.h  # SM100/SM103 kernel templates
│       ├── nvfp4_nvfp4_gemm_template_sm120.h  # SM120 kernel templates
│       ├── fp4_gemm_fp16.cu        # Template instantiations (half, 51 kernels)
│       └── fp4_gemm_bf16.cu        # Template instantiations (bf16, 51 kernels)
└── tests/
    ├── test_quantize.py            # Quantize correctness + perf
    └── test_nvfp4_gemm.py          # GEMM correctness (vs nn.Linear) + perf
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.4 (with CUDA support)
- CUDA Toolkit 12.x+
- GPU: Blackwell (sm_100a) required for FP4 E2M1 PTX instructions

## Installation

```bash
cd tllm_linear_lite/

# Init CUTLASS submodule (header-only, needed by cuda_core + CUTLASS backends)
git submodule update --init --depth 1

# Build for Blackwell (B200/B100)
TORCH_CUDA_ARCH_LIST="10.0a" pip install -e . --no-build-isolation

# Or with uv
TORCH_CUDA_ARCH_LIST="10.0a" uv pip install -e . --system --break-system-packages --no-build-isolation
```

> **Note**: `sm_100a` (not `sm_100`) is required. The `a` suffix enables architecture-specific
> instructions like `cvt.rn.satfinite.e2m1x2.f32` and `__nv_cvt_float_to_e8m0`.

## Usage

### FP4 Quantization

```python
import torch
import tllm_linear_lite  # loads CUDA extension, registers ops

x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)

# Compute global scale
global_scale = 448.0 * 6.0 / x.abs().amax().float()

# FP4 quantize (NVFP4: sfVecSize=16, sfUseUE8M0=False)
fp4_packed, scale_factors = torch.ops.tllm_linear_lite.fp4_quantize(
    x, global_scale, 16, False, True
)
```

### NVFP4 GEMM

```python
from tllm_linear_lite.nvfp4_gemm import nvfp4_gemm

# Auto-dispatch: cuda_core for M<=16, cutlass for K%32==0, else cuBLASLt
output = nvfp4_gemm(
    act_fp4, weight_fp4, act_sf, weight_sf, alpha,
    output_dtype=torch.bfloat16,
    backend="auto",  # or "cutlass", "cublaslt", "cuda_core"
)

# Add bias (epilogue fusion planned as follow-up)
output = output + bias
```

### Triton Amax

```python
from tllm_linear_lite.amax.triton_amax import triton_amax

x = torch.randn(14400, 6144, device="cuda", dtype=torch.bfloat16)
amax = triton_amax(x)
```

## Testing

```bash
# FP4 quantize: correctness (vs PyTorch reference) + performance
python tests/test_quantize.py
python tests/test_quantize.py --shape 4096,4096 --dtype float16

# NVFP4 GEMM: correctness (vs nn.Linear) + performance, with/without bias
python tests/test_nvfp4_gemm.py
python tests/test_nvfp4_gemm.py --m 1 --n 4096 --k 4096 --backend cuda_core
python tests/test_nvfp4_gemm.py --m 128 --n 4096 --k 4096 --backend cutlass
python tests/test_nvfp4_gemm.py --m 128 --n 4096 --k 4096 --backend cublaslt

# Cross-validate with TensorRT-LLM (requires tensorrt_llm installed)
python tests/test_quantize.py --compare-trtllm
python tests/test_nvfp4_gemm.py --compare-trtllm

# Triton amax
python tllm_linear_lite/amax/triton_amax.py

# NCU profiling
ncu --set full -o quantize_report python tests/test_quantize.py --ncu
```

## What Changed from TensorRT-LLM

### Quantize kernels (`quantize/`)

Derived from `tensorrt_llm/kernels/quantization.*`:
- **Replaced** 8 TRT-LLM internal headers with a single `tllm_compat.cuh`
- **Removed** non-FP4 code (INT8, per-token INT8/FP8, MXFP8)
- **Kept** FP4 block quantization, block scale interleave, per-token global scale
- See comment blocks at top of each `.h` / `.cuh` / `.cu` file for detailed diff

### GEMM kernels (`gemm/`)

- **CUTLASS** (`gemm/cutlass_fp4/`): copied from `kernels/cutlass_kernels/fp4_gemm/`, replaced all TRT-LLM headers with `trtllm_cutlass_compat.h`, removed MXFP8 dispatch/instantiations, created minimal `cutlass_type_conversion.h` (no NvInferRuntime.h). Covers SM100, SM103, SM120 with 51 kernel instantiations per dtype.
- **cuBLASLt** (`cublaslt_fp4_gemm.cpp`): self-contained rewrite of `cublasFp4ScaledMM.cpp` + `cublasMMWrapper.cpp`, with inline handle management (no `opUtils.h` dependency)
- **CUDA core** (`cuda_core_nvfp4_gemm.cu`): extracted from `cudaCoreGemmNVFP4.cu` + `cudaNvfp4MM.cpp`, replaced `NvInferRuntime.h`, `SizeType32`, `TllmToCutlassTypeAdapter` with local equivalents

See comment blocks at top of each file for detailed diff from TRT-LLM originals.

## Roadmap

- [x] `fp4_quantize` -- NVFP4 block quantization (CUDA kernel)
- [x] `calculate_nvfp4_global_scale` -- Per-token global scale (CUDA kernel)
- [x] `triton_amax` -- Global amax reduction (Triton kernel)
- [x] `cublaslt_fp4_gemm` -- cuBLASLt NVFP4 GEMM (all shapes)
- [x] `cuda_core_nvfp4_gemm` -- CUDA core NVFP4 GEMM (M<=16, decode)
- [x] `nvfp4_gemm` -- Unified Python dispatch layer
- [x] `cutlass_fp4_gemm` -- CUTLASS 3.x NVFP4 GEMM (SM100/103/120, best perf)
- [ ] Cute DSL NVFP4 GEMM -- Python-based Blackwell kernels
- [ ] Bias epilogue fusion -- Fuse bias into cuBLASLt GEMM epilogue
- [ ] FourOverSix -- Adaptive block scaling in quantize module
- [ ] End-to-end NVFP4 Linear -- Fused quantize + GEMM `nn.Module`

## License

CUDA kernel code is derived from TensorRT-LLM, licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
