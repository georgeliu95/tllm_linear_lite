# tllm_linear_lite

Standalone modules for the NVFP4 GEMM pipeline, extracted from [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) with **no TensorRT-LLM dependency**. Includes FP4 quantization, NVFP4 GEMM (Cute DSL + CUTLASS + cuBLASLt + CUDA core), and Triton-based amax.

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
| `torch.ops.tllm_linear_lite.cublaslt_fp4_gemm` | `(a, b, scale_a, scale_b, alpha, bias?, out_dtype?) -> Tensor` | cuBLASLt FP4 BlockScaleGemm, all shapes, supports fused bias epilogue |
| `torch.ops.tllm_linear_lite.cuda_core_nvfp4_gemm` | `(a, b, scale_a, scale_b, alpha, out_dtype?) -> Tensor` | CUDA core FP4 GEMM, M <= 16 (decode phase) |

### Python API

| Function | Module | Description |
|----------|--------|-------------|
| `nvfp4_gemm(...)` | `tllm_linear_lite.nvfp4_gemm` | Unified dispatch: `cuda_core` if M<=16 and N%2==0 and K%16==0; else `cutedsl` (Blackwell SM100/103 + bf16 + K%32==0 + cutlass-dsl); else `cutlass` if K%32==0 and N%32==0; else `cublaslt` |
| `fp4_quantize(...)` | `tllm_linear_lite.quantize` | Unified quantize wrapper (`backend="tllm"` or optional `backend="fouroversix"`) |
| `calculate_nvfp4_global_scale(...)` | `tllm_linear_lite.quantize` | Per-token / global NVFP4 scale (Python wrapper over C++ op) |
| `CuteDSLNVFP4Runner` | `tllm_linear_lite.cutedsl.runner` | Cute DSL JIT runner with SimpleTuner (SM100/103 only, bf16) |
| `triton_amax(...)` | `tllm_linear_lite.amax` | Global absolute maximum with autotuning |
| `triton_amax_partial(...)` | `tllm_linear_lite.amax` | Block-level partial amax (no final reduction) |
| `NVFP4DynamicLinear` | `tllm_linear_lite.nvfp4_linear` | End-to-end W4A4 dynamic-quantized `nn.Linear` replacement (tllm or fouroversix quant, auto GEMM dispatch) |

## Project Structure

```
tllm_linear_lite/
├── setup.py                        # Build script (torch CUDAExtension)
├── 3rdparty/
│   └── cutlass/                    # CUTLASS v4.3.0 (git submodule, header-only)
├── tllm_linear_lite/               # Python package
│   ├── __init__.py                 # Loads .so, registers ops
│   ├── nvfp4_gemm.py              # Unified GEMM dispatch (auto/cutedsl/cutlass/cublaslt/cuda_core)
│   ├── nvfp4_linear.py            # NVFP4DynamicLinear nn.Module (fused quantize + GEMM)
│   ├── quantize/                   # FP4 quantize Python API
│   │   └── __init__.py             # fp4_quantize(), calculate_nvfp4_global_scale(), backend dispatch
│   ├── cutedsl/                    # Cute DSL backend (Python-only, optional)
│   │   ├── __init__.py             # IS_CUTLASS_DSL_AVAILABLE check
│   │   ├── runner.py               # CuteDSLNVFP4Runner (simplified from TRT-LLM)
│   │   ├── tuner.py                # SimpleTuner (minimal tactic profiler)
│   │   └── kernels/                # Copied from TRT-LLM (no modifications)
│   │       ├── __init__.py
│   │       ├── dense_blockscaled_gemm_persistent.py  # Blackwell GEMM kernel (~2000 lines)
│   │       ├── custom_pipeline.py  # PipelineTmaUmma, PipelineUmmaAsync
│   │       └── utils.py            # make_ptr, griddepcontrol helpers
│   └── amax/
│       ├── __init__.py
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
│       ├── cutlass_type_conversion.h # Minimal dtype conversion (no NvInferRuntime.h)
│       ├── archCondition.h         # SM architecture condition macros
│       ├── gemm_configs.h          # GEMM tile/stage configuration definitions
│       ├── fp4_gemm.h              # CutlassFp4GemmRunner interface
│       ├── fp4_gemm_template.h     # GEMM dispatch logic (NVFP4 only, MXFP8 removed)
│       ├── nvfp4_nvfp4_gemm_template_sm100.h  # SM100/SM103 kernel templates
│       ├── nvfp4_nvfp4_gemm_template_sm120.h  # SM120 kernel templates
│       ├── fp4_gemm_fp16.cu        # Template instantiations (half, 51 kernels)
│       └── fp4_gemm_bf16.cu        # Template instantiations (bf16, 51 kernels)
└── tests/
    ├── test_quantize.py            # Quantize correctness + perf
    ├── test_nvfp4_gemm.py          # GEMM correctness (vs nn.Linear) + perf
    ├── test_nvfp4_linear.py        # NVFP4DynamicLinear end-to-end correctness
    └── test_fouroversix/           # fouroversix integration example
        ├── fouroversix_example.py  # Quantize + GEMM with fouroversix
        ├── fouroversix.patch       # Patch for QuantizedTensor.to() method
        └── README.md               # Setup instructions + reference results
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

# Optional: install Cute DSL backend dependencies (Blackwell SM100/SM103 only)
pip install nvidia-cutlass-dsl>=4.3.4 cuda-core apache-tvm-ffi==0.1.6
# Or: pip install -e ".[cutedsl]"
```

> **Note**: `sm_100a` (not `sm_100`) is required. The `a` suffix enables architecture-specific
> instructions like `cvt.rn.satfinite.e2m1x2.f32` and `__nv_cvt_float_to_e8m0`.

## Usage

### FP4 Quantization

```python
import torch
import tllm_linear_lite  # loads CUDA extension, registers ops
from tllm_linear_lite.quantize import fp4_quantize

x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)

# High-level API (auto-computes global_scale if not provided)
fp4_packed, scale_factors = fp4_quantize(x)

# With fouroversix backend (optional dependency, adaptive block scaling)
# qt = fp4_quantize(x, backend="fouroversix")  # returns QuantizedTensor

# Low-level torch.ops API (explicit parameters)
global_scale = 448.0 * 6.0 / x.abs().amax().float()
fp4_packed, scale_factors = torch.ops.tllm_linear_lite.fp4_quantize(
    x, global_scale, 16, False, True  # sfVecSize=16, sfUseUE8M0=False, swizzled=True
)
```

### NVFP4 GEMM

```python
from tllm_linear_lite.nvfp4_gemm import nvfp4_gemm

# Auto-dispatch:
# - cuda_core: M<=16 and N%2==0 and K%16==0
# - cutedsl: Blackwell (SM100/103), bf16 output, K%32==0, and cutlass-dsl installed
# - cutlass: K%32==0 and N%32==0
# - fallback: cublaslt
output = nvfp4_gemm(
    act_fp4, weight_fp4, act_sf, weight_sf, alpha,
    bias=bias,  # fused in cublaslt; post-add in other backends
    output_dtype=torch.bfloat16,
    backend="auto",  # or "cutedsl", "cutlass", "cublaslt", "cuda_core"
)
```

### NVFP4DynamicLinear (nn.Module)

```python
import torch
import torch.nn as nn
from tllm_linear_lite.nvfp4_linear import NVFP4DynamicLinear

# Convert a pretrained nn.Linear to NVFP4 (weights quantized, activations dynamic)
linear = nn.Linear(4096, 4096, bias=True, device="cuda", dtype=torch.bfloat16)
nvfp4 = NVFP4DynamicLinear.from_linear(linear)

# Supports N-D input just like nn.Linear
x = torch.randn(2, 128, 4096, device="cuda", dtype=torch.bfloat16)
output = nvfp4(x)  # [2, 128, 4096]

# With fouroversix quantization backend (optional dependency)
# from fouroversix import QuantizationConfig
# nvfp4_fox = NVFP4DynamicLinear.from_linear(
#     linear, quant_backend="fouroversix",
#     quant_config=QuantizationConfig(scale_rule="mse"),
# )

# Explicit GEMM backend selection
nvfp4_cublaslt = NVFP4DynamicLinear.from_linear(linear, gemm_backend="cublaslt")
```

> **Known limitations**:
> - K must be a multiple of 32 for cutlass/cutedsl backends (auto-fallback to cublaslt otherwise)
> - Only SWIZZLED scale factor layout (cuda_core backend not used by the module)
> - Bias epilogue fusion only on cublaslt backend; other backends use post-GEMM addition

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

# NVFP4 GEMM: correctness (vs nn.Linear), with/without bias
python tests/test_nvfp4_gemm.py
python tests/test_nvfp4_gemm.py --backend cuda_core
python tests/test_nvfp4_gemm.py --backend cutlass
python tests/test_nvfp4_gemm.py --backend cutedsl  # requires nvidia-cutlass-dsl
python tests/test_nvfp4_gemm.py --backend cublaslt
python tests/test_nvfp4_gemm.py --backend auto --no-bias

# NVFP4DynamicLinear: end-to-end correctness (vs nn.Linear)
python tests/test_nvfp4_linear.py
python tests/test_nvfp4_linear.py --gemm-backend cublaslt
python tests/test_nvfp4_linear.py --quant-backend all   # includes fouroversix (if installed)
python tests/test_nvfp4_linear.py --no-bias

# Cross-validate with TensorRT-LLM (requires tensorrt_llm installed)
python tests/test_quantize.py --compare-trtllm
python tests/test_nvfp4_gemm.py --compare-trtllm  # placeholder, currently prints SKIP

# fouroversix integration example (requires fouroversix, see tests/test_fouroversix/README.md)
python tests/test_fouroversix/fouroversix_example.py
python tests/test_fouroversix/fouroversix_example.py --shape 128,4096,4096 --scale-rules mse,static_6

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

#### Kernel versions

| Version | Kernel | Elements/thread | Load width | Status |
|---------|--------|-----------------|------------|--------|
| v0 | `quantize_with_block_size` | 8 | 128-bit (LDG.E.128×1) | ✅ ported |
| v1 | `opt_quantize_with_block_size_v1` | 16 | 256-bit (LDG.E.128×2) | ✅ ported |

**v1 优化说明**：v0 每 thread 处理 8 个元素（16B load），v1 将其翻倍至 16 个元素（32B load，通过两次连续 LDG.E.128 实现）。更多的 compute-per-load 使 scheduler 有更多独立指令可调度，从而隐藏 L1TEX scoreboard stall（约 7 cycles，占 CPI 的 40%）。v1 引入的新组件：
- `CVT_OPT_ELTS_PER_THREAD = 16`（常量）
- `PackedVec_Opt<T>`（32B 对齐的 32 字节向量类型）
- `load_256bit()`（两次 128-bit inline asm load，保证生成 LDG.E.128）
- `cvt_warp_fp16_to_fp4_impl_opt()`（16 元素内层实现，输出 `uint64_t`）
- `opt_quantize_with_block_size_v1`（只支持 FP16/BF16→FP4，`__launch_bounds__(512, 4)`）
- dispatch 侧新增 `kernelVersion` 参数（0=v0, 1=v1）

> v1 只支持 FP16/BF16→FP4，FP8→FP4 和 FP16→MXFP8 仍使用 v0。

### GEMM kernels (`gemm/`)

- **CUTLASS** (`gemm/cutlass_fp4/`): copied from `kernels/cutlass_kernels/fp4_gemm/`, replaced all TRT-LLM headers with `trtllm_cutlass_compat.h`, removed MXFP8 dispatch/instantiations, created minimal `cutlass_type_conversion.h` (no NvInferRuntime.h). Covers SM100, SM103, SM120 with 51 kernel instantiations per dtype.
- **cuBLASLt** (`cublaslt_fp4_gemm.cpp`): self-contained rewrite of `cublasFp4ScaledMM.cpp` + `cublasMMWrapper.cpp`, with inline handle management (no `opUtils.h` dependency)
- **CUDA core** (`cuda_core_nvfp4_gemm.cu`): extracted from `cudaCoreGemmNVFP4.cu` + `cudaNvfp4MM.cpp`, replaced `NvInferRuntime.h`, `SizeType32`, `TllmToCutlassTypeAdapter` with local equivalents

### Cute DSL kernels (`tllm_linear_lite/cutedsl/`)

- **Kernel files** (`kernels/`): copied as-is from `_torch/cute_dsl_kernels/blackwell/` -- no TRT-LLM imports (only cutlass DSL + cuda-python)
- **Runner** (`runner.py`): simplified from `CuteDSLNVFP4BlackwellLinear`, removed TRT-LLM `TunableRunner` ABC and `AutoTuner` dependency
- **Tuner** (`tuner.py`): ~100-line `SimpleTuner` replaces TRT-LLM's 700-line `AutoTuner`

See comment blocks at top of each file for detailed diff from TRT-LLM originals.

## Roadmap

- [x] `fp4_quantize` -- NVFP4 block quantization (CUDA kernel)
- [x] `calculate_nvfp4_global_scale` -- Per-token global scale (CUDA kernel)
- [x] `triton_amax` -- Global amax reduction (Triton kernel)
- [x] `cublaslt_fp4_gemm` -- cuBLASLt NVFP4 GEMM (all shapes)
- [x] `cuda_core_nvfp4_gemm` -- CUDA core NVFP4 GEMM (M<=16, decode)
- [x] `nvfp4_gemm` -- Unified Python dispatch layer
- [x] `cutlass_fp4_gemm` -- CUTLASS 3.x NVFP4 GEMM (SM100/103/120, best perf)
- [x] Cute DSL NVFP4 GEMM -- Python-based Blackwell kernels (SM100/103, JIT compiled)
- [x] cuBLASLt bias epilogue -- `nvfp4_gemm(..., bias=...)` uses fused epilogue on cublaslt backend
- [x] Optional fouroversix wrapper -- `tllm_linear_lite.quantize.fp4_quantize(..., backend="fouroversix")`
- [x] **Quantize kernel v1** -- 16 elements/thread, 32B vectorized load (LDG.E.128×2), 更多 ILP 以隐藏 L1TEX scoreboard stall；`invokeFP4Quantization` 新增 `kernelVersion` 参数（0=v0, 1=v1，默认 v1）；仅 FP16/BF16→FP4 路径
- [ ] Bias epilogue fusion for non-cuBLASLt backends (cutlass/cuda_core/cutedsl)
- [ ] FourOverSix adaptive block scaling in native CUDA quantize path
- [x] End-to-end NVFP4 Linear -- `NVFP4DynamicLinear` fused quantize + GEMM `nn.Module` (tllm + fouroversix quant backends)

## License

CUDA kernel code is derived from TensorRT-LLM, licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
