# tllm_linear_lite

Standalone modules for the NVFP4 GEMM pipeline, extracted from [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) with **no TensorRT-LLM dependency**. Includes FP4 quantization CUDA kernels, Triton-based amax, and (TODO) NVFP4 GEMM.

## Provided Ops

### Quantize (CUDA extension)

| Op | Signature | Description |
|----|-----------|-------------|
| `torch.ops.tllm_linear_lite.fp4_quantize` | `(input, globalScale?, sfVecSize, sfUseUE8M0, isSfSwizzledLayout) -> (Tensor, Tensor)` | Block-wise FP4 (E2M1) quantization with FP8 E4M3 scale factors |
| `torch.ops.tllm_linear_lite.calculate_nvfp4_global_scale` | `(input, tokensPerBatch?) -> Tensor` | Per-token global scale computation for NVFP4 |

### Amax (Triton kernel)

| Function | Signature | Description |
|----------|-----------|-------------|
| `triton_amax` | `(input, config?) -> Tensor` | Global absolute maximum with autotuning (two-stage reduction) |
| `triton_amax_partial` | `(input, config?) -> Tensor` | Block-level partial amax (first stage only, for fused pipelines) |

## Project Structure

```
tllm_linear_lite/
├── setup.py                        # Build script (torch CUDAExtension)
├── tllm_linear_lite/
│   ├── __init__.py                 # Loads the .so and registers ops
│   └── amax/
│       └── triton_amax.py          # Triton-based amax kernel (autotuned)
├── quantize/
│   ├── tllm_compat.cuh             # Standalone compat header (replaces all TRT-LLM common/)
│   ├── quantization.h              # Kernel declarations (FP4-only)
│   ├── quantization.cuh            # CUDA kernel implementations
│   ├── quantization.cu             # Host wrappers + template instantiations
│   └── fp4_quantize_op.cu          # PyTorch op registration (TORCH_LIBRARY)
└── tests/
    └── test_quantize.py            # Correctness + performance tests
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.4 (with CUDA support)
- CUDA Toolkit 12.x+
- GPU: Blackwell (sm_100a) required for FP4 E2M1 PTX instructions

## Installation

```bash
cd tllm_linear_lite/

# Build for Blackwell (B200)
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

# Per-token global scale
per_token_scale = torch.ops.tllm_linear_lite.calculate_nvfp4_global_scale(x, None)
```

### Triton Amax

```python
from tllm_linear_lite.amax.triton_amax import triton_amax

x = torch.randn(14400, 6144, device="cuda", dtype=torch.bfloat16)

# Autotuned global amax
amax = triton_amax(x)

# With specific kernel config
amax = triton_amax(x, config="B8192_W32")
```

## Testing

```bash
# FP4 quantize: correctness + performance
python tests/test_quantize.py
python tests/test_quantize.py --shape 4096,4096 --dtype float16

# FP4 quantize: compare with TensorRT-LLM (requires tensorrt_llm installed)
python tests/test_quantize.py --compare-trtllm

# FP4 quantize: NCU profiling
ncu --set full -o quantize_report python tests/test_quantize.py --ncu

# Triton amax: correctness + benchmark
python tllm_linear_lite/amax/triton_amax.py
python tllm_linear_lite/amax/triton_amax.py --shape 14400,6144 --dtype bfloat16
```

## What Changed from TensorRT-LLM

The CUDA kernels (`quantization.{h,cuh,cu}`) are derived from TensorRT-LLM's `tensorrt_llm/kernels/quantization.*`. Changes:

- **Replaced** 8 TRT-LLM internal headers with a single `tllm_compat.cuh` standalone compat header
- **Removed** non-FP4 code: INT8 scalar quantization, per-token INT8/FP8 quantization, MXFP8 quantization
- **Kept** FP4 block quantization, block scale interleave, and per-token global scale for FP4
- **Added** `fp4_quantize_op.cu` for PyTorch op registration under the `tllm_linear_lite` namespace

See the comment block at the top of each `.h` / `.cuh` / `.cu` file for the detailed diff.

## Roadmap

- [x] `fp4_quantize` -- NVFP4 block quantization (CUDA kernel)
- [x] `calculate_nvfp4_global_scale` -- Per-token global scale (CUDA kernel)
- [x] `triton_amax` -- Global amax reduction (Triton kernel)
- [ ] NVFP4 GEMM -- FP4 matrix multiplication (CUTLASS / cuBLASLt)
- [ ] FourOverSix -- Support FourOverSix in quantize module
- [ ] End-to-end NVFP4 Linear -- Fused quantize + GEMM module

## License

CUDA kernel code is derived from TensorRT-LLM, licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
