# fouroversix Example

Demonstrates [fouroversix](https://github.com/mit-han-lab/fouroversix) FP4 (NVFP4)
quantization with end-to-end GEMM verification (cuBLASLt via tllm_linear_lite).

## Files

```
test_fouroversix/
├── fouroversix_example.py   # Quantize + GEMM example script
├── fouroversix.patch        # Local changes to QuantizedTensor (.to() method)
└── README.md                # This file
```

## Prerequisites

- Blackwell GPU (cuBLASLt FP4 GEMM requires SM100+)
- PyTorch with CUDA support
- **tllm_linear_lite** installed (provides cuBLASLt FP4 GEMM backend)

## Setup

This example is tested against fouroversix **v1.0.0** (commit `3d1ef6b`).
Newer versions may have breaking API changes — pin to this version if you
encounter issues.

```bash
# 1. Clone fouroversix at the tested version
git clone --branch v1.0.0 https://github.com/mit-han-lab/fouroversix.git
cd fouroversix

# 2. Apply patch (adds QuantizedTensor.to() method + __future__ annotations)
git apply /path/to/fouroversix.patch

# 3. Install from source
pip install -e .
```

### What the Patch Does

The included `fouroversix.patch` modifies several files:

**`quantized_tensor.py`** (convenience API):
1. `from __future__ import annotations` — enables PEP 604 union syntax (`X | Y`)
2. `QuantizedTensor.to()` method — dtype conversion (dequantize) and device transfer

**Split quantize op** (profiling support — splits the single `quantize_to_fp4` op
into two independently measurable PyTorch ops, with zero kernel change):
3. `fp4_quant.h` — adds `phase` field to `FP4_quant_params` (0=both, 1=prologue, 2=quant)
4. `fp4_quant_launch_template.h` — conditional launch based on `phase`
5. `fp4_quant.cu` — refactors host code, adds `quantize_fp4_prologue` and `quantize_fp4_main`
6. `bindings.cpp` — registers the two new op schemas
7. `cuda/ops.py` — Python wrappers + `register_fake` for split ops
8. `cuda/backend.py` — `CUDAQuantizeBackend.quantize_fp4_prologue/main` classmethods

## Usage

```bash
# Default: input=[4096,4096], weight=[4096,4096], bfloat16, all scale rules
python fouroversix_example.py

# Custom M,N,K shape
python fouroversix_example.py --shape 128,4096,4096   # M=128, N=4096, K=4096

# M,K format (N defaults to K)
python fouroversix_example.py --shape 4096,4096

# Benchmark only (skip accuracy verification)
python fouroversix_example.py --skip-verify

# Specific scale rules
python fouroversix_example.py --scale-rules mse,static_6
```

## What It Does

The example runs three stages:

1. **Basic Usage Demo** — Minimal quantize → dequantize round-trip showing
   `QuantizedTensor` attributes and the `.to()` convenience API.

2. **Accuracy Verification** — Uses tllm_linear_lite's standard NVFP4 as an
   independent baseline for cross-validation. Two parts:
   - **Step 2a**: Quantize input and weight separately, report per-tensor and
     averaged SQNR (dB) + mean abs error. Columns: `tllm` baseline +
     fouroversix scale rules. SQNR gain measured against the tllm baseline.
   - **Step 2b**: Run cuBLASLt FP4 GEMM, compare against BF16 reference GEMM.
     Reports cosine similarity, mean/max abs error, and mean relative error.
     Both tllm and fouroversix use the same cuBLASLt backend for a fair
     comparison — only the quantization differs.

3. **Performance Benchmark** — Measures quantization latency, effective memory
   bandwidth, and MBU% for each scale rule.

## Scale Rules

| Rule | Description |
|------|-------------|
| `mse` | Per-block pick max_e2m1 ∈ {4, 6} to minimize mean squared error (default) |
| `abs_max` | Per-block pick to minimize max absolute error |
| `mae` | Per-block pick to minimize mean absolute error |
| `static_6` | Standard NVFP4: always max_e2m1=6 (same as TRT-LLM) |
| `static_4` | Always max_e2m1=4 |

## Reference Results (B200)

Tested on NVIDIA B200, shape `[4096, 4096]`, BF16, fouroversix v1.0.0.

### Quantization Error

`tllm` = tllm_linear_lite standard NVFP4 (independent baseline). `static_6` should
match `tllm` closely — both are standard NVFP4, different implementations.

```
                                tllm       mse    abs_max       mae  static_6
Avg SQNR (dB)                 20.44     21.21✓     20.98     21.08     20.43
Avg mean abs err            0.07145   0.06859    0.07188   0.06766✓  0.07145

SQNR gain over tllm:
  mse       +0.78 dB
  abs_max   +0.54 dB
  mae       +0.64 dB
  static_6  -0.00 dB          ← validates fouroversix correctness
```

### End-to-End GEMM (cuBLASLt)

```
                                tllm       mse    abs_max       mae  static_6
Cosine similarity            0.9910    0.9925✓    0.9920    0.9922    0.9910
Mean abs error                6.858     6.275✓     6.444     6.373     6.858
```

All configs pass (cosine > 0.99). `mse` gives the best GEMM accuracy.

### Quantize Latency

```
                                 mse    abs_max       mae  static_6
Avg latency (us)              125.2      126.1     123.5      70.3
```

4/6 scale selection adds ~1.8x overhead vs standard NVFP4 (`static_6`).
This is quantize-only latency; GEMM time dominates in practice.

