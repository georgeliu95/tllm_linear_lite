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

The included `fouroversix.patch` modifies
`src/fouroversix/quantize/quantized_tensor.py` with two changes:

1. `from __future__ import annotations` — enables PEP 604 union syntax (`X | Y`)
2. `QuantizedTensor.to()` method — convenience API for dtype conversion (dequantize)
   and device transfer

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

