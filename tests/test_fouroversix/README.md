# fouroversix Standalone Example

Standalone example demonstrating [fouroversix](https://github.com/mit-han-lab/fouroversix)
FP4 (NVFP4) quantization, independent of tllm_linear_lite.

## Files

```
test_fouroversix/
├── fouroversix_example.py   # Standalone example script
├── fouroversix.patch        # Local changes to QuantizedTensor (.to() method)
└── README.md                # This file
```

## Prerequisites

- CUDA GPU (Blackwell recommended; PyTorch backend works on any GPU)
- PyTorch with CUDA support

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
# Default: 4096x4096, bfloat16, all scale rules
python fouroversix_example.py

# Custom shape and dtype
python fouroversix_example.py --shape 14400,6144 --dtype bfloat16

# Benchmark only (skip accuracy verification)
python fouroversix_example.py --skip-verify

# Specific scale rules
python fouroversix_example.py --scale-rules mse,static_6
```

## What It Does

The example runs three stages:

1. **Basic Usage Demo** — Minimal quantize → dequantize round-trip showing
   `QuantizedTensor` attributes and the `.to()` convenience API.

2. **Accuracy Verification** — Compares quantization error across scale rules
   (mse, abs_max, mae, static_6). Reports max/mean absolute and relative error,
   plus improvement over standard NVFP4 (`static_6`).

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

