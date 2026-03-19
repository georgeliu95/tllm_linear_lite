# Tests

Correctness, performance benchmarks, and cross-validation for tllm_linear_lite.

## Test Scripts

```
tests/
├── test_quantize.py        # FP4 quantize: correctness + performance + adaptive 4/6
├── test_nvfp4_gemm.py      # NVFP4 GEMM: multi-backend correctness (vs nn.Linear)
├── test_nvfp4_linear.py    # NVFP4DynamicLinear: end-to-end correctness
└── test_fouroversix/       # fouroversix integration example (has its own README)
```

## Prerequisites

- Blackwell GPU (SM100+)
- `tllm_linear_lite` installed (`pip install -e .`)
- Optional: `fouroversix` (for cross-validation, see `test_fouroversix/README.md`)
- Optional: `tensorrt_llm` (for `--compare-trtllm`)

---

## test_quantize.py

FP4 (E2M1) block quantization test — correctness against a pure-PyTorch reference, kernel performance, and optional cross-validation with TensorRT-LLM / fouroversix.

### CLI

| Flag | Default | Description |
|------|---------|-------------|
| `--shape` | `14400,6144` | Input shape (comma-separated) |
| `--dtype` | `bfloat16` | `float16` or `bfloat16` |
| `--warmup` | `5` | Warmup iterations |
| `--repeat` | `20` | Benchmark iterations |
| `--skip-verify` | off | Skip correctness, benchmark only |
| `--compare-trtllm` | off | Bit-exact comparison with TensorRT-LLM |
| `--scale-rule` | — | Standalone adaptive 4/6 benchmark (`mse`, `mae`, `abs_max`) |
| `--ncu` | off | NCU profiling mode (single iteration per op) |

### Examples

```bash
# Correctness + performance (default shape 14400×6144, bf16)
python tests/test_quantize.py

# Custom shape, float16
python tests/test_quantize.py --shape 4096,4096 --dtype float16

# Standalone adaptive 4/6 benchmark for a single rule
python tests/test_quantize.py --scale-rule mse

# NCU profiling (wraps kernels in NVTX ranges)
ncu --set full -o quantize_report python tests/test_quantize.py --ncu
```

### Accuracy Output

Correctness runs first: bit-exact comparison against a pure-PyTorch NVFP4 reference, then dequant error for all backends.

Sample output (B200, `(14400, 6144)`, bf16):

```
  Bit-exact check (tllm vs PyTorch ref):
    Scale factors: [PASS] exact match
    FP4 values:    [PASS] 189980/88473600 differ (0.2147%)

  Metric                      PyTorch Ref          tllm     tllm(mse)     tllm(mae) tllm(abs_max)      f46(mse)  f46(abs_max)      f46(mae)
  -----------------------------------------------------------------------------------------------------------------------------------------
  Max abs error                  0.709821      0.709821      0.640625      0.703125      0.507813      0.640625      0.515625      0.703125
  Mean abs error                 0.071473      0.071473      0.068572      0.067645      0.072024      0.068570      0.071830      0.067648
  Mean relative error              0.1786        0.1786        0.1904        0.1833        0.2022        0.1904        0.2012        0.1833

  Error ratio vs PyTorch Ref (mean abs error):
    tllm                 1.0000  [PASS]
    tllm(mse)            0.9594  [PASS]
    tllm(mae)            0.9464  [PASS]
    tllm(abs_max)        1.0077  [PASS]
    f46(mse)             0.9594  [PASS]
    f46(abs_max)         1.0050  [PASS]
    f46(mae)             0.9465  [PASS]
```

- **Bit-exact check**: tllm CUDA kernel vs pure-PyTorch reference (minor diffs from FP rounding)
- **tllm** = standard NVFP4 (r=6 fixed); **tllm(mse/mae/abs_max)** = adaptive 4/6 (native CUDA)
- **f46(...)** = fouroversix (should match tllm adaptive numerically — error ratios ~1.0)
- **Error ratio < 1** means lower error than the PyTorch reference (adaptive 4/6 improves quality)

### Benchmark Output

Default mode runs correctness then prints a comparison table covering:
- **tllm** — standard NVFP4 kernel (v1, 16 elems/thread)
- **tllm(mse/abs_max/mae)** — adaptive 4/6 kernel (native CUDA, streaming)
- **f46(mse/abs_max/mae)** — fouroversix (if installed)

All backends are split into **Prologue** (amax/scale computation) and **Quant** (quantization kernel) for fair comparison.

Sample output (B200, `(14400, 6144)`, bf16):

```
  Metric                        tllm     tllm/mse tllm/abs_max     tllm/mae      f46/mse  f46/abs_max      f46/mae  f46/static_6
  -------------------------------------------------------------------------------------------------------------------------------
  Prologue (us)                75.37        75.15        75.12        75.37       101.98       102.81        99.07       102.81
  Quant (us)                   54.35        86.10        79.29        83.83       383.84       167.08       371.69       167.08
  Total (us)                  129.72       161.25       154.41       159.20       485.82       269.89       470.76       269.89
  BW (TB/s)                     1.75         1.41         1.47         1.42         0.47         0.84         0.48         0.84
  MBU %                        21.8%        17.6%        18.4%        17.8%         5.8%        10.5%         6.0%        10.5%
  Speedup vs tllm              1.00x        0.80x        0.84x        0.81x        0.27x        0.48x        0.28x        0.48x
```

- **Prologue**: `triton_amax` → global scale (tllm); `quantize_fp4_prologue` (fouroversix — includes buffer allocation + zeroing + prologue kernel)
- **Quant**: `fp4_quantize` kernel (tllm); `quantize_fp4_main` (fouroversix)
- **Total**: Prologue + Quant
- **BW**: `(input_bytes + output_bytes) / total_time`
- **MBU %**: BW / peak HBM BW (B200 = 8.0 TB/s)
- **Speedup**: `tllm_total / other_total` (>1 = faster than tllm)

The `--scale-rule` flag runs a focused single-rule benchmark:

```
Adaptive 4/6 benchmark (scale_rule=mse)
------------------------------------------------------------
  Kernel time:    22.16 us
  Effective BW:   5.06 TB/s
  Output shape:   fp4=torch.Size([14400, 3072]), sf=torch.Size([14400, 384])

  Mean abs error:  tllm(mse)=0.013916  f46(mse)=0.013916
  Error ratio (tllm/f46): 1.0000
```

When fouroversix is installed, the standalone mode also reports mean absolute error against the fouroversix reference to validate numerical equivalence.

---

## test_nvfp4_gemm.py

NVFP4 GEMM correctness test — compares each backend against `torch.nn.Linear` (bf16) as the reference baseline.

### CLI

| Flag | Default | Description |
|------|---------|-------------|
| `--dtype` | `bfloat16` | `float16` or `bfloat16` |
| `--backend` | `all` | `all`, `cutedsl`, `cutlass`, `cublaslt`, `cuda_core`, `auto` |
| `--no-bias` | off | Skip bias tests |
| `--compare-trtllm` | off | Cross-validate with TensorRT-LLM |

### Examples

```bash
python tests/test_nvfp4_gemm.py
python tests/test_nvfp4_gemm.py --backend cutlass
python tests/test_nvfp4_gemm.py --backend cublaslt --no-bias
```

### Output

Per-shape, per-backend results with cosine similarity and mean/max absolute error against `nn.Linear`:

```
  Testing (4096,4096,4096) backend=cutlass bias=no... [PASS] cos=0.9910
  Testing (4096,4096,4096) backend=cublaslt bias=no... [PASS] cos=0.9912
```

Followed by a summary table (pandas DataFrame) of all results.

---

## test_nvfp4_linear.py

End-to-end correctness and benchmark for `NVFP4DynamicLinear` — the `nn.Module` that fuses dynamic quantization + GEMM.

### CLI

| Flag | Default | Description |
|------|---------|-------------|
| `--dtype` | `bfloat16` | `float16` or `bfloat16` |
| `--gemm-backend` | `auto` | `auto`, `cutlass`, `cublaslt`, `cutedsl` |
| `--quant-backend` | `all` | `tllm`, `fouroversix`, `all` |
| `--no-bias` | off | Skip bias tests |
| `--skip-verify` | off | Skip correctness, benchmark only |
| `--warmup` | `5` | Benchmark warmup iterations |
| `--repeat` | `20` | Benchmark timed iterations |

### Test Cases

| Test | What it checks |
|------|---------------|
| Correctness (multi-shape) | Cosine similarity vs `nn.Linear` across LLM-typical shapes |
| tllm adaptive (mse/abs_max/mae) | Adaptive 4/6 scale rules via tllm backend |
| N-D input | 3-D input `(B, S, K)` propagation |
| state_dict round-trip | Save → load → output bit-exact |
| Zero input | Numerical stability with all-zero activations |
| Auto backend | `auto` must not select `cuda_core` (module uses SWIZZLED layout) |
| fouroversix | `quant_backend="fouroversix"` correctness (when `--quant-backend all`) |

### Examples

```bash
python tests/test_nvfp4_linear.py --gemm-backend cublaslt
python tests/test_nvfp4_linear.py --skip-verify   # benchmark only
python tests/test_nvfp4_linear.py --quant-backend tllm --no-bias
```

### Benchmark Output

Per-shape unified table with Prologue / Quant / GEMM breakdown across all quant
configs. Sample output (B200, bf16, gemm=cublaslt):

```
  Shape: (128, 6144, 6144)
  Metric                        tllm      tllm/mse  tllm/abs_max      tllm/mae       f46/mse       f46/mae  f46/static_6
  ------------------------------------------------------------------------------------------------------------------------
  Prologue (us)                65.26         64.84         65.08         65.03         38.15         38.71         38.54
  Quant (us)                   14.00         14.54         14.47         14.72         34.04         36.27         19.54
  GEMM (us)                    40.00         40.31         40.28         39.91         39.89         40.03         40.24
  Total (us)                  119.26        119.68        119.83        119.66        112.08        115.01         98.32
  TFLOPS                       81.03         80.74         80.65         80.76         86.22         84.03         98.29
  Speedup vs tllm              1.00x         1.00x         1.00x         1.00x         1.06x         1.04x         1.21x

  Shape: (14400, 6144, 6144)
  Metric                        tllm      tllm/mse  tllm/abs_max      tllm/mae       f46/mse       f46/mae  f46/static_6
  ------------------------------------------------------------------------------------------------------------------------
  Prologue (us)                76.06         75.80         77.54         77.00        102.05        102.42        102.33
  Quant (us)                   54.18         85.81         79.01         83.53        383.17        375.67        167.02
  GEMM (us)                   203.99        202.50        202.94        202.25        202.68        203.14        202.68
  Total (us)                  334.23        364.11        359.49        362.78        687.90        681.23        472.03
  TFLOPS                     3252.72       2985.79       3024.21       2996.79       1580.41       1595.89       2303.17
  Speedup vs tllm              1.00x         0.92x         0.93x         0.92x         0.49x         0.49x         0.71x
```

- **Prologue**: `triton_amax` → global scale (tllm); `quantize_fp4_prologue` (fouroversix — includes buffer allocation + zeroing + prologue kernel)
- **Quant**: `fp4_quantize` kernel (tllm); `quantize_fp4_main` (fouroversix)
- **GEMM**: `nvfp4_gemm` (same backend for all configs within a shape)

---

## test_fouroversix/

Standalone fouroversix integration example with end-to-end GEMM verification. See [`test_fouroversix/README.md`](test_fouroversix/README.md) for setup, usage, and reference results.
