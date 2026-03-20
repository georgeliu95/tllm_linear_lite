# Tests

Correctness, performance benchmarks, and cross-validation for tllm_linear_lite.

## Test Scripts

```
tests/
├── test_prologue.py        # Prologue (amax → global_scale): correctness + benchmark + tuning
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

## test_prologue.py

Prologue (amax → global_scale) correctness and performance test.  Compares all
available implementations: PyTorch baseline, Triton two-stage, CUDA single-kernel
(`cuda_prologue`), and fouroversix.  Includes a `--tune` sweep to find the optimal
implementation across different M values.

### CLI

| Flag | Default | Description |
|------|---------|-------------|
| `--shapes` | `1,6144 128,6144 ...` | Shapes as M,K |
| `--dtype` | `bfloat16` | `float16` or `bfloat16` |
| `--warmup` | `10` | Warmup iterations |
| `--repeat` | `50` | Benchmark iterations |
| `--skip-verify` | off | Skip correctness |
| `--tune` | off | Sweep M=1..14400 to find crossover between implementations |
| `--tune-k` | `6144` | K dimension for tuning sweep |

### Examples

```bash
# Correctness + benchmark for a single shape
python tests/test_prologue.py --shapes 128,6144

# Tuning sweep: find where each implementation wins
python tests/test_prologue.py --tune

# Full run: multiple shapes + tune
python tests/test_prologue.py --shapes 128,6144 14400,6144 --tune
```

### Output

Per-shape accuracy table followed by performance comparison:

```
  Reference:  amax = 4.968750,  global_scale = 540.981132

  Candidate                amax   global_scale      rel_err   status
  ------------------------------------------------------------------
  pytorch              4.968750     540.981140     1.49e-08     PASS
  triton_amax          4.968750     540.981140     1.49e-08     PASS
  cuda_prologue        4.968750     540.981140     1.49e-08     PASS
  f46_prologue         4.968750     540.981132     0.00e+00     PASS

  Candidate        Time(us)   BW(GB/s)    vs best
  -----------------------------------------------
  pytorch              39.04      40.29      3.18x
  triton_amax          54.33      28.95      4.43x
  cuda_prologue        12.27     128.15      1.00x <<<
  f46_prologue         30.21      52.07      2.46x
```

- **cuda_prologue**: single-kernel last-block reduction (fused amax + clamp + division)
- **triton_amax**: Triton two-stage reduction + PyTorch `.max()` + `.clamp_min()` + division
- **f46_prologue**: fouroversix `quantize_fp4_prologue` (includes buffer alloc + kernel)

The `--tune` flag sweeps M from 1 to 14400 and reports the winner at each point:

```
  M     Elements    pytorch     triton   cuda_pro    f46_pro   winner         speedup
  ------- ----------  ---------- ---------- ---------- ----------  -------------- --------
        1       6,144      39.0      55.0      12.0      30.0  cuda_pro        2.50x
      128     786,432      39.0      55.0      12.0      30.0  cuda_pro        2.50x
    14400  88,473,600      76.0      60.0      65.0     102.0  triton          1.10x

Crossover: cuda_prologue → triton_amax at M ≈ 14400
```

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

Sample output (B200, bf16):

```
  Shape: (128, 6144)
  Metric                             tllm     tllm(mse) tllm(abs_max)     tllm(mae)      f46(mse)  f46(abs_max)      f46(mae)
  ---------------------------------------------------------------------------------------------------------------------------
  Prologue (us)                     13.47         13.47         13.47         13.47         28.30         28.31         27.75
  Quant (us)                        12.67         12.96         12.88         12.66         34.04         34.08         33.86
  Total (us)                        26.14         26.43         26.35         26.13         62.34         62.40         61.61
  Speedup vs tllm                   1.00x         0.99x         0.99x         1.00x         0.42x         0.42x         0.42x

  Shape: (14400, 6144)
  Metric                             tllm     tllm(mse) tllm(abs_max)     tllm(mae)      f46(mse)  f46(abs_max)      f46(mae)
  ---------------------------------------------------------------------------------------------------------------------------
  Prologue (us)                     65.70         65.70         65.70         65.70         99.71         99.40         99.33
  Quant (us)                        55.27         85.25         78.47         82.81        383.93        397.46        389.55
  Total (us)                       120.96        150.95        144.17        148.50        483.64        496.86        488.88
  BW (TB/s)                          1.87          1.50          1.57          1.53          0.47          0.46          0.46
  MBU %                             23.4%         18.8%         19.7%         19.1%          5.9%          5.7%          5.8%
  Speedup vs tllm                   1.00x         0.80x         0.84x         0.81x         0.25x         0.24x         0.25x
```

- **Prologue**: `cuda_prologue` single-kernel amax+scale (tllm); `quantize_fp4_prologue` (fouroversix — includes buffer allocation + zeroing + prologue kernel)
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
  Shape: (1, 6144, 6144)
  Metric                        tllm      tllm/mse  tllm/abs_max      tllm/mae       f46/mse       f46/mae  f46/static_6
  ----------------------------------------------------------------------------------------------------------------------
  Prologue (us)                15.45         15.36         15.38         15.40         39.15         38.96         39.41
  Quant (us)                   16.36         16.16         16.11         16.03         36.44         36.86         19.38
  GEMM (us)                    45.88         46.27         46.60         46.64         40.06         40.18         39.90
  Total (us)                   77.69         77.78         78.09         78.07        115.65        116.00         98.69
  TFLOPS                        0.97          0.97          0.97          0.97          0.65          0.65          0.76
  Speedup vs tllm              1.00x         1.00x         0.99x         1.00x         0.67x         0.67x         0.79x

  Shape: (128, 6144, 6144)
  Metric                        tllm      tllm/mse  tllm/abs_max      tllm/mae       f46/mse       f46/mae  f46/static_6
  ----------------------------------------------------------------------------------------------------------------------
  Prologue (us)                15.79         15.78         15.39         15.55         38.89         38.89         39.27
  Quant (us)                   16.10         16.15         15.95         16.11         36.20         35.74         19.65
  GEMM (us)                    45.35         45.63         45.73         45.91         42.20         42.25         43.09
  Total (us)                   77.24         77.56         77.06         77.58        117.29        116.88        102.02
  TFLOPS                      125.12        124.60        125.40        124.57         82.39         82.68         94.73
  Speedup vs tllm              1.00x         1.00x         1.00x         1.00x         0.66x         0.66x         0.76x

  Shape: (14400, 6144, 6144)
  Metric                        tllm      tllm/mse  tllm/abs_max      tllm/mae       f46/mse       f46/mae  f46/static_6
  ----------------------------------------------------------------------------------------------------------------------
  Prologue (us)                66.41         66.22         66.05         66.31        102.27        102.14        102.64
  Quant (us)                   54.89         85.58         79.43         85.09        388.15        393.56        166.80
  GEMM (us)                   203.14        201.76        201.69        203.32        204.31        204.18        204.76
  Total (us)                  324.44        353.56        347.16        354.72        694.73        699.87        474.20
  TFLOPS                     3350.94       3074.93       3131.56       3064.84       1564.87       1553.38       2292.63
  Speedup vs tllm              1.00x         0.92x         0.93x         0.91x         0.47x         0.46x         0.68x
```

- **Prologue**: `cuda_prologue` single-kernel amax+scale (tllm, ~15 us for M≤128); `quantize_fp4_prologue` (fouroversix — includes buffer allocation + zeroing + prologue kernel, ~39 us)
- **Quant**: `fp4_quantize` kernel (tllm); `quantize_fp4_main` (fouroversix)
- **GEMM**: `nvfp4_gemm` (same backend for all configs within a shape)

---

## test_fouroversix/

Standalone fouroversix integration example with end-to-end GEMM verification. See [`test_fouroversix/README.md`](test_fouroversix/README.md) for setup, usage, and reference results.
