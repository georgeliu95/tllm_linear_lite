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

The table reports two timing rows for fair comparison:
- **Kernel only** — quantize kernel alone (tllm scale precomputed; f46 includes its own prologue)
- **End-to-end** — prologue (`triton_amax` → global scale) + quantize kernel for tllm; same as kernel-only for f46 (prologue is internal)

Sample output (B200, `(14400, 6144)`, bf16):

```
  Metric                             tllm     tllm(mse) tllm(abs_max)     tllm(mae)      f46(mse)  f46(abs_max)      f46(mae)
  ---------------------------------------------------------------------------------------------------------------------------
  Kernel only (us)                  52.00         83.44         79.11         81.87        454.53        466.24        459.67
  End-to-end (us)                  122.49        183.80        180.90        181.41        454.53        466.24        459.67
  E2E BW (TB/s)                      1.85          1.23          1.25          1.25          0.50          0.49          0.49
  E2E MBU %                         23.1%         15.4%         15.7%         15.6%          6.2%          6.1%          6.2%
  Kernel speedup                    1.00x         0.62x         0.66x         0.64x           N/A           N/A           N/A
  E2E speedup                       1.00x         0.67x         0.68x         0.68x         0.27x         0.26x         0.27x

  Prologue: triton_amax = 70.49 us (included in end-to-end; f46 includes its own prologue)
```

- **Kernel only**: CUDA event elapsed time for the quantize kernel alone
- **End-to-end**: Prologue (`triton_amax` → global scale) + quantize kernel for tllm; f46 includes its built-in prologue
- **E2E BW**: `(input_bytes + output_bytes) / end_to_end_time`
- **E2E MBU %**: E2E BW / peak HBM BW (B200 = 8.0 TB/s)
- **Kernel speedup**: `tllm_kernel / other_kernel` (>1 = faster); N/A for f46 (kernel time includes prologue)
- **E2E speedup**: `tllm_e2e / other_e2e` (>1 = faster); apples-to-apples across all columns

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

End-to-end correctness test for `NVFP4DynamicLinear` — the `nn.Module` that fuses dynamic quantization + GEMM.

### CLI

| Flag | Default | Description |
|------|---------|-------------|
| `--dtype` | `bfloat16` | `float16` or `bfloat16` |
| `--gemm-backend` | `auto` | `auto`, `cutlass`, `cublaslt`, `cutedsl` |
| `--quant-backend` | `tllm` | `tllm`, `fouroversix`, `all` |
| `--no-bias` | off | Skip bias tests |

### Test Cases

| Test | What it checks |
|------|---------------|
| Correctness (multi-shape) | Cosine similarity vs `nn.Linear` across LLM-typical shapes |
| N-D input | 3-D input `(B, S, K)` propagation |
| state_dict round-trip | Save → load → output bit-exact |
| Zero input | Numerical stability with all-zero activations |
| Auto backend | `auto` must not select `cuda_core` (module uses SWIZZLED layout) |
| fouroversix | `quant_backend="fouroversix"` correctness (when `--quant-backend all`) |

### Examples

```bash
python tests/test_nvfp4_linear.py
python tests/test_nvfp4_linear.py --gemm-backend cublaslt
python tests/test_nvfp4_linear.py --quant-backend all   # includes fouroversix
```

---

## test_fouroversix/

Standalone fouroversix integration example with end-to-end GEMM verification. See [`test_fouroversix/README.md`](test_fouroversix/README.md) for setup, usage, and reference results.
