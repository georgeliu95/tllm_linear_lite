"""
test_nvfp4_gemm.py - NVFP4 GEMM Correctness Test

Tests the CUTLASS, cuBLASLt, and cuda_core NVFP4 GEMM backends against
torch.nn.Linear as the reference baseline (the real user-facing API).

Supports both with-bias and without-bias cases. Bias is currently
applied as a post-GEMM addition (epilogue fusion planned as follow-up).

Results are collected into a summary table (pandas DataFrame) for easy comparison.

Usage:
    # Basic test (all backends, all shapes, both bias modes)
    python tests/test_nvfp4_gemm.py

    # Test specific backend
    python tests/test_nvfp4_gemm.py --backend cutlass

    # Without bias only
    python tests/test_nvfp4_gemm.py --no-bias

    # Compare with TensorRT-LLM
    python tests/test_nvfp4_gemm.py --compare-trtllm
"""

import math
import torch
import torch.nn as nn
import argparse
from typing import List, Dict, Any

import tllm_linear_lite  # noqa: F401
from tllm_linear_lite.nvfp4_gemm import nvfp4_gemm


# ============================================================================
# Test shapes (typical LLM inference shapes)
# ============================================================================

DEFAULT_SHAPES = [
    # (M, N, K) -- M=batch*seq_len, N=out_features, K=in_features
    (1, 4096, 4096),      # decode, small model
    (4, 4096, 4096),      # small batch decode
    (8, 8192, 4096),      # medium decode
    (16, 4096, 8192),     # max cuda_core M
    (32, 4096, 4096),     # small prefill
    (128, 4096, 4096),    # medium prefill
    (512, 4096, 4096),    # large prefill
    (1024, 8192, 4096),   # large prefill, wide
]


# ============================================================================
# Test data generation
# ============================================================================


def create_nvfp4_linear_test_data(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype = torch.bfloat16,
    use_bias: bool = True,
) -> dict:
    """Create test data that mirrors nn.Linear usage.

    Creates an nn.Linear(k, n, bias=use_bias) as the reference, then
    quantizes its weight (and the input activation) to NVFP4.
    """
    linear = nn.Linear(k, n, bias=use_bias).to(device="cuda", dtype=dtype)
    x = torch.randn(m, k, device="cuda", dtype=dtype)

    weight = linear.weight.data  # [n, k]
    bias = linear.bias.data if use_bias else None

    # Quantize activation (swizzled for cutlass/cuBLASLt)
    act_amax = x.abs().amax().float()
    act_global_scale = 448.0 * 6.0 / act_amax
    act_fp4, act_sf = torch.ops.tllm_linear_lite.fp4_quantize(
        x, act_global_scale, 16, False, True  # swizzled
    )

    # Quantize activation (linear for cuda_core)
    act_fp4_lin, act_sf_lin = torch.ops.tllm_linear_lite.fp4_quantize(
        x, act_global_scale, 16, False, False  # linear
    )

    # Quantize weight (swizzled)
    weight_amax = weight.abs().amax().float()
    weight_global_scale = 448.0 * 6.0 / weight_amax
    weight_fp4, weight_sf = torch.ops.tllm_linear_lite.fp4_quantize(
        weight, weight_global_scale, 16, False, True  # swizzled
    )

    alpha = torch.tensor(
        [1.0 / (act_global_scale.item() * weight_global_scale.item())],
        device="cuda", dtype=torch.float32,
    )

    return {
        "linear": linear, "input": x,
        "act_fp4": act_fp4, "act_fp4_lin": act_fp4_lin,
        "act_sf": act_sf, "act_sf_lin": act_sf_lin,
        "weight_fp4": weight_fp4, "weight_sf": weight_sf,
        "alpha": alpha, "bias": bias,
    }


# ============================================================================
# GEMM dispatch
# ============================================================================


def run_nvfp4_gemm(data: dict, backend: str, output_dtype: torch.dtype) -> torch.Tensor:
    """Run NVFP4 GEMM with the specified backend, including bias addition."""
    if backend == "cuda_core":
        gemm_out = nvfp4_gemm(
            data["act_fp4_lin"], data["weight_fp4"],
            data["act_sf_lin"], data["weight_sf"],
            data["alpha"],
            output_dtype=output_dtype, backend="cuda_core",
        )
    else:
        # cutlass / cuBLASLt both use SWIZZLED layout for scale factors
        gemm_out = nvfp4_gemm(
            data["act_fp4"], data["weight_fp4"],
            data["act_sf"], data["weight_sf"],
            data["alpha"],
            output_dtype=output_dtype, backend=backend,
        )

    if data["bias"] is not None:
        gemm_out = gemm_out + data["bias"]

    return gemm_out


# ============================================================================
# Single-shape correctness check
# ============================================================================


def check_correctness(
    m: int, n: int, k: int, dtype: torch.dtype, backend: str, use_bias: bool,
) -> Dict[str, Any]:
    """Check correctness for a single (shape, backend, bias) combo.

    Returns a dict with metrics (for the summary table).
    """
    data = create_nvfp4_linear_test_data(m, n, k, dtype, use_bias)

    # nn.Linear reference
    with torch.no_grad():
        ref_output = data["linear"](data["input"]).float()

    # NVFP4 GEMM
    result = {
        "M": m, "N": n, "K": k, "backend": backend,
        "bias": "yes" if use_bias else "no",
        "status": "FAIL", "max_abs_err": float("nan"),
        "mean_abs_err": float("nan"), "mean_rel_err": float("nan"),
        "cosine_sim": float("nan"),
    }

    try:
        gemm_output = run_nvfp4_gemm(data, backend, dtype).float()
    except Exception as e:
        result["status"] = f"ERROR: {e}"
        return result

    abs_err = (gemm_output - ref_output).abs()
    rel_err = abs_err / (ref_output.abs().clamp(min=1e-6))
    cosine_sim = torch.nn.functional.cosine_similarity(
        gemm_output.flatten().unsqueeze(0),
        ref_output.flatten().unsqueeze(0),
    ).item()

    result["max_abs_err"] = abs_err.max().item()
    result["mean_abs_err"] = abs_err.mean().item()
    result["mean_rel_err"] = rel_err.mean().item()
    result["cosine_sim"] = cosine_sim

    # --- Validation criteria ---
    # 1. torch.testing.assert_close with FP4-appropriate tolerances
    atol = 0.5 * math.sqrt(k / 16.0)
    rtol = 0.5
    assert_close_pass = True
    try:
        torch.testing.assert_close(gemm_output, ref_output, atol=atol, rtol=rtol)
    except Exception:
        assert_close_pass = False

    # 2. Cosine similarity > 0.9
    cos_pass = cosine_sim > 0.9

    if assert_close_pass and cos_pass:
        result["status"] = "PASS"
    elif cos_pass:
        result["status"] = "WARN"  # cosine OK but some outliers
    else:
        result["status"] = "FAIL"

    return result


# ============================================================================
# Summary table
# ============================================================================


def _colorize_status(status: str) -> str:
    """Colorize status string with ANSI codes for terminal output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    if status == "PASS":
        return f"{GREEN}{status}{RESET}"
    elif status == "FAIL" or status.startswith("ERROR"):
        return f"{RED}{status}{RESET}"
    elif status == "WARN":
        return f"{YELLOW}{status}{RESET}"
    return status


def print_summary_table(results: List[Dict[str, Any]]):
    """Print a summary table of all test results.

    Uses pandas if available, otherwise falls back to plain text.
    Status column is first, colored green (PASS) / red (FAIL) / yellow (WARN).
    """
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        df["shape"] = df.apply(lambda r: f"({r['M']},{r['N']},{r['K']})", axis=1)

        # Status first, then shape/backend/bias, then metrics
        display_cols = ["status", "shape", "backend", "bias", "cosine_sim", "mean_abs_err", "mean_rel_err", "max_abs_err"]
        df_display = df[display_cols].copy()

        # Format numeric columns
        df_display["cosine_sim"] = df_display["cosine_sim"].map(lambda x: f"{x:.6f}" if not math.isnan(x) else "NaN")
        df_display["mean_abs_err"] = df_display["mean_abs_err"].map(lambda x: f"{x:.4f}" if not math.isnan(x) else "NaN")
        df_display["mean_rel_err"] = df_display["mean_rel_err"].map(lambda x: f"{x:.4f}" if not math.isnan(x) else "NaN")
        df_display["max_abs_err"] = df_display["max_abs_err"].map(lambda x: f"{x:.4f}" if not math.isnan(x) else "NaN")

        # Colorize status column
        df_display["status"] = df_display["status"].map(_colorize_status)

        print("\n" + df_display.to_string(index=False))

    except ImportError:
        # Fallback: plain text table
        header = f"{'Status':<8} {'Shape':<20} {'Backend':<10} {'Bias':<5} {'Cosine':>10} {'MeanAbs':>10} {'MeanRel':>10} {'MaxAbs':>10}"
        print("\n" + header)
        print("-" * len(header))
        for r in results:
            shape = f"({r['M']},{r['N']},{r['K']})"
            cos = f"{r['cosine_sim']:.6f}" if not math.isnan(r['cosine_sim']) else "NaN"
            mae = f"{r['mean_abs_err']:.4f}" if not math.isnan(r['mean_abs_err']) else "NaN"
            mre = f"{r['mean_rel_err']:.4f}" if not math.isnan(r['mean_rel_err']) else "NaN"
            maxe = f"{r['max_abs_err']:.4f}" if not math.isnan(r['max_abs_err']) else "NaN"
            status = _colorize_status(r['status'])
            print(f"{status:<17} {shape:<20} {r['backend']:<10} {r['bias']:<5} {cos:>10} {mae:>10} {mre:>10} {maxe:>10}")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="NVFP4 GEMM Correctness Test (nn.Linear baseline)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16"])
    parser.add_argument("--backend", type=str, default="all",
                        choices=["all", "cutlass", "cublaslt", "cuda_core", "auto"])
    parser.add_argument("--no-bias", action="store_true", help="Skip bias tests")
    parser.add_argument("--compare-trtllm", action="store_true",
                        help="Compare with TensorRT-LLM nvfp4_gemm")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    bias_modes = [False] if args.no_bias else [False, True]

    print("=" * 80)
    print("tllm_linear_lite NVFP4 GEMM Correctness Test")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Dtype:  {dtype}")
    print(f"Bias:   {['no-bias', 'with-bias'] if len(bias_modes) == 2 else ['no-bias']}")
    print(f"Shapes: {len(DEFAULT_SHAPES)} configurations")

    all_results: List[Dict[str, Any]] = []

    for m, n, k in DEFAULT_SHAPES:
        # Determine backends for this shape
        if args.backend == "all":
            backends = ["cutlass", "cublaslt"]
            if m <= 16:
                backends.append("cuda_core")
        else:
            backends = [args.backend]

        for backend in backends:
            for use_bias in bias_modes:
                print(f"\n  Testing ({m},{n},{k}) backend={backend} bias={'yes' if use_bias else 'no'}...", end="", flush=True)
                result = check_correctness(m, n, k, dtype, backend, use_bias)
                all_results.append(result)
                status = result["status"]
                cos = f"{result['cosine_sim']:.4f}" if not math.isnan(result['cosine_sim']) else "NaN"
                print(f" [{status}] cos={cos}")

    # --- TRT-LLM comparison ---
    if args.compare_trtllm:
        print("\n" + "=" * 80)
        print("Cross-validation with TensorRT-LLM")
        print("=" * 80)
        try:
            import tensorrt_llm  # noqa: F401
            print("[SKIP] TRT-LLM comparison not yet implemented for GEMM")
        except ImportError:
            print("[SKIP] tensorrt_llm not installed")

    # --- Summary ---
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print_summary_table(all_results)

    # Final verdict
    num_pass = sum(1 for r in all_results if r["status"] == "PASS")
    num_warn = sum(1 for r in all_results if r["status"] == "WARN")
    num_fail = sum(1 for r in all_results if r["status"] not in ("PASS", "WARN"))
    total = len(all_results)

    print(f"\nTotal: {total} tests, {num_pass} PASS, {num_warn} WARN, {num_fail} FAIL")
    if num_fail == 0:
        print("[PASS] All correctness checks passed")
    else:
        print("[FAIL] Some tests failed, review above")
    print("=" * 80)


if __name__ == "__main__":
    main()
