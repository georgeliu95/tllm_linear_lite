# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
test_nvfp4_linear.py - NVFP4DynamicLinear End-to-End Correctness Test

Tests the NVFP4DynamicLinear module against nn.Linear as reference, covering:
  - Correctness (cosine similarity + assert_close) across LLM-typical shapes
  - With/without bias
  - N-D (3-D) input compatibility
  - state_dict save/load round-trip
  - Zero / near-zero activation input (numerical stability)
  - Module-level GEMM backend resolution (auto must not pick cuda_core)
  - fouroversix quant_backend (when available)

Results are collected into a summary table matching test_nvfp4_gemm.py style.

Usage:
    # Basic test (tllm quant_backend, all shapes, both bias modes)
    python tests/test_nvfp4_linear.py

    # Test specific GEMM backend
    python tests/test_nvfp4_linear.py --gemm-backend cublaslt

    # Skip bias tests
    python tests/test_nvfp4_linear.py --no-bias

    # Include fouroversix quant_backend tests (requires fouroversix)
    python tests/test_nvfp4_linear.py --quant-backend all
"""

import math
import tempfile
import os
import argparse
from typing import List, Dict, Any

import torch
import torch.nn as nn

import tllm_linear_lite  # noqa: F401
from tllm_linear_lite.nvfp4_linear import NVFP4DynamicLinear
from tllm_linear_lite.cutedsl import IS_CUTLASS_DSL_AVAILABLE
from tllm_linear_lite.quantize import FOUROVERSIX_AVAILABLE


# ============================================================================
# Test shapes
# ============================================================================

DEFAULT_SHAPES = [
    # (M, N, K) -- M=batch*seq_len, N=out_features, K=in_features
    (1, 4096, 4096),
    (4, 4096, 4096),
    (16, 4096, 8192),
    (32, 4096, 4096),
    (128, 4096, 4096),
    (512, 4096, 4096),
    (1024, 8192, 4096),
]


# ============================================================================
# Correctness check (single configuration)
# ============================================================================


def check_correctness(
    m: int, n: int, k: int,
    dtype: torch.dtype,
    use_bias: bool,
    gemm_backend: str = "auto",
    quant_backend: str = "tllm",
    quant_config: object = None,
) -> Dict[str, Any]:
    """Check NVFP4DynamicLinear output against nn.Linear reference.

    Args:
        m: Batch dimension (number of tokens).
        n: Output features.
        k: Input features.
        dtype: Weight / activation dtype.
        use_bias: Whether to include bias.
        gemm_backend: GEMM backend for the module.
        quant_backend: Quantization backend for the module.
        quant_config: fouroversix QuantizationConfig (or None).

    Returns:
        Dict with test metrics for the summary table.
    """
    linear = nn.Linear(k, n, bias=use_bias).to(device="cuda", dtype=dtype)
    x = torch.randn(m, k, device="cuda", dtype=dtype)

    result: Dict[str, Any] = {
        "M": m, "N": n, "K": k,
        "gemm": gemm_backend, "quant": quant_backend,
        "bias": "yes" if use_bias else "no",
        "status": "FAIL",
        "max_abs_err": float("nan"), "mean_abs_err": float("nan"),
        "mean_rel_err": float("nan"), "cosine_sim": float("nan"),
    }

    try:
        nvfp4 = NVFP4DynamicLinear.from_linear(
            linear,
            gemm_backend=gemm_backend,
            quant_backend=quant_backend,
            quant_config=quant_config,
        )
    except Exception as e:
        result["status"] = f"ERROR(init): {e}"
        return result

    with torch.no_grad():
        ref_output = linear(x).float()

    try:
        with torch.no_grad():
            nvfp4_output = nvfp4(x).float()
    except Exception as e:
        result["status"] = f"ERROR(fwd): {e}"
        return result

    abs_err = (nvfp4_output - ref_output).abs()
    rel_err = abs_err / ref_output.abs().clamp(min=1e-6)
    cosine_sim = nn.functional.cosine_similarity(
        nvfp4_output.flatten().unsqueeze(0),
        ref_output.flatten().unsqueeze(0),
    ).item()

    result["max_abs_err"] = abs_err.max().item()
    result["mean_abs_err"] = abs_err.mean().item()
    result["mean_rel_err"] = rel_err.mean().item()
    result["cosine_sim"] = cosine_sim

    atol = 0.5 * math.sqrt(k / 16.0)
    rtol = 0.5
    assert_close_pass = True
    try:
        torch.testing.assert_close(nvfp4_output, ref_output, atol=atol, rtol=rtol)
    except Exception:
        assert_close_pass = False

    cos_pass = cosine_sim > 0.9
    if assert_close_pass and cos_pass:
        result["status"] = "PASS"
    elif cos_pass:
        result["status"] = "WARN"
    else:
        result["status"] = "FAIL"

    return result


# ============================================================================
# Structural tests (from_linear buffers, N-D input, state_dict, zero input)
# ============================================================================


def test_from_linear_buffers(dtype: torch.dtype) -> Dict[str, Any]:
    """Verify that from_linear produces buffers with correct shapes/dtypes.

    Args:
        dtype: Weight dtype.

    Returns:
        Result dict.
    """
    n, k = 4096, 4096
    result: Dict[str, Any] = {
        "M": "-", "N": n, "K": k,
        "gemm": "-", "quant": "tllm", "bias": "yes",
        "status": "FAIL", "max_abs_err": float("nan"),
        "mean_abs_err": float("nan"), "mean_rel_err": float("nan"),
        "cosine_sim": float("nan"),
    }

    linear = nn.Linear(k, n, bias=True).to(device="cuda", dtype=dtype)
    nvfp4 = NVFP4DynamicLinear.from_linear(linear)

    checks = []
    checks.append(("weight_fp4.shape", nvfp4.weight_fp4.shape == (n, k // 2)))
    checks.append(("weight_fp4.dtype", nvfp4.weight_fp4.dtype == torch.uint8))
    checks.append(("weight_sf.dtype", nvfp4.weight_sf.dtype == torch.uint8))
    checks.append(("weight_sf.numel>0", nvfp4.weight_sf.numel() > 0))
    checks.append(("alpha_multiplier.shape", nvfp4.alpha_multiplier.shape == (1,)))
    checks.append(("alpha_multiplier.finite", nvfp4.alpha_multiplier.isfinite().all().item()))
    checks.append(("alpha_multiplier>0", (nvfp4.alpha_multiplier > 0).all().item()))
    checks.append(("bias.shape", nvfp4.bias is not None and nvfp4.bias.shape == (n,)))

    failed = [name for name, ok in checks if not ok]
    if not failed:
        result["status"] = "PASS"
    else:
        result["status"] = f"FAIL({','.join(failed)})"
    return result


def test_nd_input(dtype: torch.dtype) -> Dict[str, Any]:
    """Verify that 3-D input [..., K] -> [..., N] matches nn.Linear.

    Args:
        dtype: Weight dtype.

    Returns:
        Result dict.
    """
    B, S, K, N = 2, 17, 4096, 4096
    result: Dict[str, Any] = {
        "M": f"{B}x{S}", "N": N, "K": K,
        "gemm": "auto", "quant": "tllm", "bias": "yes",
        "status": "FAIL", "max_abs_err": float("nan"),
        "mean_abs_err": float("nan"), "mean_rel_err": float("nan"),
        "cosine_sim": float("nan"),
    }

    linear = nn.Linear(K, N, bias=True).to(device="cuda", dtype=dtype)
    nvfp4 = NVFP4DynamicLinear.from_linear(linear, gemm_backend="cublaslt")
    x = torch.randn(B, S, K, device="cuda", dtype=dtype)

    result["gemm"] = "cublaslt"

    try:
        with torch.no_grad():
            ref = linear(x).float()
            out = nvfp4(x).float()
    except Exception as e:
        result["status"] = f"ERROR: {e}"
        return result

    if out.shape != ref.shape:
        result["status"] = f"FAIL(shape: {out.shape} != {ref.shape})"
        return result

    cosine_sim = nn.functional.cosine_similarity(
        out.flatten().unsqueeze(0), ref.flatten().unsqueeze(0),
    ).item()
    result["cosine_sim"] = cosine_sim
    result["mean_abs_err"] = (out - ref).abs().mean().item()
    result["status"] = "PASS" if cosine_sim > 0.9 else "FAIL"
    return result


def test_state_dict_roundtrip(dtype: torch.dtype) -> Dict[str, Any]:
    """Save state_dict, load into fresh module, verify identical output.

    Args:
        dtype: Weight dtype.

    Returns:
        Result dict.
    """
    N, K = 4096, 4096
    result: Dict[str, Any] = {
        "M": 32, "N": N, "K": K,
        "gemm": "auto", "quant": "tllm", "bias": "yes",
        "status": "FAIL", "max_abs_err": float("nan"),
        "mean_abs_err": float("nan"), "mean_rel_err": float("nan"),
        "cosine_sim": float("nan"),
    }

    linear = nn.Linear(K, N, bias=True).to(device="cuda", dtype=dtype)
    nvfp4_orig = NVFP4DynamicLinear.from_linear(linear, gemm_backend="cublaslt")
    x = torch.randn(32, K, device="cuda", dtype=dtype)

    result["gemm"] = "cublaslt"

    try:
        with torch.no_grad():
            out_orig = nvfp4_orig(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nvfp4.pt")
            torch.save(nvfp4_orig.state_dict(), path)

            nvfp4_loaded = NVFP4DynamicLinear(
                K, N, bias=True, dtype=dtype, gemm_backend="cublaslt",
            ).to("cuda")
            nvfp4_loaded.load_state_dict(torch.load(path, weights_only=True))

            with torch.no_grad():
                out_loaded = nvfp4_loaded(x)

        if not torch.equal(out_orig, out_loaded):
            diff = (out_orig.float() - out_loaded.float()).abs().max().item()
            result["status"] = f"FAIL(max_diff={diff:.2e})"
        else:
            result["status"] = "PASS"
    except Exception as e:
        result["status"] = f"ERROR: {e}"
    return result


def test_zero_input() -> Dict[str, Any]:
    """Verify that all-zero activation produces finite output.

    Returns:
        Result dict.
    """
    M, N, K = 4, 4096, 4096
    result: Dict[str, Any] = {
        "M": M, "N": N, "K": K,
        "gemm": "auto", "quant": "tllm", "bias": "no",
        "status": "FAIL", "max_abs_err": float("nan"),
        "mean_abs_err": float("nan"), "mean_rel_err": float("nan"),
        "cosine_sim": float("nan"),
    }

    linear = nn.Linear(K, N, bias=False).to(device="cuda", dtype=torch.bfloat16)
    nvfp4 = NVFP4DynamicLinear.from_linear(linear, gemm_backend="cublaslt")
    x = torch.zeros(M, K, device="cuda", dtype=torch.bfloat16)

    result["gemm"] = "cublaslt"

    try:
        with torch.no_grad():
            out = nvfp4(x)

        if out.isfinite().all().item():
            result["status"] = "PASS"
        else:
            n_inf = (~out.isfinite()).sum().item()
            result["status"] = f"FAIL({n_inf} non-finite)"
    except Exception as e:
        result["status"] = f"ERROR: {e}"
    return result


def test_auto_no_cuda_core() -> Dict[str, Any]:
    """Verify auto GEMM backend does not select cuda_core for decode shapes.

    Returns:
        Result dict.
    """
    result: Dict[str, Any] = {
        "M": 1, "N": 4096, "K": 4096,
        "gemm": "auto", "quant": "tllm", "bias": "no",
        "status": "FAIL", "max_abs_err": float("nan"),
        "mean_abs_err": float("nan"), "mean_rel_err": float("nan"),
        "cosine_sim": float("nan"),
    }

    linear = nn.Linear(4096, 4096, bias=False).to(device="cuda", dtype=torch.bfloat16)
    nvfp4 = NVFP4DynamicLinear.from_linear(linear, gemm_backend="auto")

    resolved = nvfp4._resolve_gemm_backend(M=1, N=4096, K=4096)
    if resolved == "cuda_core":
        result["status"] = "FAIL(resolved to cuda_core)"
    else:
        result["status"] = f"PASS(resolved={resolved})"
    return result


# ============================================================================
# fouroversix quant_backend tests
# ============================================================================


def test_fouroversix_correctness(
    m: int, n: int, k: int, dtype: torch.dtype, scale_rule: str,
) -> Dict[str, Any]:
    """Check fouroversix quant_backend correctness.

    Args:
        m: Batch dim.
        n: Out features.
        k: In features.
        dtype: Dtype.
        scale_rule: fouroversix scale rule (e.g. "mse", "static_6").

    Returns:
        Result dict.
    """
    result: Dict[str, Any] = {
        "M": m, "N": n, "K": k,
        "gemm": "cublaslt", "quant": f"fox/{scale_rule}",
        "bias": "no",
        "status": "FAIL", "max_abs_err": float("nan"),
        "mean_abs_err": float("nan"), "mean_rel_err": float("nan"),
        "cosine_sim": float("nan"),
    }

    try:
        from fouroversix import QuantizationConfig
        config = QuantizationConfig(scale_rule=scale_rule)
    except Exception as e:
        result["status"] = f"SKIP({e})"
        return result

    linear = nn.Linear(k, n, bias=False).to(device="cuda", dtype=dtype)
    x = torch.randn(m, k, device="cuda", dtype=dtype)

    try:
        nvfp4 = NVFP4DynamicLinear.from_linear(
            linear,
            gemm_backend="cublaslt",
            quant_backend="fouroversix",
            quant_config=config,
        )
    except Exception as e:
        result["status"] = f"ERROR(init): {e}"
        return result

    with torch.no_grad():
        ref = linear(x).float()

    try:
        with torch.no_grad():
            out = nvfp4(x).float()
    except Exception as e:
        result["status"] = f"ERROR(fwd): {e}"
        return result

    cosine_sim = nn.functional.cosine_similarity(
        out.flatten().unsqueeze(0), ref.flatten().unsqueeze(0),
    ).item()
    abs_err = (out - ref).abs()

    result["cosine_sim"] = cosine_sim
    result["max_abs_err"] = abs_err.max().item()
    result["mean_abs_err"] = abs_err.mean().item()
    result["mean_rel_err"] = (abs_err / ref.abs().clamp(min=1e-6)).mean().item()

    cos_pass = cosine_sim > 0.9
    result["status"] = "PASS" if cos_pass else "FAIL"
    return result


def test_fouroversix_unavailable() -> Dict[str, Any]:
    """Verify that quant_backend='fouroversix' raises ImportError when unavailable.

    Returns:
        Result dict.
    """
    result: Dict[str, Any] = {
        "M": "-", "N": "-", "K": "-",
        "gemm": "-", "quant": "fox/unavail", "bias": "-",
        "status": "SKIP", "max_abs_err": float("nan"),
        "mean_abs_err": float("nan"), "mean_rel_err": float("nan"),
        "cosine_sim": float("nan"),
    }

    if FOUROVERSIX_AVAILABLE:
        result["status"] = "SKIP(fouroversix installed)"
        return result

    try:
        NVFP4DynamicLinear(4096, 4096, quant_backend="fouroversix")
        result["status"] = "FAIL(no error raised)"
    except ImportError:
        result["status"] = "PASS"
    except Exception as e:
        result["status"] = f"FAIL(wrong error: {type(e).__name__})"
    return result


# ============================================================================
# Summary table (reuses pattern from test_nvfp4_gemm.py)
# ============================================================================


def _colorize_status(status: str) -> str:
    GREEN, RED, YELLOW, CYAN, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[96m", "\033[0m"
    if status == "PASS" or status.startswith("PASS("):
        return f"{GREEN}{status}{RESET}"
    elif status.startswith("FAIL") or status.startswith("ERROR"):
        return f"{RED}{status}{RESET}"
    elif status == "WARN":
        return f"{YELLOW}{status}{RESET}"
    elif status.startswith("SKIP"):
        return f"{CYAN}{status}{RESET}"
    return status


def print_summary_table(results: List[Dict[str, Any]]):
    """Print summary table of all test results.

    Args:
        results: List of result dicts.
    """
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        display_cols = [c for c in ["status", "M", "N", "K", "gemm", "quant", "bias",
                                    "cosine_sim", "mean_abs_err", "mean_rel_err", "max_abs_err"]
                        if c in df.columns]
        df_display = df[display_cols].copy()

        for col in ["cosine_sim", "mean_abs_err", "mean_rel_err", "max_abs_err"]:
            if col in df_display.columns:
                df_display[col] = df_display[col].map(
                    lambda v: f"{v:.6f}" if isinstance(v, float) and not math.isnan(v) else "NaN"
                )

        df_display["status"] = df_display["status"].map(_colorize_status)
        print("\n" + df_display.to_string(index=False))

    except ImportError:
        header = f"{'Status':<30} {'M':<8} {'N':<6} {'K':<6} {'GEMM':<10} {'Quant':<14} {'Bias':<5} {'Cosine':>10}"
        print("\n" + header)
        print("-" * len(header))
        for r in results:
            cos = f"{r['cosine_sim']:.6f}" if isinstance(r.get('cosine_sim'), float) and not math.isnan(r['cosine_sim']) else "NaN"
            status = _colorize_status(r['status'])
            print(f"{status:<39} {str(r['M']):<8} {str(r['N']):<6} {str(r['K']):<6} "
                  f"{r['gemm']:<10} {r['quant']:<14} {r['bias']:<5} {cos:>10}")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="NVFP4DynamicLinear End-to-End Correctness Test"
    )
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16"])
    parser.add_argument("--gemm-backend", type=str, default="auto",
                        choices=["auto", "cutlass", "cublaslt", "cutedsl"])
    parser.add_argument("--quant-backend", type=str, default="tllm",
                        choices=["tllm", "fouroversix", "all"])
    parser.add_argument("--no-bias", action="store_true",
                        help="Skip bias tests")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    bias_modes = [False] if args.no_bias else [False, True]

    print("=" * 80)
    print("tllm_linear_lite NVFP4DynamicLinear Correctness Test")
    print("=" * 80)
    print(f"Device:       {torch.cuda.get_device_name(0)}")
    print(f"Dtype:        {dtype}")
    print(f"GEMM backend: {args.gemm_backend}")
    print(f"Quant backend:{args.quant_backend}")
    print(f"Bias:         {['no-bias', 'with-bias'] if len(bias_modes) == 2 else ['no-bias']}")
    print(f"Shapes:       {len(DEFAULT_SHAPES)} configurations")
    print(f"fouroversix:  {'available' if FOUROVERSIX_AVAILABLE else 'not installed'}")
    print(f"cutlass DSL:  {'available' if IS_CUTLASS_DSL_AVAILABLE else 'not available'}")

    all_results: List[Dict[str, Any]] = []

    # ---- 1. Correctness across shapes (tllm quant_backend) ----
    if args.quant_backend in ("tllm", "all"):
        print("\n--- Correctness: quant_backend=tllm ---")
        for m, n, k in DEFAULT_SHAPES:
            for use_bias in bias_modes:
                label = f"({m},{n},{k}) gemm={args.gemm_backend} bias={'yes' if use_bias else 'no'}"
                print(f"  {label} ...", end="", flush=True)
                r = check_correctness(
                    m, n, k, dtype, use_bias,
                    gemm_backend=args.gemm_backend,
                    quant_backend="tllm",
                )
                all_results.append(r)
                cos = f"{r['cosine_sim']:.4f}" if not math.isnan(r['cosine_sim']) else "NaN"
                print(f" [{r['status']}] cos={cos}")

    # ---- 2. Structural tests ----
    print("\n--- Structural tests ---")

    print("  from_linear buffers ...", end="", flush=True)
    r = test_from_linear_buffers(dtype)
    all_results.append(r)
    print(f" [{r['status']}]")

    print("  N-D input (2,17,K) ...", end="", flush=True)
    r = test_nd_input(dtype)
    all_results.append(r)
    cos = f"{r['cosine_sim']:.4f}" if isinstance(r.get('cosine_sim'), float) and not math.isnan(r['cosine_sim']) else "NaN"
    print(f" [{r['status']}] cos={cos}")

    print("  state_dict round-trip ...", end="", flush=True)
    r = test_state_dict_roundtrip(dtype)
    all_results.append(r)
    print(f" [{r['status']}]")

    print("  zero input ...", end="", flush=True)
    r = test_zero_input()
    all_results.append(r)
    print(f" [{r['status']}]")

    print("  auto != cuda_core ...", end="", flush=True)
    r = test_auto_no_cuda_core()
    all_results.append(r)
    print(f" [{r['status']}]")

    # ---- 3. fouroversix quant_backend ----
    if args.quant_backend in ("fouroversix", "all"):
        print("\n--- fouroversix quant_backend ---")
        if FOUROVERSIX_AVAILABLE:
            fox_shapes = [(32, 4096, 4096), (128, 4096, 4096)]
            fox_rules = ["mse", "static_6"]
            for m, n, k in fox_shapes:
                for rule in fox_rules:
                    print(f"  ({m},{n},{k}) rule={rule} ...", end="", flush=True)
                    r = test_fouroversix_correctness(m, n, k, dtype, rule)
                    all_results.append(r)
                    cos = f"{r['cosine_sim']:.4f}" if isinstance(r.get('cosine_sim'), float) and not math.isnan(r['cosine_sim']) else "NaN"
                    print(f" [{r['status']}] cos={cos}")
        else:
            print("  [SKIP] fouroversix not installed")
            r = test_fouroversix_unavailable()
            all_results.append(r)
            print(f"  unavailable guard ... [{r['status']}]")

    # ---- Summary ----
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print_summary_table(all_results)

    num_pass = sum(1 for r in all_results if r["status"].startswith("PASS"))
    num_warn = sum(1 for r in all_results if r["status"] == "WARN")
    num_skip = sum(1 for r in all_results if r["status"].startswith("SKIP"))
    num_fail = len(all_results) - num_pass - num_warn - num_skip
    total = len(all_results)

    print(f"\nTotal: {total} tests, {num_pass} PASS, {num_warn} WARN, "
          f"{num_skip} SKIP, {num_fail} FAIL")
    if num_fail == 0:
        print("[PASS] All correctness checks passed")
    else:
        print("[FAIL] Some tests failed, review above")
    print("=" * 80)


if __name__ == "__main__":
    main()
