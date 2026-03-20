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
    (1, 6144, 6144),
    (4, 6144, 6144),
    (8, 6144, 6144),
    (128, 6144, 6144),
    (14400, 6144, 6144),
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
    scale_rule: str = "static_6",
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
        scale_rule: Block scale selection rule (tllm backend only).
        quant_config: fouroversix QuantizationConfig (or None).

    Returns:
        Dict with test metrics for the summary table.
    """
    linear = nn.Linear(k, n, bias=use_bias).to(device="cuda", dtype=dtype)
    x = torch.randn(m, k, device="cuda", dtype=dtype)

    quant_label = quant_backend if scale_rule == "static_6" else f"{quant_backend}/{scale_rule}"
    result: Dict[str, Any] = {
        "M": m, "N": n, "K": k,
        "gemm": gemm_backend, "quant": quant_label,
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
            scale_rule=scale_rule,
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
        "test_name": "from_linear buffers",
        "status": "FAIL",
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
        "test_name": "N-D input (2,17,K)",
        "status": "FAIL",
    }

    linear = nn.Linear(K, N, bias=True).to(device="cuda", dtype=dtype)
    nvfp4 = NVFP4DynamicLinear.from_linear(linear, gemm_backend="cublaslt")
    x = torch.randn(B, S, K, device="cuda", dtype=dtype)

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
    result["note"] = f"cos={cosine_sim:.4f}"
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
        "test_name": "state_dict round-trip",
        "status": "FAIL",
    }

    linear = nn.Linear(K, N, bias=True).to(device="cuda", dtype=dtype)
    nvfp4_orig = NVFP4DynamicLinear.from_linear(linear, gemm_backend="cublaslt")
    x = torch.randn(32, K, device="cuda", dtype=dtype)

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
        "test_name": "zero input",
        "status": "FAIL",
    }

    linear = nn.Linear(K, N, bias=False).to(device="cuda", dtype=torch.bfloat16)
    nvfp4 = NVFP4DynamicLinear.from_linear(linear, gemm_backend="cublaslt")
    x = torch.zeros(M, K, device="cuda", dtype=torch.bfloat16)

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
        "test_name": "auto != cuda_core",
        "status": "FAIL",
    }

    linear = nn.Linear(4096, 4096, bias=False).to(device="cuda", dtype=torch.bfloat16)
    nvfp4 = NVFP4DynamicLinear.from_linear(linear, gemm_backend="auto")

    resolved = nvfp4._resolve_gemm_backend(M=1, N=4096, K=4096)
    if resolved == "cuda_core":
        result["status"] = "FAIL"
        result["note"] = "resolved to cuda_core"
    else:
        result["status"] = "PASS"
        result["note"] = f"resolved={resolved}"
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
        "gemm": "cublaslt", "quant": f"f46/{scale_rule}",
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
        "test_name": "f46/unavailable guard",
        "status": "SKIP",
    }

    if FOUROVERSIX_AVAILABLE:
        result["note"] = "fouroversix installed"
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
# Summary table output
# ============================================================================

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_RESET = "\033[0m"


def _colorize_padded(status: str, width: int) -> str:
    """Pad status to fixed visible width, then wrap with ANSI color."""
    padded = f"{status:<{width}}"
    if status.startswith("PASS"):
        return f"{_GREEN}{padded}{_RESET}"
    if status.startswith("FAIL") or status.startswith("ERROR"):
        return f"{_RED}{padded}{_RESET}"
    if status == "WARN":
        return f"{_YELLOW}{padded}{_RESET}"
    if status.startswith("SKIP"):
        return f"{_CYAN}{padded}{_RESET}"
    return padded


def _fmt_float(v: float) -> str:
    if isinstance(v, float) and not math.isnan(v):
        return f"{v:.4f}"
    return "NaN"


def print_correctness_table(results: List[Dict[str, Any]]):
    """Print correctness test results with metrics.

    Args:
        results: List of result dicts (must have M/N/K/cosine_sim etc).
    """
    if not results:
        return

    header = (
        f"  {'#':>3}  {'status':<6}  {'M':>5} {'N':>5} {'K':>5}"
        f"  {'gemm':<10} {'quant':<14} {'bias':<4}"
        f"  {'cosine':>8} {'mean_err':>9} {'rel_err':>9} {'max_err':>9}"
    )
    sep = "  " + "-" * (len(header) - 2)

    print(f"\n  Correctness Tests")
    print(header)
    print(sep)

    for i, r in enumerate(results, 1):
        status_col = _colorize_padded(r["status"], 6)
        cos = _fmt_float(r.get("cosine_sim", float("nan")))
        mean_e = _fmt_float(r.get("mean_abs_err", float("nan")))
        rel_e = _fmt_float(r.get("mean_rel_err", float("nan")))
        max_e = _fmt_float(r.get("max_abs_err", float("nan")))

        print(
            f"  {i:>3}  {status_col}  {r['M']:>5} {r['N']:>5} {r['K']:>5}"
            f"  {r['gemm']:<10} {r['quant']:<14} {r['bias']:<4}"
            f"  {cos:>8} {mean_e:>9} {rel_e:>9} {max_e:>9}"
        )


def print_structural_table(results: List[Dict[str, Any]], start_idx: int = 1):
    """Print structural test results (pass/fail, with footnotes for notes).

    Args:
        results: List of result dicts (must have test_name).
        start_idx: Starting row number (continues from correctness table).
    """
    if not results:
        return

    notes: list[tuple[int, str]] = []
    note_counter = 0

    header = f"  {'#':>3}  {'status':<8}  {'test'}"
    sep = "  " + "-" * 40

    print(f"\n  Structural Tests (buffer shapes, N-D input, state_dict, numerical stability)")
    print(header)
    print(sep)

    for i, r in enumerate(results, start_idx):
        status = r["status"]
        note = r.get("note")
        suffix = ""
        if note:
            note_counter += 1
            notes.append((note_counter, note))
            suffix = f" [{note_counter}]"

        status_with_suffix = status + suffix
        status_col = _colorize_padded(status_with_suffix, 8)

        print(f"  {i:>3}  {status_col}  {r['test_name']}")

    if notes:
        print()
        for idx, note_text in notes:
            print(f"  [{idx}] {note_text}")


# ============================================================================
# Benchmark
# ============================================================================


def _make_quant_label(quant_backend: str, scale_rule: str, quant_config: object) -> str:
    if quant_backend == "fouroversix" and quant_config is not None:
        return f"f46/{quant_config.scale_rule.value}"
    if scale_rule != "static_6":
        return f"{quant_backend}/{scale_rule}"
    return quant_backend


def _measure_us(fn, warmup: int, repeat: int) -> float:
    """Warmup + measure average latency in microseconds."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        times = []
        for _ in range(repeat):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e) * 1000)
    return sum(times) / len(times)


def benchmark_linear(
    shapes: List[tuple],
    configs: List[Dict[str, Any]],
    dtype: torch.dtype = torch.bfloat16,
    gemm_backend: str = "auto",
    warmup: int = 5,
    repeat: int = 20,
):
    """Benchmark NVFP4DynamicLinear: per-shape unified table with Prologue/Quant/GEMM.

    For each shape, measures activation prologue (amax/scale), quant kernel,
    and GEMM separately across all configs, then prints a unified table.

    For tllm backend: prologue = cuda_prologue (single-kernel amax + scale), quant = fp4_quantize.
    For fouroversix: prologue = quantize_fp4_prologue, quant = quantize_fp4_main.

    Args:
        shapes: List of (M, N, K) tuples.
        configs: List of dicts with keys: label, quant_backend, scale_rule,
                 quant_config (each describes one column in the table).
        dtype: Weight / activation dtype.
        gemm_backend: GEMM backend.
        warmup: Warmup iterations.
        repeat: Timed iterations.
    """
    from tllm_linear_lite.nvfp4_gemm import nvfp4_gemm
    from tllm_linear_lite.amax import cuda_prologue
    from tllm_linear_lite.quantize import (
        fp4_quantize as _fp4_quantize,
        _SCALE_RULE_MAP as _srm,
        _TLLM_QUANT_RANGE as _tqr,
        _FOUROVERSIX_QUANT_RANGE as _fqr,
    )

    _EPS = 1e-12
    col_w = 14
    col_names = [c["label"] for c in configs]
    header = f"  {'Metric':<20}" + "".join(f"{n:>{col_w}}" for n in col_names)
    sep = "  " + "-" * (20 + col_w * len(col_names))

    for m, n, k in shapes:
        print(f"\n  Shape: ({m}, {n}, {k})")
        print(header)
        print(sep)

        linear = nn.Linear(k, n, bias=False).to(device="cuda", dtype=dtype)
        x_2d = torch.randn(m, k, device="cuda", dtype=dtype)

        prologue_us_list: list[float] = []
        quant_us_list: list[float] = []
        gemm_us_list: list[float] = []

        for cfg in configs:
            qb = cfg["quant_backend"]
            sr = cfg.get("scale_rule", "static_6")
            qc = cfg.get("quant_config")

            try:
                nvfp4 = NVFP4DynamicLinear.from_linear(
                    linear, gemm_backend=gemm_backend,
                    quant_backend=qb, scale_rule=sr, quant_config=qc,
                )
            except Exception as e:
                print(f"  [WARN] {cfg['label']}: {e}")
                prologue_us_list.append(float("nan"))
                quant_us_list.append(float("nan"))
                gemm_us_list.append(float("nan"))
                continue

            if qb == "tllm":
                is_adaptive = _srm.get(sr, 0) != 0
                qr = _fqr if is_adaptive else _tqr

                def _prologue():
                    return cuda_prologue(x_2d, quant_range=qr, eps=_EPS)

                prologue_us = _measure_us(_prologue, warmup, repeat)

                with torch.no_grad():
                    act_amax, act_gs = _prologue()

                def _quant():
                    return _fp4_quantize(x_2d, global_scale=act_gs, swizzled=True, scale_rule=sr)

                quant_us = _measure_us(_quant, warmup, repeat)

                with torch.no_grad():
                    act_fp4, act_sf = _quant()

            else:
                from fouroversix.quantize.cuda import CUDAQuantizeBackend
                f46_cfg = qc

                def _prologue():
                    return CUDAQuantizeBackend.quantize_fp4_prologue(x_2d, f46_cfg)

                prologue_us = _measure_us(_prologue, warmup, repeat)

                with torch.no_grad():
                    intermediates = _prologue()

                def _quant():
                    return CUDAQuantizeBackend.quantize_fp4_main(x_2d, intermediates, f46_cfg)

                quant_us = _measure_us(_quant, warmup, repeat)

                with torch.no_grad():
                    qt = _quant()
                    act_fp4 = qt.values
                    act_sf = qt.scale_factors.view(torch.uint8)
                    act_amax = qt.amax.float().clamp_min(_EPS)

            alpha = (act_amax * nvfp4.alpha_multiplier).to(torch.float32)
            be = nvfp4._resolve_gemm_backend(m, n, k)

            def _gemm():
                return nvfp4_gemm(
                    act_fp4, nvfp4.weight_fp4,
                    act_sf, nvfp4.weight_sf,
                    alpha, output_dtype=dtype, backend=be,
                )

            gemm_us = _measure_us(_gemm, warmup, repeat)

            prologue_us_list.append(prologue_us)
            quant_us_list.append(quant_us)
            gemm_us_list.append(gemm_us)

        total_us_list = [p + q + g for p, q, g in
                         zip(prologue_us_list, quant_us_list, gemm_us_list)]
        tllm_total = total_us_list[0] if total_us_list else 1.0
        flops = 2.0 * m * n * k

        def _f(v: float) -> str:
            return f"{v:>{col_w}.2f}" if not math.isnan(v) else f"{'N/A':>{col_w}}"

        print(f"  {'Prologue (us)':<20}" + "".join(_f(t) for t in prologue_us_list))
        print(f"  {'Quant (us)':<20}" + "".join(_f(t) for t in quant_us_list))
        print(f"  {'GEMM (us)':<20}" + "".join(_f(t) for t in gemm_us_list))
        print(f"  {'Total (us)':<20}" + "".join(_f(t) for t in total_us_list))

        tflops = [flops / (t * 1e-6) / 1e12 if not math.isnan(t) else float("nan")
                  for t in total_us_list]
        print(f"  {'TFLOPS':<20}" + "".join(_f(t) for t in tflops))

        if len(configs) > 1:
            print(f"  {'Speedup vs tllm':<20}" + "".join(
                f"{'1.00x':>{col_w}}" if i == 0
                else (f"{tllm_total / t:>{col_w - 1}.2f}x" if not math.isnan(t) else f"{'N/A':>{col_w}}")
                for i, t in enumerate(total_us_list)
            ))


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="NVFP4DynamicLinear End-to-End Test (Correctness + Benchmark)"
    )
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16"])
    parser.add_argument("--gemm-backend", type=str, default="auto",
                        choices=["auto", "cutlass", "cublaslt", "cutedsl"])
    parser.add_argument("--quant-backend", type=str, default="all",
                        choices=["tllm", "fouroversix", "all"])
    parser.add_argument("--no-bias", action="store_true",
                        help="Skip bias tests")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip correctness verification, benchmark only")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Benchmark warmup iterations")
    parser.add_argument("--repeat", type=int, default=20,
                        help="Benchmark timed iterations")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    bias_modes = [False] if args.no_bias else [False, True]

    print("=" * 80)
    print("tllm_linear_lite NVFP4DynamicLinear Test")
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

    # ==================================================================
    # Correctness
    # ==================================================================
    if not args.skip_verify:
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

        # ---- 1b. tllm adaptive scale rules (mse/abs_max/mae) ----
        if args.quant_backend in ("tllm", "all"):
            _ADAPTIVE_SHAPES = [(128, 6144, 6144)]
            _ADAPTIVE_RULES = ("mse", "abs_max", "mae")
            print("\n--- Correctness: quant_backend=tllm, adaptive scale rules ---")
            for rule in _ADAPTIVE_RULES:
                for m, n, k in _ADAPTIVE_SHAPES:
                    label = f"({m},{n},{k}) rule={rule}"
                    print(f"  {label} ...", end="", flush=True)
                    r = check_correctness(
                        m, n, k, dtype, False,
                        gemm_backend=args.gemm_backend,
                        quant_backend="tllm",
                        scale_rule=rule,
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
        note = r.get("note", "")
        print(f" [{r['status']}] {note}")

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
                fox_shapes = [(128, 6144, 6144)]
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

        correctness = [r for r in all_results if "test_name" not in r]
        structural = [r for r in all_results if "test_name" in r]

        print_correctness_table(correctness)
        print_structural_table(structural, start_idx=len(correctness) + 1)

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

    # ==================================================================
    # Benchmark
    # ==================================================================
    print("\n" + "=" * 80)
    print("Performance Benchmark")
    print("=" * 80)

    bench_configs: List[Dict[str, Any]] = []

    if args.quant_backend in ("tllm", "all"):
        bench_configs.append({"label": "tllm", "quant_backend": "tllm", "scale_rule": "static_6"})
        for rule in ("mse", "abs_max", "mae"):
            bench_configs.append({"label": f"tllm/{rule}", "quant_backend": "tllm", "scale_rule": rule})

    if args.quant_backend in ("fouroversix", "all") and FOUROVERSIX_AVAILABLE:
        from fouroversix import QuantizationConfig as _F46Config
        for rule in ("mse", "mae", "static_6"):
            cfg = _F46Config(scale_rule=rule)
            bench_configs.append({
                "label": f"f46/{rule}", "quant_backend": "fouroversix",
                "quant_config": cfg,
            })

    if bench_configs:
        benchmark_linear(
            DEFAULT_SHAPES, bench_configs, dtype,
            gemm_backend=args.gemm_backend,
            warmup=args.warmup, repeat=args.repeat,
        )

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
