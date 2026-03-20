#!/usr/bin/env python3
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
test_prologue.py - Prologue (amax → global_scale) Correctness & Performance Test

Compares all available prologue implementations that produce a usable
global_scale value (= quant_range / amax).  Every candidate must pass
correctness before being benchmarked.

Candidates:
  - pytorch:       x.abs().max() baseline (multi-kernel)
  - triton_amax:   Triton two-stage reduction + PyTorch .max()
  - cuda_prologue: CUDA single-kernel last-block reduction (fused amax + scale)
  - f46_prologue:  fouroversix quantize_fp4_prologue (reference, optional)

Usage:
    python tests/test_prologue.py
    python tests/test_prologue.py --shapes 128,6144 256,6144 14400,6144
    python tests/test_prologue.py --warmup 10 --repeat 50 --skip-verify
"""

from __future__ import annotations

import argparse
from typing import Callable, Tuple

import torch

try:
    import nvtx
    _HAS_NVTX = True
except ImportError:
    _HAS_NVTX = False

import tllm_linear_lite  # noqa: F401 — ensure CUDA extension is loaded
from tllm_linear_lite.amax.triton_amax import triton_amax
from tllm_linear_lite.amax.cuda_amax import cuda_prologue


# ============================================================================
# Constants
# ============================================================================

_EPS: float = 1e-12
_QUANT_RANGE: float = 6.0 * 448.0  # 2688, standard NVFP4

DEFAULT_SHAPES = [
    (1, 6144),
    (128, 6144),
    (256, 6144),
    (1024, 6144),
    (4096, 6144),
    (14400, 6144),
]


# ============================================================================
# Measurement
# ============================================================================

def measure_us(fn: Callable, warmup: int, repeat: int, name: str = "") -> float:
    """Warmup + measure average GPU latency in microseconds.

    Each timed iteration is wrapped in an NVTX range for nsys profiling.
    """
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        times: list[float] = []
        for i in range(repeat):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            if _HAS_NVTX:
                nvtx.push_range(f"prologue/{name}/{i}")
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            if _HAS_NVTX:
                nvtx.pop_range()
            times.append(s.elapsed_time(e) * 1000)  # ms → us
    return sum(times) / len(times)


# ============================================================================
# Candidate builders
# ============================================================================

def make_candidates(
    x: torch.Tensor,
    quant_range: float = _QUANT_RANGE,
) -> list[tuple[str, Callable[[], torch.Tensor]]]:
    """Build (name, fn) pairs.  Each fn() returns a scalar global_scale tensor."""
    candidates: list[tuple[str, Callable]] = []

    # --- PyTorch baseline ---
    def _pytorch():
        return quant_range / x.abs().max().float().clamp_min(_EPS)

    candidates.append(("pytorch", _pytorch))

    # --- Triton amax (two-stage: kernel + .max()) ---
    def _triton():
        return quant_range / triton_amax(x).float().clamp_min(_EPS)

    candidates.append(("triton_amax", _triton))

    # --- CUDA single-kernel prologue (fused amax + global_scale) ---
    x_2d = x if x.ndim == 2 else x.reshape(-1, x.shape[-1])

    def _cuda():
        _amax, gs = cuda_prologue(x_2d, quant_range=quant_range, eps=_EPS)
        return gs

    candidates.append(("cuda_prologue", _cuda))

    # --- fouroversix prologue (optional) ---
    try:
        from tllm_linear_lite.quantize import FOUROVERSIX_AVAILABLE

        if FOUROVERSIX_AVAILABLE:
            from fouroversix.quantize.cuda import CUDAQuantizeBackend
            from fouroversix import QuantizationConfig

            f46_cfg = QuantizationConfig()
            if CUDAQuantizeBackend.is_available() and CUDAQuantizeBackend.is_supported(x_2d, f46_cfg):
                def _f46():
                    return CUDAQuantizeBackend.quantize_fp4_prologue(x_2d, f46_cfg)

                candidates.append(("f46_prologue", _f46))
    except Exception:
        pass

    return candidates


# ============================================================================
# Correctness
# ============================================================================

def check_correctness(
    candidates: list[tuple[str, Callable]],
    ref_amax: float,
    ref_gs: float,
) -> tuple[bool, list[dict]]:
    """Verify all candidates produce the correct global_scale.

    Args:
        candidates: List of (name, fn) pairs.
        ref_amax: Reference amax from PyTorch.
        ref_gs: Reference global_scale from PyTorch.

    Returns:
        (all_pass, rows) where rows is a list of dicts with per-candidate details.
    """
    rows: list[dict] = []
    all_pass = True

    for name, fn in candidates:
        row: dict = {"name": name}
        try:
            with torch.no_grad():
                out = fn()
            if isinstance(out, tuple):
                amax_val = out[-1].float().item()
                gs_val = _QUANT_RANGE / max(amax_val, _EPS)
            else:
                gs_val = out.float().item()

            amax_val = _QUANT_RANGE / max(gs_val, _EPS)
            rel_err = abs(gs_val - ref_gs) / max(abs(ref_gs), 1e-30)
            passed = rel_err < 0.01

            row["amax"] = amax_val
            row["global_scale"] = gs_val
            row["rel_err"] = rel_err
            row["status"] = "PASS" if passed else "FAIL"
            if not passed:
                all_pass = False
        except Exception as exc:
            row["amax"] = float("nan")
            row["global_scale"] = float("nan")
            row["rel_err"] = float("nan")
            row["status"] = f"ERR({type(exc).__name__})"
            all_pass = False

        rows.append(row)

    return all_pass, rows


def print_correctness_table(
    ref_amax: float, ref_gs: float, rows: list[dict], name_w: int,
) -> None:
    """Print a formatted accuracy table."""
    print(f"  Reference:  amax = {ref_amax:.6f},  global_scale = {ref_gs:.6f}")
    print()
    hdr = (
        f"  {'Candidate':<{name_w}}"
        f"{'amax':>14} {'global_scale':>14} {'rel_err':>12} {'status':>8}"
    )
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")
    for r in rows:
        n = r["name"]
        if r["status"].startswith("ERR"):
            print(f"  {n:<{name_w}}{'—':>14} {'—':>14} {'—':>12} {r['status']:>8}")
        else:
            print(
                f"  {n:<{name_w}}"
                f"{r['amax']:>14.6f} {r['global_scale']:>14.6f} "
                f"{r['rel_err']:>12.2e} {r['status']:>8}"
            )
    print()


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prologue (amax → global_scale) correctness & benchmark",
    )
    parser.add_argument(
        "--shapes", nargs="+",
        default=[f"{m},{k}" for m, k in DEFAULT_SHAPES],
        help="Shapes as M,K (comma-separated)",
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--skip-verify", action="store_true", help="Skip correctness")
    parser.add_argument(
        "--tune", action="store_true",
        help="Sweep M from 1..14400 at fixed K to find crossover point between cuda_prologue and triton_amax",
    )
    parser.add_argument("--tune-k", type=int, default=6144, help="K for tuning sweep")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    shapes = [tuple(int(d) for d in s.split(",")) for s in args.shapes]

    print("=" * 80)
    print("Prologue (amax → global_scale) Benchmark")
    print("=" * 80)
    print(f"Device:  {torch.cuda.get_device_name(0)}")
    print(f"Dtype:   {dtype}")
    print(f"Warmup:  {args.warmup}    Repeat: {args.repeat}")
    print(f"Shapes:  {shapes}")
    print()

    all_pass = True

    for M, K in shapes:
        x = torch.randn(M, K, device="cuda", dtype=dtype)
        n_elements = x.numel()
        data_bytes = n_elements * x.element_size()
        ref_amax = x.abs().max().float().item()
        ref_gs = (_QUANT_RANGE / max(ref_amax, _EPS))

        print("-" * 80)
        print(
            f"Shape: ({M}, {K})  |  "
            f"Elements: {n_elements:,}  |  "
            f"Data: {data_bytes / 1024:.1f} KB"
        )
        print("-" * 80)

        candidates = make_candidates(x)
        name_w = max(len(n) for n, _ in candidates) + 2

        # --- Correctness ---
        if not args.skip_verify:
            passed, rows = check_correctness(candidates, ref_amax, ref_gs)
            print_correctness_table(ref_amax, ref_gs, rows, name_w)
            if not passed:
                all_pass = False

        # --- Benchmark ---
        hdr = (
            f"  {'Candidate':<{name_w}}"
            f"{'Time(us)':>10} {'BW(GB/s)':>10} {'vs best':>10}"
        )
        print(hdr)
        print(f"  {'-' * (len(hdr) - 2)}")

        timings: list[tuple[str, float]] = []
        for name, fn in candidates:
            try:
                t_us = measure_us(fn, args.warmup, args.repeat, name=name)
                timings.append((name, t_us))
            except Exception as exc:
                print(f"  {name:<{name_w}} ERROR: {exc}")

        if not timings:
            print("  No candidates succeeded.\n")
            continue

        best_us = min(t for _, t in timings)
        for name, t_us in timings:
            bw = data_bytes / (t_us * 1e-6) / 1e9
            ratio = t_us / best_us
            marker = " <<<" if abs(t_us - best_us) < 0.01 else ""
            print(f"  {name:<{name_w}} {t_us:>10.2f} {bw:>10.2f} {ratio:>9.2f}x{marker}")

        print()

    if not args.skip_verify and not args.tune:
        print("=" * 80)
        print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILURES'}")
        print("=" * 80)

    # ==================================================================
    # Tuning sweep: find crossover point between cuda_prologue and triton_amax
    # ==================================================================
    if args.tune:
        _run_tune(args.tune_k, dtype, args.warmup, args.repeat)


def _run_tune(K: int, dtype: torch.dtype, warmup: int, repeat: int) -> None:
    """Sweep M values to find where triton_amax overtakes cuda_prologue."""

    # Sample points: dense near crossover region, sparse at extremes
    m_values = sorted(set(
        [1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, 768, 1024,
         1536, 2048, 3072, 4096, 6144, 8192, 10240, 12288, 14400]
    ))

    qr = _QUANT_RANGE

    print()
    print("=" * 80)
    print(f"Tuning Sweep: M = 1..{m_values[-1]},  K = {K},  dtype = {dtype}")
    print("=" * 80)
    print()
    print(f"  {'M':>7} {'Elements':>12}  "
          f"{'pytorch':>10} {'triton':>10} {'cuda_pro':>10} {'f46_pro':>10}  "
          f"{'winner':<14} {'speedup':>8}")
    print(f"  {'-'*7} {'-'*12}  "
          f"{'-'*10} {'-'*10} {'-'*10} {'-'*10}  "
          f"{'-'*14} {'-'*8}")

    # Try to import f46
    f46_available = False
    try:
        from tllm_linear_lite.quantize import FOUROVERSIX_AVAILABLE
        if FOUROVERSIX_AVAILABLE:
            from fouroversix.quantize.cuda import CUDAQuantizeBackend
            from fouroversix import QuantizationConfig
            f46_available = True
    except Exception:
        pass

    crossover_m: int | None = None
    prev_winner: str = ""

    for M in m_values:
        x = torch.randn(M, K, device="cuda", dtype=dtype)
        n_elems = x.numel()
        x_2d = x

        results: dict[str, float] = {}

        # pytorch
        def _py():
            return qr / x.abs().max().float().clamp_min(_EPS)
        results["pytorch"] = measure_us(_py, warmup, repeat)

        # triton
        def _tr():
            return qr / triton_amax(x).float().clamp_min(_EPS)
        results["triton"] = measure_us(_tr, warmup, repeat)

        # cuda_prologue
        def _cu():
            _a, gs = cuda_prologue(x_2d, quant_range=qr, eps=_EPS)
            return gs
        results["cuda_pro"] = measure_us(_cu, warmup, repeat)

        # f46
        if f46_available:
            f46_cfg = QuantizationConfig()
            if CUDAQuantizeBackend.is_available() and CUDAQuantizeBackend.is_supported(x_2d, f46_cfg):
                def _f46():
                    return CUDAQuantizeBackend.quantize_fp4_prologue(x_2d, f46_cfg)
                results["f46_pro"] = measure_us(_f46, warmup, repeat)

        # Find winner
        winner = min(results, key=results.get)  # type: ignore[arg-type]
        best_us = results[winner]
        second_us = sorted(results.values())[1] if len(results) > 1 else best_us
        speedup = second_us / best_us if best_us > 0 else 1.0

        # Detect crossover between cuda_pro and triton
        if crossover_m is None and prev_winner == "cuda_pro" and winner == "triton":
            crossover_m = M

        def _fmt(name: str) -> str:
            v = results.get(name, float("nan"))
            return f"{v:>10.2f}" if not (v != v) else f"{'—':>10}"

        print(
            f"  {M:>7} {n_elems:>12,}  "
            f"{_fmt('pytorch')} {_fmt('triton')} {_fmt('cuda_pro')} {_fmt('f46_pro')}  "
            f"{winner:<14} {speedup:>7.2f}x"
        )

        prev_winner = winner

    print()
    print("=" * 80)
    if crossover_m is not None:
        print(f"Crossover: cuda_prologue → triton_amax at M ≈ {crossover_m}")
        print(f"Recommendation: use cuda_prologue for M < {crossover_m}, triton_amax for M >= {crossover_m}")
    else:
        print(f"No crossover detected in range M=[{m_values[0]}, {m_values[-1]}].")
        print(f"Winner at M={m_values[-1]}: {prev_winner}")
    print("=" * 80)


if __name__ == "__main__":
    main()
