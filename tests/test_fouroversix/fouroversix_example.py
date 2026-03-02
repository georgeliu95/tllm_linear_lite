"""
fouroversix_example.py — Standalone FP4 (NVFP4) Quantization Example

Demonstrates direct usage of the fouroversix library for NVFP4 block quantization:
  1. Basic quantize → dequantize round-trip
  2. Multiple scale rule comparison (mse, abs_max, mae, static_6)
  3. Quantization accuracy measurement
  4. Performance benchmark

Prerequisites:
  pip install fouroversix   # or install from source

Usage:
    python fouroversix_example.py
    python fouroversix_example.py --shape 14400,6144 --dtype bfloat16
    python fouroversix_example.py --skip-verify
    python fouroversix_example.py --scale-rules mse,static_6
"""

import argparse

import torch

from fouroversix import (
    QuantizationConfig,
    QuantizedTensor,
    ScaleRule,
    quantize_to_fp4,
)

# Available NVFP4 scale rules:
#   mse       – pick max_e2m1 ∈ {4, 6} per block to minimize mean squared error
#   abs_max   – pick per block to minimize max absolute error
#   mae       – pick per block to minimize mean absolute error
#   static_6  – standard NVFP4: always use max_e2m1=6 (same as TRT-LLM)
#   static_4  – always use max_e2m1=4
DEFAULT_SCALE_RULES = ("mse", "abs_max", "mae", "static_6")


# ============================================================================
# Basic usage demo
# ============================================================================


def basic_usage_demo(input_tensor: torch.Tensor) -> None:
    """Show the simplest quantize → dequantize flow and QuantizedTensor API.

    Args:
        input_tensor: [M, K] BF16/FP16 tensor on CUDA.
    """
    print("\n" + "=" * 80)
    print("1. Basic Usage Demo")
    print("=" * 80)

    # Default config: NVFP4, MSE-based 4/6 scale selection, round-to-nearest
    config = QuantizationConfig()
    print(f"Config: {config}")

    # Quantize
    q: QuantizedTensor = quantize_to_fp4(input_tensor, config=config)

    print(f"\nQuantizedTensor attributes:")
    print(f"  values.shape:        {q.values.shape}  (packed FP4, uint8)")
    print(f"  scale_factors.shape: {q.scale_factors.shape}")
    print(f"  amax:                {q.amax}")
    print(f"  dtype:               {q.dtype}")
    print(f"  scale_rule:          {q.scale_rule}")
    print(f"  original_shape:      {q.original_shape}")
    print(f"  padded_shape:        {q.padded_shape}")

    # Dequantize back to high precision
    dequant = q.dequantize(dtype=torch.bfloat16)
    print(f"\nDequantized: shape={dequant.shape}, dtype={dequant.dtype}")

    # .to(dtype) convenience — equivalent to .dequantize(dtype)
    dequant_via_to = q.to(torch.bfloat16)
    assert torch.equal(dequant, dequant_via_to)
    print("  q.to(torch.bfloat16) == q.dequantize(torch.bfloat16): OK")

    # Round-trip quantization error
    M, K = q.original_shape
    orig = input_tensor[:M, :K].float()
    recon = dequant[:M, :K].float()
    abs_err = (recon - orig).abs()
    print(f"\nRound-trip error (mse scale rule):")
    print(f"  Max abs error:  {abs_err.max().item():.6f}")
    print(f"  Mean abs error: {abs_err.mean().item():.6f}")


# ============================================================================
# Accuracy verification
# ============================================================================


def verify_accuracy(
    input_tensor: torch.Tensor,
    scale_rules: tuple[str, ...] = DEFAULT_SCALE_RULES,
) -> bool:
    """Compare quantize→dequantize accuracy across scale rules.

    For each scale rule, quantizes input_tensor, dequantizes, and measures
    error against the original.

    Args:
        input_tensor: [M, K] BF16/FP16 input on CUDA.
        scale_rules: Scale rules to compare.

    Returns:
        True if all scale rules produce reasonable error levels.
    """
    print("\n" + "=" * 80)
    print("2. Quantization Accuracy (dequant error vs original input)")
    print("=" * 80)
    print(f"Input shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")

    input_f32 = input_tensor.float()
    results: dict[str, dict[str, float]] = {}

    for rule in scale_rules:
        try:
            config = QuantizationConfig(scale_rule=rule)
            q = quantize_to_fp4(input_tensor, config=config)
            dequant = q.dequantize(dtype=torch.bfloat16).float()

            M, K = q.original_shape
            orig = input_f32[:M, :K]
            recon = dequant[:M, :K]

            abs_err = (recon - orig).abs()
            rel_err = abs_err / orig.abs().clamp(min=1e-6)

            results[rule] = {
                "max_abs": abs_err.max().item(),
                "mean_abs": abs_err.mean().item(),
                "max_rel": rel_err.max().item(),
                "mean_rel": rel_err.mean().item(),
            }
        except Exception as e:
            print(f"  [WARN] scale_rule={rule} failed: {e}")

    valid = [r for r in scale_rules if r in results]
    if not valid:
        print("  [FAIL] No scale rules succeeded")
        return False

    # Comparison table
    col_w = 15
    header = "".join(f"{r:>{col_w}}" for r in valid)
    print(f"\n  {'Metric':<25}{header}")
    print(f"  {'-' * (25 + col_w * len(valid))}")

    for label, key, fmt in [
        ("Max abs error", "max_abs", ".6f"),
        ("Mean abs error", "mean_abs", ".6f"),
        ("Max relative error", "max_rel", ".4f"),
        ("Mean relative error", "mean_rel", ".4f"),
    ]:
        vals = "".join(f"{results[r][key]:>{col_w}{fmt}}" for r in valid)
        print(f"  {label:<25}{vals}")

    # Sanity check: mean error should be small relative to input std
    all_pass = True
    input_std = input_f32.std().item()
    print(f"\n  Input std: {input_std:.6f}")
    print(f"  Error ratio (mean_abs_err / input_std):")
    for rule in valid:
        ratio = results[rule]["mean_abs"] / max(input_std, 1e-10)
        ok = ratio < 0.5
        tag = "[PASS]" if ok else "[WARN]"
        print(f"    {rule:<12} {ratio:.4f}  {tag}")
        if not ok:
            all_pass = False

    # Relative improvement of 4/6 rules vs static_6 baseline
    if "static_6" in results and len(valid) > 1:
        baseline = results["static_6"]["mean_abs"]
        print(f"\n  Improvement over static_6 (standard NVFP4):")
        for rule in valid:
            if rule == "static_6":
                continue
            improvement = (1.0 - results[rule]["mean_abs"] / max(baseline, 1e-10)) * 100
            print(f"    {rule:<12} {improvement:+.2f}% mean abs error reduction")

    return all_pass


# ============================================================================
# Performance benchmark
# ============================================================================


def benchmark(
    input_tensor: torch.Tensor,
    scale_rules: tuple[str, ...] = DEFAULT_SCALE_RULES,
    warmup: int = 5,
    repeat: int = 20,
) -> None:
    """Benchmark quantization latency for each scale rule.

    Args:
        input_tensor: [M, K] BF16/FP16 input on CUDA.
        scale_rules: Scale rules to benchmark.
        warmup: Number of warmup iterations.
        repeat: Number of timed iterations.
    """
    print("\n" + "=" * 80)
    print("3. Performance Benchmark")
    print("=" * 80)

    n_elements = input_tensor.numel()
    data_bytes = n_elements * input_tensor.element_size()
    print(f"Input shape:  {input_tensor.shape}")
    print(f"Input dtype:  {input_tensor.dtype}")
    print(f"Elements:     {n_elements:,}")
    print(f"Data size:    {data_bytes / 1024 / 1024:.2f} MB")
    print(f"Warmup: {warmup}, Repeat: {repeat}")

    gpu_name = torch.cuda.get_device_name(0).lower()
    if "b200" in gpu_name or "b100" in gpu_name:
        peak_bw = 8.0
    elif "h100" in gpu_name or "h200" in gpu_name:
        peak_bw = 3.35
    else:
        peak_bw = 2.0

    results: dict[str, dict[str, float]] = {}

    for rule in scale_rules:
        try:
            config = QuantizationConfig(scale_rule=rule)

            for _ in range(warmup):
                _ = quantize_to_fp4(input_tensor, config=config)
            torch.cuda.synchronize()

            times: list[float] = []
            for _ in range(repeat):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _ = quantize_to_fp4(input_tensor, config=config)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end) * 1000)  # ms → us

            avg_us = sum(times) / len(times)
            bw_tbs = data_bytes / (avg_us * 1e-6) / 1e12
            results[rule] = {
                "avg_us": avg_us,
                "bw_tbs": bw_tbs,
                "mbu_pct": bw_tbs / peak_bw * 100,
            }
        except Exception as e:
            print(f"  [WARN] scale_rule={rule} benchmark failed: {e}")

    valid = [r for r in scale_rules if r in results]
    if not valid:
        print("  No benchmarks succeeded")
        return

    col_w = 15
    header = "".join(f"{r:>{col_w}}" for r in valid)
    print(f"\n  {'Metric':<30}{header}")
    print(f"  {'-' * (30 + col_w * len(valid))}")
    print(f"  {'Avg latency (us)':<30}" + "".join(
        f"{results[r]['avg_us']:>{col_w}.2f}" for r in valid
    ))
    print(f"  {'Effective BW (TB/s)':<30}" + "".join(
        f"{results[r]['bw_tbs']:>{col_w}.2f}" for r in valid
    ))
    print(f"  {'MBU %':<30}" + "".join(
        f"{results[r]['mbu_pct']:>{col_w - 1}.1f}%" for r in valid
    ))

    if "static_6" in results and len(valid) > 1:
        baseline = results["static_6"]["avg_us"]
        print(f"\n  Latency ratio vs static_6 (standard NVFP4):")
        for rule in valid:
            ratio = results[rule]["avg_us"] / max(baseline, 1e-10)
            print(f"    {rule:<12} {ratio:.2f}x")


# ============================================================================
# Entry point
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="fouroversix FP4 Quantization Example"
    )
    parser.add_argument(
        "--shape", type=str, default="4096,4096",
        help="Input shape M,K (comma-separated, default: 4096,4096)",
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Input dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--warmup", type=int, default=5,
        help="Warmup iterations for benchmark (default: 5)",
    )
    parser.add_argument(
        "--repeat", type=int, default=20,
        help="Timed iterations for benchmark (default: 20)",
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip accuracy verification, run benchmark only",
    )
    parser.add_argument(
        "--scale-rules", type=str, default=None,
        help="Comma-separated scale rules (default: mse,abs_max,mae,static_6)",
    )
    args = parser.parse_args()

    shape = tuple(map(int, args.shape.split(",")))
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    scale_rules = (
        tuple(args.scale_rules.split(",")) if args.scale_rules
        else DEFAULT_SCALE_RULES
    )

    print("=" * 80)
    print("fouroversix FP4 Quantization Example")
    print("=" * 80)
    print(f"Device:       {torch.cuda.get_device_name(0)}")
    print(f"Shape:        {shape}")
    print(f"Dtype:        {dtype}")
    print(f"Scale rules:  {', '.join(scale_rules)}")
    print(f"fouroversix:  {__import__('fouroversix').__version__}")

    input_tensor = torch.randn(shape, device="cuda", dtype=dtype)

    basic_usage_demo(input_tensor)

    if not args.skip_verify:
        if not verify_accuracy(input_tensor, scale_rules):
            print("\n[WARN] Some accuracy checks had warnings, review above.")

    benchmark(input_tensor, scale_rules, warmup=args.warmup, repeat=args.repeat)

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
