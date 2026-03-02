"""
fouroversix_example.py — FP4 (NVFP4) Quantization + GEMM Example

Demonstrates fouroversix NVFP4 quantization with end-to-end GEMM verification:
  1. Basic quantize → dequantize round-trip (QuantizedTensor API)
  2. Accuracy: input/weight quantization error (SQNR) + FP4 GEMM vs BF16 reference
  3. Performance benchmark (quantize latency per scale rule)

GEMM uses tllm_linear_lite's cuBLASLt backend with fouroversix-quantized tensors.

Prerequisites:
  pip install fouroversix       # or install from source (see README)
  pip install tllm_linear_lite  # for cuBLASLt FP4 GEMM

Usage:
    python fouroversix_example.py
    python fouroversix_example.py --shape 128,4096,4096   # M,N,K
    python fouroversix_example.py --shape 4096,4096        # M,K (N defaults to K)
    python fouroversix_example.py --skip-verify
    python fouroversix_example.py --scale-rules mse,static_6
"""

import argparse
import traceback

import torch

import tllm_linear_lite  # noqa: F401  — registers torch.ops.tllm_linear_lite.*

from fouroversix import (
    QuantizationConfig,
    QuantizedTensor,
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


def _sqnr(signal: torch.Tensor, noise: torch.Tensor) -> float:
    """Signal-to-Quantization-Noise Ratio in dB.

    Args:
        signal: Original tensor.
        noise: Quantization error (reconstructed - original).

    Returns:
        SQNR in dB. Higher is better.
    """
    return (
        10 * torch.log10(
            signal.square().mean() / noise.square().mean().clamp(min=1e-20)
        )
    ).item()


def _dequant_error(
    q: QuantizedTensor, original: torch.Tensor,
) -> tuple[float, float]:
    """Compute SQNR and mean abs error for a single quantized tensor.

    Args:
        q: Quantized tensor from quantize_to_fp4.
        original: Original float tensor (same logical shape as q).

    Returns:
        (sqnr_db, mean_abs_error)
    """
    recon = q.dequantize(dtype=torch.bfloat16).float()
    M, K = q.original_shape
    orig = original[:M, :K].float()
    recon = recon[:M, :K]
    noise = recon - orig
    return _sqnr(orig, noise), noise.abs().mean().item()


def _fouroversix_to_cublaslt_alpha(
    q_a: QuantizedTensor, q_b: QuantizedTensor,
) -> torch.Tensor:
    """Compute cuBLASLt alpha from two fouroversix QuantizedTensors.

    cuBLASLt computes: C = alpha * (A_fp4 @ B_fp4^T) where FP4 values are
    in the E2M1 domain. alpha rescales back to the original value range.

    Formula mirrors fouroversix CUTLASS backend (matmul/cutlass/backend.py).

    Args:
        q_a: Quantized activation.
        q_b: Quantized weight.

    Returns:
        Scalar float32 tensor on the same device.
    """
    return (
        (q_a.amax * q_b.amax)
        / (
            q_a.scale_rule.max_allowed_e2m1_value()
            * q_a.scale_rule.max_allowed_e4m3_value()
            * q_b.scale_rule.max_allowed_e2m1_value()
            * q_b.scale_rule.max_allowed_e4m3_value()
        )
    ).to(torch.float32).reshape(1)


_E2M1_DECODE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _tllm_dequant(
    fp4_packed: torch.Tensor,
    sf_packed: torch.Tensor,
    global_scale: float,
    M: int,
    K: int,
    block_size: int = 16,
) -> torch.Tensor:
    """Dequantize tllm_linear_lite fp4_quantize output (LINEAR sf layout).

    Args:
        fp4_packed: [M, K/2] uint8 packed E2M1x2 from fp4_quantize.
        sf_packed: Flat uint8 tensor (FP8 E4M3 bytes, linear layout).
        global_scale: Global scale value used during quantization.
        M: Number of rows.
        K: Number of columns (original, not packed).
        block_size: NVFP4 block size (16).

    Returns:
        [M, K] float32 dequantized tensor.
    """
    decode = _E2M1_DECODE.to(device=fp4_packed.device)
    packed = fp4_packed.view(M, K // 2)
    low = decode[(packed & 0x0F).long()]
    high = decode[((packed >> 4) & 0x0F).long()]
    fp4_float = torch.stack([low, high], dim=-1).reshape(M, K)

    sf_float = sf_packed.view(torch.float8_e4m3fn).reshape(M, K // block_size).float()
    scale = (sf_float / global_scale).unsqueeze(-1).expand(-1, -1, block_size)
    return fp4_float * scale.reshape(M, K)


def _compute_gemm_error(
    gemm_f32: torch.Tensor, ref_output: torch.Tensor,
) -> dict[str, float]:
    """Compute GEMM error metrics against a reference.

    Args:
        gemm_f32: FP4 GEMM output in float32.
        ref_output: BF16 reference GEMM output in float32.

    Returns:
        Dict with cosine, mean_abs, max_abs, mean_rel.
    """
    err = (gemm_f32 - ref_output).abs()
    cosine = torch.nn.functional.cosine_similarity(
        gemm_f32.flatten().unsqueeze(0),
        ref_output.flatten().unsqueeze(0),
    ).item()
    return {
        "cosine": cosine,
        "mean_abs": err.mean().item(),
        "max_abs": err.max().item(),
        "mean_rel": (err / ref_output.abs().clamp(min=1e-6)).mean().item(),
    }


_BASELINE = "tllm"


def verify_accuracy(
    input_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    scale_rules: tuple[str, ...] = DEFAULT_SCALE_RULES,
) -> bool:
    """Verify quantization + GEMM accuracy across scale rules.

    Uses tllm_linear_lite's standard NVFP4 as an independent baseline for
    cross-validation, then compares each fouroversix scale rule against it.

    For each config (tllm baseline + fouroversix scale rules):
      Step 2a — Quantize input [M,K] and weight [N,K], report SQNR + MAE.
      Step 2b — Run cuBLASLt FP4 GEMM, compare against BF16 reference.

    Args:
        input_tensor: [M, K] BF16/FP16 activation on CUDA.
        weight_tensor: [N, K] BF16/FP16 weight on CUDA.
        scale_rules: fouroversix scale rules to compare.

    Returns:
        True if all configs produce reasonable GEMM cosine similarity.
    """
    M, K = input_tensor.shape
    N = weight_tensor.shape[0]

    print("\n" + "=" * 80)
    print("2. Accuracy Verification")
    print("=" * 80)
    print(f"Input: [{M}, {K}],  Weight: [{N}, {K}],  dtype: {input_tensor.dtype}")

    ref_output = input_tensor.float() @ weight_tensor.float().T  # [M, N]

    quant_results: dict[str, dict[str, float]] = {}
    gemm_results: dict[str, dict[str, float]] = {}

    # ---- tllm baseline (independent standard NVFP4 implementation) ----
    try:
        act_gs = 448.0 * 6.0 / input_tensor.abs().amax().float()
        wt_gs = 448.0 * 6.0 / weight_tensor.abs().amax().float()

        # LINEAR layout for dequant error measurement
        act_fp4_lin, act_sf_lin = torch.ops.tllm_linear_lite.fp4_quantize(
            input_tensor, act_gs, 16, False, False,
        )
        wt_fp4_lin, wt_sf_lin = torch.ops.tllm_linear_lite.fp4_quantize(
            weight_tensor, wt_gs, 16, False, False,
        )
        act_dq = _tllm_dequant(act_fp4_lin, act_sf_lin, act_gs.item(), M, K)
        wt_dq = _tllm_dequant(wt_fp4_lin, wt_sf_lin, wt_gs.item(), N, K)

        in_noise = act_dq - input_tensor.float()
        wt_noise = wt_dq - weight_tensor.float()
        in_sqnr = _sqnr(input_tensor.float(), in_noise)
        wt_sqnr = _sqnr(weight_tensor.float(), wt_noise)
        quant_results[_BASELINE] = {
            "in_sqnr": in_sqnr, "wt_sqnr": wt_sqnr,
            "avg_sqnr": (in_sqnr + wt_sqnr) / 2,
            "in_mae": in_noise.abs().mean().item(),
            "wt_mae": wt_noise.abs().mean().item(),
            "avg_mae": (in_noise.abs().mean().item() + wt_noise.abs().mean().item()) / 2,
        }

        # SWIZZLED layout for cuBLASLt GEMM
        act_fp4, act_sf = torch.ops.tllm_linear_lite.fp4_quantize(
            input_tensor, act_gs, 16, False, True,
        )
        wt_fp4, wt_sf = torch.ops.tllm_linear_lite.fp4_quantize(
            weight_tensor, wt_gs, 16, False, True,
        )
        alpha = torch.tensor(
            [1.0 / (act_gs.item() * wt_gs.item())],
            device="cuda", dtype=torch.float32,
        )
        tllm_gemm = torch.ops.tllm_linear_lite.cublaslt_fp4_gemm(
            act_fp4, wt_fp4, act_sf, wt_sf, alpha, None, torch.bfloat16,
        )
        gemm_results[_BASELINE] = _compute_gemm_error(
            tllm_gemm[:M, :N].float(), ref_output,
        )
    except Exception as e:
        print(f"  [WARN] tllm baseline failed: {e}")
        traceback.print_exc()

    # ---- fouroversix scale rules ----
    for rule in scale_rules:
        try:
            config = QuantizationConfig(scale_rule=rule)
            q_input = quantize_to_fp4(input_tensor, config=config)
            q_weight = quantize_to_fp4(weight_tensor, config=config)

            in_sqnr, in_mae = _dequant_error(q_input, input_tensor)
            wt_sqnr, wt_mae = _dequant_error(q_weight, weight_tensor)
            quant_results[rule] = {
                "in_sqnr": in_sqnr, "wt_sqnr": wt_sqnr,
                "avg_sqnr": (in_sqnr + wt_sqnr) / 2,
                "in_mae": in_mae, "wt_mae": wt_mae,
                "avg_mae": (in_mae + wt_mae) / 2,
            }

            alpha = _fouroversix_to_cublaslt_alpha(q_input, q_weight)
            gemm_out = torch.ops.tllm_linear_lite.cublaslt_fp4_gemm(
                q_input.values, q_weight.values,
                q_input.scale_factors.view(torch.uint8),
                q_weight.scale_factors.view(torch.uint8),
                alpha, None, torch.bfloat16,
            )
            gemm_results[rule] = _compute_gemm_error(
                gemm_out[:M, :N].float(), ref_output,
            )
        except Exception as e:
            print(f"  [WARN] scale_rule={rule} failed: {e}")
            traceback.print_exc()

    # Column ordering: tllm baseline first, then fouroversix rules
    all_q = ([_BASELINE] if _BASELINE in quant_results else []) + \
            [r for r in scale_rules if r in quant_results]
    if not all_q:
        print("  [FAIL] No configs succeeded")
        return False

    # ---- Step 2a: Quantization error table ----
    col_w = 15
    header = "".join(f"{r:>{col_w}}" for r in all_q)
    sep = "-" * (25 + col_w * len(all_q))

    print(f"\n--- Step 2a: Quantization Error (dequant vs original) ---")
    print(f"\n  {'Metric':<25}{header}")
    print(f"  {sep}")
    for label, key, fmt, higher_better in [
        ("Input  SQNR (dB)", "in_sqnr", ".2f", True),
        ("Weight SQNR (dB)", "wt_sqnr", ".2f", True),
        ("Avg    SQNR (dB)", "avg_sqnr", ".2f", True),
        ("Input  mean abs err", "in_mae", ".6f", False),
        ("Weight mean abs err", "wt_mae", ".6f", False),
        ("Avg    mean abs err", "avg_mae", ".6f", False),
    ]:
        best_r = (max if higher_better else min)(all_q, key=lambda r: quant_results[r][key])
        vals = "".join(
            f"{quant_results[r][key]:>{col_w}{fmt}}{'✓' if r == best_r else ' '}"
            for r in all_q
        )
        print(f"  {label:<25}{vals}")

    if _BASELINE in quant_results and len(all_q) > 1:
        baseline_sqnr = quant_results[_BASELINE]["avg_sqnr"]
        print(f"\n  SQNR gain over tllm baseline (standard NVFP4):")
        for col in all_q:
            if col == _BASELINE:
                continue
            gain = quant_results[col]["avg_sqnr"] - baseline_sqnr
            print(f"    {col:<12} {gain:+.2f} dB")

    # ---- Step 2b: GEMM error table ----
    all_g = ([_BASELINE] if _BASELINE in gemm_results else []) + \
            [r for r in scale_rules if r in gemm_results]
    all_pass = True

    if all_g:
        header = "".join(f"{r:>{col_w}}" for r in all_g)
        sep = "-" * (25 + col_w * len(all_g))

        print(f"\n--- Step 2b: GEMM Error (FP4 cuBLASLt vs BF16 reference) ---")
        print(f"  GEMM: [{M},{K}] x [{N},{K}]^T → [{M},{N}]")
        print(f"\n  {'Metric':<25}{header}")
        print(f"  {sep}")
        for label, key, fmt, higher_better in [
            ("Cosine similarity", "cosine", ".6f", True),
            ("Mean abs error", "mean_abs", ".4f", False),
            ("Max abs error", "max_abs", ".4f", False),
            ("Mean relative error", "mean_rel", ".4f", False),
        ]:
            best_r = (max if higher_better else min)(all_g, key=lambda r: gemm_results[r][key])
            vals = "".join(
                f"{gemm_results[r][key]:>{col_w}{fmt}}{'✓' if r == best_r else ' '}"
                for r in all_g
            )
            print(f"  {label:<25}{vals}")

        print(f"\n  Verdict:")
        for col in all_g:
            cos = gemm_results[col]["cosine"]
            ok = cos > 0.9
            tag = "[PASS]" if ok else "[WARN]"
            print(f"    {col:<12} cosine={cos:.6f}  {tag}")
            if not ok:
                all_pass = False
    else:
        print("\n  [WARN] No GEMM results available")
        all_pass = False

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
        help="Shape: M,K or M,N,K (default: 4096,4096 → N=K)",
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

    dims = tuple(map(int, args.shape.split(",")))
    if len(dims) == 2:
        M, K = dims
        N = K
    elif len(dims) == 3:
        M, N, K = dims
    else:
        parser.error("--shape must be M,K or M,N,K")

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    scale_rules = (
        tuple(args.scale_rules.split(",")) if args.scale_rules
        else DEFAULT_SCALE_RULES
    )

    print("=" * 80)
    print("fouroversix FP4 Quantization + GEMM Example")
    print("=" * 80)
    print(f"Device:       {torch.cuda.get_device_name(0)}")
    print(f"Shape:        input=[{M},{K}], weight=[{N},{K}]")
    print(f"Dtype:        {dtype}")
    print(f"Scale rules:  {', '.join(scale_rules)}")
    print(f"fouroversix:  {__import__('fouroversix').__version__}")

    input_tensor = torch.randn(M, K, device="cuda", dtype=dtype)
    weight_tensor = torch.randn(N, K, device="cuda", dtype=dtype)

    basic_usage_demo(input_tensor)

    if not args.skip_verify:
        if not verify_accuracy(input_tensor, weight_tensor, scale_rules):
            print("\n[WARN] Some accuracy checks had warnings, review above.")

    benchmark(input_tensor, scale_rules, warmup=args.warmup, repeat=args.repeat)

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
