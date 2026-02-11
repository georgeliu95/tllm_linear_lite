"""
test_nvfp4_gemm.py - NVFP4 GEMM Correctness and Performance Test

Tests the cuBLASLt and cuda_core NVFP4 GEMM backends against
torch.nn.Linear as the reference baseline (the real user-facing API).

Supports both with-bias and without-bias cases. Bias is currently
applied as a post-GEMM addition (epilogue fusion planned as follow-up).

Usage:
    # Basic test (correctness + performance, both bias modes)
    python tests/test_nvfp4_gemm.py

    # Custom shape
    python tests/test_nvfp4_gemm.py --m 1 --n 4096 --k 4096

    # Test specific backend
    python tests/test_nvfp4_gemm.py --backend cublaslt

    # Without bias only
    python tests/test_nvfp4_gemm.py --no-bias

    # Compare with TensorRT-LLM
    python tests/test_nvfp4_gemm.py --compare-trtllm
"""

import torch
import torch.nn as nn
import argparse
from typing import Tuple, Optional

import tllm_linear_lite  # noqa: F401
from tllm_linear_lite.nvfp4_gemm import nvfp4_gemm


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

    Returns:
        Dictionary with keys:
            linear:       nn.Linear reference module
            input:        [m, k] activation tensor
            act_fp4:      [m, k/2] quantized activation (swizzled SF)
            act_fp4_lin:  [m, k/2] quantized activation (linear SF, for cuda_core)
            act_sf:       activation scale factors (swizzled)
            act_sf_lin:   activation scale factors (linear)
            weight_fp4:   [n, k/2] quantized weight (swizzled SF)
            weight_sf:    weight scale factors (swizzled)
            alpha:        [1] global GEMM scale (float32, device)
            bias:         [n] bias tensor or None
    """
    # Create reference nn.Linear
    linear = nn.Linear(k, n, bias=use_bias).to(device="cuda", dtype=dtype)

    # Input activation
    x = torch.randn(m, k, device="cuda", dtype=dtype)

    weight = linear.weight.data  # [n, k]
    bias = linear.bias.data if use_bias else None

    # Quantize activation (swizzled for cuBLASLt)
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

    # Alpha: inverse of the combined quantization scales
    alpha = torch.tensor(
        [1.0 / (act_global_scale.item() * weight_global_scale.item())],
        device="cuda", dtype=torch.float32,
    )

    return {
        "linear": linear,
        "input": x,
        "act_fp4": act_fp4,
        "act_fp4_lin": act_fp4_lin,
        "act_sf": act_sf,
        "act_sf_lin": act_sf_lin,
        "weight_fp4": weight_fp4,
        "weight_sf": weight_sf,
        "alpha": alpha,
        "bias": bias,
    }


# ============================================================================
# Reference
# ============================================================================


def pytorch_linear_reference(
    x: torch.Tensor, linear: nn.Linear
) -> torch.Tensor:
    """PyTorch nn.Linear reference (full precision, no quantization).

    This is the ground truth that the NVFP4 GEMM should approximate.
    output = x @ weight^T + bias
    """
    with torch.no_grad():
        return linear(x)


# ============================================================================
# Correctness verification
# ============================================================================


def run_nvfp4_gemm(
    data: dict,
    backend: str,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Run NVFP4 GEMM with the specified backend, including bias addition."""

    if backend == "cuda_core":
        # cuda_core uses LINEAR layout for activation scale factors
        gemm_out = nvfp4_gemm(
            data["act_fp4_lin"], data["weight_fp4"],
            data["act_sf_lin"], data["weight_sf"],
            data["alpha"],
            output_dtype=output_dtype, backend="cuda_core",
        )
    else:
        # cuBLASLt uses SWIZZLED layout for both
        gemm_out = nvfp4_gemm(
            data["act_fp4"], data["weight_fp4"],
            data["act_sf"], data["weight_sf"],
            data["alpha"],
            output_dtype=output_dtype, backend=backend,
        )

    # Post-GEMM bias addition (epilogue fusion planned as follow-up)
    if data["bias"] is not None:
        gemm_out = gemm_out + data["bias"]

    return gemm_out


def verify_correctness(
    m: int, n: int, k: int, dtype: torch.dtype, backend: str, use_bias: bool,
) -> bool:
    """Verify NVFP4 GEMM correctness against nn.Linear reference."""
    bias_str = "bias" if use_bias else "no-bias"
    print(f"\n--- Correctness: backend={backend}, ({m},{n},{k}), {dtype}, {bias_str} ---")

    data = create_nvfp4_linear_test_data(m, n, k, dtype, use_bias)

    # nn.Linear reference (full precision)
    ref_output = pytorch_linear_reference(data["input"], data["linear"]).float()

    # NVFP4 GEMM
    try:
        gemm_output = run_nvfp4_gemm(data, backend, dtype).float()
    except Exception as e:
        print(f"  [FAIL] {e}")
        import traceback
        traceback.print_exc()
        return False

    # Compare
    abs_err = (gemm_output - ref_output).abs()
    rel_err = abs_err / (ref_output.abs().clamp(min=1e-6))

    # Cosine similarity: the primary metric for FP4 (very lossy)
    cosine_sim = torch.nn.functional.cosine_similarity(
        gemm_output.flatten().unsqueeze(0),
        ref_output.flatten().unsqueeze(0),
    ).item()

    print(f"  Output shape:      {gemm_output.shape}")
    print(f"  Max abs error:     {abs_err.max().item():.4f}")
    print(f"  Mean abs error:    {abs_err.mean().item():.4f}")
    print(f"  Mean rel error:    {rel_err.mean().item():.4f}")
    print(f"  Cosine similarity: {cosine_sim:.6f}")

    # FP4 GEMM with random data: cosine sim > 0.85 is reasonable
    threshold = 0.85
    if cosine_sim > threshold:
        print(f"  [PASS] Cosine similarity {cosine_sim:.6f} > {threshold}")
        return True
    else:
        print(f"  [FAIL] Cosine similarity {cosine_sim:.6f} < {threshold}")
        return False


# ============================================================================
# Benchmark
# ============================================================================


def benchmark_gemm(
    m: int, n: int, k: int, dtype: torch.dtype, backend: str,
    use_bias: bool = True, warmup: int = 10, repeat: int = 50,
):
    """Benchmark NVFP4 GEMM (including post-GEMM bias addition)."""
    bias_str = "bias" if use_bias else "no-bias"
    print(f"\n--- Benchmark: backend={backend}, ({m},{n},{k}), {bias_str} ---")

    data = create_nvfp4_linear_test_data(m, n, k, dtype, use_bias)

    # Warmup
    for _ in range(warmup):
        _ = run_nvfp4_gemm(data, backend, dtype)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = run_nvfp4_gemm(data, backend, dtype)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # us

    avg_us = sum(times) / len(times)

    # TFLOPS (2*M*N*K for matmul + M*N for bias add)
    flops = 2.0 * m * n * k
    if use_bias:
        flops += m * n
    tflops = flops / (avg_us * 1e-6) / 1e12

    print(f"  Avg time:  {avg_us:.2f} us")
    print(f"  TFLOPS:    {tflops:.2f}")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="NVFP4 GEMM Test (nn.Linear baseline)")
    parser.add_argument("--m", type=int, default=1, help="M dimension (batch * seq_len)")
    parser.add_argument("--n", type=int, default=4096, help="N dimension (out_features)")
    parser.add_argument("--k", type=int, default=4096, help="K dimension (in_features)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16"])
    parser.add_argument("--backend", type=str, default="all",
                        choices=["all", "cublaslt", "cuda_core", "auto"])
    parser.add_argument("--no-bias", action="store_true", help="Skip bias tests")
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--compare-trtllm", action="store_true",
                        help="Compare with TensorRT-LLM nvfp4_gemm")
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    bias_modes = [False] if args.no_bias else [False, True]

    print("=" * 80)
    print("tllm_linear_lite NVFP4 GEMM Test (nn.Linear baseline)")
    print("=" * 80)
    print(f"Device:   {torch.cuda.get_device_name(0)}")
    print(f"Shape:    M={args.m}, N={args.n}, K={args.k}")
    print(f"Dtype:    {dtype}")
    print(f"Bias:     {['no-bias', 'with-bias'] if len(bias_modes) == 2 else bias_modes}")

    # Determine backends
    if args.backend == "all":
        backends = ["cublaslt"]
        if args.m <= 16:
            backends.append("cuda_core")
    else:
        backends = [args.backend]

    all_pass = True

    # --- Correctness ---
    if not args.skip_verify:
        for use_bias in bias_modes:
            for backend in backends:
                ok = verify_correctness(args.m, args.n, args.k, dtype, backend, use_bias)
                all_pass = all_pass and ok

        if not all_pass:
            print("\n[FAIL] Some correctness checks failed.")

    # --- Benchmark ---
    for use_bias in bias_modes:
        for backend in backends:
            benchmark_gemm(args.m, args.n, args.k, dtype, backend, use_bias,
                           warmup=args.warmup, repeat=args.repeat)

    # --- Multi-shape sweep ---
    if args.backend == "all" and not args.skip_verify:
        print("\n" + "=" * 80)
        print("Multi-shape correctness sweep")
        print("=" * 80)
        shapes = [
            (1, 4096, 4096),
            (4, 4096, 4096),
            (8, 8192, 4096),
            (16, 4096, 8192),
            (128, 4096, 4096),
            (512, 4096, 4096),
        ]
        for m_s, n_s, k_s in shapes:
            for backend in ["cublaslt"] + (["cuda_core"] if m_s <= 16 else []):
                for use_bias in bias_modes:
                    ok = verify_correctness(m_s, n_s, k_s, dtype, backend, use_bias)
                    all_pass = all_pass and ok

    # --- TRT-LLM comparison ---
    if args.compare_trtllm:
        print("\n" + "=" * 80)
        print("Cross-validation with TensorRT-LLM")
        print("=" * 80)
        try:
            import tensorrt_llm  # noqa: F401
            # TODO: implement bit-exact comparison with trtllm.nvfp4_gemm
            print("[SKIP] TRT-LLM comparison not yet implemented for GEMM")
        except ImportError:
            print("[SKIP] tensorrt_llm not installed")

    print("\n" + "=" * 80)
    print("[PASS] All tests passed" if all_pass else "[WARN] Some tests had issues")
    print("=" * 80)


if __name__ == "__main__":
    main()
