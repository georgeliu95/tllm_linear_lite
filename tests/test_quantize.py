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
test_quantize.py - tllm_linear_lite FP4 Quantization Test

Goals:
1. Correctness: Compare CUDA kernel output against a pure PyTorch NVFP4 reference
2. Performance: Measure fp4_quantize kernel latency and memory bandwidth
3. Cross-validation (optional): Bit-exact comparison with TensorRT-LLM's trtllm.fp4_quantize
4. fouroversix (optional): If installed, report fouroversix quantize->dequant accuracy vs input

Usage:
    # Basic test (correctness + performance)
    python test_quantize.py

    # Custom shape and dtype
    python test_quantize.py --shape 14400,6144 --dtype bfloat16

    # Skip correctness, benchmark only
    python test_quantize.py --skip-verify

    # Compare with TensorRT-LLM (requires tensorrt_llm installed)
    python test_quantize.py --compare-trtllm

    # NCU profiling
    ncu --set full -o quantize_report python test_quantize.py --ncu
"""

import torch
import argparse
from typing import Tuple

import nvtx

# Load tllm_linear_lite CUDA extension
import tllm_linear_lite  # noqa: F401
from tllm_linear_lite.quantize import FOUROVERSIX_AVAILABLE, fp4_quantize as fp4_quantize_unified
from tllm_linear_lite.amax.triton_amax import triton_amax


# ============================================================================
# E2M1 (FP4) lookup tables and helpers
# ============================================================================

# E2M1 encoding: 1 sign + 2 exponent + 1 mantissa
# Positive representable values (4-bit index -> float):
#   0b0000 -> +0.0    0b0100 -> +2.0
#   0b0001 -> +0.5    0b0101 -> +3.0
#   0b0010 -> +1.0    0b0110 -> +4.0
#   0b0011 -> +1.5    0b0111 -> +6.0
E2M1_POSITIVE_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)

# Full 4-bit decode table (index 0..15 -> float value)
E2M1_DECODE_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def round_to_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round float tensor to nearest E2M1 representable value (with sign).

    This simulates the PTX instruction: cvt.rn.satfinite.e2m1x2.f32

    Args:
        x: Float tensor, values should already be scaled to E2M1 range [-6, 6].

    Returns:
        Tensor with values rounded to nearest E2M1 representable value.
    """
    e2m1 = E2M1_POSITIVE_VALUES.to(device=x.device)
    sign = x.sign()
    abs_x = x.abs().clamp(max=6.0)  # satfinite: clamp to max E2M1 value

    # Find nearest E2M1 value: broadcast [*, 1] vs [8]
    diffs = (abs_x.unsqueeze(-1) - e2m1).abs()  # [*, 8]
    nearest_idx = diffs.argmin(dim=-1)
    nearest_val = e2m1[nearest_idx]

    return sign * nearest_val


def unpack_fp4_bytes(packed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unpack uint8 tensor containing E2M1x2 packed values.

    Each byte contains two 4-bit E2M1 values:
      - low nibble  (bits 0-3): first FP4 value
      - high nibble (bits 4-7): second FP4 value

    Args:
        packed: uint8 tensor of shape [*, N/2] (N/2 bytes for N FP4 values)

    Returns:
        (val_low, val_high): two float tensors, each [*, N/2]
    """
    decode = E2M1_DECODE_TABLE.to(device=packed.device)
    low_nibble = (packed & 0x0F).long()
    high_nibble = ((packed >> 4) & 0x0F).long()
    return decode[low_nibble], decode[high_nibble]


def dequantize_fp4(
    fp4_float: torch.Tensor,
    sf_float: torch.Tensor,
    global_scale: float,
    block_size: int = 16,
) -> torch.Tensor:
    """Dequantize FP4 (E2M1) values back to float.

    Shared by both PyTorch reference and CUDA kernel output paths.

    Formula: dequant_value = fp4_value * sf / global_scale

    Args:
        fp4_float: [M, K] float tensor of E2M1 representable values
        sf_float:  [M, K//block_size] float tensor of scale factors
        global_scale: the global scale value used during quantization
        block_size: quantization block size (default 16)

    Returns:
        [M, K] dequantized float tensor
    """
    M, K = fp4_float.shape
    dequant_scale = (sf_float / global_scale).unsqueeze(-1).expand(-1, -1, block_size)
    return fp4_float * dequant_scale.reshape(M, K)


def unpack_cuda_fp4_output(
    fp4_packed: torch.Tensor,
    sf_packed: torch.Tensor,
    M: int,
    K: int,
    block_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unpack CUDA kernel's packed output into float tensors.

    NOTE: Assumes LINEAR scale factor layout (isSfSwizzledLayout=False).

    Args:
        fp4_packed: uint8 tensor [M, K/2] from fp4_quantize
        sf_packed:  uint8 tensor [M * K/block_size] (FP8 E4M3 as bytes, linear layout)
        M: number of rows
        K: number of columns (original)
        block_size: quantization block size (default 16)

    Returns:
        (fp4_float, sf_float):
            fp4_float: [M, K] float tensor of E2M1 values
            sf_float:  [M, K//block_size] float tensor of scale factors
    """
    # Unpack FP4: each byte -> two E2M1 float values
    val_low, val_high = unpack_fp4_bytes(fp4_packed.view(M, K // 2))
    fp4_float = torch.stack([val_low, val_high], dim=-1).reshape(M, K)

    # Decode scale factors: uint8 -> FP8 E4M3 -> float
    sf_float = sf_packed.view(torch.float8_e4m3fn).reshape(M, K // block_size).float()

    return fp4_float, sf_float


def fp4_quantize_reference(
    input_tensor: torch.Tensor,
    global_scale: float,
    block_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch reference: NVFP4 block quantization (fake quantize).

    Computes block scale factors and rounds values to E2M1 representable values,
    returning them as float tensors (not packed).

    Args:
        input_tensor: [M, K] fp16/bf16 input
        global_scale: precomputed = 448 * 6 / tensor_amax
        block_size: NVFP4 block size (16)

    Returns:
        (fp4_float, sf_float):
            fp4_float: [M, K] float, E2M1 representable values (scaled domain)
            sf_float:  [M, K//block_size] float, scale factors (FP8 E4M3 rounded)
    """
    orig_shape = input_tensor.shape
    x = input_tensor.float().reshape(-1, orig_shape[-1])
    M, K = x.shape
    assert K % block_size == 0

    # Reshape into blocks: [M, num_blocks, block_size]
    x_blocks = x.reshape(M, K // block_size, block_size)

    # Block-wise abs max
    block_max = x_blocks.abs().amax(dim=-1)  # [M, num_blocks]

    # Scale factor = global_scale * (block_max / 6.0), rounded to FP8 E4M3
    sf_raw = global_scale * (block_max / 6.0)
    sf_fp8 = sf_raw.to(torch.float8_e4m3fn)
    sf_float = sf_fp8.float()  # [M, num_blocks]

    # Output scale per block: global_scale / float(fp8_sf)
    # Kernel uses reciprocal_approximate_ftz which is approximately the same
    output_scale = torch.where(
        block_max > 0,
        global_scale / sf_float,
        torch.zeros_like(sf_float),
    )  # [M, num_blocks]

    # Scale each block's values, then round to nearest E2M1
    scaled = x_blocks * output_scale.unsqueeze(-1)  # [M, num_blocks, block_size]
    fp4_float = round_to_e2m1(scaled).reshape(M, K)  # [M, K]

    return fp4_float, sf_float.reshape(M, -1)


# ============================================================================
# CUDA kernel wrapper
# ============================================================================


def quantize(
    input_tensor: torch.Tensor,
    swizzled: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FP4 quantization using the tllm_linear_lite standalone CUDA op.

    Args:
        input_tensor: Input tensor (BF16/FP16)
        swizzled: Whether to use swizzled SF layout

    Returns:
        act_fp4: Quantized FP4 tensor (packed uint8)
        act_sf: Scale factor tensor (FP8 E4M3 as uint8)
    """
    act_global_scale = 448.0 * 6.0 / input_tensor.abs().amax().float()
    scaling_vector_size = 16
    act_fp4, act_sf = torch.ops.tllm_linear_lite.fp4_quantize(
        input_tensor, act_global_scale, scaling_vector_size, False, swizzled
    )
    return act_fp4, act_sf


def measure_scale_computation(
    input_tensor: torch.Tensor,
    quant_range: float = 448.0 * 6.0,
    warmup: int = 3,
    repeat: int = 10,
) -> Tuple[float, float]:
    """Measure the time for global scale computation (triton_amax + division).

    Uses triton_amax (Triton partial reduction + PyTorch .max() final) instead
    of the naive abs().amax() path.

    Args:
        input_tensor: Input tensor.
        quant_range: Numerator for scale computation (default: 2688 for NVFP4).
        warmup: Number of warmup iterations.
        repeat: Number of timed iterations.

    Returns:
        (act_global_scale, avg_time_us)
    """
    with nvtx.annotate("scale_computation/warmup"):
        for _ in range(warmup):
            _ = quant_range / triton_amax(input_tensor)
        torch.cuda.synchronize()

    times = []
    with nvtx.annotate("scale_computation/timed"):
        for i in range(repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            rng = nvtx.start_range(f"scale_computation/iter_{i}")
            start.record()
            act_global_scale = quant_range / triton_amax(input_tensor)
            end.record()
            nvtx.end_range(rng)

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000)  # ms -> us

    avg_time_us = sum(times) / len(times)
    return act_global_scale, avg_time_us


def run_quantize_kernel(
    input_tensor: torch.Tensor,
    act_global_scale: float,
    warmup: int = 3,
    repeat: int = 10,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor], float]:
    """Measure only the fp4_quantize kernel time.

    Args:
        input_tensor: Input tensor
        act_global_scale: Precomputed global scale
        warmup: Number of warmup iterations
        repeat: Number of timed iterations

    Returns:
        ((act_fp4, act_sf), avg_time_us)
    """
    scaling_vector_size = 16

    with nvtx.annotate("tllm_fp4_quantize/warmup"):
        for _ in range(warmup):
            _ = torch.ops.tllm_linear_lite.fp4_quantize(
                input_tensor, act_global_scale, scaling_vector_size, False, True
            )
        torch.cuda.synchronize()

    times = []
    with nvtx.annotate("tllm_fp4_quantize/timed"):
        for i in range(repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            rng = nvtx.start_range(f"tllm_fp4_quantize/iter_{i}")
            start.record()
            act_fp4, act_sf = torch.ops.tllm_linear_lite.fp4_quantize(
                input_tensor, act_global_scale, scaling_vector_size, False, True
            )
            end.record()
            nvtx.end_range(rng)

            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000)  # ms -> us

    avg_time_us = sum(times) / len(times)
    return (act_fp4, act_sf), avg_time_us


def test_accuracy(input_tensor: torch.Tensor) -> bool:
    """Test quantization accuracy of all backends against original input.

    Compares dequantized outputs from tllm and fouroversix (multiple scale rules)
    against the original input, using PyTorch fake-quantize as reference baseline.
    All backends are shown in one unified table.

    Args:
        input_tensor: [M, K] BF16/FP16 input tensor on CUDA.

    Returns:
        True if all tllm checks pass. fouroversix results are informational only.
    """
    print("\n" + "=" * 80)
    print("Accuracy (dequant error vs original input)")
    print("=" * 80)

    try:
        M = (
            input_tensor.shape[0]
            if input_tensor.dim() == 2
            else input_tensor.reshape(-1, input_tensor.shape[-1]).shape[0]
        )
        K = input_tensor.shape[-1]
        block_size = 16
        global_scale_val = (448.0 * 6.0 / input_tensor.abs().amax().float()).item()
        input_flat = input_tensor.float().reshape(M, K)

        print(f"\nInput shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
        print(f"Global scale: {global_scale_val:.4f}")

        # ================================================================
        # Run all backends, collect dequant error vs original input
        # ================================================================

        # backends[name] = {"err": abs_error, "rel_err": relative_error}
        backends: dict[str, dict[str, torch.Tensor]] = {}

        # --- PyTorch reference (golden baseline) ---
        ref_fp4_float, ref_sf = fp4_quantize_reference(
            input_tensor, global_scale_val, block_size
        )
        ref_fp4_float = ref_fp4_float.to(input_tensor.device)
        ref_sf = ref_sf.to(input_tensor.device)
        ref_dequant = dequantize_fp4(
            ref_fp4_float, ref_sf, global_scale_val, block_size
        )
        ref_err = (ref_dequant - input_flat).abs()
        backends["PyTorch Ref"] = {
            "err": ref_err,
            "rel_err": ref_err / input_flat.abs().clamp(min=1e-6),
        }

        # --- tllm CUDA kernel (linear layout for comparison) ---
        act_fp4, act_sf = quantize(input_tensor, swizzled=False)
        kernel_fp4_float, kernel_sf = unpack_cuda_fp4_output(
            act_fp4, act_sf, M, K, block_size
        )
        kernel_dequant = dequantize_fp4(
            kernel_fp4_float, kernel_sf, global_scale_val, block_size
        )
        kernel_err = (kernel_dequant - input_flat).abs()
        backends["tllm"] = {
            "err": kernel_err,
            "rel_err": kernel_err / input_flat.abs().clamp(min=1e-6),
        }

        # --- tllm adaptive 4/6 (native CUDA, streaming kernel) ---
        _ADAPTIVE_RULES = ("mse", "mae", "abs_max")
        for rule in _ADAPTIVE_RULES:
            try:
                adapt_fp4, adapt_sf = fp4_quantize_unified(
                    input_tensor, backend="tllm", swizzled=False, scale_rule=rule,
                )
                adapt_fp4_float, adapt_sf_float = unpack_cuda_fp4_output(
                    adapt_fp4, adapt_sf, M, K, block_size
                )
                # FourOverSix uses encode_scale=1536/amax
                adapt_gs = (6.0 * 256.0 / input_tensor.abs().amax().float()).item()
                adapt_dequant = dequantize_fp4(
                    adapt_fp4_float, adapt_sf_float, adapt_gs, block_size
                )
                adapt_err = (adapt_dequant - input_flat).abs()
                backends[f"tllm({rule})"] = {
                    "err": adapt_err,
                    "rel_err": adapt_err / input_flat.abs().clamp(min=1e-6),
                }
            except Exception as e:
                print(f"\n  [WARN] tllm adaptive ({rule}) failed: {e}")

        # --- fouroversix (multiple scale rules) ---
        _F46_SCALE_RULES = ("mse", "abs_max", "mae")
        if FOUROVERSIX_AVAILABLE:
            from fouroversix import QuantizationConfig as F46Config

            for rule in _F46_SCALE_RULES:
                try:
                    config = F46Config(scale_rule=rule)
                    x_f46 = fp4_quantize_unified(
                        input_tensor, backend="fouroversix", config=config
                    )
                    f46_dequant = x_f46.dequantize(dtype=torch.bfloat16).float()
                    if f46_dequant.shape != input_flat.shape:
                        f46_dequant = f46_dequant[
                            : input_flat.shape[0], : input_flat.shape[1]
                        ]
                    err = (f46_dequant - input_flat).abs()
                    backends[f"f46({rule})"] = {
                        "err": err,
                        "rel_err": err / input_flat.abs().clamp(min=1e-6),
                    }
                except Exception as e:
                    print(f"\n  [WARN] fouroversix ({rule}) failed: {e}")
        else:
            print("\n  fouroversix: [SKIP] not installed")

        # ================================================================
        # Bit-exact sub-check: tllm vs PyTorch reference
        # ================================================================
        sf_match = torch.equal(kernel_sf, ref_sf)
        fp4_match = torch.equal(kernel_fp4_float, ref_fp4_float)

        print(f"\n  Bit-exact check (tllm vs PyTorch ref):")
        if sf_match:
            print(f"    Scale factors: [PASS] exact match")
        else:
            n_diff = (kernel_sf != ref_sf).sum().item()
            sf_max_diff = (kernel_sf - ref_sf).abs().max().item()
            tag = "PASS" if sf_max_diff < 1e-3 else "WARN"
            print(f"    Scale factors: [{tag}] {n_diff}/{kernel_sf.numel()} differ (max_diff={sf_max_diff:.6f})")

        fp4_mismatch_pct = 0.0
        if fp4_match:
            print(f"    FP4 values:    [PASS] exact match")
        else:
            n_diff = (kernel_fp4_float != ref_fp4_float).sum().item()
            fp4_mismatch_pct = 100 * n_diff / kernel_fp4_float.numel()
            # FP4 rounding near decision boundaries + reciprocal_approximate_ftz
            # can cause ~0.1% of values to differ by one E2M1 step
            tag = "PASS" if fp4_mismatch_pct < 1.0 else "WARN"
            print(f"    FP4 values:    [{tag}] {n_diff}/{kernel_fp4_float.numel()} differ ({fp4_mismatch_pct:.4f}%)")

        # ================================================================
        # Unified accuracy table
        # ================================================================
        names = list(backends.keys())
        col_w = 14
        header = f"  {'Metric':<25}" + "".join(f"{n:>{col_w}}" for n in names)
        sep = f"  {'-' * (25 + col_w * len(names))}"

        def _fmt(v: float, w: int = col_w) -> str:
            return f"{v:>{w}.6f}"

        def _fmt_rel(v: float, w: int = col_w) -> str:
            return f"{v:>{w}.4f}"

        print(f"\n{header}")
        print(sep)
        print(f"  {'Max abs error':<25}" + "".join(
            _fmt(backends[n]["err"].max().item()) for n in names
        ))
        print(f"  {'Mean abs error':<25}" + "".join(
            _fmt(backends[n]["err"].mean().item()) for n in names
        ))
        print(f"  {'Max relative error':<25}" + "".join(
            _fmt_rel(backends[n]["rel_err"].max().item()) for n in names
        ))
        print(f"  {'Mean relative error':<25}" + "".join(
            _fmt_rel(backends[n]["rel_err"].mean().item()) for n in names
        ))

        # Error ratio vs PyTorch Ref (mean abs error)
        ref_mean = max(backends["PyTorch Ref"]["err"].mean().item(), 1e-10)
        print(f"\n  Error ratio vs PyTorch Ref (mean abs error):")
        tllm_ratio_ok = True
        for n in names:
            if n == "PyTorch Ref":
                continue
            ratio = backends[n]["err"].mean().item() / ref_mean
            ok = ratio < 2.0
            tag = "PASS" if ok else "WARN"
            print(f"    {n:<20} {ratio:.4f}  [{tag}]")
            if not ok and n == "tllm":
                tllm_ratio_ok = False

        # ================================================================
        # calculate_nvfp4_global_scale sub-check
        # ================================================================
        print(f"\n  calculate_nvfp4_global_scale:")

        per_token_scale = torch.ops.tllm_linear_lite.calculate_nvfp4_global_scale(
            input_tensor, None
        )
        ref_per_token = 448.0 * 6.0 / input_flat.abs().amax(dim=-1, keepdim=True)
        kernel_pts = per_token_scale.float().reshape(M, 1)

        pts_diff = (kernel_pts - ref_per_token).abs()
        pts_max_diff = pts_diff.max().item()

        assert not torch.isnan(per_token_scale).any(), "Per-token scale contains NaN"
        assert not torch.isinf(per_token_scale).any(), "Per-token scale contains Inf"

        pts_pass = pts_max_diff < 1e-2
        tag = "PASS" if pts_pass else "WARN"
        print(f"    [{tag}] max_diff={pts_max_diff:.6f}")

        # ================================================================
        # Summary: only tllm checks determine pass/fail
        # ================================================================
        sf_max_diff = 0.0 if sf_match else (kernel_sf - ref_sf).abs().max().item()
        all_pass = (
            tllm_ratio_ok
            and (sf_match or sf_max_diff < 1e-3)
            and fp4_mismatch_pct < 1.0
            and pts_pass
        )

        print("\n" + "=" * 80)
        if all_pass:
            print("[PASS] All accuracy checks passed")
        else:
            print("[WARN] Some checks have warnings, review above")
        print("=" * 80)
        return all_pass

    except Exception as e:
        print(f"\n[FAIL] Accuracy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_with_trtllm(input_tensor: torch.Tensor) -> bool:
    """Cross-validate against TensorRT-LLM's fp4_quantize (bit-exact comparison)."""
    print("\n" + "=" * 80)
    print("Cross-Validation: tllm_linear_lite vs TensorRT-LLM")
    print("=" * 80)

    try:
        import tensorrt_llm  # noqa: F401
    except ImportError:
        print("\n[SKIP] tensorrt_llm not installed, skipping comparison.")
        return True

    try:
        act_global_scale = 448.0 * 6.0 / input_tensor.abs().amax().float()
        sf_vec_size = 16

        # tllm_linear_lite version
        fp4_ours, sf_ours = torch.ops.tllm_linear_lite.fp4_quantize(
            input_tensor, act_global_scale, sf_vec_size, False, True
        )

        # TensorRT-LLM version
        fp4_trtllm, sf_trtllm = torch.ops.trtllm.fp4_quantize(
            input_tensor, act_global_scale, sf_vec_size, False, True
        )

        # Compare FP4 output
        fp4_match = torch.equal(fp4_ours, fp4_trtllm)
        if fp4_match:
            print("\n[PASS] FP4 outputs match exactly")
        else:
            diff_count = (fp4_ours != fp4_trtllm).sum().item()
            total_count = fp4_ours.numel()
            pct = 100 * diff_count / total_count
            print(f"\n[WARN] FP4 outputs differ: {diff_count}/{total_count} ({pct:.4f}%)")

        # Compare scale factors
        sf_match = torch.equal(sf_ours, sf_trtllm)
        if sf_match:
            print("[PASS] Scale factors match exactly")
        else:
            diff_count = (sf_ours != sf_trtllm).sum().item()
            total_count = sf_ours.numel()
            pct = 100 * diff_count / total_count
            print(f"[WARN] Scale factors differ: {diff_count}/{total_count} ({pct:.4f}%)")

        if fp4_match and sf_match:
            print("\n[PASS] tllm_linear_lite output is bit-exact with TensorRT-LLM")
        return fp4_match and sf_match

    except Exception as e:
        print(f"\n[FAIL] Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def benchmark(
    input_tensor: torch.Tensor,
    warmup: int = 5,
    repeat: int = 20,
):
    """Benchmark quantization performance: tllm as baseline, fouroversix variants compared.

    Measures kernel latency, effective bandwidth, and MBU% for tllm and each
    fouroversix scale rule (mse, abs_max, mae) in one unified table.

    Args:
        input_tensor: Input tensor (BF16/FP16, CUDA).
        warmup: Number of warmup iterations.
        repeat: Number of timed iterations.
    """
    print("\n" + "=" * 80)
    print("Performance Benchmark")
    print("=" * 80)

    n_elements = input_tensor.numel()
    data_bytes = n_elements * input_tensor.element_size()

    print(f"Input shape:  {input_tensor.shape}")
    print(f"Input dtype:  {input_tensor.dtype}")
    print(f"Elements:     {n_elements:,}")
    print(f"Data size:    {data_bytes / 1024 / 1024:.2f} MB")
    print(f"Warmup: {warmup}, Repeat: {repeat}")

    # GPU peak bandwidth for MBU calculation (B200 = 8 TB/s, H100 SXM = 3.35 TB/s)
    gpu_name = torch.cuda.get_device_name(0).lower()
    if "b200" in gpu_name or "b100" in gpu_name:
        peak_bw = 8.0
    elif "h100" in gpu_name or "h200" in gpu_name:
        peak_bw = 3.35
    else:
        peak_bw = 2.0  # conservative estimate

    # ================================================================
    # tllm baseline: prologue (triton_amax) + kernel, measured both ways
    # ================================================================
    from tllm_linear_lite.quantize import (
        _SCALE_RULE_MAP, _TLLM_QUANT_RANGE, _FOUROVERSIX_QUANT_RANGE,
    )

    with nvtx.annotate("benchmark/scale_computation"):
        act_global_scale, scale_time_us = measure_scale_computation(
            input_tensor, quant_range=_TLLM_QUANT_RANGE,
            warmup=warmup, repeat=repeat,
        )

    with nvtx.annotate("benchmark/tllm_fp4_quantize"):
        (act_fp4, act_sf), tllm_kernel_us = run_quantize_kernel(
            input_tensor, act_global_scale, warmup=warmup, repeat=repeat
        )

    tllm_e2e_us = scale_time_us + tllm_kernel_us

    output_bytes = (
        act_fp4.numel() * act_fp4.element_size()
        + act_sf.numel() * act_sf.element_size()
    )
    total_bytes = data_bytes + output_bytes

    # ================================================================
    # tllm adaptive 4/6: end-to-end (triton_amax → scale → kernel)
    # ================================================================
    _ADAPTIVE_RULES = ("mse", "abs_max", "mae")
    adaptive_kernel_times: dict[str, float] = {}
    adaptive_e2e_times: dict[str, float] = {}

    for rule in _ADAPTIVE_RULES:
        try:
            scale_rule_int = _SCALE_RULE_MAP[rule]
            scaling_vector_size = 16

            # Kernel-only timing (scale precomputed)
            adaptive_global_scale = (_FOUROVERSIX_QUANT_RANGE / triton_amax(input_tensor)).item()
            adaptive_gs_tensor = torch.tensor(adaptive_global_scale, dtype=torch.float32, device=input_tensor.device)

            with nvtx.annotate(f"benchmark/tllm_{rule}/warmup"):
                for _ in range(warmup):
                    _ = torch.ops.tllm_linear_lite.fp4_quantize(
                        input_tensor, adaptive_gs_tensor, scaling_vector_size, False, True,
                        1, scale_rule_int,
                    )
                torch.cuda.synchronize()

            kernel_times = []
            with nvtx.annotate(f"benchmark/tllm_{rule}/kernel"):
                for i in range(repeat):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    rng = nvtx.start_range(f"tllm_{rule}/kernel_{i}")
                    start.record()
                    _ = torch.ops.tllm_linear_lite.fp4_quantize(
                        input_tensor, adaptive_gs_tensor, scaling_vector_size, False, True,
                        1, scale_rule_int,
                    )
                    end.record()
                    nvtx.end_range(rng)
                    torch.cuda.synchronize()
                    kernel_times.append(start.elapsed_time(end) * 1000)

            adaptive_kernel_times[rule] = sum(kernel_times) / len(kernel_times)

            # End-to-end timing (triton_amax → scale → kernel)
            with nvtx.annotate(f"benchmark/tllm_{rule}/e2e_warmup"):
                for _ in range(warmup):
                    gs = _FOUROVERSIX_QUANT_RANGE / triton_amax(input_tensor)
                    gs_t = torch.tensor(gs.item(), dtype=torch.float32, device=input_tensor.device)
                    _ = torch.ops.tllm_linear_lite.fp4_quantize(
                        input_tensor, gs_t, scaling_vector_size, False, True,
                        1, scale_rule_int,
                    )
                torch.cuda.synchronize()

            e2e_times = []
            with nvtx.annotate(f"benchmark/tllm_{rule}/e2e"):
                for i in range(repeat):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    rng = nvtx.start_range(f"tllm_{rule}/e2e_{i}")
                    start.record()
                    gs = _FOUROVERSIX_QUANT_RANGE / triton_amax(input_tensor)
                    gs_t = torch.tensor(gs.item(), dtype=torch.float32, device=input_tensor.device)
                    _ = torch.ops.tllm_linear_lite.fp4_quantize(
                        input_tensor, gs_t, scaling_vector_size, False, True,
                        1, scale_rule_int,
                    )
                    end.record()
                    nvtx.end_range(rng)
                    torch.cuda.synchronize()
                    e2e_times.append(start.elapsed_time(end) * 1000)

            adaptive_e2e_times[rule] = sum(e2e_times) / len(e2e_times)
        except Exception as e:
            print(f"\n  [WARN] tllm adaptive ({rule}) benchmark failed: {e}")

    # ================================================================
    # fouroversix: each scale rule measured separately
    # ================================================================
    _F46_SCALE_RULES = ("mse", "abs_max", "mae")
    f46_times: dict[str, float] = {}

    if FOUROVERSIX_AVAILABLE:
        from fouroversix import QuantizationConfig as F46Config
        from fouroversix.utils import ScaleRule as F46ScaleRule

        _F46_RULE_MAP = {"mse": F46ScaleRule.mse, "abs_max": F46ScaleRule.abs_max, "mae": F46ScaleRule.mae}

        for rule in _F46_SCALE_RULES:
            try:
                f46_rule_int = _F46_RULE_MAP[rule].cuda_id()

                with nvtx.annotate(f"benchmark/f46_{rule}/warmup"):
                    for _ in range(warmup):
                        _ = torch.ops.fouroversix.quantize_to_fp4(
                            input_tensor, True, True, False, False, False, f46_rule_int, 0,
                        )
                    torch.cuda.synchronize()

                times = []
                with nvtx.annotate(f"benchmark/f46_{rule}/timed"):
                    for i in range(repeat):
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        rng = nvtx.start_range(f"f46_{rule}/iter_{i}")
                        start.record()
                        _ = torch.ops.fouroversix.quantize_to_fp4(
                            input_tensor, True, True, False, False, False, f46_rule_int, 0,
                        )
                        end.record()
                        nvtx.end_range(rng)
                        torch.cuda.synchronize()
                        times.append(start.elapsed_time(end) * 1000)  # ms -> us

                f46_times[rule] = sum(times) / len(times)
            except Exception as e:
                print(f"\n  [WARN] fouroversix ({rule}) benchmark failed: {e}")
    else:
        print("\n  fouroversix: [SKIP] not installed")

    # ================================================================
    # Unified performance table (tllm = baseline)
    # ================================================================
    col_names: list[str] = ["tllm"]
    kernel_times: list[float] = [tllm_kernel_us]
    e2e_times_list: list[float] = [tllm_e2e_us]
    kernel_only_col: list[bool] = [True]
    for rule in _ADAPTIVE_RULES:
        if rule in adaptive_kernel_times:
            col_names.append(f"tllm({rule})")
            kernel_times.append(adaptive_kernel_times[rule])
            e2e_times_list.append(adaptive_e2e_times.get(rule, adaptive_kernel_times[rule]))
            kernel_only_col.append(True)
    for rule in _F46_SCALE_RULES:
        if rule in f46_times:
            col_names.append(f"f46({rule})")
            kernel_times.append(f46_times[rule])
            e2e_times_list.append(f46_times[rule])
            kernel_only_col.append(False)

    col_w = 14
    header = f"  {'Metric':<25}" + "".join(f"{n:>{col_w}}" for n in col_names)
    sep = f"  {'-' * (25 + col_w * len(col_names))}"

    print(f"\n{header}")
    print(sep)

    # Kernel-only time
    print(f"  {'Kernel only (us)':<25}" + "".join(
        f"{t:>{col_w}.2f}" for t in kernel_times
    ))

    # End-to-end time (prologue + kernel for tllm; f46 already includes prologue)
    print(f"  {'End-to-end (us)':<25}" + "".join(
        f"{t:>{col_w}.2f}" for t in e2e_times_list
    ))

    # Effective bandwidth (based on end-to-end time)
    bws = [total_bytes / (t * 1e-6) / 1e12 for t in e2e_times_list]
    print(f"  {'E2E BW (TB/s)':<25}" + "".join(
        f"{bw:>{col_w}.2f}" for bw in bws
    ))

    # Memory bandwidth utilization (based on end-to-end time)
    mbus = [bw / peak_bw * 100 for bw in bws]
    print(f"  {'E2E MBU %':<25}" + "".join(
        f"{m:>{col_w - 1}.1f}%" for m in mbus
    ))

    # Kernel speedup vs tllm (N/A for f46 whose kernel time includes prologue)
    if len(kernel_times) > 1:
        def _kernel_speedup(i: int, t: float) -> str:
            if i == 0:
                return f"{'1.00x':>{col_w}}"
            if not kernel_only_col[i]:
                return f"{'N/A':>{col_w}}"
            return f"{tllm_kernel_us / t:>{col_w - 1}.2f}x"

        print(f"  {'Kernel speedup':<25}" + "".join(
            _kernel_speedup(i, t) for i, t in enumerate(kernel_times)
        ))

    # E2E speedup vs tllm (>1 means faster than tllm)
    if len(e2e_times_list) > 1:
        print(f"  {'E2E speedup':<25}" + "".join(
            f"{'1.00x':>{col_w}}" if i == 0
            else f"{tllm_e2e_us / t:>{col_w - 1}.2f}x"
            for i, t in enumerate(e2e_times_list)
        ))

    print(f"\n  Prologue: triton_amax = {scale_time_us:.2f} us"
          f" (included in end-to-end; f46 includes its own prologue)")

    # Output statistics
    print(f"\n  Output statistics:")
    print(f"    FP4 output:    {act_fp4.numel() * act_fp4.element_size() / 1024:.2f} KB")
    print(f"    Scale factors: {act_sf.numel() * act_sf.element_size() / 1024:.2f} KB")
    print(f"    Compression:   {data_bytes / (act_fp4.numel() * act_fp4.element_size()):.2f}x")


def run_ncu_mode(input_tensor: torch.Tensor):
    """NCU profiling mode: single iteration for each op."""
    print("\n" + "=" * 80)
    print("NCU Profiling Mode (tllm_linear_lite + fouroversix)")
    print("=" * 80)

    # Warmup: run all variants to avoid cold-start effects
    print("Warmup...")
    for _ in range(3):
        _ = quantize(input_tensor)
    for rule in ("mse", "abs_max", "mae"):
        for _ in range(3):
            _ = fp4_quantize_unified(
                input_tensor, backend="tllm", swizzled=True, scale_rule=rule,
            )
    if FOUROVERSIX_AVAILABLE:
        from fouroversix import QuantizationConfig as F46Config
        for rule in ("mse", "abs_max", "mae"):
            for _ in range(3):
                _ = fp4_quantize_unified(
                    input_tensor, backend="fouroversix", config=F46Config(scale_rule=rule)
                )
    torch.cuda.synchronize()
    print("Warmup complete.\n")

    # "ncu_capture" marks the exact region NCU should capture via --nvtx-include.
    # Warmup above is intentionally outside this range.
    with nvtx.annotate("ncu_capture"):
        # ------------------------------------------------------------------ tllm
        print("Profiling fp4_quantize (tllm_linear_lite)...")
        with nvtx.annotate("tllm_linear_lite/fp4_quantize"):
            act_fp4, act_sf = quantize(input_tensor)
            torch.cuda.synchronize()

        # ------------------------------------------------------------------ tllm adaptive 4/6
        for rule in ("mse", "abs_max", "mae"):
            print(f"Profiling fp4_quantize adaptive (scale_rule={rule})...")
            with nvtx.annotate(f"tllm_linear_lite/fp4_quantize_adaptive/{rule}"):
                _ = fp4_quantize_unified(
                    input_tensor, backend="tllm", swizzled=True, scale_rule=rule,
                )
                torch.cuda.synchronize()

        print("Profiling calculate_nvfp4_global_scale (tllm_linear_lite)...")
        with nvtx.annotate("tllm_linear_lite/global_scale"):
            _ = torch.ops.tllm_linear_lite.calculate_nvfp4_global_scale(input_tensor, None)
            torch.cuda.synchronize()

        # ------------------------------------------------------------------ fouroversix
        if FOUROVERSIX_AVAILABLE:
            from fouroversix import QuantizationConfig as F46Config
            for rule in ("mse", "abs_max", "mae"):
                print(f"Profiling fouroversix fp4_quantize (scale_rule={rule})...")
                with nvtx.annotate(f"fouroversix/fp4_quantize/{rule}"):
                    _ = fp4_quantize_unified(
                        input_tensor, backend="fouroversix", config=F46Config(scale_rule=rule)
                    )
                    torch.cuda.synchronize()
        else:
            print("fouroversix: [SKIP] not installed")

    print(f"\nFP4 output shape:   {act_fp4.shape}")
    print(f"Scale factor shape: {act_sf.shape}")
    print("\nNCU profiling complete!")


def main():
    parser = argparse.ArgumentParser(description="tllm_linear_lite FP4 Quantization Test")
    parser.add_argument("--ncu", action="store_true", help="NCU profiling mode")
    parser.add_argument(
        "--shape", type=str, default="14400,6144",
        help="Input shape (comma-separated)"
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=20, help="Benchmark iterations")
    parser.add_argument(
        "--dtype", type=str, default="bfloat16",
        choices=["float16", "bfloat16"],
        help="Input dtype"
    )
    parser.add_argument("--skip-verify", action="store_true", help="Skip correctness verification")
    parser.add_argument("--compare-trtllm", action="store_true", help="Compare with TensorRT-LLM")
    parser.add_argument(
        "--scale-rule", type=str, default=None,
        choices=["mse", "mae", "abs_max"],
        help="Run adaptive 4/6 quantize with specified scale rule (standalone benchmark)"
    )
    args = parser.parse_args()

    shape = tuple(map(int, args.shape.split(",")))
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16

    print("\n" + "=" * 80)
    print("tllm_linear_lite FP4 Quantization Test")
    print("=" * 80)
    print(f"Device:    {torch.cuda.get_device_name(0)}")
    print(f"Shape:     {shape}")
    print(f"Dtype:     {dtype}")
    print(f"Elements:  {torch.tensor(shape).prod().item():,}")

    input_tensor = torch.randn(shape, device="cuda", dtype=dtype)

    if args.ncu:
        run_ncu_mode(input_tensor)
    elif args.scale_rule:
        # Standalone adaptive 4/6 benchmark
        rule = args.scale_rule
        print(f"\nAdaptive 4/6 benchmark (scale_rule={rule})")
        print("-" * 60)

        for _ in range(args.warmup):
            _ = fp4_quantize_unified(input_tensor, backend="tllm", swizzled=True, scale_rule=rule)
        torch.cuda.synchronize()

        times = []
        for _ in range(args.repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fp4, sf = fp4_quantize_unified(input_tensor, backend="tllm", swizzled=True, scale_rule=rule)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000)  # us

        avg_us = sum(times) / len(times)
        n_bytes = input_tensor.numel() * input_tensor.element_size()
        out_bytes = fp4.numel() * fp4.element_size() + sf.numel() * sf.element_size()
        bw_tb = (n_bytes + out_bytes) / (avg_us * 1e-6) / 1e12

        print(f"  Kernel time:    {avg_us:.2f} us")
        print(f"  Effective BW:   {bw_tb:.2f} TB/s")
        print(f"  Output shape:   fp4={fp4.shape}, sf={sf.shape}")

        # Quick accuracy check vs fouroversix if available
        if FOUROVERSIX_AVAILABLE:
            from fouroversix import QuantizationConfig as F46Config
            config = F46Config(scale_rule=rule)
            f46_qt = fp4_quantize_unified(input_tensor, backend="fouroversix", config=config)
            f46_dq = f46_qt.dequantize(dtype=torch.bfloat16).float()
            M, K = input_tensor.shape[0], input_tensor.shape[-1]
            input_flat = input_tensor.float().reshape(-1, K)
            if f46_dq.shape != input_flat.shape:
                f46_dq = f46_dq[:input_flat.shape[0], :input_flat.shape[1]]

            # tllm adaptive dequant
            gs = (6.0 * 256.0 / triton_amax(input_tensor)).item()
            fp4_f, sf_f = unpack_cuda_fp4_output(
                *fp4_quantize_unified(input_tensor, backend="tllm", swizzled=False, scale_rule=rule),
                M, K, 16
            )
            tllm_dq = dequantize_fp4(fp4_f, sf_f, gs, 16)

            tllm_err = (tllm_dq - input_flat).abs().mean().item()
            f46_err = (f46_dq - input_flat).abs().mean().item()
            print(f"\n  Mean abs error:  tllm({rule})={tllm_err:.6f}  f46({rule})={f46_err:.6f}")
            print(f"  Error ratio (tllm/f46): {tllm_err / max(f46_err, 1e-10):.4f}")
    else:
        if not args.skip_verify:
            if not test_accuracy(input_tensor):
                print("\n[FAIL] Accuracy test failed, aborting.")
                return

        if args.compare_trtllm:
            compare_with_trtllm(input_tensor)

        benchmark(input_tensor, warmup=args.warmup, repeat=args.repeat)

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
