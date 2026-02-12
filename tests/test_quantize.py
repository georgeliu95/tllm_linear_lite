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

# Load tllm_linear_lite CUDA extension
import tllm_linear_lite  # noqa: F401


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
    warmup: int = 3,
    repeat: int = 10,
) -> Tuple[float, float]:
    """Measure the time for global scale computation (fixed overhead).

    Args:
        input_tensor: Input tensor
        warmup: Number of warmup iterations
        repeat: Number of timed iterations

    Returns:
        (act_global_scale, avg_time_us)
    """
    for _ in range(warmup):
        _ = 448.0 * 6.0 / input_tensor.abs().amax().float()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        act_global_scale = 448.0 * 6.0 / input_tensor.abs().amax().float()
        end.record()

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

    for _ in range(warmup):
        _ = torch.ops.tllm_linear_lite.fp4_quantize(
            input_tensor, act_global_scale, scaling_vector_size, False, True
        )
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        act_fp4, act_sf = torch.ops.tllm_linear_lite.fp4_quantize(
            input_tensor, act_global_scale, scaling_vector_size, False, True
        )
        end.record()

        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000)  # ms -> us

    avg_time_us = sum(times) / len(times)
    return (act_fp4, act_sf), avg_time_us


def verify_correctness(input_tensor: torch.Tensor) -> bool:
    """Verify numerical correctness of tllm_linear_lite.fp4_quantize.

    Comparison structure:
    1. PyTorch fake NVFP4 quantize vs CUDA kernel:
       - Compare block scale factors (FP8 E4M3)
       - Compare FP4 output values (unpacked E2M1)
    2. Dequantize both outputs independently, compare each against original input
    3. Verify calculate_nvfp4_global_scale op
    """
    print("\n" + "=" * 80)
    print("Correctness Verification")
    print("=" * 80)

    try:
        M = input_tensor.shape[0] if input_tensor.dim() == 2 else input_tensor.reshape(-1, input_tensor.shape[-1]).shape[0]
        K = input_tensor.shape[-1]
        block_size = 16
        global_scale_val = (448.0 * 6.0 / input_tensor.abs().amax().float()).item()
        input_flat = input_tensor.float().reshape(M, K)

        print(f"\nInput shape: {input_tensor.shape}, dtype: {input_tensor.dtype}")
        print(f"Global scale: {global_scale_val:.4f}")

        # ================================================================
        # Run both implementations
        # ================================================================

        # PyTorch reference (fake quantize)
        ref_fp4_float, ref_sf = fp4_quantize_reference(
            input_tensor, global_scale_val, block_size
        )
        ref_fp4_float = ref_fp4_float.to(input_tensor.device)
        ref_sf = ref_sf.to(input_tensor.device)
        ref_dequant = dequantize_fp4(ref_fp4_float, ref_sf, global_scale_val, block_size)

        # CUDA kernel (LINEAR layout for easy comparison)
        act_fp4, act_sf = quantize(input_tensor, swizzled=False)
        kernel_fp4_float, kernel_sf = unpack_cuda_fp4_output(
            act_fp4, act_sf, M, K, block_size
        )
        kernel_dequant = dequantize_fp4(kernel_fp4_float, kernel_sf, global_scale_val, block_size)

        # ================================================================
        # Step 1: PyTorch fake quantize vs CUDA kernel (FP4 output + scales)
        # ================================================================
        print("\n--- Step 1: Quantized output comparison (PyTorch ref vs CUDA kernel) ---")

        # 1a. Scale factors
        sf_match = torch.equal(kernel_sf, ref_sf)
        sf_diff = (kernel_sf - ref_sf).abs()
        sf_mismatch = (sf_diff > 0).sum().item()
        sf_max_diff = sf_diff.max().item()

        print(f"\n  Block scale factors ({kernel_sf.numel()} values):")
        if sf_match:
            print(f"    [PASS] Exact match")
        else:
            print(f"    Mismatches: {sf_mismatch}/{kernel_sf.numel()} ({100*sf_mismatch/kernel_sf.numel():.4f}%)")
            print(f"    Max diff:   {sf_max_diff:.6f}")
            print(f"    {'[PASS] Within tolerance' if sf_max_diff < 1e-3 else '[WARN] Exceeds tolerance'}")

        # 1b. FP4 values (E2M1 representable, before dequant)
        fp4_match = torch.equal(kernel_fp4_float, ref_fp4_float)
        fp4_diff = (kernel_fp4_float - ref_fp4_float).abs()
        fp4_mismatch = (fp4_diff > 0).sum().item()
        fp4_max_diff = fp4_diff.max().item()

        print(f"\n  FP4 values ({kernel_fp4_float.numel()} values):")
        if fp4_match:
            print(f"    [PASS] Exact match")
        else:
            print(f"    Mismatches: {fp4_mismatch}/{kernel_fp4_float.numel()} ({100*fp4_mismatch/kernel_fp4_float.numel():.4f}%)")
            print(f"    Max diff:   {fp4_max_diff:.6f}")
            # FP4 rounding near decision boundaries + reciprocal_approximate_ftz
            # can cause ~0.1% of values to differ by one E2M1 step
            pct = 100 * fp4_mismatch / kernel_fp4_float.numel()
            print(f"    {'[PASS] Within tolerance' if pct < 1.0 else '[WARN] Exceeds 1% mismatch rate'}")

        # ================================================================
        # Step 2: Dequantized outputs vs original input (quantization error)
        # ================================================================
        print("\n--- Step 2: Dequantized output vs original input ---")

        # PyTorch reference dequant error
        ref_err = (ref_dequant - input_flat).abs()
        ref_rel_err = ref_err / (input_flat.abs().clamp(min=1e-6))

        # CUDA kernel dequant error
        kernel_err = (kernel_dequant - input_flat).abs()
        kernel_rel_err = kernel_err / (input_flat.abs().clamp(min=1e-6))

        print(f"\n  {'Metric':<35} {'PyTorch Ref':>15} {'CUDA Kernel':>15}")
        print(f"  {'-'*65}")
        print(f"  {'Max abs error':<35} {ref_err.max().item():>15.6f} {kernel_err.max().item():>15.6f}")
        print(f"  {'Mean abs error':<35} {ref_err.mean().item():>15.6f} {kernel_err.mean().item():>15.6f}")
        print(f"  {'Max relative error':<35} {ref_rel_err.max().item():>15.4f} {kernel_rel_err.max().item():>15.4f}")
        print(f"  {'Mean relative error':<35} {ref_rel_err.mean().item():>15.4f} {kernel_rel_err.mean().item():>15.4f}")

        # Sanity: both should have similar error magnitude
        err_ratio = kernel_err.mean().item() / max(ref_err.mean().item(), 1e-10)
        print(f"\n  Kernel/Reference error ratio: {err_ratio:.4f}")
        if 0.8 < err_ratio < 1.2:
            print(f"  [PASS] Both implementations have similar quantization error")
        else:
            print(f"  [WARN] Error ratio outside [0.8, 1.2], implementations may diverge")

        # ================================================================
        # Step 3: calculate_nvfp4_global_scale
        # ================================================================
        print("\n--- Step 3: calculate_nvfp4_global_scale ---")

        per_token_scale = torch.ops.tllm_linear_lite.calculate_nvfp4_global_scale(
            input_tensor, None
        )

        # PyTorch reference
        ref_per_token = 448.0 * 6.0 / input_flat.abs().amax(dim=-1, keepdim=True)
        kernel_pts = per_token_scale.float().reshape(M, 1)

        pts_diff = (kernel_pts - ref_per_token).abs()
        pts_max_diff = pts_diff.max().item()

        assert not torch.isnan(per_token_scale).any(), "Per-token scale contains NaN"
        assert not torch.isinf(per_token_scale).any(), "Per-token scale contains Inf"

        if pts_max_diff < 1e-2:
            print(f"  [PASS] Per-token global scale matches (max_diff={pts_max_diff:.6f})")
        else:
            print(f"  [WARN] Per-token global scale max_diff={pts_max_diff:.6f}")

        # ================================================================
        # Summary
        # ================================================================
        print("\n" + "=" * 80)
        all_pass = (
            (sf_max_diff < 1e-3)
            and (100 * fp4_mismatch / kernel_fp4_float.numel() < 1.0)
            and (0.5 < err_ratio < 2.0)
            and (pts_max_diff < 1e-2)
        )
        if all_pass:
            print("[PASS] All correctness checks passed")
        else:
            print("[WARN] Some checks have warnings, review above")
        print("=" * 80)
        return all_pass

    except Exception as e:
        print(f"\n[FAIL] Verification failed: {e}")
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
    """Run performance benchmark."""
    print("\n" + "=" * 80)
    print("Performance Benchmark (tllm_linear_lite)")
    print("=" * 80)

    n_elements = input_tensor.numel()
    data_bytes = n_elements * input_tensor.element_size()

    print(f"Input shape:  {input_tensor.shape}")
    print(f"Input dtype:  {input_tensor.dtype}")
    print(f"Elements:     {n_elements:,}")
    print(f"Data size:    {data_bytes / 1024 / 1024:.2f} MB")
    print(f"Warmup: {warmup}, Repeat: {repeat}")

    # Step 1: Scale computation time (fixed overhead)
    print("\n--- Scale Computation (Fixed Overhead) ---")
    act_global_scale, scale_time_us = measure_scale_computation(
        input_tensor, warmup=warmup, repeat=repeat
    )
    print(f"Scale computation: {scale_time_us:.2f} us")

    # Step 2: Kernel time
    print("\n--- fp4_quantize Kernel ---")
    (act_fp4, act_sf), kernel_time_us = run_quantize_kernel(
        input_tensor, act_global_scale, warmup=warmup, repeat=repeat
    )

    total_time_us = scale_time_us + kernel_time_us

    # Bandwidth calculation
    output_bytes = act_fp4.numel() * act_fp4.element_size() + act_sf.numel() * act_sf.element_size()
    kernel_bytes = data_bytes + output_bytes
    bandwidth_tb_s = kernel_bytes / (kernel_time_us * 1e-6) / 1e12

    # GPU peak bandwidth (B200 = 8 TB/s, H100 SXM = 3.35 TB/s)
    gpu_name = torch.cuda.get_device_name(0).lower()
    if "b200" in gpu_name or "b100" in gpu_name:
        peak_bw = 8.0
    elif "h100" in gpu_name or "h200" in gpu_name:
        peak_bw = 3.35
    else:
        peak_bw = 2.0  # conservative estimate
    mbu_pct = bandwidth_tb_s / peak_bw * 100

    print(f"\n{'Metric':<30} {'Value':>15}")
    print("-" * 50)
    print(f"{'Kernel time':<30} {kernel_time_us:>12.2f} us")
    print(f"{'Scale computation':<30} {scale_time_us:>12.2f} us")
    print(f"{'Total time':<30} {total_time_us:>12.2f} us")
    print(f"{'Effective bandwidth':<30} {bandwidth_tb_s:>12.2f} TB/s")
    print(f"{'Memory bandwidth utilization':<30} {mbu_pct:>11.1f} %")

    print(f"\n--- Output Statistics ---")
    print(f"  FP4 output:      {act_fp4.numel() * act_fp4.element_size() / 1024:.2f} KB")
    print(f"  Scale factors:   {act_sf.numel() * act_sf.element_size() / 1024:.2f} KB")
    print(f"  Compression:     {data_bytes / (act_fp4.numel() * act_fp4.element_size()):.2f}x")


def run_ncu_mode(input_tensor: torch.Tensor):
    """NCU profiling mode: single iteration for each op."""
    print("\n" + "=" * 80)
    print("NCU Profiling Mode (tllm_linear_lite)")
    print("=" * 80)

    # Warmup
    print("Warmup...")
    for _ in range(3):
        _ = quantize(input_tensor)
    torch.cuda.synchronize()
    print("Warmup complete.\n")

    torch.cuda.cudart().cudaProfilerStart()

    print("Profiling fp4_quantize...")
    torch.cuda.nvtx.range_push("tllm_linear_lite_fp4_quantize")
    act_fp4, act_sf = quantize(input_tensor)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    print("Profiling calculate_nvfp4_global_scale...")
    torch.cuda.nvtx.range_push("tllm_linear_lite_global_scale")
    _ = torch.ops.tllm_linear_lite.calculate_nvfp4_global_scale(input_tensor, None)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    torch.cuda.cudart().cudaProfilerStop()

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
    else:
        if not args.skip_verify:
            if not verify_correctness(input_tensor):
                print("\n[FAIL] Verification failed, aborting.")
                return

        if args.compare_trtllm:
            compare_with_trtllm(input_tensor)

        benchmark(input_tensor, warmup=args.warmup, repeat=args.repeat)

    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
