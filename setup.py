"""
Build script for tllm_linear_lite NVFP4 CUDA extension.

Usage:
    pip install -e .          # editable install
    python setup.py build_ext --inplace  # build only

After building, use the ops as:
    import tllm_linear_lite
    torch.ops.tllm_linear_lite.fp4_quantize(...)
    torch.ops.tllm_linear_lite.cublaslt_fp4_gemm(...)
"""

import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent.resolve()
QUANTIZE_DIR = ROOT_DIR / "quantize"
GEMM_DIR = ROOT_DIR / "gemm"

# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------
sources = [
    # Quantize ops
    str(QUANTIZE_DIR / "quantization.cu"),
    str(QUANTIZE_DIR / "fp4_quantize_op.cu"),
    # GEMM ops
    str(GEMM_DIR / "cublaslt_fp4_gemm.cpp"),
    str(GEMM_DIR / "cuda_core_nvfp4_gemm.cu"),
    str(GEMM_DIR / "cutlass_fp4_gemm_op.cu"),
    # CUTLASS FP4 GEMM template instantiations (slow to compile, ~50 kernels each)
    str(GEMM_DIR / "cutlass_fp4" / "fp4_gemm_fp16.cu"),
    str(GEMM_DIR / "cutlass_fp4" / "fp4_gemm_bf16.cu"),
]

# ---------------------------------------------------------------------------
# CUTLASS FP4 GEMM headers
# ---------------------------------------------------------------------------
CUTLASS_FP4_DIR = GEMM_DIR / "cutlass_fp4"

# ---------------------------------------------------------------------------
# CUTLASS (vendored, header-only)
# ---------------------------------------------------------------------------
CUTLASS_DIR = ROOT_DIR / "3rdparty" / "cutlass"
CUTLASS_INCLUDE = CUTLASS_DIR / "include"
CUTLASS_TOOLS_INCLUDE = CUTLASS_DIR / "tools" / "util" / "include"

# ---------------------------------------------------------------------------
# Compile flags
# ---------------------------------------------------------------------------
cxx_flags = [
    "-std=c++17",
    "-O3",
]

nvcc_flags = [
    "-std=c++17",
    "-O3",
    "--use_fast_math",
    "-DENABLE_BF16",
    "-DENABLE_FP8",
    "-DENABLE_FP4",
    "-DUSING_OSS_CUTLASS_FP4_GEMM",
    # Suppress annoying warnings from CUDA headers
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
]

# Let PyTorch's BuildExtension handle CUDA architectures via TORCH_CUDA_ARCH_LIST.
# Set a default if not already in the environment.
# NOTE: sm_100a (not sm_100) is required for Blackwell-specific instructions
# used in FP4 kernels (e2m1 conversion PTX, __nv_fp8_e8m0, etc.).
# The "a" suffix means architecture-specific (non-forward-compatible).
if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;8.9;9.0;10.0a;12.0a"

# Include paths
include_dirs = [
    str(QUANTIZE_DIR),
    str(GEMM_DIR),
    str(CUTLASS_FP4_DIR),
    str(CUTLASS_INCLUDE),
    str(CUTLASS_TOOLS_INCLUDE),
]

# ---------------------------------------------------------------------------
# Extension
# ---------------------------------------------------------------------------
ext_modules = [
    CUDAExtension(
        name="tllm_linear_lite._C",
        sources=sources,
        include_dirs=include_dirs,
        libraries=["cublasLt"],
        extra_compile_args={
            "cxx": cxx_flags,
            "nvcc": nvcc_flags,
        },
    ),
]

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
setup(
    name="tllm_linear_lite",
    version="0.1.0",
    description="Standalone NVFP4 quantize + GEMM ops (no TensorRT-LLM dependency)",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.10",
)
