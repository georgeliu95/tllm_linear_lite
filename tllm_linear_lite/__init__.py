"""
tllm_linear_lite: Standalone FP4 quantization ops.

After building, the following PyTorch ops are available:
    torch.ops.tllm_linear_lite.fp4_quantize(input, globalScale, sfVecSize, sfUseUE8M0, isSfSwizzledLayout)
    torch.ops.tllm_linear_lite.calculate_nvfp4_global_scale(input, tokensPerBatch)
"""

import os
import glob

import torch

# Load the compiled CUDA extension .so which registers ops via TORCH_LIBRARY.
# We find the .so next to this __init__.py (placed there by setuptools build_ext --inplace).
_dir = os.path.dirname(os.path.abspath(__file__))
_so_files = glob.glob(os.path.join(_dir, "_C*.so"))

if not _so_files:
    # Also check parent directory (non-editable install may place it differently)
    _parent = os.path.dirname(_dir)
    _so_files = glob.glob(os.path.join(_parent, "tllm_linear_lite", "_C*.so"))

if _so_files:
    torch.ops.load_library(_so_files[0])
else:
    raise ImportError(
        "tllm_linear_lite CUDA extension not found. "
        "Build with: cd tllm_linear_lite && TORCH_CUDA_ARCH_LIST='10.0a' pip install -e . --no-build-isolation"
    )
