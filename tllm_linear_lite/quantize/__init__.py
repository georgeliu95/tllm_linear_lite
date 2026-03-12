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
tllm_linear_lite.quantize: FP4 quantization with optional fouroversix backend.

- backend="tllm": uses torch.ops.tllm_linear_lite.fp4_quantize (native CUDA).
  Supports scale_rule="static_6" (default NVFP4) or "mse"/"mae"/"abs_max"
  for FourOverSix adaptive 4/6 block scaling (native CUDA, no fouroversix dependency).
- backend="fouroversix": uses fouroversix.quantize_to_fp4 (optional dependency).
  When fouroversix is used, returns a fouroversix QuantizedTensor; use .dequantize()
  for high-precision values. When backend="tllm", returns (packed_fp4, scale_factors).
"""

from __future__ import annotations

import logging
from typing import Any, Tuple, Union

import torch

logger = logging.getLogger(__name__)
_FOUROVERSIX_IMPORT_ERROR: Exception | None = None

# Optional fouroversix: defaults to MSE-based four-over-six scale selection.
try:
    import fouroversix
    from fouroversix import QuantizationConfig, quantize_to_fp4 as _fouroversix_quantize_to_fp4

    FOUROVERSIX_AVAILABLE = True
except Exception as exc:
    fouroversix = None  # type: ignore[assignment]
    _fouroversix_quantize_to_fp4 = None
    QuantizationConfig = None  # type: ignore[misc, assignment]
    FOUROVERSIX_AVAILABLE = False
    _FOUROVERSIX_IMPORT_ERROR = exc
    logger.warning(
        "fouroversix import failed (%s: %s). backend='fouroversix' disabled.",
        type(exc).__name__,
        exc,
    )


_SCALE_RULE_MAP = {
    "static_6": 0,
    "static_4": 0,
    "mse": 1,
    "mae": 2,
    "abs_max": 3,
}

# Standard NVFP4: max_e2m1=6, max_e4m3=448 → encode_scale = 2688/amax
_TLLM_QUANT_RANGE = 6.0 * 448.0  # 2688
# FourOverSix adaptive: max_e2m1=6, max_e4m3=256 → encode_scale = 1536/amax
_FOUROVERSIX_QUANT_RANGE = 6.0 * 256.0  # 1536


def fp4_quantize(
    x: torch.Tensor,
    *,
    backend: str = "tllm",
    global_scale: torch.Tensor | float | None = None,
    swizzled: bool = True,
    scale_rule: str = "static_6",
    **kwargs: Any,
) -> Union[Tuple[torch.Tensor, torch.Tensor], Any]:
    """
    FP4 quantize with backend selection.

    Args:
        x: Input tensor (2D, bfloat16 or float16, CUDA).
        backend: "tllm" (native CUDA op) or "fouroversix" (optional).
        global_scale: For tllm backend only. If None, auto-computed from amax.
        swizzled: For tllm backend only. Scale factor layout.
        scale_rule: Block scale selection rule (tllm backend only).
            "static_6" (default) — standard NVFP4, all blocks use scale 6.
            "mse" / "mae" / "abs_max" — FourOverSix adaptive 4/6 selection,
            choosing per-block between scale 4 and 6 to minimize the given
            error metric. Uses encode_scale=1536/amax instead of 2688/amax.
        **kwargs: Passed to fouroversix when backend="fouroversix" (e.g. config=...).

    Returns:
        - If backend="tllm": (packed_fp4, scale_factors) as torch.Tensor.
        - If backend="fouroversix": fouroversix QuantizedTensor (has .dequantize()).
    """
    if backend == "tllm":
        scale_rule_int = _SCALE_RULE_MAP.get(scale_rule)
        if scale_rule_int is None:
            raise ValueError(
                f"Unknown scale_rule: {scale_rule!r}. "
                f"Choose from {list(_SCALE_RULE_MAP.keys())}."
            )

        is_adaptive = scale_rule_int != 0
        quant_range = _FOUROVERSIX_QUANT_RANGE if is_adaptive else _TLLM_QUANT_RANGE

        if global_scale is None:
            global_scale = (quant_range / x.abs().amax().float()).item()
        if isinstance(global_scale, float):
            global_scale_t = torch.tensor(
                global_scale, dtype=torch.float32, device=x.device
            )
        else:
            global_scale_t = global_scale
        scaling_vector_size = 16
        packed, sf = torch.ops.tllm_linear_lite.fp4_quantize(
            x, global_scale_t, scaling_vector_size, False, swizzled,
            1,  # kernelVersion (v1 default)
            scale_rule_int,
        )
        return (packed, sf)

    if backend == "fouroversix":
        if not FOUROVERSIX_AVAILABLE:
            if _FOUROVERSIX_IMPORT_ERROR is not None:
                raise ImportError(
                    "fouroversix is installed but failed to import. "
                    f"Original error: {_FOUROVERSIX_IMPORT_ERROR}"
                ) from _FOUROVERSIX_IMPORT_ERROR
            raise ImportError("fouroversix is not installed. Install it to use backend='fouroversix'.")
        config = kwargs.get("config")
        if config is None:
            config = QuantizationConfig()
        return _fouroversix_quantize_to_fp4(x, config=config)

    raise ValueError(f"Unknown backend: {backend}. Use 'tllm' or 'fouroversix'.")


def calculate_nvfp4_global_scale(
    x: torch.Tensor,
    tokens_per_batch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute per-token or global NVFP4 scale (tllm op)."""
    return torch.ops.tllm_linear_lite.calculate_nvfp4_global_scale(x, tokens_per_batch)


__all__ = [
    "FOUROVERSIX_AVAILABLE",
    "fp4_quantize",
    "calculate_nvfp4_global_scale",
]
