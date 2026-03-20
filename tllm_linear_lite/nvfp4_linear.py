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
nvfp4_linear.py - End-to-end NVFP4 Dynamic Linear Module.

Drop-in replacement for ``nn.Linear`` that fuses FP4 weight storage with
dynamic activation quantization and NVFP4 GEMM dispatch.

Weights are quantized once at construction time (SWIZZLED layout, FP4 packed
uint8 + FP8 E4M3 scale factors).  Activations are quantized on-the-fly in
each ``forward()`` call.  The module resolves the GEMM backend internally
among {cutedsl, cutlass, cublaslt} — cuda_core is excluded because this
module only produces SWIZZLED activation scale factors.

Supports two quantization backends:
  - ``"tllm"``:       native CUDA ``fp4_quantize`` (default)
  - ``"fouroversix"``: MSE-based adaptive block scaling (optional dependency)

Usage:
    from tllm_linear_lite.nvfp4_linear import NVFP4DynamicLinear

    linear = nn.Linear(4096, 4096, device="cuda", dtype=torch.bfloat16)
    nvfp4 = NVFP4DynamicLinear.from_linear(linear)
    output = nvfp4(x)  # x: [..., 4096] -> [..., 4096]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tllm_linear_lite.quantize import (
    FOUROVERSIX_AVAILABLE,
    fp4_quantize,
    _SCALE_RULE_MAP,
    _TLLM_QUANT_RANGE,
    _FOUROVERSIX_QUANT_RANGE,
)
from tllm_linear_lite.amax import triton_amax, cuda_prologue
from tllm_linear_lite.nvfp4_gemm import nvfp4_gemm, _is_blackwell_sm
from tllm_linear_lite.cutedsl import IS_CUTLASS_DSL_AVAILABLE

# torch.finfo(torch.float32).tiny ≈ 1.175e-38 is too small: dividing
# quant_range (2688) by it overflows float32 (max ~3.4e38), producing inf
# global_scale and NaN in downstream FP4 quantization.  1e-12 keeps the
# global_scale safely within float32 range while still being negligible
# compared to any realistic activation magnitude.
_EPS: float = 1e-12

# Below this element count, PyTorch abs().max() is faster than triton_amax
# due to lower launch overhead.  Crossover measured on B200 at ~16M elements
# (e.g. M=2048, K=6144).  See benchmark in test_nvfp4_linear.py.
_TRITON_AMAX_THRESHOLD: int = 16_000_000

# Valid GEMM backends for this module (cuda_core excluded — SWIZZLED only)
_VALID_GEMM_BACKENDS = ("auto", "cutedsl", "cutlass", "cublaslt")
_VALID_QUANT_BACKENDS = ("tllm", "fouroversix")


def _compute_quant_range(
    quant_backend: str,
    quant_config: object | None,
    scale_rule: str = "static_6",
) -> float:
    """Derive quant range (max_e2m1 * max_e4m3) from quant configuration.

    For tllm backend the range depends on the scale rule: static rules use
    the full E4M3 range (6 * 448 = 2688), while adaptive 4/6 rules (mse,
    abs_max, mae) use a reduced range (6 * 256 = 1536).  For fouroversix the
    range is derived from the QuantizationConfig.

    Args:
        quant_backend: ``"tllm"`` or ``"fouroversix"``.
        quant_config: A resolved ``fouroversix.QuantizationConfig``.
                      Must not be ``None`` when ``quant_backend="fouroversix"``
                      (caller is responsible for creating the default config).
        scale_rule: Block scale selection rule (tllm backend only).

    Returns:
        Quant range as a Python float.

    Raises:
        ValueError: If ``quant_backend="fouroversix"`` but ``quant_config``
                    is None (indicates a caller bug).
    """
    if quant_backend == "tllm":
        is_adaptive = _SCALE_RULE_MAP.get(scale_rule, 0) != 0
        return _FOUROVERSIX_QUANT_RANGE if is_adaptive else _TLLM_QUANT_RANGE

    if quant_config is None:
        raise ValueError(
            "quant_config must be resolved before computing quant_range "
            "for fouroversix backend"
        )

    max_e2m1 = float(quant_config.scale_rule.max_allowed_e2m1_value())
    max_e4m3 = float(quant_config.scale_rule.max_allowed_e4m3_value())
    return max_e2m1 * max_e4m3


class NVFP4DynamicLinear(nn.Module):
    """NVFP4 W4A4 dynamic-quantized linear layer.

    Weights are stored as FP4 (E2M1) packed uint8 with FP8 E4M3 block scale
    factors in SWIZZLED layout.  Activations are dynamically quantized per
    forward call.  GEMM is dispatched via ``nvfp4_gemm`` to the best backend
    among cutedsl / cutlass / cublaslt.

    Args:
        in_features:  Input feature dimension (K).
        out_features: Output feature dimension (N).
        bias:         If ``True``, adds a learnable bias.
        dtype:        Logical dtype of the original weight (bf16 or fp16).
                      Controls the GEMM output dtype.
        gemm_backend: GEMM dispatch policy.
                      ``"auto"`` selects cutedsl > cutlass > cublaslt based on
                      shape and hardware.  ``"cutlass"`` / ``"cublaslt"`` /
                      ``"cutedsl"`` force a specific backend.
        quant_backend: ``"tllm"`` (native CUDA) or ``"fouroversix"``
                       (optional dependency, MSE-based adaptive scaling).
        scale_rule:   Block scale selection rule (tllm backend only).
                      ``"static_6"`` (default) — standard NVFP4.
                      ``"mse"`` / ``"mae"`` / ``"abs_max"`` — adaptive 4/6.
                      Ignored when ``quant_backend="fouroversix"``.
        quant_config:  ``fouroversix.QuantizationConfig`` for the fouroversix
                       backend.  Ignored when ``quant_backend="tllm"``.
                       ``None`` defaults to ``QuantizationConfig()`` (MSE).

    Example:
        >>> linear = nn.Linear(4096, 4096, device="cuda", dtype=torch.bfloat16)
        >>> nvfp4 = NVFP4DynamicLinear.from_linear(linear)
        >>> x = torch.randn(2, 128, 4096, device="cuda", dtype=torch.bfloat16)
        >>> out = nvfp4(x)  # [2, 128, 4096]
    """

    _version: int = 1

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        gemm_backend: str = "auto",
        quant_backend: str = "tllm",
        scale_rule: str = "static_6",
        quant_config: object | None = None,
    ) -> None:
        super().__init__()

        if gemm_backend not in _VALID_GEMM_BACKENDS:
            raise ValueError(
                f"gemm_backend must be one of {_VALID_GEMM_BACKENDS}, "
                f"got '{gemm_backend}'"
            )
        if quant_backend not in _VALID_QUANT_BACKENDS:
            raise ValueError(
                f"quant_backend must be one of {_VALID_QUANT_BACKENDS}, "
                f"got '{quant_backend}'"
            )
        if quant_backend == "tllm" and scale_rule not in _SCALE_RULE_MAP:
            raise ValueError(
                f"scale_rule must be one of {list(_SCALE_RULE_MAP.keys())}, "
                f"got '{scale_rule}'"
            )
        if quant_backend == "fouroversix" and not FOUROVERSIX_AVAILABLE:
            raise ImportError(
                "fouroversix is not installed. "
                "Install it to use quant_backend='fouroversix'."
            )

        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self.gemm_backend = gemm_backend
        self.quant_backend = quant_backend
        self.scale_rule = scale_rule

        # Resolve fouroversix config BEFORE computing quant_range, since QR
        # depends on the scale_rule (e.g. MSE uses max_e4m3=256, not 448).
        if quant_backend == "fouroversix" and quant_config is None:
            from fouroversix import QuantizationConfig as _QC
            quant_config = _QC()

        self.quant_config = quant_config
        self.quant_range = _compute_quant_range(
            quant_backend, quant_config, scale_rule
        )

        # Weight buffers — correct shapes for strict load_state_dict
        self.register_buffer(
            "weight_fp4",
            torch.zeros(out_features, in_features // 2, dtype=torch.uint8),
        )
        # Scale factor shape depends on swizzle pattern; allocate a flat
        # placeholder that _quantize_weight will resize on first call.
        # For load_state_dict, _load_from_state_dict handles shape mismatches.
        self.register_buffer(
            "weight_sf",
            torch.zeros(0, dtype=torch.uint8),
        )
        self.register_buffer(
            "alpha_multiplier",
            torch.zeros(1, dtype=torch.float32),
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.bias = None

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        gemm_backend: str = "auto",
        quant_backend: str = "tllm",
        scale_rule: str = "static_6",
        quant_config: object | None = None,
    ) -> "NVFP4DynamicLinear":
        """Create an ``NVFP4DynamicLinear`` from a pretrained ``nn.Linear``.

        The original float weight is quantized to FP4 and discarded.  If the
        source linear has a bias, it is copied over.

        Args:
            linear:       Source ``nn.Linear`` (must be on CUDA).
            gemm_backend: See class docstring.
            quant_backend: See class docstring.
            scale_rule:   See class docstring.
            quant_config:  See class docstring.

        Returns:
            A new ``NVFP4DynamicLinear`` with quantized weights.
        """
        module = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            dtype=linear.weight.dtype,
            gemm_backend=gemm_backend,
            quant_backend=quant_backend,
            scale_rule=scale_rule,
            quant_config=quant_config,
        )
        device = linear.weight.device
        module = module.to(device)

        module._quantize_weight(linear.weight.data)

        if linear.bias is not None:
            module.bias = nn.Parameter(linear.bias.data.clone())

        return module

    def _quantize_weight(self, weight: torch.Tensor) -> None:
        """Quantize a float weight tensor and fill internal buffers.

        Args:
            weight: [out_features, in_features] tensor in bf16/fp16.
        """
        if self.quant_backend == "tllm":
            weight_amax = weight.abs().amax().float().clamp_min(_EPS)
            global_scale = self.quant_range / weight_amax
            fp4, sf = fp4_quantize(
                weight, global_scale=global_scale, swizzled=True,
                scale_rule=self.scale_rule,
            )
        else:
            from fouroversix import quantize_to_fp4 as _fox_quantize
            q = _fox_quantize(weight, config=self.quant_config)
            fp4 = q.values
            sf = q.scale_factors.view(torch.uint8)
            weight_amax = q.amax.float().clamp_min(_EPS)

        alpha_mult = weight_amax / (self.quant_range ** 2)
        alpha_mult_t = alpha_mult.to(torch.float32).reshape(1)

        self.weight_fp4 = fp4
        self.weight_sf = sf
        self.alpha_multiplier = alpha_mult_t

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: quantize activation, run FP4 GEMM, reshape back.

        Args:
            x: Input tensor of shape ``[..., in_features]``.

        Returns:
            Output tensor of shape ``[..., out_features]``.
        """
        orig_shape = x.shape
        x_2d = x.reshape(-1, self.in_features)
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()

        M = x_2d.shape[0]
        act_fp4, act_sf, act_amax = self._quantize_activation(x_2d)

        # alpha = act_amax * alpha_multiplier  (single scalar multiply)
        alpha = (act_amax * self.alpha_multiplier).to(torch.float32)

        # fouroversix pads activation rows to multiples of 128 and weight
        # rows/cols to Blackwell alignment boundaries, so the actual GEMM
        # operand dimensions may exceed the logical (M, N, K).  Use the
        # real padded shapes for backend constraint checks (K%32, N%32);
        # tllm never pads, so the original dimensions are used directly.
        if self.quant_backend == "fouroversix":
            gemm_M = act_fp4.shape[0]
            gemm_N = self.weight_fp4.shape[0]
            gemm_K = act_fp4.shape[1] * 2
        else:
            gemm_M = M
            gemm_N = self.out_features
            gemm_K = self.in_features

        backend = self._resolve_gemm_backend(gemm_M, gemm_N, gemm_K)
        out = nvfp4_gemm(
            act_fp4, self.weight_fp4,
            act_sf, self.weight_sf,
            alpha,
            output_dtype=self.dtype,
            bias=self.bias,
            backend=backend,
        )

        # Trim padded rows/cols produced by fouroversix back to the
        # original unpadded (M, N) before reshaping to the caller's
        # expected output shape.
        if self.quant_backend == "fouroversix":
            out = out[:M, :self.out_features]

        return out.reshape(*orig_shape[:-1], self.out_features)

    # ------------------------------------------------------------------
    # Activation quantization (dispatched by quant_backend)
    # ------------------------------------------------------------------

    def _quantize_activation(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize a 2-D activation tensor to FP4.

        Args:
            x: [M, K] activation tensor.

        Returns:
            (act_fp4, act_sf, act_amax) where act_amax is a scalar float32
            tensor on the same device.
        """
        if self.quant_backend == "tllm":
            if x.numel() >= _TRITON_AMAX_THRESHOLD:
                act_amax = triton_amax(x).float().clamp_min(_EPS)
                act_gs = self.quant_range / act_amax
            else:
                act_amax, act_gs = cuda_prologue(
                    x, quant_range=self.quant_range, eps=_EPS,
                )
            act_fp4, act_sf = fp4_quantize(
                x, global_scale=act_gs, swizzled=True,
                scale_rule=self.scale_rule,
            )
            return act_fp4, act_sf, act_amax

        # fouroversix path
        from fouroversix import quantize_to_fp4 as _fox_quantize
        q = _fox_quantize(x, config=self.quant_config)
        act_fp4 = q.values
        act_sf = q.scale_factors.view(torch.uint8)
        act_amax = q.amax.float()
        return act_fp4, act_sf, act_amax

    # ------------------------------------------------------------------
    # GEMM backend resolution (excludes cuda_core)
    # ------------------------------------------------------------------

    def _resolve_gemm_backend(self, M: int, N: int, K: int) -> str:
        """Select the best GEMM backend for the given shape.

        Resolution order: cutedsl -> cutlass -> cublaslt.
        cuda_core is excluded because this module only produces SWIZZLED
        activation scale factors.

        Args:
            M: Batch dimension (rows of activation).
            N: Output features.
            K: Input features.

        Returns:
            Resolved backend name (never ``"auto"`` or ``"cuda_core"``).
        """
        if self.gemm_backend != "auto":
            return self.gemm_backend

        cutedsl_ok = (
            IS_CUTLASS_DSL_AVAILABLE
            and K % 32 == 0
            and self.dtype == torch.bfloat16
            and _is_blackwell_sm()
        )
        if cutedsl_ok:
            return "cutedsl"
        if K % 32 == 0 and N % 32 == 0:
            return "cutlass"
        return "cublaslt"

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys,
        unexpected_keys, error_msgs,
    ):
        """Handle weight_sf shape mismatch during load_state_dict.

        weight_sf shape depends on the swizzle pattern and cannot be
        pre-computed without running the quantization op.  We resize the
        buffer to match the checkpoint before the default load.
        """
        sf_key = prefix + "weight_sf"
        if sf_key in state_dict:
            stored = state_dict[sf_key]
            if self.weight_sf.shape != stored.shape:
                self.weight_sf = torch.zeros_like(stored)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        parts = [
            f"in_features={self.in_features}",
            f"out_features={self.out_features}",
            f"bias={self.bias is not None}",
            f"dtype={self.dtype}",
            f"quant_backend='{self.quant_backend}'",
            f"gemm_backend='{self.gemm_backend}'",
            f"quant_range={self.quant_range}",
        ]
        if self.quant_backend == "tllm" and self.scale_rule != "static_6":
            parts.append(f"scale_rule='{self.scale_rule}'")
        return ", ".join(parts)
