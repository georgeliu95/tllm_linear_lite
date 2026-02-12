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
Cute DSL NVFP4 GEMM backend for tllm_linear_lite.

Requires optional dependencies:
    pip install nvidia-cutlass-dsl>=4.3.4 cuda-core apache-tvm-ffi==0.1.6

Usage:
    from tllm_linear_lite.cutedsl import IS_CUTLASS_DSL_AVAILABLE
    if IS_CUTLASS_DSL_AVAILABLE:
        from tllm_linear_lite.cutedsl.runner import CuteDSLNVFP4Runner
"""

import platform
import logging

logger = logging.getLogger(__name__)

IS_CUTLASS_DSL_AVAILABLE = False

if platform.system() == "Windows":
    logger.debug("cutlass DSL is not supported on Windows")
else:
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401
        IS_CUTLASS_DSL_AVAILABLE = True
        logger.info("cutlass DSL is available (nvidia-cutlass-dsl found)")
    except ImportError as e:
        logger.warning(
            f"cutlass DSL is not available: {e}. "
            f"Install with: pip install nvidia-cutlass-dsl>=4.3.4 cuda-core apache-tvm-ffi==0.1.6"
        )
