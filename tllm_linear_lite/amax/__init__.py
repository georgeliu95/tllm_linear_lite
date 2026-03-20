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

from tllm_linear_lite.amax.triton_amax import triton_amax, triton_amax_partial  # noqa: F401
from tllm_linear_lite.amax.cuda_amax import cuda_amax, cuda_prologue  # noqa: F401

try:
    from tllm_linear_lite.amax.cutedsl_amax import cutedsl_amax, CUTEDSL_AMAX_AVAILABLE  # noqa: F401
except ImportError:
    cutedsl_amax = None  # type: ignore[assignment]
    CUTEDSL_AMAX_AVAILABLE = False
