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
SimpleTuner -- minimal tactic profiler for Cute DSL NVFP4 GEMM.

Replaces TRT-LLM's 700-line AutoTuner with a lightweight version:
- Profiles a fixed set of tactics on first call for each shape bucket
- Caches the best tactic by (M_bucket, N, K)
- Thread-safe with a simple dict + lock

Typical usage is internal to CuteDSLNVFP4Runner.
"""

import threading
import torch
from typing import Any, Callable, Dict, List, Tuple


def _last_power_of_2(x: int) -> int:
    """Return the largest power of 2 <= x."""
    if x <= 0:
        return 1
    p = 1
    while p * 2 <= x:
        p *= 2
    return p


class SimpleTuner:
    """Minimal tactic profiler for Cute DSL kernels.

    On first call for a given shape bucket, profiles all candidate tactics
    and caches the fastest one.

    Args:
        warmup: Number of warmup iterations before timing.
        repeat: Number of timed iterations for averaging.
    """

    def __init__(self, warmup: int = 3, repeat: int = 5):
        self._cache: Dict[Tuple, Any] = {}
        self._lock = threading.Lock()
        self._warmup = warmup
        self._repeat = repeat

    def choose_best(
        self,
        shape_key: Tuple[int, int, int],
        tactics: List[Any],
        run_fn: Callable[[Any], torch.Tensor],
    ) -> Any:
        """Choose the best tactic for a given shape.

        Args:
            shape_key: (M, N, K) -- M is bucketed to last power of 2.
            tactics: List of candidate tactics to profile.
            run_fn: Function that takes a tactic and runs the GEMM, returning output tensor.

        Returns:
            The tactic with the lowest average runtime.
        """
        m, n, k = shape_key
        cache_key = (_last_power_of_2(m), n, k)

        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Profile outside the lock (GPU work)
        best_tactic = tactics[0]
        best_time = float("inf")

        for tactic in tactics:
            try:
                # Warmup
                for _ in range(self._warmup):
                    run_fn(tactic)
                torch.cuda.synchronize()

                # Time
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                times = []
                for _ in range(self._repeat):
                    start.record()
                    run_fn(tactic)
                    end.record()
                    torch.cuda.synchronize()
                    times.append(start.elapsed_time(end))

                avg_ms = sum(times) / len(times)
                if avg_ms < best_time:
                    best_time = avg_ms
                    best_tactic = tactic
            except Exception:
                # Skip tactics that fail (e.g. insufficient shared memory)
                continue

        with self._lock:
            self._cache[cache_key] = best_tactic

        return best_tactic
