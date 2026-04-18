# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA Event-based timer for accurate GPU timing."""

from __future__ import annotations

# Third Party
import torch

# CuRobo
from curobo._src import runtime


class CudaEventTimer:
    """Lightweight CUDA event-based timer for accurate GPU timing.

    Uses CUDA events instead of CPU time + synchronize for more accurate
    GPU timing without blocking the entire device.

    Example:
        >>> timer = CudaEventTimer().start()
        >>> # ... GPU work ...
        >>> elapsed_seconds = timer.stop()

    The timer respects `runtime.cuda_event_timers`. When disabled,
    `stop()` returns 0.0 with minimal overhead.
    """

    def __init__(self):
        """Initialize the timer."""
        self._start: torch.cuda.Event | None = None
        self._end: torch.cuda.Event | None = None

    def start(self) -> CudaEventTimer:
        """Start the timer by recording a CUDA event.

        Returns:
            Self for method chaining.
        """
        if runtime.cuda_event_timers:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed time in seconds.

        This synchronizes only on the end event, not the entire device.

        Returns:
            Elapsed time in seconds, or 0.0 if timing is disabled.
        """
        if runtime.cuda_event_timers and self._start is not None:
            self._end.record()
            self._end.synchronize()
            return self._start.elapsed_time(self._end) / 1000.0
        return 0.0
