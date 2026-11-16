# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUDA Event-based timer for accurate GPU timing."""

from __future__ import annotations

# Standard Library
from typing import Dict, Optional

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
        self._checkpoints: list[tuple[str, torch.cuda.Event]] = []
        self._intervals: Dict[str, float] = {}

    def start(self, stream: Optional[torch.cuda.Stream] = None) -> CudaEventTimer:
        """Start the timer by recording a CUDA event.

        Args:
            stream: Optional CUDA stream on which to record the event.

        Returns:
            Self for method chaining.
        """
        self._checkpoints = []
        self._intervals = {}
        if runtime.cuda_event_timers:
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record(stream)
        return self

    def checkpoint(
        self,
        name: str,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> CudaEventTimer:
        """Record a named, non-synchronizing interval boundary.

        The elapsed interval ends at this checkpoint and starts at the
        previous checkpoint (or :meth:`start`). Reusing a name accumulates
        repeated intervals, which is useful for iterative algorithms. Values
        become available through :attr:`intervals` after :meth:`stop`.

        Args:
            name: Stable name for the interval ending at this checkpoint.
            stream: Optional CUDA stream on which to record the event.

        Returns:
            Self for method chaining.
        """
        if not name:
            raise ValueError("checkpoint name must be non-empty")
        if runtime.cuda_event_timers and self._start is not None:
            event = torch.cuda.Event(enable_timing=True)
            event.record(stream)
            self._checkpoints.append((name, event))
        return self

    def stop(
        self,
        final_interval: Optional[str] = None,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> float:
        """Stop the timer and return elapsed time in seconds.

        This synchronizes only on the end event, not the entire device.

        Args:
            final_interval: Optional name for the interval from the last
                checkpoint (or start) through the stop event.
            stream: Optional CUDA stream on which to record the event.

        Returns:
            Elapsed time in seconds, or 0.0 if timing is disabled.
        """
        if runtime.cuda_event_timers and self._start is not None:
            self._end.record(stream)
            self._end.synchronize()
            previous = self._start
            intervals: Dict[str, float] = {}
            for name, event in self._checkpoints:
                intervals[name] = intervals.get(name, 0.0) + (
                    previous.elapsed_time(event) / 1000.0
                )
                previous = event
            if final_interval is not None:
                if not final_interval:
                    raise ValueError("final interval name must be non-empty")
                intervals[final_interval] = intervals.get(final_interval, 0.0) + (
                    previous.elapsed_time(self._end) / 1000.0
                )
            self._intervals = intervals
            return self._start.elapsed_time(self._end) / 1000.0
        self._intervals = {}
        return 0.0

    @property
    def intervals(self) -> Dict[str, float]:
        """Return named interval durations in seconds after :meth:`stop`."""
        return dict(self._intervals)
