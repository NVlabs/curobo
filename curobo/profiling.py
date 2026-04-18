# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Profiling utilities for cuRobo.

Lightweight timing helpers for measuring GPU work. Timer behavior is gated by
:data:`curobo.runtime.cuda_event_timers`; when disabled, timers are no-ops and return
``0.0`` with minimal overhead.

Example:
    .. code-block:: python

        from curobo.profiling import CudaEventTimer

        timer = CudaEventTimer().start()
        # ... GPU work ...
        elapsed_seconds = timer.stop()
"""
from curobo._src.util.cuda_event_timer import CudaEventTimer

__all__ = [
    "CudaEventTimer",
]
