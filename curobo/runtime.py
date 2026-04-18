# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public runtime configuration for cuRobo.

Re-exports compile-time and runtime flags from :mod:`curobo._src.runtime` (for example CUDA
graph usage, debug switches, cache directory, and torch compile options) so callers can use
``import curobo.runtime`` or ``from curobo import runtime`` without depending on private paths.
"""
from curobo._src.runtime import (
    cache_dir,
    cuda_core_backend,
    cuda_event_timers,
    cuda_graph_reset,
    cuda_graphs,
    cuda_streams,
    debug,
    debug_cuda_compile,
    debug_cuda_graphs,
    debug_nan,
    debug_timers,
    debug_trajopt,
    kernel_backend,
    profiler,
    torch_compile,
    torch_compile_slow,
    torch_jit,
)

__all__ = [
    "cache_dir",
    "cuda_core_backend",
    "cuda_event_timers",
    "cuda_graph_reset",
    "cuda_graphs",
    "cuda_streams",
    "debug",
    "debug_cuda_compile",
    "debug_cuda_graphs",
    "debug_nan",
    "debug_timers",
    "debug_trajopt",
    "kernel_backend",
    "profiler",
    "torch_compile",
    "torch_compile_slow",
    "torch_jit",
]
