# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cuRobo runtime configuration flags.

These module-level flags control runtime behavior. Set them in Python **before**
creating any cuRobo objects:

.. code-block:: python

   import curobo._src.runtime as runtime

   runtime.cuda_graphs = False   # disable CUDA graph capture
   runtime.torch_compile = True  # enable torch.compile
   runtime.kernel_launch_sm_count = 32  # select small-GPU launch policies
"""

from pathlib import Path

from curobo._src.util.logging import log_and_raise

# --- torch.compile ---

#: Enable ``torch.compile`` for eligible cuRobo kernels.
torch_compile: bool = True

#: Enable ``torch.compile`` for operations that are slow to compile.
torch_compile_slow: bool = False

#: Enable ``torch.jit.script`` for eligible cuRobo functions.
torch_jit: bool = True

# --- CUDA execution ---

#: Enable CUDA graph capture for solvers. Significantly improves throughput
#: at the cost of a one-time warm-up overhead.
cuda_graphs: bool = True

#: Skip repeat runtime validation on performance-sensitive paths. Enable only
#: when tensor references and shapes remain stable between calls.
skip_checks: bool = True

#: Enable CUDA graph reset when tensor memory references change.
#: Requires CUDA 12.0+.
cuda_graph_reset: bool = False

#: Enable separate CUDA streams for parallel kernel execution.
cuda_streams: bool = True

#: Enable CUDA event-based timers for profiling.
cuda_event_timers: bool = True

#: Override the SM count used to select kernel launch policies. This does not
#: partition the GPU or restrict kernels to this number of physical SMs. Set it
#: before constructing cuRobo objects or capturing CUDA graphs.
kernel_launch_sm_count: int | None = None

_SMALL_GPU_MAX_SM_COUNT = 32


def resolve_kernel_launch_sm_count(actual_sm_count: int) -> int:
    """Return the SM count to use for launch-policy selection."""
    if actual_sm_count <= 0:
        log_and_raise("actual_sm_count must be positive")
    if kernel_launch_sm_count is None:
        return actual_sm_count
    if (
        isinstance(kernel_launch_sm_count, bool)
        or not isinstance(kernel_launch_sm_count, int)
        or kernel_launch_sm_count <= 0
    ):
        log_and_raise("kernel_launch_sm_count must be a positive integer or None")
    return kernel_launch_sm_count


def use_small_gpu_launch_policy(actual_sm_count: int) -> bool:
    """Return whether kernels should use the small-GPU launch policy."""
    return resolve_kernel_launch_sm_count(actual_sm_count) <= _SMALL_GPU_MAX_SM_COUNT


# --- Paths ---

#: Directory for cuRobo outputs (example results, saved configs, etc.).
#: Defaults to ``~/.cache/curobo``.
cache_dir: str = str(Path.home() / ".cache" / "curobo")

# --- Debug ---

#: Enable general debug mode (additional checks and logging).
debug: bool = False

#: Enable CUDA graph debug mode.
debug_cuda_graphs: bool = False

#: Enable debug compilation flags for CUDA kernels.
debug_cuda_compile: bool = False

#: If ``True``, disable CUDA graphs and check for NaN values during optimization.
debug_nan: bool = False

#: Enable debug timing output and print results to stdout.
debug_timers: bool = False

#: Print trajectory optimization results for debugging.
debug_trajopt: bool = False

#: Enable profiler hooks (e.g., for Nsight Systems).
profiler: bool = False
