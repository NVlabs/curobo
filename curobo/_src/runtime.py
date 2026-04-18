# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""cuRobo runtime configuration flags.

These module-level flags control runtime behavior. Set them in Python **before**
creating any cuRobo objects:

.. code-block:: python

   import curobo._src.runtime as runtime

   runtime.cuda_graphs = False   # disable CUDA graph capture
   runtime.torch_compile = True  # enable torch.compile
"""

from pathlib import Path

# --- torch.compile ---

#: Enable ``torch.compile`` for eligible cuRobo kernels.
torch_compile: bool = False

#: Enable ``torch.compile`` for operations that are slow to compile.
torch_compile_slow: bool = False

#: Enable ``torch.jit.script`` for eligible cuRobo functions.
torch_jit: bool = False

# --- CUDA execution ---

#: Enable CUDA graph capture for solvers. Significantly improves throughput
#: at the cost of a one-time warm-up overhead.
cuda_graphs: bool = True

#: Enable CUDA graph reset when tensor memory references change.
#: Requires CUDA 12.0+.
cuda_graph_reset: bool = False

#: Enable separate CUDA streams for parallel kernel execution.
cuda_streams: bool = True

#: Enable CUDA event-based timers for profiling.
cuda_event_timers: bool = True

# --- Backend ---

#: Enable the CUDA core runtime backend.
cuda_core_backend: bool = True

#: Kernel backend to use. One of ``"auto"``, ``"cuda_core"``, or ``"pybind"``.
#: ``"auto"`` tries ``cuda_core`` first and falls back to ``pybind``.
kernel_backend: str = "auto"

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
