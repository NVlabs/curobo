# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#


class CuroboRuntime:
    """Global runtime for CuRobo kernel management.

    Manages caches for multiple backends (cuda.core, pybind, warp).
    Backends are lazily initialized on first use.
    """

    def __init__(self):
        # Backend-specific caches (lazy initialized)
        self._cuda_core_cache = None
        self._warp_cache = None  # Future
        self._pybind_cache = None  # If needed

        # Legacy cache names (kept for compatibility)
        self.warp_runtime_kernel_cache = {}
        self.curobo_runtime_kernel_cache = {}

    def get_cuda_core_cache(self):
        """Get cuda.core kernel cache (lazy initialization).

        Returns:
            CudaCoreKernelCache instance
        """
        if self._cuda_core_cache is None:
            from curobo._src.curobolib.backends.cuda_core_backend import CudaCoreKernelCache

            self._cuda_core_cache = CudaCoreKernelCache()
        return self._cuda_core_cache

    def get_warp_cache(self):
        """Get warp kernel cache (future implementation).

        Returns:
            WarpKernelCache instance
        """
        if self._warp_cache is None:
            # Future: from curobo.curobolib.backends.warp_backend import WarpKernelCache
            # self._warp_cache = WarpKernelCache()
            raise NotImplementedError("Warp backend not yet implemented")
        return self._warp_cache


# Global runtime
runtime = None


def init():
    """Initialize global runtime"""
    global runtime
    if runtime is None:
        runtime = CuroboRuntime()


def get_runtime() -> CuroboRuntime:
    """Get or initialize global runtime"""
    if runtime is None:
        init()
    return runtime
