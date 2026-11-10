# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Lazy access to CUDA kernel modules.

The proxies keep configuration, documentation, and CPU-only imports independent of
``cuda.core``. The CUDA runtime is imported only when a kernel attribute is first used.
"""

# Standard Library
from importlib import import_module
from types import ModuleType
from typing import Any

# CuRobo
from curobo._src.util.logging import log_and_raise

__all__ = [
    "dynamics",
    "geometry",
    "kinematics",
    "optimization",
    "pba",
    "trajectory",
    "trajectory_seed_fitter",
]


class _CudaCoreModuleProxy:
    """Load one CUDA kernel module when its first attribute is accessed."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._module: ModuleType | None = None

    def __getattr__(self, attribute: str) -> Any:
        if self._module is None:
            module_name = "curobo._src.curobolib.backends.cuda_core_backend." + self._name
            try:
                self._module = import_module(module_name)
            except ImportError as error:
                missing_module = getattr(error, "name", None)
                if missing_module != "cuda" and not str(missing_module).startswith("cuda."):
                    raise
                log_and_raise(
                    "CUDA kernel execution requires cuda.core. Install the CUDA 12 or "
                    f"CUDA 13 package extra for this environment. Original error: {error}",
                    exception_type=ImportError,
                )
        return getattr(self._module, attribute)


_PROXIES = {name: _CudaCoreModuleProxy(name) for name in __all__}


def __getattr__(name: str) -> Any:
    """Return a lazy proxy for a CUDA kernel module."""
    if name in _PROXIES:
        return _PROXIES[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
