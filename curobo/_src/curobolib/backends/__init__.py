# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Backend selector for curobolib kernels.

Supports two backends:
- 'cuda_core': Runtime compilation using cuda.core (default)
- 'pybind': Pre-compiled PyBind11 extensions (fallback)

Backend selection priority:
1. Environment variable CUROBO_KERNEL_BACKEND (if set)
2. Config option cuda_core_backend
3. Auto-detect available backends with intelligent fallback
"""

import os
from typing import Any, Optional

import curobo._src.runtime as runtime
from curobo._src.runtime import cuda_core_backend as enable_cuda_core_runtime
from curobo._src.util.logging import log_and_raise, log_info, log_warn

__all__ = ["get_backend", "get_backend_name"]

_backend_modules: Optional[dict] = None
_backend_name: Optional[str] = None


def _try_load_cuda_core_backend() -> Optional[dict]:
    """Try to load cuda.core runtime compilation backend"""
    try:
        from curobo._src.curobolib.backends.cuda_core_backend import (
            dynamics as cc_dyn,
        )
        from curobo._src.curobolib.backends.cuda_core_backend import (
            geometry as cc_geo,
        )
        from curobo._src.curobolib.backends.cuda_core_backend import (
            kinematics as cc_kin,
        )
        from curobo._src.curobolib.backends.cuda_core_backend import (
            optimization as cc_opt,
        )
        from curobo._src.curobolib.backends.cuda_core_backend import (
            pba as cc_pba,
        )
        from curobo._src.curobolib.backends.cuda_core_backend import (
            trajectory as cc_traj,
        )

        return {
            "kinematics": cc_kin,
            "optimization": cc_opt,
            "trajectory": cc_traj,
            "geometry": cc_geo,
            "dynamics": cc_dyn,
            "pba": cc_pba,
        }
    except ImportError as e:
        # cuda.core not available
        log_and_raise(f"Error loading cuda.core backend: {e}")
        return None


def _try_load_pybind_backend() -> Optional[dict]:
    """Try to load pre-compiled PyBind11 extensions"""
    try:
        from curobo._src.curobolib.backends.pybind import (
            geometry as _geo,
        )
        from curobo._src.curobolib.backends.pybind import (
            kinematics as _kin,
        )
        from curobo._src.curobolib.backends.pybind import (
            optimization as _opt,
        )
        from curobo._src.curobolib.backends.pybind import (
            trajectory as _traj,
        )

        return {
            "kinematics": _kin,
            "optimization": _opt,
            "trajectory": _traj,
            "geometry": _geo,
        }
    except ImportError:
        # Pybind extensions not compiled
        return None


def _load_cuda_core_backend() -> dict:
    """Load cuda.core backend or raise error"""
    modules = _try_load_cuda_core_backend()
    if modules is None:
        raise RuntimeError(
            "CUDA.core backend not available. Install with: pip install 'cuda-core[cu12]'"
        )
    return modules


def _load_pybind_backend() -> dict:
    """Load pybind backend or raise error"""
    modules = _try_load_pybind_backend()
    if modules is None:
        raise RuntimeError("PyBind backend not available. Compile with: pip install .[compiled]")
    return modules


def _auto_select_backend() -> tuple:
    """Auto-select the best available backend.

    Priority:
    1. Environment variable CUROBO_KERNEL_BACKEND (if set)
    2. Config option cuda_core_backend
    3. Auto-detect available backends with intelligent fallback

    Returns:
        tuple: (backend_name, backend_modules)

    Raises:
        RuntimeError: If no backend is available
    """
    # Check runtime flag override
    backend_pref = runtime.kernel_backend.lower()

    if backend_pref == "cuda_core":
        return ("cuda_core", _load_cuda_core_backend())
    elif backend_pref == "pybind":
        return ("pybind", _load_pybind_backend())

    # Use config preference

    if enable_cuda_core_runtime:
        # Try cuda_core first (preferred)
        cuda_core_modules = _try_load_cuda_core_backend()
        if cuda_core_modules:
            return ("cuda_core", cuda_core_modules)

        # Fallback to pybind
        pybind_modules = _try_load_pybind_backend()
        if pybind_modules:
            log_warn(
                "cuda.core backend not available, using compiled pybind backend. "
                "For runtime compilation, install: pip install 'cuda-core[cu12]'"
            )
            return ("pybind", pybind_modules)
    else:
        # Try pybind first
        pybind_modules = _try_load_pybind_backend()
        if pybind_modules:
            return ("pybind", pybind_modules)

        # Fallback to cuda_core
        cuda_core_modules = _try_load_cuda_core_backend()
        if cuda_core_modules:
            log_warn(
                "Pybind backend not available, using cuda.core runtime compilation. "
                "To compile extensions: pip install .[compiled]",
                UserWarning,
            )
            return ("cuda_core", cuda_core_modules)

    # No backend available
    raise RuntimeError(
        "No curobo kernel backend available!\n"
        "Install one of:\n"
        "  1. cuda.core (recommended, no compilation): pip install 'cuda-core[cu12]'\n"
        "  2. pybind (compile from source): CUROBO_USE_PYBIND=1 pip install -e . --no-build-isolation"
    )


def get_backend() -> dict:
    """Get the active backend modules"""
    global _backend_modules, _backend_name

    if _backend_modules is None:
        _backend_name, _backend_modules = _auto_select_backend()
        log_info(f"CuRobo kernel backend: '{_backend_name}'")

    return _backend_modules


def get_backend_name() -> str:
    """Get the name of the active backend"""
    if _backend_modules is None:
        get_backend()  # Initialize
    return _backend_name


class _BackendProxy:
    """Lazy proxy for a backend kernel module.

    Returned by the package-level ``__getattr__`` so that
    ``from curobo._src.curobolib.backends import kinematics as kinematics_cu``
    succeeds at import time even when no backend (cuda.core / pybind) can be
    loaded. Backend selection only runs the first time an attribute is accessed
    on the proxy, allowing curobo modules to be imported on hosts without
    cuda-core (CPU-only docs builds, type-checkers, IDE indexers, examples that
    only construct configs).

    If the active backend does not expose the requested module (e.g. ``dynamics``
    and ``pba`` are only implemented for cuda.core), accessing any attribute on
    the proxy raises a clear error via ``log_and_raise`` instead of a bare
    ``KeyError``.
    """

    _MODULES = frozenset(
        {"kinematics", "optimization", "trajectory", "geometry", "dynamics", "pba"}
    )

    def __init__(self, name: str) -> None:
        self._name = name
        self._mod: Optional[Any] = None

    def __getattr__(self, attr: str) -> Any:
        if self._mod is None:
            backends = get_backend()
            if self._name not in backends:
                log_and_raise(
                    f"Backend module '{self._name}' is not available in the "
                    f"active backend '{get_backend_name()}'. This kernel module "
                    f"is currently only implemented in the cuda_core backend."
                )
            self._mod = backends[self._name]
        return getattr(self._mod, attr)


def __getattr__(name: str) -> Any:
    """Return a lazy proxy for a backend kernel module.

    Backend selection is deferred until the first attribute access on the
    returned proxy, so ``from curobo._src.curobolib.backends import X`` does
    not require ``cuda-core`` (or any backend) to be installed at import time.
    """
    if name in _BackendProxy._MODULES:
        return _BackendProxy(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
