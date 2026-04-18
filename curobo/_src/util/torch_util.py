# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Standard Library
import inspect
from functools import wraps

# Third Party
import torch
import torch.autograd.profiler as profiler
from packaging import version

# CuRobo
from curobo import runtime as curobo_runtime
from curobo._src.util.logging import log_info, log_warn


def is_cuda_graph_available():
    if version.parse(torch.__version__) < version.parse("1.10"):
        log_warn("Disabling CUDA Graph as pytorch < 1.10")
        return False
    return True


def is_cuda_graph_reset_available():
    if not curobo_runtime.cuda_graph_reset:
        return False

    # CUDA graph reset requires CUDA 12.0+
    if version.parse(torch.version.cuda) >= version.parse("12.0"):
        return True

    log_warn("CUDA graph reset requires CUDA 12.0+, current version: " + torch.version.cuda)
    return False


def is_torch_compile_available():
    if not curobo_runtime.torch_jit:
        return False

    if not curobo_runtime.torch_compile:
        log_info("torch.compile is explicitly disabled via runtime config")
        return False

    # Check PyTorch version
    if version.parse(torch.__version__) < version.parse("2.0"):
        log_info("Disabling torch.compile as pytorch < 2.0")
        log_warn("Using pytorch < 2.0 is not recommended for performance reasons.")
        return False

    # Check if torch.compile components are available
    try:
        torch.compile
    except:
        log_info("Could not find torch.compile, disabling Torch Compile.")
        return False
    try:
        torch._dynamo
    except:
        log_info("Could not find torch._dynamo, disabling Torch Compile.")
        return False

    log_info("torch.compile is available and enabled")
    return True


def get_torch_compile_options() -> dict:
    options = {}
    if is_torch_compile_available():
        # Third Party
        from torch._inductor import config

        torch._dynamo.config.suppress_errors = True
        use_options = {
            "max_autotune": True,
            "use_mixed_mm": True,
            "conv_1x1_as_mm": True,
            "coordinate_descent_tuning": True,
            "epilogue_fusion": False,
            "coordinate_descent_check_all_directions": True,
            "force_fuse_int_mm_with_mul": True,
            "triton.cudagraphs": False,
            "aggressive_fusion": True,
            "split_reductions": False,
            "worker_start_method": "spawn",
        }
        for k in use_options.keys():
            if hasattr(config, k):
                options[k] = use_options[k]
            else:
                log_info("Not found in torch.compile: " + k)
    return options


def disable_torch_compile_global():
    if is_torch_compile_available():
        torch._dynamo.config.disable = True
        return True
    return False


def set_torch_compile_global_options():
    if is_torch_compile_available():
        # Third Party
        from torch._inductor import config

        torch._dynamo.config.suppress_errors = True
        if hasattr(config, "conv_1x1_as_mm"):
            torch._inductor.config.conv_1x1_as_mm = True
        if hasattr(config, "coordinate_descent_tuning"):
            torch._inductor.config.coordinate_descent_tuning = True
        if hasattr(config, "epilogue_fusion"):
            torch._inductor.config.epilogue_fusion = False
        if hasattr(config, "coordinate_descent_check_all_directions"):
            torch._inductor.config.coordinate_descent_check_all_directions = True
        if hasattr(config, "force_fuse_int_mm_with_mul"):
            torch._inductor.config.force_fuse_int_mm_with_mul = True
        if hasattr(config, "use_mixed_mm"):
            torch._inductor.config.use_mixed_mm = True
        return True
    return False


def get_torch_jit_decorator(
    force_jit: bool = False,
    dynamic: bool = True,
    only_valid_for_compile: bool = False,
    extreme_trace: bool = False,
    slow_to_compile: bool = False,
):
    if not curobo_runtime.torch_jit:
        return empty_decorator
    if not force_jit and is_torch_compile_available():
        if slow_to_compile and not curobo_runtime.torch_compile_slow:
            return empty_decorator
        return torch.compile(options=get_torch_compile_options(), dynamic=dynamic)
    elif not only_valid_for_compile:
        if slow_to_compile and not curobo_runtime.torch_compile_slow:
            return empty_decorator
        return torch.jit.script
    else:
        return empty_decorator


def get_profiler_decorator(str_name: str):
    if curobo_runtime.profiler:
        return profiler.record_function(str_name)
    return empty_decorator


def profile_class_methods(cls):
    """Class decorator that wraps all methods with :func:`torch.autograd.profiler.record_function`.

    Each method is labelled as ``"ClassName/method_name"`` in profiler traces.
    Wrapping is guarded by :attr:`curobo.runtime.profiler` so there
    is zero overhead when profiling is disabled.  Dunder methods (``__init__``,
    ``__repr__``, …) are skipped.

    Args:
        cls: The class whose methods should be profiled.

    Returns:
        The modified class with profiler-wrapped methods.

    Example:
        >>> @profile_class_methods
        ... class MyMapper:
        ...     def integrate(self, data):
        ...         ...
    """
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith("__"):
            continue

        label = f"{cls.__name__}/{name}"

        @wraps(method)
        def _wrapper(*args, _m=method, _l=label, **kwargs):
            if not curobo_runtime.profiler:
                return _m(*args, **kwargs)
            with profiler.record_function(_l):
                return _m(*args, **kwargs)

        setattr(cls, name, _wrapper)
    return cls


def empty_decorator(function):
    return function
