#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Standard Library
import os
from functools import lru_cache
from typing import Optional

# Third Party
import torch
from packaging import version

# CuRobo
from curobo.util.logger import log_info, log_warn


def find_first_idx(array, value, EQUAL=False):
    if EQUAL:
        f_idx = torch.nonzero(array >= value, as_tuple=False)[0].item()
    else:
        f_idx = torch.nonzero(array > value, as_tuple=False)[0].item()
    return f_idx


def find_last_idx(array, value):
    f_idx = torch.nonzero(array <= value, as_tuple=False)[-1].item()
    return f_idx


def is_cuda_graph_available():
    if version.parse(torch.__version__) < version.parse("1.10"):
        log_warn("Disabling CUDA Graph as pytorch < 1.10")
        return False
    return True


def is_cuda_graph_reset_available():
    reset_cuda_graph = os.environ.get("CUROBO_TORCH_CUDA_GRAPH_RESET")
    if reset_cuda_graph is not None:
        if bool(int(reset_cuda_graph)):
            if version.parse(torch.version.cuda) >= version.parse("12.0"):
                return True
        if not bool(int(reset_cuda_graph)):
            return False
    return False


def is_torch_compile_available():
    force_compile = os.environ.get("CUROBO_TORCH_COMPILE_FORCE")
    if force_compile is not None and bool(int(force_compile)):
        return True
    if version.parse(torch.__version__) < version.parse("2.0"):
        log_info("Disabling torch.compile as pytorch < 2.0")
        return False

    env_variable = os.environ.get("CUROBO_TORCH_COMPILE_DISABLE")

    if env_variable is None:
        log_info("Environment variable for CUROBO_TORCH_COMPILE is not set, Disabling.")

        return False

    if bool(int(env_variable)):
        log_info("Environment variable for CUROBO_TORCH_COMPILE is set to Disable")
        return False

    log_info("Environment variable for CUROBO_TORCH_COMPILE is set to Enable")

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
    try:
        # Third Party
        import triton
    except:
        log_info("Could not find triton, disabling Torch Compile.")
        return False

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
    force_jit: bool = False, dynamic: bool = True, only_valid_for_compile: bool = False
):
    if not force_jit and is_torch_compile_available():
        return torch.compile(options=get_torch_compile_options(), dynamic=dynamic)
    elif not only_valid_for_compile:
        return torch.jit.script
    else:
        return empty_decorator


def is_lru_cache_avaiable():
    use_lru_cache = os.environ.get("CUROBO_USE_LRU_CACHE")
    if use_lru_cache is not None:
        return bool(int(use_lru_cache))
    log_info("Environment variable for CUROBO_USE_LRU_CACHE is not set, Enabling as default.")
    return False


def get_cache_fn_decorator(maxsize: Optional[int] = None):
    if is_lru_cache_avaiable():
        return lru_cache(maxsize=maxsize)
    else:
        return empty_decorator


def empty_decorator(function):
    return function


@get_torch_jit_decorator()
def round_away_from_zero(x: torch.Tensor) -> torch.Tensor:
    y = torch.trunc(x + 0.5 * torch.sign(x))
    return y
