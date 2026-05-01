# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Standard Library
from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any, Optional, Tuple, Union

# Third Party
import torch
import warp as wp
from packaging import version

from curobo._src.runtime import debug_cuda_compile as cuda_debug_compile

# CuRobo
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_debug, log_info


def _set_warp_name(func: Callable[..., Any], name: str) -> Callable[..., Any]:
    func.__name__ = name
    func.__qualname__ = name
    return func


def warp_func(name: str) -> Callable[[Callable[..., Any]], Any]:
    """Return a decorator that names a template before ``wp.func`` sees it."""

    def _decorate(func: Callable[..., Any]) -> Any:
        return wp.func(module="unique")(_set_warp_name(func, name))

    return _decorate


def warp_kernel(name: str, **kwargs: Any) -> Callable[[Callable[..., Any]], Any]:
    """Return a decorator that names a template before ``wp.kernel`` sees it."""

    def _decorate(func: Callable[..., Any]) -> Any:
        return wp.kernel(module="unique", **kwargs)(_set_warp_name(func, name))

    return _decorate


def warp_constant_suffix(*values: object) -> str:
    """Hash compile-time constants into a stable Warp symbol suffix.

    Warp function/kernel names must change when their closed-over
    compile-time constants change, otherwise generated modules can collide.
    Callers should pass resolved primitive constants or tuples/lists of
    primitive constants, not tensors or mutable runtime objects.
    """
    return hashlib.sha1(repr(values).encode("ascii")).hexdigest()[:12]


def init_warp(
    quiet=True,
    verbose=False,
    lineinfo=False,
    line_directives=False,
    print_launches=False,
    device_cfg: DeviceCfg = DeviceCfg()
):
    wp.config.quiet = quiet
    wp.config.verbose = verbose
    wp.config.lineinfo = lineinfo
    wp.config.line_directives = line_directives
    wp.config.print_launches = print_launches
    wp.config.mode = "release" if not cuda_debug_compile else "debug"
    wp.config.verify_cuda = cuda_debug_compile
    # wp.config.max_unroll = 8
    # wp.config.optimization_level = 2


    wp.init()

    # Print GPU info for debugging
    log_debug(f"Warp initialized - Version: {wp.config.version}")
    log_debug(f"Warp Devices: {wp.get_devices()}")

    return True


def warp_support_sdf_struct(wp_module=None):
    if wp_module is None:
        wp_module = wp
    wp_version = wp_module.config.version

    if version.parse(wp_version) < version.parse("1.0.0"):
        log_info(
            "Warp version is "
            + wp_version
            + " < 1.0.0, using older sdf kernels."
            + "No issues expected."
        )
        return False
    return True


def warp_support_kernel_key(wp_module=None):
    if wp_module is None:
        wp_module = wp
    wp_version = wp_module.config.version

    if version.parse(wp_version) < version.parse("1.2.1"):
        log_info(
            "Warp version is "
            + wp_version
            + " < 1.2.1, using, creating global constant to trigger kernel generation."
        )
        return False
    return True


def warp_support_bvh_constructor_type(wp_module=None):
    if wp_module is None:
        wp_module = wp
    wp_version = wp_module.config.version

    if version.parse(wp_version) < version.parse("1.6.0"):
        log_info(
            "Warp version is "
            + wp_version
            + " < 1.6.0, using, creating global constant to trigger kernel generation."
        )
        return False
    return True



def get_warp_device_stream(
    tensor_or_device: Union[torch.Tensor, torch.device],
) -> Tuple[wp.Device, Optional[wp.Stream]]:
    """Get Warp device and stream for a tensor or device.

    The stream respects the current PyTorch stream context, which is
    critical for CUDA graph capture compatibility. During CUDA graph
    capture, this returns the capture stream rather than the default stream.

    Args:
        tensor_or_device: A PyTorch CUDA/CPU tensor or torch.device

    Returns:
        Tuple of (wp.Device, wp.Stream or None).
        Stream is None for CPU tensors/devices.

    Example:
        >>> device, stream = get_warp_device_stream(my_tensor)
        >>> wp.launch(kernel, dim=n, inputs=[...], device=device, stream=stream)
    """
    if isinstance(tensor_or_device, torch.Tensor):
        torch_device = tensor_or_device.device
        is_cuda = tensor_or_device.is_cuda
    else:
        torch_device = tensor_or_device
        is_cuda = torch_device.type == "cuda"

    wp_device = wp.device_from_torch(torch_device)
    if is_cuda:
        pt_stream = torch.cuda.current_stream(torch_device)
        stream = wp.stream_from_torch(pt_stream)
    else:
        stream = None
    return wp_device, stream
