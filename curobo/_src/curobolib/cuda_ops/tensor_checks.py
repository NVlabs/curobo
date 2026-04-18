# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tensor validation utilities for kernel launches.

All tensors passed to CUDA/Warp kernels must be on the expected device, contiguous, and have the
expected dtype. Calling `.contiguous()` as a fallback is unsafe under CUDA graphs because the
conditional allocation may capture a no-op or a copy depending on tensor state at capture time.

These batch check functions assert correctness and raise immediately on violation.
"""

from __future__ import annotations

import torch

from curobo._src.util.logging import log_and_raise


def _check_tensors(
    device: torch.device, expected_dtype: torch.dtype, **tensors: torch.Tensor
) -> None:
    """Validate that every tensor is on the expected device, contiguous, and has the expected dtype.

    Args:
        device: Required device for all tensors.
        expected_dtype: Required dtype for all tensors.
        **tensors: Named tensors to validate. Names are used in error messages.

    Raises:
        ValueError: If any tensor fails a check.
    """
    for name, t in tensors.items():
        if t is None:
            log_and_raise(f"{name}: expected a tensor, got None")
        if t.device != device:
            log_and_raise(
                f"{name}: expected device {device}, got {t.device}",
            )
        if not t.is_contiguous():
            log_and_raise(
                f"{name}: expected contiguous tensor, got strides={t.stride()} "
                f"for shape={tuple(t.shape)}",
            )
        if t.dtype != expected_dtype:
            log_and_raise(
                f"{name}: expected dtype {expected_dtype}, got {t.dtype}",
            )


def check_float32_tensors(device: torch.device, **tensors: torch.Tensor) -> None:
    """Validate all tensors: device + contiguous + float32."""
    _check_tensors(device, torch.float32, **tensors)


def check_float16_tensors(device: torch.device, **tensors: torch.Tensor) -> None:
    """Validate all tensors: device + contiguous + float16."""
    _check_tensors(device, torch.float16, **tensors)


def check_int8_tensors(device: torch.device, **tensors: torch.Tensor) -> None:
    """Validate all tensors: device + contiguous + int8."""
    _check_tensors(device, torch.int8, **tensors)


def check_uint8_tensors(device: torch.device, **tensors: torch.Tensor) -> None:
    """Validate all tensors: device + contiguous + uint8."""
    _check_tensors(device, torch.uint8, **tensors)


def check_int16_tensors(device: torch.device, **tensors: torch.Tensor) -> None:
    """Validate all tensors: device + contiguous + int16."""
    _check_tensors(device, torch.int16, **tensors)


def check_int32_tensors(device: torch.device, **tensors: torch.Tensor) -> None:
    """Validate all tensors: device + contiguous + int32."""
    _check_tensors(device, torch.int32, **tensors)


def check_bool_tensors(device: torch.device, **tensors: torch.Tensor) -> None:
    """Validate all tensors: device + contiguous + bool."""
    _check_tensors(device, torch.bool, **tensors)
