# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers shared by the cuda.core launch configuration modules."""

# Standard Library
from functools import lru_cache
from typing import Optional

# Third Party
import torch

# CuRobo
from curobo._src.util.logging import log_warn

#: Shared memory per SM assumed when the device query is unavailable.
FALLBACK_SM_SHARED_MEM_CAPACITY = 100 * 1024


def ceil_div(a: int, b: int) -> int:
    """Ceiling division helper"""
    return (a + b - 1) // b


@lru_cache(maxsize=None)
def _query_sm_shared_memory_capacity(device_id: int) -> int:
    """Query shared memory per SM for a device, cached per ordinal.

    Args:
        device_id: CUDA device ordinal.

    Returns:
        Shared memory per SM in bytes, or the fallback if the query fails.
    """
    # Third Party
    import cuda.bindings.runtime as cudart

    err, capacity = cudart.cudaDeviceGetAttribute(
        cudart.cudaDeviceAttr.cudaDevAttrMaxSharedMemoryPerMultiprocessor, device_id
    )
    if err != cudart.cudaError_t.cudaSuccess or capacity <= 0:
        log_warn(
            f"Failed to query shared memory per SM for device {device_id} ({err}); "
            f"assuming {FALLBACK_SM_SHARED_MEM_CAPACITY} bytes."
        )
        return FALLBACK_SM_SHARED_MEM_CAPACITY
    return int(capacity)


def get_sm_shared_memory_capacity(device_id: Optional[int] = None) -> int:
    """Get total shared memory per SM, which is architecture dependent.

    Args:
        device_id: CUDA device ordinal. Defaults to the current device.

    Returns:
        Shared memory per SM in bytes.
    """
    if device_id is None:
        if not torch.cuda.is_available():
            return FALLBACK_SM_SHARED_MEM_CAPACITY
        device_id = torch.cuda.current_device()
    return _query_sm_shared_memory_capacity(device_id)
