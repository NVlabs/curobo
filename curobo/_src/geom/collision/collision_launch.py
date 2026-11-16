# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared block sizing for tiled sphere-obstacle collision kernels."""

from curobo._src.util.logging import log_and_raise


def select_collision_block_dim(num_spheres: int) -> int:
    """Round sphere count up to 32, 64, or 128 threads, capped at 128."""
    if num_spheres < 1:
        log_and_raise("num_spheres must be positive")
    if num_spheres <= 32:
        return 32
    if num_spheres <= 64:
        return 64
    return 128
