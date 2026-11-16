# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared storage helpers for unsigned 32-bit collision-layer membership."""

from __future__ import annotations

from numbers import Integral

import torch
import warp as wp

from curobo._src.util.logging import log_and_raise

COMMON_COLLISION_LAYER = 1


def layer_bits_to_int32(layers: int) -> int:
    """Validate unsigned layer bits and return their signed storage-view value."""
    if not isinstance(layers, Integral) or isinstance(layers, bool) or not 0 <= layers < 2**32:
        log_and_raise("collision layers must be an integer in [0, 2**32)")
    return int(layers) if layers < 2**31 else int(layers) - 2**32


def collision_layers_to_warp(layers: torch.Tensor) -> wp.array:
    """Expose uint32 storage to Warp without a copy or dtype conversion."""
    # Warp's Torch bridge does not accept torch.uint32 directly. A signed view
    # exposes the same bytes; Warp consumes them as unsigned, including bit 31.
    return wp.from_torch(layers.view(torch.int32).view(-1), dtype=wp.uint32)
