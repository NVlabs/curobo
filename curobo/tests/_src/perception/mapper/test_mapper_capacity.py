# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU configuration checks for explicit mapper pool capacities."""
from __future__ import annotations

import pytest

from curobo._src.perception.mapper.constants import DEFAULT_HASH_LAYOUT
from curobo._src.perception.mapper.mapper_cfg import MapperCfg


def test_pool_override_updates_dependent_capacities() -> None:
    """An explicit small pool bypasses the heuristic floor and sizes scratch/hash."""
    cfg = MapperCfg(
        extent_meters_xyz=(2.0, 2.0, 2.0),
        image_height=480,
        image_width=640,
        voxel_size=0.02,
        truncation_distance=0.08,
        max_blocks_override=1349,
        color_grid_size=8,
        device="cpu",
    )
    assert cfg.max_blocks == 1349
    assert cfg.hash_capacity == 2698
    assert cfg.max_visible_blocks_per_integration == 1349
    assert cfg.color_grid_size == 8


def test_default_pool_stays_heuristic() -> None:
    """Omitting the override preserves existing allocation policy."""
    cfg = MapperCfg(
        extent_meters_xyz=(2.0, 2.0, 2.0),
        image_height=480,
        image_width=640,
        voxel_size=0.005,
        truncation_distance=0.08,
        device="cpu",
    )
    assert cfg.max_blocks == 225000


@pytest.mark.parametrize("capacity", [0, -1, 1.5, True, DEFAULT_HASH_LAYOUT.max_pool_idx + 1])
def test_invalid_pool_capacity_rejected(capacity: object) -> None:
    """Reject capacities that cannot be represented by the native hash pool."""
    with pytest.raises(ValueError, match="max_blocks_override"):
        MapperCfg(
            extent_meters_xyz=(2.0, 2.0, 2.0),
            image_height=480,
            image_width=640,
            max_blocks_override=capacity,
            device="cpu",
        )
