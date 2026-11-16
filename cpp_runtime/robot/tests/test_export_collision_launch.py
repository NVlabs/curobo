# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keep native collision exports consistent with Python launch sizing."""

import pytest

from cpp_runtime.robot.tools.export_robot_aot import _collision_block_dim
from curobo._src.geom.collision.collision_launch import select_collision_block_dim


@pytest.mark.parametrize("num_spheres", [1, 32, 33, 64, 65, 128, 129, 176, 512, 1025])
def test_native_collision_width_matches_python(num_spheres: int) -> None:
    """All native discrete and swept export sites share the Python block width."""
    assert _collision_block_dim(num_spheres) == select_collision_block_dim(num_spheres)


@pytest.mark.parametrize("num_spheres", [0, -1])
def test_invalid_sphere_count_is_rejected(num_spheres: int) -> None:
    """Reject invalid layouts consistently in execution and export."""
    for select in (select_collision_block_dim, _collision_block_dim):
        with pytest.raises(ValueError, match="num_spheres must be positive"):
            select(num_spheres)
