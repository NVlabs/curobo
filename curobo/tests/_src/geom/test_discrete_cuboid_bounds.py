# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for discrete cuboid pruning across pages and timesteps."""

import pytest
import torch

from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.warp import init_warp


@pytest.mark.parametrize("summed", [False, True])
def test_last_page_contact_and_empty_pose_overwrite(
    cuda_device_cfg: DeviceCfg, summed: bool
) -> None:
    """Keep a contact beyond the first page and clear only the pose that moved away."""
    init_warp(quiet=True, device_cfg=cuda_device_cfg)
    boxes = [
        Cuboid(name=f"box_{i}", pose=[x, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1])
        for i, x in enumerate([10.0] * 256 + [0.0])
    ]
    scene = SceneCollision.from_config(
        SceneCollisionCfg(device_cfg=cuda_device_cfg, scene_model=SceneCfg(cuboid=boxes))
    )
    # Only the last sphere touches the last box, at both timesteps initially.
    query = cuda_device_cfg.to_device([3.0, 0.0, 0.0, 0.1]).repeat(1, 2, 129, 1)
    query[0, :, -1, 0] = 0.58
    buffer = CollisionBuffer.from_shape(query.shape, cuda_device_cfg)
    weight = cuda_device_cfg.to_device([1.0])
    margin = cuda_device_cfg.to_device([0.0])
    for step in range(2):
        if step == 1:
            query[0, 1, -1, 0] = 3.0
        cost = scene.get_sphere_distance_raw(query, buffer, weight, margin, sum_distance=summed)
        expected_cost = torch.zeros_like(cost)
        expected_cost[0, : 2 - step, -1] = 0.02
        expected_gradient = torch.zeros_like(buffer.gradient)
        expected_gradient[0, : 2 - step, -1, 0] = -1.0
        torch.testing.assert_close(cost, expected_cost)
        torch.testing.assert_close(buffer.gradient, expected_gradient)
