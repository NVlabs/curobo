# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Regression tests for mesh obstacle SDF collision queries."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.types import Cuboid, Mesh, SceneCfg


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_small_mesh_collision_cost_matches_cuboid(cuda_device_cfg):
    """Small mesh collision costs must match the same geometry as an analytic cuboid."""
    cube_edge = 0.05
    pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    sphere_radius = 0.05
    query_distances_m = [0.08, 0.10, 0.50, 1.00]

    def make_checker(scene_cfg: SceneCfg) -> SceneCollision:
        cfg = SceneCollisionCfg(
            device_cfg=cuda_device_cfg,
            scene_model=scene_cfg,
            cache={"cuboid": 4, "mesh": 4},
        )
        return SceneCollision.from_config(cfg)

    def collision_cost(checker: SceneCollision) -> torch.Tensor:
        spheres = torch.tensor(
            [[[[d, 0.0, 0.0, sphere_radius] for d in query_distances_m]]],
            device=cuda_device_cfg.device,
            dtype=torch.float32,
        )
        buf = CollisionBuffer.from_shape(spheres.shape, cuda_device_cfg)
        return checker.get_sphere_distance_raw(
            query_spheres=spheres,
            collision_buffer=buf,
            weight=torch.tensor([1.0], device=cuda_device_cfg.device),
            activation_distance=torch.tensor([0.01], device=cuda_device_cfg.device),
        )

    cuboid = Cuboid(name="box", dims=[cube_edge, cube_edge, cube_edge], pose=pose)
    trimesh_box = cuboid.get_trimesh_mesh()
    mesh = Mesh(
        name="box",
        vertices=trimesh_box.vertices.tolist(),
        faces=trimesh_box.faces.reshape(-1).tolist(),
        pose=pose,
    )

    cuboid_cost = collision_cost(make_checker(SceneCfg(cuboid=[cuboid])))
    mesh_cost = collision_cost(make_checker(SceneCfg(mesh=[mesh])))

    assert torch.allclose(mesh_cost, cuboid_cost)
    assert mesh_cost.flatten()[0] > 0.0
    assert torch.all(mesh_cost.flatten()[1:] == 0.0)
