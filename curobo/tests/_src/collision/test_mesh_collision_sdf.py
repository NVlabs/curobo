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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("as_mesh", [False, True], ids=["cuboid", "mesh"])
@pytest.mark.parametrize(
    "batch,horizon,num_spheres",
    [(1, 1, 1), (1, 3, 43), (2, 3, 43), (1, 3, 128), (1, 3, 129)],
)
def test_swept_collision_preserves_sample_weights(
    cuda_device_cfg, as_mesh: bool, batch: int, horizon: int, num_spheres: int
) -> None:
    """Check analytic costs and gradients across full tiles, tails, and batch boundaries."""
    boxes = [
        Cuboid(
            name=f"box_{i}",
            dims=[1.0, 1.0, 1.0],
            pose=[0.0, 0.01 * i, 0.0, 1.0, 0.0, 0.0, 0.0],
        )
        for i in range(3)
    ]
    scene = SceneCfg(cuboid=boxes)
    if as_mesh:
        scene = scene.get_mesh_world(merge_meshes=False)
    checker = SceneCollision.from_config(
        SceneCollisionCfg(
            device_cfg=cuda_device_cfg,
            scene_model=scene,
            cache={"cuboid": 4, "mesh": 4},
        )
    )
    spheres = torch.zeros(
        (batch, horizon, num_spheres, 4), device=cuda_device_cfg.device, dtype=torch.float32
    )
    spheres[..., 0] = (
        0.55
        + 0.002 * torch.arange(batch, device=cuda_device_cfg.device)[:, None, None]
        + 0.01 * torch.arange(horizon, device=cuda_device_cfg.device)[None, :, None]
        + 0.0001 * (torch.arange(num_spheres, device=cuda_device_cfg.device) % 43)[None, None]
    )
    spheres[..., 3] = 0.1
    if num_spheres > 1:
        # Keep the last sphere active so a missing partial tile changes the result.
        spheres[:, :, num_spheres // 2, 3] = -1.0
    spheres.requires_grad_(True)
    buffer = CollisionBuffer.from_shape(spheres.shape, cuda_device_cfg)
    weight = torch.tensor([2.0], device=cuda_device_cfg.device)
    eta = torch.tensor([0.05], device=cuda_device_cfg.device)
    distance = checker.get_swept_sphere_distance_raw(
        query_spheres=spheres,
        collision_buffer=buffer,
        weight=weight,
        activation_distance=eta,
        trajectory_dt=torch.full((batch,), 0.1, device=cuda_device_cfg.device),
    )

    # Every nonzero segment first samples its current endpoint. Its initial
    # penetration exceeds the segment half-length, so it takes no further samples.
    multiplicity = torch.ones((1, horizon, 1), device=cuda_device_cfg.device)
    if horizon > 1:
        multiplicity[:, 1:] += 1.0
        multiplicity[:, :-1] += 1.0
    enabled = spheres[..., 3] >= 0.0
    expected_cost = 3.0 * weight * multiplicity * (0.125 - (spheres[..., 0] - 0.5))
    expected_cost = torch.where(enabled, expected_cost, 0.0)
    torch.testing.assert_close(distance, expected_cost, atol=2e-5, rtol=2e-5)

    distance.sum().backward()
    expected_grad = torch.zeros_like(spheres)
    expected_grad[..., 0] = torch.where(enabled, -3.0 * weight * multiplicity, 0.0)
    torch.testing.assert_close(spheres.grad, expected_grad, atol=2e-4, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("as_mesh", [False, True], ids=["cuboid", "mesh"])
def test_swept_collision_detects_obstacle_between_free_endpoints(
    cuda_device_cfg, as_mesh: bool
) -> None:
    """Require adaptive samples to detect a box between collision-free endpoints."""
    box = Cuboid(
        name="box",
        dims=[1.0, 1.0, 1.0],
        pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    )
    scene = SceneCfg(cuboid=[box])
    if as_mesh:
        scene = scene.get_mesh_world(merge_meshes=False)
    checker = SceneCollision.from_config(
        SceneCollisionCfg(
            device_cfg=cuda_device_cfg,
            scene_model=scene,
            cache={"cuboid": 1, "mesh": 1},
        )
    )
    spheres = torch.tensor(
        [[[[-0.8, 0.0, 0.0, 0.05]], [[0.8, 0.0, 0.0, 0.05]]]],
        device=cuda_device_cfg.device,
        dtype=torch.float32,
        requires_grad=True,
    )
    weight = torch.ones(1, device=cuda_device_cfg.device)
    eta = torch.tensor([0.02], device=cuda_device_cfg.device)
    static = checker.get_sphere_distance_raw(
        query_spheres=spheres,
        collision_buffer=CollisionBuffer.from_shape(spheres.shape, cuda_device_cfg),
        weight=weight,
        activation_distance=eta,
    )
    torch.testing.assert_close(static, torch.zeros_like(static))
    swept = checker.get_swept_sphere_distance_raw(
        query_spheres=spheres,
        collision_buffer=CollisionBuffer.from_shape(spheres.shape, cuda_device_cfg),
        weight=weight,
        activation_distance=eta,
        trajectory_dt=torch.tensor([0.1], device=cuda_device_cfg.device),
    )
    assert torch.all(swept > 0.0)
    swept.sum().backward()
    assert torch.isfinite(spheres.grad).all()
