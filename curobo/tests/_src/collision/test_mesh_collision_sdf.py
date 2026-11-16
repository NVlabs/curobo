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
from curobo._src.types.device_cfg import DeviceCfg


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
def test_swept_mesh_collision_cost_matches_cuboid(cuda_device_cfg):
    """Swept mesh queries must match the equivalent analytic cuboid."""
    cube_edge = 0.05
    pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    cuboid = Cuboid(name="box", dims=[cube_edge, cube_edge, cube_edge], pose=pose)
    trimesh_box = cuboid.get_trimesh_mesh()
    mesh = Mesh(
        name="mesh_box",
        vertices=trimesh_box.vertices.tolist(),
        faces=trimesh_box.faces.reshape(-1).tolist(),
        pose=pose,
    )

    def swept_cost(scene_cfg: SceneCfg) -> torch.Tensor:
        checker = SceneCollision.from_config(
            SceneCollisionCfg(
                device_cfg=cuda_device_cfg,
                scene_model=scene_cfg,
                cache={"cuboid": 4, "mesh": 4},
            )
        )
        spheres = torch.tensor(
            [
                [
                    [[-0.1, 0.0, 0.0, 0.02]],
                    [[0.0, 0.0, 0.0, 0.02]],
                    [[0.1, 0.0, 0.0, 0.02]],
                ]
            ],
            device=cuda_device_cfg.device,
            dtype=torch.float32,
        )
        buffer = CollisionBuffer.from_shape(spheres.shape, cuda_device_cfg)
        return checker.get_swept_sphere_distance_raw(
            query_spheres=spheres,
            collision_buffer=buffer,
            weight=torch.tensor([1.0], device=cuda_device_cfg.device),
            activation_distance=torch.tensor([0.01], device=cuda_device_cfg.device),
            trajectory_dt=torch.tensor([0.1], device=cuda_device_cfg.device),
        )

    cuboid_cost = swept_cost(SceneCfg(cuboid=[cuboid]))
    mesh_cost = swept_cost(SceneCfg(mesh=[mesh]))
    combined_cost = swept_cost(SceneCfg(cuboid=[cuboid], mesh=[mesh]))

    torch.testing.assert_close(mesh_cost, cuboid_cost, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(combined_cost, cuboid_cost + mesh_cost, rtol=1e-5, atol=1e-6)
    assert torch.any(mesh_cost > 0.0)


def _swept_mesh_shape_outputs(
    cuda_device_cfg: DeviceCfg,
    horizon: int,
    num_spheres: int,
    mesh_cache: int,
    split_states: bool,
    enable_speed_metric: bool,
    sum_distance: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exercise normal native mesh calls across shapes and both state-count branches."""
    sms = torch.cuda.get_device_properties(cuda_device_cfg.device).multi_processor_count
    batch = (4 * sms) // horizon + 1 if split_states else 2
    assert (batch * horizon > 4 * sms) == split_states
    boxes = [
        Cuboid(name=f"box_{i}", dims=[0.05, 0.05, 0.05], pose=[-0.02 * i, 0, 0, 1, 0, 0, 0])
        for i in range(min(mesh_cache, 2))
    ]
    meshes = []
    for box in boxes:
        tri = box.get_trimesh_mesh()
        meshes.append(
            Mesh(
                name=box.name,
                vertices=tri.vertices.tolist(),
                faces=tri.faces.reshape(-1).tolist(),
                pose=box.pose,
            )
        )
    spheres = torch.zeros((batch, horizon, num_spheres, 4), **cuda_device_cfg.as_torch_dict())
    spheres[..., 0] = torch.linspace(0.04, 0.12, num_spheres, **cuda_device_cfg.as_torch_dict())
    spheres[..., 1] = torch.linspace(
        -0.004, 0.004, horizon, **cuda_device_cfg.as_torch_dict()
    ).view(1, horizon, 1)
    spheres[..., 2] = 0.003
    spheres[..., 3] = 0.02

    def evaluate(scene: SceneCfg) -> tuple[torch.Tensor, torch.Tensor]:
        checker = SceneCollision.from_config(
            SceneCollisionCfg(
                device_cfg=cuda_device_cfg,
                scene_model=scene,
                cache={"cuboid": mesh_cache, "mesh": mesh_cache},
            )
        )
        buffer = CollisionBuffer.from_shape(spheres.shape, cuda_device_cfg)
        weight = torch.tensor([1.0], device=cuda_device_cfg.device)
        activation = torch.tensor([0.04], device=cuda_device_cfg.device)
        dt = torch.tensor([0.1], device=cuda_device_cfg.device)

        def invoke() -> torch.Tensor:
            return checker.get_swept_sphere_distance_raw(
                query_spheres=spheres,
                collision_buffer=buffer,
                weight=weight,
                activation_distance=activation,
                trajectory_dt=dt,
                enable_speed_metric=enable_speed_metric,
                sum_distance=sum_distance,
            )

        cost = invoke().clone()
        gradient = buffer.gradient.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replay_cost = invoke()
        buffer.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(cost, replay_cost)
        assert torch.equal(gradient, buffer.gradient)
        assert torch.count_nonzero(gradient[..., 3]) == 0
        if sum_distance:
            assert torch.count_nonzero(buffer.distance.view(-1)[batch * horizon :]) == 0
        return cost, gradient

    cuboid_cost, cuboid_gradient = evaluate(SceneCfg(cuboid=boxes))
    mesh_cost, mesh_gradient = evaluate(SceneCfg(mesh=meshes))
    torch.testing.assert_close(mesh_cost, cuboid_cost, atol=1e-6, rtol=2e-5)
    # Mesh and cuboid gradients have different historical sign conventions.
    # The separate-process native baseline is the mesh-gradient oracle.
    assert torch.count_nonzero(mesh_cost[cuboid_cost == 0]) == 0
    assert torch.any(mesh_cost > 0)

    return mesh_cost, mesh_gradient


@pytest.mark.parametrize("enable_speed_metric", [False, True])
@pytest.mark.parametrize("sum_distance", [False, True])
@pytest.mark.parametrize(
    "horizon,num_spheres,mesh_cache,split_states",
    [(1, 1, 1, False), (2, 65, 3, False), (5, 129, 2, False), (2, 67, 5, True)],
    ids=["h1", "h2-tail", "general-horizon-tail", "split-state-tail"],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_swept_mesh_shape_generality_matches_cuboid(
    cuda_device_cfg: DeviceCfg,
    horizon: int,
    num_spheres: int,
    mesh_cache: int,
    split_states: bool,
    enable_speed_metric: bool,
    sum_distance: bool,
) -> None:
    """Check native mesh costs and graph determinism across supported mesh shapes."""
    _swept_mesh_shape_outputs(
        cuda_device_cfg,
        horizon,
        num_spheres,
        mesh_cache,
        split_states,
        enable_speed_metric,
        sum_distance,
    )
