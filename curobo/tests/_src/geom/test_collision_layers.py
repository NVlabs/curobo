# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collision-layer contracts through scene and rollout-facing query interfaces."""

from __future__ import annotations

import pytest
import torch

from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.types import Capsule, Cuboid, Cylinder, Mesh, SceneCfg, VoxelGrid
from curobo._src.robot.kinematics.kinematics_state import KinematicsState
from curobo._src.robot.types.collision_geometry import RobotCollisionGeometry
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.warp import init_warp


@pytest.fixture(scope="module", autouse=True)
def setup_warp(cuda_device_cfg: DeviceCfg) -> None:
    """Initialize the production Warp backend."""
    init_warp(quiet=True, device_cfg=cuda_device_cfg)


def make_scene(kind: str, device: DeviceCfg, num_envs: int = 1) -> SceneCollision:
    """Create colliding geometry with the same obstacle name in each environment."""
    pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    if kind == "cuboid":
        obstacle = Cuboid(name="target", pose=pose, dims=[0.4, 0.4, 0.4])
    elif kind == "capsule":
        obstacle = Capsule(
            name="target", pose=pose, radius=0.2, base=[0.0, 0.0, -0.2], tip=[0.0, 0.0, 0.2]
        )
    elif kind == "cylinder":
        obstacle = Cylinder(name="target", pose=pose, radius=0.2, height=0.4)
    elif kind == "mesh":
        mesh = Cuboid(name="box", pose=pose, dims=[0.4, 0.4, 0.4]).get_trimesh_mesh()
        obstacle = Mesh(
            name="target", pose=pose, vertices=mesh.vertices.tolist(), faces=mesh.faces.tolist()
        )
    else:
        obstacle = VoxelGrid(
            name="target",
            pose=pose,
            dims=[0.8, 0.8, 0.8],
            voxel_size=0.1,
            feature_dtype=torch.float16,
            feature_tensor=torch.full((512, 1), 0.05, device=device.device, dtype=torch.float16),
        )
    configs = [SceneCfg(**{kind: [obstacle]}) for _ in range(num_envs)]
    return SceneCollision.from_config(
        SceneCollisionCfg(device_cfg=device, scene_model=configs if num_envs > 1 else configs[0])
    )


def query_cost(
    scene: SceneCollision,
    query: torch.Tensor,
    masks: torch.Tensor,
    swept: bool = False,
    indices: torch.Tensor | None = None,
    buffer: CollisionBuffer | None = None,
) -> torch.Tensor:
    """Exercise the same raw query wrappers used by model-backed scene costs."""
    buffer = buffer or CollisionBuffer.from_shape(query.shape, scene.device_cfg)
    args = dict(
        query_spheres=query,
        collision_buffer=buffer,
        weight=torch.ones(1, device=query.device),
        activation_distance=torch.zeros(1, device=query.device),
        env_query_idx=indices,
        sphere_collision_mask=masks,
    )
    if swept:
        return scene.get_swept_sphere_distance_raw(
            **args, trajectory_dt=torch.full((query.shape[0],), 0.1, device=query.device)
        )
    return scene.get_sphere_distance_raw(**args)


@pytest.mark.parametrize("kind", ["cuboid", "capsule", "cylinder", "mesh", "voxel"])
@pytest.mark.parametrize("swept", [False, True])
def test_subsets_and_high_bit(kind: str, swept: bool, cuda_device_cfg: DeviceCfg) -> None:
    """Only matching spheres contribute; matching several bits counts an obstacle once."""
    scene = make_scene(kind, cuda_device_cfg)
    query = torch.tensor([0.25, 0.03, 0.01, 0.1], device=cuda_device_cfg.device)
    query = query.repeat(1, 3, 3, 1).requires_grad_()
    masks = torch.tensor([[1, 1, 1]], device=query.device, dtype=torch.uint32)
    baseline = query_cost(scene, query, masks, swept).clone()
    assert torch.all(baseline > 0)
    baseline.sum().backward()
    baseline_grad = query.grad.clone()
    query.grad = None
    scene.data.set_obstacle_layers("target", (1 << 31) | 4)
    masks.copy_(
        torch.tensor([[1 << 31, 2, (1 << 31) | 4]], device=query.device, dtype=torch.uint32)
    )
    result = query_cost(scene, query, masks, swept)
    expected = baseline.clone()
    expected[:, :, 1] = 0
    torch.testing.assert_close(result, expected)
    result.sum().backward()
    baseline_grad[:, :, 1] = 0
    torch.testing.assert_close(query.grad, baseline_grad)


@pytest.mark.parametrize("swept", [False, True])
def test_environment_rows_and_broadcast(swept: bool, cuda_device_cfg: DeviceCfg) -> None:
    """Environment indices select mask rows, not batch or timestep indices."""
    scene = make_scene("cuboid", cuda_device_cfg, num_envs=2)
    scene.data.set_obstacle_layers("target", 2, env_idx=0)
    scene.data.set_obstacle_layers("target", 4, env_idx=1)
    query = torch.tensor([0.25, 0.03, 0.01, 0.1], device=cuda_device_cfg.device)
    query = query.repeat(3, 3, 2, 1)
    indices = torch.tensor([1, 0, 1], device=query.device, dtype=torch.int32)
    masks = torch.tensor([[2, 4], [4, 2]], device=query.device, dtype=torch.uint32)
    result = query_cost(scene, query, masks, swept, indices).clone()
    assert torch.all(result[:, :, 0] > 0)
    assert torch.count_nonzero(result[:, :, 1]) == 0
    shared = query_cost(scene, query, masks[:1], swept, indices)
    torch.testing.assert_close(shared[1], result[1])
    torch.testing.assert_close(shared[0], result[0].flip(-1))
    torch.testing.assert_close(shared[2], shared[0])


def test_scene_clear_resets_membership(cuda_device_cfg: DeviceCfg) -> None:
    """Reused obstacle slots start on the common layer and retain their allocation."""
    scene = make_scene("cuboid", cuda_device_cfg)
    layers = scene.data.cuboids.obstacle_layers
    scene.data.set_obstacle_layers("target", 4)
    scene.data.clear()
    scene.data.add_obstacle(Cuboid(name="next", pose=[0, 0, 0, 1, 0, 0, 0], dims=[0.4, 0.4, 0.4]))
    assert layers.data_ptr() == scene.data.cuboids.obstacle_layers.data_ptr()
    assert layers[0, 0].item() == 1


@pytest.mark.parametrize("swept", [False, True])
@pytest.mark.parametrize("summed", [False, True])
def test_graph_replays_mutable_layers(swept, summed, cuda_device_cfg):
    """Layer and environment updates change graph results without replacing any pointer."""
    scene = make_scene("cuboid", cuda_device_cfg, num_envs=2)
    query = torch.tensor([0.25, 0.03, 0.01, 0.1], device=cuda_device_cfg.device)
    query = query.repeat(2, 3, 2, 1)
    masks = torch.tensor([[2, 4], [4, 2]], dtype=torch.uint32, device=query.device)
    indices = torch.tensor([0, 1], dtype=torch.int32, device=query.device)
    descriptor = RobotCollisionGeometry(
        torch.zeros(2, dtype=torch.int32, device=query.device), 1, masks
    )
    state = KinematicsState(robot_spheres=query, robot_collision_geometry=descriptor)
    buffer = CollisionBuffer.from_shape(query.shape, cuda_device_cfg)
    args = dict(
        state=state,
        collision_buffer=buffer,
        weight=torch.ones(1, device=query.device),
        activation_distance=torch.zeros(1, device=query.device),
        env_query_idx=indices,
        sum_distance=summed,
    )
    if swept:
        args["trajectory_dt"] = torch.full((2,), 0.1, device=query.device)
        run = scene.get_swept_sphere_distance
    else:
        run = scene.get_sphere_distance
    scene.data.set_obstacle_layers("target", 2, 0)
    scene.data.set_obstacle_layers("target", 4, 1)
    run(**args)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(**args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = run(**args)
    graph.replay()
    baseline = result.clone()
    pointers = [
        masks.data_ptr(),
        indices.data_ptr(),
        scene.data.cuboids.obstacle_layers.data_ptr(),
    ]
    masks.zero_()
    graph.replay()
    assert torch.count_nonzero(result) == 0
    masks.fill_(2**32 - 1)
    graph.replay()
    assert torch.all(result > 0)
    masks.copy_(torch.tensor([[2, 4], [4, 2]], dtype=torch.uint32, device=query.device))
    indices.copy_(torch.tensor([1, 0], dtype=torch.int32, device=query.device))
    graph.replay()
    torch.testing.assert_close(result, baseline)
    scene.data.set_obstacle_layers("target", 0, 1)
    graph.replay()
    assert torch.count_nonzero(result[0]) == 0
    torch.testing.assert_close(result[1], baseline[1])
    assert pointers == [
        masks.data_ptr(),
        indices.data_ptr(),
        scene.data.cuboids.obstacle_layers.data_ptr(),
    ]


@pytest.mark.parametrize("swept", [False, True])
def test_invalid_environment_is_infeasible(swept, cuda_device_cfg):
    """Bad GPU indices must not read out of bounds or return a collision-free result."""
    scene = make_scene("cuboid", cuda_device_cfg, num_envs=2)
    query = torch.tensor([0.25, 0.03, 0.01, 0.1], device=cuda_device_cfg.device)
    query = query.repeat(3, 3, 2, 1).requires_grad_()
    masks = torch.ones((2, 2), device=query.device, dtype=torch.uint32)
    indices = torch.tensor([-1, 2, 0], device=query.device, dtype=torch.int32)
    result = query_cost(scene, query, masks, swept, indices)
    assert torch.all(torch.isposinf(result[:2]))
    assert torch.all(torch.isfinite(result[2]))
    result.sum().backward()
    assert torch.count_nonzero(query.grad[:2]) == 0


@pytest.mark.parametrize("layers", [-1, 2**32, 1.5, True])
def test_invalid_layer_bits(layers, cuda_device_cfg):
    """Never silently truncate invalid layer values."""
    scene = make_scene("cuboid", cuda_device_cfg)
    with pytest.raises(ValueError, match="collision layers"):
        scene.data.set_obstacle_layers("target", layers)


@pytest.mark.parametrize(
    "shape,dtype", [((1, 3), torch.uint32), ((0, 2), torch.uint32), ((1, 2), torch.int32)]
)
def test_invalid_mask_layout(shape, dtype, cuda_device_cfg):
    """Reject incompatible host-visible layouts before launching a kernel."""
    scene = make_scene("cuboid", cuda_device_cfg, num_envs=2)
    query = torch.zeros((1, 1, 2, 4), device=cuda_device_cfg.device)
    masks = torch.ones(shape, dtype=dtype, device=query.device)
    with pytest.raises(ValueError, match="sphere_collision_mask"):
        query_cost(scene, query, masks)


@pytest.mark.parametrize("swept", [False, True])
def test_independent_environment_capacities(swept, cuda_device_cfg):
    """An index must fit both owners; extra unused scene rows do not resize model masks."""
    scene = make_scene("cuboid", cuda_device_cfg, num_envs=3)
    query = torch.tensor([0.25, 0.03, 0.01, 0.1], device=cuda_device_cfg.device)
    query = query.repeat(3, 3, 2, 1)
    masks = torch.ones((2, 2), device=query.device, dtype=torch.uint32)
    indices = torch.tensor([1, 0, 2], dtype=torch.int32, device=query.device)
    result = query_cost(scene, query, masks, swept, indices)
    assert torch.all(torch.isfinite(result[:2]))
    assert torch.all(torch.isposinf(result[2]))
    broadcast = query_cost(scene, query, masks[:1], swept, indices)
    assert torch.all(torch.isfinite(broadcast))


@pytest.mark.parametrize("swept", [False, True])
def test_two_fields_and_common_obstacle(swept, cuda_device_cfg):
    """Choose one of two overlapping ESDFs while keeping a shared analytic obstacle."""
    pose = [0, 0, 0, 1, 0, 0, 0]
    grids = [
        VoxelGrid(
            name=name,
            pose=pose,
            dims=[0.8, 0.8, 0.8],
            voxel_size=0.1,
            feature_dtype=torch.float16,
            feature_tensor=torch.full(
                (512, 1), value, dtype=torch.float16, device=cuda_device_cfg.device
            ),
        )
        for name, value in [("full", 0.05), ("without_target", 0.5)]
    ]
    table = Cuboid(name="table", pose=pose, dims=[0.4, 0.4, 0.4])
    scene = SceneCollision.from_config(
        SceneCollisionCfg(
            device_cfg=cuda_device_cfg, scene_model=SceneCfg(cuboid=[table], voxel=grids)
        )
    )
    scene.data.set_obstacle_layers("full", 2)
    scene.data.set_obstacle_layers("without_target", 4)
    query = torch.tensor([0.25, 0.03, 0.01, 0.1], device=cuda_device_cfg.device)
    query = query.repeat(1, 3, 2, 1)
    common = torch.ones((1, 2), dtype=torch.uint32, device=query.device)
    base = query_cost(scene, query, common, swept).clone()
    masks = torch.tensor([[3, 5]], dtype=torch.uint32, device=query.device)
    result = query_cost(scene, query, masks, swept)
    assert torch.all(base > 0)
    assert torch.all(result[:, :, 0] > base[:, :, 0])
    torch.testing.assert_close(result[:, :, 1], base[:, :, 1])
