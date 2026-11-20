# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare obstacle-frame sweep rejection with an unfiltered leaf-query reference."""

from typing import Any

import pytest
import torch
import warp as wp

from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.collision.wp_sweep_collision_kernel import (
    apply_speed_metric_to_contribution,
    compute_prepared_swept_collision_contribution,
)
from curobo._src.geom.data.data_obstacle_query import is_obs_enabled, load_obstacle_transform
from curobo._src.geom.types import Capsule, Cuboid, Cylinder, Mesh, SceneCfg, VoxelGrid
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.util.warp import init_warp


@wp.kernel(module="unique", enable_backward=False, module_options={"fast_math": True})
def _unfiltered_sweep(
    obs: Any,
    spheres: wp.array(dtype=wp.vec4),
    masks: wp.array(dtype=wp.uint32),
    layers: wp.array(dtype=wp.uint32),
    dt: wp.array(dtype=wp.float32),
    horizon: int,
    count: int,
    speed_metric: int,
    costs: wp.array(dtype=wp.float32),
    gradients: wp.array(dtype=wp.vec4),
) -> None:
    """Evaluate every enabled obstacle without either broadphase rejection."""
    index = wp.tid()
    state = index // count
    frame = state % horizon
    sphere_index = index % count
    sphere = spheres[index]
    current = wp.vec3(sphere[0], sphere[1], sphere[2])
    previous = current
    following = current
    if frame > 0:
        item = spheres[index - count]
        previous = wp.vec3(item[0], item[1], item[2])
    if frame < horizon - 1:
        item = spheres[index + count]
        following = wp.vec3(item[0], item[1], item[2])
    total = wp.vec4(0.0)
    if sphere[3] >= 0.0:
        for obstacle in range(obs.max_n):
            if is_obs_enabled(obs, 0, obstacle) and (masks[sphere_index] & layers[obstacle]) != 0:
                inv_t = load_obstacle_transform(obs, 0, obstacle)
                total += compute_prepared_swept_collision_contribution(
                    obs,
                    0,
                    obstacle,
                    inv_t,
                    wp.transform_point(inv_t, current),
                    wp.transform_point(inv_t, previous),
                    wp.transform_point(inv_t, following),
                    frame > 0,
                    frame < horizon - 1,
                    sphere[3] + 0.01,
                    0.01,
                )
        if speed_metric == 1 and total[0] > 0.0:
            total = apply_speed_metric_to_contribution(
                total, spheres, dt, index, frame, horizon, count
            )
    costs[index] = total[0]
    gradients[index] = wp.vec4(total[1], total[2], total[3], 0.0)


def _make_scene(kind: str, device: DeviceCfg) -> SceneCollision:
    """Create rotated, offset, or field-backed geometry with the same query interface."""
    pose = [0.3, -0.2, 0.1, 0.9238795, 0.0, 0.0, 0.3826834]
    if kind == "cuboid_team":
        # A 64-lane block shares 40 spheres. Nine nearby boxes trigger cooperative
        # evaluation, and the final nearby box is in a second obstacle chunk.
        boxes = [
            Cuboid(
                name="target" if i == 0 else f"box_{i}",
                pose=[pose[0] + (0.02 * i if i < 9 or i == 64 else 100.0), *pose[1:]],
                dims=[0.4, 0.8, 0.3],
            )
            for i in range(65)
        ]
        return SceneCollision.from_config(
            SceneCollisionCfg(device_cfg=device, scene_model=SceneCfg(cuboid=boxes))
        )
    if kind == "cuboid":
        obstacle = Cuboid(name="target", pose=pose, dims=[0.4, 0.8, 0.3])
    elif kind == "capsule":
        obstacle = Capsule(
            name="target", pose=pose, base=[0.3, -0.1, -0.4], tip=[0.7, 0.2, 0.3], radius=0.2
        )
    elif kind == "capsule_point":
        obstacle = Capsule(
            name="target", pose=pose, base=[0.3, -0.1, 0.2], tip=[0.3, -0.1, 0.2], radius=0.2
        )
    elif kind == "cylinder":
        obstacle = Cylinder(name="target", pose=pose, radius=0.2, height=0.8)
    elif kind.startswith("mesh"):
        offset = 1.5 if kind == "mesh_offset" else 0.0
        shape = Cuboid(
            name="local", pose=[offset, 0, 0, 1, 0, 0, 0], dims=[0.4, 0.8, 0.3]
        ).get_trimesh_mesh(transform_with_pose=True)
        obstacle = Mesh(
            name="target", pose=pose, vertices=shape.vertices.tolist(), faces=shape.faces.tolist()
        )
    else:
        obstacle = VoxelGrid(
            name="target",
            pose=pose,
            dims=[0.8] * 3,
            voxel_size=0.1,
            feature_tensor=torch.full((512, 1), 0.05, device=device.device, dtype=torch.float16),
            feature_dtype=torch.float16,
        )
    adapter = kind
    if kind.startswith("mesh"):
        adapter = "mesh"
    elif kind.startswith("voxel"):
        adapter = "voxel"
    elif kind == "capsule_point":
        adapter = "capsule"
    scene = SceneCollision.from_config(
        SceneCollisionCfg(device_cfg=device, scene_model=SceneCfg(**{adapter: [obstacle]}))
    )
    if kind == "mesh_unknown":
        scene.data.meshes.world_bounds[..., 4:7].fill_(torch.inf)
    if kind == "voxel_sentinel":
        scene.data.voxels.max_esdf_distance = 0.02
    return scene


@pytest.mark.parametrize(
    "kind",
    [
        "cuboid",
        "cuboid_team",
        "capsule",
        "capsule_point",
        "cylinder",
        "mesh",
        "mesh_offset",
        "mesh_unknown",
        "voxel",
        "voxel_sentinel",
    ],
)
@pytest.mark.parametrize("summed", [False, True])
@pytest.mark.parametrize("speed_metric", [False, True])
def test_local_sweep_bounds_preserve_leaf_queries(
    kind: str, summed: bool, speed_metric: bool, cuda_device_cfg: DeviceCfg
) -> None:
    """Preserve costs/gradients through graph replay, poses, masks, and strided sphere work."""
    init_warp(quiet=True, device_cfg=cuda_device_cfg)
    torch.manual_seed(73)
    scene = _make_scene(kind, cuda_device_cfg)
    num_spheres = 40 if kind == "cuboid_team" else 129
    query = torch.randn((2, 5, num_spheres, 4), device=cuda_device_cfg.device)
    query[..., :3].mul_(0.8)
    query[..., 3] = 0.12
    query[:, :, ::11, 3] = -1.0
    query.requires_grad_()
    mask = torch.ones((1, num_spheres), device=query.device, dtype=torch.uint32)
    mask[:, 1::7] = 0
    dt = torch.full((2,), 0.1, device=query.device)
    args = dict(
        query_spheres=query,
        collision_buffer=CollisionBuffer.from_shape(query.shape, cuda_device_cfg),
        weight=torch.ones(1, device=query.device),
        activation_distance=torch.full((1,), 0.01, device=query.device),
        trajectory_dt=dt,
        sphere_collision_mask=mask,
        sum_distance=summed,
        enable_speed_metric=speed_metric,
    )
    run = scene.get_swept_sphere_distance_raw
    run(**args)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(**args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = run(**args)
    reference_cost = torch.empty(query.shape[:-1], device=query.device)
    reference_gradient = torch.empty_like(query)
    for moved in (False, True):
        if moved:
            scene.update_obstacle_pose("target", Pose.from_list([0.1, 0.2, -0.1, 1, 0, 0, 0]))
        graph.replay()
        obs = scene.data.get_valid_data()[0]
        wp.launch(
            _unfiltered_sweep,
            dim=query.numel() // 4,
            inputs=[
                obs.to_warp(),
                wp.from_torch(query.detach().reshape(-1, 4), dtype=wp.vec4),
                wp.from_torch(mask.view(torch.int32).reshape(-1), dtype=wp.uint32),
                wp.from_torch(obs.obstacle_layers.view(torch.int32).reshape(-1), dtype=wp.uint32),
                wp.from_torch(dt),
                5,
                num_spheres,
                int(speed_metric),
                wp.from_torch(reference_cost.reshape(-1)),
                wp.from_torch(reference_gradient.reshape(-1, 4), dtype=wp.vec4),
            ],
        )
        wp.synchronize()
        expected = reference_cost.sum(-1, keepdim=True) if summed else reference_cost
        torch.testing.assert_close(result, expected, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(
            args["collision_buffer"].gradient, reference_gradient, atol=2e-5, rtol=2e-5
        )


def test_offset_mesh_collision_is_not_rejected(cuda_device_cfg: DeviceCfg) -> None:
    """Keep the off-center mesh collision that the cuboid-only prototype discarded."""
    init_warp(quiet=True, device_cfg=cuda_device_cfg)
    scene = _make_scene("mesh_offset", cuda_device_cfg)
    scene.update_obstacle_pose("target", Pose.from_list([0, 0, 0, 1, 0, 0, 0]))
    query = torch.tensor([1.75, 0.03, 0.01, 0.1], device=cuda_device_cfg.device)
    query = query.repeat(1, 3, 1, 1).requires_grad_()
    cost = scene.get_swept_sphere_distance_raw(
        query,
        CollisionBuffer.from_shape(query.shape, cuda_device_cfg),
        torch.ones(1, device=query.device),
        torch.zeros(1, device=query.device),
        torch.full((1,), 0.1, device=query.device),
    )
    torch.testing.assert_close(cost, torch.full_like(cost, 0.05), atol=1e-6, rtol=1e-6)
    cost.sum().backward()
    assert torch.isfinite(query.grad).all()
    assert torch.all(query.grad[..., 0].abs() > 0.9)


@pytest.mark.parametrize("single_voxel_axis", [False, True])
def test_voxel_sampling_halo_is_not_rejected(
    single_voxel_axis: bool, cuda_device_cfg: DeviceCfg
) -> None:
    """Preserve partial interpolation and nearest sampling beyond the nominal grid box."""
    init_warp(quiet=True, device_cfg=cuda_device_cfg)
    dims = [0.1 if single_voxel_axis else 0.8, 0.8, 0.8]
    count = 64 if single_voxel_axis else 512
    obstacle = VoxelGrid(
        name="field",
        pose=[0, 0, 0, 1, 0, 0, 0],
        dims=dims,
        voxel_size=0.1,
        feature_tensor=torch.full(
            (count, 1), -0.05, dtype=torch.float16, device=cuda_device_cfg.device
        ),
        feature_dtype=torch.float16,
    )
    scene = SceneCollision.from_config(
        SceneCollisionCfg(device_cfg=cuda_device_cfg, scene_model=SceneCfg(voxel=[obstacle]))
    )
    x = -0.14 if single_voxel_axis else -0.44
    query = torch.tensor([x, 0.0, 0.0, 0.01], device=cuda_device_cfg.device)
    query = query.repeat(1, 3, 1, 1)
    cost = scene.get_swept_sphere_distance_raw(
        query,
        CollisionBuffer.from_shape(query.shape, cuda_device_cfg),
        torch.ones(1, device=query.device),
        torch.zeros(1, device=query.device),
        torch.full((1,), 0.1, device=query.device),
    )
    assert torch.all(cost > 0.05)
