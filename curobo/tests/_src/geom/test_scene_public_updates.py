# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public scene updates preserve obstacle storage and captured collision queries."""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy

import pytest
import torch

from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.util.warp import init_warp
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import (
    Capsule,
    Cuboid,
    Cylinder,
    Mesh,
    Obstacle,
    Scene,
    SceneData,
    Sphere,
    VoxelGrid,
)
from curobo.types import DeviceCfg, Pose


def make_obstacle(kind: str, name: str) -> Obstacle:
    """Construct public scene objects without allocating GPU data."""
    common = dict(name=name, pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    if kind == "sphere":
        return Sphere(**common, radius=0.2)
    if kind == "cuboid":
        return Cuboid(**common, dims=[0.4, 0.4, 0.4])
    if kind == "capsule":
        return Capsule(**common, radius=0.2, base=[0.0, 0.0, -0.2], tip=[0.0, 0.0, 0.2])
    if kind == "cylinder":
        return Cylinder(**common, radius=0.2, height=0.4)
    if kind == "mesh":
        return Mesh(
            **common,
            vertices=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            faces=[[0, 1, 2]],
        )
    return VoxelGrid(**common, dims=[0.8, 0.8, 0.8], voxel_size=0.1)


@pytest.mark.parametrize("kind", ["sphere", "cuboid", "capsule", "cylinder", "mesh", "voxel"])
def test_public_scene_removal_updates_typed_list(kind: str) -> None:
    """Removing a public scene object must also remove it from reload inputs."""
    removed = make_obstacle(kind, "removed")
    kept = make_obstacle(kind, "kept")
    scene = Scene(**{kind: [removed, kept]})
    typed_list = getattr(scene, kind)
    scene.remove_obstacle("removed")
    assert scene.get_obstacle("removed") is None
    assert [item.name for item in typed_list] == ["kept"]
    assert [item.name for item in scene.objects] == ["kept"]
    scene.remove_obstacle("missing")
    assert scene.get_obstacle("kept") is kept
    scene.remove_obstacle("kept")
    assert not scene.objects
    assert not typed_list


def test_public_scene_removal_preserves_aliased_list() -> None:
    """An explicitly shared objects/type list must be updated exactly once."""
    obstacles = [make_obstacle("cuboid", "removed"), make_obstacle("cuboid", "kept")]
    scene = Scene(cuboid=obstacles, objects=obstacles)
    scene.remove_obstacle("removed")
    assert [item.name for item in scene.cuboid] == ["kept"]
    assert scene.objects is obstacles


def test_public_scene_data_reload_retains_multiple_voxel_grids(
    cuda_device_cfg: DeviceCfg,
) -> None:
    """Reloading a scene must retain every configured voxel layer."""
    data = SceneData.create_cache(
        num_envs=1,
        device_cfg=cuda_device_cfg,
        voxel_cache={"layers": 2, "dims": [0.8, 0.8, 0.8], "voxel_size": 0.1},
    )
    grids = [
        VoxelGrid(
            name=name,
            pose=[x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            dims=[0.8, 0.8, 0.8],
            voxel_size=0.1,
            feature_tensor=torch.full(
                (512, 1), 0.05, device=cuda_device_cfg.device, dtype=torch.float16
            ),
        )
        for name, x in (("first", 0.0), ("second", 1.0))
    ]
    data.load_from_scene_cfg(Scene(voxel=grids))
    assert data.get_obstacle_names() == ["first", "second"]
    assert data.voxels.get_active_count() == 2
    data.load_from_scene_cfg(Scene(voxel=grids[:1]))
    assert data.get_obstacle_names() == ["first"]
    assert data.voxels.get_active_count() == 1


@pytest.fixture(scope="module")
def public_planner(cuda_device_cfg: DeviceCfg) -> Iterator[MotionPlanner]:
    """Build the public planner with fixed-capacity world storage and CUDA graphs."""
    init_warp(quiet=True, device_cfg=cuda_device_cfg)
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        device_cfg=cuda_device_cfg,
        collision_cache={"cuboid": 41},
        num_ik_seeds=2,
        num_trajopt_seeds=2,
        max_goalset=1,
        use_cuda_graph=True,
    )
    with MotionPlanner(config) as planner:
        yield planner


@pytest.mark.parametrize("swept", [False, True])
def test_public_planner_world_updates_replay(
    public_planner: MotionPlanner,
    cuda_device_cfg: DeviceCfg,
    swept: bool,
) -> None:
    """Public world updates must remain visible to a previously captured query."""
    planner = public_planner
    planner.clear_scene_cache()
    collision = planner.scene_collision_checker
    data = collision.data
    assert isinstance(data, SceneData)
    query = torch.tensor(
        [
            [0.25, 0.03, 0.01, 0.1],
            [1.25, 0.03, 0.01, 0.1],
            [2.25, 0.03, 0.01, 0.1],
            [3.25, 0.03, 0.01, 0.1],
        ],
        **cuda_device_cfg.as_torch_dict(),
    ).repeat(1, 3, 1, 1)
    buffer = CollisionBuffer.from_shape(query.shape, cuda_device_cfg)
    arguments = dict(
        query_spheres=query,
        collision_buffer=buffer,
        weight=torch.ones(1, device=query.device),
        activation_distance=torch.zeros(1, device=query.device),
    )
    if swept:
        arguments["trajectory_dt"] = torch.full((1,), 0.1, device=query.device)
    query_fn = (
        collision.get_swept_sphere_distance_raw if swept else collision.get_sphere_distance_raw
    )
    for _ in range(2):
        query_fn(**arguments)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cost = query_fn(**arguments)

    def check(expected: list[bool], count: int) -> None:
        """Check actual collision output and loaded count after graph replay."""
        graph.replay()
        actual = cost > 0
        target = torch.tensor(expected, device=query.device).repeat(1, 3, 1)
        torch.testing.assert_close(actual, target)
        assert data.cuboids.get_active_count() == count

    def box(name: str, x: float) -> Cuboid:
        """Create separated obstacles addressed by stable public names."""
        return Cuboid(name=name, pose=[x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dims=[0.4, 0.4, 0.4])

    check([False, False, False, False], 0)
    planner.update_world(Scene(cuboid=[box("a", 0.0), box("b", 1.0), box("c", 2.0)]))
    check([True, True, True, False], 3)
    data.enable_obstacle("b", False)
    check([True, False, True, False], 3)
    data.add_obstacle(box("d", 3.0))
    check([True, False, True, True], 4)
    data.update_obstacle_pose(
        "c", Pose.from_list([8.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], cuda_device_cfg)
    )
    check([True, False, False, True], 4)
    data.enable_obstacle("b", True)
    check([True, True, False, True], 4)
    replacement = Scene(cuboid=[box("d", 3.0), box("c", 2.0), box("b", 1.0), box("a", 0.0)])
    replacement.remove_obstacle("b")
    planner.update_world(replacement)
    assert data.get_obstacle_names() == ["d", "c", "a"]
    check([True, False, True, True], 3)
    planner.update_world(Scene(cuboid=[box("b", 1.0)]))
    check([False, True, False, False], 1)
    planner.clear_scene_cache()
    check([False, False, False, False], 0)
    data.add_obstacle(box("new", 3.0))
    check([False, False, False, True], 1)


@pytest.mark.parametrize("swept", [False, True])
def test_public_scene_data_environment_reloads(
    cuda_device_cfg: DeviceCfg,
    swept: bool,
) -> None:
    """Per-environment replacement must update only its own captured prefix."""
    init_warp(quiet=True, device_cfg=cuda_device_cfg)
    boxes = [
        Cuboid(name=name, pose=[x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dims=[0.4, 0.4, 0.4])
        for name, x in (("a", 0.0), ("b", 1.0), ("c", 2.0))
    ]
    collision = SceneCollision.from_config(
        SceneCollisionCfg(
            device_cfg=cuda_device_cfg,
            scene_model=[Scene(cuboid=boxes[:1]), Scene(cuboid=boxes[1:])],
            cache={"cuboid": 41},
        )
    )
    data = collision.data
    query = torch.tensor(
        [[0.25, 0.03, 0.01, 0.1], [1.25, 0.03, 0.01, 0.1], [2.25, 0.03, 0.01, 0.1]],
        **cuda_device_cfg.as_torch_dict(),
    ).repeat(2, 3, 1, 1)
    arguments = dict(
        query_spheres=query,
        collision_buffer=CollisionBuffer.from_shape(query.shape, cuda_device_cfg),
        weight=torch.ones(1, device=query.device),
        activation_distance=torch.zeros(1, device=query.device),
        env_query_idx=torch.tensor([1, 0], device=query.device, dtype=torch.int32),
    )
    if swept:
        arguments["trajectory_dt"] = torch.full((2,), 0.1, device=query.device)
    query_fn = (
        collision.get_swept_sphere_distance_raw if swept else collision.get_sphere_distance_raw
    )
    for _ in range(2):
        query_fn(**arguments)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cost = query_fn(**arguments)

    def check(expected: list[list[bool]], counts: list[int]) -> None:
        """Compare per-query collisions and loaded counts after replay."""
        graph.replay()
        target = torch.tensor(expected, device=query.device).unsqueeze(1).repeat(1, 3, 1)
        torch.testing.assert_close(cost > 0, target)
        assert [len(data.get_obstacle_names(env_idx=i)) for i in range(2)] == counts

    check([[False, True, True], [True, False, False]], [1, 2])
    data.load_from_scene_cfg(Scene(cuboid=boxes[2:]), env_idx=1)
    check([[False, False, True], [True, False, False]], [1, 1])
    data.enable_obstacle("a", False, env_idx=0)
    check([[False, False, True], [False, False, False]], [1, 1])
    data.clear(env_idx=1)
    check([[False, False, False], [False, False, False]], [1, 0])
    data.add_obstacle(boxes[1], env_idx=1)
    check([[False, True, False], [False, False, False]], [1, 1])


def make_voxel(name: str, x: float, device: DeviceCfg) -> VoxelGrid:
    """Create a constant ESDF grid for append and graph-replay tests."""
    return VoxelGrid(
        name=name,
        pose=[x, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        dims=[0.8, 0.8, 0.8],
        voxel_size=0.1,
        feature_tensor=torch.full((512, 1), 0.05, device=device.device),
    )


@pytest.mark.parametrize("swept", [False, True])
def test_public_voxel_append_replays(cuda_device_cfg: DeviceCfg, swept: bool) -> None:
    """Appending preserves disabled grids, layers, storage, and captured queries."""
    init_warp(quiet=True, device_cfg=cuda_device_cfg)
    collision = SceneCollision.from_config(
        SceneCollisionCfg(
            device_cfg=cuda_device_cfg,
            cache={"voxel": {"layers": 3, "dims": [0.8] * 3, "voxel_size": 0.1}},
        )
    )
    data = collision.data
    voxels = data.voxels
    fields = ("params", "dims", "inv_pose", "features", "enable", "obstacle_layers", "count")
    pointers = [getattr(voxels, field).data_ptr() for field in fields]
    query = torch.tensor(
        [[0.05, 0.03, 0.01, 0.1], [1.05, 0.03, 0.01, 0.1], [2.05, 0.03, 0.01, 0.1]],
        **cuda_device_cfg.as_torch_dict(),
    ).repeat(1, 3, 1, 1)
    arguments = dict(
        query_spheres=query,
        collision_buffer=CollisionBuffer.from_shape(query.shape, cuda_device_cfg),
        weight=torch.ones(1, device=query.device),
        activation_distance=torch.zeros(1, device=query.device),
    )
    if swept:
        arguments["trajectory_dt"] = torch.full((1,), 0.1, device=query.device)
    query_fn = (
        collision.get_swept_sphere_distance_raw if swept else collision.get_sphere_distance_raw
    )
    for _ in range(2):
        query_fn(**arguments)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cost = query_fn(**arguments)

    def check(expected: list[bool]) -> None:
        """Verify actual collisions on the original captured storage."""
        graph.replay()
        target = torch.tensor(expected, device=query.device).repeat(1, 3, 1)
        torch.testing.assert_close(cost > 0, target)
        assert [getattr(voxels, field).data_ptr() for field in fields] == pointers

    check([False, False, False])
    assert data.add_obstacle(make_voxel("first", 0.0, cuda_device_cfg)) == 0
    check([True, False, False])
    data.set_obstacle_layers("first", 5)
    data.enable_obstacle("first", False)
    first = [getattr(voxels, field)[:, :1].clone() for field in fields[:-1]]
    assert data.add_obstacle(make_voxel("second", 1.0, cuda_device_cfg)) == 1
    for field, original in zip(fields[:-1], first):
        torch.testing.assert_close(getattr(voxels, field)[:, :1], original)
    check([False, True, False])
    data.enable_obstacle("first", True)
    check([True, True, False])
    smaller = make_voxel("third", 2.0, cuda_device_cfg)
    smaller.dims = [0.4] * 3
    smaller.feature_tensor = torch.full((64, 2), 0.05, device=query.device)[:, :1]
    assert data.add_obstacle(smaller) == 2
    assert voxels.get_names() == ["first", "second", "third"]
    assert voxels.get_active_count() == 3
    assert voxels.obstacle_layers[0].tolist() == [5, 1, 1]
    torch.testing.assert_close(
        voxels.features[0, 2, :64], smaller.feature_tensor.to(torch.float16)
    )
    assert torch.count_nonzero(voxels.features[0, 2, 64:]) == 0
    check([True, True, True])
    data.clear()
    assert data.add_obstacle(make_voxel("replacement", 2.0, cuda_device_cfg)) == 0
    check([False, False, True])


@pytest.mark.parametrize(
    "failure",
    [
        "duplicate",
        "full",
        "oversized",
        "shape",
        "missing",
        "negative_env",
        "env",
        "dims",
        "size",
        "empty_shape",
    ],
)
def test_public_voxel_append_failure_preserves_scene(
    cuda_device_cfg: DeviceCfg, failure: str
) -> None:
    """Rejected voxel additions leave all environments and metadata unchanged."""
    data = SceneData.create_cache(
        num_envs=2,
        device_cfg=cuda_device_cfg,
        voxel_cache={"layers": 2, "dims": [0.8] * 3, "voxel_size": 0.1},
    )
    assert data.add_obstacle(make_voxel("first", 0.0, cuda_device_cfg), env_idx=1) == 0
    data.enable_obstacle("first", False, env_idx=1)
    data.set_obstacle_layers("first", 5, env_idx=1)
    candidate = make_voxel("new", 1.0, cuda_device_cfg)
    env_idx = 1
    if failure == "duplicate":
        candidate.name = "first"
    elif failure == "full":
        data.add_obstacle(make_voxel("second", 2.0, cuda_device_cfg), env_idx=1)
    elif failure == "oversized":
        candidate.dims = [1.0] * 3
        candidate.feature_tensor = torch.zeros((1000, 1), device=cuda_device_cfg.device)
    elif failure == "shape":
        candidate.feature_tensor = candidate.feature_tensor[:32]
    elif failure == "missing":
        candidate.feature_tensor = None
    elif failure == "dims":
        candidate.dims = [0.0, 0.8, 0.8]
    elif failure == "size":
        candidate.voxel_size = 0.0
    elif failure == "empty_shape":
        candidate.dims = [0.001] * 3
    elif failure == "negative_env":
        env_idx = -1
    elif failure == "env":
        env_idx = 2
    voxels = data.voxels
    fields = ("params", "dims", "inv_pose", "features", "enable", "obstacle_layers", "count")
    original = [getattr(voxels, field).clone() for field in fields]
    names = deepcopy(voxels.names)
    with pytest.raises(ValueError):
        data.add_obstacle(candidate, env_idx=env_idx)
    assert voxels.names == names
    for field, expected in zip(fields, original):
        torch.testing.assert_close(getattr(voxels, field), expected)
