# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for analytic capsule and cylinder scene collision."""

# Standard Library
from typing import Union

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.collision_scene import (
    SceneCollision,
    SceneCollisionCfg,
    create_scene_collision,
)
from curobo._src.geom.types import Capsule, Cylinder, SceneCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.util.warp import init_warp

AnalyticObstacle = Union[Capsule, Cylinder]


@pytest.fixture(scope="module", autouse=True)
def setup_warp(cuda_device_cfg: DeviceCfg) -> None:
    """Initialize Warp for analytic primitive collision tests."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    init_warp(quiet=True, device_cfg=cuda_device_cfg)


def _scene_with_obstacle(obstacle: AnalyticObstacle) -> SceneCfg:
    """Create a scene containing one analytic obstacle."""
    if isinstance(obstacle, Capsule):
        return SceneCfg(capsule=[obstacle])
    return SceneCfg(cylinder=[obstacle])


def _create_checker(
    scene: Union[SceneCfg, list[SceneCfg]],
    device_cfg: DeviceCfg,
    cache: dict[str, int] | None = None,
) -> SceneCollision:
    """Create a scene checker for one or more environments."""
    return create_scene_collision(
        SceneCollisionCfg(
            device_cfg=device_cfg,
            scene_model=scene,
            cache=cache,
            max_distance=1.0,
        )
    )


def _query_distance(
    checker: SceneCollision,
    query_spheres: torch.Tensor,
    device_cfg: DeviceCfg,
    env_query_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Query unsmoothed collision penetration for raw spheres."""
    return checker.get_sphere_distance_raw(
        query_spheres,
        CollisionBuffer.from_shape(query_spheres.shape, device_cfg),
        weight=device_cfg.to_device([1.0]),
        activation_distance=device_cfg.to_device([0.0]),
        env_query_idx=env_query_idx,
    )


@pytest.mark.parametrize(
    ("obstacle", "query_center", "expected_gradient"),
    [
        (
            Capsule(
                name="capsule",
                pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                base=[-0.5, 0.0, 0.0],
                tip=[0.5, 0.0, 0.0],
                radius=0.2,
            ),
            [0.0, 0.25, 0.0],
            [0.0, -1.0, 0.0, 0.0],
        ),
        (
            Cylinder(
                name="cylinder",
                pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                radius=0.2,
                height=1.0,
            ),
            [0.25, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
        ),
    ],
    ids=["arbitrary-axis-capsule", "cylinder-side"],
)
def test_analytic_primitive_distance_and_gradient(
    obstacle: AnalyticObstacle,
    query_center: list[float],
    expected_gradient: list[float],
    cuda_device_cfg: DeviceCfg,
) -> None:
    """Verify exact side distance and autograd gradient for each primitive."""
    checker = _create_checker(_scene_with_obstacle(obstacle), cuda_device_cfg)
    query = torch.tensor(
        [[[[*query_center, 0.1]]]],
        device=cuda_device_cfg.device,
        dtype=torch.float32,
        requires_grad=True,
    )

    distance = _query_distance(checker, query, cuda_device_cfg)
    distance.sum().backward()

    torch.testing.assert_close(distance, torch.full_like(distance, 0.05), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        query.grad,
        torch.tensor(
            [[[[*expected_gradient]]]], device=cuda_device_cfg.device, dtype=torch.float32
        ),
        atol=1e-6,
        rtol=1e-6,
    )


def test_non_unit_diagonal_capsule_distance_and_gradient(
    cuda_device_cfg: DeviceCfg,
) -> None:
    """Verify projection onto a non-unit, non-axis-aligned capsule segment."""
    inverse_sqrt_two = 2.0**-0.5
    capsule = Capsule(
        name="diagonal_capsule",
        pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        base=[0.0, 0.0, 0.0],
        tip=[2.0, 2.0, 0.0],
        radius=0.2,
    )
    query = torch.tensor(
        [[[[1.0 + 0.25 * inverse_sqrt_two, 1.0 - 0.25 * inverse_sqrt_two, 0.0, 0.1]]]],
        device=cuda_device_cfg.device,
        dtype=torch.float32,
        requires_grad=True,
    )
    checker = _create_checker(SceneCfg(capsule=[capsule]), cuda_device_cfg)

    distance = _query_distance(checker, query, cuda_device_cfg)
    distance.sum().backward()

    torch.testing.assert_close(distance, torch.full_like(distance, 0.05), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        query.grad,
        torch.tensor(
            [[[[-inverse_sqrt_two, inverse_sqrt_two, 0.0, 0.0]]]],
            device=cuda_device_cfg.device,
            dtype=torch.float32,
        ),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    ("obstacle", "query_center", "expected_gradient"),
    [
        (
            Capsule(
                name="capsule",
                pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                base=[0.0, 0.0, -0.5],
                tip=[0.0, 0.0, 0.5],
                radius=0.2,
            ),
            [0.0, 0.0, 0.75],
            [0.0, 0.0, -1.0, 0.0],
        ),
        (
            Cylinder(
                name="cylinder",
                pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                radius=0.2,
                height=1.0,
            ),
            [0.0, 0.0, 0.55],
            [0.0, 0.0, -1.0, 0.0],
        ),
    ],
    ids=["capsule-end", "cylinder-cap"],
)
def test_analytic_primitive_end_distance_and_gradient(
    obstacle: AnalyticObstacle,
    query_center: list[float],
    expected_gradient: list[float],
    cuda_device_cfg: DeviceCfg,
) -> None:
    """Verify capsule-end and cylinder-cap distances and gradients."""
    checker = _create_checker(_scene_with_obstacle(obstacle), cuda_device_cfg)
    query = torch.tensor(
        [[[[*query_center, 0.1]]]],
        device=cuda_device_cfg.device,
        dtype=torch.float32,
        requires_grad=True,
    )

    distance = _query_distance(checker, query, cuda_device_cfg)
    distance.sum().backward()

    torch.testing.assert_close(distance, torch.full_like(distance, 0.05), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        query.grad,
        torch.tensor(
            [[[[*expected_gradient]]]], device=cuda_device_cfg.device, dtype=torch.float32
        ),
        atol=1e-6,
        rtol=1e-6,
    )


def test_zero_length_capsule_matches_sphere(cuda_device_cfg: DeviceCfg) -> None:
    """Treat coincident capsule endpoints as a sphere without numerical instability."""
    capsule = Capsule(
        name="sphere_capsule",
        pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        base=[0.0, 0.0, 0.0],
        tip=[0.0, 0.0, 0.0],
        radius=0.2,
    )
    checker = _create_checker(SceneCfg(capsule=[capsule]), cuda_device_cfg)
    query = torch.tensor(
        [[[[0.25, 0.0, 0.0, 0.1]]]],
        device=cuda_device_cfg.device,
        dtype=torch.float32,
        requires_grad=True,
    )

    distance = _query_distance(checker, query, cuda_device_cfg)
    distance.sum().backward()

    torch.testing.assert_close(distance, torch.full_like(distance, 0.05), atol=1e-6, rtol=1e-6)
    assert query.grad is not None
    assert torch.isfinite(query.grad).all()
    torch.testing.assert_close(
        query.grad,
        torch.tensor(
            [[[[-1.0, 0.0, 0.0, 0.0]]]],
            device=cuda_device_cfg.device,
            dtype=torch.float32,
        ),
    )


@pytest.mark.parametrize(
    ("obstacle", "expected_gradient"),
    [
        (
            Capsule(
                name="centered_capsule",
                pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                base=[0.0, 0.0, -0.5],
                tip=[0.0, 0.0, 0.5],
                radius=0.2,
            ),
            [0.0, -1.0, 0.0, 0.0],
        ),
        (
            Cylinder(
                name="centered_cylinder",
                pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                radius=0.2,
                height=1.0,
            ),
            [-1.0, 0.0, 0.0, 0.0],
        ),
    ],
    ids=["capsule-centerline", "cylinder-axis"],
)
def test_analytic_primitive_interior_singularity_is_stable(
    obstacle: AnalyticObstacle,
    expected_gradient: list[float],
    cuda_device_cfg: DeviceCfg,
) -> None:
    """Return a deterministic escape gradient at non-unique interior points."""
    checker = _create_checker(_scene_with_obstacle(obstacle), cuda_device_cfg)
    query = torch.tensor(
        [[[[0.0, 0.0, 0.0, 0.1]]]],
        device=cuda_device_cfg.device,
        dtype=torch.float32,
        requires_grad=True,
    )

    distance = _query_distance(checker, query, cuda_device_cfg)
    distance.sum().backward()

    torch.testing.assert_close(distance, torch.full_like(distance, 0.3))
    assert query.grad is not None
    assert torch.isfinite(query.grad).all()
    torch.testing.assert_close(
        query.grad,
        torch.tensor(
            [[[[*expected_gradient]]]], device=cuda_device_cfg.device, dtype=torch.float32
        ),
    )


def test_cylinder_rim_distance_and_gradient(cuda_device_cfg: DeviceCfg) -> None:
    """Verify the diagonal distance and gradient outside a cylinder rim."""
    cylinder = Cylinder(
        name="cylinder",
        pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        radius=0.2,
        height=1.0,
    )
    checker = _create_checker(SceneCfg(cylinder=[cylinder]), cuda_device_cfg)
    query = torch.tensor(
        [[[[0.23, 0.0, 0.53, 0.1]]]],
        device=cuda_device_cfg.device,
        dtype=torch.float32,
        requires_grad=True,
    )

    distance = _query_distance(checker, query, cuda_device_cfg)
    distance.sum().backward()

    expected_distance = 0.1 - (2.0 * 0.03**2) ** 0.5
    torch.testing.assert_close(
        distance, torch.full_like(distance, expected_distance), atol=1e-6, rtol=1e-6
    )
    assert query.grad is not None
    torch.testing.assert_close(
        query.grad,
        torch.tensor(
            [[[[-(2.0**-0.5), 0.0, -(2.0**-0.5), 0.0]]]],
            device=cuda_device_cfg.device,
            dtype=torch.float32,
        ),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    ("obstacle", "query_center"),
    [
        (
            Capsule(
                name="rotated_capsule",
                pose=[0.0, 0.0, 0.0, 2.0**-0.5, 0.0, 2.0**-0.5, 0.0],
                base=[0.0, 0.0, -0.5],
                tip=[0.0, 0.0, 0.5],
                radius=0.2,
            ),
            [0.75, 0.0, 0.0],
        ),
        (
            Cylinder(
                name="rotated_cylinder",
                pose=[0.0, 0.0, 0.0, 2.0**-0.5, 0.0, 2.0**-0.5, 0.0],
                radius=0.2,
                height=1.0,
            ),
            [0.55, 0.0, 0.0],
        ),
    ],
    ids=["capsule", "cylinder"],
)
def test_analytic_primitive_rotated_pose(
    obstacle: AnalyticObstacle,
    query_center: list[float],
    cuda_device_cfg: DeviceCfg,
) -> None:
    """Apply obstacle orientation to distances and world-frame gradients."""
    checker = _create_checker(_scene_with_obstacle(obstacle), cuda_device_cfg)
    query = torch.tensor(
        [[[[*query_center, 0.1]]]],
        device=cuda_device_cfg.device,
        dtype=torch.float32,
        requires_grad=True,
    )

    distance = _query_distance(checker, query, cuda_device_cfg)
    distance.sum().backward()

    torch.testing.assert_close(distance, torch.full_like(distance, 0.05), atol=1e-6, rtol=1e-6)
    assert query.grad is not None
    torch.testing.assert_close(
        query.grad,
        torch.tensor(
            [[[[-1.0, 0.0, 0.0, 0.0]]]],
            device=cuda_device_cfg.device,
            dtype=torch.float32,
        ),
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "obstacle",
    [
        Capsule(
            name="moving",
            pose=[2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            base=[0.0, 0.0, -0.5],
            tip=[0.0, 0.0, 0.5],
            radius=0.2,
        ),
        Cylinder(
            name="moving",
            pose=[2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            radius=0.2,
            height=1.0,
        ),
    ],
    ids=["capsule", "cylinder"],
)
def test_analytic_primitive_pose_and_enable_updates(
    obstacle: AnalyticObstacle, cuda_device_cfg: DeviceCfg
) -> None:
    """Apply pose and enable updates without rebuilding primitive storage."""
    checker = _create_checker(_scene_with_obstacle(obstacle), cuda_device_cfg)
    query = torch.tensor(
        [[[[0.25, 0.0, 0.0, 0.1]]]],
        device=cuda_device_cfg.device,
        dtype=torch.float32,
    )

    torch.testing.assert_close(
        _query_distance(checker, query, cuda_device_cfg),
        torch.zeros((1, 1, 1), device=cuda_device_cfg.device),
    )
    checker.update_obstacle_pose(
        "moving", Pose.from_list([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], cuda_device_cfg)
    )
    torch.testing.assert_close(
        _query_distance(checker, query, cuda_device_cfg),
        torch.full((1, 1, 1), 0.05, device=cuda_device_cfg.device),
        atol=1e-6,
        rtol=1e-6,
    )
    checker.enable_obstacle("moving", False)
    torch.testing.assert_close(
        _query_distance(checker, query, cuda_device_cfg),
        torch.zeros((1, 1, 1), device=cuda_device_cfg.device),
    )


def test_capsule_and_cylinder_cache_load(cuda_device_cfg: DeviceCfg) -> None:
    """Load both analytic types into preallocated empty caches."""
    checker = _create_checker(SceneCfg(), cuda_device_cfg, cache={"capsule": 1, "cylinder": 1})
    checker.load_collision_model(
        SceneCfg(
            capsule=[
                Capsule(
                    name="capsule",
                    pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    base=[0.0, 0.0, -0.5],
                    tip=[0.0, 0.0, 0.5],
                    radius=0.2,
                )
            ],
            cylinder=[
                Cylinder(
                    name="cylinder",
                    pose=[2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    radius=0.2,
                    height=1.0,
                )
            ],
        )
    )

    assert checker.collision_types["capsule"]
    assert checker.collision_types["cylinder"]
    assert checker.get_obstacle_names() == ["capsule", "cylinder"]
    assert checker.check_obstacle_exists("capsule")
    assert checker.check_obstacle_exists("cylinder")


def test_capsule_and_cylinder_multi_environment(cuda_device_cfg: DeviceCfg) -> None:
    """Select capsule and cylinder obstacles from different environments."""
    far_pose = [3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    origin_pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    scenes = [
        SceneCfg(
            capsule=[
                Capsule(
                    name="capsule_0",
                    pose=origin_pose,
                    base=[0.0, 0.0, -0.5],
                    tip=[0.0, 0.0, 0.5],
                    radius=0.2,
                )
            ],
            cylinder=[Cylinder(name="cylinder_0", pose=far_pose, radius=0.2, height=1.0)],
        ),
        SceneCfg(
            capsule=[
                Capsule(
                    name="capsule_1",
                    pose=far_pose,
                    base=[0.0, 0.0, -0.5],
                    tip=[0.0, 0.0, 0.5],
                    radius=0.2,
                )
            ],
            cylinder=[Cylinder(name="cylinder_1", pose=origin_pose, radius=0.2, height=1.0)],
        ),
    ]
    checker = _create_checker(scenes, cuda_device_cfg)
    query = torch.tensor(
        [
            [[[0.25, 0.0, 0.0, 0.1]]],
            [[[0.25, 0.0, 0.0, 0.1]]],
        ],
        device=cuda_device_cfg.device,
        dtype=torch.float32,
    )
    env_query_idx = torch.tensor([0, 1], device=cuda_device_cfg.device, dtype=torch.int32)

    distance = _query_distance(checker, query, cuda_device_cfg, env_query_idx)

    torch.testing.assert_close(distance, torch.full_like(distance, 0.05), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "obstacle",
    [
        Capsule(
            name="capsule",
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            base=[0.0, 0.0, -0.5],
            tip=[0.0, 0.0, 0.5],
            radius=0.1,
        ),
        Cylinder(
            name="cylinder",
            pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            radius=0.1,
            height=1.0,
        ),
    ],
    ids=["capsule", "cylinder"],
)
def test_swept_query_detects_analytic_primitive(
    obstacle: AnalyticObstacle, cuda_device_cfg: DeviceCfg
) -> None:
    """Detect a primitive crossed between two collision-free trajectory samples."""
    checker = _create_checker(_scene_with_obstacle(obstacle), cuda_device_cfg)
    query = torch.tensor(
        [[[[-0.5, 0.0, 0.0, 0.05]], [[0.5, 0.0, 0.0, 0.05]]]],
        device=cuda_device_cfg.device,
        dtype=torch.float32,
    )
    buffer = CollisionBuffer.from_shape(query.shape, cuda_device_cfg)
    weight = cuda_device_cfg.to_device([1.0])
    activation_distance = cuda_device_cfg.to_device([0.0])

    static_distance = checker.get_sphere_distance_raw(
        query, buffer, weight, activation_distance
    ).clone()
    swept_distance = checker.get_swept_sphere_distance_raw(
        query,
        buffer,
        weight,
        activation_distance,
        trajectory_dt=cuda_device_cfg.to_device([0.1]),
    )

    assert torch.count_nonzero(static_distance) == 0
    assert torch.any(swept_distance > 0.0)


def test_swept_query_detects_non_unit_diagonal_capsule(
    cuda_device_cfg: DeviceCfg,
) -> None:
    """Use reconstructed endpoint bounds for a non-unit diagonal capsule."""
    capsule = Capsule(
        name="diagonal_capsule",
        pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        base=[0.0, 0.0, 0.0],
        tip=[2.0, 2.0, 0.0],
        radius=0.1,
    )
    checker = _create_checker(SceneCfg(capsule=[capsule]), cuda_device_cfg)
    query = torch.tensor(
        [[[[1.0, 1.0, -0.5, 0.05]], [[1.0, 1.0, 0.5, 0.05]]]],
        device=cuda_device_cfg.device,
        dtype=torch.float32,
    )
    buffer = CollisionBuffer.from_shape(query.shape, cuda_device_cfg)
    weight = cuda_device_cfg.to_device([1.0])
    activation_distance = cuda_device_cfg.to_device([0.0])

    static_distance = checker.get_sphere_distance_raw(
        query, buffer, weight, activation_distance
    ).clone()
    swept_distance = checker.get_swept_sphere_distance_raw(
        query,
        buffer,
        weight,
        activation_distance,
        trajectory_dt=cuda_device_cfg.to_device([0.1]),
    )

    assert torch.count_nonzero(static_distance) == 0
    assert torch.any(swept_distance > 0.0)


def test_mixed_analytic_primitives_cuda_graph(cuda_device_cfg: DeviceCfg) -> None:
    """Replay mixed capsule and cylinder queries through a CUDA graph."""
    scene = SceneCfg(
        capsule=[
            Capsule(
                name="capsule",
                pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                base=[0.0, 0.0, -0.5],
                tip=[0.0, 0.0, 0.5],
                radius=0.2,
            )
        ],
        cylinder=[
            Cylinder(
                name="cylinder",
                pose=[2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                radius=0.2,
                height=1.0,
            )
        ],
    )
    checker = _create_checker(scene, cuda_device_cfg)
    query = torch.tensor(
        [[[[0.25, 0.0, 0.0, 0.1], [2.25, 0.0, 0.0, 0.1]]]],
        device=cuda_device_cfg.device,
        dtype=torch.float32,
    )
    buffer = CollisionBuffer.from_shape(query.shape, cuda_device_cfg)
    weight = cuda_device_cfg.to_device([1.0])
    activation_distance = cuda_device_cfg.to_device([0.0])
    env_query_idx = torch.zeros((1,), device=cuda_device_cfg.device, dtype=torch.int32)

    for _ in range(3):
        checker.get_sphere_distance_raw(
            query,
            buffer,
            weight,
            activation_distance,
            env_query_idx=env_query_idx,
        )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_distance = checker.get_sphere_distance_raw(
            query,
            buffer,
            weight,
            activation_distance,
            env_query_idx=env_query_idx,
        )
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        captured_distance,
        torch.full_like(captured_distance, 0.05),
        atol=1e-6,
        rtol=1e-6,
    )
