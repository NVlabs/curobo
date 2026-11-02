# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""End-to-end tests for getting_started examples.

Each example supports a ``--test`` flag that runs with assertions.
This file invokes them via subprocess so each gets a clean process.
"""

import subprocess
import sys

import numpy as np
import pytest
import torch

from curobo.examples.getting_started.motion_planning import (
    _combine_grasp_phases,
    _prepare_interpolated_trajectory,
)
from curobo.types import JointState

EXAMPLES = [
    "curobo.examples.getting_started.forward_kinematics",
    "curobo.examples.getting_started.inverse_kinematics",
    "curobo.examples.getting_started.motion_planning",
    "curobo.examples.getting_started.cartesian_waypoints",
    "curobo.examples.getting_started.reactive_control",
    "curobo.examples.getting_started.build_robot_model",
]


@pytest.mark.parametrize(
    "pattern",
    [
        "Straight line",
        "Arc",
        "Circle",
        "Crane pick/place",
        "Front to back",
        "Zigzag",
    ],
)
@pytest.mark.parametrize("count", [1, 3, 4, 8, 9, 17, 32, 64, 256])
def test_waypoint_presets_have_valid_geometry(pattern: str, count: int) -> None:
    """Presets cover every supported count and retain their defining targets."""
    from curobo.examples.getting_started.cartesian_waypoints import (
        _PATTERN_MIN_COUNTS,
        _pattern_offsets,
    )

    offsets = _pattern_offsets(pattern, count)
    assert offsets.shape == (count, 3)
    assert offsets.dtype == np.float32 and np.isfinite(offsets).all()
    np.testing.assert_array_equal(offsets[0], np.zeros(3, dtype=np.float32))
    bounds = np.ptp(np.concatenate((np.zeros((1, 3), dtype=np.float32), offsets)), axis=0)
    assert bounds.max() <= 0.25 + 1e-7
    if count >= _PATTERN_MIN_COUNTS[pattern] and (pattern != "Circle" or (count - 1) % 4 == 0):
        np.testing.assert_allclose(bounds.max(), np.float32(0.25), atol=1e-7)
    if pattern == "Circle":
        np.testing.assert_array_equal(offsets[-1], np.zeros(3, dtype=np.float32))
        radius = np.linalg.norm(offsets[:, 1:] - np.array([0, 0.125], dtype=np.float32), axis=-1)
        np.testing.assert_allclose(radius, np.full(count, 0.125, dtype=np.float32), atol=1e-7)
    if pattern == "Straight line":
        np.testing.assert_array_equal(offsets[:, (0, 2)], np.zeros((count, 2), dtype=np.float32))
    if pattern == "Crane pick/place" and count >= 4:
        for anchor in (
            (0, 0, 0),
            (0, 0, 0.06),
            (0, 0.1, 0.06),
            (0, 0.1, 0),
        ):
            scaled_anchor = np.array(anchor, dtype=np.float32) * np.float32(0.25 / 0.1)
            distance = np.linalg.norm(offsets - scaled_anchor, axis=-1)
            assert distance.min() < 1e-7
        assert offsets[:, 2].min() >= 0
        moves = np.abs(np.diff(offsets, axis=0)) > 1e-7
        assert (moves.sum(axis=-1) == 1).all()
    if pattern == "Front to back" and count >= 5:
        np.testing.assert_allclose(offsets[:, 0].max(), np.float32(0.125), atol=1e-7)
        np.testing.assert_allclose(offsets[:, 0].min(), np.float32(-0.125), atol=1e-7)


def test_waypoint_group_transform_composes_positions_and_orientations() -> None:
    """The planner receives the same rigid transform as the displayed parent frame."""
    from curobo.examples.getting_started.cartesian_waypoints import _transform_waypoint_poses

    position = np.arange(12, dtype=np.float32).reshape(2, 2, 3) / 100
    original = position.copy()
    quaternion = np.zeros((2, 2, 4), dtype=np.float32)
    quaternion[..., 0] = 1
    half_root = np.float32(np.sqrt(0.5))
    group_rotation = np.array([half_root, 0, 0, half_root], dtype=np.float32)
    quaternion[-1, -1] = group_rotation
    translation = np.array([0.3, -0.2, 0.1], dtype=np.float32)
    world_position, world_quaternion = _transform_waypoint_poses(
        position,
        quaternion,
        translation,
        group_rotation,
    )
    expected = np.stack((-position[..., 1], position[..., 0], position[..., 2]), axis=-1)
    np.testing.assert_allclose(world_position, expected + translation, atol=1e-6)
    np.testing.assert_allclose(world_quaternion[0, 0], group_rotation, atol=1e-6)
    np.testing.assert_allclose(
        world_quaternion[-1, -1],
        np.array([0, 0, 0, 1], dtype=np.float32),
        atol=1e-6,
    )
    np.testing.assert_array_equal(position, original)
    assert world_position.dtype == world_quaternion.dtype == np.float32


def test_waypoint_example_warms_both_modes_before_first_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First GUI plans reuse startup captures and never expose a warmup trajectory."""
    from curobo._src.util.cuda_graph_util import GraphExecutor
    from curobo.examples.getting_started.cartesian_waypoints import WaypointDemo

    demo = WaypointDemo(port=0)
    try:
        assert demo.knots.value == "32"
        assert tuple(demo.knots.options) == ("16", "32", "64", "128", "256", "512", "1024")
        assert demo.planner.config.spline_layout.num_knots == 32
        assert demo.status.content.startswith("Ready")
        assert not demo.plan_button.disabled
        assert demo.trajectory is None and not demo.planned_lines
        assert not demo.play.value and demo.play.disabled
        assert demo.motion_time.value == demo.planning_time.value == "--"
        assert demo.path_scale.value == 1.0
        demo.path_scale.value = 0.3
        demo._synchronize()
        demo.waypoint_group.position = np.asarray(demo.waypoint_group.position) + (0.02, 0, 0)
        first_rotation = np.asarray(demo.handles[0][0].wxyz).copy()
        home_seed = demo.start.position.clone()

        def unexpected_capture(executor: GraphExecutor, inputs: tuple) -> None:
            pytest.fail("The first Plan call captured a graph after startup warmup")

        monkeypatch.setattr(GraphExecutor, "_initialize_cuda_graph", unexpected_capture)
        for mode in ("Position only", "Position + orientation"):
            demo.tracking_mode.value = mode
            demo.handles[0][0].wxyz = (
                np.array([1, 0, 0, 0], dtype=np.float32)
                if mode == "Position only"
                else first_rotation
            )
            demo.handles[-1][0].position += (0.0, 0.001, 0.0)
            assert demo.plan(), demo.status.content
            assert demo.status.content == "**Success**"
            assert float(demo.motion_time.value.removesuffix(" s")) >= 0
            assert float(demo.planning_time.value.removesuffix(" ms")) > 0
            assert demo.path_start is not None
            torch.testing.assert_close(demo.start.position, home_seed, atol=0, rtol=0)
            assert not torch.allclose(demo.path_start.position, demo.start.position, atol=1e-4)
            active = demo.planner.kinematics.get_active_js(demo.trajectory)
            torch.testing.assert_close(active.position[0], demo.path_start.position[0])
            first = demo._waypoints().position[0, 0, :, 0]
            achieved = demo.planner.compute_kinematics(demo.path_start).tool_poses.position[0, 0]
            assert (achieved - first).norm(dim=-1).max() <= demo.path_position.value / 100
            np.testing.assert_allclose(
                demo.reference_lines[0].points[0, 0],
                first[0].cpu(),
                atol=1e-6,
            )
        from curobo.examples.getting_started.cartesian_waypoints import _PATTERN_MIN_COUNTS

        initial_group = np.asarray(demo.waypoint_group.position).copy()
        before_scale = np.array(
            [[h.position for h in row] for row in demo.handles],
            dtype=np.float32,
        )
        before_rotation = np.array(
            [[h.wxyz for h in row] for row in demo.handles],
            dtype=np.float32,
        )
        demo.path_scale.value = 0.6
        demo.step()
        after_scale = np.array(
            [[h.position for h in row] for row in demo.handles],
            dtype=np.float32,
        )
        anchor = before_scale[0]
        np.testing.assert_allclose(after_scale, anchor + 2 * (before_scale - anchor), atol=1e-6)
        np.testing.assert_array_equal(
            np.array([[h.wxyz for h in row] for row in demo.handles], dtype=np.float32),
            before_rotation,
        )
        np.testing.assert_array_equal(demo.waypoint_group.position, initial_group)
        assert demo.trajectory is None and demo.play.disabled
        demo.waypoint_group.position += (0.01, 0.02, 0.0)
        demo.move_group.value = True
        demo._mark_edited(None)
        demo._last_preview = 0
        demo.step()
        assert demo.waypoint_group.visible
        assert all(not h.visible for row in demo.handles for h in row)
        assert demo.trajectory is None and not demo.planned_lines
        assert demo.motion_time.value == demo.planning_time.value == "--"
        mode = demo.tracking_mode.value
        for pattern, minimum in _PATTERN_MIN_COUNTS.items():
            demo._request("pattern:" + pattern)
            demo.step()
            assert demo._pattern == pattern and len(demo.handles) >= minimum
            assert demo.tracking_mode.value == mode
            assert demo.trajectory is None and demo.play.disabled
            np.testing.assert_allclose(
                demo.waypoint_group.position,
                initial_group + (0.01, 0.02, 0.0),
                atol=1e-6,
            )
            assert demo._waypoints().horizon == len(demo.handles)
        demo._request("reset")
        demo.step()
        np.testing.assert_allclose(
            demo.waypoint_group.position,
            demo.start_position.mean(axis=0),
        )
        demo.count.value = 1
        demo._synchronize()
        assert demo.plan(), demo.status.content
        assert demo.path_start is not None
        active = demo.planner.kinematics.get_active_js(demo.trajectory)
        torch.testing.assert_close(active.position[0], demo.path_start.position[0])

        def unexpected_path_plan(*args: object, **kwargs: object) -> None:
            pytest.fail("Path optimization ran after start IK failed")

        monkeypatch.setattr(demo.planner, "plan_pose_waypoints", unexpected_path_plan)
        demo.handles[0][0].position = np.asarray(demo.handles[0][0].position) + (10, 0, 0)
        assert not demo.plan()
        assert "start IK" in demo.status.content
        assert demo.path_start is None and demo.trajectory is None and demo.play.disabled
        assert demo.motion_time.value == "--"
    finally:
        demo.close()


@pytest.mark.parametrize("large_knots", [32, 64, 1024])
def test_waypoint_knot_selector_rebuilds_and_warms_preserving_edits(
    large_knots: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply knot changes on the main thread without losing poses or delaying capture."""
    from curobo._src.util.cuda_graph_util import GraphExecutor
    from curobo.examples.getting_started.cartesian_waypoints import WaypointDemo
    from curobo.motion_planner import MotionPlanner

    demo = WaypointDemo(port=0, num_knots=16)
    try:
        assert tuple(demo.knots.options) == ("16", "32", "64", "128", "256", "512", "1024")
        assert demo.knots.value == "16"
        demo.path_scale.value = 0.3
        demo._synchronize()
        demo.waypoint_group.position += (0.02, -0.01, 0.0)
        demo.handles[-1][0].position += (0.003, 0.0, 0.0)
        demo.handles[-1][0].wxyz = -np.asarray(demo.handles[-1][0].wxyz)
        demo.selected.value = "2"
        demo.tracking_mode.value = "Position only"
        before = demo._waypoints().clone()
        previous = demo.planner
        warmup = demo.warmup
        warmed = []

        def checked_warmup() -> bool:
            assert demo.plan_button.disabled and demo.knots.disabled
            warmed.append(demo.planner.config.spline_layout.num_knots)
            return warmup()

        monkeypatch.setattr(demo, "warmup", checked_warmup)
        demo.knots.value = str(large_knots)
        demo.step()
        assert demo.planner is not previous and previous._destroyed
        assert demo.planner.config.spline_layout.num_knots == large_knots
        max_count = 137 if large_knots == 32 else 256
        assert demo.count.max == max_count and demo.selected.value == "2"
        assert demo.tracking_mode.value == "Position only"
        assert not demo.plan_button.disabled and not demo.knots.disabled
        assert demo.trajectory is None and demo.play.disabled
        assert demo.motion_time.value == demo.planning_time.value == "--"
        after = demo._waypoints()
        torch.testing.assert_close(after.position, before.position, rtol=0, atol=0)
        torch.testing.assert_close(after.quaternion, before.quaternion, rtol=0, atol=0)

        def unexpected_capture(executor: GraphExecutor, inputs: tuple) -> None:
            pytest.fail("Changing knots deferred graph capture until Plan")

        with monkeypatch.context() as graph_guard:
            graph_guard.setattr(GraphExecutor, "_initialize_cuda_graph", unexpected_capture)
            for mode in ("Position only", "Position + orientation"):
                demo.tracking_mode.value = mode
                assert demo.plan(), demo.status.content
            demo.count.value = max_count
            demo._synchronize()
            previous = demo.planner
            demo.knots.value = "16"
            demo.step()
            assert demo.planner is previous and not previous._destroyed
            assert demo.knots.value == str(large_knots) and len(demo.handles) == max_count
            assert "at most 73" in demo.status.content

        demo.count.value = 3
        demo.knots.value = "16"
        demo.step()
        assert demo.planner.config.spline_layout.num_knots == 16 and previous._destroyed
        assert demo.count.max == 73 and len(demo.handles) == 3
        assert warmed == [large_knots, 16]
        assert not demo.plan_button.disabled and demo.status.content.startswith("Ready")
        with monkeypatch.context() as graph_guard:
            graph_guard.setattr(GraphExecutor, "_initialize_cuda_graph", unexpected_capture)
            for mode in ("Position only", "Position + orientation"):
                demo.tracking_mode.value = mode
                assert demo.plan(), demo.status.content

        def failed_create(num_knots: int) -> None:
            raise RuntimeError("test initialization failure")

        previous = demo.planner
        create_planner = demo._create_planner
        monkeypatch.setattr(demo, "_create_planner", failed_create)
        demo.knots.value = "32"
        demo.step()
        assert demo.planner is previous and not previous._destroyed
        assert demo.knots.value == "16" and not demo.plan_button.disabled
        assert "test initialization failure" in demo.status.content

        failed_replacements = []

        def create_failed_replacement(num_knots: int) -> MotionPlanner:
            replacement = create_planner(num_knots)
            failed_replacements.append(replacement)
            return replacement

        def failed_warmup() -> bool:
            return False

        monkeypatch.setattr(demo, "_create_planner", create_failed_replacement)
        monkeypatch.setattr(demo, "warmup", failed_warmup)
        demo.knots.value = "32"
        demo.step()
        assert failed_replacements[0]._destroyed
        assert demo.planner is previous and not previous._destroyed
        assert demo.knots.value == "16" and not demo.plan_button.disabled
        assert "Warmup did not succeed" in demo.status.content
    finally:
        demo.close()


def test_prepare_interpolated_trajectory_normalizes_and_trims() -> None:
    """Select one seed and trim the interpolation backing buffer."""
    position = torch.arange(10, dtype=torch.float32).reshape(1, 1, 5, 2)
    trajectory = JointState.from_position(position)
    trajectory.dt = torch.tensor([[0.1]], dtype=torch.float32)

    prepared = _prepare_interpolated_trajectory(
        trajectory,
        torch.tensor([[3]], dtype=torch.int32),
    )

    assert prepared.position.shape == (1, 3, 2)
    torch.testing.assert_close(prepared.position, position[:, 0, :3])
    assert prepared.dt is not None
    torch.testing.assert_close(prepared.dt, trajectory.dt)


def test_combine_grasp_phases_uses_each_dt() -> None:
    """Build continuous timestamps without repeating phase boundary states."""
    phases = []
    for label, horizon, dt in (
        ("Approach", 3, 0.1),
        ("Grasp", 4, 0.2),
        ("Lift", 2, 0.05),
    ):
        trajectory = JointState.from_position(torch.zeros(1, horizon, 2))
        trajectory.dt = torch.tensor([dt], dtype=torch.float32)
        phases.append((label, trajectory))

    positions, timestamps, boundaries, labels = _combine_grasp_phases(phases)

    assert positions.shape == (7, 2)
    torch.testing.assert_close(
        timestamps,
        torch.tensor([0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.85]),
    )
    assert boundaries == [0, 3, 6]
    assert labels == ["Approach", "Grasp", "Lift"]


@pytest.mark.parametrize("example", EXAMPLES, ids=[e.rsplit(".", 1)[-1] for e in EXAMPLES])
def test_example(example):
    result = subprocess.run(
        [sys.executable, "-m", example, "--test"],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        pytest.fail(
            f"{example} --test failed (exit {result.returncode})\n"
            f"--- stdout ---\n{result.stdout[-2000:]}\n"
            f"--- stderr ---\n{result.stderr[-2000:]}"
        )
