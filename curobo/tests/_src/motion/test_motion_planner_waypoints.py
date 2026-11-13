# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline Cartesian waypoint planning, dense validation, and graph reuse."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.motion.motion_planner_batch import BatchMotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util.cuda_graph_util import GraphExecutor


@pytest.mark.parametrize("planner_type", [MotionPlanner, BatchMotionPlanner])
def test_grasp_requires_trajopt_before_changing_collision_state(
    planner_type: type[MotionPlanner] | type[BatchMotionPlanner],
) -> None:
    """Reject unsupported graph-only grasp phases before accessing any mutable state."""
    planner = object.__new__(planner_type)
    planner._destroyed = True
    planner.config = SimpleNamespace(enable_trajopt=False)
    with pytest.raises(ValueError, match="grasp insertion and lift require enabled TrajOpt"):
        planner.plan_grasp(None, None)


@pytest.mark.parametrize("mode", ["single", "batch", "multi_env"])
def test_grasp_phases_follow_ordered_references(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Insertion/lift track ordered references with the existing phase budget and masks."""
    batch = 1 if mode == "single" else 2
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        num_knots=16,
        max_batch_size=batch,
        multi_env=mode == "multi_env",
        collision_cache={"cuboid": 1},
        num_ik_seeds=16,
        num_trajopt_seeds=2,
        use_cuda_graph=True,
    )
    planner = MotionPlanner(config) if mode == "single" else BatchMotionPlanner(config)
    try:
        planner.warmup(enable_graph=False, num_warmup_iterations=1)
        start = JointState.from_position(
            planner.default_joint_state.position.view(1, -1).repeat(batch, 1),
            joint_names=planner.joint_names,
        )
        if batch > 1:
            start.position[1, 0] += 0.1
        goals = planner.compute_kinematics(start).tool_poses.as_goal().clone()
        original = planner._grasp_phase_planner._plan_pose_waypoints
        solve_ik = planner.ik_solver.solve_pose
        ik_calls = []
        phases = []
        expected_success = torch.ones(batch, device=start.position.device, dtype=torch.bool)
        spheres = planner.kinematics.config.kinematics_config.link_spheres
        original_spheres = spheres.clone()

        def record_ik(*args: object, **kwargs: object) -> object:
            ik_calls.append(True)
            return solve_ik(*args, **kwargs)

        def record_phase(goal: GoalToolPose, state: JointState, **kwargs: object) -> object:
            count = len(ik_calls)
            result = original(goal, state, **kwargs)
            assert result is not None and result.success[expected_success].all()
            assert result.trajectory_source == "waypoints"
            assert result.debug_info["initial_iterations"] == 50
            assert goal.horizon == 1
            measured = planner.compute_kinematics(
                planner.kinematics.get_active_js(result.js_solution.squeeze(1))
            ).tool_poses.position.clone()
            reference = planner.prepare_pose_waypoints(goal, state)
            error = torch.linalg.vector_norm(measured - reference.position.squeeze(3), dim=-1)
            assert float(error[expected_success].max()) <= kwargs["path_tolerance"][0] + 1e-4
            phases.append((len(ik_calls) - count, kwargs, result.js_solution.clone()))
            return result

        monkeypatch.setattr(planner.ik_solver, "solve_pose", record_ik)
        monkeypatch.setattr(planner._grasp_phase_planner, "_plan_pose_waypoints", record_phase)
        cases = [(False, False), (True, False)]
        if batch > 1:
            cases.append((False, True))
        for refinement, partial_failure in cases:
            if partial_failure:
                goals.position[1, ..., 0] += 10.0
                expected_success[1] = False
            phases.clear()
            result = planner.plan_grasp(
                goals,
                start,
                grasp_approach_offset=-0.05,
                grasp_lift_offset=-0.05,
                grasp_path_tolerance=(0.02, math.radians(10)),
                lift_path_tolerance=(0.02, math.radians(10)),
                use_double_pass_trajopt_for_constrained_phases=refinement,
            )
            torch.testing.assert_close(result.success.reshape(batch), expected_success)
            torch.testing.assert_close(spheres, original_spheres)
            assert len(phases) == 2
            assert phases[0][0] == 1, "insertion must solve IK from the approach endpoint"
            assert phases[1][0] == 1, "lift must solve its new endpoint"
            assert all(phase[1]["finetune_attempts"] == int(refinement) for phase in phases)
    finally:
        planner.destroy()


@pytest.mark.parametrize("batched", [False, True])
def test_outer_iterations_bound_waypoints_and_restore_pose_defaults(batched: bool) -> None:
    """Per-call pass counts bound failed paths without changing later ordinary solves."""
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        num_ik_seeds=16,
        num_trajopt_seeds=2,
        use_cuda_graph=True,
    )
    planner = BatchMotionPlanner(config) if batched else MotionPlanner(config)
    try:
        assert planner.warmup(enable_graph=False, num_warmup_iterations=1)
        start = JointState.from_position(
            planner.default_joint_state.position.view(1, -1), joint_names=planner.joint_names
        )
        target = start.clone()
        target.position[:, 0] += 0.04
        terminal = planner.compute_kinematics(target).tool_poses.as_goal()
        waypoints = GoalToolPose(
            terminal.tool_frames,
            terminal.position.repeat(1, 3, 1, 1, 1),
            terminal.quaternion.repeat(1, 3, 1, 1, 1),
        )
        waypoints.position[:, 1, :, :, 2] += 3.0
        for passes in (None, 1, 2, 4):
            options = {} if passes is None else {"outer_iterations": passes}
            result = planner.plan_pose_waypoints(
                waypoints,
                start,
                path_tolerance=(0.01, 0.05),
                final_tolerance=(0.005, 0.05),
                max_attempts=1,
                finetune_attempts=0,
                **options,
            )
            assert result is not None and not result.success.any()
            expected_iterations = (4 if passes is None else passes) * 50
            assert result.debug_info["initial_iterations"] == expected_iterations
        ordinary = planner.plan_pose(terminal, start, max_attempts=1, finetune_attempts=0)
        assert ordinary is not None and ordinary.success.all()
        assert ordinary.debug_info["initial_iterations"] == 50
        assert planner.trajopt_solver.config.initial_max_iters == 50
    finally:
        planner.destroy()


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("num_knots,count", [(16, 32), (64, 256)])
def test_normal_warmup_initializes_waypoint_graph_dependencies(
    batched: bool,
    num_knots: int,
    count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordinary warmup prepares the shared graphs for both waypoint orientation modes."""
    batch = 2 if batched else 1
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        max_batch_size=batch,
        num_knots=num_knots,
        num_ik_seeds=16,
        num_trajopt_seeds=2,
        use_cuda_graph=True,
    )
    planner = BatchMotionPlanner(config) if batched else MotionPlanner(config)
    try:
        assert planner.warmup(enable_graph=False, num_warmup_iterations=1)
        start = JointState.from_position(
            planner.default_joint_state.position.view(1, -1).repeat(batch, 1),
            joint_names=planner.joint_names,
        )
        targets = start.position[:, None].repeat(1, count, 1)
        targets[..., 0] += torch.linspace(0.001, 0.04, count, device=targets.device)
        poses = planner.compute_kinematics(
            JointState.from_position(
                targets.reshape(-1, targets.shape[-1]),
                joint_names=planner.joint_names,
            )
        ).tool_poses
        waypoints = GoalToolPose(
            poses.tool_frames,
            poses.position.reshape(batch, count, poses.num_links, 1, 3),
            poses.quaternion.reshape(batch, count, poses.num_links, 1, 4),
        )

        def unexpected_capture(executor: GraphExecutor, inputs: tuple) -> None:
            pytest.fail("Waypoint planning captured a new graph after normal warmup")

        monkeypatch.setattr(GraphExecutor, "_initialize_cuda_graph", unexpected_capture)
        for track_orientation in (True, False):
            result = planner.plan_pose_waypoints(
                waypoints,
                start,
                path_tolerance=(0.05, 0.26),
                final_tolerance=(0.005, 0.05),
                track_orientation=track_orientation,
                max_attempts=1,
            )
            assert result is not None and result.success.all()
    finally:
        planner.destroy()


def test_batched_waypoints_retain_solved_rows_and_slice_padded_results() -> None:
    """One impossible row cannot invalidate a good row or corrupt its saved diagnostics."""
    planner = BatchMotionPlanner(
        MotionPlannerCfg.create(
            robot="franka.yml",
            max_batch_size=2,
            num_ik_seeds=16,
            num_trajopt_seeds=2,
            use_cuda_graph=True,
        )
    )
    try:
        state = JointState.from_position(
            planner.default_joint_state.position.view(1, -1).repeat(2, 1),
            joint_names=planner.joint_names,
        )
        targets = state.position[:, None].repeat(1, 2, 1)
        targets[0, :, 0] += targets.new_tensor([0.025, 0.05])
        targets[1, :, 0] -= targets.new_tensor([0.025, 0.05])
        poses = planner.compute_kinematics(
            JointState.from_position(
                targets.reshape(4, -1),
                joint_names=planner.joint_names,
            )
        ).tool_poses
        waypoints = GoalToolPose(
            poses.tool_frames,
            poses.position.reshape(2, 2, poses.num_links, 1, 3),
            poses.quaternion.reshape(2, 2, poses.num_links, 1, 4),
        )
        good_reference = planner.prepare_pose_waypoints(waypoints, state)
        waypoints.position[1, 0, :, :, 0] += 10.0
        result = planner.plan_pose_waypoints(
            waypoints,
            state,
            path_tolerance=(0.01, 0.05),
            final_tolerance=(0.005, 0.05),
            max_attempts=2,
            finetune_attempts=0,
        )
        assert result is not None
        assert result.success.tolist() == [[True], [False]]
        assert result.path_position_error.shape == (2, 1, 81, poses.num_links)
        assert result.path_position_error[0].max() <= 0.01
        assert result.debug_info["validation"]["path_valid"].tolist() == [[True], [False]]
        saved = result.clone()
        result.debug_info["validation"]["path_valid"].zero_()
        assert saved.debug_info["validation"]["path_valid"][0].all()

        start = state[0:1]
        reference = GoalToolPose(
            good_reference.tool_frames,
            good_reference.position[:1].contiguous(),
            good_reference.quaternion[:1].contiguous(),
        )
        seed_config = targets[:1, -1:].repeat(1, 2, 1)
        fraction = torch.linspace(0, 1, 16, device=targets.device).view(1, 1, 16, 1)
        seed = start.position[:, None, None] + fraction * (
            seed_config[:, :, None] - start.position[:, None, None]
        )
        single = planner.trajopt_solver.solve_pose(
            reference,
            start,
            seed_config=seed_config,
            seed_traj=seed,
            use_implicit_goal=True,
            track_intermediate=True,
            path_tolerance=(0.01, 0.05),
            final_tolerance=(0.005, 0.05),
            finetune_attempts=0,
        )
        assert single.success.shape == (1, 1)
        assert single.path_position_error.shape == (1, 1, 81, poses.num_links)
        assert single.interpolated_trajectory.position.shape[0] == 1
        assert single.timeopt_duration.shape[0] == 1
        assert single.debug_info["validation"]["path_valid"].shape == (1, 1)
    finally:
        planner.destroy()


@pytest.mark.parametrize(
    "timeopt,count,path_first,bend",
    [
        (False, 2, False, 0.02),
        (True, 2, False, 0.02),
        (True, 32, False, 0.02),
        (True, 2, True, 0.02),
        (True, 32, False, 0.03),
    ],
)
def test_goal_waypoints_goal_reuses_graphs_and_fixed_controls(
    timeopt: bool,
    count: int,
    path_first: bool,
    bend: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Switch both directions without changing shape, endpoint semantics, or captures."""
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        num_ik_seeds=16,
        num_trajopt_seeds=2,
        use_cuda_graph=True,
        enable_timeopt=timeopt,
    )
    planner = MotionPlanner(config)
    try:
        state = JointState.from_position(
            planner.default_joint_state.position.view(1, -1).clone(),
            joint_names=planner.joint_names,
        )
        targets = state.position.repeat(count, 1)
        targets[:, 0] += torch.linspace(0.08 / count, 0.08, count, device=targets.device)
        # Include a harder 32-target case that may not converge within 50 iterations.
        targets[:, 1] += bend * torch.sin(
            torch.linspace(torch.pi / count, torch.pi, count, device=targets.device)
        )
        poses = planner.compute_kinematics(
            JointState.from_position(
                targets,
                joint_names=planner.joint_names,
            )
        ).tool_poses
        waypoints = GoalToolPose(
            poses.tool_frames,
            poses.position.view(1, count, poses.num_links, 1, 3),
            poses.quaternion.view(1, count, poses.num_links, 1, 4),
        )
        terminal = GoalToolPose(
            poses.tool_frames,
            waypoints.position[:, -1:].contiguous(),
            waypoints.quaternion[:, -1:].contiguous(),
        )
        captures = []
        iteration_budgets = []
        solver = planner.trajopt_solver
        optimize = solver.optimizer.optimize

        def record_iterations(seed: torch.Tensor) -> torch.Tensor:
            iteration_budgets.append(solver.optimizer.optimizers[-1].config.num_iters)
            return optimize(seed)

        monkeypatch.setattr(solver.optimizer, "optimize", record_iterations)
        initialize = GraphExecutor._initialize_cuda_graph

        def record_capture(executor: GraphExecutor, inputs: tuple) -> None:
            captures.append(executor)
            initialize(executor, inputs)

        monkeypatch.setattr(GraphExecutor, "_initialize_cuda_graph", record_capture)
        if path_first:
            first = planner.plan_pose_waypoints(
                waypoints,
                state,
                path_tolerance=(0.01, 0.05),
                final_tolerance=(0.005, 0.05),
                max_attempts=1,
                finetune_attempts=0,
            )
        else:
            first = planner.plan_pose(terminal, state, max_attempts=1, finetune_attempts=0)
        assert first is not None and first.success.all()
        capture_count = len(captures)
        solver = planner.trajopt_solver
        reference_ptr = solver.core._goal_buffer.link_goal_poses.position.data_ptr()
        metrics_graph = solver.metrics_rollout._compute_metrics_from_action_executor._graph
        dense_graph = solver.additional_metrics_rollouts["interpolated_rollout"]
        dense_graph = dense_graph._compute_metrics_from_state_executor._graph

        result = planner.plan_pose_waypoints(
            waypoints,
            state,
            path_tolerance=(0.01, 0.05),
            final_tolerance=(0.005, 0.05),
            max_attempts=1,
            finetune_attempts=0,
        )
        assert result is not None
        if bend <= 0.02:
            assert result.success.all()
        within_path_bounds = bool(
            (result.path_position_error <= 0.01).all() & (result.path_rotation_error <= 0.05).all()
        )
        dense_valid = all(value.all() for value in result.debug_info["validation"].values())
        if not within_path_bounds or not dense_valid:
            assert not result.success.any()
        assert solver.action_horizon == 16 and solver.horizon == 81
        assert solver.core._goal_buffer.link_goal_poses.position.data_ptr() == reference_ptr
        assert solver.metrics_rollout._compute_metrics_from_action_executor._graph is metrics_graph
        dense_rollout = solver.additional_metrics_rollouts["interpolated_rollout"]
        assert dense_rollout._compute_metrics_from_state_executor._graph is dense_graph
        if result.success.all():
            assert dense_valid and within_path_bounds
        assert set(iteration_budgets) == {50}
        assert len(captures) == capture_count
        assert result.waypoint_indices.tolist() == [
            4 + round(72 * i / count) for i in range(1, count + 1)
        ]

        ordinary = planner.plan_pose(terminal, state, max_attempts=1, finetune_attempts=0)
        assert ordinary is not None and ordinary.success.all()
        assert ordinary.path_position_error is None
        assert solver.metrics_rollout._compute_metrics_from_action_executor._graph is metrics_graph
        assert len(captures) == capture_count

        moving = state.clone()
        moving.velocity = torch.ones_like(moving.position)
        with pytest.raises(ValueError, match="rest-to-rest"):
            planner.prepare_pose_waypoints(waypoints, moving)
        impossible = waypoints.clone()
        impossible.position[:, 0, :, :, 0] += 10.0
        failed = planner.plan_pose_waypoints(
            impossible,
            state,
            path_tolerance=(0.01, 0.05),
            final_tolerance=(0.005, 0.05),
            max_attempts=1,
            finetune_attempts=0,
        )
        assert failed is not None and not failed.success.any()
        assert not failed.debug_info["validation"]["path_valid"].any()
    finally:
        planner.destroy()


def test_position_only_waypoints_ignore_rotation_and_restore_pose_ik(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Switch between position-only and full poses without recapture or leaked IK criteria."""
    planner = MotionPlanner(
        MotionPlannerCfg.create(
            robot="franka.yml",
            num_ik_seeds=16,
            num_trajopt_seeds=2,
            use_cuda_graph=True,
        )
    )
    try:
        start = JointState.from_position(
            planner.default_joint_state.position.view(1, -1).clone(),
            joint_names=planner.joint_names,
        )
        target = start.position.repeat(2, 1)
        target[:, 0] += target.new_tensor([0.02, 0.04])
        poses = planner.compute_kinematics(
            JointState.from_position(
                target,
                joint_names=planner.joint_names,
            )
        ).tool_poses
        reference = GoalToolPose(
            poses.tool_frames,
            poses.position.reshape(1, 2, poses.num_links, 1, 3),
            poses.quaternion.reshape(1, 2, poses.num_links, 1, 4),
        )
        position_only = reference.clone()
        position_only.quaternion.zero_()
        position_only.quaternion[..., 0] = 1
        result = planner.plan_pose_waypoints(
            position_only,
            start,
            track_orientation=False,
            path_tolerance=(0.01, 0.001),
            final_tolerance=(0.005, 0.001),
            max_attempts=1,
            finetune_attempts=0,
        )
        assert result is not None and result.success.all()
        assert result.path_rotation_error.max() > 0.05
        assert result.path_position_error.max() <= 0.01

        def unexpected_capture(executor: GraphExecutor, inputs: tuple) -> None:
            pytest.fail("Orientation mode changed the captured graph")

        monkeypatch.setattr(GraphExecutor, "_initialize_cuda_graph", unexpected_capture)
        result = planner.plan_pose_waypoints(
            reference,
            start,
            track_orientation=True,
            path_tolerance=(0.01, 0.05),
            final_tolerance=(0.005, 0.05),
            max_attempts=1,
            finetune_attempts=0,
        )
        assert result is not None and result.success.all()
        assert result.path_rotation_error.max() <= 0.05
    finally:
        planner.destroy()


@pytest.mark.parametrize(
    "num_knots,count",
    [
        (12, 32),
        (13, 32),
        (16, 72),
        (17, 32),
        (32, 32),
        (62, 256),
        (63, 256),
        (64, 256),
        (128, 256),
        (256, 256),
        (512, 256),
        (513, 256),
        (1024, 256),
    ],
)
def test_configured_spline_waypoints_and_dense_validation(num_knots: int, count: int) -> None:
    """Retain up to 256 targets and reuse the configured graphs for normal pose planning."""
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        num_knots=num_knots,
        num_ik_seeds=16,
        num_trajopt_seeds=2,
        use_cuda_graph=True,
    )
    planner = MotionPlanner(config)
    try:
        state = planner.default_joint_state.clone()
        if state.position.ndim == 1:
            state = JointState.from_position(state.position[None], joint_names=planner.joint_names)
        joints = state.position[:, None].repeat(1, count, 1)
        joints[..., 0] += torch.linspace(0.001, 0.08, count, device=joints.device)
        poses = planner.compute_kinematics(
            JointState.from_position(joints.reshape(count, -1), joint_names=planner.joint_names)
        ).tool_poses
        targets = GoalToolPose(
            poses.tool_frames,
            poses.position.reshape(1, count, poses.num_links, 1, 3),
            poses.quaternion.reshape(1, count, poses.num_links, 1, 4),
        )
        assert planner.native_horizon == config.native_horizon
        assert planner.trajopt_solver.native_horizon == planner.native_horizon
        for orientation in (True, False):
            result = planner.plan_pose_waypoints(
                targets,
                state,
                track_orientation=orientation,
                max_attempts=1,
                finetune_attempts=0,
                path_tolerance=(0.05, 0.26),
                final_tolerance=(0.005, 0.05),
            )
            assert result is not None and result.success.all()
            assert result.path_position_error.shape[-2] == planner.native_horizon
            assert all(value.all() for value in result.debug_info["validation"].values())
            assert result.waypoint_indices.tolist() == config.spline_layout.waypoint_indices(count)
        goal = GoalToolPose(
            targets.tool_frames, targets.position[:, -1:], targets.quaternion[:, -1:]
        )
        result = planner.plan_pose(goal, state, max_attempts=1, finetune_attempts=0)
        assert result is not None and result.success.all()
        result = planner.plan_pose(
            goal, state, max_attempts=1, enable_graph_attempt=0, finetune_attempts=0
        )
        assert result is not None and result.success.all()
        assert planner.last_graph_calls > 0
    finally:
        planner.destroy()
