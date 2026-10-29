# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Integration of all-seed timeopt with TrajOpt and motion planning."""

from __future__ import annotations

import pytest
import torch

from curobo._src.motion.motion_planner import MotionPlanner, _fit_linear_seed_trajectories
from curobo._src.motion.motion_planner_batch import BatchMotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.solver.solver_trajopt import TrajOptSolver
from curobo._src.solver.solver_trajopt_cfg import TrajOptSolverCfg
from curobo._src.state.state_joint import JointState
from curobo._src.util.config_io import join_path, load_yaml
from curobo.content import get_robot_configs_path


@pytest.mark.parametrize("use_cuda_graph", [False, True])
@pytest.mark.parametrize("capacity", [1000, 2200])
def test_timeopt_solver_cspace_replay(use_cuda_graph: bool, capacity: int) -> None:
    """Every seed is timeopt, selected output is executable, and replay refreshes it."""
    config = TrajOptSolverCfg.create(
        "franka.yml", num_seeds=2, enable_timeopt=True, use_cuda_graph=use_cuda_graph,
        interpolation_buffer_size=capacity,
    )
    solver = TrajOptSolver(config)
    try:
        start = JointState.from_position(
            solver.default_joint_state.position.view(1, -1).clone(), joint_names=solver.joint_names
        )
        saved = None
        for delta in (0.0, 0.15, 0.3):
            goal = start.clone()
            goal.position[:, 0] += delta
            result = solver.solve_cspace(goal, start, return_seeds=2, finetune_attempts=2)
            assert result.success.any()
            assert result.timeopt_duration.shape == (1, 2)
            assert solver._timeopt.optimizer.batch_size == 2
            assert torch.equal(result.motion_time(), result.timeopt_duration)
            assert result.js_solution.knot is not None
            assert result.js_solution.knot_dt is not None
            for seed in range(2):
                if not result.success[0, seed]:
                    continue
                last = int(result.interpolated_last_tstep[0, seed])
                state = result.interpolated_trajectory
                torch.testing.assert_close(
                    state.position[0, seed, 0, :7], start.position[0], atol=2e-4, rtol=0
                )
                torch.testing.assert_close(
                    state.position[0, seed, last - 1, :7], goal.position[0], atol=2e-4, rtol=0
                )
                torch.testing.assert_close(
                    state.dt[0, seed] * (last - 1), result.motion_time()[0, seed]
                )
                assert state.dt[0, seed] <= config.interpolation_dt + 1e-6
                limits = solver.kinematics.config.kinematics_config.joint_limits
                for name in ("velocity", "acceleration", "jerk"):
                    values = getattr(state, name)[0, seed, :last, :7]
                    bound = getattr(limits, name)
                    assert torch.all(values <= bound[1] + 1e-3)
                    assert torch.all(values >= bound[0] - 1e-3)
            if saved is None:
                saved = result.clone()
                first_position = saved.interpolated_trajectory.position.clone()
        torch.testing.assert_close(saved.interpolated_trajectory.position, first_position)
    finally:
        solver.destroy()


def test_ik_rrt_timeopt_without_optimization(monkeypatch: pytest.MonkeyPatch) -> None:
    """Graph-only mode runs IK and RRT and timeopt fitted seeds without invoking TrajOpt."""
    config = MotionPlannerCfg.create(
        "franka.yml",
        num_ik_seeds=16,
        num_trajopt_seeds=2,
        graph_planner_algorithm="rrt",
        enable_trajopt=False,
        enable_timeopt=True,
        create_constrained_rrt=False,
        use_cuda_graph=True,
    )
    with MotionPlanner(config) as planner:
        start = JointState.from_position(
            planner.default_joint_state.position.view(1, -1).clone(),
            joint_names=planner.joint_names,
        )
        goal = start.clone()
        goal.position[:, 0] += 0.2
        goal_pose = planner.compute_kinematics(goal).tool_poses.as_goal()

        def fail_optimization(*args, **kwargs) -> None:
            pytest.fail("Graph-only planning invoked trajectory optimization")

        monkeypatch.setattr(planner.trajopt_solver, "solve_pose", fail_optimization)
        evaluate_seed = planner.trajopt_solver.evaluate_pose_seed
        evaluation_calls = 0

        def reject_first_fit(*args, **kwargs):
            nonlocal evaluation_calls
            evaluation_calls += 1
            fitted = evaluate_seed(*args, **kwargs)
            if evaluation_calls == 1:
                fitted.success.zero_()
            return fitted

        monkeypatch.setattr(planner.trajopt_solver, "evaluate_pose_seed", reject_first_fit)
        result = planner.plan_pose(goal_pose, start, max_attempts=2, enable_graph_attempt=0)
        assert planner.last_graph_calls == 2
        assert planner.last_planning_attempts == 2
        assert result is not None and result.success.item()
        assert result.trajectory_source == "graph_fit"
        assert result.timeopt_duration is not None
        assert result.success.shape == (1, 1)
        assert planner.trajopt_solver._timeopt.optimizer.batch_size == 2


def test_fitted_seed_masks_payload_and_moving_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """Preserve moving boundaries, reject masked seeds, and refresh payload dynamics."""
    robot = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
    robot["load_dynamics"] = True
    config = TrajOptSolverCfg.create(robot, num_seeds=2, enable_timeopt=True)
    solver = TrajOptSolver(config)
    try:
        start = JointState.from_position(
            solver.default_joint_state.position.view(1, -1).clone(), joint_names=solver.joint_names
        )
        start.velocity = torch.zeros_like(start.position)
        start.acceleration = torch.zeros_like(start.position)
        start.jerk = torch.zeros_like(start.position)
        start.velocity[:, 0] = 0.02
        start.acceleration[:, 0] = 0.01
        goal = JointState.from_position(start.position.clone(), joint_names=solver.joint_names)
        goal.position[:, 0] += 0.2
        seeds = _fit_linear_seed_trajectories(
            start, goal.position[:, None].repeat(1, 2, 1), solver.action_horizon, None
        )
        valid = torch.tensor([[False, True]], device=start.position.device)
        result = solver.evaluate_cspace_seed(goal, start, seeds, seed_valid_mask=valid)
        assert result.success.item()
        assert result.seed_rank.item() == 1
        state = result.interpolated_trajectory
        for name in ("position", "velocity", "acceleration"):
            torch.testing.assert_close(
                getattr(state, name)[0, 0, 0, :7], getattr(start, name)[0], atol=3e-4, rtol=0
            )
        valid.zero_()
        assert not solver.evaluate_cspace_seed(
            goal, start, seeds, seed_valid_mask=valid
        ).success.item()
        previous_timeopt = solver._timeopt
        reduced = previous_timeopt.optimizer._dynamics._reduced_model
        assert reduced.link_masses_com is (
            solver.transition_model.robot_dynamics._reduced_model.link_masses_com
        )

        def fail_duplicate_torque(*args, **kwargs) -> None:
            pytest.fail("Validation recomputed torque already available from the rollout")

        monkeypatch.setattr(previous_timeopt.optimizer, "_compute_torque", fail_duplicate_torque)
        solver.update_link_inertial("attached_object", mass=0.5)
        assert solver._timeopt is previous_timeopt
        valid[:] = True
        result = solver.evaluate_cspace_seed(goal, start, seeds, seed_valid_mask=valid)
        assert result.success.item()
        assert solver._timeopt is previous_timeopt
        # Overflow must reject the trajectory instead of accepting truncated output.
        solver.config.interpolation_dt = 1e-6
        result = solver.evaluate_cspace_seed(goal, start, seeds, seed_valid_mask=valid)
        assert not result.success.item()
    finally:
        solver.destroy()


def test_batched_timeopt_selects_per_problem() -> None:
    """Batched planning returns matching durations and endpoints for each problem."""
    config = MotionPlannerCfg.create(
        "franka.yml",
        num_ik_seeds=16,
        num_trajopt_seeds=2,
        max_batch_size=2,
        enable_timeopt=True,
        create_constrained_rrt=False,
    )
    planner = BatchMotionPlanner(config)
    try:
        start = JointState.from_position(
            planner.default_joint_state.position.view(1, -1).repeat(2, 1),
            joint_names=planner.joint_names,
        )
        goal = start.clone()
        goal.position[:, 0] += torch.tensor([0.15, 0.3], device=start.position.device)
        result = planner.plan_cspace(goal, start, enable_graph_attempt=1)
        assert result.success.all()
        assert result.timeopt_duration.shape == (2, 1)
        assert planner.trajopt_solver._timeopt.optimizer.batch_size == 4
        for batch in range(2):
            last = int(result.interpolated_last_tstep[batch, 0])
            torch.testing.assert_close(
                result.interpolated_trajectory.position[batch, 0, last - 1, :7],
                goal.position[batch],
                atol=2e-4,
                rtol=0,
            )
    finally:
        planner.destroy()
