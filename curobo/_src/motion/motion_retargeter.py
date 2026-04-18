# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Motion retargeter: IK and MPC-based retargeting for humanoid robots.

This module provides :class:`MotionRetargeter`, a high-level API that
takes per-frame :class:`GoalToolPose` targets and produces joint trajectories
using either frame-by-frame warm-started IK or MPC.

Example:

.. code-block:: python

    from curobo.motion_retargeter import MotionRetargeter, MotionRetargeterCfg
    from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria

    cfg = MotionRetargeterCfg.create(
        robot="unitree_g1_29dof_retarget.yml",
        tool_pose_criteria={
            "pelvis": ToolPoseCriteria.track_position_and_orientation(
                xyz=[1.0, 1.0, 1.0], rpy=[0.5, 0.5, 0.5],
            ),
            "left_hand": ToolPoseCriteria.track_position_and_orientation(
                xyz=[1.0, 1.0, 1.0], rpy=[0.3, 0.3, 0.3],
            ),
            "right_hand": ToolPoseCriteria.track_position_and_orientation(
                xyz=[1.0, 1.0, 1.0], rpy=[0.3, 0.3, 0.3],
            ),
        },
    )
    retargeter = MotionRetargeter(cfg)
    result = retargeter.solve_frame(goal_tool_pose)
"""

from __future__ import annotations

from typing import List, Optional

import torch
from tqdm import trange

from curobo._src.motion.motion_retargeter_cfg import MotionRetargeterCfg
from curobo._src.motion.motion_retargeter_result import RetargetResult
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.solver.solver_mpc import MPCSolver
from curobo._src.solver.solver_mpc_cfg import MPCSolverCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.sequence_tool_pose import SequenceGoalToolPose
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util.logging import log_and_raise


class MotionRetargeter:
    """IK / MPC-based motion retargeter.

    Tracks internal state across calls to :meth:`solve_frame`. The first
    call uses global IK (many seeds). Subsequent calls use either
    warm-started local IK or MPC depending on ``config.use_mpc``.

    Call :meth:`reset` to clear state and start a new sequence.
    """

    def __init__(self, config: MotionRetargeterCfg):
        self._config = config
        self._num_envs = config.num_envs
        self._tool_pose_criteria = config.tool_pose_criteria
        self._global_ik_solver = self._build_global_ik_solver()
        self._joint_names = list(self._global_ik_solver.joint_names)
        self._action_dim = self._global_ik_solver.action_dim

        if config.use_mpc:
            self._mpc_solver = self._build_mpc_solver()
            self._local_ik_solver = None
        else:
            self._local_ik_solver = self._build_local_ik_solver()
            self._mpc_solver = None

        self._prev_solution: Optional[torch.Tensor] = None
        self._prev_velocity: Optional[torch.Tensor] = None
        self._mpc_state: Optional[JointState] = None

    @property
    def joint_names(self) -> List[str]:
        """Joint names in solver DOF order."""
        return self._joint_names

    @property
    def action_dim(self) -> int:
        return self._global_ik_solver.action_dim

    @property
    def tool_frames(self) -> List[str]:
        return self._global_ik_solver.tool_frames

    @property
    def kinematics(self) -> Kinematics:
        return self._global_ik_solver.kinematics

    @property
    def default_joint_state(self) -> JointState:
        return self._global_ik_solver.default_joint_state

    @property
    def num_dof(self) -> int:
        """Deprecated alias for :attr:`action_dim`."""
        return self.action_dim

    @property
    def config(self) -> MotionRetargeterCfg:
        """Current configuration."""
        return self._config

    def reset(self) -> None:
        """Clear warm-start state. Next :meth:`solve_frame` uses global IK."""
        self._prev_solution = None
        self._prev_velocity = None
        self._mpc_state = None

    def solve_frame(self, goal_tool_poses: GoalToolPose) -> RetargetResult:
        """Solve one frame of retargeting.

        Args:
            goal_tool_poses: Target poses for tracked links. Position shape
                ``(num_envs, 1, num_links, num_goalset, 3)``, quaternion wxyz shape
                ``(num_envs, 1, num_links, num_goalset, 4)``.

        Returns:
            :class:`RetargetResult` with ``joint_state`` shape
            ``(num_envs, num_dof)`` and optional MPC ``trajectory``.

        Raises:
            ValueError: If ``goal_tool_poses`` batch size does not match
                ``num_envs`` from the config.
        """
        if goal_tool_poses.batch_size != self._num_envs:
            log_and_raise(
                f"GoalToolPose batch size ({goal_tool_poses.batch_size}) does not "
                f"match num_envs ({self._num_envs}) from config"
            )

        is_first_frame = self._prev_solution is None

        if is_first_frame:
            return self._solve_global_ik(goal_tool_poses)
        elif self._config.use_mpc:
            return self._solve_mpc_frame(goal_tool_poses)
        else:
            return self._solve_local_ik(goal_tool_poses)

    def solve_sequence(self, tool_poses: SequenceGoalToolPose) -> RetargetResult:
        """Solve a full sequence of retargeting targets.

        Args:
            tool_poses: Sequence of goal tool poses with shape
                ``(num_frames, num_envs, num_links, num_goalset, 3/4)``.

        Returns:
            :class:`RetargetResult` with ``joint_state`` position shape
            ``(num_envs, num_output_frames, num_dof)`` and optional
            stacked MPC ``trajectory``.

        Raises:
            ValueError: If ``tool_poses.num_envs`` does not match
                ``num_envs`` from the config.
        """
        if tool_poses.num_envs != self._num_envs:
            log_and_raise(
                f"SequenceGoalToolPose num_envs ({tool_poses.num_envs}) does not "
                f"match num_envs ({self._num_envs}) from config"
            )

        self.reset()

        joint_states: list[RetargetResult] = []
        trajectories: list[JointState] = []

        for t in trange(tool_poses.num_frames, desc="Retargeting"):
            frame_tool_pose = tool_poses.get_frame(t)
            result = self.solve_frame(frame_tool_pose)
            joint_states.append(result.joint_state)
            if result.trajectory is not None:
                trajectories.append(result.trajectory)

        stacked_js = JointState(
            position=torch.stack([joint_state.position for joint_state in joint_states], dim=1),
            velocity=torch.stack([joint_state.velocity for joint_state in joint_states], dim=1)
            if joint_states[0].velocity is not None
            else None,
            acceleration=torch.stack(
                [joint_state.acceleration for joint_state in joint_states], dim=1
            )
            if joint_states[0].acceleration is not None
            else None,
            joint_names=self._joint_names,
        )

        stacked_traj = None
        if trajectories:
            stacked_traj = JointState(
                position=torch.cat(
                    [t.position for t in trajectories], dim=1
                ),
                velocity=torch.cat(
                    [t.velocity for t in trajectories], dim=1
                )
                if trajectories[0].velocity is not None
                else None,
                acceleration=torch.cat(
                    [t.acceleration for t in trajectories], dim=1
                )
                if trajectories[0].acceleration is not None
                else None,
                joint_names=self._joint_names,
            )

        return RetargetResult(
            joint_state=stacked_js,
            trajectory=stacked_traj,
        )

    def _solve_global_ik(self, goal_tool_poses: GoalToolPose) -> RetargetResult:
        """First frame: global IK with many seeds."""
        result = self._global_ik_solver.solve_pose(
            goal_tool_poses=goal_tool_poses,
            return_seeds=1,
        )
        sol = result.js_solution.position.view(self._num_envs, self._action_dim)
        self._prev_solution = sol.clone()

        if self._config.use_mpc and self._mpc_solver is not None:
            self._mpc_state = JointState.from_position(
                sol.clone(), joint_names=self._joint_names
            )
            self._mpc_solver.setup(self._mpc_state.clone())

        return RetargetResult(
            joint_state=JointState.from_position(
                sol, joint_names=self._joint_names
            ),
        )

    def _solve_local_ik(self, goal_tool_poses: GoalToolPose) -> RetargetResult:
        """Subsequent frames in IK mode: warm-started, velocity-limited."""
        seed = self._prev_solution.view(self._num_envs, 1, -1)
        current_state = JointState.from_position(
            self._prev_solution.clone(), joint_names=self._joint_names
        )
        if self._prev_velocity is not None:
            current_state.velocity = self._prev_velocity.clone()

        result = self._local_ik_solver.solve_pose(
            goal_tool_poses=goal_tool_poses,
            seed_config=seed,
            current_state=current_state,
            return_seeds=1,
        )
        sol = result.js_solution.position.view(self._num_envs, self._action_dim)

        if result.js_solution.velocity is not None:
            self._prev_velocity = result.js_solution.velocity.view(
                self._num_envs, self._action_dim
            )
        self._prev_solution = sol.clone()

        return RetargetResult(
            joint_state=JointState.from_position(
                sol, joint_names=self._joint_names
            ),
        )

    def _solve_mpc_frame(self, goal_tool_poses: GoalToolPose) -> RetargetResult:
        """Subsequent frames in MPC mode: optimize action sequence."""
        self._mpc_solver.update_goal_tool_poses(
            goal_tool_poses, run_ik=False,
        )

        traj_positions = []
        traj_velocities = []
        traj_accelerations = []

        for _ in range(self._config.steps_per_target):
            mpc_result = self._mpc_solver.optimize_action_sequence(
                self._mpc_state
            )
            if (
                mpc_result.action_sequence is not None
                and mpc_result.action_sequence.position.shape[1] > 0
            ):
                act_seq = mpc_result.action_sequence

                self._mpc_state = JointState(
                    position=act_seq.position[:, -1, :].clone(),
                    velocity=act_seq.velocity[:, -1, :].clone(),
                    acceleration=act_seq.acceleration[:, -1, :].clone(),
                    jerk=act_seq.jerk[:, -1, :].clone(),
                    joint_names=self._joint_names,
                )

                # Store only the executed endpoint, not the full horizon.
                traj_positions.append(self._mpc_state.position.unsqueeze(1))
                traj_velocities.append(self._mpc_state.velocity.unsqueeze(1))
                traj_accelerations.append(
                    self._mpc_state.acceleration.unsqueeze(1)
                )

        sol_pos = self._mpc_state.position.view(-1, self._action_dim)
        sol_vel = self._mpc_state.velocity.view(-1, self._action_dim)
        sol_acc = self._mpc_state.acceleration.view(-1, self._action_dim)
        self._prev_solution = sol_pos.clone()

        trajectory = None
        if traj_positions:
            trajectory = JointState(
                position=torch.cat(traj_positions, dim=1),
                velocity=torch.cat(traj_velocities, dim=1),
                acceleration=torch.cat(traj_accelerations, dim=1),
                joint_names=self._joint_names,
            )

        return RetargetResult(
            joint_state=JointState(
                position=sol_pos,
                velocity=sol_vel,
                acceleration=sol_acc,
                joint_names=self._joint_names,
            ),
            trajectory=trajectory,
        )

    def _build_global_ik_solver(self) -> IKSolver:
        """Build IK solver for frame 0: global search, no velocity limit."""
        cfg = self._config
        override_num_iters = {"lbfgs": cfg.global_ik_num_iters}
        ik_config = IKSolverCfg.create(
            robot=cfg.robot,
            num_seeds=cfg.num_seeds_global,
            position_tolerance=cfg.position_tolerance,
            orientation_tolerance=cfg.orientation_tolerance,
            self_collision_check=cfg.self_collision_check,
            scene_model=cfg.scene_model,
            optimization_dt=None,
            override_iters_for_multi_link_ik=None,
            override_optimizer_num_iters=override_num_iters,
            optimizer_configs=cfg.ik_optimizer_configs,
            device_cfg=cfg.device_cfg,
            load_collision_spheres=cfg.load_collision_spheres,
            optimizer_collision_activation_distance=cfg.collision_activation_distance,
        )
        solver = IKSolver(ik_config)
        solver.update_tool_pose_criteria(self._tool_pose_criteria)
        return solver

    def _build_local_ik_solver(self) -> IKSolver:
        """Build IK solver for frames 1+: warm-started, velocity-limited."""
        cfg = self._config
        override_num_iters = {"lbfgs": cfg.local_ik_num_iters}
        ik_config = IKSolverCfg.create(
            robot=cfg.robot,
            optimizer_configs=cfg.ik_optimizer_configs,
            num_seeds=cfg.num_seeds_local,
            position_tolerance=cfg.position_tolerance,
            orientation_tolerance=cfg.orientation_tolerance,
            self_collision_check=cfg.self_collision_check,
            scene_model=cfg.scene_model,
            optimization_dt=cfg.optimization_dt,
            override_iters_for_multi_link_ik=None,
            override_optimizer_num_iters=override_num_iters,
            device_cfg=cfg.device_cfg,
            load_collision_spheres=cfg.load_collision_spheres,
            velocity_regularization_weight=cfg.velocity_regularization_weight,
            acceleration_regularization_weight=cfg.acceleration_regularization_weight,
            optimizer_collision_activation_distance=cfg.collision_activation_distance,
        )
        solver = IKSolver(ik_config)
        solver.update_tool_pose_criteria(self._tool_pose_criteria)
        solver.use_lm_seed = False
        solver.exit_early = False
        return solver

    def _build_mpc_solver(self):
        """Build MPC solver for frames 1+: trajectory optimization."""
        cfg = self._config
        mpc_config = MPCSolverCfg.create(
            robot=cfg.robot,
            optimization_dt=cfg.optimization_dt,
            self_collision_check=cfg.self_collision_check,
            scene_model=cfg.scene_model,
            optimizer_configs=cfg.mpc_optimizer_configs,
            device_cfg=cfg.device_cfg,
            load_collision_spheres=cfg.load_collision_spheres,
            num_control_points=cfg.num_control_points,
            optimizer_collision_activation_distance=cfg.collision_activation_distance,
            warm_start_optimization_num_iters=cfg.mpc_warm_start_num_iters,
            cold_start_optimization_num_iters=cfg.mpc_cold_start_num_iters,
        )
        solver = MPCSolver(mpc_config)
        solver.update_tool_pose_criteria(self._tool_pose_criteria)
        return solver
