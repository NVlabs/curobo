#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

# CuRobo
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState


class ReacherSolveType(Enum):
    # TODO: how to differentiate between goal pose and goal config?
    SINGLE = 0
    GOALSET = 1
    BATCH = 2
    BATCH_GOALSET = 3
    BATCH_ENV = 4
    BATCH_ENV_GOALSET = 5


@dataclass
class ReacherSolveState:
    solve_type: ReacherSolveType
    batch_size: int
    n_envs: int
    n_goalset: int = 1
    batch_env: bool = False
    batch_retract: bool = False
    batch_mode: bool = False
    num_seeds: Optional[int] = None
    num_ik_seeds: Optional[int] = None
    num_graph_seeds: Optional[int] = None
    num_trajopt_seeds: Optional[int] = None
    num_mpc_seeds: Optional[int] = None

    def __post_init__(self):
        if self.n_envs == 1:
            self.batch_env = False
        else:
            self.batch_env = True
        if self.batch_size > 1:
            self.batch_mode = True
        if self.num_seeds is None:
            self.num_seeds = self.num_ik_seeds
        if self.num_seeds is None:
            self.num_seeds = self.num_trajopt_seeds
        if self.num_seeds is None:
            self.num_seeds = self.num_graph_seeds
        if self.num_seeds is None:
            self.num_seeds = self.num_mpc_seeds

    def clone(self):
        return ReacherSolveState(
            solve_type=self.solve_type,
            n_envs=self.n_envs,
            batch_size=self.batch_size,
            n_goalset=self.n_goalset,
            batch_env=self.batch_env,
            batch_retract=self.batch_retract,
            batch_mode=self.batch_mode,
            num_seeds=self.num_seeds,
            num_ik_seeds=self.num_ik_seeds,
            num_graph_seeds=self.num_graph_seeds,
            num_trajopt_seeds=self.num_trajopt_seeds,
            num_mpc_seeds=self.num_mpc_seeds,
        )

    def get_batch_size(self):
        return self.num_seeds * self.batch_size

    def get_ik_batch_size(self):
        return self.num_ik_seeds * self.batch_size

    def create_goal_buffer(
        self,
        goal_pose: Pose,
        goal_state: Optional[JointState] = None,
        retract_config: Optional[T_BDOF] = None,
        link_poses: Optional[Dict[str, Pose]] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ):
        # TODO: Refactor to account for num_ik_seeds or num_trajopt_seeds
        batch_retract = True
        if retract_config is None or retract_config.shape[0] != goal_pose.batch:
            batch_retract = False
        goal_buffer = Goal.create_idx(
            pose_batch_size=self.batch_size,
            batch_env=self.batch_env,
            batch_retract=batch_retract,
            num_seeds=self.num_seeds,
            tensor_args=tensor_args,
        )
        goal_buffer.goal_pose = goal_pose
        goal_buffer.retract_state = retract_config
        goal_buffer.goal_state = goal_state
        goal_buffer.links_goal_pose = link_poses
        return goal_buffer

    def update_goal_buffer(
        self,
        goal_pose: Pose,
        goal_state: Optional[JointState] = None,
        retract_config: Optional[T_BDOF] = None,
        link_poses: Optional[List[Pose]] = None,
        current_solve_state: Optional[ReacherSolveState] = None,
        current_goal_buffer: Optional[Goal] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ):
        solve_state = self
        # create goal buffer by comparing to existing solve type
        update_reference = False
        if (
            current_solve_state is None
            or current_goal_buffer is None
            or current_solve_state != solve_state
            or (current_goal_buffer.retract_state is None and retract_config is not None)
            or (current_goal_buffer.goal_state is None and goal_state is not None)
            or (current_goal_buffer.links_goal_pose is None and link_poses is not None)
        ):
            current_solve_state = solve_state
            current_goal_buffer = solve_state.create_goal_buffer(
                goal_pose, goal_state, retract_config, link_poses, tensor_args
            )
            update_reference = True
        else:
            current_goal_buffer.goal_pose.copy_(goal_pose)
            if retract_config is not None:
                current_goal_buffer.retract_state.copy_(retract_config)
            if goal_state is not None:
                current_goal_buffer.goal_state.copy_(goal_state)
            if link_poses is not None:
                for k in link_poses.keys():
                    current_goal_buffer.links_goal_pose[k].copy_(link_poses[k])
        return current_solve_state, current_goal_buffer, update_reference

    def update_goal(
        self,
        goal: Goal,
        current_solve_state: Optional[ReacherSolveState] = None,
        current_goal_buffer: Optional[Goal] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ):
        solve_state = self

        update_reference = False
        if (
            current_solve_state is None
            or current_goal_buffer is None
            or current_solve_state != solve_state
            or (current_goal_buffer.goal_state is None and goal.goal_state is not None)
            or (current_goal_buffer.goal_state is not None and goal.goal_state is None)
        ):
            # TODO: Check for change in update idx buffers, currently we assume
            # that solve_state captures difference in idx buffers
            current_solve_state = solve_state
            current_goal_buffer = goal.create_index_buffers(
                solve_state.batch_size,
                solve_state.batch_env,
                solve_state.batch_retract,
                solve_state.num_seeds,
                tensor_args,
            )
            update_reference = True
        else:
            current_goal_buffer.copy_(goal, update_idx_buffers=False)
        return current_solve_state, current_goal_buffer, update_reference


@dataclass
class MotionGenSolverState:
    solve_type: ReacherSolveType
    ik_solve_state: ReacherSolveState
    trajopt_solve_state: ReacherSolveState
