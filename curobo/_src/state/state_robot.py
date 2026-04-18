# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import List, Optional, Union

# Third Party
import torch

# CuRobo
from curobo._src.robot.kinematics.kinematics import KinematicsState
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import ToolPose
from curobo._src.util.helpers import list_idx_if_not_none
from curobo._src.util.logging import log_and_raise

from .state_base import State
from .state_joint import JointState
from .state_joint_trajectory_ops import (
    copy_joint_state_at_batch_seed_indices,
    copy_joint_state_only_index,
)


def _kinematics_uses_merged_batch_seed_dim(
    joint_position: torch.Tensor,
    robot_spheres: Optional[torch.Tensor],
    tool_position: Optional[torch.Tensor] = None,
) -> bool:
    """Detect FK layout ``[batch * num_seeds, horizon, ...]`` vs joint ``[B, S, ...]``."""
    ref = robot_spheres if robot_spheres is not None else tool_position
    if ref is None or ref.ndim != 4:
        return False
    if joint_position.ndim == 3:
        return ref.shape[0] == joint_position.shape[0] * joint_position.shape[1]
    if joint_position.ndim == 4:
        return ref.shape[0] == joint_position.shape[0] * joint_position.shape[1]
    return False


@dataclass
class RobotState(State):
    joint_state: JointState
    joint_torque: Optional[torch.Tensor] = None
    cuda_robot_model_state: Optional[KinematicsState] = None

    def data_ptr(self):
        return self.joint_state.data_ptr()

    def __getitem__(self, idx: Union[int, torch.Tensor]):
        d_list = [
            self.joint_state,
            self.joint_torque,
            self.cuda_robot_model_state,
        ]
        idx_vals = list_idx_if_not_none(d_list, idx)
        return RobotState(*idx_vals)

    def detach(self):
        return RobotState(
            joint_state=self.joint_state.detach(),
            joint_torque=self.joint_torque.detach() if self.joint_torque is not None else None,
            cuda_robot_model_state=(
                self.cuda_robot_model_state.detach()
                if self.cuda_robot_model_state is not None
                else None
            ),
        )

    @property
    def robot_spheres(self) -> Optional[torch.Tensor]:
        if self.cuda_robot_model_state is None:
            return None
        return self.cuda_robot_model_state.robot_spheres

    @property
    def link_poses(self) -> Optional[ToolPose]:
        if self.cuda_robot_model_state is None:
            return None
        return self.cuda_robot_model_state.tool_poses

    @property
    def tool_poses(self) -> Optional[ToolPose]:
        if self.cuda_robot_model_state is None:
            return None
        return self.cuda_robot_model_state.tool_poses

    @property
    def tool_frames(self) -> List[str]:
        if self.tool_poses is not None:
            return self.tool_poses.tool_frames
        else:
            return []

    def __len__(self):
        return len(self.joint_state)

    def copy_at_batch_seed_indices(
        self, other: RobotState, batch_idx: torch.Tensor, seed_idx: torch.Tensor
    ):
        """Copy robot state at specific batch and seed indices"""
        copy_joint_state_at_batch_seed_indices(
            self.joint_state, other.joint_state, batch_idx, seed_idx
        )
        q = self.joint_state.position
        tp_pos = self.tool_poses.position if self.tool_poses is not None else None
        use_linear = _kinematics_uses_merged_batch_seed_dim(q, self.robot_spheres, tp_pos)
        if use_linear:
            num_seeds = q.shape[1]
            lin = batch_idx * num_seeds + seed_idx
            if self.robot_spheres is not None and other.robot_spheres is not None:
                self.robot_spheres[lin] = other.robot_spheres[lin]
            if self.tool_poses is not None and other.tool_poses is not None:
                self.tool_poses.position[lin] = other.tool_poses.position[lin]
                self.tool_poses.quaternion[lin] = other.tool_poses.quaternion[lin]
        else:
            if self.robot_spheres is not None and other.robot_spheres is not None:
                self.robot_spheres[batch_idx, seed_idx] = other.robot_spheres[batch_idx, seed_idx]
            if self.tool_poses is not None and other.tool_poses is not None:
                self.tool_poses.position[batch_idx, seed_idx] = other.tool_poses.position[
                    batch_idx, seed_idx
                ]
                self.tool_poses.quaternion[batch_idx, seed_idx] = other.tool_poses.quaternion[
                    batch_idx, seed_idx
                ]
        if self.joint_torque is not None:
            self.joint_torque[batch_idx, seed_idx] = other.joint_torque[batch_idx, seed_idx]
        return self

    def copy_only_index(self, other: RobotState, index: Union[int, torch.Tensor]):
        """Copy robot state at specific indices"""
        copy_joint_state_only_index(self.joint_state, other.joint_state, index)
        if self.robot_spheres is not None:
            self.robot_spheres[index] = other.robot_spheres[index]
        if self.tool_poses is not None:
            self.tool_poses.position[index] = other.tool_poses.position[index]
            self.tool_poses.quaternion[index] = other.tool_poses.quaternion[index]
        if self.joint_torque is not None:
            self.joint_torque[index] = other.joint_torque[index]
        return self

    def get_link_pose(self, link_name: str) -> Pose:
        if self.tool_poses is None:
            log_and_raise("Link poses are not set")
        return self.tool_poses.get_link_pose(link_name)

    def clone(self) -> RobotState:
        return RobotState(
            joint_state=self.joint_state.clone() if self.joint_state is not None else None,
            joint_torque=self.joint_torque.clone() if self.joint_torque is not None else None,
            cuda_robot_model_state=(
                self.cuda_robot_model_state.clone()
                if self.cuda_robot_model_state is not None
                else None
            ),
        )

    def copy_(self, other: RobotState):
        self.joint_state.copy_(other.joint_state)
        if self.joint_torque is not None:
            self.joint_torque.copy_(other.joint_torque)
        if self.cuda_robot_model_state is not None:
            self.cuda_robot_model_state.copy_(other.cuda_robot_model_state)
        return self

