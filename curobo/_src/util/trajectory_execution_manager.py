# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Standard Library
from typing import Optional

# Third Party
import torch

# CuRobo
from curobo._src.rollout.metrics import RolloutMetrics
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_trajectory_ops import (
    get_joint_state_at_horizon_index,
    trim_joint_state_trajectory,
)
from curobo._src.state.state_robot import RobotState
from curobo._src.util.logging import log_and_raise


class TrajectoryExecutionManager:
    """This class holds the current solution of trajectory optimization.

    It provides interface to get next command from the current solution by keeping track of
    the previously executed command index. This is mainly used for executing trajectory one time
    step at a time with re-optimization after each action (e.g., MPC). The assumption is that the
    state trajectory is obtained by rolling out actions  interpolated by some
    interpolation steps to obtain the state trajectory.

    Args:
        interpolation_steps: Number of interpolation steps between two consecutive commands.
        command_start_idx: First state trajectory index to expose as a command.
        command_end_idx: End index for the command sequence. Defaults to two interpolation
            windows, which preserves the existing non-offset behavior.
    """

    def __init__(
        self,
        interpolation_steps: int,
        command_start_idx: int = 0,
        command_end_idx: Optional[int] = None,
    ):
        self._current_robot_state_trajectory = None
        self._current_joint_state_trajectory = None
        self._current_action_trajectory = None
        self._current_metrics = None
        self._current_command_idx = 0
        self.interpolation_steps = interpolation_steps
        self.command_start_idx = command_start_idx
        self.command_end_idx = (
            command_end_idx if command_end_idx is not None else interpolation_steps * 2
        )

    def get_current_metrics(self) -> RolloutMetrics:
        return self._current_metrics

    def update_robot_state_trajectory(self, robot_state_trajectory: RobotState):
        self._current_robot_state_trajectory = robot_state_trajectory

    def update_state_action_buffers(
        self, state_trajectory: JointState, action_trajectory: torch.Tensor
    ):
        self._current_joint_state_trajectory = state_trajectory
        self._current_action_trajectory = action_trajectory
        self._current_command_idx = 0
        self._current_metrics = None

    def update_state_action_metrics_buffers(
        self, state_trajectory: JointState, action_trajectory: torch.Tensor, metrics: RolloutMetrics
    ):
        """Update the state and action buffers.

        Args:
            state_trajectory: State trajectory. Shape: (batch_size, n_states, action_dim).
            action_trajectory: Action trajectory. Shape: (batch_size, n_actions, action_dim).
        """
        self._current_joint_state_trajectory = state_trajectory
        self._current_action_trajectory = action_trajectory
        self._current_metrics = metrics
        self._current_command_idx = 0

    def get_next_command(self) -> JointState:
        """Get the next command from the current state trajectory.

        Returns:
            Next command. Shape: (batch_size, action_dim).
        """
        if not self.has_valiaction_dim_buffer():
            log_and_raise("No valid action buffer, call update_action_trajectory first")
        horizon_index = self.command_start_idx + self._current_command_idx
        next_command = get_joint_state_at_horizon_index(
            self._current_joint_state_trajectory, horizon_index
        )
        self._current_command_idx += 1
        return next_command

    def get_command_sequence(self) -> torch.Tensor:
        """Get the action sequence.

        Returns:
            Action sequence. Shape: (batch_size, interpolation_steps, action_dim).
        """
        if not self.has_valiaction_dim_buffer():
            log_and_raise("No valid action buffer, call update_action_trajectory first")
        action_sequence = trim_joint_state_trajectory(
            self._current_joint_state_trajectory,
            start_idx=self.command_start_idx,
            end_idx=self.command_end_idx,
        )
        return action_sequence

    def has_valid_next_command(self) -> bool:
        """Check if the next command is valid.

        Returns:
            True if the next command is valid, False otherwise.
        """
        if not self.has_valiaction_dim_buffer():
            return False
        elif self._current_command_idx >= self.interpolation_steps:
            return False
        else:
            return True

    def has_valiaction_dim_buffer(self) -> bool:
        """Check if the action buffer is valid.

        Returns:
            True if the action buffer is valid, False otherwise.
        """
        if self._current_action_trajectory is None:
            return False
        else:
            return True

    def get_action_buffer(self) -> torch.Tensor:
        if not self.has_valiaction_dim_buffer():
            log_and_raise("No valid action buffer, call update_action_trajectory first")
        return self._current_action_trajectory

    def get_robot_state_sequence(self) -> RobotState:
        if not self.has_valid_robot_state_trajectory():
            log_and_raise("No valid robot state trajectory, call update_robot_state_trajectory first")
        return self._current_robot_state_trajectory

    def has_valid_robot_state_trajectory(self) -> bool:
        if self._current_robot_state_trajectory is None:
            return False
        else:
            return True

    def get_shifteaction_dim_buffer(self) -> torch.Tensor:
        """Get the action buffer.

        Returns:
            Action buffer. Shape: (batch_size, n_actions, action_dim).
        """
        if not self.has_valiaction_dim_buffer():
            log_and_raise("No valid action buffer, call update_action_trajectory first")
        action_index = self._current_command_idx // self.interpolation_steps
        if action_index >= self._current_action_trajectory.shape[1]:
            log_and_raise("Action index out of bounds, call update_action_trajectory first")
        action_buffer = self._current_action_trajectory.clone()

        # Only roll and repeat the last action if there's more than one action
        if action_buffer.shape[-2] > 1:
            action_buffer = action_buffer.roll(-1, dims=-2)
            action_buffer[..., -1:, :] = action_buffer[..., -2:-1, :]
        action_buffer = action_buffer
        return action_buffer

    def update_action_buffer(self, action_buffer: torch.Tensor):
        """Update the action buffer.

        Args:
            action_buffer: Action buffer. Shape: (batch_size, n_actions, action_dim).
        """
        self._current_action_trajectory = action_buffer
        self._current_command_idx = 0
        self._current_joint_state_trajectory = None
