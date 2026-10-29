# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Goal registry manager for optimization-based solvers."""

from __future__ import annotations

# Standard Library
from typing import Optional, Tuple, Union, cast

# Third Party
import torch

# CuRobo
from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.solver.solve_state import SolveState
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.target_tool_relative_transform import ToolRelativeTransformTarget
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util.logging import log_and_raise


def _relative_targets_are_compatible(
    current: ToolRelativeTransformTarget,
    incoming: ToolRelativeTransformTarget,
) -> bool:
    """Return whether target values can be copied without changing captured storage."""
    return (
        current.reference_tool_frame == incoming.reference_tool_frame
        and current.tool_frames == incoming.tool_frames
        and current.position.shape == incoming.position.shape
        and current.quaternion.shape == incoming.quaternion.shape
        and current.device == incoming.device
        and current.dtype == incoming.dtype
    )


_UNSET_RELATIVE_TARGET = object()


class GoalManager:
    """Manages goal buffer creation, updates, and access for solvers.

    This class encapsulates all operations related to the goal buffer, which stores
    target poses, states, and configuration for trajectory optimization and motion planning.
    """

    def __init__(self, device_cfg: DeviceCfg):
        """Initialize the goal manager.

        Args:
            device_cfg: Device and dtype configuration.
        """
        self.device_cfg = device_cfg
        self._solve_state = None
        self._goal_buffer = None
        self._col = None

    def create_goal_buffer(
        self,
        solve_state: SolveState,
        goal_tool_poses: Optional[GoalToolPose] = None,
        goal_js: Optional[JointState] = None,
        current_js: Optional[JointState] = None,
        seed_goal_js: Optional[JointState] = None,
        current_state_dt: Optional[torch.Tensor] = None,
        tool_relative_transform_target: Optional[ToolRelativeTransformTarget] = None,
    ) -> GoalRegistry:
        """Create a goal buffer from goal pose and other problem targets.

        Args:
            solve_state: The SolveState containing problem information.
            goal_tool_poses: Target poses for specified links.
            goal_js: Joint configuration to reach.
            current_js: Start state of the robot.
            seed_goal_js: Seed goal state for trajectory optimization.
            current_state_dt: Time since the current state was observed.
                Shape: ``(n_problems,)``. When ``None`` and ``current_js.dt`` is set,
                ``current_js.dt`` is used instead.
            tool_relative_transform_target: Per-problem transforms expressed in one
                moving reference tool frame.

        Returns:
            Goal buffer with the goal pose, goal state, and link poses.
        """
        if solve_state.num_seeds is None:
            log_and_raise("Number of seeds is not set. Please set the number of seeds.")
        goal_buffer = GoalRegistry.create_idx(
            pose_batch_size=solve_state.batch_size,
            multi_env=solve_state.multi_env,
            num_seeds=solve_state.num_seeds,
            device_cfg=self.device_cfg,
            seed_goal_state=seed_goal_js,
            repeat_seed_idx_buffers=False,
            tool_relative_transform_target=tool_relative_transform_target,
        )

        goal_buffer.goal_js = goal_js

        if goal_tool_poses is not None:
            # Subsequent solves update this storage in place. Own the tensors
            # so approach planning cannot overwrite the caller's grasp poses.
            goal_buffer.link_goal_poses = goal_tool_poses.clone()

        if tool_relative_transform_target is not None:
            num_target_groups = tool_relative_transform_target.num_target_groups
            if num_target_groups == 1:
                goal_buffer.idxs_tool_relative_transform_target.zero_()
            elif num_target_groups != solve_state.batch_size:
                log_and_raise(
                    "tool_relative_transform_target must have either one target group or "
                    f"one per problem ({solve_state.batch_size}), got {num_target_groups}"
                )
            goal_buffer.tool_relative_transform_target = tool_relative_transform_target

        if current_js is not None:
            goal_buffer.current_js = current_js

        # Extract current_state_dt: prefer explicit argument, fall back to current_js.dt
        if current_state_dt is not None:
            goal_buffer.current_state_dt = current_state_dt
        elif current_js is not None and current_js.dt is not None:
            goal_buffer.current_state_dt = current_js.dt

        return goal_buffer

    def update_goal_buffer(
        self,
        solve_state: SolveState,
        goal_tool_poses: Optional[GoalToolPose] = None,
        current_js: Optional[JointState] = None,
        seed_goal_js: Optional[JointState] = None,
        goal_js: Optional[JointState] = None,
        use_implicit_goal: bool = False,
        current_state_dt: Optional[torch.Tensor] = None,
        tool_relative_transform_target: Union[
            ToolRelativeTransformTarget, None, object
        ] = _UNSET_RELATIVE_TARGET,
    ) -> Tuple[GoalRegistry, bool]:
        """Update the internal goal buffer.

        Args:
            solve_state: Defines the problem type, batch size, seeds, etc.
            goal_tool_poses: Target poses for specified links.
            current_js: The initial state.
            seed_goal_js: State based on seed trajectory end.
            goal_js: Optional explicit target joint state.
            use_implicit_goal: Flag to use seed_goal_js as the goal.
            current_state_dt: Time since the current state was observed.
                Shape: ``(n_problems,)``. When ``None`` and ``current_js.dt`` is set,
                ``current_js.dt`` is used instead.
            tool_relative_transform_target: Per-problem transforms expressed in one
                moving reference tool frame. Omitting this argument preserves an
                existing target; passing ``None`` removes it.

        Returns:
            Tuple of (goal buffer, whether reference was updated).
        """
        update_reference = False
        target_was_supplied = tool_relative_transform_target is not _UNSET_RELATIVE_TARGET
        effective_relative_target = cast(
            Optional[ToolRelativeTransformTarget],
            (
                tool_relative_transform_target
                if target_was_supplied
                else (
                    self._goal_buffer.tool_relative_transform_target
                    if self._goal_buffer is not None
                    else None
                )
            ),
        )

        if (
            self._solve_state is None
            or self._goal_buffer is None
            or (self._goal_buffer.goal_js is None and goal_js is not None)
            or (self._goal_buffer.link_goal_poses is None and goal_tool_poses is not None)
            or (self._goal_buffer.current_js is None and current_js is not None)
            or (self._goal_buffer.seed_goal_js is None and seed_goal_js is not None)
            or (
                self._goal_buffer.tool_relative_transform_target is None
                and effective_relative_target is not None
            )
            or (
                self._goal_buffer.tool_relative_transform_target is not None
                and effective_relative_target is None
            )
            or (
                self._goal_buffer.tool_relative_transform_target is not None
                and effective_relative_target is not None
                and not _relative_targets_are_compatible(
                    self._goal_buffer.tool_relative_transform_target,
                    effective_relative_target,
                )
            )
        ):
            update_reference = True
        elif self._solve_state != solve_state:
            padded_link_goal_poses = self._get_padded_goalset_for_links(
                solve_state, self._solve_state, self._goal_buffer, goal_tool_poses
            )
            if padded_link_goal_poses is not None:
                goal_tool_poses = padded_link_goal_poses
            else:
                update_reference = True

        if update_reference:
            self._solve_state = solve_state
            self._goal_buffer = self.create_goal_buffer(
                solve_state=solve_state,
                goal_tool_poses=goal_tool_poses,
                goal_js=goal_js,
                current_js=current_js,
                seed_goal_js=seed_goal_js,
                current_state_dt=current_state_dt,
                tool_relative_transform_target=effective_relative_target,
            )
            self.update_batch_helper(solve_state.batch_size)
        else:
            if current_js is not None:
                self._goal_buffer.current_js.copy_(current_js)
            if goal_js is not None:
                self._goal_buffer.goal_js.copy_(goal_js)
            if goal_tool_poses is not None:
                if self._goal_buffer.link_goal_poses is not None:
                    self._goal_buffer.link_goal_poses.position.copy_(goal_tool_poses.position)
                    self._goal_buffer.link_goal_poses.quaternion.copy_(goal_tool_poses.quaternion)
            if target_was_supplied and effective_relative_target is not None:
                self._goal_buffer.tool_relative_transform_target.copy_(effective_relative_target)
            if seed_goal_js is not None:
                self._goal_buffer.seed_goal_js.copy_(seed_goal_js)
            # Update current_state_dt in the existing buffer
            effective_dt = current_state_dt
            if effective_dt is None and current_js is not None and current_js.dt is not None:
                effective_dt = current_js.dt
            if effective_dt is not None:
                if self._goal_buffer.current_state_dt is not None:
                    self._goal_buffer.current_state_dt.copy_(effective_dt)
                else:
                    self._goal_buffer.current_state_dt = effective_dt

        if use_implicit_goal and self._goal_buffer.seed_enable_implicit_goal_js is not None:
            self._goal_buffer.seed_enable_implicit_goal_js[:] = int(use_implicit_goal)
        return self._goal_buffer, update_reference

    def update_from_goal_registry(
        self,
        solve_state: SolveState,
        goal: GoalRegistry,
    ) -> Tuple[GoalRegistry, bool]:
        """Update the goal buffer with values from a Rollout goal.

        Args:
            solve_state: Defines the problem type, batch size, seeds, etc.
            goal: Rollout goal to update the goal buffer.

        Returns:
            Tuple of (goal buffer, whether reference was updated).
        """
        update_reference = False

        if (
            self._solve_state is None
            or self._goal_buffer is None
            or (self._goal_buffer.goal_js is None and goal.goal_js is not None)
            or (self._goal_buffer.goal_js is not None and goal.goal_js is None)
            or (self._goal_buffer.seed_goal_js is not None and goal.seed_goal_js is None)
            or (self._goal_buffer.seed_goal_js is None and goal.seed_goal_js is not None)
            or (
                self._goal_buffer.tool_relative_transform_target is None
                and goal.tool_relative_transform_target is not None
            )
            or (
                self._goal_buffer.tool_relative_transform_target is not None
                and goal.tool_relative_transform_target is None
            )
            or (
                self._goal_buffer.tool_relative_transform_target is not None
                and goal.tool_relative_transform_target is not None
                and not _relative_targets_are_compatible(
                    self._goal_buffer.tool_relative_transform_target,
                    goal.tool_relative_transform_target,
                )
            )
        ):
            update_reference = True
        elif self._solve_state != solve_state:
            new_links_goal_pose = self._get_padded_goalset_for_links(
                solve_state, self._solve_state, self._goal_buffer, goal.link_goal_poses
            )
            if new_links_goal_pose is not None:
                goal = goal.clone()
                goal.link_goal_poses = new_links_goal_pose
            else:
                update_reference = True

        if update_reference:
            self._solve_state = solve_state
            self._goal_buffer = goal.create_index_buffers(
                solve_state.batch_size,
                solve_state.multi_env,
                solve_state.num_seeds,
                self.device_cfg,
            )
            if goal.idxs_seed_goal_js is not None:
                self._goal_buffer.idxs_seed_goal_js = goal.idxs_seed_goal_js
            if goal.seed_enable_implicit_goal_js is not None:
                self._goal_buffer.seed_enable_implicit_goal_js = goal.seed_enable_implicit_goal_js
        else:
            self._goal_buffer.copy_(goal, update_idx_buffers=False)

        return self._goal_buffer, update_reference

    def update_batch_helper(self, batch_size: int) -> torch.Tensor:
        """Update and return the batch helper column tensor for indexing.

        Args:
            batch_size: The batch size to create the helper for.

        Returns:
            Column index helper of shape (batch_size, 1).
        """
        self._col = torch.arange(
            0, batch_size, device=self.device_cfg.device, dtype=torch.long
        ).view(-1, 1)
        return self._col

    def update_goal_tool_poses(self, goal_tool_poses: GoalToolPose) -> GoalRegistry:
        """Update the goal pose in the goal buffer."""
        if self._goal_buffer is None:
            log_and_raise("Goal buffer has not been initialized.")
        if self._solve_state is None:
            log_and_raise("Solve state has not been initialized.")

        if self._goal_buffer.link_goal_poses is not None:
            if set(goal_tool_poses.tool_frames) != set(
                self._goal_buffer.link_goal_poses.tool_frames
            ):
                log_and_raise("Goal link poses do not match existing goal buffer.")
            expected_pos_shape = self._goal_buffer.link_goal_poses.position.shape
            expected_quat_shape = self._goal_buffer.link_goal_poses.quaternion.shape
            if (
                goal_tool_poses.position.shape != expected_pos_shape
                or goal_tool_poses.quaternion.shape != expected_quat_shape
            ):
                log_and_raise("Shape of goal link poses does not match existing goal buffer.")

        new_goal_buffer, update_reference = self.update_goal_buffer(
            self._solve_state, goal_tool_poses=goal_tool_poses
        )
        if update_reference:
            log_and_raise("Goal link poses do not match existing goal buffer.")

        return new_goal_buffer

    def update_seed_goal_dt(self, dt: torch.Tensor) -> GoalRegistry:
        """Update only the per-seed trajectory dt in the current goal buffer.

        Args:
            dt: Per-seed trajectory timesteps with the same shape as
                ``goal_buffer.seed_goal_js.dt``.

        Returns:
            The existing goal buffer after updating its dt tensor in place.
        """
        if self._goal_buffer is None:
            log_and_raise("Goal buffer has not been initialized.")
        if self._goal_buffer.seed_goal_js is None:
            log_and_raise("seed_goal_js is None in goal buffer")
        goal_dt = self._goal_buffer.seed_goal_js.dt
        if goal_dt is None:
            log_and_raise("dt is None in seed_goal_js in goal buffer")
        if dt.shape != goal_dt.shape:
            log_and_raise(f"dt.shape: {dt.shape} != goal dt shape: {goal_dt.shape}")
        goal_dt.copy_(dt)
        return self._goal_buffer

    def update_tool_relative_transform_target(
        self,
        target: ToolRelativeTransformTarget,
    ) -> GoalRegistry:
        """Update relative-transform target values without reallocating the goal buffer."""
        if self._goal_buffer is None or self._solve_state is None:
            log_and_raise("Goal buffer has not been initialized.")
        if self._goal_buffer.tool_relative_transform_target is None:
            log_and_raise("Relative-transform target has not been initialized.")

        new_goal_buffer, update_reference = self.update_goal_buffer(
            self._solve_state,
            tool_relative_transform_target=target,
        )
        if update_reference:
            log_and_raise("Relative-transform target does not match existing goal buffer.")
        return new_goal_buffer

    def update_current_state(self, current_state: JointState) -> GoalRegistry:
        """Update the current state in the goal buffer."""
        if self._goal_buffer is None:
            log_and_raise("Goal buffer has not been initialized.")
        if self._solve_state is None:
            log_and_raise("Solve state has not been initialized.")
        if current_state.shape != self._goal_buffer.current_js.shape:
            log_and_raise("Current state does not match existing goal buffer.")

        new_goal_buffer, update_reference = self.update_goal_buffer(
            self._solve_state, current_js=current_state
        )
        if update_reference:
            log_and_raise("Current state does not match existing goal buffer.")

        return new_goal_buffer

    def update_goal_state(self, goal_state: JointState) -> GoalRegistry:
        """Update the goal state in the goal buffer."""
        if self._goal_buffer is None:
            log_and_raise("Goal buffer has not been initialized.")
        if self._solve_state is None:
            log_and_raise("Solve state has not been initialized.")
        if goal_state.shape != self._goal_buffer.goal_js.shape:
            log_and_raise("Goal state does not match existing goal buffer.")

        new_goal_buffer, update_reference = self.update_goal_buffer(
            self._solve_state, goal_js=goal_state
        )
        if update_reference:
            log_and_raise("Goal state does not match existing goal buffer.")

        return new_goal_buffer

    @property
    def goal_buffer(self) -> GoalRegistry:
        """Get the current goal buffer."""
        if self._goal_buffer is None:
            log_and_raise("Goal buffer has not been initialized.")
        return self._goal_buffer

    @property
    def solve_state(self) -> SolveState:
        """Get the current solve state."""
        if self._solve_state is None:
            log_and_raise("Solve state has not been initialized.")
        return self._solve_state

    @property
    def batch_helper(self) -> torch.Tensor:
        """Get the batch helper column tensor for indexing."""
        return self._col.clone()

    def get_batch_size(self) -> int:
        """Get the current batch size."""
        if self._solve_state is None:
            return 0
        return self._solve_state.batch_size

    def get_ik_batch_size(self) -> int:
        """Get the batch size for IK problems."""
        if self._solve_state is None:
            return 0
        return self._solve_state.get_ik_batch_size()

    def get_trajopt_batch_size(self) -> int:
        """Get the batch size for TrajOpt problems."""
        if self._solve_state is None:
            return 0
        return self._solve_state.get_trajopt_batch_size()

    @staticmethod
    def _get_padded_goalset_for_links(
        solve_state: SolveState,
        current_solve_state: SolveState,
        current_goal_buffer: GoalRegistry,
        links_goal_pose: Optional[GoalToolPose],
    ) -> Union[GoalToolPose, None]:
        """Pad goal poses to reuse a cached goal buffer when num_goalset shrinks.

        When the first call allocates a buffer for a large num_goalset (e.g. 50),
        subsequent calls with fewer goals (including num_goalset=1 for single-pose
        planning) reuse that buffer by copying the new goals into the first slots
        and filling the remainder with copies of the first goal.

        Returns None (forcing buffer reallocation) when the solve_type, batch size,
        seed count, or tool frames change, or when the new num_goalset exceeds the
        current buffer.

        Args:
            solve_state: New problem's solve state.
            current_solve_state: Cached solve state from previous call.
            current_goal_buffer: Cached goal buffer to pad into.
            links_goal_pose: New goal poses (may have fewer goals than buffer).

        Returns:
            Padded GoalToolPose matching the buffer shape, or None if reuse is not
            possible.
        """
        if links_goal_pose is None or current_goal_buffer.link_goal_poses is None:
            return None

        if solve_state.solve_type != current_solve_state.solve_type:
            return None

        if solve_state.batch_size != current_solve_state.batch_size:
            return None

        if solve_state.num_seeds != current_solve_state.num_seeds:
            return None

        if set(links_goal_pose.tool_frames) != set(
            current_goal_buffer.link_goal_poses.tool_frames
        ):
            return None

        cur_n = current_solve_state.num_goalset
        new_n = solve_state.num_goalset
        if new_n > cur_n:
            return None

        goal_pose = current_goal_buffer.link_goal_poses.clone()

        goal_pose_pos = goal_pose.position
        goal_pose_quat = goal_pose.quaternion

        new_goal_pos = links_goal_pose.position
        new_goal_quat = links_goal_pose.quaternion

        if new_goal_pos.shape[0] != goal_pose_pos.shape[0]:
            return None

        goal_pose_pos[:, :, :, :new_n, :] = new_goal_pos
        goal_pose_quat[:, :, :, :new_n, :] = new_goal_quat
        if new_n < cur_n:
            goal_pose_pos[:, :, :, new_n:, :] = new_goal_pos[:, :, :, :1, :]
            goal_pose_quat[:, :, :, new_n:, :] = new_goal_quat[:, :, :, :1, :]

        goal_pose.position = goal_pose_pos
        goal_pose.quaternion = goal_pose_quat

        return goal_pose
