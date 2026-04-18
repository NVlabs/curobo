# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import Dict, Optional

# Third Party
import torch
import torch.autograd.profiler as profiler

from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg

# CuRobo
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util.logging import log_and_raise
from curobo._src.util.tensor_util import (
    copy_or_clone,
    tensor_repeat_seeds,
)


@dataclass
class GoalRegistry:
    """Stores goal specifications and handles indexing for optimizing across seeds and batch.

    Current implementation assumes that num_goalset is the same for all link_goal_poses.
    """

    name: str = "goal"
    batch_size: int = -1
    num_goalset: int = 1  # NOTE: This currently does not get updated if goal_pose is updated later.
    num_seeds: int = 1

    #: Contains goal state for each seed
    goal_js: Optional[JointState] = None
    seed_goal_js: Optional[JointState] = None
    link_goal_poses: Optional[GoalToolPose] = None
    current_js: Optional[JointState] = None

    #: Time elapsed since the current robot state was observed. Shape: ``(n_problems,)``.
    #: Used by :class:`~curobo._src.cost.cost_cspace_position.PositionCSpaceCost` to tighten
    #: joint position bounds via velocity limits. Separate from the per-seed trajectory dt
    #: stored on ``seed_goal_js.dt``.
    current_state_dt: Optional[torch.Tensor] = None

    idxs_link_pose: Optional[torch.Tensor] = None  # shape: [batch]
    idxs_goal_js: Optional[torch.Tensor] = None
    idxs_current_js: Optional[torch.Tensor] = None  # shape: [batch]
    idxs_seed_goal_js: Optional[torch.Tensor] = None
    idxs_enable: Optional[torch.Tensor] = None  # shape: [batch, n]
    idxs_env: Optional[torch.Tensor] = None  # shape: [batch, n]

    seed_enable_implicit_goal_js: Optional[torch.Tensor] = None
    update_idxs_buffers: bool = True

    def __post_init__(self):
        self._update_batch_size()
        if self.link_goal_poses is not None:
            # Simplified validation - GoalToolPose ensures consistency
            self.batch_size = self.link_goal_poses.batch_size
            self.num_goalset = self.link_goal_poses.num_goalset

            if self.idxs_link_pose is None:
                self.idxs_link_pose = torch.arange(
                    0,
                    self.link_goal_poses.position.shape[0],
                    1,
                    device=self.link_goal_poses.position.device,
                    dtype=torch.int32,
                ).unsqueeze(-1)

        if self.current_js is not None:
            if self.idxs_current_js is None:
                self.idxs_current_js = torch.arange(
                    0,
                    self.current_js.position.shape[0],
                    1,
                    device=self.current_js.position.device,
                    dtype=torch.int32,
                ).unsqueeze(-1)
        if self.seed_goal_js is not None:
            if self.idxs_seed_goal_js is None:
                # batch_seed_goal_state_idx shape is [batch, num_seeds]
                if len(self.seed_goal_js.shape) != 3:  # batch, num_seeds, dof
                    log_and_raise(f"seed_goal_state shape: {self.seed_goal_js.shape} is not supported")

                self.idxs_seed_goal_js = torch.arange(
                    0,
                    self.seed_goal_js.shape[0] * self.seed_goal_js.shape[1],
                    1,
                    device=self.seed_goal_js.position.device,
                    dtype=torch.int32,
                ).unsqueeze(-1)
            if self.seed_enable_implicit_goal_js is None:
                self.seed_enable_implicit_goal_js = torch.zeros(
                    (self.seed_goal_js.shape[0], self.seed_goal_js.shape[1]),
                    device=self.seed_goal_js.position.device,
                    dtype=torch.uint8,
                )

    def _update_batch_size(self):
        if self.link_goal_poses is not None:
            self.batch_size = self.link_goal_poses.batch_size
        elif self.goal_js is not None:
            self.batch_size = self.goal_js.position.shape[0]

    @property
    def link_goal_pose_dict(self) -> Optional[Dict[str, Pose]]:
        """Convert to dict format for backward compatibility in cost computations."""
        if self.link_goal_poses is None:
            return None
        return self.link_goal_poses.to_dict()

    @profiler.record_function("GoalRegistry/repeat_seeds")
    def repeat_seeds(self, num_seeds: int, repeat_seed_idx_buffers: bool = False):
        """Repeat seeds across batch and seeds.

        Args:
            num_seeds (int): Number of seeds to repeat across
            repeat_seed_idx_buffers (bool, optional): Whether to repeat the seed index buffers.
                Defaults to False.

        Returns:
            GoalRegistry: A new GoalRegistry object with the seeds repeated across batch and seeds.
        """
        # across seeds, the data is the same, so could we just expand batch_idx
        goal_js = current_js = links_goal_pose = None
        seed_goal_js = None

        idxs_enable = idxs_link_pose = idxs_env = idxs_current_js = None
        idxs_goal_js = None
        idxs_seed_goal_js = None
        seed_enable_implicit_goal_js = None
        batch_size = None
        current_state_dt = None
        if self.link_goal_poses is not None:
            links_goal_pose = self.link_goal_poses
            batch_size = self.link_goal_poses.batch_size
        if self.goal_js is not None:
            goal_js = self.goal_js.clone()
            if batch_size is None:
                batch_size = self.goal_js.position.shape[0]
        if self.current_js is not None:
            current_js = self.current_js.clone()
        if self.current_state_dt is not None:
            current_state_dt = self.current_state_dt.clone()

        if self.seed_goal_js is not None:
            seed_goal_js = self.seed_goal_js.clone()

        # repeat seeds for indexing:
        if self.idxs_link_pose is not None:
            idxs_link_pose = tensor_repeat_seeds(self.idxs_link_pose, num_seeds)
        if self.idxs_goal_js is not None:
            idxs_goal_js = tensor_repeat_seeds(self.idxs_goal_js, num_seeds)
        if self.idxs_enable is not None:
            idxs_enable = tensor_repeat_seeds(self.idxs_enable, num_seeds)
        if self.idxs_env is not None:
            idxs_env = tensor_repeat_seeds(self.idxs_env, num_seeds)
        if self.idxs_current_js is not None:
            idxs_current_js = tensor_repeat_seeds(self.idxs_current_js, num_seeds)
        if self.idxs_seed_goal_js is not None:
            idxs_seed_goal_js = self.idxs_seed_goal_js
            if repeat_seed_idx_buffers:
                idxs_seed_goal_js = tensor_repeat_seeds(self.idxs_seed_goal_js, num_seeds)
            if (
                idxs_seed_goal_js.shape[0]
                > num_seeds * self.seed_goal_js.shape[0] * self.seed_goal_js.shape[1]
            ):
                log_and_raise(
                    "idxs_seed_goal_js shape: "
                    + str(idxs_seed_goal_js.shape)
                    + " does not match number of seeds: "
                    + str(num_seeds * self.seed_goal_js.shape[0] * self.seed_goal_js.shape[1])
                )
        if self.seed_enable_implicit_goal_js is not None:
            seed_enable_implicit_goal_js = self.seed_enable_implicit_goal_js.clone()
            if idxs_seed_goal_js is None:
                log_and_raise("seed_goal_js is None")
            if seed_goal_js.shape[0:2] != seed_enable_implicit_goal_js.shape:
                log_and_raise(
                    f"seed_goal_js.shape {seed_goal_js.shape} "
                    + f"!= seed_enable_implicit_goal_js.shape {seed_enable_implicit_goal_js.shape}"
                )

        num_seeds = self.num_seeds * num_seeds
        batch_size = self.batch_size
        return GoalRegistry(
            goal_js=goal_js,
            seed_goal_js=seed_goal_js,
            current_js=current_js,
            current_state_dt=current_state_dt,
            idxs_link_pose=idxs_link_pose,
            idxs_env=idxs_env,
            idxs_enable=idxs_enable,
            idxs_current_js=idxs_current_js,
            idxs_goal_js=idxs_goal_js,
            idxs_seed_goal_js=idxs_seed_goal_js,
            link_goal_poses=links_goal_pose,
            seed_enable_implicit_goal_js=seed_enable_implicit_goal_js,
            num_seeds=num_seeds,
            batch_size=batch_size,
        )

    def clone(self):
        return GoalRegistry(
            goal_js=self.goal_js if self.goal_js is not None else None,
            current_js=self.current_js if self.current_js is not None else None,
            current_state_dt=(
                self.current_state_dt.clone() if self.current_state_dt is not None else None
            ),
            idxs_link_pose=self.idxs_link_pose if self.idxs_link_pose is not None else None,
            idxs_env=self.idxs_env if self.idxs_env is not None else None,
            idxs_enable=self.idxs_enable if self.idxs_enable is not None else None,
            idxs_current_js=self.idxs_current_js if self.idxs_current_js is not None else None,
            idxs_goal_js=self.idxs_goal_js if self.idxs_goal_js is not None else None,
            link_goal_poses=(
                self.link_goal_poses.clone() if self.link_goal_poses is not None else None
            ),
            num_goalset=self.num_goalset,
            idxs_seed_goal_js=(
                self.idxs_seed_goal_js if self.idxs_seed_goal_js is not None else None
            ),
            seed_goal_js=self.seed_goal_js if self.seed_goal_js is not None else None,
            seed_enable_implicit_goal_js=(
                self.seed_enable_implicit_goal_js
                if self.seed_enable_implicit_goal_js is not None
                else None
            ),
            num_seeds=self.num_seeds,
            batch_size=self.batch_size,
        )

    def apply_kernel(self, kernel_mat):
        # For each seed in optimization, we use kernel_mat to transform to many parallel goals
        # This can be modified to just multiply self.batch and update self.batch by the shape of
        # self.batch
        # NOTE: seed variables are not implemented
        goal_state = current_state = links_goal_pose = None
        batch_enable_idx = batch_pose_idx = env_idx = batch_current_state_idx = None
        batch_goal_state_idx = None
        if self.link_goal_poses is not None:
            links_goal_pose = self.link_goal_poses
        if self.goal_js is not None:
            goal_state = self.goal_js  # .apply_kernel(kernel_mat)
        if self.current_js is not None:
            current_state = self.current_js  # .apply_kernel(kernel_mat)
        if self.idxs_enable is not None:
            batch_enable_idx = kernel_mat @ self.idxs_enable
        if self.idxs_goal_js is not None:
            batch_goal_state_idx = (kernel_mat @ self.idxs_goal_js.to(dtype=torch.float32)).to(
                dtype=torch.int32
            )

        if self.idxs_current_js is not None:
            batch_current_state_idx = (
                kernel_mat @ self.idxs_current_js.to(dtype=torch.float32)
            ).to(dtype=torch.int32)
        if self.idxs_link_pose is not None:
            batch_pose_idx = (kernel_mat @ self.idxs_link_pose.to(dtype=torch.float32)).to(
                dtype=torch.int32
            )
        if self.idxs_env is not None:
            env_idx = (kernel_mat @ self.idxs_env.to(dtype=torch.float32)).to(
                dtype=torch.int32
            )

        return GoalRegistry(
            goal_js=goal_state,
            link_goal_poses=links_goal_pose,
            current_js=current_state,
            current_state_dt=self.current_state_dt,
            idxs_link_pose=batch_pose_idx,
            idxs_enable=batch_enable_idx,
            idxs_env=env_idx,
            idxs_current_js=batch_current_state_idx,
            idxs_goal_js=batch_goal_state_idx,
        )

    @profiler.record_function("GoalRegistry/copy_")
    def copy_(self, goal: GoalRegistry, update_idx_buffers: bool = True, allow_clone: bool = True):
        """Copy data from another goal object.

        Args:
            goal: Source ``GoalRegistry`` whose joint states, tool poses, index buffers, and
                related fields are merged or copied into this registry.

        Returns:
            ``None``. Updates this instance in place.
        """
        if allow_clone:
            if self.goal_js is None and goal.goal_js is not None:
                self.goal_js = goal.goal_js
            if self.current_js is None and goal.current_js is not None:
                self.current_js = goal.current_js
            if self.seed_goal_js is None and goal.seed_goal_js is not None:
                self.seed_goal_js = goal.seed_goal_js
            if self.current_state_dt is None and goal.current_state_dt is not None:
                self.current_state_dt = goal.current_state_dt

        if goal.goal_js is not None and self.goal_js is not None:
            self.goal_js.copy_(goal.goal_js, allow_clone=allow_clone)
        if goal.current_js is not None and self.current_js is not None:
            self.current_js.copy_(goal.current_js, allow_clone=allow_clone)
        if goal.seed_goal_js is not None and self.seed_goal_js is not None:
            self.seed_goal_js.copy_(goal.seed_goal_js, allow_clone=allow_clone)
        if goal.current_state_dt is not None:
            if self.current_state_dt is not None:
                self.current_state_dt = copy_or_clone(
                    goal.current_state_dt, self.current_state_dt, allow_clone=allow_clone
                )
            elif allow_clone:
                self.current_state_dt = goal.current_state_dt.clone()
            else:
                self.current_state_dt = goal.current_state_dt

        if goal.link_goal_poses is not None:
            if self.link_goal_poses is None:
                self.link_goal_poses = goal.link_goal_poses
            else:
                self.link_goal_poses.copy_(goal.link_goal_poses)

        self._update_batch_size()

        if (
            self.seed_enable_implicit_goal_js is not None
            and goal.seed_enable_implicit_goal_js is not None
        ):
            self.seed_enable_implicit_goal_js = copy_or_clone(
                goal.seed_enable_implicit_goal_js,
                self.seed_enable_implicit_goal_js,
                allow_clone=allow_clone,
            )
        # copy pose indices as well?
        if goal.update_idxs_buffers and update_idx_buffers:
            self.idxs_link_pose = copy_or_clone(
                goal.idxs_link_pose, self.idxs_link_pose, allow_clone=allow_clone
            )
            self.idxs_env = copy_or_clone(
                goal.idxs_env, self.idxs_env, allow_clone=allow_clone
            )
            self.idxs_current_js = copy_or_clone(
                goal.idxs_current_js, self.idxs_current_js, allow_clone=allow_clone
            )
            self.idxs_seed_goal_js = copy_or_clone(
                goal.idxs_seed_goal_js, self.idxs_seed_goal_js, allow_clone=allow_clone
            )
            self.idxs_enable = copy_or_clone(
                goal.idxs_enable, self.idxs_enable, allow_clone=allow_clone
            )
            self.idxs_goal_js = copy_or_clone(
                goal.idxs_goal_js, self.idxs_goal_js, allow_clone=allow_clone
            )

    def get_batch_goal_state(self):
        return self.goal_js[self.idxs_link_pose[:, 0]]

    def create_index_buffers(
        self,
        batch_size: int,
        multi_env: bool,
        num_seeds: int,
        device_cfg: DeviceCfg,
    ):
        new_goal = GoalRegistry.create_idx(
            batch_size, multi_env, num_seeds, device_cfg
        )
        new_goal.copy_(self, update_idx_buffers=False)
        return new_goal

    @classmethod
    def create_idx(
        cls,
        pose_batch_size: int,
        multi_env: bool,
        num_seeds: int,
        device_cfg: DeviceCfg,
        seed_goal_state: Optional[JointState] = None,
        repeat_seed_idx_buffers: bool = False,
    ):
        batch_pose_idx = torch.arange(
            0, pose_batch_size, 1, device=device_cfg.device, dtype=torch.int32
        ).unsqueeze(-1)
        if multi_env:
            env_idx = batch_pose_idx.clone()
        else:
            env_idx = 0 * batch_pose_idx
        batch_currernt_state_idx = batch_pose_idx.clone()
        batch_goal_state_idx = batch_pose_idx.clone()

        g = GoalRegistry(
            idxs_link_pose=batch_pose_idx,
            idxs_env=env_idx,
            idxs_current_js=batch_currernt_state_idx,
            idxs_goal_js=batch_goal_state_idx,
            seed_goal_js=seed_goal_state,
        )
        g_seeds = g.repeat_seeds(num_seeds, repeat_seed_idx_buffers)
        return g_seeds

    def get_index_size(self):
        if self.idxs_link_pose is not None:
            return self.idxs_link_pose.shape[0]
        elif self.idxs_goal_js is not None:
            return self.idxs_goal_js.shape[0]
        else:
            return None
