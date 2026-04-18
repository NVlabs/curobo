# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

# Standard Library
from typing import TYPE_CHECKING, Optional

# Third Party
import torch

# CuRobo
from curobo._src.cost.cost_base import BaseCost
from curobo._src.geom.collision import CollisionBuffer
from curobo._src.util.logging import log_and_raise, log_info
from curobo._src.util.torch_util import get_torch_jit_decorator

if TYPE_CHECKING:
    # CuRobo
    from curobo._src.cost.cost_scene_collision_cfg import SceneCollisionCostCfg
from curobo._src.robot.kinematics.kinematics import KinematicsState


class SceneCollisionCost(BaseCost):
    def __init__(self, config: SceneCollisionCostCfg):
        """Creates a world collision cost instance. This only works on CUDA devices.

        See note on :ref:`collision_checking_note` for details on the cost formulation.

        Args:
            config: configuration for the world collision cost.
        """
        super().__init__(config)
        if self.config.scene_collision_checker is None:
            log_and_raise("scene_collision_checker must be set before using world collision cost")
        self._collision_buffer: Optional[CollisionBuffer] = None

    def setup_batch_tensors(self, batch_size: int, horizon: int):
        self._batch_size = batch_size
        self._horizon = horizon

        robot_spheres_shape = (batch_size, horizon, self.config.num_spheres, 4)
        self._collision_buffer = CollisionBuffer.from_shape(
            robot_spheres_shape,
            self.device_cfg,
        )

    def update_num_spheres(
        self, num_spheres: int, batch_size: Optional[int] = None, horizon: Optional[int] = None
    ):
        self.config.update_num_spheres(num_spheres)
        if batch_size is None:
            batch_size = self._batch_size
        if horizon is None:
            horizon = self._horizon
        self.setup_batch_tensors(batch_size, horizon)

    def forward(
        self,
        state: KinematicsState,
        idxs_env_query: Optional[torch.Tensor] = None,
        trajectory_dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the world collision cost.

        Args:
            robot_spheres_in: The robot spheres. This is of shape (batch, horizon, num_spheres, 4),
            with last dimension being (position_x, position_y, position_z, radius) in meters.
            env_query_idx: The environment query index.
            trajectory_dt: The trajectory dt. This is of shape (batch,).

        Returns:
            The world collision cost. This is of shape (batch, horizon).
        """
        self.validate_input(
            state.robot_spheres,
            idxs_env_query,
            trajectory_dt,
        )

        if self.config.use_sweep:
            return self._sweep_fn(
                state,
                idxs_env_query,
                trajectory_dt,
            )
        else:
            return self._discrete_fn(
                state,
                idxs_env_query,
            )

    def validate_input(
        self,
        robot_spheres_in: torch.Tensor,
        idxs_env_query: Optional[torch.Tensor] = None,
        trajectory_dt: Optional[torch.Tensor] = None,
    ):
        if robot_spheres_in.shape != (self._batch_size, self._horizon, self.config.num_spheres, 4):
            log_and_raise(
                "robot_spheres_in.shape must be equal to "
                + "(batch_size, horizon, num_spheres, 4)"
                + "robot_spheres_in.shape: "
                + str(robot_spheres_in.shape)
                + "batch_size: "
                + str(self._batch_size)
                + "horizon: "
                + str(self._horizon)
                + "num_spheres: "
                + str(self.config.num_spheres)
                + "Call update_num_spheres() to set the number of spheres or "
                + "setup_batch_tensors() to set the batch size and horizon"
            )

        if idxs_env_query is not None:
            if idxs_env_query.shape != (self._batch_size,):
                log_and_raise("env_query_idx.shape must be equal to (batch_size,)")
        if self.config.use_sweep:
            if trajectory_dt is None:
                log_and_raise("trajectory_dt must be set if use_sweep is True")
            if trajectory_dt.shape != (self._batch_size,):
                log_and_raise("trajectory_dt.shape must be equal to (batch_size,)")

    def _sweep_fn(
        self,
        state: KinematicsState,
        env_query_idx: Optional[torch.Tensor] = None,
        trajectory_dt: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the world collision cost for swept spheres.

        Args:
            robot_spheres_in: The robot spheres. This is of shape (batch, horizon, num_spheres, 4),
            with last dimension being (position_x, position_y, position_z, radius) in meters.
            env_query_idx: The environment query index. This is only required if there are multiple
            world environments in the world collision checker.

        Returns:
            The world collision cost. This is of shape (batch, horizon).
        """
        if not self.config.sum_distance:
            log_info("sum_distance=False will be slower than sum_distance=True")
            self.use_grad_input = True

        if self.config.convert_to_binary:
            assert False, "Not implemented"

        dist = self.config.scene_collision_checker.get_swept_sphere_distance(
            state,
            self._collision_buffer,
            self._weight,
            activation_distance=self.config.activation_distance,
            trajectory_dt=trajectory_dt,
            enable_speed_metric=self.config.use_speed_metric,
            env_query_idx=env_query_idx,
            return_loss=self.config.use_grad_input,
        )
        return dist

    def _discrete_fn(
        self,
        state: KinematicsState,
        env_query_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the world collision cost for discrete spheres.

        Args:
            robot_spheres_in: The robot spheres. This is of shape (batch, horizon, num_spheres, 4),
            with last dimension being (position_x, position_y, position_z, radius) in meters.
            env_query_idx: The environment query index. This is only required if there are multiple
            world environments in the world collision checker.

        Returns:
            The world collision cost. This is of shape (batch, horizon).
        """
        if not self.config.sum_distance:
            log_info("sum_distance=False will be slower than sum_distance=True")
            self.use_grad_input = True

        if self.config.convert_to_binary:
            dist = self.config.scene_collision_checker.get_sphere_collision(
                state,
                self._collision_buffer,
                self._weight,
                activation_distance=self.config.activation_distance,
                env_query_idx=env_query_idx,
                return_loss=self.config.use_grad_input,
            )
        else:
            dist = self.config.scene_collision_checker.get_sphere_distance(
                state,
                self._collision_buffer,
                self._weight,
                activation_distance=self.config.activation_distance,
                env_query_idx=env_query_idx,
                return_loss=self.config.use_grad_input,
            )
        return dist

    def get_gradient_buffer(self) -> torch.Tensor:
        """Get the gradient buffer for the world collision cost.

        Returns:
            The gradient buffer. This is of shape (batch, horizon, num_spheres, 4).
        """
        return self._collision_buffer.gradient

    @staticmethod
    @get_torch_jit_decorator()
    def jit_weight_distance(dist: torch.Tensor, sum_cost: bool) -> torch.Tensor:
        """Weight the distance cost.

        Args:
            dist: The distance.
            sum_cost: Whether to sum the cost.

        Returns:
            The weighted distance cost.
        """
        if sum_cost:
            dist = torch.sum(dist, dim=-1)
        else:
            dist = torch.max(dist, dim=-1)[0]
        return dist

    @staticmethod
    @get_torch_jit_decorator()
    def jit_weight_collision(dist: torch.Tensor, sum_cost: bool) -> torch.Tensor:
        """Weight the collision cost.

        Args:
            dist: The distance.
            sum_cost: Whether to sum the cost.
        """
        if sum_cost:
            dist = torch.sum(dist, dim=-1)
        else:
            dist = torch.max(dist, dim=-1)[0]

        dist = torch.where(dist > 0, dist + 1.0, dist)
        return dist
