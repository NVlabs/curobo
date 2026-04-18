# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

# Standard Library
from typing import TYPE_CHECKING, Optional

# Third Party
import torch

from curobo._src.cost.cost_base import BaseCost

# CuRobo
from curobo._src.curobolib.cuda_ops.geometry import SelfCollisionDistance
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    # CuRobo
    from curobo._src.cost.cost_self_collision_cfg import SelfCollisionCostCfg


class SelfCollisionCost(BaseCost):
    def __init__(self, config: SelfCollisionCostCfg):
        if config.self_collision_kin_config is None:
            log_and_raise("SelfCollisionCostConfig must contain self_collision_kin_config")
        super().__init__(config)

        self._batch_size = None

    def setup_batch_tensors(self, batch_size: int, horizon: int):
        # Assuming n stays constant

        num_spheres = self.config.self_collision_kin_config.num_spheres
        sphere_dim = 4

        if self._batch_size is None or self._batch_size != batch_size or self._horizon != horizon:
            # b, h, n, k = robot_spheres.shape
            self._out_distance = torch.zeros(
                (batch_size, horizon, 1),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )
            self._out_grad = torch.zeros(
                (batch_size, horizon, num_spheres, sphere_dim),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )

            # Sparse index stores index of gradients that are non-zero.
            self._sparse_sphere_idx = torch.zeros(
                (batch_size, horizon, num_spheres),
                device=self.device_cfg.device,
                dtype=torch.uint8,
            )
            if self.config.store_pair_distance:
                num_collision_pairs = self.config.self_collision_kin_config.collision_pairs.shape[0]
                self._pair_distance = torch.zeros(
                    (batch_size, horizon, num_collision_pairs),
                    device=self.device_cfg.device,
                    dtype=self.device_cfg.dtype,
                )
            else:
                self._pair_distance = torch.zeros(
                    (1),
                    device=self.device_cfg.device,
                    dtype=self.device_cfg.dtype,
                )

            # calculate block parallelization parameters:

            self._block_batch_max_index = torch.full(
                (
                    batch_size,
                    horizon,
                    self.config.self_collision_kin_config.num_blocks_per_batch,
                    2,
                ),
                fill_value=0,
                device=self.device_cfg.device,
                dtype=torch.int16,
            )
            self._block_batch_max_value = torch.full(
                (batch_size, horizon, self.config.self_collision_kin_config.num_blocks_per_batch),
                fill_value=0,
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )
            super().setup_batch_tensors(batch_size, horizon)

    def forward(
        self,
        robot_spheres: torch.Tensor,
    ):
        """Compute the self collision cost.

        This will return the largest penetration distance between spheres across the batch and
        horizon. The sphere pairs are chosen using
        self.config.self_collision_kin_config (SelfCollisionKinematicsCfg).

        Args:
            robot_spheres: The robot spheres. This is of shape (batch, horizon, num_spheres, 4),
            with last dimension being (position_x, position_y, position_z, radius) in meters.

        Returns:
            The self collision cost.
        """
        self.validate_input(robot_spheres)

        dist = SelfCollisionDistance.apply(
            robot_spheres,
            self._out_distance,
            self._out_grad,
            self._pair_distance,
            self._sparse_sphere_idx,
            self._weight,
            self.config.self_collision_kin_config.sphere_padding,
            self.config.self_collision_kin_config.collision_pairs,
            self._block_batch_max_value,
            self._block_batch_max_index,
            self.config.self_collision_kin_config.num_blocks_per_batch,
            self.config.self_collision_kin_config.max_threads_per_block,
            self.config.store_pair_distance,
            self.config.use_grad_input,
        )
        if self.config.convert_to_binary:
            dist = torch.clamp(dist, max=1.0)
            dist = torch.where(dist > 0, dist + 1.0, dist)

        return dist

    def reset(
        self,
        reset_problem_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):


        super().reset(
            reset_problem_ids=reset_problem_ids,
            **kwargs,
        )

    def validate_input(
        self, robot_spheres: torch.Tensor
    ):
        if robot_spheres.ndim != 4:
            log_and_raise("robot_spheres must be of shape (batch, horizon, num_spheres, 4)")
        if robot_spheres.shape[3] != 4:
            log_and_raise("robot_spheres must have 4 dimensions (x, y, z, radius)")
        if robot_spheres.shape[1] != self._horizon:
            log_and_raise("robot_spheres.shape[1] must be equal to self._horizon")
        if robot_spheres.shape[2] != self.config.self_collision_kin_config.num_spheres:
            log_and_raise(
                "robot_spheres.shape[2] must be equal to self.config.self_collision_kin_config.num_spheres"
            )
        if robot_spheres.shape[0] != self._batch_size:
            log_and_raise("robot_spheres.shape[0] must be equal to self._batch_size")
        if self.config.store_pair_distance:
            if self._pair_distance.ndim != 3:
                log_and_raise(
                    "self._pair_distance must be of shape (batch, horizon, num_collision_pairs)"
                )
            if self._pair_distance.shape[0] != self._batch_size:
                log_and_raise("self._pair_distance.shape[0] must be equal to self._batch_size")
            if self._pair_distance.shape[1] != self._horizon:
                log_and_raise("self._pair_distance.shape[1] must be equal to self._horizon")
            if (
                self._pair_distance.shape[2]
                != self.config.self_collision_kin_config.collision_pairs.shape[0]
            ):
                log_and_raise(
                    "self._pair_distance.shape[2] must be equal to self.config.self_collision_kin_config.collision_pairs.shape[0]"
                )
