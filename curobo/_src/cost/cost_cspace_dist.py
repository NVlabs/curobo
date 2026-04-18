# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

# Standard Library
from typing import TYPE_CHECKING, Tuple

# Third Party
import torch

# CuRobo
from curobo._src.cost.cost_base import BaseCost
from curobo._src.cost.wp_torch_cspace_dist import L2DistFunction
from curobo._src.util.logging import log_and_raise
from curobo._src.util.torch_util import get_torch_jit_decorator
from curobo._src.util.warp import init_warp

if TYPE_CHECKING:
    # CuRobo
    from curobo._src.cost.cost_cspace_dist_cfg import CSpaceDistCostCfg


# create a bound cost tensor:
class CSpaceDistCost(BaseCost):
    def __init__(self, config: CSpaceDistCostCfg):
        if config.dof == -1:
            log_and_raise("dof should be set before initializing the cost class")
        super().__init__(config)
        init_warp()

    def setup_batch_tensors(self, batch, horizon):
        if self._batch_size != batch or self._horizon != horizon:
            self._out_cv_buffer = torch.zeros(
                (batch, horizon, self.config.dof),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )

            self._out_g_buffer = torch.zeros(
                (batch, horizon, self.config.dof),
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )

            self._batch_size = batch
            self._horizon = horizon

    def validate_input(
        self,
        current_vec: torch.Tensor,
        goal_vec: torch.Tensor,
        idxs_goal: torch.Tensor,
    ):
        """Validate the input to the cost function. If this raises an error, call
        setup_batch_tensors with the correct batch size and horizon. If dof is the error,
        initialize config with the correct dof.

        Args:
            current_vec: The current vector. Shape is (batch, horizon, dof).
            goal_vec: The goal vector. Shape is (-1, dof). The first dimension can have any size as
            the index is accessed via the `idxs_goal` tensor. i.e., goal for batch element 0 is
            goal_vec[idxs_goal[0]].
            idxs_goal: The indices of the goal vector. Shape is (batch).

        Raises:
            ValueError: If the input is not valid.
        """
        if current_vec.ndim != 3:
            log_and_raise("current_vec should have 3 dimensions")
        if goal_vec.ndim != 2:
            log_and_raise("goal_vec should have 2 dimensions")
        if idxs_goal.ndim != 1:
            log_and_raise(f"idxs_goal should have 1 dimension, but got {idxs_goal.ndim}, shape {idxs_goal.shape}")
        current_vec_shape = current_vec.shape
        goal_vec_shape = goal_vec.shape
        idxs_goal_shape = idxs_goal.shape

        if current_vec_shape[0] != self._batch_size:
            log_and_raise(
                f"current_vec should have the same batch size as the cost \
                    class: {current_vec_shape[0]} != {self._batch_size} \
                    Call setup_batch_tensors with the correct batch size"
            )
        if current_vec_shape[1] != self._horizon:
            log_and_raise(
                f"current_vec should have the same horizon as the cost \
                    class: {current_vec_shape[1]} != {self._horizon} \
                    Call setup_batch_tensors with the correct horizon"
            )
        if current_vec_shape[2] != self.config.dof:
            log_and_raise(
                f"current_vec should have the same dof as the cost \
                    class: {current_vec_shape[2]} != {self.config.dof} \
                    initialize config with the correct dof"
            )
        if goal_vec_shape[1] != self.config.dof:
            log_and_raise(
                f"goal_vec should have the same dof as the cost \
                    class: {goal_vec_shape[1]} != {self.config.dof} \
                    initialize config with the correct dof"
            )
        if idxs_goal_shape[0] != self._batch_size:
            log_and_raise(
                f"idxs_goal should have the same batch size as the cost \
                    class: {idxs_goal_shape[0]} != {self._batch_size} \
                    Call setup_batch_tensors with the correct batch size"
            )

    def forward(
        self,
        current_vec: torch.Tensor,
        goal_vec: torch.Tensor,
        idxs_goal: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the distance cost for a given input.

        Args:
            current_vec: The current vector. Shape is (batch, horizon, dof).
            goal_vec: The goal vector. Shape is (-1, dof). The first dimension can have any size as
            the index is accessed via the `idxs_goal` tensor. i.e., goal for batch element 0 is
            goal_vec[idxs_goal[0]].

            idxs_goal: The indices of the goal vector. Shape is (batch).

        Returns:
            cost: The distance cost. Shape is (batch, horizon).
        """
        self.validate_input(current_vec, goal_vec, idxs_goal)

        cost = L2DistFunction.apply(
            current_vec,
            goal_vec,
            idxs_goal,
            self._weight,
            self.config.terminal_dof_weight,
            self.config.non_terminal_dof_weight,
            self._out_cv_buffer,
            self._out_g_buffer,
            self.config.use_grad_input,
        )

        return cost

    def forward_out_distance(
        self,
        current_vec: torch.Tensor,
        goal_vec: torch.Tensor,
        idxs_goal: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the distance cost and distance for a given input.

        Args:
            goal_vec: The goal vector. Shape is (-1, dof). The first dimension can have any size as
            the index is accessed via the `idxs_goal` tensor. i.e., goal for batch element 0 is
            goal_vec[idxs_goal[0]].
            current_vec: The current vector. Shape is (batch, horizon, dof).
            idxs_goal: The indices of the goal vector. Shape is (batch).

        Returns:
            cost: The distance cost. Shape is (batch, horizon).
            distance: The distance. Shape is (batch, horizon).
        """
        cost = self.forward(current_vec, goal_vec, idxs_goal)
        distance = self.jit_squared_cost_to_l2(cost, self._weight, self._run_weight_vec)
        return cost, distance

    @staticmethod
    @get_torch_jit_decorator(only_valid_for_compile=True, dynamic=True)
    def jit_squared_cost_to_l2(cost, weight, run_weight_vec) -> torch.Tensor:
        """Convert the squared cost to a distance by taking the square root of the cost and
        dividing by the weight.

        Args:
            cost: The cost. Shape is (batch, horizon).
            weight: The weight. Shape is (batch, horizon).
            run_weight_vec: The run weight vector. Shape is (batch, horizon).

        Returns:
            distance: The distance. Shape is (batch, horizon).
        """
        weight_inv = weight * run_weight_vec
        weight_inv = 1.0 / weight_inv
        weight_inv = torch.nan_to_num(weight_inv, 0.0)
        distance = torch.sqrt(cost * weight_inv)
        return distance
