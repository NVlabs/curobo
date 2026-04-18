# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from curobo._src.cost.cost_base_cfg import BaseCostCfg
from curobo._src.cost.cost_cspace_position import PositionCSpaceCost
from curobo._src.cost.cost_cspace_state import StateCSpaceCost
from curobo._src.cost.cost_cspace_type import CSpaceCostType
from curobo._src.curobolib.cuda_ops.tensor_checks import check_float16_tensors, check_float32_tensors

# CuRobo
from curobo._src.robot.types.joint_limits import JointLimits
from curobo._src.transition.robot_state_transition import RobotStateTransition
from curobo._src.util.logging import log_and_raise


@dataclass
class CSpaceCostCfg(BaseCostCfg):
    #: Number of degrees of freedom. This should before initializing the cost class.
    dof: int = 0

    cost_type: Optional[CSpaceCostType] = None
    """ Type of bound cost to compute.

    For CSpaceCostType.POSITION, the cost is only computed for joint position and torque. In addition,
    a cost is computed between the current joint position and the target joint position.

    For CSpaceCostType.STATE, the cost is computed for joint position, velocity, acceleration,
    jerk and torque.



    """
    joint_limits: Optional[JointLimits] = None
    """ Joint limits for the bound cost. This can be initialized from a transition model using
    `initialize_from_transition_model`.
    """

    squared_l2_regularization_weight: Optional[List[float]] = None
    """Squared L2 regularization weights for joint state derivatives.

    For CSpaceCostType.POSITION, shape is (2,) for [velocity, acceleration].  These penalise
    the *implied* velocity ``(q - q_prev) / dt`` and *implied* acceleration
    ``(v_implied - v_prev) / dt`` computed from :attr:`current_joint_state`.
    Active only when ``current_joint_state`` carries ``dt > 0`` and velocity.

    For CSpaceCostType.STATE, shape is (5,) for [velocity, acceleration, jerk, torque, energy].
    """

    retime_weights: bool = False
    """If true, the weights are retimed based on current dt."""

    retime_regularization_weights: bool = False
    """If true, the smooth weights are retimed to be used for the bounds smooth cost."""

    activation_distance: Union[torch.Tensor, float] = 0.0
    """Activation distance is the distance at which the cost is activated.

    The distance here is a multiplier on the range of the joint limits. E.g., if the joint limit
    for position is [-1, 1], and the activation distance is 0.1, then the cost will be activated
    at 10% of the joint limit range, i.e., at 0.1.

    For BoundCostType.POSITION, should be of shape (2,). One for position and one for torque.

    For BoundCostType.BOUNDS and BoundCostType.BOUNDS_SMOOTH, this is the activation distance for
    position, velocity, acceleration, jerk, torque terms. This should be of shape (5,).
    """

    cspace_target_weight: Optional[torch.Tensor] = None
    """Weight for the cspace target cost term."""

    cspace_non_terminal_weight_factor: Optional[torch.Tensor] = None
    """Factor to scale the cspace non-terminal weight. This is a multiplier on the cspace target weight."""

    cspace_target_dof_weight: Optional[torch.Tensor] = None
    """Per dof weight for the cspace target cost term."""

    def set_bounds(self, bounds: JointLimits, teleport_mode: bool = False):
        self.joint_limits = bounds.clone()

        if teleport_mode:
            if self.cost_type == CSpaceCostType.STATE:
                self.cost_type = CSpaceCostType.POSITION
                self.weight = self.weight[[0, 4]]
                self.activation_distance = self.activation_distance[[0, 4]]
                sliced_l2 = (
                    self.squared_l2_regularization_weight is not None
                    and len(self.squared_l2_regularization_weight) == 5
                )
                if sliced_l2:
                    self.squared_l2_regularization_weight = (
                        self.squared_l2_regularization_weight[:2]
                    )
                check_fn = (
                    check_float16_tensors
                    if self.weight.dtype == torch.float16
                    else check_float32_tensors
                )
                if sliced_l2:
                    check_fn(
                        self.weight.device,
                        weight=self.weight,
                        activation_distance=self.activation_distance,
                        squared_l2_regularization_weight=self.squared_l2_regularization_weight,
                    )
                else:
                    check_fn(
                        self.weight.device,
                        weight=self.weight,
                        activation_distance=self.activation_distance,
                    )

                self.class_type = PositionCSpaceCost

        if self.cost_type != CSpaceCostType.POSITION:
            if torch.max(self.joint_limits.velocity[1] - self.joint_limits.velocity[0]) == 0.0:
                log_and_raise("Joint velocity limits is zero")
            if (
                torch.max(self.joint_limits.acceleration[1] - self.joint_limits.acceleration[0])
                == 0.0
            ):
                log_and_raise("Joint acceleration limits is zero")
            if torch.max(self.joint_limits.jerk[1] - self.joint_limits.jerk[0]) == 0.0:
                log_and_raise("Joint jerk limits is zero")

    def __post_init__(self):
        if isinstance(self.activation_distance, List):
            self.activation_distance = self.device_cfg.to_device(self.activation_distance)
        elif isinstance(self.activation_distance, float):
            raise ValueError(f"Activation distance is a list for cspace cost. got {self.activation_distance}")
        if self.squared_l2_regularization_weight is not None:
            self.squared_l2_regularization_weight = self.device_cfg.to_device(
                self.squared_l2_regularization_weight
            )

        if self.cost_type is None:
            log_and_raise("specify cost type for bound cost")
        if isinstance(self.cost_type, str):
            self.cost_type = CSpaceCostType[self.cost_type]

        if self.cost_type in [CSpaceCostType.STATE]:
            if self.squared_l2_regularization_weight is None:
                self.squared_l2_regularization_weight = torch.zeros(5, device=self.device_cfg.device, dtype=self.device_cfg.dtype)
            if len(self.squared_l2_regularization_weight) != 5:
                log_and_raise(
                    "smooth weight needs to be of length 5, "
                    + "for velocity, acceleration, jerk, torque and energy"
                )
        if self.cost_type == CSpaceCostType.POSITION:
            if self.squared_l2_regularization_weight is None:
                self.squared_l2_regularization_weight = torch.zeros(
                    2, device=self.device_cfg.device, dtype=self.device_cfg.dtype
                )
            if len(self.squared_l2_regularization_weight) != 2:
                log_and_raise(
                    "squared_l2_regularization_weight for POSITION cost should be length 2, "
                    + "for velocity and acceleration"
                )

        if self.cspace_target_weight is None:
            self.cspace_target_weight = torch.zeros(1, device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        else:
            if isinstance(self.cspace_target_weight, float):
                self.cspace_target_weight = torch.tensor([self.cspace_target_weight], device=self.device_cfg.device, dtype=self.device_cfg.dtype)
            elif isinstance(self.cspace_target_weight, list):
                self.cspace_target_weight = torch.tensor(self.cspace_target_weight, device=self.device_cfg.device, dtype=self.device_cfg.dtype)
            elif isinstance(self.cspace_target_weight, torch.Tensor):
                self.cspace_target_weight = self.cspace_target_weight.to(self.device_cfg.device, dtype=self.device_cfg.dtype)
            if self.cspace_target_weight.shape != (1,):
                log_and_raise("cspace_target_weight should be of shape (1,)")
        if self.cspace_target_dof_weight is None:
            self.cspace_target_dof_weight = torch.ones(self.dof, device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        else:
            if isinstance(self.cspace_target_dof_weight, float):
                self.cspace_target_dof_weight = torch.tensor([self.cspace_target_dof_weight], device=self.device_cfg.device, dtype=self.device_cfg.dtype)
            elif isinstance(self.cspace_target_dof_weight, list):
                self.cspace_target_dof_weight = torch.tensor(self.cspace_target_dof_weight, device=self.device_cfg.device, dtype=self.device_cfg.dtype)
            elif isinstance(self.cspace_target_dof_weight, torch.Tensor):
                self.cspace_target_dof_weight = self.cspace_target_dof_weight.to(self.device_cfg.device, dtype=self.device_cfg.dtype)
            if self.cspace_target_dof_weight.shape != (self.dof,):
                log_and_raise("cspace_target_dof_weight should be of shape (dof,)")

        if self.cost_type == CSpaceCostType.POSITION:
            if len(self.weight) != 2:
                log_and_raise("bound weight should be of length 2")
            if len(self.activation_distance) != 2:
                log_and_raise("activation distance should be of length 2")
        if self.cost_type in [CSpaceCostType.STATE]:
            if len(self.weight) != 5:
                log_and_raise("bound weight should be of length 5")
            if len(self.activation_distance) != 5:
                log_and_raise("activation distance should be of length 5")

        if self.cspace_non_terminal_weight_factor is None:
            self.cspace_non_terminal_weight_factor = torch.ones(1, device=self.device_cfg.device, dtype=self.device_cfg.dtype)
        else:
            if isinstance(self.cspace_non_terminal_weight_factor, float):
                self.cspace_non_terminal_weight_factor = torch.tensor([self.cspace_non_terminal_weight_factor], device=self.device_cfg.device, dtype=self.device_cfg.dtype)
            elif isinstance(self.cspace_non_terminal_weight_factor, list):
                self.cspace_non_terminal_weight_factor = torch.tensor(self.cspace_non_terminal_weight_factor, device=self.device_cfg.device, dtype=self.device_cfg.dtype)
            elif isinstance(self.cspace_non_terminal_weight_factor, torch.Tensor):
                self.cspace_non_terminal_weight_factor = self.cspace_non_terminal_weight_factor.to(self.device_cfg.device, dtype=self.device_cfg.dtype)

        if self.cost_type == CSpaceCostType.STATE:
            self.class_type = StateCSpaceCost
        elif self.cost_type == CSpaceCostType.POSITION:
            self.class_type = PositionCSpaceCost
        if self.dof is not None:
            self.update_dof(self.dof)

        return super().__post_init__()

    def initialize_from_transition_model(self, transition_model: RobotStateTransition):
        self.update_dof(transition_model.action_dim)

        self.set_bounds(
            transition_model.get_state_bounds(),
            teleport_mode=transition_model.teleport_mode,
        )

    def update_dof(self, dof: int):
        self.dof = dof
        self.cspace_target_dof_weight = torch.ones(self.dof, device=self.device_cfg.device,
        dtype=self.device_cfg.dtype)
