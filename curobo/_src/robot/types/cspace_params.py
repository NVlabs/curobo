# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Configuration space parameters for robot kinematics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union

import torch

from curobo._src.robot.types.joint_limits import JointLimits
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tensor import T_DOF
from curobo._src.util.logging import log_and_raise
from curobo._src.util.tensor_util import clone_if_not_none, copy_or_clone


@dataclass
class CSpaceParams:
    """Configuration space parameters of the robot."""

    #: Names of the joints.
    joint_names: List[str]

    #: Default joint position for the robot. This is the configuration used to bias graph search
    #: and also regularize inverse kinematics. This configuration is also used to initialize
    #: the robot during warmup phase of an optimizer. Set this to a collision-free configuration
    #: for good performance. When this configuration is in collision, it's not used to bias
    #: graph search.
    default_joint_position: T_DOF | None = None

    #: Weight for each joint in configuration space. Used to measure distance between nodes in
    #: graph search-based planning.
    cspace_distance_weight: T_DOF | None = None

    #: Weight for each joint, used in regularization cost term for inverse kinematics.
    null_space_weight: T_DOF | None = None

    #: Maximum distance for each joint. This is a multiplier on the total range of the joint limits.
    #: So the largest value is 1.0.
    null_space_maximum_distance: T_DOF | None = None

    #: Device and floating point precision for tensors.
    device_cfg: DeviceCfg = DeviceCfg()

    #: Maximum acceleration for each joint. Accepts a scalar or a list of values for each joint.
    max_acceleration: Union[float, List[float]] = 10.0

    #: Maximum jerk for each joint. Accepts a scalar or a list of values for each joint.
    max_jerk: Union[float, List[float]] = 500.0

    #: Velocity scale for each joint. Accepts a scalar or a list of values for each joint.
    #: This is used to scale the velocity limits for each joint.
    velocity_scale: Union[float, List[float]] = 1.0

    #: Acceleration scale for each joint. Accepts a scalar or a list of values for each joint.
    #: This is used to scale the acceleration limits for each joint.
    acceleration_scale: Union[float, List[float]] = 1.0

    #: Jerk scale for each joint. Accepts a scalar or a list of values for each joint.
    #: This is used to scale the jerk limits for each joint.
    jerk_scale: Union[float, List[float]] = 1.0

    #: Position limit clip value. This is used to clip the position limits for each joint.
    #: Accepts a scalar or a list of values for each joint. This is useful to truncate limits
    #: to account for any safety margins imposed by real robot controllers.
    position_limit_clip: Union[float, List[float]] = 0.0

    def __post_init__(self):
        """Post initialization checks and data transfer to device tensors."""
        if self.default_joint_position is not None:
            self.default_joint_position = self.device_cfg.to_device(self.default_joint_position)
        if self.cspace_distance_weight is not None:
            self.cspace_distance_weight = self.device_cfg.to_device(self.cspace_distance_weight)
        if self.null_space_weight is not None:
            self.null_space_weight = self.device_cfg.to_device(self.null_space_weight)
        if isinstance(self.max_acceleration, float):
            self.max_acceleration = self.device_cfg.to_device(
                [self.max_acceleration for _ in self.joint_names]
            )

        if isinstance(self.velocity_scale, float) or len(self.velocity_scale) == 1:
            self.velocity_scale = self.device_cfg.to_device(
                [self.velocity_scale for _ in self.joint_names]
            ).view(-1)

        if isinstance(self.acceleration_scale, float) or len(self.acceleration_scale) == 1:
            self.acceleration_scale = self.device_cfg.to_device(
                [self.acceleration_scale for _ in self.joint_names]
            ).view(-1)

        if isinstance(self.jerk_scale, float) or len(self.jerk_scale) == 1 and len(self.joint_names) > 1:
            self.jerk_scale = self.device_cfg.to_device(
                [self.jerk_scale for _ in self.joint_names]
            ).view(-1)

        if isinstance(self.max_acceleration, List):
            self.max_acceleration = self.device_cfg.to_device(self.max_acceleration)
        if isinstance(self.max_jerk, float):
            self.max_jerk = [self.max_jerk for _ in self.joint_names]
        if isinstance(self.max_jerk, List):
            self.max_jerk = self.device_cfg.to_device(self.max_jerk)
        if isinstance(self.velocity_scale, List):
            self.velocity_scale = self.device_cfg.to_device(self.velocity_scale)

        if isinstance(self.acceleration_scale, List):
            self.acceleration_scale = self.device_cfg.to_device(self.acceleration_scale)
        if isinstance(self.jerk_scale, List):
            self.jerk_scale = self.device_cfg.to_device(self.jerk_scale)
        if isinstance(self.position_limit_clip, List):
            self.position_limit_clip = self.device_cfg.to_device(self.position_limit_clip)

        if self.null_space_maximum_distance is None and self.null_space_weight is not None:
            self.null_space_maximum_distance = self.device_cfg.to_device(
                [0.1 for _ in self.joint_names]
            )
        if self.null_space_maximum_distance is not None:
            self.null_space_maximum_distance = self.device_cfg.to_device(
                self.null_space_maximum_distance
            )

        # check shapes:

        if self.default_joint_position is not None:
            dof = self.default_joint_position.shape

            if self.cspace_distance_weight is not None and self.cspace_distance_weight.shape != dof:
                log_and_raise(
                    "cspace_distance_weight shape: "
                    + str(self.cspace_distance_weight.shape)
                    + " does not match default_joint_position shape: "
                    + str(dof)
                )
            if self.null_space_weight is not None and self.null_space_weight.shape != dof:
                log_and_raise("null_space_weight shape does not match default_joint_position")
            if (
                self.null_space_maximum_distance is not None
                and self.null_space_maximum_distance.shape != dof
            ):
                log_and_raise(
                    "null_space_maximum_distance shape: "
                    + str(self.null_space_maximum_distance.shape)
                    + " does not match default_joint_position shape: "
                    + str(dof)
                )

    def inplace_reindex(self, joint_names: List[str]):
        """Change order of joints in configuration space tensors to match given order of names.

        Args:
            joint_names: New order of joint names.

        """
        new_index = [self.joint_names.index(j) for j in joint_names]
        if self.default_joint_position is not None:
            self.default_joint_position = self.default_joint_position[new_index].clone()
        if self.cspace_distance_weight is not None:
            self.cspace_distance_weight = self.cspace_distance_weight[new_index].clone()
        if self.null_space_weight is not None:
            self.null_space_weight = self.null_space_weight[new_index].clone()
        if self.null_space_maximum_distance is not None:
            self.null_space_maximum_distance = self.null_space_maximum_distance[new_index].clone()
        self.max_acceleration = self.max_acceleration[new_index].clone()
        self.max_jerk = self.max_jerk[new_index].clone()
        self.velocity_scale = self.velocity_scale[new_index].clone()
        self.acceleration_scale = self.acceleration_scale[new_index].clone()
        self.jerk_scale = self.jerk_scale[new_index].clone()
        joint_names = [self.joint_names[n] for n in new_index]
        self.joint_names = joint_names

    def copy_(self, new_config: CSpaceParams) -> CSpaceParams:
        """Copy parameters from another instance.

        This maintains reference and copies the data. Assumes that the new instance has the same
        number of joints as the current instance and also same shape of tensors.

        Args:
            new_config: New parameters to copy into current instance.

        Returns:
            CSpaceParams: Same instance of cspace configuration which has updated parameters.
        """
        self.joint_names = new_config.joint_names.copy()
        self.default_joint_position = copy_or_clone(new_config.default_joint_position, self.default_joint_position)
        self.null_space_weight = copy_or_clone(new_config.null_space_weight, self.null_space_weight)
        self.cspace_distance_weight = copy_or_clone(
            new_config.cspace_distance_weight, self.cspace_distance_weight
        )
        self.device_cfg = self.device_cfg
        self.max_jerk = copy_or_clone(new_config.max_jerk, self.max_jerk)
        self.max_acceleration = copy_or_clone(new_config.max_acceleration, self.max_acceleration)
        self.velocity_scale = copy_or_clone(new_config.velocity_scale, self.velocity_scale)
        self.acceleration_scale = copy_or_clone(
            new_config.acceleration_scale, self.acceleration_scale
        )
        self.jerk_scale = copy_or_clone(new_config.jerk_scale, self.jerk_scale)
        self.null_space_maximum_distance = copy_or_clone(
            new_config.null_space_maximum_distance, self.null_space_maximum_distance
        )
        return self

    def clone(self) -> CSpaceParams:
        """Clone configuration space parameters."""
        return CSpaceParams(
            joint_names=self.joint_names.copy(),
            default_joint_position=clone_if_not_none(self.default_joint_position),
            null_space_weight=clone_if_not_none(self.null_space_weight),
            null_space_maximum_distance=clone_if_not_none(self.null_space_maximum_distance),
            cspace_distance_weight=clone_if_not_none(self.cspace_distance_weight),
            device_cfg=self.device_cfg,
            max_jerk=self.max_jerk.clone(),
            max_acceleration=self.max_acceleration.clone(),
            velocity_scale=self.velocity_scale.clone(),
            acceleration_scale=self.acceleration_scale.clone(),
            jerk_scale=self.jerk_scale.clone(),
            position_limit_clip=(
                self.position_limit_clip.clone()
                if isinstance(self.position_limit_clip, torch.Tensor)
                else self.position_limit_clip
            ),
        )

    def scale_joint_limits(self, joint_limits: JointLimits) -> JointLimits:
        """Scale joint limits by the given scale factors.

        Args:
            joint_limits: Joint limits to scale.

        Returns:
            JointLimits: Scaled joint limits.
        """
        if self.velocity_scale is not None:
            joint_limits.velocity = joint_limits.velocity * self.velocity_scale
        if self.acceleration_scale is not None:
            joint_limits.acceleration = joint_limits.acceleration * self.acceleration_scale
        if self.jerk_scale is not None:
            joint_limits.jerk = joint_limits.jerk * self.jerk_scale

        return joint_limits

    @staticmethod
    def load_from_joint_limits(
        joint_position_upper: torch.Tensor,
        joint_position_lower: torch.Tensor,
        joint_names: List[str],
        device_cfg: DeviceCfg = DeviceCfg(),
    ) -> CSpaceParams:
        """Load CSpace configuration from joint limits.

        Args:
            joint_position_upper: Upper position limits for each joint.
            joint_position_lower: Lower position limits for each joint.
            joint_names: Names of the joints. This should match the order of joints in the upper
                and lower limits.
            device_cfg: Device and floating point precision for tensors.

        Returns:
            CSpaceParams: CSpace configuration with default joint configuration set to the middle
                of the joint limits and all weights set to 1.
        """
        default_joint_position = ((joint_position_upper + joint_position_lower) / 2).flatten()
        num_dof = default_joint_position.shape[-1]
        null_space_weight = torch.ones(num_dof, **(device_cfg.as_torch_dict()))
        cspace_distance_weight = torch.ones(num_dof, **(device_cfg.as_torch_dict()))
        null_space_maximum_distance = torch.ones(num_dof, **(device_cfg.as_torch_dict()))
        return CSpaceParams(
            joint_names,
            default_joint_position,
            cspace_distance_weight,
            null_space_weight,
            null_space_maximum_distance=null_space_maximum_distance,
            device_cfg=device_cfg,
        )

