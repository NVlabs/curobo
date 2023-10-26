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
# Standard Library
from dataclasses import dataclass
from typing import Optional

# CuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.enum import StateType
from curobo.types.state import FilterCoeff, JointState
from curobo.types.tensor import T_DOF


@dataclass(frozen=True)
class FilterConfig:
    filter_coeff: FilterCoeff
    dt: float
    control_space: StateType
    tensor_args: TensorDeviceType = TensorDeviceType()
    enable: bool = True
    teleport_mode: bool = False

    @staticmethod
    def from_dict(
        coeff_dict,
        enable=True,
        dt=0.0,
        control_space=StateType.ACCELERATION,
        tensor_args=TensorDeviceType(),
        teleport_mode=False,
    ):
        data = {}
        data["filter_coeff"] = FilterCoeff(**coeff_dict)
        data["dt"] = dt
        data["control_space"] = control_space
        data["enable"] = enable
        data["teleport_mode"] = teleport_mode
        return FilterConfig(**data, tensor_args=tensor_args)


class JointStateFilter(FilterConfig):
    def __init__(self, filter_config: FilterConfig):
        super().__init__(**vars(filter_config))
        self.cmd_joint_state = None
        if self.control_space == StateType.ACCELERATION:
            self.integrate_action = self.integrate_acc
        elif self.control_space == StateType.VELOCITY:
            self.integrate_action = self.integrate_vel
        elif self.control_space == StateType.POSITION:
            self.integrate_action = self.integrate_pos

    def filter_joint_state(self, raw_joint_state: JointState):
        if not self.enable:
            return raw_joint_state
        raw_joint_state = raw_joint_state.to(self.tensor_args)

        if self.cmd_joint_state is None:
            self.cmd_joint_state = raw_joint_state.clone()
            return self.cmd_joint_state
        self.cmd_joint_state.blend(self.filter_coeff, raw_joint_state)
        return self.cmd_joint_state

    def integrate_jerk(
        self, qddd_des, cmd_joint_state: Optional[JointState] = None, dt: Optional[float] = None
    ):
        dt = self.dt if dt is None else dt
        if cmd_joint_state is not None:
            if self.cmd_joint_state is None:
                self.cmd_joint_state = cmd_joint_state.clone()
            else:
                self.cmd_joint_state.copy_(cmd_joint_state)
        self.cmd_joint_state.acceleration[:] = self.cmd_joint_state.acceleration + qddd_des * dt
        self.cmd_joint_state.velocity[:] = (
            self.cmd_joint_state.velocity + self.cmd_joint_state.acceleration * dt
        )
        self.cmd_joint_state.position[:] = (
            self.cmd_joint_state.position + self.cmd_joint_state.velocity * dt
        )

        return self.cmd_joint_state

    def integrate_acc(
        self,
        qdd_des: T_DOF,
        cmd_joint_state: Optional[JointState] = None,
        dt: Optional[float] = None,
    ):
        dt = self.dt if dt is None else dt
        if cmd_joint_state is not None:
            if self.cmd_joint_state is None:
                self.cmd_joint_state = cmd_joint_state.clone()
            else:
                self.cmd_joint_state.copy_(cmd_joint_state)
        self.cmd_joint_state.acceleration[:] = qdd_des
        self.cmd_joint_state.velocity[:] = self.cmd_joint_state.velocity + qdd_des * dt
        self.cmd_joint_state.position[:] = (
            self.cmd_joint_state.position + self.cmd_joint_state.velocity * dt
        )
        # TODO: for now just have zero jerl:
        if self.cmd_joint_state.jerk is None:
            self.cmd_joint_state.jerk = qdd_des * 0.0
        else:
            self.cmd_joint_state.jerk[:] = qdd_des * 0.0
        return self.cmd_joint_state.clone()

    def integrate_vel(
        self,
        qd_des: T_DOF,
        cmd_joint_state: Optional[JointState] = None,
        dt: Optional[float] = None,
    ):
        dt = self.dt if dt is None else dt
        if cmd_joint_state is not None:
            self.cmd_joint_state = cmd_joint_state
        self.cmd_joint_state.velocity = qd_des
        self.cmd_joint_state.position = (
            self.cmd_joint_state.position + self.cmd_joint_state.velocity * dt
        )

        return self.cmd_joint_state

    def integrate_pos(
        self, q_des: T_DOF, cmd_joint_state: Optional[JointState] = None, dt: Optional[float] = None
    ):
        dt = self.dt if dt is None else dt
        if cmd_joint_state is not None:
            self.cmd_joint_state = cmd_joint_state

        if not self.teleport_mode:
            self.cmd_joint_state.velocity = (q_des - self.cmd_joint_state.position) / dt
        self.cmd_joint_state.position = q_des
        return self.cmd_joint_state

    def reset(self):
        self.cmd_joint_state = None
