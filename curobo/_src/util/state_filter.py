# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Standard Library
from dataclasses import dataclass
from typing import Optional

from curobo._src.state.filter_coeff import FilterCoeff
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_ops import blend_joint_states

# CuRobo
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tensor import T_DOF


@dataclass(frozen=True)
class FilterCfg:
    filter_coeff: FilterCoeff
    dt: float
    control_space: ControlSpace
    device_cfg: DeviceCfg = DeviceCfg()
    enable: bool = True
    teleport_mode: bool = False

    @staticmethod
    def create(
        coeff_dict,
        enable=True,
        dt=0.0,
        control_space=ControlSpace.ACCELERATION,
        device_cfg=DeviceCfg(),
        teleport_mode=False,
    ):
        data = {}
        data["filter_coeff"] = FilterCoeff(**coeff_dict)
        data["dt"] = dt
        data["control_space"] = control_space
        data["enable"] = enable
        data["teleport_mode"] = teleport_mode
        return FilterCfg(**data, device_cfg=device_cfg)


class JointStateFilter(FilterCfg):
    def __init__(self, filter_config: FilterCfg):
        super().__init__(**vars(filter_config))
        self.cmd_joint_state = None
        if self.control_space == ControlSpace.ACCELERATION:
            self.integrate_action = self.integrate_acc
        elif self.control_space == ControlSpace.VELOCITY:
            self.integrate_action = self.integrate_vel
        elif self.control_space in [ControlSpace.POSITION] + ControlSpace.bspline_types():
            self.integrate_action = self.integrate_pos

    def filter_joint_state(self, raw_joint_state: JointState):
        if not self.enable:
            return raw_joint_state
        raw_joint_state = raw_joint_state.to(self.device_cfg)

        if self.cmd_joint_state is None:
            self.cmd_joint_state = raw_joint_state.clone()
            return self.cmd_joint_state
        blend_joint_states(self.cmd_joint_state, raw_joint_state, self.filter_coeff)
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
        self,
        q_des: T_DOF,
        cmd_joint_state: Optional[JointState] = None,
        dt: Optional[float] = None,
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
