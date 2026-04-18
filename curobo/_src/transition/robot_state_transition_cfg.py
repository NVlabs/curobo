# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Standard Library
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional, Type, Union

# Third Party
import torch

# CuRobo
from curobo._src.transition.robot_state_transition import RobotStateTransition
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util.logging import log_and_raise
from curobo._src.util.state_filter import FilterCfg


@dataclass
class TimeTrajCfg:
    base_dt: float
    base_ratio: float
    max_dt: float

    def get_dt_array(self, num_points: int):
        dt_array = [self.base_dt] * int(self.base_ratio * num_points)

        smooth_blending = torch.linspace(
            self.base_dt,
            self.max_dt,
            steps=int((1 - self.base_ratio) * num_points),
        ).tolist()
        dt_array += smooth_blending
        if len(dt_array) != num_points:
            dt_array.insert(0, dt_array[0])
        return dt_array

    def update_dt(
        self,
        all_dt: float = None,
        base_dt: float = None,
        max_dt: float = None,
        base_ratio: float = None,
    ):
        if all_dt is not None:
            self.base_dt = all_dt
            self.max_dt = all_dt
            return
        if base_dt is not None:
            self.base_dt = base_dt
        if base_ratio is not None:
            self.base_ratio = base_ratio
        if max_dt is not None:
            self.max_dt = max_dt


@dataclass(frozen=False)
class RobotStateTransitionCfg:
    robot_config: RobotCfg
    dt_traj_params: TimeTrajCfg
    device_cfg: DeviceCfg
    vel_scale: float = 1.0
    state_estimation_variance: float = 0.0
    batch_size: int = 1
    horizon: int = 5
    n_knots: int = 0
    control_space: ControlSpace = ControlSpace.ACCELERATION
    state_filter_cfg: Optional[FilterCfg] = None
    teleport_mode: bool = False
    return_full_act_buffer: bool = False
    state_finite_difference_mode: str = "BACKWARD"
    filter_robot_command: bool = False
    interpolation_steps: int = 1
    class_type: Type[RobotStateTransition] = RobotStateTransition

    @staticmethod
    def create(
        data_dict_in, robot_cfg: Union[Dict, RobotCfg], device_cfg=DeviceCfg()
    ):
        data_dict = deepcopy(data_dict_in)
        if isinstance(robot_cfg, dict):
            robot_cfg = RobotCfg.create(robot_cfg, device_cfg)
        data_dict["robot_config"] = robot_cfg
        data_dict["dt_traj_params"] = TimeTrajCfg(**data_dict["dt_traj_params"])
        data_dict["control_space"] = ControlSpace[data_dict["control_space"]]
        data_dict["state_filter_cfg"] = FilterCfg.create(
            data_dict["state_filter_cfg"]["filter_coeff"],
            enable=data_dict["state_filter_cfg"]["enable"],
            dt=data_dict["dt_traj_params"].base_dt,
            control_space=data_dict["control_space"],
            device_cfg=device_cfg,
            teleport_mode=data_dict["teleport_mode"],
        )

        return RobotStateTransitionCfg(**data_dict, device_cfg=device_cfg)

    def __post_init__(self):
        if self.control_space in ControlSpace.bspline_types():
            if self.interpolation_steps < 1:
                log_and_raise("interpolation_steps needs to be greater than 0")
            if self.interpolation_steps > 32:
                log_and_raise("interpolation_steps needs to be less than 32")
            if self.n_knots <= 5:
                log_and_raise("n_knots needs to be greater than 5")
            if self.control_space in ControlSpace.bspline_types():
                if self.teleport_mode:
                    log_and_raise("Teleport mode is not supported for bspline control spaces")
                self.horizon = ControlSpace.spline_total_interpolation_steps(
                    self.control_space, self.n_knots, self.interpolation_steps
                )
