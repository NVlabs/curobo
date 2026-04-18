# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

# Standard Library
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg

# CuRobo
from curobo._src.robot.loader.kinematics_loader_cfg import KinematicsLoaderCfg
from curobo._src.robot.types.cspace_params import CSpaceParams
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util_file import write_yaml


@dataclass
class RobotCfg:
    """Robot configuration to load in CuRobo.

    Tutorial: :ref:`tut_robot_configuration`
    """

    #: Robot kinematics configuration to load into cuda robot model.
    kinematics: KinematicsCfg

    #: Robot dynamics configuration (optional, for inverse dynamics computation).
    dynamics: Optional[DynamicsCfg] = None

    device_cfg: DeviceCfg = DeviceCfg()

    @staticmethod
    def _create_dynamics_config(
        kinematics_config,
        device_cfg: DeviceCfg,
    ) -> Optional[DynamicsCfg]:
        """Create dynamics configuration.

        Args:
            kinematics_config: KinematicsParams with inertial properties.
            device_cfg: Device configuration.

        Returns:
            DynamicsCfg instance.
        """
        return DynamicsCfg(
            kinematics_config=kinematics_config, device_cfg=device_cfg
        )

    @staticmethod
    def create(
        data: Union[Dict[str, Any], "RobotCfg"],
        device_cfg: DeviceCfg = DeviceCfg(),
        load_collision_spheres: bool = True,
        num_envs: int = 1,
    ) -> "RobotCfg":
        """Create RobotCfg from a dictionary or return existing RobotCfg.

        Args:
            data: Robot configuration dictionary or existing RobotCfg object.
                If dict, can include:
                - load_dynamics (bool): Whether to load dynamics model.
            device_cfg: Device configuration.
            load_collision_spheres: When False, skip loading collision spheres
                from the robot config. Saves memory and compute when collision
                checking is not needed.
            num_envs: Number of environment configurations for link_spheres.

        Returns:
            Configured RobotCfg instance.
        """
        if isinstance(data, RobotCfg):
            return data

        data_dict_in = data
        if "robot_cfg" in data_dict_in:
            data_dict_in = data_dict_in["robot_cfg"]
        data_dict = deepcopy(data_dict_in)
        if isinstance(data_dict["kinematics"], dict):
            data_dict["kinematics"] = KinematicsCfg.from_config(
                KinematicsLoaderCfg(
                    **data_dict_in["kinematics"],
                    device_cfg=device_cfg,
                    load_collision_spheres=load_collision_spheres,
                    num_envs=num_envs,
                )
            )

        load_dynamics = data_dict.pop("load_dynamics", False)

        dynamics_config = None
        if load_dynamics:
            dynamics_config = RobotCfg._create_dynamics_config(
                kinematics_config=data_dict["kinematics"].kinematics_config,
                device_cfg=device_cfg,
            )

        data_dict["dynamics"] = dynamics_config
        return RobotCfg(**data_dict, device_cfg=device_cfg)

    @staticmethod
    def from_basic(
        urdf_path: str,
        base_link: str,
        tool_frames: List[str],
        device_cfg: DeviceCfg = DeviceCfg(),
        load_dynamics: bool = False,
    ):
        """Create RobotCfg from basic URDF parameters.

        Args:
            urdf_path: Path to URDF file.
            base_link: Name of the robot's base link.
            tool_frames: List of end-effector link names.
            device_cfg: Device configuration.
            load_dynamics: Whether to load dynamics model.

        Returns:
            Configured RobotCfg instance.
        """
        cuda_robot_model_config = KinematicsCfg.from_basic_urdf(
            urdf_path=urdf_path, base_link=base_link, tool_frames=tool_frames, device_cfg=device_cfg
        )

        dynamics_config = None
        if load_dynamics:
            dynamics_config = RobotCfg._create_dynamics_config(
                kinematics_config=cuda_robot_model_config.kinematics_config,
                device_cfg=device_cfg,
            )

        return RobotCfg(
            cuda_robot_model_config,
            dynamics=dynamics_config,
            device_cfg=device_cfg,
        )

    def write_config(self, file_path):
        dictionary = vars(self)
        dictionary["kinematics"] = vars(dictionary["kinematics"])
        # Handle dynamics config - just store a flag
        if dictionary["dynamics"] is not None:
            dictionary["load_dynamics"] = True
            dictionary.pop("dynamics")  # Don't serialize the full config
        else:
            dictionary["load_dynamics"] = False
            dictionary.pop("dynamics", None)
        write_yaml(dictionary, file_path)

    @property
    def cspace(self) -> CSpaceParams:
        return self.kinematics.cspace
