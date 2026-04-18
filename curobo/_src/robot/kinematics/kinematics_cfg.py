# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from curobo._src.robot.loader.kinematics_loader import KinematicsLoader
from curobo._src.robot.loader.kinematics_loader_cfg import KinematicsLoaderCfg
from curobo._src.robot.loader.util import load_robot_yaml
from curobo._src.robot.parser.parser_base import RobotParser
from curobo._src.robot.types.cspace_params import CSpaceParams
from curobo._src.robot.types.joint_limits import JointLimits
from curobo._src.robot.types.kinematics_params import KinematicsParams
from curobo._src.robot.types.self_collision_params import SelfCollisionKinematicsCfg
from curobo._src.types.content_path import ContentPath
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.config_io import is_file_xrdf
from curobo._src.util.logging import log_and_raise


@dataclass
class KinematicsCfg:
    """Configuration for robot kinematics on GPU.

    Helper functions are provided to load this configuration from an URDF file or from
    a cuRobo robot configuration file (:ref:`tut_robot_configuration`). To create from a XRDF, use
    :ref:`curobo.util.xrdf_utils.convert_xrdf_to_curobo`.
    """

    #: Device and floating point precision to use for kinematics.
    device_cfg: DeviceCfg

    #: Names of links to compute poses with forward kinematics.
    tool_frames: List[str]

    #: Tensors representing kinematics of the robot. This can be created using
    #: :class:`~curobo.cuda_robot_model.cuda_robot_generator.KinematicsLoader`.
    kinematics_config: KinematicsParams

    #: Collision pairs to ignore when computing self collision between spheres across all
    #: robot links. This also contains distance threshold between spheres pairs and which thread
    #: indices for calculating the distances. More details on computing these parameters is in
    #: :func:`~curobo.cuda_robot_model.cuda_robot_generator.KinematicsLoader._build_collision_model`.
    self_collision_config: Optional[SelfCollisionKinematicsCfg] = None

    #: Parser to load kinematics from URDF or USD files. This is used to load kinematics
    #: representation of the robot. This is created using
    #: :class:`~curobo.cuda_robot_model.kinematics_parser.RobotParser`.
    #: USD is an experimental feature and might not work for all robots.
    kinematics_parser: Optional[RobotParser] = None

    #: Generator config used to create this robot kinematics model.
    generator_config: Optional[KinematicsLoaderCfg] = None

    def __post_init__(self):
        self.kinematics_config.make_contiguous()

    def get_joint_limits(self) -> JointLimits:
        """Get limits of actuated joints of the robot.

        Returns:
            JointLimits: Joint limits of the robot's actuated joints.
        """
        return self.kinematics_config.joint_limits

    @staticmethod
    def from_basic_urdf(
        urdf_path: str,
        base_link: str,
        tool_frames: List[str],
        device_cfg: DeviceCfg = DeviceCfg(),
    ) -> KinematicsCfg:
        """Load a cuda robot model from only urdf. This does not support collision queries.

        Args:
            urdf_path : Path of urdf file.
            base_link : Name of base link.
            tool_frames : List of tool frames to compute poses for. If None, all links are computed.
            device_cfg : Device to load robot model. Defaults to DeviceCfg().

        Returns:
            KinematicsCfg: cuda robot model configuration.
        """
        config = KinematicsLoaderCfg(
            base_link=base_link, tool_frames=tool_frames, device_cfg=device_cfg, urdf_path=urdf_path)
        return KinematicsCfg.from_config(config)

    @staticmethod
    def from_content_path(
        content_path: ContentPath,
        tool_frames: Optional[List[str]] = None,
        device_cfg: DeviceCfg = DeviceCfg(),
        **kwargs: Any,
    ) -> KinematicsCfg:
        """Load robot from Contentpath containing paths to robot description files.

        Args:
            content_path: Path to robot configuration files.
            tool_frames: List of tool frames to compute poses for. If None, all links are computed.
            device_cfg: Device to load robot model, defaults to cuda:0.

        Returns:
            KinematicsCfg: cuda robot model configuration.
        """
        config_file = load_robot_yaml(content_path)
        if "robot_cfg" in config_file:
            config_file = config_file["robot_cfg"]
        if "kinematics" in config_file:
            config_file = config_file["kinematics"]
        if tool_frames is not None:
            config_file["tool_frames"] = tool_frames
        if kwargs:
            config_file.update(kwargs)

        return KinematicsCfg.from_config(
            KinematicsLoaderCfg(**config_file, device_cfg=device_cfg),
        )

    @staticmethod
    def from_robot_yaml_file(
        file_path: Union[str, Dict],
        tool_frames: Optional[List[str]] = None,
        device_cfg: DeviceCfg = DeviceCfg(),
        urdf_path: Optional[str] = None,
        **kwargs: Any,
    ) -> KinematicsCfg:
        """Load robot from a yaml file that is in cuRobo's format (:ref:`tut_robot_configuration`).

        Args:
            file_path: Path to robot configuration file (yml or xrdf).
            tool_frames: List of tool frames to compute poses for. If None, all links are computed.
            device_cfg: Device to load robot model, defaults to cuda:0.
            urdf_path: Path to urdf file. This is required when loading a xrdf file.

        Returns:
            KinematicsCfg: cuda robot model configuration.
        """
        if isinstance(file_path, dict):
            content_path = ContentPath(robot_urdf_file=urdf_path, robot_config_file=file_path)
        else:
            if is_file_xrdf(file_path):
                content_path = ContentPath(robot_urdf_file=urdf_path, robot_xrdf_file=file_path)
            else:
                content_path = ContentPath(robot_urdf_file=urdf_path, robot_config_file=file_path)

        if tool_frames is not None:
            if not isinstance(tool_frames, list):
                log_and_raise(f"tool_frames must be a list, but is {type(tool_frames)}")
        return KinematicsCfg.from_content_path(
            content_path, tool_frames=tool_frames, device_cfg=device_cfg,
            **kwargs,
        )

    @staticmethod
    def from_config_file(
        file_path: Union[str, Dict],
        tool_frames: Optional[List[str]] = None,
        device_cfg: DeviceCfg = DeviceCfg(),
        urdf_path: Optional[str] = None,
    ) -> KinematicsCfg:
        """Load robot from a yaml file that is in cuRobo's format (:ref:`tut_robot_configuration`).

        Args:
            file_path: Path to robot configuration file (yml or xrdf).
            tool_frames: List of tool frames to compute poses for. If None, all links are computed.
            device_cfg: Device to load robot model, defaults to cuda:0.
            urdf_path: Path to urdf file. This is required when loading a xrdf file.

        Returns:
            KinematicsCfg: cuda robot model configuration.
        """
        if isinstance(file_path, dict):
            content_path = ContentPath(robot_urdf_file=urdf_path, robot_config_file=file_path)
        else:
            if is_file_xrdf(file_path):
                content_path = ContentPath(robot_urdf_file=urdf_path, robot_xrdf_file=file_path)
            else:
                content_path = ContentPath(robot_urdf_file=urdf_path, robot_config_file=file_path)

        return KinematicsCfg.from_content_path(
            content_path, tool_frames=tool_frames, device_cfg=device_cfg
        )

    @staticmethod
    def from_data_dict(
        data_dict: Dict[str, Any],
        tool_frames: Optional[List[str]] = None,
        device_cfg: DeviceCfg = DeviceCfg(),
    ) -> KinematicsCfg:
        """Load robot from a dictionary containing data for :class:`~curobo.cuda_robot_model.cuda_robot_generator.KinematicsLoaderCfg`.

        :tut_robot_configuration discusses the data required to load a robot.

        Args:
            data_dict: Input dictionary containing robot configuration.
            device_cfg: Device to load robot model, defaults to cuda:0.

        Returns:
            KinematicsCfg: cuda robot model configuration.
        """
        if "robot_cfg" in data_dict:
            data_dict = data_dict["robot_cfg"]
        if "kinematics" in data_dict:
            data_dict = data_dict["kinematics"]
        if "device_cfg" not in data_dict:
            data_dict["device_cfg"] = device_cfg
        if tool_frames is not None:
            data_dict["tool_frames"] = tool_frames
        return KinematicsCfg.from_config(KinematicsLoaderCfg(**data_dict))

    @staticmethod
    def from_config(config: KinematicsLoaderCfg) -> KinematicsCfg:
        """Create a robot model configuration from a generator configuration.

        Args:
            config: Input robot generator configuration.

        Returns:
            KinematicsCfg: robot model configuration.
        """
        # create a config generator and load all values
        generator = KinematicsLoader(config)
        return KinematicsCfg(
            device_cfg=generator.device_cfg,
            tool_frames=generator.tool_frames,
            kinematics_config=generator.kinematics_config,
            self_collision_config=generator.self_collision_config,
            kinematics_parser=generator.kinematics_parser,
            generator_config=config,
        )

    @property
    def cspace(self) -> CSpaceParams:
        """Get cspace parameters of the robot."""
        return self.kinematics_config.cspace

    @property
    def dof(self) -> int:
        """Get the number of actuated joints (degrees of freedom) of the robot"""
        return self.kinematics_config.num_dof
