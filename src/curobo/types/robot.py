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
from __future__ import annotations

# Standard Library
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

# CuRobo
from curobo.cuda_robot_model.cuda_robot_generator import CudaRobotGeneratorConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModelConfig
from curobo.cuda_robot_model.types import CSpaceConfig
from curobo.types.base import TensorDeviceType
from curobo.types.state import JointState, State  # For compatibility with older versions.
from curobo.util_file import write_yaml


@dataclass
class NNConfig:
    ik_seeder: Optional[Any] = None


@dataclass
class RobotConfig:
    """Robot configuration to load in CuRobo.

    Tutorial: :ref:`tut_robot_configuration`
    """

    #: Robot kinematics configuration to load into cuda robot model.
    kinematics: CudaRobotModelConfig

    tensor_args: TensorDeviceType = TensorDeviceType()

    @staticmethod
    def from_dict(data_dict_in, tensor_args=TensorDeviceType()):
        if "robot_cfg" in data_dict_in:
            data_dict_in = data_dict_in["robot_cfg"]
        data_dict = deepcopy(data_dict_in)
        if isinstance(data_dict["kinematics"], dict):
            data_dict["kinematics"] = CudaRobotModelConfig.from_config(
                CudaRobotGeneratorConfig(**data_dict_in["kinematics"], tensor_args=tensor_args)
            )

        return RobotConfig(**data_dict, tensor_args=tensor_args)

    @staticmethod
    def from_basic(urdf_path, base_link, ee_link, tensor_args=TensorDeviceType()):
        cuda_robot_model_config = CudaRobotModelConfig.from_basic_urdf(
            urdf_path, base_link, ee_link, tensor_args
        )

        return RobotConfig(
            cuda_robot_model_config,
            tensor_args=tensor_args,
        )

    def write_config(self, file_path):
        dictionary = vars(self)
        dictionary["kinematics"] = vars(dictionary["kinematics"])
        write_yaml(dictionary, file_path)

    @property
    def cspace(self) -> CSpaceConfig:
        return self.kinematics.cspace
