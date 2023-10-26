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

# CuRobo
from curobo.cuda_robot_model.cuda_robot_generator import (
    CudaRobotGenerator,
    CudaRobotGeneratorConfig,
)
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModelConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml


def test_cuda_robot_generator_config():
    robot_file = "franka.yml"
    robot_params = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"][
        "kinematics"
    ]
    config = CudaRobotGeneratorConfig(**robot_params)
    assert config.ee_link == robot_params["ee_link"]


def test_cuda_robot_generator():
    robot_file = "franka.yml"
    robot_params = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"][
        "kinematics"
    ]
    config = CudaRobotGeneratorConfig(**robot_params)
    robot_generator = CudaRobotGenerator(config)
    assert robot_generator.kinematics_config.n_dof == 7


def test_cuda_robot_config():
    robot_file = "franka.yml"
    config = CudaRobotModelConfig.from_robot_yaml_file(robot_file)
    assert config.kinematics_config.n_dof == 7


def test_cuda_robot_generator_config_cspace():
    robot_file = "franka.yml"
    robot_params = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"][
        "kinematics"
    ]
    config = CudaRobotGeneratorConfig(**robot_params)
    assert len(config.cspace.max_jerk) == len(config.cspace.joint_names)
    robot_generator = CudaRobotGenerator(config)

    assert len(robot_generator.cspace.max_jerk) == 7
