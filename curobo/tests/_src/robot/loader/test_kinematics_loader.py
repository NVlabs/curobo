# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# CuRobo
from curobo._src.robot.kinematics.kinematics import KinematicsCfg
from curobo._src.robot.loader.kinematics_loader import (
    KinematicsLoader,
    KinematicsLoaderCfg,
)
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


def test_cuda_robot_generator_config():
    robot_file = "franka.yml"
    robot_params = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"][
        "kinematics"
    ]
    config = KinematicsLoaderCfg(**robot_params)
    # Test that tool_frames is set
    assert config.tool_frames is not None
    assert len(config.tool_frames) > 0


def test_cuda_robot_generator():
    robot_file = "franka.yml"
    robot_params = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"][
        "kinematics"
    ]
    config = KinematicsLoaderCfg(**robot_params)
    robot_generator = KinematicsLoader(config)
    assert robot_generator.kinematics_config.num_dof == 7


def test_cuda_robot_config():
    robot_file = "franka.yml"
    config = KinematicsCfg.from_robot_yaml_file(robot_file)
    assert config.kinematics_config.num_dof == 7


def test_cuda_robot_generator_config_cspace():
    robot_file = "franka.yml"
    robot_params = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"][
        "kinematics"
    ]
    config = KinematicsLoaderCfg(**robot_params)
    assert len(config.cspace.max_jerk) == len(config.cspace.joint_names)
    robot_generator = KinematicsLoader(config)

    assert len(robot_generator.cspace.max_jerk) == 7
