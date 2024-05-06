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

# Third Party
import torch
import pytest
import numpy as np

# CuRobo
from curobo.cuda_robot_model.cuda_robot_generator import (
    CudaRobotGenerator,
    CudaRobotGeneratorConfig,
)
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModelConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml

def unscaled_limits():
    robot_file = "franka.yml"
    robot_params = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"][
        "kinematics"
    ]
    robot_params["cspace"]["velocity_scale"] = 1.0
    robot_params["cspace"]["acceleration_scale"] = 1.0 
    robot_params["cspace"]["jerk_scale"] = 1.0
    config = CudaRobotGeneratorConfig(**robot_params)
    robot_generator = CudaRobotGenerator(config)
    vel_limits = robot_generator.get_joint_limits.velocity
    acc_limits = robot_generator.get_joint_limits.acceleration
    jerk_limits = robot_generator.get_joint_limits.jerk
    return vel_limits, acc_limits, jerk_limits


@pytest.fixture(scope="module")
def unscaled_limits_fixture():
    return unscaled_limits()

@pytest.mark.parametrize("vel_scale, acc_scale, jerk_scale", [
    (np.random.uniform(0.1, 1.0), np.random.uniform(0.1, 1.0), np.random.uniform(0.1, 1.0)) for _ in range(10)
])
def test_cuda_robot_generator_limits(unscaled_limits_fixture, vel_scale, acc_scale, jerk_scale):
    unscaled_vel_limits, unscaled_acc_limits, unscaled_jerk_limits = unscaled_limits_fixture

    robot_file = "franka.yml"
    robot_params = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"][
        "kinematics"
    ]
    robot_params["cspace"]["velocity_scale"] = vel_scale
    robot_params["cspace"]["acceleration_scale"] = acc_scale
    robot_params["cspace"]["jerk_scale"] = jerk_scale
    config = CudaRobotGeneratorConfig(**robot_params)
    robot_generator = CudaRobotGenerator(config)
    assert robot_generator.kinematics_config.n_dof == 7
    print(robot_generator.get_joint_limits.position)
    vel_limits = robot_generator.get_joint_limits.velocity
    acc_limits = robot_generator.get_joint_limits.acceleration
    jerk_limits = robot_generator.get_joint_limits.jerk

    assert torch.equal(vel_limits, vel_scale * unscaled_vel_limits)
    assert torch.equal(acc_limits, acc_scale * unscaled_acc_limits)
    assert torch.equal(jerk_limits, jerk_scale * unscaled_jerk_limits)