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
import pytest
import torch

# CuRobo
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig


@pytest.mark.parametrize(
    "b_size",
    [1, 10, 100],
)
def test_multi_pose_franka(b_size: int):
    tensor_args = TensorDeviceType()
    world_file = "collision_cubby.yml"

    robot_file = "franka.yml"
    robot_data = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]

    robot_data["kinematics"]["link_names"] = robot_data["kinematics"]["collision_link_names"]
    robot_cfg = RobotConfig.from_dict(robot_data)

    world_cfg = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_file)))
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=16,
        self_collision_check=True,
        self_collision_opt=True,
        use_cuda_graph=True,
        tensor_args=tensor_args,
        regularization=False,
    )
    ik_solver = IKSolver(ik_config)

    q_sample = ik_solver.sample_configs(b_size)
    kin_state = ik_solver.fk(q_sample)
    link_poses = kin_state.link_pose
    goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)
    result = ik_solver.solve_batch(goal, link_poses=link_poses)

    success = result.success
    assert (
        torch.count_nonzero(success).item() / b_size >= 0.9
    )  # we check if atleast 90% are successful


@pytest.mark.parametrize(
    "b_size",
    [1, 10, 100],
)
def test_multi_pose_hand(b_size: int):
    tensor_args = TensorDeviceType()
    world_file = "collision_table.yml"

    robot_file = "iiwa_allegro.yml"
    robot_data = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]

    robot_cfg = RobotConfig.from_dict(robot_data)

    world_cfg = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_file)))
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=16,
        use_cuda_graph=True,
        tensor_args=tensor_args,
        regularization=False,
    )
    ik_solver = IKSolver(ik_config)

    q_sample = ik_solver.sample_configs(b_size)
    kin_state = ik_solver.fk(q_sample)
    link_poses = kin_state.link_pose
    goal = Pose(kin_state.ee_position, kin_state.ee_quaternion).clone()
    result = ik_solver.solve_batch(goal, link_poses=link_poses)

    success = result.success
    assert (
        torch.count_nonzero(success).item() / b_size >= 0.9
    )  # we check if atleast 90% are successful
