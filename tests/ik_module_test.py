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
from curobo.geom.sdf.world import (
    CollisionCheckerType,
    WorldCollisionConfig,
    WorldPrimitiveCollision,
)
from curobo.geom.sdf.world_mesh import WorldMeshCollision
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig


@pytest.fixture(scope="module")
def ik_solver():
    tensor_args = TensorDeviceType()
    world_file = "collision_cubby.yml"

    robot_file = "franka.yml"
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )
    world_cfg = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_file)))
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=100,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=False,
    )
    ik_solver = IKSolver(ik_config)
    return ik_solver


@pytest.fixture(scope="module")
def ik_solver_batch_env():
    tensor_args = TensorDeviceType()
    world_files = ["collision_table.yml", "collision_cubby.yml", "collision_test.yml"]
    world_cfg = [
        WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_file)))
        for world_file in world_files
    ]
    robot_file = "franka.yml"
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=100,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=False,
    )
    ik_solver = IKSolver(ik_config)
    return ik_solver


def test_ik_single(ik_solver):
    q_sample = ik_solver.sample_configs(1)
    kin_state = ik_solver.fk(q_sample)
    goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)
    result = ik_solver.solve_single(goal)

    success = result.success
    assert success.item()


def test_ik_goalset(ik_solver):
    q_sample = ik_solver.sample_configs(10)
    kin_state = ik_solver.fk(q_sample)
    goal = Pose(kin_state.ee_position.unsqueeze(0), kin_state.ee_quaternion.unsqueeze(0))
    result = ik_solver.solve_goalset(goal)

    assert result.success.item()


def test_ik_batch(ik_solver):
    q_sample = ik_solver.sample_configs(10)
    kin_state = ik_solver.fk(q_sample)
    goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)
    result = ik_solver.solve_batch(goal)
    assert torch.count_nonzero(result.success) > 5


def test_ik_batch_goalset(ik_solver):
    q_sample = ik_solver.sample_configs(100)
    kin_state = ik_solver.fk(q_sample)
    goal = Pose(kin_state.ee_position.view(10, 10, 3), kin_state.ee_quaternion.view(10, 10, 4))
    result = ik_solver.solve_batch_goalset(goal)
    assert torch.count_nonzero(result.success) > 5


def test_ik_batch_env(ik_solver_batch_env):
    q_sample = ik_solver_batch_env.sample_configs(3)
    kin_state = ik_solver_batch_env.fk(q_sample)
    goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)
    result = ik_solver_batch_env.solve_batch_env(goal)

    assert torch.count_nonzero(result.success) >= 1


def test_ik_batch_env_goalset(ik_solver_batch_env):
    q_sample = ik_solver_batch_env.sample_configs(3 * 3)
    kin_state = ik_solver_batch_env.fk(q_sample)
    goal = Pose(kin_state.ee_position.view(3, 3, 3), kin_state.ee_quaternion.view(3, 3, 4))
    result = ik_solver_batch_env.solve_batch_env_goalset(goal)
    assert torch.count_nonzero(result.success) >= 2
