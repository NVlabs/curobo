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
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.trajopt import TrajOptSolver, TrajOptSolverConfig


def trajopt_base_config():
    tensor_args = TensorDeviceType()
    world_file = "collision_table.yml"

    robot_file = "franka.yml"
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )
    world_cfg = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_file)))

    trajopt_config = TrajOptSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        use_cuda_graph=False,
        use_fixed_samples=True,
        n_collision_envs=1,
        collision_cache={"obb": 10},
        seed_ratio={"linear": 0.5, "start": 0.25, "goal": 0.25},
        num_seeds=10,
    )

    return trajopt_config


def trajopt_es_config():
    tensor_args = TensorDeviceType()
    world_file = "collision_table.yml"

    robot_file = "franka.yml"
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )
    world_cfg = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_file)))

    trajopt_config = TrajOptSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        use_cuda_graph=False,
        use_es=True,
        es_learning_rate=0.01,
    )
    return trajopt_config


def trajopt_gd_config():
    tensor_args = TensorDeviceType()
    world_file = "collision_table.yml"

    robot_file = "franka.yml"
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )
    world_cfg = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_file)))

    trajopt_config = TrajOptSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        use_cuda_graph=False,
        use_gradient_descent=True,
        grad_trajopt_iters=500,
    )
    return trajopt_config


def trajopt_no_particle_opt_config():
    tensor_args = TensorDeviceType()
    world_file = "collision_table.yml"

    robot_file = "franka.yml"
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )
    world_cfg = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_file)))

    trajopt_config = TrajOptSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        tensor_args,
        use_cuda_graph=False,
        use_particle_opt=False,
    )
    return trajopt_config


@pytest.mark.parametrize(
    "config,expected",
    [
        (trajopt_base_config(), True),
        (trajopt_es_config(), True),
        (trajopt_gd_config(), True),
        (trajopt_no_particle_opt_config(), True),
    ],
)
def test_eval(config, expected):
    trajopt_solver = TrajOptSolver(config)
    q_start = trajopt_solver.retract_config
    q_goal = q_start.clone() + 0.1
    kin_state = trajopt_solver.fk(q_goal)
    goal_pose = Pose(kin_state.ee_position, kin_state.ee_quaternion)
    goal_state = JointState.from_position(q_goal)
    current_state = JointState.from_position(q_start)
    js_goal = Goal(goal_pose=goal_pose, goal_state=goal_state, current_state=current_state)
    result = trajopt_solver.solve_single(js_goal)

    assert result.success.item() == expected
