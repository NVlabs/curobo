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

# CuRobo
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig


def load_robot_world():
    robot_file = "franka.yml"
    world_file = "collision_table.yml"
    config = RobotWorldConfig.load_from_config(
        robot_file, world_file, collision_activation_distance=0.2
    )
    model = RobotWorld(config)
    return model


def load_robot_batch_world():
    robot_file = "franka.yml"
    world_file = ["collision_table.yml", "collision_test.yml"]
    config = RobotWorldConfig.load_from_config(
        robot_file, world_file, collision_activation_distance=0.0
    )
    model = RobotWorld(config)
    return model


def test_robot_world_config_load():
    # test if curobo robot world can be loaded
    load_robot_world()
    assert True


def test_robot_batch_world_config_load():
    # test if curobo robot world can be loaded
    load_robot_batch_world()
    assert True


def test_robot_world_kinematics():
    # test if kinematics can be queried
    model = load_robot_world()
    n = 10
    q = model.sample(n)
    state = model.get_kinematics(q)
    assert state.ee_position.shape[-1] == 3


def test_robot_world_sample():
    # test if joint configurations can be sampled
    model = load_robot_world()
    n = 10
    q = model.sample(n)
    assert q.shape[0] == n


def test_robot_world_collision():
    # test computing collisions given robot joint configurations
    model = load_robot_world()
    n = 10
    q = model.sample(n)
    state = model.get_kinematics(q)
    d_world = model.get_collision_constraint(state.link_spheres_tensor.view(n, 1, -1, 4))
    assert d_world.shape[0] == n
    assert torch.sum(d_world) == 0.0


def test_robot_world_collision_vector():
    # test computing collisions given robot joint configurations
    model = load_robot_world()
    n = 10
    q = model.sample(n)
    state = model.get_kinematics(q)
    d_world, d_vec = model.get_collision_vector(state.link_spheres_tensor.view(n, 1, -1, 4))

    assert d_world.shape[0] == n
    assert torch.norm(d_world[0] - 0.1385) < 0.005
    assert torch.abs(d_vec[0, 0, 0, 2] + 0.7350) > 0.002


def test_robot_world_self_collision():
    model = load_robot_world()
    n = 10
    q = model.sample(n)
    state = model.get_kinematics(q)
    d_self = model.get_self_collision(state.link_spheres_tensor.view(n, 1, -1, 4))
    assert torch.sum(d_self) == 0.0


def test_robot_world_pose():
    model = load_robot_world()
    n = 10
    q = model.sample(n)
    state = model.get_kinematics(q)
    pose = state.ee_pose
    des_pose = state.ee_pose
    d = model.pose_distance(des_pose, pose)
    assert torch.sum(d) < 1e-3


def test_trajectory_sample():
    model = load_robot_world()
    b = 10
    horizon = 21
    q = model.sample_trajectory(b, horizon)
    assert q.shape[0] == b and q.shape[1] == horizon


def test_batch_trajectory_sample():
    model = load_robot_batch_world()
    b = 2
    horizon = 21
    env_query_idx = torch.zeros((b), dtype=torch.int32, device=torch.device("cuda", 0))
    q = model.sample_trajectory(b, horizon, env_query_idx=env_query_idx)
    assert q.shape[0] == b and q.shape[1] == horizon


def test_batch_trajectory_1env_sample():
    model = load_robot_batch_world()
    b = 2
    horizon = 21
    env_query_idx = None
    q = model.sample_trajectory(b, horizon, env_query_idx=env_query_idx)
    assert q.shape[0] == b and q.shape[1] == horizon


def test_robot_batch_world_collision():
    # test computing collisions given robot joint configurations
    model = load_robot_batch_world()
    b = 2
    horizon = 21
    env_query_idx = torch.zeros((b), dtype=torch.int32, device=torch.device("cuda", 0))

    q = model.sample_trajectory(b, horizon, env_query_idx=env_query_idx)

    d_world, d_self = model.get_world_self_collision_distance_from_joint_trajectory(
        q, env_query_idx
    )
    assert d_world.shape[0] == b
    assert torch.sum(d_world) == 0.0
