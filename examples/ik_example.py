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
# Standard Library
import time

# Third Party
import torch

# CuRobo
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def demo_basic_ik():
    tensor_args = TensorDeviceType()

    config_file = load_yaml(join_path(get_robot_configs_path(), "ur10e.yml"))
    urdf_file = config_file["robot_cfg"]["kinematics"][
        "urdf_path"
    ]  # Send global path starting with "/"
    base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
    ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
    robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)

    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        None,
        rotation_threshold=0.05,
        position_threshold=0.005,
        num_seeds=20,
        self_collision_check=False,
        self_collision_opt=False,
        tensor_args=tensor_args,
        use_cuda_graph=True,
    )
    ik_solver = IKSolver(ik_config)

    # print(kin_state)
    for _ in range(10):
        q_sample = ik_solver.sample_configs(100)
        kin_state = ik_solver.fk(q_sample)
        goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)

        st_time = time.time()
        result = ik_solver.solve_batch(goal)
        torch.cuda.synchronize()
        print(
            "Success, Solve Time(s), hz ",
            torch.count_nonzero(result.success).item() / len(q_sample),
            result.solve_time,
            q_sample.shape[0] / (time.time() - st_time),
            torch.mean(result.position_error),
            torch.mean(result.rotation_error),
        )


def demo_full_config_collision_free_ik():
    tensor_args = TensorDeviceType()
    world_file = "collision_cage.yml"

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
        num_seeds=20,
        self_collision_check=True,
        self_collision_opt=True,
        tensor_args=tensor_args,
        use_cuda_graph=True,
        # use_fixed_samples=True,
    )
    ik_solver = IKSolver(ik_config)

    # print(kin_state)
    print("Running Single IK")
    for _ in range(10):
        q_sample = ik_solver.sample_configs(1)
        kin_state = ik_solver.fk(q_sample)
        goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)

        st_time = time.time()
        result = ik_solver.solve_batch(goal)
        torch.cuda.synchronize()
        total_time = (time.time() - st_time) / q_sample.shape[0]
        print(
            "Success, Solve Time(s), Total Time(s)",
            torch.count_nonzero(result.success).item(),
            result.solve_time,
            total_time,
            1.0 / total_time,
            torch.mean(result.position_error) * 100.0,
            torch.mean(result.rotation_error) * 100.0,
        )
    exit()
    print("Running Batch IK (10 goals)")
    q_sample = ik_solver.sample_configs(10)
    kin_state = ik_solver.fk(q_sample)
    goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)

    for _ in range(3):
        st_time = time.time()
        result = ik_solver.solve_batch(goal)
        torch.cuda.synchronize()
        print(
            "Success, Solve Time(s), Total Time(s)",
            torch.count_nonzero(result.success).item() / len(q_sample),
            result.solve_time,
            time.time() - st_time,
        )

    print("Running Goalset IK (10 goals in 1 set)")
    q_sample = ik_solver.sample_configs(10)
    kin_state = ik_solver.fk(q_sample)
    goal = Pose(kin_state.ee_position.unsqueeze(0), kin_state.ee_quaternion.unsqueeze(0))

    for _ in range(3):
        st_time = time.time()
        result = ik_solver.solve_goalset(goal)
        torch.cuda.synchronize()
        print(
            "Success, Solve Time(s), Total Time(s)",
            torch.count_nonzero(result.success).item() / len(result.success),
            result.solve_time,
            time.time() - st_time,
        )

    print("Running Batch Goalset IK (10 goals in 10 sets)")
    q_sample = ik_solver.sample_configs(100)
    kin_state = ik_solver.fk(q_sample)
    goal = Pose(
        kin_state.ee_position.view(10, 10, 3).contiguous(),
        kin_state.ee_quaternion.view(10, 10, 4).contiguous(),
    )

    for _ in range(3):
        st_time = time.time()
        result = ik_solver.solve_batch_goalset(goal)
        torch.cuda.synchronize()
        print(
            "Success, Solve Time(s), Total Time(s)",
            torch.count_nonzero(result.success).item() / len(result.success.view(-1)),
            result.solve_time,
            time.time() - st_time,
        )


def demo_full_config_batch_env_collision_free_ik():
    tensor_args = TensorDeviceType()
    world_file = ["collision_test.yml", "collision_cubby.yml"]

    robot_file = "franka.yml"
    robot_cfg = RobotConfig.from_dict(
        load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    )
    world_cfg = [
        WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), x))) for x in world_file
    ]
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
        # use_fixed_samples=True,
    )
    ik_solver = IKSolver(ik_config)
    q_sample = ik_solver.sample_configs(len(world_file))
    kin_state = ik_solver.fk(q_sample)
    goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)

    print("Running Batch Env IK")
    for _ in range(3):
        st_time = time.time()
        result = ik_solver.solve_batch_env(goal)
        print(result.success)
        torch.cuda.synchronize()
        print(
            "Success, Solve Time(s), Total Time(s)",
            torch.count_nonzero(result.success).item() / len(q_sample),
            result.solve_time,
            time.time() - st_time,
        )

    q_sample = ik_solver.sample_configs(10 * len(world_file))
    kin_state = ik_solver.fk(q_sample)
    goal = Pose(
        kin_state.ee_position.view(len(world_file), 10, 3),
        kin_state.ee_quaternion.view(len(world_file), 10, 4),
    )

    print("Running Batch Env Goalset IK")
    for _ in range(3):
        st_time = time.time()
        result = ik_solver.solve_batch_env_goalset(goal)
        torch.cuda.synchronize()
        print(
            "Success, Solve Time(s), Total Time(s)",
            torch.count_nonzero(result.success).item() / len(result.success.view(-1)),
            result.solve_time,
            time.time() - st_time,
        )


if __name__ == "__main__":
    demo_basic_ik()
    # demo_full_config_collision_free_ik()
    # demo_full_config_batch_env_collision_free_ik()
