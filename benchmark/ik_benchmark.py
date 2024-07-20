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
import argparse
import time

# Third Party
import numpy as np
import torch

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.geom.types import WorldConfig
from curobo.rollout.arm_base import ArmBase, ArmBaseConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import (
    get_motion_gen_robot_list,
    get_multi_arm_robot_list,
    get_robot_configs_path,
    get_robot_list,
    get_task_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
    write_yaml,
)
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

# set seeds
torch.manual_seed(2)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def run_full_config_collision_free_ik(
    robot_file,
    world_file,
    batch_size,
    use_cuda_graph=False,
    collision_free=True,
    high_precision=False,
    num_seeds=12,
):
    tensor_args = TensorDeviceType()
    robot_data = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    if not collision_free:
        robot_data["kinematics"]["collision_link_names"] = None
        robot_data["kinematics"]["lock_joints"] = {}
    robot_data["kinematics"]["collision_sphere_buffer"] = 0.0
    robot_cfg = RobotConfig.from_dict(robot_data)
    world_cfg = WorldConfig.from_dict(load_yaml(join_path(get_world_configs_path(), world_file)))
    position_threshold = 0.005
    grad_iters = None
    if high_precision:
        position_threshold = 0.001
        grad_iters = 100
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        position_threshold=position_threshold,
        num_seeds=num_seeds,
        self_collision_check=collision_free,
        self_collision_opt=collision_free,
        tensor_args=tensor_args,
        use_cuda_graph=use_cuda_graph,
        high_precision=high_precision,
        regularization=False,
        grad_iters=grad_iters,
    )
    ik_solver = IKSolver(ik_config)

    for _ in range(3):
        q_sample = ik_solver.sample_configs(batch_size)
        while q_sample.shape[0] == 0:
            q_sample = ik_solver.sample_configs(batch_size)

        kin_state = ik_solver.fk(q_sample)
        goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)

        st_time = time.time()
        result = ik_solver.solve_batch(goal)
        torch.cuda.synchronize()
        total_time = (time.time() - st_time) / q_sample.shape[0]
    return (
        total_time,
        100.0 * torch.count_nonzero(result.success).item() / len(q_sample),
        # np.mean(result.position_error[result.success].cpu().numpy()).item(),
        np.percentile(result.position_error[result.success].cpu().numpy(), 98).item(),
        np.percentile(result.rotation_error[result.success].cpu().numpy(), 98).item(),
    )


if __name__ == "__main__":
    setup_curobo_logger("error")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="path to save file",
    )
    parser.add_argument(
        "--high_precision",
        action="store_true",
        help="When True, enables IK for 1 mm precision, when False 5mm precision",
        default=False,
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="ik",
        help="File name prefix to use to save benchmark results",
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=16,
        help="Number of seeds to use for IK",
    )
    args = parser.parse_args()

    b_list = [1, 10, 100, 2000][-1:]

    robot_list = get_motion_gen_robot_list()
    world_file = "collision_test.yml"

    print("running...")
    data = {
        "robot": [],
        "IK-time(ms)": [],
        "Collision-Free-IK-time(ms)": [],
        "Batch-Size": [],
        "Success-IK": [],
        "Success-Collision-Free-IK": [],
        "Position-Error(mm)": [],
        "Orientation-Error": [],
        "Position-Error-Collision-Free-IK(mm)": [],
        "Orientation-Error-Collision-Free-IK": [],
    }
    for robot_file in robot_list[:-1]:
        print("running for robot: ", robot_file)
        # create a sampler with dof:
        for b_size in b_list:
            # sample test configs:

            dt_cu_ik, succ, p_err, q_err = run_full_config_collision_free_ik(
                robot_file,
                world_file,
                batch_size=b_size,
                use_cuda_graph=True,
                collision_free=False,
                high_precision=args.high_precision,
                num_seeds=args.num_seeds,
            )
            dt_cu_ik_cfree, success, p_err_c, q_err_c = run_full_config_collision_free_ik(
                robot_file,
                world_file,
                batch_size=b_size,
                use_cuda_graph=True,
                collision_free=True,
                num_seeds=args.num_seeds,
                # high_precision=args.high_precision,
            )
            # print(dt_cu/b_size, dt_cu_cg/b_size)
            data["robot"].append(robot_file)
            data["IK-time(ms)"].append(dt_cu_ik * 1000.0)
            data["Batch-Size"].append(b_size)
            data["Success-Collision-Free-IK"].append(success)
            data["Success-IK"].append(succ)

            data["Position-Error(mm)"].append(p_err * 1000.0)
            data["Orientation-Error"].append(q_err)
            data["Position-Error-Collision-Free-IK(mm)"].append(p_err_c * 1000.0)
            data["Orientation-Error-Collision-Free-IK"].append(q_err_c)

            data["Collision-Free-IK-time(ms)"].append(dt_cu_ik_cfree * 1000.0)

    if args.save_path is not None:
        file_path = join_path(args.save_path, args.file_name)
    else:
        file_path = args.file_name

    write_yaml(data, file_path + ".yml")

    try:
        # Third Party
        import pandas as pd

        df = pd.DataFrame(data)
        print("Reported errors are 98th percentile")
        df.to_csv(file_path + ".csv")
        try:
            # Third Party
            from tabulate import tabulate

            print(tabulate(df, headers="keys", tablefmt="grid"))
        except ImportError:
            print(df)

            pass
    except ImportError:
        pass
