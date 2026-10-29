# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark batched IK accuracy and solve time across supported robots."""

import argparse
from typing import Optional

import numpy as np
import torch

import curobo.runtime as runtime
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.cuda_event_timer import CudaEventTimer
from curobo._src.util.logging import setup_curobo_logger
from curobo._src.util_file import join_path, write_yaml

runtime.enable_cuda_event_timer = True

# set seeds
torch.manual_seed(2)
np.random.seed(2)

torch._dynamo.config.compiled_autograd = True


torch._dynamo.config.cache_size_limit = 64


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def run_full_config_collision_free_ik(
    robot_file: str,
    world_file: Optional[str],
    batch_size: int,
    use_cuda_graph: bool = False,
    collision_free: bool = True,
    num_seeds: int = 12,
) -> tuple[float, float, float, float]:
    """Run five timed IK batches for one robot and optimizer pipeline."""
    device_cfg = DeviceCfg()
    position_threshold = 0.005

    # Configure optimizer files
    optimizer_config_yamls = ["ik/lbfgs_ik.yml"]

    config = IKSolverCfg.create(
        robot=robot_file,
        optimizer_configs=optimizer_config_yamls,
        metrics_rollout="metrics_base.yml",
        transition_model="ik/transition_ik.yml",
        scene_model=world_file if collision_free else None,
        use_cuda_graph=use_cuda_graph,
        num_seeds=num_seeds,
        position_tolerance=position_threshold,
        optimizer_collision_activation_distance=0.0025,
        self_collision_check=collision_free,
        device_cfg=device_cfg,
        override_iters_for_multi_link_ik=(
            160 if robot_file in ["unitree_g1_29dof_retarget.yml"] else None
        ),
        seed_solver_num_seeds=(
            32 if robot_file in ["unitree_g1_29dof_retarget.yml"] else max(32, num_seeds * 2)
        ),
        max_batch_size=batch_size,
        load_collision_spheres=collision_free,
    )
    config.use_lm_seed = True

    ik_solver = IKSolver(config)

    rej_ratio = 10
    q_sample_list = []
    for _ in range(5):
        q_sample = ik_solver.sample_configs(batch_size, rejection_ratio=rej_ratio)
        while q_sample.shape[0] == 0 or q_sample.shape[0] < batch_size:
            q_sample = ik_solver.sample_configs(batch_size, rejection_ratio=rej_ratio)
            rej_ratio = int(1.2 * rej_ratio)
            if rej_ratio > 100:
                print("Rejection ratio too high")
                exit()
        q_sample_list.append(q_sample)

    time_list = []
    success_list = []
    p_err_list = []
    q_err_list = []
    torch.cuda.empty_cache()

    # warmup:
    ik_solver.config.exit_early = False
    for i in range(3):
        ik_solver.reset_seed()
        q_sample = q_sample_list[0]
        kin_state = ik_solver.compute_kinematics(JointState.from_position(q_sample))
        goal_tool_poses = kin_state.tool_poses.as_goal()
        result = ik_solver.solve_pose(
            goal_tool_poses=goal_tool_poses,
            seed_config=None,
        )
    ik_solver.config.exit_early = True
    for i in range(len(q_sample_list)):
        ik_solver.reset_seed()
        q_sample = q_sample_list[i]
        kin_state = ik_solver.compute_kinematics(JointState.from_position(q_sample))
        goal_tool_poses = kin_state.tool_poses.as_goal()
        seed_config = None
        timer = CudaEventTimer().start()

        result = ik_solver.solve_pose(
            goal_tool_poses=goal_tool_poses,
            seed_config=seed_config,
        )
        total_time = timer.stop()
        success = 100.0 * torch.count_nonzero(result.success).item() / len(q_sample)
        if success > 0.0:
            time_list.append(total_time)
            success_list.append(success)
            p_err_list.append(
                np.percentile(
                    result.position_error[result.success.view(-1)].cpu().numpy(), 90
                ).item()
            )
            q_err_list.append(
                np.percentile(
                    result.rotation_error[result.success.view(-1)].cpu().numpy(), 90
                ).item()
            )
    if len(time_list) == 0:
        return (0, 0, 100, 100)
    return (
        np.mean(time_list).item(),
        np.mean(success_list).item(),
        np.mean(p_err_list).item(),
        np.mean(q_err_list).item() * 180.0 / np.pi,
    )



def run_benchmark(world_file: Optional[str] = None) -> None:
    """Run the standard LM-seeded L-BFGS IK benchmark."""
    setup_curobo_logger("error")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="path to save file",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="curobo_ik",
        help="File name prefix to use to save benchmark results",
    )

    args = parser.parse_args()
    b_list = [100]
    num_seeds = 2
    collision_free_num_seeds = 8
    if world_file is None:
        world_file = "collision_table.yml"

    data = {
        "robot": [],
        "IK-time(ms)": [],
        "Batch-Size": [],
        "Success-IK": [],
        "Position-Error(mm)": [],
        "Orientation-Error(deg)": [],
        "C-Free-IK-time(ms)": [],
        "Success-C-Free-IK": [],
        "Position-Error-C-Free-IK(mm)": [],
        "Orientation-Error-C-Free-IK(deg)": [],
    }

    robot_list = ["unitree_g1_29dof_retarget.yml", "dual_ur10e.yml", "franka.yml"]

    for robot_file in robot_list:
        print("running for robot: ", robot_file)
        if robot_file == "unitree_g1_29dof_retarget.yml":
            world = "collision_table.yml"
        else:
            world = world_file

        for b_size in b_list:
            print(robot_file, b_size)

            torch.cuda.empty_cache()
            dt_cu_ik, succ, p_err, q_err = run_full_config_collision_free_ik(
                robot_file,
                world,
                batch_size=b_size,
                use_cuda_graph=True,
                collision_free=False,
                num_seeds=num_seeds,
            )

            torch.cuda.empty_cache()
            cfree_num_seeds = collision_free_num_seeds
            if robot_file in ["unitree_g1_29dof_retarget.yml", "dual_ur10e.yml"]:
                cfree_num_seeds = max(16, collision_free_num_seeds)

            dt_cu_ik_cfree, success, p_err_c, q_err_c = run_full_config_collision_free_ik(
                robot_file,
                world,
                batch_size=b_size,
                use_cuda_graph=True,
                collision_free=True,
                num_seeds=cfree_num_seeds,
            )
            data["Success-C-Free-IK"].append(success)

            data["Position-Error-C-Free-IK(mm)"].append(p_err_c * 1000.0)
            data["Orientation-Error-C-Free-IK(deg)"].append(q_err_c)

            data["C-Free-IK-time(ms)"].append(dt_cu_ik_cfree * 1000.0)

            data["robot"].append(robot_file)
            data["IK-time(ms)"].append(dt_cu_ik * 1000.0)
            data["Batch-Size"].append(b_size)
            data["Success-IK"].append(succ)

            data["Position-Error(mm)"].append(p_err * 1000.0)
            data["Orientation-Error(deg)"].append(q_err)
            torch.cuda.empty_cache()

    if args.save_path is not None:
        file_path = join_path(args.save_path, args.file_name)
    else:
        file_path = args.file_name
    write_yaml(data, file_path + ".yml")

    try:
        # Third Party
        import pandas as pd

        df = pd.DataFrame(data)
        try:
            # Third Party
            from tabulate import tabulate

            print(tabulate(df, headers="keys", tablefmt="grid"))
        except ImportError:
            print(df)

            pass
    except ImportError:
        pass


if __name__ == "__main__":
    run_benchmark()
