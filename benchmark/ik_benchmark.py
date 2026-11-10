# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark batched IK accuracy and solve time across supported robots."""

import argparse
from pathlib import Path
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


def create_ik_config(
    robot_file: str,
    world_file: Optional[str] = "collision_table.yml",
    batch_size: int = 100,
    use_cuda_graph: bool = True,
    collision_free: bool = True,
    num_seeds: Optional[int] = None,
) -> IKSolverCfg:
    """Build the shared Python and native benchmark IK configuration."""
    if num_seeds is None:
        num_seeds = 2 if not collision_free else (8 if Path(robot_file).stem == "franka" else 16)
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
        override_iters_for_multi_link_ik=(160 if robot_file in ["unitree_g1_29dof.yml"] else None),
        seed_solver_num_seeds=(
            32 if robot_file in ["unitree_g1_29dof.yml"] else max(32, num_seeds * 2)
        ),
        max_batch_size=batch_size,
        load_collision_spheres=collision_free,
    )
    config.use_lm_seed = True

    return config


def sample_ik_configs(sampler: IKSolver, num_samples: int) -> torch.Tensor:
    """Collect exactly [num_samples, dof] valid configurations across bounded draws."""
    accepted = []
    count = 0
    for ratio in range(10, 101, 10):
        samples = sampler.sample_configs(num_samples, rejection_ratio=ratio)
        chunk = samples[: num_samples - count]
        if len(chunk):
            accepted.append(chunk)
            count += len(chunk)
        if count == num_samples:
            return torch.cat(accepted, dim=0)
    raise RuntimeError(f"could not sample {num_samples} valid configurations; collected {count}")


def run_full_config_collision_free_ik(
    robot_file: str,
    world_file: Optional[str],
    batch_size: int,
    use_cuda_graph: bool = False,
    collision_free: bool = True,
    num_seeds: Optional[int] = None,
) -> tuple[float, float, float, float]:
    """Run five timed IK batches for one robot and optimizer pipeline."""
    config = create_ik_config(
        robot_file, world_file, batch_size, use_cuda_graph, collision_free, num_seeds
    )
    ik_solver = IKSolver(config)

    q_sample_list = [sample_ik_configs(ik_solver, batch_size) for _ in range(5)]

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
        time_list.append(total_time)
        success_list.append(success)
        if success > 0.0:
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
    return (
        np.mean(time_list).item(),
        np.mean(success_list).item(),
        np.mean(p_err_list).item() if p_err_list else float("nan"),
        np.mean(q_err_list).item() * 180.0 / np.pi if q_err_list else float("nan"),
    )


def run_benchmark(world_file: Optional[str] = None) -> None:
    """Run the standard LM-seeded L-BFGS IK benchmark."""
    setup_curobo_logger("error")
    runtime.enable_cuda_event_timer = True

    # set seeds
    torch.manual_seed(2)
    np.random.seed(2)

    torch._dynamo.config.compiled_autograd = True

    torch._dynamo.config.cache_size_limit = 64

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
    parser.add_argument("--robot", choices=["unitree_g1_29dof", "dual_ur10e", "franka"])
    parser.add_argument(
        "--num-seeds",
        type=int,
        help="Override the number of L-BFGS seeds in both modes",
    )

    args = parser.parse_args()
    b_list = [100]
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

    robot_list = ["unitree_g1_29dof.yml", "dual_ur10e.yml", "franka.yml"]
    if args.robot is not None:
        robot_list = [args.robot + ".yml"]

    for robot_file in robot_list:
        print("running for robot: ", robot_file)
        if robot_file == "unitree_g1_29dof.yml":
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
                num_seeds=args.num_seeds,
            )

            torch.cuda.empty_cache()
            dt_cu_ik_cfree, success, p_err_c, q_err_c = run_full_config_collision_free_ik(
                robot_file,
                world,
                batch_size=b_size,
                use_cuda_graph=True,
                collision_free=True,
                num_seeds=args.num_seeds,
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
        Path(args.save_path).mkdir(parents=True, exist_ok=True)
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
