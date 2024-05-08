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
from copy import deepcopy
from typing import Optional

# Third Party
import numpy as np
import torch
from robometrics.datasets import demo_raw, motion_benchmaker_raw, mpinets_raw
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType, WorldConfig
from curobo.geom.types import Cuboid
from curobo.geom.types import Cuboid as curobo_Cuboid
from curobo.geom.types import Mesh, VoxelGrid
from curobo.types.base import TensorDeviceType
from curobo.types.camera import CameraObservation
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util.metrics import CuroboGroupMetrics, CuroboMetrics
from curobo.util_file import (
    get_assets_path,
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
    write_yaml,
)
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

torch.manual_seed(0)

torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

np.random.seed(0)


def load_curobo(
    n_cubes: int,
    enable_debug: bool = False,
    tsteps: int = 30,
    trajopt_seeds: int = 4,
    mpinets: bool = False,
    graph_mode: bool = False,
    cuda_graph: bool = True,
    collision_activation_distance: float = 0.025,
    finetune_dt_scale: float = 1.0,
    parallel_finetune: bool = True,
):
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
    robot_cfg["kinematics"]["collision_sphere_buffer"] = 0.0

    ik_seeds = 30
    if graph_mode:
        trajopt_seeds = 4
    if trajopt_seeds >= 14:
        ik_seeds = max(100, trajopt_seeds * 2)
    if mpinets:
        robot_cfg["kinematics"]["lock_joints"] = {
            "panda_finger_joint1": 0.025,
            "panda_finger_joint2": 0.025,
        }
    world_cfg = WorldConfig.from_dict(
        {
            "voxel": {
                "base": {
                    "dims": [2.0, 2.0, 3.0],
                    "pose": [0, 0, 0, 1, 0, 0, 0],
                    "voxel_size": 0.01,
                    "feature_dtype": torch.float8_e4m3fn,
                },
            }
        }
    )
    interpolation_steps = 2000
    if graph_mode:
        interpolation_steps = 100
    robot_cfg_instance = RobotConfig.from_dict(robot_cfg, tensor_args=TensorDeviceType())
    K = robot_cfg_instance.kinematics.kinematics_config.joint_limits
    K.position[0, :] -= 0.2
    K.position[1, :] += 0.2

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg_instance,
        world_cfg,
        trajopt_tsteps=tsteps,
        collision_checker_type=CollisionCheckerType.VOXEL,
        use_cuda_graph=cuda_graph,
        position_threshold=0.005,  # 0.5 cm
        rotation_threshold=0.05,
        num_ik_seeds=ik_seeds,
        num_graph_seeds=trajopt_seeds,
        num_trajopt_seeds=trajopt_seeds,
        interpolation_dt=0.025,
        store_ik_debug=enable_debug,
        store_trajopt_debug=enable_debug,
        interpolation_steps=interpolation_steps,
        collision_activation_distance=collision_activation_distance,
        trajopt_dt=0.25,
        finetune_dt_scale=finetune_dt_scale,
        maximum_trajectory_dt=0.16,
        finetune_trajopt_iters=300,
    )
    mg = MotionGen(motion_gen_config)
    mg.warmup(enable_graph=True, warmup_js_trajopt=False, parallel_finetune=True)
    # create a ground truth collision checker:
    config = RobotWorldConfig.load_from_config(
        robot_cfg_instance,
        "collision_table.yml",
        collision_activation_distance=0.0,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        n_cuboids=50,
    )
    robot_world = RobotWorld(config)

    return mg, robot_cfg, robot_world


def benchmark_mb(
    write_usd=False,
    save_log=False,
    write_plot=False,
    write_benchmark=False,
    plot_cost=False,
    override_tsteps: Optional[int] = None,
    args=None,
):
    # load dataset:
    graph_mode = args.graph
    interpolation_dt = 0.02
    file_paths = [demo_raw, motion_benchmaker_raw, mpinets_raw][:1]

    enable_debug = save_log or plot_cost
    all_files = []
    og_tsteps = 32
    if override_tsteps is not None:
        og_tsteps = override_tsteps

    og_trajopt_seeds = 12
    og_collision_activation_distance = 0.01  # 0.03
    if args.graph:
        og_trajopt_seeds = 4
    for file_path in file_paths:
        all_groups = []
        mpinets_data = False
        problems = file_path()

        for key, v in tqdm(problems.items()):
            scene_problems = problems[key]
            m_list = []
            i = -1
            ik_fail = 0
            trajopt_seeds = og_trajopt_seeds
            tsteps = og_tsteps
            collision_activation_distance = og_collision_activation_distance
            finetune_dt_scale = 1.0
            parallel_finetune = True
            if "cage_panda" in key:
                trajopt_seeds = 16
                finetune_dt_scale = 0.95
                parallel_finetune = True
            if "table_under_pick_panda" in key:
                tsteps = 36
                trajopt_seeds = 16
                finetune_dt_scale = 0.95
                parallel_finetune = True
                # collision_activation_distance = 0.015

            if "table_pick_panda" in key:
                collision_activation_distance = 0.005

            if "cubby_task_oriented" in key:  # and "merged" not in key:
                trajopt_seeds = 16
                finetune_dt_scale = 0.95
                collision_activation_distance = 0.005
                parallel_finetune = True
            if "dresser_task_oriented" in key:
                trajopt_seeds = 16
                finetune_dt_scale = 0.95
                parallel_finetune = True
            if key in [
                "tabletop_neutral_start",
                "merged_cubby_neutral_start",
                "cubby_neutral_start",
                "cubby_neutral_goal",
                "dresser_neutral_start",
                "tabletop_task_oriented",
            ]:
                collision_activation_distance = 0.005
            if "dresser_task_oriented" in list(problems.keys()):
                mpinets_data = True

            mg, robot_cfg, robot_world = load_curobo(
                0,
                enable_debug,
                tsteps,
                trajopt_seeds,
                mpinets_data,
                graph_mode,
                not args.disable_cuda_graph,
                collision_activation_distance=collision_activation_distance,
                finetune_dt_scale=finetune_dt_scale,
                parallel_finetune=parallel_finetune,
            )
            for problem in tqdm(scene_problems, leave=False):
                i += 1
                if problem["collision_buffer_ik"] < 0.0:
                    continue

                plan_config = MotionGenPlanConfig(
                    max_attempts=10,
                    enable_graph_attempt=1,
                    enable_finetune_trajopt=True,
                    partial_ik_opt=False,
                    enable_graph=graph_mode,
                    timeout=60,
                    enable_opt=not graph_mode,
                    parallel_finetune=True,
                )

                q_start = problem["start"]
                pose = (
                    problem["goal_pose"]["position_xyz"] + problem["goal_pose"]["quaternion_wxyz"]
                )

                # reset planner
                mg.reset(reset_seed=False)
                world_coll = WorldConfig.from_dict(problem["obstacles"]).get_obb_world()

                robot_world.update_world(world_coll)

                esdf = robot_world.world_model.get_esdf_in_bounding_box(
                    Cuboid(name="base", pose=[0, 0, 0, 1, 0, 0, 0], dims=[2, 2, 3]), voxel_size=0.01
                )
                world_voxel_collision = mg.world_coll_checker
                world_voxel_collision.update_voxel_data(esdf)

                start_state = JointState.from_position(mg.tensor_args.to_device([q_start]))
                for _ in range(2):
                    result = mg.plan_single(
                        start_state,
                        Pose.from_list(pose),
                        plan_config,
                    )
                print("Profiling...")
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    torch.cuda.profiler.start()

                    result = mg.plan_single(
                        start_state,
                        Pose.from_list(pose),
                        plan_config,
                    )
                    torch.cuda.profiler.stop()

                print("Exporting the trace..")
                prof.export_chrome_trace("benchmark/log/trace/motion_gen_voxel.json")

                exit()


if __name__ == "__main__":
    setup_curobo_logger("error")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--graph",
        action="store_true",
        help="When True, runs only geometric planner",
        default=False,
    )
    parser.add_argument(
        "--disable_cuda_graph",
        action="store_true",
        help="When True, disable cuda graph during benchmarking",
        default=False,
    )
    args = parser.parse_args()
    benchmark_mb(
        save_log=False,
        write_usd=False,
        write_plot=False,
        write_benchmark=False,
        plot_cost=False,
        args=args,
    )
