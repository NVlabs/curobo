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
from typing import Any, Dict, List

# Third Party
import numpy as np
import torch

# from geometrout.primitive import Cuboid, Cylinder
# from geometrout.transform import SE3
# from robometrics.robot import CollisionSphereConfig, Robot
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType, WorldConfig
from curobo.geom.types import Mesh
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

# torch.set_num_threads(8)
# ttorch.use_deterministic_algorithms(True)
torch.manual_seed(0)

torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
np.random.seed(10)
# Third Party
from robometrics.datasets import demo_raw


def load_curobo(
    n_cubes: int, enable_log: bool = False, mesh_mode: bool = False, cuda_graph: bool = False
):
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
    robot_cfg["kinematics"]["collision_sphere_buffer"] = -0.0
    world_cfg = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    ).get_obb_world()

    c_checker = CollisionCheckerType.PRIMITIVE
    c_cache = {"obb": n_cubes}
    if mesh_mode:
        c_checker = CollisionCheckerType.MESH
        c_cache = {"mesh": n_cubes}
        world_cfg = world_cfg.get_mesh_world()

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        trajopt_tsteps=32,
        collision_checker_type=c_checker,
        use_cuda_graph=cuda_graph,
        collision_cache=c_cache,
        ee_link_name="panda_hand",
        position_threshold=0.005,
        rotation_threshold=0.05,
        num_ik_seeds=30,
        num_trajopt_seeds=12,
        interpolation_dt=0.025,
        finetune_trajopt_iters=200,
        # grad_trajopt_iters=200,
        store_ik_debug=enable_log,
        store_trajopt_debug=enable_log,
    )
    mg = MotionGen(motion_gen_config)
    mg.warmup(enable_graph=False, warmup_js_trajopt=False)
    return mg


def benchmark_mb(args):
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
    spheres = robot_cfg["kinematics"]["collision_spheres"]
    if isinstance(spheres, str):
        spheres = load_yaml(join_path(get_robot_configs_path(), spheres))["collision_spheres"]

    plan_config = MotionGenPlanConfig(
        max_attempts=1,
        enable_graph_attempt=3,
        enable_finetune_trajopt=True,
        partial_ik_opt=False,
        enable_graph=False,
    )
    # load dataset:

    file_paths = [demo_raw]
    all_files = []
    for file_path in file_paths:
        all_groups = []

        problems = file_path()

        for key, v in tqdm(problems.items()):
            # if key not in ["table_under_pick_panda"]:
            #    continue
            scene_problems = problems[key]  # [:2]
            n_cubes = check_problems(scene_problems)
            mg = load_curobo(n_cubes, False, args.mesh, args.cuda_graph)
            m_list = []
            i = 0
            for problem in tqdm(scene_problems, leave=False):
                q_start = problem["start"]

                pose = (
                    problem["goal_pose"]["position_xyz"] + problem["goal_pose"]["quaternion_wxyz"]
                )

                # reset planner
                mg.reset(reset_seed=False)
                if args.mesh:
                    world = WorldConfig.from_dict(problem["obstacles"]).get_mesh_world()

                else:
                    world = WorldConfig.from_dict(problem["obstacles"]).get_obb_world()

                mg.update_world(world)
                start_state = JointState.from_position(mg.tensor_args.to_device([q_start]))

                result = mg.plan_single(
                    start_state,
                    Pose.from_list(pose),
                    plan_config,
                )
                print(result.total_time, result.solve_time)
                # continue
                # load obstacles
                # exit()
                # run planner
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    # torch.cuda.profiler.start()
                    result = mg.plan_single(
                        start_state,
                        Pose.from_list(pose),
                        plan_config,
                    )
                    # torch.cuda.profiler.stop()

                print("Exporting the trace..")
                prof.export_chrome_trace(join_path(args.save_path, args.file_name) + ".json")
                print(result.success, result.status)
                exit()


def check_problems(all_problems):
    n_cube = 0
    for problem in all_problems:
        cache = WorldConfig.from_dict(problem["obstacles"]).get_obb_world().get_cache_dict()
        n_cube = max(n_cube, cache["obb"])
    return n_cube


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="benchmark/log/trace",
        help="path to save file",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="motion_gen_trace",
        help="File name prefix to use to save benchmark results",
    )
    parser.add_argument(
        "--mesh",
        action="store_true",
        help="When True, converts obstacles to meshes",
        default=False,
    )
    parser.add_argument(
        "--cuda_graph",
        action="store_true",
        help="When True, uses cuda graph during profiing",
        default=False,
    )

    args = parser.parse_args()

    setup_curobo_logger("warn")
    benchmark_mb(args)
