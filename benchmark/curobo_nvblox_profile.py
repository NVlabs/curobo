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
from typing import Any, Dict, List

# Third Party
import numpy as np
import torch
import torch.autograd.profiler as profiler
from nvblox_torch.datasets.sun3d_dataset import Sun3dDataset
from robometrics.datasets import demo_raw
from torch.profiler import ProfilerActivity, profile, record_function
from tqdm import tqdm

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType, WorldConfig
from curobo.geom.types import Mesh
from curobo.types.math import Pose
from curobo.types.state import JointState
from curobo.util.logger import setup_curobo_logger
from curobo.util_file import (
    get_assets_path,
    get_robot_configs_path,
    get_task_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

# torch.set_num_threads(8)
# ttorch.use_deterministic_algorithms(True)
torch.manual_seed(0)

torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
np.random.seed(10)
# Third Party
from nvblox_torch.datasets.mesh_dataset import MeshDataset

# CuRobo
from curobo.types.camera import CameraObservation


def load_curobo(n_cubes: int, enable_log: bool = False):
    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
    robot_cfg["kinematics"]["collision_sphere_buffer"] = -0.0
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg,
        "collision_nvblox_online.yml",
        trajopt_tsteps=32,
        collision_checker_type=CollisionCheckerType.BLOX,
        use_cuda_graph=False,
        position_threshold=0.005,
        rotation_threshold=0.05,
        num_ik_seeds=30,
        num_trajopt_seeds=12,
        interpolation_dt=0.02,
        store_ik_debug=enable_log,
        store_trajopt_debug=enable_log,
    )
    mg = MotionGen(motion_gen_config)
    mg.warmup(enable_graph=False)
    # print("warmed up")
    # exit()
    return mg


def benchmark_mb(write_usd=False, save_log=False):
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
            mg = load_curobo(n_cubes, save_log)
            m_list = []
            i = 1
            for problem in tqdm(scene_problems, leave=False):
                q_start = problem["start"]
                pose = (
                    problem["goal_pose"]["position_xyz"] + problem["goal_pose"]["quaternion_wxyz"]
                )

                # reset planner
                mg.reset(reset_seed=False)
                world = WorldConfig.from_dict(problem["obstacles"]).get_mesh_world(
                    merge_meshes=True
                )
                # clear cache:
                mesh = world.mesh[0].get_trimesh_mesh()
                mg.clear_world_cache()
                obs = []
                # get camera_observations:
                save_path = "benchmark/log/nvblox/" + key + "_" + str(i)

                m_dataset = Sun3dDataset(save_path)

                # m_dataset = MeshDataset(
                #    None, n_frames=200, image_size=640, save_data_dir=None, trimesh_mesh=mesh
                # )
                obs = []
                tensor_args = mg.tensor_args
                for j in range(len(m_dataset)):
                    with profiler.record_function("nvblox/create_camera_images"):
                        data = m_dataset[j]
                        cam_obs = CameraObservation(
                            rgb_image=tensor_args.to_device(data["rgba"]),
                            depth_image=tensor_args.to_device(data["depth"]),
                            intrinsics=data["intrinsics"],
                            pose=Pose.from_matrix(data["pose"].to(device=mg.tensor_args.device)),
                        )
                        obs.append(cam_obs)
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    for j in range(len(obs)):
                        cam_obs = obs[j]
                        cam_obs.rgb_image = None
                        with profiler.record_function("nvblox/add_camera_images"):
                            mg.add_camera_frame(cam_obs, "world")

                        with profiler.record_function("nvblox/process_camera_images"):
                            mg.process_camera_frames("world", False)
                    mg.world_coll_checker.update_blox_hashes()

                    # run planner
                    start_state = JointState.from_position(mg.tensor_args.to_device([q_start]))
                    result = mg.plan_single(
                        start_state,
                        Pose.from_list(pose),
                        plan_config,
                    )
                print("Exporting the trace..")
                prof.export_chrome_trace("benchmark/log/trace/motion_gen_nvblox.json")
                print(result.success, result.status)
                exit()


def get_metrics_obstacles(obs: Dict[str, List[Any]]):
    obs_list = []
    if "cylinder" in obs and len(obs["cylinder"].items()) > 0:
        for _, vi in enumerate(obs["cylinder"].values()):
            obs_list.append(
                Cylinder(
                    np.ravel(vi["pose"][:3]), vi["radius"], vi["height"], np.ravel(vi["pose"][3:])
                )
            )
    if "cuboid" in obs and len(obs["cuboid"].items()) > 0:
        for _, vi in enumerate(obs["cuboid"].values()):
            obs_list.append(
                Cuboid(np.ravel(vi["pose"][:3]), np.ravel(vi["dims"]), np.ravel(vi["pose"][3:]))
            )
    return obs_list


def check_problems(all_problems):
    n_cube = 0
    for problem in all_problems:
        cache = WorldConfig.from_dict(problem["obstacles"]).get_obb_world().get_cache_dict()
        n_cube = max(n_cube, cache["obb"])
    return n_cube


if __name__ == "__main__":
    setup_curobo_logger("error")
    benchmark_mb()
