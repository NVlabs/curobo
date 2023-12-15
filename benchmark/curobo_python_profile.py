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
import cProfile
import time

# Third Party
import torch
from torch.profiler import ProfilerActivity, profile, record_function

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import get_robot_configs_path, get_robot_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


def demo_motion_gen():
    # Standard Library

    st_time = time.time()
    tensor_args = TensorDeviceType()
    world_file = "collision_table.yml"
    robot_file = "franka.yml"
    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_file,
        world_file,
        tensor_args,
        trajopt_tsteps=44,
        collision_checker_type=CollisionCheckerType.PRIMITIVE,
        use_cuda_graph=False,
        num_trajopt_seeds=24,
        num_graph_seeds=24,
        evaluate_interpolated_trajectory=True,
        interpolation_dt=0.02,
    )
    motion_gen = MotionGen(motion_gen_config)

    # st_time = time.time()
    motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
    print("motion gen time:", time.time() - st_time)

    # print(time.time() - st_time)
    # return
    retract_cfg = motion_gen.get_retract_config()
    print(retract_cfg)
    state = motion_gen.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg.view(1, -1))
    )

    retract_pose = Pose(state.ee_pos_seq.squeeze(), quaternion=state.ee_quat_seq.squeeze())
    start_state = JointState.from_position(retract_cfg.view(1, -1) + 0.3)
    result = motion_gen.plan(
        start_state,
        retract_pose,
        enable_graph=True,
        enable_opt=True,
        max_attempts=1,
        need_graph_success=True,
    )
    print(result.status)
    print(result.optimized_plan.position.shape)
    traj = result.get_interpolated_plan()  # $.position.view(-1, 7)  # optimized plan
    print("Trajectory Generated: ", result.success, result.optimized_dt.item())


def demo_basic_robot():
    st_time = time.time()
    tensor_args = TensorDeviceType()
    # load a urdf:
    config_file = load_yaml(join_path(get_robot_path(), "franka.yml"))

    urdf_file = config_file["robot_cfg"]["kinematics"][
        "urdf_path"
    ]  # Send global path starting with "/"
    base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
    ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
    robot_cfg = RobotConfig.from_basic(urdf_file, base_link, ee_link, tensor_args)

    kin_model = CudaRobotModel(robot_cfg.kinematics)
    print("base kin time:", time.time() - st_time)
    return
    # compute forward kinematics:

    # q = torch.rand((10, kin_model.get_dof()), **vars(tensor_args))
    # out = kin_model.get_state(q)
    # here is the kinematics state:
    # print(out)


def demo_full_config_robot(config_file):
    st_time = time.time()
    tensor_args = TensorDeviceType()
    # load a urdf:

    robot_cfg = RobotConfig.from_dict(config_file, tensor_args)

    # kin_model = CudaRobotModel(robot_cfg.kinematics)
    print("full kin time: ", time.time() - st_time)
    # compute forward kinematics:
    # q = torch.rand((10, kin_model.get_dof()), **vars(tensor_args))
    # out = kin_model.get_state(q)
    # here is the kinematics state:
    # print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        type=str,
        default="log/trace",
        help="path to save file",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="startup_trace",
        help="File name prefix to use to save benchmark results",
    )
    parser.add_argument(
        "--motion_gen",
        action="store_true",
        help="When True, runs startup for motion generation",
        default=False,
    )
    parser.add_argument(
        "--kinematics",
        action="store_true",
        help="When True, runs startup for kinematics",
        default=False,
    )
    parser.add_argument(
        "--motion_gen_once",
        action="store_true",
        help="When True, runs startup for kinematics",
        default=False,
    )
    args = parser.parse_args()

    # cProfile.run('demo_motion_gen()')
    config_file = load_yaml(join_path(get_robot_path(), "franka.yml"))["robot_cfg"]

    # Third Party

    if args.kinematics:
        for _ in range(5):
            demo_full_config_robot(config_file)

        pr = cProfile.Profile()
        pr.enable()
        demo_full_config_robot(config_file)
        pr.disable()
        filename = join_path(args.save_path, args.file_name) + "_kinematics_cprofile.prof"
        pr.dump_stats(filename)

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            demo_full_config_robot(config_file)
        filename = join_path(args.save_path, args.file_name) + "_kinematics_trace.json"
        prof.export_chrome_trace(filename)

    if args.motion_gen_once:
        demo_motion_gen()

    if args.motion_gen:
        for _ in range(5):
            demo_motion_gen()

        pr = cProfile.Profile()
        pr.enable()
        demo_motion_gen()
        pr.disable()
        filename = join_path(args.save_path, args.file_name) + "_motion_gen_cprofile.prof"
        pr.dump_stats(filename)

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            demo_motion_gen()
        filename = join_path(args.save_path, args.file_name) + "_motion_gen_trace.json"
        prof.export_chrome_trace(filename)
