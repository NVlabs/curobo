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
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.util.usd_helper import UsdHelper
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig


def demo_motion_gen_api():
    tensor_args = TensorDeviceType(device=torch.device("cuda:0"))

    interpolation_dt = 0.02
    collision_activation_distance = 0.02  # meters
    # create motion gen with a cuboid cache to be able to load obstacles later:
    motion_gen_cfg = MotionGenConfig.load_from_robot_config(
        "franka.yml",
        "collision_table.yml",
        tensor_args,
        trajopt_tsteps=34,
        interpolation_steps=5000,
        num_ik_seeds=50,
        num_trajopt_seeds=6,
        grad_trajopt_iters=500,
        trajopt_dt=0.5,
        interpolation_dt=interpolation_dt,
        evaluate_interpolated_trajectory=True,
        js_trajopt_dt=0.5,
        js_trajopt_tsteps=34,
        collision_activation_distance=collision_activation_distance,
    )
    motion_gen = MotionGen(motion_gen_cfg)

    motion_gen.warmup()

    # create world representation:
    motion_gen.world_coll_checker.clear_cache()
    motion_gen.reset(reset_seed=False)
    cuboids = [Cuboid(name="obs_1", pose=[0.9, 0.0, 0.5, 1, 0, 0, 0], dims=[0.1, 0.5, 0.5])]
    world = WorldConfig(cuboid=cuboids)

    motion_gen.update_world(world)

    q_start = JointState.from_position(
        tensor_args.to_device([[0.0, -1.3, 0.0, -2.5, 0.0, 1.0, 0.0]]),
        joint_names=[
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ],
    )

    goal_pose = Pose(
        position=tensor_args.to_device([[0.5, 0.0, 0.3]]),
        quaternion=tensor_args.to_device([[1, 0, 0, 0]]),
    )

    result = motion_gen.plan_single(q_start, goal_pose)

    if result.success.item():
        # get result:
        # this contains trajectory with 34 tsteps and the final
        # result.optimized_plan
        # result.optimized_dt

        # this contains a linearly interpolated trajectory with fixed dt
        interpolated_solution = result.get_interpolated_plan()

        UsdHelper.write_trajectory_animation_with_robot_usd(
            "franka.yml",
            world,
            q_start,
            interpolated_solution,
            dt=result.interpolation_dt,
            save_path="demo.usd",
            base_frame="/world_base",
        )
    else:
        print("Failed")


if __name__ == "__main__":
    demo_motion_gen_api()
