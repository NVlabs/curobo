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
"""Example computing collisions using curobo"""
# Third Party
import torch

# CuRobo
from curobo.types.base import TensorDeviceType
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

if __name__ == "__main__":
    robot_file = "franka.yml"
    world_file = "collision_test.yml"
    tensor_args = TensorDeviceType()
    # config = RobotWorldConfig.load_from_config(robot_file, world_file, pose_weight=[10, 200, 1, 10],
    #                                           collision_activation_distance=0.0)
    # curobo_fn = RobotWorld(config)
    robot_file = "franka.yml"
    world_config = {
        "cuboid": {
            "table": {"dims": [2, 2, 0.2], "pose": [0.4, 0.0, 0.3, 1, 0, 0, 0]},
            "cube_1": {"dims": [0.1, 0.1, 0.2], "pose": [0.4, 0.0, 0.5, 1, 0, 0, 0]},
        },
        "mesh": {
            "scene": {
                "pose": [1.5, 0.080, 1.6, 0.043, -0.471, 0.284, 0.834],
                "file_path": "scene/nvblox/srl_ur10_bins.obj",
            }
        },
    }
    tensor_args = TensorDeviceType()
    config = RobotWorldConfig.load_from_config(
        robot_file, world_file, collision_activation_distance=0.0
    )
    curobo_fn = RobotWorld(config)

    q_sph = torch.randn((10, 1, 1, 4), device=tensor_args.device, dtype=tensor_args.dtype)
    q_sph[..., 3] = 0.2
    d = curobo_fn.get_collision_distance(q_sph)
    print(d)

    q_s = curobo_fn.sample(5, mask_valid=False)

    d_world, d_self = curobo_fn.get_world_self_collision_distance_from_joints(q_s)
    print("Collision Distance:")
    print("World:", d_world)
    print("Self:", d_self)
