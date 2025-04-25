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

try:
    # Third Party
    import isaacsim
except ImportError:
    pass


# Third Party
import torch

a = torch.zeros(4, device="cuda:0")

# Standard Library

# Standard Library
import argparse

# Third Party
from omni.isaac.kit import SimulationApp

parser = argparse.ArgumentParser()

parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)
args = parser.parse_args()

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)
# Third Party
import carb
import numpy as np
from helper import add_extensions
from omni.isaac.core import World

try:
    from omni.isaac.core.materials import OmniPBR
except ImportError:
    from isaacsim.core.api.materials import OmniPBR


from omni.isaac.core.objects import sphere

# CuRobo
# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.util.logger import setup_curobo_logger
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_world_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

########### OV #################


def main():
    usd_help = UsdHelper()
    act_distance = 0.2

    n_envs = 2
    # assuming obstacles are in objects_path:
    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane()

    stage = my_world.stage
    usd_help.load_stage(stage)
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")

    stage = my_world.stage
    target_list = []
    target_material_list = []
    offset_x = 3.5
    radius = 0.1
    pose = Pose.from_list([0, 0, 0, 1, 0, 0, 0])

    for i in range(n_envs):
        if i > 0:
            pose.position[0, 0] += offset_x
        usd_help.add_subroot("/World", "/World/world_" + str(i), pose)

        target_material = OmniPBR("/World/looks/t_" + str(i), color=np.array([0, 1, 0]))

        target = sphere.VisualSphere(
            "/World/world_" + str(i) + "/target",
            position=np.array([0.5, 0, 0.5]) + pose.position[0].cpu().numpy(),
            orientation=np.array([1, 0, 0, 0]),
            radius=radius,
            visual_material=target_material,
        )
        target_list.append(target)
        target_material_list.append(target_material)

    setup_curobo_logger("warn")

    # warmup curobo instance

    tensor_args = TensorDeviceType()
    robot_file = "franka.yml"
    world_file = ["collision_thin_walls.yml", "collision_test.yml"]
    world_cfg_list = []
    for i in range(n_envs):
        world_cfg = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), world_file[i]))
        )  # .get_mesh_world()
        world_cfg.objects[0].pose[2] += 0.1
        world_cfg.randomize_color(r=[0.2, 0.3], b=[0.0, 0.05], g=[0.2, 0.3])
        usd_help.add_world_to_stage(world_cfg, base_frame="/World/world_" + str(i))
        world_cfg_list.append(world_cfg)
    config = RobotWorldConfig.load_from_config(
        robot_file, world_cfg_list, collision_activation_distance=act_distance
    )
    model = RobotWorld(config)
    i = 0
    max_distance = 0.5
    x_sph = torch.zeros((n_envs, 1, 1, 4), device=tensor_args.device, dtype=tensor_args.dtype)
    x_sph[..., 3] = radius
    env_query_idx = torch.arange(n_envs, device=tensor_args.device, dtype=torch.int32)
    add_extensions(simulation_app, args.headless_mode)
    while simulation_app.is_running():
        my_world.step(render=True)
        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            continue
        step_index = my_world.current_time_step_index

        if step_index == 0:
            my_world.reset()

        if step_index < 20:
            continue
        sp_buffer = []
        for k in target_list:
            sph_position, _ = k.get_local_pose()
            sp_buffer.append(sph_position)

        x_sph[..., :3] = tensor_args.to_device(sp_buffer).view(n_envs, 1, 1, 3)

        d, d_vec = model.get_collision_vector(x_sph, env_query_idx=env_query_idx)

        d = d.view(-1).cpu()

        for i in range(d.shape[0]):
            p = d[i].item()
            p = max(1, p * 5)
            if d[i].item() == 0.0:
                target_material_list[i].set_color(np.ravel([0, 1, 0]))
            elif d[i].item() <= model.contact_distance:
                target_material_list[i].set_color(np.array([0, 0, p]))
            elif d[i].item() >= model.contact_distance:
                target_material_list[i].set_color(np.array([p, 0, 0]))


if __name__ == "__main__":
    main()
