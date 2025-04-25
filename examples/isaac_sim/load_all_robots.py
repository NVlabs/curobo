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
parser.add_argument(
    "--visualize_spheres",
    action="store_true",
    help="When True, visualizes robot spheres",
    default=False,
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
from helper import add_extensions, add_robot_to_scene
from omni.isaac.core import World
from omni.isaac.core.objects import sphere

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

# from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig

########### OV #################
from curobo.util.logger import setup_curobo_logger, log_warn
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_motion_gen_robot_list, get_robot_configs_path, join_path, load_yaml


def main():
    log_warn("This example will not work correctly in isaac sim 4.5+")
    list_of_robots = get_motion_gen_robot_list()  # [:2]
    # remove tm12 as meshes don't load correctly in isaac sim 4.5
    if "tm12.yml" in list_of_robots:
        list_of_robots.remove("tm12.yml")
    usd_help = UsdHelper()

    # assuming obstacles are in objects_path:
    my_world = World(stage_units_in_meters=1.0)
    my_world.scene.add_default_ground_plane()

    stage = my_world.stage
    usd_help.load_stage(stage)
    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    stage.DefinePrim("/curobo", "Xform")

    # my_world.stage.SetDefaultPrim(my_world.stage.GetPrimAtPath("/World"))
    stage = my_world.stage
    # stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    # Make a target to follow
    offset_y = 2
    pose = Pose.from_list([0, 0, 0, 1, 0, 0, 0])

    robot_cfg_list = []
    robot_list = []
    tensor_args = TensorDeviceType()
    spheres = []

    for i in range(len(list_of_robots)):
        if i > 0:
            pose.position[0, 1] += offset_y
        if i == int(len(list_of_robots) / 2):
            pose.position[0, 0] = -offset_y
            pose.position[0, 1] = 0
        robot_cfg = load_yaml(join_path(get_robot_configs_path(), list_of_robots[i]))["robot_cfg"]
        robot_cfg_list.append(robot_cfg)
        print("Loading robot: ", list_of_robots[i])
        r = add_robot_to_scene(
            robot_cfg,
            my_world,
            "/World/world_" + str(i) + "/",
            robot_name="/World/world_" + str(i) + "/" "robot_" + str(i),
            position=pose.position[0].cpu().numpy(),
            initialize_world=False,
        )

        robot_list.append(r[0])
        robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)

        kin_model = CudaRobotModel(robot_cfg.kinematics)
        default_config = kin_model.cspace.retract_config

        sph_list = kin_model.get_robot_as_spheres(default_config)
        for si, s in enumerate(sph_list[0]):
            sp = sphere.VisualSphere(
                prim_path="/curobo/robot_sphere_" + str(i) + "_" + str(si),
                position=np.ravel(s.position)
                + pose.position[0].cpu().numpy()
                + np.ravel([0, 0.5, 0.0]),
                radius=float(s.radius),
                color=np.array([0, 0.8, 0.2]),
            )
            spheres.append(sp)

    setup_curobo_logger("warn")

    my_world.initialize_physics()
    my_world.reset()

    add_extensions(simulation_app, args.headless_mode)
    while simulation_app.is_running():
        my_world.step(render=True)

        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            continue
        step_index = my_world.current_time_step_index

        if step_index <= 10:
            for ri, robot in enumerate(robot_list):
                robot._articulation_view.initialize()
                j_names = robot_cfg_list[ri]["kinematics"]["cspace"]["joint_names"]
                default_config = robot_cfg_list[ri]["kinematics"]["cspace"]["retract_config"]

                idx_list = [robot.get_dof_index(x) for x in j_names]
                robot.set_joint_positions(default_config, idx_list)
                robot._articulation_view.set_max_efforts(
                    values=np.array([5000 for i in range(len(idx_list))]), joint_indices=idx_list
                )


if __name__ == "__main__":
    main()
