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
# Third Party
import cv2
import numpy as np
import torch
from matplotlib import cm
from nvblox_torch.datasets.realsense_dataset import RealsenseDataloader
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
        "width": "1920",
        "height": "1080",
    }
)
# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.camera import CameraObservation
from curobo.types.math import Pose
from curobo.util_file import get_world_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

simulation_app.update()
# Standard Library
import argparse

# Third Party
from omni.isaac.core import World
from omni.isaac.core.materials import OmniPBR
from omni.isaac.core.objects import cuboid, sphere

parser = argparse.ArgumentParser()

parser.add_argument(
    "--show-window",
    action="store_true",
    help="When True, shows camera image in a CV window",
    default=False,
)
args = parser.parse_args()


def draw_points(voxels):
    # Third Party

    # Third Party
    try:
        from omni.isaac.debug_draw import _debug_draw
    except ImportError:
        from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
    # if draw.get_num_points() > 0:
    draw.clear_points()
    if len(voxels) == 0:
        return

    jet = cm.get_cmap("plasma").reversed()

    cpu_pos = voxels[..., :3].view(-1, 3).cpu().numpy()
    z_val = cpu_pos[:, 1]
    # add smallest and largest values:
    # z_val = np.append(z_val, 1.0)
    # z_val = np.append(z_val,0.4)
    # scale values
    # z_val += 0.4
    # z_val[z_val>1.0] = 1.0
    # z_val = 1.0/z_val
    # z_val = z_val/1.5
    # z_val[z_val!=z_val] = 0.0
    # z_val[z_val==0.0] = 0.4

    jet_colors = jet(z_val)

    b, _ = cpu_pos.shape
    point_list = []
    colors = []
    for i in range(b):
        # get list of points:
        point_list += [(cpu_pos[i, 0], cpu_pos[i, 1], cpu_pos[i, 2])]
        colors += [(jet_colors[i][0], jet_colors[i][1], jet_colors[i][2], 1.0)]
    sizes = [10.0 for _ in range(b)]

    draw.draw_points(point_list, colors, sizes)


def clip_camera(camera_data):
    # clip camera image to bounding box:
    h_ratio = 0.15
    w_ratio = 0.15
    depth = camera_data["raw_depth"]
    depth_tensor = camera_data["depth"]
    h, w = depth_tensor.shape
    depth[: int(h_ratio * h), :] = 0.0
    depth[int((1 - h_ratio) * h) :, :] = 0.0
    depth[:, : int(w_ratio * w)] = 0.0
    depth[:, int((1 - w_ratio) * w) :] = 0.0

    depth_tensor[: int(h_ratio * h), :] = 0.0
    depth_tensor[int(1 - h_ratio * h) :, :] = 0.0
    depth_tensor[:, : int(w_ratio * w)] = 0.0
    depth_tensor[:, int(1 - w_ratio * w) :] = 0.0


def draw_line(start, gradient):
    # Third Party
    try:
        from omni.isaac.debug_draw import _debug_draw
    except ImportError:
        from isaacsim.util.debug_draw import _debug_draw

    draw = _debug_draw.acquire_debug_draw_interface()
    # if draw.get_num_points() > 0:
    draw.clear_lines()
    start_list = [start]
    end_list = [start + gradient]

    colors = [(0.0, 0, 0.8, 0.9)]

    sizes = [10.0]
    draw.draw_lines(start_list, end_list, colors, sizes)


if __name__ == "__main__":
    radius = 0.05
    act_distance = 0.4
    my_world = World(stage_units_in_meters=1.0)
    stage = my_world.stage
    my_world.scene.add_default_ground_plane()
    # my_world.scene.add_ground_plane(color=np.array([0.2,0.2,0.2]))

    xform = stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(xform)
    target_material = OmniPBR("/World/looks/t", color=np.array([0, 1, 0]))

    target = sphere.VisualSphere(
        "/World/target",
        position=np.array([0.0, 0, 0.5]),
        orientation=np.array([1, 0, 0, 0]),
        radius=radius,
        visual_material=target_material,
    )

    # Make a target to follow
    camera_marker = cuboid.VisualCuboid(
        "/World/camera_nvblox",
        position=np.array([0.0, -0.1, 0.25]),
        orientation=np.array([0.843, -0.537, 0.0, 0.0]),
        color=np.array([0.1, 0.1, 0.5]),
        size=0.03,
    )
    collision_checker_type = CollisionCheckerType.BLOX
    world_cfg = WorldConfig.from_dict(
        {
            "blox": {
                "world": {
                    "pose": [0, 0, 0, 1, 0, 0, 0],
                    "integrator_type": "occupancy",
                    "voxel_size": 0.03,
                }
            }
        }
    )

    config = RobotWorldConfig.load_from_config(
        "franka.yml",
        world_cfg,
        collision_activation_distance=act_distance,
        collision_checker_type=collision_checker_type,
    )

    model = RobotWorld(config)

    realsense_data = RealsenseDataloader(clipping_distance_m=1.0)
    data = realsense_data.get_data()

    camera_pose = Pose.from_list([0, 0, 0, 0.707, 0.707, 0, 0])
    i = 0
    tensor_args = TensorDeviceType()
    x_sph = torch.zeros((1, 1, 1, 4), device=tensor_args.device, dtype=tensor_args.dtype)
    x_sph[..., 3] = radius
    while simulation_app.is_running():
        my_world.step(render=True)
        if not my_world.is_playing():
            if i % 100 == 0:
                print("**** Click Play to start simulation *****")
            i += 1
            # if step_index == 0:
            #    my_world.play()
            continue

        sp_buffer = []
        sph_position, _ = target.get_local_pose()

        x_sph[..., :3] = tensor_args.to_device(sph_position).view(1, 1, 1, 3)

        model.world_model.decay_layer("world")
        data = realsense_data.get_data()
        clip_camera(data)
        cube_position, cube_orientation = camera_marker.get_local_pose()
        camera_pose = Pose(
            position=tensor_args.to_device(cube_position),
            quaternion=tensor_args.to_device(cube_orientation),
        )
        # print(data["rgba"].shape, data["depth"].shape, data["intrinsics"])

        data_camera = CameraObservation(  # rgb_image = data["rgba_nvblox"],
            depth_image=data["depth"], intrinsics=data["intrinsics"], pose=camera_pose
        )
        data_camera = data_camera.to(device=model.tensor_args.device)
        # print(data_camera.depth_image, data_camera.rgb_image, data_camera.intrinsics)
        # print("got new message")
        model.world_model.add_camera_frame(data_camera, "world")
        # print("added camera frame")
        model.world_model.process_camera_frames("world", False)
        torch.cuda.synchronize()
        model.world_model.update_blox_hashes()
        bounding = Cuboid("t", dims=[1, 1, 1], pose=[0, 0, 0, 1, 0, 0, 0])
        voxels = model.world_model.get_voxels_in_bounding_box(bounding, 0.025)
        # print(data_camera.depth_image)
        if args.show_window:
            depth_image = data["raw_depth"]
            color_image = data["raw_rgb"]
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=100), cv2.COLORMAP_VIRIDIS
            )
            images = np.hstack((color_image, depth_colormap))

            cv2.namedWindow("Align Example", cv2.WINDOW_NORMAL)
            cv2.imshow("Align Example", images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord("q") or key == 27:
                cv2.destroyAllWindows()
                break

        draw_points(voxels)
        d, d_vec = model.get_collision_vector(x_sph)

        p = d.item()
        p = max(1, p * 5)
        if d.item() == 0.0:
            target_material.set_color(np.ravel([0, 1, 0]))
        elif d.item() <= model.contact_distance:
            target_material.set_color(np.array([0, 0, p]))
        elif d.item() >= model.contact_distance:
            target_material.set_color(np.array([p, 0, 0]))

        if d.item() != 0.0:
            print(d, d_vec)

            draw_line(sph_position, d_vec[..., :3].view(3).cpu().numpy())
        else:
            # Third Party
            try:
                from omni.isaac.debug_draw import _debug_draw
            except ImportError:
                from isaacsim.util.debug_draw import _debug_draw

            draw = _debug_draw.acquire_debug_draw_interface()
            # if draw.get_num_points() > 0:
            draw.clear_lines()

    realsense_data.stop_device()
    print("finished program")
    simulation_app.close()
