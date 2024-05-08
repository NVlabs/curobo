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
import pytest
import torch

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.types import PointCloud, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.camera import CameraObservation
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_segmenter import RobotSegmenter

torch.manual_seed(30)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    # Third Party
    from nvblox_torch.datasets.mesh_dataset import MeshDataset

except ImportError:
    pytest.skip(
        "Nvblox Torch is not available or pyrender is not installed", allow_module_level=True
    )


def create_render_dataset(
    robot_file,
    fov_deg: float = 60,
    n_frames: int = 20,
    retract_delta: float = 0.0,
):
    # load robot:
    robot_dict = load_yaml(join_path(get_robot_configs_path(), robot_file))
    robot_dict["robot_cfg"]["kinematics"]["load_link_names_with_mesh"] = True
    robot_dict["robot_cfg"]["kinematics"]["load_meshes"] = True

    robot_cfg = RobotConfig.from_dict(robot_dict["robot_cfg"])

    kin_model = CudaRobotModel(robot_cfg.kinematics)

    q = kin_model.retract_config

    q += retract_delta

    meshes = kin_model.get_robot_as_mesh(q)

    world = WorldConfig(mesh=meshes[:])
    world_table = WorldConfig.from_dict(
        load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
    )
    world_table.cuboid[0].dims = [0.5, 0.5, 0.1]
    world.add_obstacle(world_table.objects[0])
    robot_mesh = (
        WorldConfig.create_merged_mesh_world(world, process_color=False).mesh[0].get_trimesh_mesh()
    )

    mesh_dataset = MeshDataset(
        None,
        n_frames=n_frames,
        image_size=480,
        save_data_dir=None,
        trimesh_mesh=robot_mesh,
        fov_deg=fov_deg,
    )
    q_js = JointState(position=q, joint_names=kin_model.joint_names)

    return mesh_dataset, q_js


@pytest.mark.parametrize(
    "robot_file",
    ["iiwa.yml", "iiwa_allegro.yml", "franka.yml", "ur10e.yml", "ur5e.yml"],
)
def test_mask_image(robot_file):
    # create robot segmenter:
    tensor_args = TensorDeviceType()

    curobo_segmenter = RobotSegmenter.from_robot_file(
        robot_file, collision_sphere_buffer=0.01, distance_threshold=0.05, use_cuda_graph=True
    )

    mesh_dataset, q_js = create_render_dataset(robot_file, n_frames=5)

    for i in range(len(mesh_dataset)):

        data = mesh_dataset[i]
        cam_obs = CameraObservation(
            depth_image=tensor_args.to_device(data["depth"]).unsqueeze(0) * 1000,
            intrinsics=data["intrinsics"],
            pose=Pose.from_matrix(data["pose"].to(device=tensor_args.device)),
        )
        if not curobo_segmenter.ready:
            curobo_segmenter.update_camera_projection(cam_obs)

        depth_mask, filtered_image = curobo_segmenter.get_robot_mask_from_active_js(
            cam_obs,
            q_js,
        )
        if torch.count_nonzero(depth_mask) > 100:
            return
    assert False


@pytest.mark.parametrize(
    "robot_file",
    ["iiwa.yml", "iiwa_allegro.yml", "franka.yml", "ur10e.yml", "ur5e.yml"],
)
def test_batch_mask_image(robot_file):
    # create robot segmenter:
    tensor_args = TensorDeviceType()

    curobo_segmenter = RobotSegmenter.from_robot_file(
        robot_file, collision_sphere_buffer=0.01, distance_threshold=0.05, use_cuda_graph=True
    )

    mesh_dataset, q_js = create_render_dataset(robot_file, n_frames=5)
    mesh_dataset_zoom, q_js = create_render_dataset(robot_file, fov_deg=40, n_frames=5)

    for i in range(len(mesh_dataset)):

        data = mesh_dataset[i]
        data_zoom = mesh_dataset_zoom[i]

        cam_obs = CameraObservation(
            depth_image=tensor_args.to_device(data["depth"]) * 1000,
            intrinsics=data["intrinsics"],
            pose=Pose.from_matrix(data["pose"].to(device=tensor_args.device)),
        )
        cam_obs_zoom = CameraObservation(
            depth_image=tensor_args.to_device(data_zoom["depth"]) * 1000,
            intrinsics=data_zoom["intrinsics"],
            pose=Pose.from_matrix(data_zoom["pose"].to(device=tensor_args.device)),
        )
        cam_obs = cam_obs.stack(cam_obs_zoom)
        if not curobo_segmenter.ready:
            curobo_segmenter.update_camera_projection(cam_obs)

        depth_mask, filtered_image = curobo_segmenter.get_robot_mask_from_active_js(
            cam_obs,
            q_js,
        )
        if torch.count_nonzero(depth_mask[0]) > 100 and (torch.count_nonzero(depth_mask[1]) > 100):
            return
    assert False


@pytest.mark.parametrize(
    "robot_file",
    ["iiwa.yml", "iiwa_allegro.yml", "franka.yml", "ur10e.yml", "ur5e.yml"],
)
def test_batch_robot_mask_image(robot_file):
    # create robot segmenter:
    tensor_args = TensorDeviceType()

    curobo_segmenter = RobotSegmenter.from_robot_file(
        robot_file, collision_sphere_buffer=0.01, distance_threshold=0.05, use_cuda_graph=True
    )

    mesh_dataset, q_js = create_render_dataset(robot_file, n_frames=5)
    mesh_dataset_zoom, q_js_zoom = create_render_dataset(
        robot_file,
        fov_deg=40,
        n_frames=5,
        retract_delta=0.4,
    )
    q_js = q_js.unsqueeze(0).stack(q_js_zoom.unsqueeze(0))

    for i in range(len(mesh_dataset)):

        data = mesh_dataset[i]
        data_zoom = mesh_dataset_zoom[i]

        cam_obs = CameraObservation(
            depth_image=tensor_args.to_device(data["depth"]) * 1000,
            intrinsics=data["intrinsics"],
            pose=Pose.from_matrix(data["pose"].to(device=tensor_args.device)),
        )
        cam_obs_zoom = CameraObservation(
            depth_image=tensor_args.to_device(data_zoom["depth"]) * 1000,
            intrinsics=data_zoom["intrinsics"],
            pose=Pose.from_matrix(data_zoom["pose"].to(device=tensor_args.device)),
        )
        cam_obs = cam_obs.stack(cam_obs_zoom)
        if not curobo_segmenter.ready:
            curobo_segmenter.update_camera_projection(cam_obs)

        depth_mask, filtered_image = curobo_segmenter.get_robot_mask_from_active_js(
            cam_obs,
            q_js,
        )
        if torch.count_nonzero(depth_mask[0]) > 100 and (torch.count_nonzero(depth_mask[1]) > 100):
            return
    assert False
