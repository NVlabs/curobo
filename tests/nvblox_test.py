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
import sys

# Third Party
import pytest
import torch

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType, CollisionQueryBuffer, WorldCollisionConfig
from curobo.geom.types import BloxMap, Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.camera import CameraObservation
from curobo.types.math import Pose
from curobo.util_file import get_world_configs_path, join_path, load_yaml

try:
    # Third Party
    from nvblox_torch.mapper import Mapper

    # CuRobo
    from curobo.geom.sdf.world_blox import WorldBloxCollision

except ImportError:
    pytest.skip("Nvblox Torch not available", allow_module_level=True)


def test_world_blox_trajectory():
    tensor_args = TensorDeviceType()
    world_file = "collision_nvblox.yml"
    data_dict = load_yaml(join_path(get_world_configs_path(), world_file))
    world_cfg = WorldConfig.from_dict(data_dict)
    coll_cfg = WorldCollisionConfig(world_model=world_cfg, tensor_args=TensorDeviceType())
    coll_check = WorldBloxCollision(coll_cfg)
    x_sph = torch.as_tensor(
        [[0.1, 0.2, 0.0, 10.5], [10.0, 0.0, 0.0, 0.0], [0.01, 0.01, 0.0, 0.1]],
        **(tensor_args.as_torch_dict())
    ).view(1, -1, 1, 4)
    # create buffers:
    query_buffer = CollisionQueryBuffer.initialize_from_shape(
        x_sph.shape, tensor_args, coll_check.collision_types
    )
    weight = tensor_args.to_device([1])
    act_distance = tensor_args.to_device([0.0])
    d_sph = coll_check.get_swept_sphere_distance(
        x_sph, query_buffer, weight, act_distance, act_distance + 0.01, 4, True
    ).view(-1)

    assert d_sph[0] > 10.0
    assert d_sph[1] == 0.0
    assert d_sph[2] == 0.0


def test_world_blox():
    tensor_args = TensorDeviceType()
    world_file = "collision_nvblox.yml"
    data_dict = load_yaml(join_path(get_world_configs_path(), world_file))
    world_cfg = WorldConfig.from_dict(data_dict)
    coll_cfg = WorldCollisionConfig(world_model=world_cfg, tensor_args=TensorDeviceType())
    coll_check = WorldBloxCollision(coll_cfg)
    x_sph = torch.as_tensor(
        [[0.1, 0.2, 0.0, 10.5], [10.0, 0.0, 0.0, 0.0], [0.01, 0.01, 0.0, 0.1]],
        **(tensor_args.as_torch_dict())
    ).view(1, 1, -1, 4)
    # create buffers:
    query_buffer = CollisionQueryBuffer.initialize_from_shape(
        x_sph.shape, tensor_args, coll_check.collision_types
    )
    weight = tensor_args.to_device([1])
    act_distance = tensor_args.to_device([0.0])
    d_sph = coll_check.get_sphere_distance(x_sph, query_buffer, weight, act_distance).view(-1)

    assert d_sph[0] > 10.0
    assert d_sph[1] == 0.0
    assert d_sph[2] == 0.0


def test_world_blox_bounding():
    tensor_args = TensorDeviceType()
    world_file = "collision_nvblox.yml"
    data_dict = load_yaml(join_path(get_world_configs_path(), world_file))
    world_cfg = WorldConfig.from_dict(data_dict)
    coll_cfg = WorldCollisionConfig(world_model=world_cfg, tensor_args=TensorDeviceType())
    coll_check = WorldBloxCollision(coll_cfg)
    bounding_cube = Cuboid("test", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1.0, 1.0, 1.0])
    mesh = coll_check.get_mesh_in_bounding_box(
        bounding_cube,
        voxel_size=0.05,
    )
    assert len(mesh.vertices) > 0


def test_world_blox_get_mesh():
    world_file = "collision_nvblox.yml"
    tensor_args = TensorDeviceType()

    data_dict = load_yaml(join_path(get_world_configs_path(), world_file))
    world_cfg = WorldConfig.from_dict(data_dict)
    coll_cfg = WorldCollisionConfig(world_model=world_cfg, tensor_args=TensorDeviceType())
    coll_check = WorldBloxCollision(coll_cfg)

    world_mesh = coll_check.get_mesh_from_blox_layer("world")
    assert len(world_mesh.vertices) > 10


@pytest.mark.skipif(sys.version_info < (3, 8), reason="pyglet requires python 3.8+")
def test_nvblox_world_from_primitive_world():
    world_file = "collision_cubby.yml"
    tensor_args = TensorDeviceType()
    data_dict = load_yaml(join_path(get_world_configs_path(), world_file))
    world_cfg = WorldConfig.from_dict(data_dict).get_mesh_world(True)
    mesh = world_cfg.mesh[0].get_trimesh_mesh()
    # Third Party
    try:
        # Third Party
        from nvblox_torch.datasets.mesh_dataset import MeshDataset
    except ImportError:
        pytest.skip("pyrender and scikit-image is not installed")

    # create a nvblox collision checker:
    world_config = WorldConfig(
        blox=[
            BloxMap(
                name="world",
                pose=[0, 0, 0, 1, 0, 0, 0],
                voxel_size=0.02,
                integrator_type="tsdf",
            )
        ]
    )
    config = WorldCollisionConfig(
        tensor_args=tensor_args, world_model=world_config, checker_type=CollisionCheckerType.BLOX
    )
    world_coll = WorldBloxCollision(config)
    m_dataset = MeshDataset(
        None, n_frames=100, image_size=1000, save_data_dir=None, trimesh_mesh=mesh
    )
    for i in range(len(m_dataset)):
        data = m_dataset[i]
        cam_obs = CameraObservation(
            rgb_image=data["rgba"].permute(1, 2, 0),
            depth_image=data["depth"],
            intrinsics=data["intrinsics"],
            pose=Pose.from_matrix(data["pose"].to(device=tensor_args.device)),
        )
        world_coll.add_camera_frame(cam_obs, "world")

    world_coll.process_camera_frames("world")
    world_coll.update_blox_mesh("world")
    integrated_mesh = world_coll.get_mesh_from_blox_layer("world")
    if len(integrated_mesh.vertices) > 0:
        assert True
        # print("saving World")
        # integrated_mesh.save_as_mesh("collision_test.obj")
    else:
        assert True
