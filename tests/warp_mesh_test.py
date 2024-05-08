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
from curobo.geom.sdf.world import CollisionQueryBuffer, WorldCollisionConfig
from curobo.geom.sdf.world_mesh import WorldMeshCollision
from curobo.geom.types import Mesh
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.util_file import get_assets_path, join_path


def test_sdf_pose():
    mesh_file = join_path(get_assets_path(), "scene/nvblox/srl_ur10_bins.obj")
    tensor_args = TensorDeviceType()
    world_config = WorldCollisionConfig(tensor_args)
    world_ccheck = WorldMeshCollision(world_config)
    world_ccheck.create_collision_cache(1)
    new_mesh = Mesh(name="test_mesh", file_path=mesh_file, pose=[0, 0, 0, 1, 0, 0, 0])
    world_ccheck.add_mesh(
        new_mesh,
        env_idx=0,
    )
    query_spheres = torch.zeros((1, 1, 2, 4), **(tensor_args.as_torch_dict()))
    query_spheres[..., 2] = 10.0
    query_spheres[..., 1, :] = 0.0
    query_spheres[..., 3] = 1.0
    act_distance = tensor_args.to_device([0.01])

    weight = tensor_args.to_device([1.0])
    collision_buffer = CollisionQueryBuffer.initialize_from_shape(
        query_spheres.shape, tensor_args, world_ccheck.collision_types
    )
    out = world_ccheck.get_sphere_distance(query_spheres, collision_buffer, act_distance, weight)
    out = out.view(-1)
    assert out[0] <= 0.0
    assert out[1] >= 0.007


def test_swept_sdf_speed_pose():
    mesh_file = join_path(get_assets_path(), "scene/nvblox/srl_ur10_bins.obj")
    tensor_args = TensorDeviceType()
    world_config = WorldCollisionConfig(tensor_args)
    world_ccheck = WorldMeshCollision(world_config)
    world_ccheck.create_collision_cache(1)
    new_mesh = Mesh(name="test_mesh", file_path=mesh_file, pose=[0, 0, 0, 1, 0, 0, 0])
    world_ccheck.add_mesh(
        new_mesh,
        env_idx=0,
    )
    query_spheres = torch.zeros((1, 1, 2, 4), **(tensor_args.as_torch_dict()))
    query_spheres[..., 2] = 10.0
    query_spheres[..., 1, :] = 0.0
    query_spheres[..., 3] = 1.0
    act_distance = tensor_args.to_device([0.01])
    dt = act_distance.clone()
    weight = tensor_args.to_device([1.0])
    collision_buffer = CollisionQueryBuffer.initialize_from_shape(
        query_spheres.shape, tensor_args, world_ccheck.collision_types
    )
    out = world_ccheck.get_swept_sphere_distance(
        query_spheres, collision_buffer, weight, act_distance, dt, 2, True, None
    )
    out = out.view(-1)
    assert out[0] <= 0.0
    assert out[1] >= 0.3


def test_swept_sdf_pose():
    mesh_file = join_path(get_assets_path(), "scene/nvblox/srl_ur10_bins.obj")
    tensor_args = TensorDeviceType()
    world_config = WorldCollisionConfig(tensor_args)
    world_ccheck = WorldMeshCollision(world_config)
    world_ccheck.create_collision_cache(1)
    new_mesh = Mesh(name="test_mesh", file_path=mesh_file, pose=[0, 0, 0, 1, 0, 0, 0])
    world_ccheck.add_mesh(
        new_mesh,
        env_idx=0,
    )
    query_spheres = torch.zeros((1, 1, 2, 4), **(tensor_args.as_torch_dict()))
    query_spheres[..., 2] = 10.0
    query_spheres[..., 1, :] = 0.0
    query_spheres[..., 3] = 1.0
    act_distance = tensor_args.to_device([0.01])
    dt = act_distance.clone()
    weight = tensor_args.to_device([1.0])
    collision_buffer = CollisionQueryBuffer.initialize_from_shape(
        query_spheres.shape, tensor_args, world_ccheck.collision_types
    )
    out = world_ccheck.get_swept_sphere_distance(
        query_spheres, collision_buffer, weight, act_distance, dt, 2, False, None
    )
    out = out.view(-1)
    assert out[0] <= 0.0
    assert out[1] >= 0.01
