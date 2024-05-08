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
from curobo.geom.sdf.world import (
    CollisionQueryBuffer,
    WorldCollisionConfig,
    WorldPrimitiveCollision,
)
from curobo.geom.sdf.world_mesh import WorldMeshCollision
from curobo.geom.types import Cuboid, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.util_file import get_world_configs_path, join_path, load_yaml


def test_world_primitive():
    # load a world:
    tensor_args = TensorDeviceType()
    world_file = "collision_test.yml"
    data_dict = load_yaml(join_path(get_world_configs_path(), world_file))
    world_cfg = WorldConfig.from_dict(data_dict)
    coll_cfg = WorldCollisionConfig(world_model=world_cfg, tensor_args=TensorDeviceType())
    coll_check = WorldPrimitiveCollision(coll_cfg)
    x_sph = torch.as_tensor(
        [[0.0, 0.0, 0.0, 0.1], [10.0, 0.0, 0.0, 0.0], [0.01, 0.01, 0.0, 0.1]],
        **(tensor_args.as_torch_dict())
    ).view(1, 1, -1, 4)
    # create buffers:
    query_buffer = CollisionQueryBuffer.initialize_from_shape(
        x_sph.shape, tensor_args, coll_check.collision_types
    )
    weight = tensor_args.to_device([1])
    act_distance = tensor_args.to_device([0.0])

    d_sph = coll_check.get_sphere_distance(x_sph, query_buffer, weight, act_distance).view(-1)
    assert abs(d_sph[0].item() - 0.1) < 1e-3
    assert abs(d_sph[1].item() - 0.0) < 1e-9
    assert abs(d_sph[2].item() - 0.1) < 1e-3


def test_batch_world_primitive():
    """This tests collision checking across different environments"""
    tensor_args = TensorDeviceType()

    world_file = "collision_test.yml"
    data_dict = load_yaml(join_path(get_world_configs_path(), world_file))
    world_cfg = WorldConfig.from_dict(data_dict)
    coll_cfg = WorldCollisionConfig(world_model=world_cfg, tensor_args=TensorDeviceType())
    coll_check = WorldPrimitiveCollision(coll_cfg)
    x_sph = torch.as_tensor(
        [[0.0, 0.0, 0.0, 0.1], [10.0, 0.0, 0.0, 0.0], [0.01, 0.01, 0.0, 0.1]],
        **(tensor_args.as_torch_dict())
    ).view(-1, 1, 1, 4)
    env_query_idx = torch.zeros((x_sph.shape[0]), device=tensor_args.device, dtype=torch.int32)
    # create buffers:
    query_buffer = CollisionQueryBuffer.initialize_from_shape(
        x_sph.shape, tensor_args, coll_check.collision_types
    )
    act_distance = tensor_args.to_device([0.0])

    weight = tensor_args.to_device([1])
    d_sph = coll_check.get_sphere_distance(
        x_sph, query_buffer, weight, act_distance, env_query_idx
    ).view(-1)
    assert abs(d_sph[0].item() - 0.1) < 1e-3
    assert abs(d_sph[1].item() - 0.0) < 1e-9
    assert abs(d_sph[2].item() - 0.1) < 1e-3


def test_swept_world_primitive():
    """This tests collision checking across different environments"""
    tensor_args = TensorDeviceType()

    world_file = "collision_test.yml"
    data_dict = load_yaml(join_path(get_world_configs_path(), world_file))
    world_cfg = WorldConfig.from_dict(data_dict)
    coll_cfg = WorldCollisionConfig(world_model=world_cfg, tensor_args=TensorDeviceType())
    coll_cfg.cache = {"obb": 5}
    coll_check = WorldPrimitiveCollision(coll_cfg)
    # add an obstacle:
    new_cube = Cuboid("cube_1", [0, 0, 1, 1, 0, 0, 0], None, [0.1, 0.2, 0.2])
    coll_check.add_obb(new_cube, 0)
    x_sph = torch.as_tensor(
        [[0.0, 0.0, 0.0, 0.1], [10.0, 0.0, 0.0, 0.0], [0.01, 0.01, 0.0, 0.1]],
        **(tensor_args.as_torch_dict())
    ).view(1, 1, -1, 4)
    env_query_idx = None
    env_query_idx = torch.zeros((x_sph.shape[0]), device=tensor_args.device, dtype=torch.int32)
    # create buffers:
    query_buffer = CollisionQueryBuffer.initialize_from_shape(
        x_sph.shape, tensor_args, coll_check.collision_types
    )
    weight = tensor_args.to_device([1])
    act_distance = tensor_args.to_device([0.0])
    dt = act_distance.clone()
    d_sph_swept = coll_check.get_swept_sphere_distance(
        x_sph, query_buffer, weight, act_distance, dt, 10, env_query_idx
    ).view(-1)
    d_sph = coll_check.get_sphere_distance(
        x_sph, query_buffer.clone() * 0.0, weight, act_distance, env_query_idx
    ).view(-1)
    assert abs(d_sph[0].item() - 0.1) < 1e-3
    assert abs(d_sph[1].item() - 0.0) < 1e-9
    assert abs(d_sph[2].item() - 0.1) < 1e-3

    assert abs(d_sph_swept[0].item() - 0.1) < 1e-3
    assert abs(d_sph_swept[1].item() - 0.0) < 1e-9
    assert abs(d_sph_swept[2].item() - 0.1) < 1e-3


def test_world_primitive_mesh_instance():
    # load a world:
    tensor_args = TensorDeviceType()
    world_file = "collision_test.yml"
    data_dict = load_yaml(join_path(get_world_configs_path(), world_file))
    world_cfg = WorldConfig.from_dict(data_dict)
    coll_cfg = WorldCollisionConfig(world_model=world_cfg, tensor_args=TensorDeviceType())
    coll_check = WorldMeshCollision(coll_cfg)
    x_sph = torch.as_tensor(
        [[0.0, 0.0, 0.0, 0.1], [10.0, 0.0, 0.0, 0.0], [0.01, 0.01, 0.0, 0.1]],
        **(tensor_args.as_torch_dict())
    ).view(1, 1, -1, 4)
    # create buffers:
    query_buffer = CollisionQueryBuffer.initialize_from_shape(
        x_sph.shape, tensor_args, coll_check.collision_types
    )
    act_distance = tensor_args.to_device([0.0])

    weight = tensor_args.to_device([1])
    d_sph = coll_check.get_sphere_distance(x_sph, query_buffer, weight, act_distance).view(-1)
    assert abs(d_sph[0].item() - 0.1) < 1e-3
    assert abs(d_sph[1].item() - 0.0) < 1e-9
    assert abs(d_sph[2].item() - 0.1) < 1e-3


def test_batch_world_primitive_mesh_instance():
    """This tests collision checking across different environments"""
    tensor_args = TensorDeviceType()

    world_file = "collision_test.yml"
    data_dict = load_yaml(join_path(get_world_configs_path(), world_file))
    world_cfg = WorldConfig.from_dict(data_dict)
    coll_cfg = WorldCollisionConfig(world_model=world_cfg, tensor_args=TensorDeviceType())
    coll_check = WorldMeshCollision(coll_cfg)
    x_sph = torch.as_tensor(
        [[0.0, 0.0, 0.0, 0.1], [10.0, 0.0, 0.0, 0.0], [0.01, 0.01, 0.0, 0.1]],
        **(tensor_args.as_torch_dict())
    ).view(-1, 1, 1, 4)
    env_query_idx = torch.zeros((x_sph.shape[0]), device=tensor_args.device, dtype=torch.int32)
    # create buffers:
    query_buffer = CollisionQueryBuffer.initialize_from_shape(
        x_sph.shape, tensor_args, coll_check.collision_types
    )

    weight = tensor_args.to_device([1])
    act_distance = tensor_args.to_device([0.0])

    d_sph = coll_check.get_sphere_distance(
        x_sph, query_buffer, weight, act_distance, env_query_idx
    ).view(-1)
    assert abs(d_sph[0].item() - 0.1) < 1e-3
    assert abs(d_sph[1].item() - 0.0) < 1e-9
    assert abs(d_sph[2].item() - 0.1) < 1e-3


def test_swept_world_primitive_mesh_instance():
    """This tests collision checking across different environments"""
    tensor_args = TensorDeviceType()

    world_file = "collision_test.yml"
    data_dict = load_yaml(join_path(get_world_configs_path(), world_file))
    world_cfg = WorldConfig.from_dict(data_dict)
    coll_cfg = WorldCollisionConfig(world_model=world_cfg, tensor_args=TensorDeviceType())
    coll_cfg.cache = {"obb": 5}
    coll_check = WorldMeshCollision(coll_cfg)
    # add an obstacle:
    dims = tensor_args.to_device([0.1, 0.2, 0.2])
    w_obj_pose = Pose.from_list([0, 0, 1, 1, 0, 0, 0], tensor_args)
    coll_check.add_obb_from_raw("cube_1", dims, 0, w_obj_pose=w_obj_pose)
    x_sph = torch.as_tensor(
        [[0.0, 0.0, 0.0, 0.1], [10.0, 0.0, 0.0, 0.0], [0.01, 0.01, 0.0, 0.1]],
        **(tensor_args.as_torch_dict())
    ).view(1, 1, -1, 4)
    env_query_idx = None
    env_query_idx = torch.zeros((x_sph.shape[0]), device=tensor_args.device, dtype=torch.int32)
    # create buffers:
    query_buffer = CollisionQueryBuffer.initialize_from_shape(
        x_sph.shape, tensor_args, coll_check.collision_types
    )
    weight = tensor_args.to_device([1])
    act_distance = tensor_args.to_device([0.0])
    dt = act_distance.clone() + 0.01

    d_sph_swept = coll_check.get_swept_sphere_distance(
        x_sph, query_buffer, weight, act_distance, dt, 10, env_query_idx
    ).view(-1)
    d_sph = coll_check.get_sphere_distance(
        x_sph, query_buffer.clone() * 0.0, weight, act_distance, env_query_idx
    ).view(-1)
    assert abs(d_sph[0].item() - 0.1) < 1e-3
    assert abs(d_sph[1].item() - 0.0) < 1e-9
    assert abs(d_sph[2].item() - 0.1) < 1e-3

    assert abs(d_sph_swept[0].item() - 0.1) < 1e-3
    assert abs(d_sph_swept[1].item() - 0.0) < 1e-9
    assert abs(d_sph_swept[2].item() - 0.1) < 1e-3
