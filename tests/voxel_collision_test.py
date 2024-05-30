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
from curobo.geom.sdf.world import (
    CollisionCheckerType,
    CollisionQueryBuffer,
    WorldCollisionConfig,
    WorldPrimitiveCollision,
)
from curobo.geom.sdf.world_mesh import WorldMeshCollision
from curobo.geom.sdf.world_voxel import WorldVoxelCollision
from curobo.geom.types import Cuboid, VoxelGrid, WorldConfig
from curobo.types.base import TensorDeviceType


def get_world_model(single_object: bool = False):
    if single_object:

        world_model = WorldConfig.from_dict(
            {
                "cuboid": {
                    "block1": {"dims": [0.1, 0.2, 0.5], "pose": [0.25, 0.1, 0, 1, 0, 0, 0]},
                }
            }
        )
    else:
        world_model = WorldConfig.from_dict(
            {
                "cuboid": {
                    "block2": {"dims": [0.5, 0.5, 0.5], "pose": [-0.25, 0, 0, 1, 0, 0, 0]},
                    "block3": {"dims": [0.1, 0.2, 0.5], "pose": [0.25, 0.1, 0, 1, 0, 0, 0]},
                }
            }
        )
    return world_model


@pytest.fixture(scope="function")
def world_collision(request):
    world_model = get_world_model(request.param[1])

    if request.param[0]:
        world_model = world_model.get_mesh_world()
    tensor_args = TensorDeviceType()
    world_collision_config = WorldCollisionConfig.load_from_dict(
        {
            "checker_type": (
                CollisionCheckerType.PRIMITIVE
                if not request.param[0]
                else CollisionCheckerType.MESH
            ),
            "max_distance": 1.0,
            "n_envs": 1,
        },
        world_model,
        tensor_args,
    )
    if request.param[0]:
        world_collision = WorldMeshCollision(world_collision_config)
    else:
        world_collision = WorldPrimitiveCollision(world_collision_config)

    return world_collision


def world_voxel_collision_checker():
    world_model = {
        "voxel": {
            "base": {"dims": [2.0, 2.0, 2.0], "pose": [0, 0, 0, 1, 0, 0, 0], "voxel_size": 0.05},
        }
    }
    tensor_args = TensorDeviceType()
    world_collision_config = WorldCollisionConfig.load_from_dict(
        {
            "checker_type": CollisionCheckerType.VOXEL,
            "max_distance": 5.0,
            "n_envs": 1,
        },
        world_model,
        tensor_args,
    )
    world_collision = WorldVoxelCollision(world_collision_config)
    return world_collision


@pytest.mark.parametrize(
    "world_collision",
    [
        ([True, True]),
        ([False, True]),
        ([True, False]),
        ([False, False]),
    ],
    indirect=True,
)
def test_voxel_esdf(world_collision):

    # create voxel collision checker
    world_voxel_collision = world_voxel_collision_checker()

    voxel_grid = world_voxel_collision.get_voxel_grid("base")
    esdf = world_collision.get_esdf_in_bounding_box(
        Cuboid(name="base", pose=voxel_grid.pose, dims=voxel_grid.dims),
        voxel_size=voxel_grid.voxel_size,
    )

    world_voxel_collision.update_voxel_data(esdf)

    voxel_size = 0.01
    esdf = world_collision.get_esdf_in_bounding_box(
        Cuboid(name="base", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1]), voxel_size=voxel_size
    )

    esdf_voxel = world_voxel_collision.get_esdf_in_bounding_box(
        Cuboid(name="base", pose=[0, 0, 0, 1, 0, 0, 0], dims=[1, 1, 1]), voxel_size=voxel_size
    )
    esdf_data = esdf.feature_tensor
    esdf_voxel_data = esdf_voxel.feature_tensor
    esdf_data[esdf_data < -1.0] = 0.0
    esdf_voxel_data[esdf_voxel_data < -1.0] = 0.0
    error = torch.abs(esdf_data - esdf_voxel_data)

    assert torch.max(error) < 2 * voxel_grid.voxel_size


@pytest.mark.parametrize(
    "world_collision",
    [
        ([True, True]),
        ([False, True]),
        ([True, False]),
        ([False, False]),
    ],
    indirect=True,
)
def test_primitive_voxel_sphere_distance(world_collision):
    tensor_args = TensorDeviceType()
    voxel_size = 0.025
    world_voxel_collision = world_voxel_collision_checker()

    voxel_grid = world_voxel_collision.get_voxel_grid("base")
    esdf = world_collision.get_esdf_in_bounding_box(
        Cuboid(name="base", pose=voxel_grid.pose, dims=voxel_grid.dims),
        voxel_size=voxel_grid.voxel_size,
    )

    world_voxel_collision.update_voxel_data(esdf)

    # create a grid and compute distance:
    sample_grid = VoxelGrid(
        name="test", pose=[0, 0, 0, 1, 0, 0, 0], voxel_size=voxel_size, dims=[1, 1, 1]
    )
    sample_spheres = sample_grid.create_xyzr_tensor()
    sample_spheres = sample_spheres.reshape(-1, 1, 1, 4)

    act_distance = tensor_args.to_device([0.0])

    weight = tensor_args.to_device([1.0])
    cuboid_collision_buffer = CollisionQueryBuffer.initialize_from_shape(
        sample_spheres.shape, tensor_args, world_collision.collision_types
    )
    voxel_collision_buffer = CollisionQueryBuffer.initialize_from_shape(
        sample_spheres.shape, tensor_args, world_voxel_collision.collision_types
    )
    d_cuboid = world_collision.get_sphere_distance(
        sample_spheres, cuboid_collision_buffer, weight, act_distance
    )
    d_voxel = world_voxel_collision.get_sphere_distance(
        sample_spheres, voxel_collision_buffer, weight, act_distance
    )

    error = torch.abs(d_cuboid.view(-1) - d_voxel.view(-1))

    assert torch.max(error) - voxel_grid.voxel_size < 1e-3


@pytest.mark.parametrize(
    "world_collision",
    [
        # ([True, True]),
        ([False, True]),
        # ([True, False]),
        # ([False, False]),
    ],
    indirect=True,
)
def test_primitive_voxel_sphere_gradient(world_collision):
    tensor_args = TensorDeviceType()
    world_voxel_collision = world_voxel_collision_checker()

    voxel_grid = world_voxel_collision.get_voxel_grid("base")
    esdf = world_collision.get_esdf_in_bounding_box(
        Cuboid(name="base", pose=voxel_grid.pose, dims=voxel_grid.dims),
        voxel_size=voxel_grid.voxel_size,
    )
    voxel_size = voxel_grid.voxel_size

    world_voxel_collision.update_voxel_data(esdf)

    # create a grid and compute distance:
    sample_grid = VoxelGrid(
        name="test", pose=[0.0, 0.0, 0, 1, 0, 0, 0], voxel_size=voxel_size, dims=[0.1, 0.1, 0.1]
    )
    # sample_grid = VoxelGrid(
    #    name="test", pose=[0.2, 0.0, 0, 1, 0, 0, 0], voxel_size=voxel_size,
    #    dims=[0.1, 0.1, 0.1]
    # )
    sample_spheres = sample_grid.create_xyzr_tensor(transform_to_origin=True)
    sample_spheres = sample_spheres.reshape(-1, 1, 1, 4)

    act_distance = tensor_args.to_device([0.0])

    weight = tensor_args.to_device([1.0])
    cuboid_collision_buffer = CollisionQueryBuffer.initialize_from_shape(
        sample_spheres.shape, tensor_args, world_collision.collision_types
    )
    voxel_collision_buffer = CollisionQueryBuffer.initialize_from_shape(
        sample_spheres.shape, tensor_args, world_voxel_collision.collision_types
    )
    sample_spheres_1 = sample_spheres.clone()
    sample_spheres_1.requires_grad = True
    d_cuboid = world_collision.get_sphere_distance(
        sample_spheres_1, cuboid_collision_buffer, weight, act_distance
    )
    sample_spheres_2 = sample_spheres.clone()
    sample_spheres_2.requires_grad = True
    d_voxel = world_voxel_collision.get_sphere_distance(
        sample_spheres_2, voxel_collision_buffer, weight, act_distance
    )

    error = torch.abs(d_cuboid.view(-1) - d_voxel.view(-1))

    assert torch.max(error) - voxel_grid.voxel_size < 1e-3

    cuboid_gradient = cuboid_collision_buffer.get_gradient_buffer()

    voxel_gradient = voxel_collision_buffer.get_gradient_buffer()
    error = torch.linalg.norm(cuboid_gradient - voxel_gradient, dim=-1)

    assert torch.max(error) - voxel_grid.voxel_size < 1e-3
