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

# CuRobo
from curobo.geom.sdf.world import (
    CollisionCheckerType,
    WorldCollisionConfig,
    WorldPrimitiveCollision,
)
from curobo.geom.sdf.world_mesh import WorldMeshCollision
from curobo.geom.types import Mesh, WorldConfig
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
                    "block": {"dims": [0.5, 0.5, 0.5], "pose": [-0.25, 0, 0, 1, 0, 0, 0]},
                    "block1": {"dims": [0.1, 0.2, 0.5], "pose": [0.25, 0.1, 0, 1, 0, 0, 0]},
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
            "max_distance": 5.0,
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


@pytest.fixture(scope="function")
def world_collision_primitive():
    world_model = get_world_model(True)

    tensor_args = TensorDeviceType()
    world_collision_config = WorldCollisionConfig.load_from_dict(
        {
            "checker_type": (CollisionCheckerType.PRIMITIVE),
            "max_distance": 5.0,
            "n_envs": 1,
        },
        world_model,
        tensor_args,
    )

    world_collision = WorldPrimitiveCollision(world_collision_config)

    return world_collision


@pytest.mark.parametrize(
    "world_collision",
    [
        (True, True),
        (False, True),
    ],
    indirect=True,
)
def test_voxels_from_world(world_collision):
    voxel_size = 0.1
    voxels = world_collision.get_voxels_in_bounding_box(voxel_size=voxel_size)
    assert voxels.shape[0] > 4


# @pytest.mark.skip(reason="Not ready yet.")
@pytest.mark.parametrize(
    "world_collision",
    [
        (True, True),
        # (False, True),
    ],
    indirect=True,
)
def test_esdf_from_world(world_collision):
    voxel_size = 0.02
    voxels = world_collision.get_voxels_in_bounding_box(voxel_size=voxel_size).clone()
    world_collision.clear_voxelization_cache()
    esdf = world_collision.get_esdf_in_bounding_box(voxel_size=voxel_size).clone()

    occupied = esdf.get_occupied_voxels(feature_threshold=0.0)

    assert voxels.shape == occupied.shape


@pytest.mark.parametrize(
    "world_collision",
    [
        (True, True),
        (False, True),
    ],
    indirect=True,
)
def test_voxels_prim_mesh(world_collision, world_collision_primitive):
    voxel_size = 0.05
    voxels = world_collision.get_voxels_in_bounding_box(voxel_size=voxel_size).clone()
    voxels_prim = world_collision_primitive.get_voxels_in_bounding_box(
        voxel_size=voxel_size
    ).clone()
    assert voxels.shape == voxels_prim.shape


@pytest.mark.parametrize(
    "world_collision",
    [
        (True, True),
        # (False, True),
    ],
    indirect=True,
)
def test_esdf_prim_mesh(world_collision, world_collision_primitive):
    voxel_size = 0.1
    esdf = world_collision.get_esdf_in_bounding_box(voxel_size=voxel_size).clone()
    esdf_prim = world_collision_primitive.get_esdf_in_bounding_box(voxel_size=voxel_size).clone()
    voxels = esdf.get_occupied_voxels(voxel_size)
    voxels_prim = esdf_prim.get_occupied_voxels(voxel_size)
    assert voxels.shape == voxels_prim.shape


@pytest.mark.parametrize(
    "world_collision",
    [
        (True, True),
        (False, True),
        (True, False),
        (False, False),
    ],
    indirect=True,
)
def test_marching_cubes_from_world(world_collision):
    voxel_size = 0.05
    voxels = world_collision.get_voxels_in_bounding_box(voxel_size=voxel_size)
    mesh = Mesh.from_pointcloud(voxels[:, :3].detach().cpu().numpy(), pitch=voxel_size * 0.1)

    mesh.save_as_mesh("test_" + str(len(voxels)) + ".stl")
    assert len(mesh.vertices) > 100  # exact value is 240
