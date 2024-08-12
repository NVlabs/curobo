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
from curobo.geom.types import Capsule, Cuboid, Cylinder, Mesh, Sphere, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.util_file import get_assets_path, join_path


def test_world_modify():
    tensor_args = TensorDeviceType()
    obstacle_1 = Cuboid(
        name="cube_1",
        pose=[0.0, 0.0, 0.0, 0.043, -0.471, 0.284, 0.834],
        dims=[0.2, 1.0, 0.2],
        color=[0.8, 0.0, 0.0, 1.0],
    )

    mesh_file = join_path(get_assets_path(), "scene/nvblox/srl_ur10_bins.obj")

    obstacle_2 = Mesh(
        name="mesh_1",
        pose=[0.0, 2, 0.5, 0.043, -0.471, 0.284, 0.834],
        file_path=mesh_file,
        scale=[0.5, 0.5, 0.5],
    )

    obstacle_3 = Capsule(
        name="capsule",
        radius=0.2,
        base=[0, 0, 0],
        tip=[0, 0, 0.5],
        pose=[0.0, 5, 0.0, 0.043, -0.471, 0.284, 0.834],
        # pose=[0.0, 5, 0.0, 1,0,0,0],
        color=[0, 1.0, 0, 1.0],
    )

    obstacle_4 = Cylinder(
        name="cylinder_1",
        radius=0.2,
        height=0.5,
        pose=[0.0, 6, 0.0, 0.043, -0.471, 0.284, 0.834],
        color=[0, 1.0, 0, 1.0],
    )

    obstacle_5 = Sphere(
        name="sphere_1",
        radius=0.2,
        pose=[0.0, 7, 0.0, 0.043, -0.471, 0.284, 0.834],
        color=[0, 1.0, 0, 1.0],
    )

    world_model = WorldConfig(
        mesh=[obstacle_2],
        cuboid=[obstacle_1],
        capsule=[obstacle_3],
        cylinder=[obstacle_4],
        sphere=[obstacle_5],
    )
    world_model.randomize_color(r=[0.2, 0.7], g=[0.4, 0.8], b=[0.0, 0.4])

    collision_support_world = WorldConfig.create_collision_support_world(world_model)

    world_collision_config = WorldCollisionConfig(tensor_args, world_model=collision_support_world)
    world_ccheck = WorldMeshCollision(world_collision_config)

    world_ccheck.enable_obstacle("sphere_1", False)

    w_pose = Pose.from_list([0, 0, 1, 1, 0, 0, 0], tensor_args)
    world_ccheck.update_obstacle_pose(name="cylinder_1", w_obj_pose=w_pose)

    x_sph = torch.as_tensor(
        [[0.0, 0.0, 0.0, 0.1], [10.0, 0.0, 0.0, 0.1], [0.01, 0.01, 0.0, 0.1]],
        **(tensor_args.as_torch_dict())
    ).view(1, 1, -1, 4)
    # create buffers:
    query_buffer = CollisionQueryBuffer.initialize_from_shape(
        x_sph.shape, tensor_args, world_ccheck.collision_types
    )
    act_distance = tensor_args.to_device([0.0])

    weight = tensor_args.to_device([1])
    d_sph = world_ccheck.get_sphere_distance(x_sph, query_buffer, weight, act_distance).view(-1)
    assert d_sph[0] >= 0.2
    assert d_sph[1] == 0.0
    assert d_sph[2] >= 0.19
    world_ccheck.update_obstacle_pose("cube_1", Pose.from_list([1, 0, 0, 1, 0, 0, 0], tensor_args))
    d_sph = world_ccheck.get_sphere_distance(x_sph, query_buffer, weight, act_distance).view(-1)
    assert torch.sum(d_sph) == 0.0

    x_sph[0, 0, 0, 1] = 5.0
    x_sph[0, 0, 0, 3] = 0.2

    d_sph = world_ccheck.get_sphere_distance(x_sph, query_buffer, weight, act_distance).view(-1)
    assert d_sph[0] >= 0.35


def test_batch_collision():
    tensor_args = TensorDeviceType()
    world_config_1 = WorldConfig(
        cuboid=[
            Cuboid(
                name="cube_env_1",
                pose=[0.0, 0.0, 0.0, 1.0, 0, 0, 0],
                dims=[0.2, 1.0, 0.2],
                color=[0.8, 0.0, 0.0, 1.0],
            )
        ]
    )
    world_config_2 = WorldConfig(
        cuboid=[
            Cuboid(
                name="cube_env_2",
                pose=[0.0, 0.0, 1.0, 1, 0, 0, 0],
                dims=[0.2, 1.0, 0.2],
                color=[0.8, 0.0, 0.0, 1.0],
            )
        ]
    )

    world_coll_config = WorldCollisionConfig(
        tensor_args, world_model=[world_config_1, world_config_2]
    )

    world_ccheck = WorldPrimitiveCollision(world_coll_config)

    x_sph = torch.zeros((2, 1, 1, 4), device=tensor_args.device, dtype=tensor_args.dtype)
    x_sph[..., 3] = 0.1

    query_buffer = CollisionQueryBuffer.initialize_from_shape(
        x_sph.shape, tensor_args, world_ccheck.collision_types
    )
    act_distance = tensor_args.to_device([0.0])

    weight = tensor_args.to_device([1])

    d = world_ccheck.get_sphere_distance(
        x_sph,
        query_buffer,
        weight,
        act_distance,
    )

    assert d[0] == 0.2 and d[1] == 0.2
    env_query_idx = torch.zeros((x_sph.shape[0]), device=tensor_args.device, dtype=torch.int32)
    env_query_idx[1] = 1
    d = world_ccheck.get_sphere_distance(
        x_sph, query_buffer, weight, act_distance, env_query_idx=env_query_idx
    )

    assert d[0] == 0.2 and d[1] == 0.0


def test_world_modify_cpu():
    tensor_args = TensorDeviceType()
    obstacle_1 = Cuboid(
        name="cube_1",
        pose=[0.0, 0.0, 0.0, 0.043, -0.471, 0.284, 0.834],
        dims=[0.2, 1.0, 0.2],
        color=[0.8, 0.0, 0.0, 1.0],
    )

    mesh_file = join_path(get_assets_path(), "scene/nvblox/srl_ur10_bins.obj")

    obstacle_2 = Mesh(
        name="mesh_1",
        pose=[0.0, 2, 0.5, 0.043, -0.471, 0.284, 0.834],
        file_path=mesh_file,
        scale=[0.5, 0.5, 0.5],
    )

    obstacle_3 = Capsule(
        name="capsule",
        radius=0.2,
        base=[0, 0, 0],
        tip=[0, 0, 0.5],
        pose=[0.0, 5, 0.0, 0.043, -0.471, 0.284, 0.834],
        # pose=[0.0, 5, 0.0, 1,0,0,0],
        color=[0, 1.0, 0, 1.0],
    )

    obstacle_4 = Cylinder(
        name="cylinder_1",
        radius=0.2,
        height=0.5,
        pose=[0.0, 6, 0.0, 0.043, -0.471, 0.284, 0.834],
        color=[0, 1.0, 0, 1.0],
    )

    obstacle_5 = Sphere(
        name="sphere_1",
        radius=0.2,
        pose=[0.0, 7, 0.0, 0.043, -0.471, 0.284, 0.834],
        color=[0, 1.0, 0, 1.0],
    )

    world_model = WorldConfig(
        mesh=[obstacle_2],
        cuboid=[obstacle_1],
        capsule=[obstacle_3],
        cylinder=[obstacle_4],
        sphere=[obstacle_5],
    )
    world_model.randomize_color(r=[0.2, 0.7], g=[0.4, 0.8], b=[0.0, 0.4])

    collision_support_world = WorldConfig.create_collision_support_world(world_model)

    world_collision_config = WorldCollisionConfig(tensor_args, world_model=collision_support_world)
    world_ccheck = WorldMeshCollision(world_collision_config)

    world_ccheck.enable_obstacle("sphere_1", False)

    w_pose = Pose.from_list([0, 0, 1, 1, 0, 0, 0], tensor_args)
    world_ccheck.update_obstacle_pose(
        name="cylinder_1", w_obj_pose=w_pose, update_cpu_reference=True
    )

    assert world_ccheck.world_model.get_obstacle("cylinder_1").pose[2] == 1
