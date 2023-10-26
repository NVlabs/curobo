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

# CuRobo
from curobo.geom.types import Capsule, Cuboid, Cylinder, Mesh, Sphere, WorldConfig
from curobo.util_file import get_assets_path, join_path


def approximate_geometry():
    # CuRobo
    from curobo.geom.sphere_fit import SphereFitType
    from curobo.geom.types import Capsule, WorldConfig

    obstacle_capsule = Capsule(
        name="capsule",
        radius=0.2,
        base=[0, 0, 0],
        tip=[0, 0, 0.5],
        pose=[0.0, 5, 0.0, 0.043, -0.471, 0.284, 0.834],
        color=[0, 1.0, 0, 1.0],
    )

    sph = obstacle_capsule.get_bounding_spheres(
        500, 0.005, SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE
    )

    WorldConfig(sphere=sph).save_world_as_mesh("bounding_spheres.obj")

    WorldConfig(capsule=[obstacle_capsule]).save_world_as_mesh("capsule.obj")


def doc_example():
    # describe a cuboid obstacle

    obstacle_1 = Cuboid(
        name="cube_1",
        pose=[0.0, 0.0, 0.0, 0.043, -0.471, 0.284, 0.834],
        dims=[0.2, 1.0, 0.2],
        color=[0.8, 0.0, 0.0, 1.0],
    )

    # describe a mesh obstacle
    # import a mesh file:

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
    file_path = "debug_mesh.obj"
    world_model.save_world_as_mesh(file_path)

    cuboid_world = WorldConfig.create_obb_world(world_model)
    cuboid_world.save_world_as_mesh("debug_cuboid_mesh.obj")

    collision_support_world = WorldConfig.create_collision_support_world(world_model)
    collision_support_world.save_world_as_mesh("debug_collision_mesh.obj")


if __name__ == "__main__":
    approximate_geometry()
