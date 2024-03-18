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
from curobo.geom.sdf.world import CollisionCheckerType, WorldCollisionConfig
from curobo.util.logger import log_error


def create_collision_checker(config: WorldCollisionConfig):
    if config.checker_type == CollisionCheckerType.PRIMITIVE:
        # CuRobo
        from curobo.geom.sdf.world import WorldPrimitiveCollision

        return WorldPrimitiveCollision(config)
    elif config.checker_type == CollisionCheckerType.BLOX:
        # CuRobo
        from curobo.geom.sdf.world_blox import WorldBloxCollision

        return WorldBloxCollision(config)
    elif config.checker_type == CollisionCheckerType.MESH:
        # CuRobo
        from curobo.geom.sdf.world_mesh import WorldMeshCollision

        return WorldMeshCollision(config)
    elif config.checker_type == CollisionCheckerType.VOXEL:
        # CuRobo
        from curobo.geom.sdf.world_voxel import WorldVoxelCollision

        return WorldVoxelCollision(config)
    else:
        log_error("Unknown Collision Checker type: " + config.checker_type, exc_info=True)
        raise NotImplementedError
