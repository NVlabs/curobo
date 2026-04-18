# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""GPU data storage for scene obstacles.

This module provides GPU tensor storage classes for obstacle types:
- CuboidData: Oriented bounding boxes
- MeshData: Triangle meshes with BVH acceleration
- VoxelData: Voxel grids with ESDF values
- SceneData: Aggregate container for all obstacle types

Each *Data class owns PyTorch tensors and provides:
- Factory methods (create_cache, from_scene_cfg)
- Add/update/query operations
- to_warp() conversion for kernel access

The corresponding *DataWarp structs and SDF functions are defined
in each data_*.py file for use with generic collision kernels.

SDF Function Registration:
    Each data module defines is_obs_enabled, compute_sdf_value, and
    compute_sdf_with_grad functions. These are registered as Warp function
    overloads by collision/TSDF kernels using OBSTACLE_SDF_MODULES.
"""

# List of data modules containing SDF functions for obstacle types.
# Each module must define:
#   - is_obs_enabled(obs_set, env_idx, local_idx) -> bool
#   - compute_sdf_value(obs_set, env_idx, local_idx, point) -> float32
#   - compute_sdf_with_grad(obs_set, env_idx, local_idx, point) -> vec4
#
# To add a new obstacle type, add its module path here.
OBSTACLE_SDF_MODULES = [
    "curobo._src.geom.data.data_cuboid",
    "curobo._src.geom.data.data_mesh",
    "curobo._src.geom.data.data_voxel",
]

# Cuboid
from curobo._src.geom.data.data_cuboid import (
    CuboidData,
    CuboidDataWarp,
    #compute_sdf_value,
    #compute_sdf_with_grad,
    #is_obs_enabled,
)

# Mesh
from curobo._src.geom.data.data_mesh import (
    MeshData,
    MeshDataWarp,
    WarpMeshCache,
    #compute_sdf_value as compute_sdf_value_mesh,
    #compute_sdf_with_grad as compute_sdf_with_grad_mesh,
    #is_obs_enabled as is_obs_enabled_mesh,
)

# Scene (aggregate)
from curobo._src.geom.data.data_scene import (
    SceneData,
    SceneDataWarp,
)

# Voxel
from curobo._src.geom.data.data_voxel import (
    VoxelData,
    VoxelDataWarp,
    #compute_sdf_value as compute_sdf_value_voxel,
    #compute_sdf_with_grad as compute_sdf_with_grad_voxel,
    #is_obs_enabled as is_obs_enabled_voxel,
)

# Pose helpers
from curobo._src.geom.data.helper_pose import (
    get_forward_quat,
    get_obs_idx,
    load_inv_position,
    load_inv_quat,
)

__all__ = [
    # SDF module registry
    "OBSTACLE_SDF_MODULES",
    # Cuboid
    "CuboidData",
    "CuboidDataWarp",
    # Mesh
    "MeshData",
    "MeshDataWarp",
    "WarpMeshCache",
    # Voxel
    "VoxelData",
    "VoxelDataWarp",
    # Scene
    "SceneData",
    "SceneDataWarp",
    # Pose helpers
    "get_forward_quat",
    "get_obs_idx",
    "load_inv_position",
    "load_inv_quat",
]
