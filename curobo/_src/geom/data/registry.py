# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Obstacle data module registry."""

# List of data modules containing SDF functions for obstacle types.
# Each module must define:
#   - is_obs_enabled(obs_set, env_idx, local_idx) -> bool
#   - load_obstacle_transform(obs_set, env_idx, local_idx) -> ObstacleTransform
#   - compute_local_sdf(obs_set, env_idx, local_idx, local_point) -> float32
#   - compute_local_sdf_with_grad(obs_set, env_idx, local_idx, local_point) -> vec4
#
# To add a new obstacle type, add its module path here.
OBSTACLE_SDF_MODULES = [
    "curobo._src.geom.data.data_cuboid",
    "curobo._src.geom.data.data_mesh",
    "curobo._src.geom.data.data_voxel",
]
