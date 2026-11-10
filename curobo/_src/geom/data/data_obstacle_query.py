# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared Warp function overloads for obstacle data queries."""

# Standard Library
from importlib import import_module

# Third Party
import warp as wp

from curobo._src.geom.data.registry import OBSTACLE_SDF_MODULES

is_obs_enabled = None
load_obstacle_transform = None
compute_local_sdf = None
compute_local_sdf_with_grad = None
compute_swept_sphere_clearance_lower_bound = None

for _module_path in OBSTACLE_SDF_MODULES:
    _data_module = import_module(_module_path)
    _is_obs_enabled_fn = getattr(_data_module, "is_obs_enabled")
    _load_obstacle_transform_fn = getattr(_data_module, "load_obstacle_transform")
    _compute_local_sdf_fn = getattr(_data_module, "compute_local_sdf")
    _compute_local_sdf_with_grad_fn = getattr(_data_module, "compute_local_sdf_with_grad")
    _compute_swept_sphere_clearance_lower_bound_fn = getattr(
        _data_module, "compute_swept_sphere_clearance_lower_bound"
    )

    is_obs_enabled = wp.func(_is_obs_enabled_fn, module=__name__)
    load_obstacle_transform = wp.func(_load_obstacle_transform_fn, module=__name__)
    compute_local_sdf = wp.func(_compute_local_sdf_fn, module=__name__)
    compute_local_sdf_with_grad = wp.func(_compute_local_sdf_with_grad_fn, module=__name__)
    compute_swept_sphere_clearance_lower_bound = wp.func(
        _compute_swept_sphere_clearance_lower_bound_fn, module=__name__
    )

del (
    _module_path,
    _data_module,
    _is_obs_enabled_fn,
    _load_obstacle_transform_fn,
    _compute_local_sdf_fn,
    _compute_local_sdf_with_grad_fn,
    _compute_swept_sphere_clearance_lower_bound_fn,
)
