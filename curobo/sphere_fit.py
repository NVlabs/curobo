# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sphere fitting utilities for approximating mesh geometry.

cuRobo represents robot links and scene obstacles as collections of spheres for fast
GPU collision checking. This module exposes the public API for fitting sphere
approximations to triangle meshes and inspecting fit quality.

Typical workflow:

1. Load a triangle mesh (e.g. with :mod:`trimesh`).
2. Optionally call :func:`estimate_sphere_count` to pick a reasonable sphere count from
   the mesh bounding volume.
3. Call :func:`fit_spheres_to_mesh` with the desired :class:`SphereFitType` strategy.
4. Inspect the returned :class:`SphereFitResult` for sphere centers, radii, and
   :class:`SphereFitMetrics` fit quality.

Example:
    .. code-block:: python

        import trimesh
        from curobo.sphere_fit import (
            SphereFitType,
            estimate_sphere_count,
            fit_spheres_to_mesh,
        )

        mesh = trimesh.load("link.obj")
        n = estimate_sphere_count(mesh)
        result = fit_spheres_to_mesh(mesh, n_spheres=n, fit_type=SphereFitType.VOXEL)
        centers, radii, metrics = result.centers, result.radii, result.metrics
"""
from curobo._src.geom.sphere_fit.fit_spheres import fit_spheres_to_mesh
from curobo._src.geom.sphere_fit.sphere_count import estimate_sphere_count
from curobo._src.geom.sphere_fit.types import (
    SphereFitMetrics,
    SphereFitResult,
    SphereFitType,
)

__all__ = [
    "SphereFitMetrics",
    "SphereFitResult",
    "SphereFitType",
    "estimate_sphere_count",
    "fit_spheres_to_mesh",
]
