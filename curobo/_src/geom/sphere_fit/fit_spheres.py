# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Top-level sphere fitting dispatcher.

This module provides the main entry point :func:`fit_spheres_to_mesh` which delegates
to the appropriate fitting backend based on the requested :class:`SphereFitType`.
"""

# Standard Library
from __future__ import annotations

import time
from typing import Optional

# Third Party
import numpy
import torch
import trimesh

from curobo._src.geom.sphere_fit.fit_morphit import MorphItLossWeights, morphit_sphere_fit
from curobo._src.geom.sphere_fit.fit_voxel import sample_even_fit_mesh, voxel_fit_mesh
from curobo._src.geom.sphere_fit.metrics import populate_metrics

# CuRobo
from curobo._src.geom.sphere_fit.sphere_count import estimate_sphere_count
from curobo._src.geom.sphere_fit.types import SphereFitResult, SphereFitType
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_info, log_warn


def _is_hollow_mesh(mesh: trimesh.Trimesh, thickness_ratio: float = 0.1) -> bool:
    """Detect if a mesh is a degenerate thin shell that needs convex-hull replacement.

    Only triggers for watertight meshes whose volume is an extremely small
    fraction of their bounding box (e.g. a closed thin panel).  Non-watertight
    meshes are left as-is; Warp SDF queries handle triangle soups correctly.
    """
    if not mesh.is_watertight:
        return False
    bbox_dims = mesh.bounds[1] - mesh.bounds[0]
    bbox_vol = float(numpy.prod(bbox_dims))
    if bbox_vol < 1e-12:
        return True
    fill_ratio = abs(float(mesh.volume)) / bbox_vol
    return fill_ratio < thickness_ratio


def _apply_clip_plane(
    result: SphereFitResult,
    clip_plane: tuple,
    buffer: float = 0.02,
) -> None:
    """Discard or shrink spheres that cross a half-plane boundary.

    Spheres whose centers are behind the plane (within *buffer*) are removed.
    Remaining spheres have their radii clamped so they don't protrude past
    the plane minus *buffer*.  Modifies *result* in place.
    """
    normal = numpy.array(clip_plane[0], dtype=numpy.float64)
    normal = normal / numpy.linalg.norm(normal)
    offset = float(clip_plane[1])

    signed_dist = result.centers @ normal - offset
    keep = signed_dist > buffer
    if not numpy.all(keep):
        result.centers = result.centers[keep]
        result.radii = result.radii[keep]
        result.num_spheres = len(result.centers)
        signed_dist = signed_dist[keep]

    if result.num_spheres > 0:
        max_radii = numpy.maximum(signed_dist - buffer, 1e-4)
        result.radii = numpy.minimum(result.radii, max_radii)


def fit_spheres_to_mesh(
    mesh: trimesh.Trimesh,
    num_spheres: Optional[int] = None,
    sphere_density: float = 1.0,
    surface_radius: float = 0.005,
    fit_type: SphereFitType = SphereFitType.MORPHIT,
    iterations: int = 200,
    compute_metrics: bool = False,
    coverage_weight: Optional[float] = None,
    protrusion_weight: Optional[float] = None,
    clip_plane: Optional[tuple] = None,
    device_cfg: DeviceCfg = DeviceCfg(),
) -> SphereFitResult:
    """Approximate a mesh with spheres.

    Args:
        mesh: Input mesh.
        num_spheres: Explicit number of spheres to fit.  When ``None`` (default),
            estimated automatically using *sphere_density*.
        sphere_density: Dimensionless density multiplier used when *num_spheres*
            is ``None``.  Scales both the sphere count estimate and the
            per-link cap.  ``1.0`` (default) gives a balanced count; ``2.0``
            doubles it; ``0.5`` halves it.  Practical range: ``0.1`` -- ``10.0``.
        surface_radius: Radius added to surface-sampled spheres.  Only affects
            the ``SURFACE`` fit type and the surface-sampling fallback.
        fit_type: Fitting algorithm; see :class:`SphereFitType`.
        iterations: Optimization iterations (only used by ``MORPHIT``).
        compute_metrics: When True, compute quality metrics (coverage,
            protrusion, surface gap, volume ratio) on the result.
        coverage_weight: MorphIt coverage loss weight.  Higher values force
            spheres to fill the mesh volume more completely.  Only used by
            ``MORPHIT``.  When ``None``, uses the default (1000.0).
        protrusion_weight: MorphIt protrusion loss weight.  Higher values
            penalise sphere surface area outside the mesh more aggressively.
            Only used by ``MORPHIT``.  When ``None``, uses the default (10.0).
        clip_plane: Half-plane constraint ``((nx, ny, nz), offset)`` in
            mesh-local coordinates.  Spheres that cross the plane are penalised
            during MorphIt optimization and hard-clamped afterwards.  For
            non-MorphIt fit types, only the hard clamp is applied.  ``None``
            (default) disables clipping.

    Returns:
        A :class:`SphereFitResult` with sphere positions, radii, and
        optionally quality metrics.
    """
    requested_n_spheres = num_spheres
    used_convex_hull = False

    if fit_type != SphereFitType.SURFACE and _is_hollow_mesh(mesh):
        log_info("sphere_fit: hollow/thin mesh detected, using convex hull")
        mesh = mesh.convex_hull
        used_convex_hull = True

    auto_mode = num_spheres is None
    if auto_mode:
        num_spheres = estimate_sphere_count(mesh, sphere_density=sphere_density)
        log_info(f"sphere_fit: auto num_spheres={num_spheres}")

    n_pts = n_radius = None
    history = []
    fallback_used = False

    t0 = time.time()

    device = device_cfg.device

    if fit_type == SphereFitType.SURFACE:
        n_pts, n_radius = sample_even_fit_mesh(mesh, num_spheres, surface_radius)

    elif fit_type == SphereFitType.VOXEL:
        n_pts, n_radius = voxel_fit_mesh(mesh, num_spheres, device=device)

    elif fit_type == SphereFitType.MORPHIT:
        init_pts, init_rad = voxel_fit_mesh(mesh, num_spheres, device=device)
        if init_pts is not None and len(init_pts) > 0:
            loss_weights = None
            if coverage_weight is not None or protrusion_weight is not None:
                loss_weights = MorphItLossWeights(
                    coverage=coverage_weight if coverage_weight is not None else 1000.0,
                    protrusion=protrusion_weight if protrusion_weight is not None else 10.0,
                )
            n_pts, n_radius, history = morphit_sphere_fit(
                mesh, num_spheres, iterations=iterations,
                init_centers=init_pts, init_radii=init_rad,
                loss_weights=loss_weights,
                clip_plane=clip_plane,
                max_spheres=num_spheres if requested_n_spheres is not None else 0,
                device=device,
            )

    if (n_pts is None or len(n_pts) < 1) and num_spheres > 0:
        log_warn("sphere_fit: primary method failed, falling back to voxel volume")
        n_pts, n_radius = voxel_fit_mesh(mesh, num_spheres, device=device)
        fallback_used = True

    if (n_pts is None or len(n_pts) < 1) and num_spheres > 0:
        log_warn("sphere_fit: voxel fallback empty (thin shell?), using surface sampling")
        n_pts, n_radius = sample_even_fit_mesh(mesh, num_spheres, surface_radius)
        fallback_used = True

    fit_time = time.time() - t0

    if n_pts is None:
        n_pts = numpy.zeros((0, 3))
    if n_radius is None:
        n_radius = numpy.zeros((0,))
    n_radius = numpy.ravel(n_radius)

    if requested_n_spheres is not None and len(n_pts) > requested_n_spheres:
        order = numpy.argsort(-n_radius)[:requested_n_spheres]
        n_pts = n_pts[order]
        n_radius = n_radius[order]

    result = SphereFitResult(
        centers=n_pts,
        radii=n_radius,
        num_spheres=len(n_pts),
        fit_time_s=fit_time,
        used_mesh=mesh,
        history=history,
        debug_info={
            "fit_type": fit_type.value,
            "used_convex_hull": used_convex_hull,
            "auto_n_spheres": auto_mode,
            "requested_n_spheres": requested_n_spheres,
            "resolved_n_spheres": num_spheres,
            "fallback_used": fallback_used,
        },
    )

    if clip_plane is not None and result.num_spheres > 0:
        _apply_clip_plane(result, clip_plane)

    result.centers = torch.as_tensor(
        result.centers, dtype=device_cfg.dtype, device=device
    ).contiguous()
    result.radii = torch.as_tensor(
        result.radii, dtype=device_cfg.dtype, device=device
    ).contiguous()

    if compute_metrics:
        populate_metrics(result, mesh, device=device)

    return result
