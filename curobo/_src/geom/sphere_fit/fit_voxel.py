# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Voxel-based and surface-sampling sphere fitting methods.

This module provides voxelization-based approaches for approximating 3D meshes
with sets of spheres.  Mesh proximity queries (signed distance, closest surface
point) are performed via Warp GPU kernels through
:class:`~curobo._src.geom.sphere_fit.wp_mesh_query.WarpMeshQuery`.
"""

# Standard Library
from __future__ import annotations

from typing import Optional, Tuple

# Third Party
import numpy as np
import torch
import trimesh

# CuRobo
from curobo._src.geom.sphere_fit.wp_mesh_query import WarpMeshQuery
from curobo._src.util.logging import log_warn

# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def sample_even_fit_mesh(
    mesh: trimesh.Trimesh,
    num_spheres: int,
    sphere_radius: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample even points on the surface of the mesh and return them with the given radius.

    Falls back to :func:`trimesh.sample.sample_surface` when even sampling
    fails (e.g. very small or degenerate meshes).

    Args:
        mesh: Mesh to sample points from.
        num_spheres: Number of spheres to sample.
        sphere_radius: Sphere radius.

    Returns:
        Tuple of points ``(N, 3)`` and radii ``(N,)``.
    """
    try:
        n_pts = trimesh.sample.sample_surface_even(mesh, num_spheres)[0]
    except Exception as e:
        log_warn(f"sample_even_fit_mesh: sample_surface_even failed ({e}), falling back to sample_surface")
        n_pts = np.zeros((0, 3))

    if len(n_pts) == 0:
        n_pts = trimesh.sample.sample_surface(mesh, num_spheres)[0]

    n_radius = np.full(len(n_pts), sphere_radius)
    return n_pts, n_radius


# ---------------------------------------------------------------------------
# Voxel fitting
# ---------------------------------------------------------------------------

def _build_bbox_grid(mesh: trimesh.Trimesh, num_spheres: int) -> np.ndarray:
    """Create a uniform 3D grid of points over the mesh bounding box.

    The grid resolution is chosen so that the total number of cells is
    approximately ``num_spheres``, distributed proportionally to the mesh
    extents along each axis.

    Args:
        mesh: Input mesh.
        num_spheres: Target number of grid cells.

    Returns:
        ``(M, 3)`` array of grid-centre coordinates.
    """
    lo, hi = mesh.bounds
    extents = hi - lo
    bbox_vol = extents[0] * extents[1] * extents[2]
    if bbox_vol <= 0:
        return np.zeros((0, 3))

    pitch = (bbox_vol / num_spheres) ** (1 / 3)
    nx = max(int(np.ceil(extents[0] / pitch)), 1)
    ny = max(int(np.ceil(extents[1] / pitch)), 1)
    nz = max(int(np.ceil(extents[2] / pitch)), 1)

    xs = np.linspace(lo[0] + pitch / 2, hi[0] - pitch / 2, nx)
    ys = np.linspace(lo[1] + pitch / 2, hi[1] - pitch / 2, ny)
    zs = np.linspace(lo[2] + pitch / 2, hi[2] - pitch / 2, nz)

    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)
    return grid.reshape(-1, 3)


def voxel_fit_mesh(
    mesh: trimesh.Trimesh,
    num_spheres: int,
    device: torch.device = torch.device("cuda", 0),
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Fit inscribed spheres by SDF-filtering a bounding-box voxel grid.

    Creates a uniform grid over the mesh bounding box, queries the signed
    distance at each centre via Warp, and keeps only interior points.  The
    radius of each sphere equals the absolute (negated) SDF value (i.e. the
    largest inscribed sphere at that centre).

    Args:
        mesh: Input mesh.
        num_spheres: Target number of spheres.

    Returns:
        Tuple of sphere positions ``(N, 3)`` and radii ``(N,)`` for interior
        points, or ``(None, None)`` if no interior points are found.
    """
    pts = _build_bbox_grid(mesh, num_spheres)
    if len(pts) == 0:
        return None, None

    if device.index is None:
        device = torch.device(device.type, 0)
    mq = WarpMeshQuery(mesh, device)
    pts_t = torch.as_tensor(pts, dtype=torch.float32, device=device).contiguous()
    sdf_t, _ = mq.query_sdf(pts_t)
    sdf = sdf_t.cpu().numpy()  # Warp convention: negative = inside

    keep = sdf < 0
    pts = pts[keep]
    rad = -sdf[keep]

    if len(pts) == 0:
        return None, None

    order = np.argsort(-rad)
    pts = pts[order]
    rad = rad[order]

    if len(pts) > num_spheres:
        pts = pts[:num_spheres]
        rad = rad[:num_spheres]

    return pts, rad


