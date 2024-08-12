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
"""Approximate mesh geometry with spheres."""

# Standard Library
from enum import Enum
from typing import List, Tuple, Union

# Third Party
import numpy
import numpy as np
import torch
import trimesh
from trimesh.voxel.creation import voxelize

# CuRobo
from curobo.util.logger import log_warn


class SphereFitType(Enum):
    """Supported sphere fit types are listed here. VOXEL_VOLUME_SAMPLE_SURFACE works best.

    See :ref:`attach_object_note` for more details.
    """

    #:  samples the surface of the mesh to approximate geometry
    SAMPLE_SURFACE = "sample_surface"
    #: voxelizes the volume and returns voxel positions that are intersecting with surface.
    VOXEL_SURFACE = "voxel_surface"
    #: voxelizes the volume and returns all ocupioed voxel positions.
    VOXEL_VOLUME = "voxel_volume"
    #: voxelizes the volume and returns voxel positions that are inside the surface of the geometry
    VOXEL_VOLUME_INSIDE = "voxel_volume_inside"
    #: voxelizes the volume and returns voxel positions that are inside the surface,
    #: along with surface sampled positions
    VOXEL_VOLUME_SAMPLE_SURFACE = "voxel_volume_sample_surface"


def sample_even_fit_mesh(
    mesh: trimesh.Trimesh,
    n_spheres: int,
    sphere_radius: float,
) -> Tuple[numpy.array, List[float]]:
    """Sample even points on the surface of the mesh and return them with the given radius.

    Args:
        mesh: Mesh to sample points from.
        n_spheres: Number of spheres to sample.
        sphere_radius: Sphere radius.

    Returns:
        Tuple of points and radius.
    """

    n_pts = trimesh.sample.sample_surface_even(mesh, n_spheres)[0]
    n_radius = [sphere_radius for _ in range(len(n_pts))]
    return n_pts, n_radius


def get_voxel_pitch(mesh: trimesh.Trimesh, n_cubes: int) -> float:
    """Get the pitch of the voxel grid based on the mesh and number of cubes.

    Args:
        mesh: Mesh to get the pitch from.
        n_cubes: Number of voxels to fit.

    Returns:
        float: Pitch of the voxel grid.
    """
    d = mesh.extents
    cube_volume = d[0] * d[1] * d[2]
    v = mesh.volume
    if v > 0:
        occupancy = 1.0 - ((cube_volume - v) / cube_volume)
    else:
        log_warn("sphere_fit: occupancy test failed, assuming cuboid volume")
        occupancy = 1.0

    # given the extents, find the radius to get number of spheres:
    pitch = (occupancy * cube_volume / n_cubes) ** (1 / 3)
    return pitch


def voxel_fit_surface_mesh(
    mesh: trimesh.Trimesh,
    n_spheres: int,
    sphere_radius: float,
    voxelize_method: str = "ray",
) -> Tuple[numpy.array, List[float]]:
    """Get voxel grid from mesh and fit spheres to the surface.

    Args:
        mesh: Input mesh.
        n_spheres: Number of spheres to fit.
        sphere_radius: Radius of the spheres.
        voxelize_method: TriMesh Voxelization method. Defaults to "ray".

    Returns:
        Tuple of sphere positions and sphere radius.
    """
    pts, rad = get_voxelgrid_from_mesh(mesh, n_spheres, voxelize_method)
    if pts is None:
        return pts, rad
    pr = trimesh.proximity.ProximityQuery(mesh)
    if len(pts) <= n_spheres:
        surface_pts, _, _ = pr.on_surface(pts)
        n_radius = [sphere_radius for _ in range(len(surface_pts))]
        return surface_pts, n_radius

    # compute signed distance:
    dist = pr.signed_distance(pts)

    # calculate distance to boundary:
    dist = numpy.abs(dist - rad)
    # get the first n points closest to boundary:
    _, idx = torch.topk(torch.as_tensor(dist), k=n_spheres, largest=False)

    n_pts = pts[idx]
    n_radius = [sphere_radius for _ in range(len(n_pts))]
    return n_pts, n_radius


def get_voxelgrid_from_mesh(
    mesh: trimesh.Trimesh, n_spheres: int, voxelize_method: str = "ray"
) -> Tuple[Union[numpy.array, None], Union[numpy.array, None]]:
    """Get voxel grid from mesh using :py:func:`trimesh.voxel.creation.voxelize`.

    Args:
        mesh: Input mesh.
        n_spheres: Number of voxels to fit.
        voxelize_method: Voxelize method. Defaults to "ray".

    Returns:
        Tuple of occupied voxels and side of voxels (length of cube). Returns [None, None] if
            voxelization fails.

    """
    pitch = get_voxel_pitch(mesh, n_spheres)
    radius = pitch / 2.0
    try:
        voxel = voxelize(mesh, pitch, voxelize_method)
        voxel = voxel.fill("base")
        pts = voxel.points
        rad = numpy.ravel([radius for _ in range(len(pts))])
    except:
        log_warn("voxelization failed")
        pts = rad = None
    return pts, rad


def voxel_fit_mesh(
    mesh: trimesh.Trimesh,
    n_spheres: int,
    surface_sphere_radius: float,
    voxelize_method: str = "ray",
) -> Tuple[numpy.array, List[float]]:
    """Voxelize mesh, fit spheres to volume and near surface. Return the fitted spheres.

    Args:
        mesh: Input mesh.
        n_spheres: Number of spheres to fit.
        surface_sphere_radius: Radius of the spheres on the surface. This radius will be added
            to points on the surface of the mesh, causing the spheres to inflate the mesh volume
            by this amount.
        voxelize_method: Voxelization method to use, select from
            :py:func:`trimesh.voxel.creation.voxelize`.

    Returns:
        Tuple of sphere positions and their radius.
    """
    pts, rad = get_voxelgrid_from_mesh(mesh, n_spheres, voxelize_method)
    if pts is None:
        return pts, rad

    # compute signed distance:
    pr = trimesh.proximity.ProximityQuery(mesh)
    dist = pr.signed_distance(pts)

    # calculate distance to boundary:
    dist = dist - rad

    # all negative values are outside the mesh:
    idx = dist < 0.0
    surface_pts, _, _ = pr.on_surface(pts[idx])
    pts[idx] = surface_pts
    rad[idx] = surface_sphere_radius
    if len(pts) > n_spheres:
        # remove some surface pts:
        inside_idx = dist >= 0.0
        inside_pts = pts[inside_idx]
        if len(inside_pts) < n_spheres:
            new_pts = numpy.zeros((n_spheres, 3))
            new_pts[: len(inside_pts)] = inside_pts
            new_radius = numpy.zeros(n_spheres)
            new_radius[: len(inside_pts)] = rad[inside_idx]

            new_pts[len(inside_pts) :] = surface_pts[: n_spheres - len(inside_pts)]
            new_radius[len(inside_pts) :] = surface_sphere_radius
            pts = new_pts
            rad = new_radius

    n_pts = pts
    n_radius = rad.tolist()

    return n_pts, n_radius


def voxel_fit_volume_inside_mesh(
    mesh: trimesh.Trimesh,
    n_spheres: int,
    voxelize_method: str = "ray",
) -> Tuple[numpy.ndarray, numpy.array]:
    """Voxelize mesh, fit spheres to volume. Return the fitted spheres.

    Args:
        mesh: Input mesh.
        n_spheres: Number of spheres to fit.
        voxelize_method: Voxelization method to use, select from
            :py:func:`trimesh.voxel.creation.voxelize`.

    Returns:
        Tuple of sphere positions and their radius.
    """
    pts, rad = get_voxelgrid_from_mesh(mesh, 2 * n_spheres, voxelize_method)
    if pts is None:
        return pts, rad
    # compute signed distance:
    pr = trimesh.proximity.ProximityQuery(mesh)
    dist = pr.signed_distance(pts)

    # calculate distance to boundary:
    dist = dist - rad
    # all negative values are outside the mesh:
    idx = dist > 0.0
    n_pts = pts[idx]
    n_radius = rad[idx].tolist()
    return n_pts, n_radius


def voxel_fit_volume_sample_surface_mesh(
    mesh: trimesh.Trimesh,
    n_spheres: int,
    surface_sphere_radius: float,
    voxelize_method: str = "ray",
) -> Tuple[numpy.ndarray, numpy.array]:
    """Voxelize mesh, fit spheres to volume, and sample surface for points.

    Args:
        mesh: Input mesh.
        n_spheres: Number of spheres to fit.
        surface_sphere_radius: Radius of the spheres on the surface. This radius will be added
            to points on the surface of the mesh, causing the spheres to inflate the mesh volume
            by this amount.
        voxelize_method: Voxelization method to use, select from
            :py:func:`trimesh.voxel.creation.voxelize`.
    Returns:
        Tuple of sphere positions and their radius.
    """
    pts, rad = voxel_fit_volume_inside_mesh(mesh, 0.75 * n_spheres, voxelize_method)
    if pts is None:
        return pts, rad
    # compute surface points:
    if len(pts) >= n_spheres:
        return pts, rad

    sample_count = n_spheres - (len(pts))

    surface_sample_pts, sample_radius = sample_even_fit_mesh(
        mesh, sample_count, surface_sphere_radius
    )
    pts = numpy.concatenate([pts, surface_sample_pts])
    rad = numpy.concatenate([rad, sample_radius])
    return pts, rad


def fit_spheres_to_mesh(
    mesh: trimesh.Trimesh,
    n_spheres: int,
    surface_sphere_radius: float = 0.01,
    fit_type: SphereFitType = SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
    voxelize_method: str = "ray",
) -> Tuple[numpy.ndarray, numpy.array]:
    """Approximate a mesh with spheres. See :ref:`attach_object_note` for more details.

    Args:
        mesh: Input mesh.
        n_spheres: Number of spheres to fit.
        surface_sphere_radius: Radius of the spheres on the surface. This radius will be added
            to points on the surface of the mesh, causing the spheres to inflate the mesh volume
            by this amount.
        fit_type: Sphere fit type, select from :py:class:`~SphereFitType`.
        voxelize_method: Voxelization method to use, select from
            :py:func:`trimesh.voxel.creation.voxelize`.

    Returns:
        Tuple of spehre positions and their radius.
    """
    n_pts = n_radius = None
    if fit_type == SphereFitType.SAMPLE_SURFACE:
        n_pts, n_radius = sample_even_fit_mesh(mesh, n_spheres, surface_sphere_radius)
    elif fit_type == SphereFitType.VOXEL_SURFACE:
        n_pts, n_radius = voxel_fit_surface_mesh(
            mesh, n_spheres, surface_sphere_radius, voxelize_method
        )
    elif fit_type == SphereFitType.VOXEL_VOLUME_INSIDE:
        n_pts, n_radius = voxel_fit_volume_inside_mesh(mesh, n_spheres, voxelize_method)
    elif fit_type == SphereFitType.VOXEL_VOLUME:
        n_pts, n_radius = voxel_fit_mesh(mesh, n_spheres, surface_sphere_radius, voxelize_method)
    elif fit_type == SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE:
        n_pts, n_radius = voxel_fit_volume_sample_surface_mesh(
            mesh, n_spheres, surface_sphere_radius, voxelize_method
        )
    if n_pts is None or len(n_pts) < 1:
        log_warn("sphere_fit: falling back to sampling surface")
        n_pts, n_radius = sample_even_fit_mesh(mesh, n_spheres, surface_sphere_radius)

    if n_pts is not None and len(n_pts) > n_spheres:
        samples = torch.as_tensor(n_pts)
        dist = torch.linalg.norm(samples - torch.mean(samples, dim=-1).unsqueeze(1), dim=-1)
        _, knn_i = dist.topk(n_spheres, largest=True)
        n_pts = samples[knn_i].cpu().numpy()
        n_radius = numpy.ravel(n_radius)[knn_i.cpu().flatten().tolist()].tolist()
    return n_pts, n_radius
