# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""CPU/Torch reference tests for mapper LiDAR spherical projection."""

from __future__ import annotations

import math

import pytest
import torch

from curobo._src.perception.mapper.integrator_tsdf import (
    BlockSparseTSDFIntegrator,
    BlockSparseTSDFIntegratorCfg,
)
from curobo._src.types.lidar import LidarObservation
from curobo._src.types.pose import Pose
from curobo._src.util.warp import init_warp


def _as_angle_range(elevation_range_rad: tuple[float, float]) -> tuple[float, float]:
    min_elev, max_elev = elevation_range_rad
    if min_elev > max_elev:
        raise ValueError("elevation_range_rad must be [min_elevation, max_elevation]")
    return min_elev, max_elev


def _lidar_pixel_to_ray(
    u: torch.Tensor,
    v: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    elevation_range_rad: tuple[float, float],
) -> torch.Tensor:
    """Convert LiDAR range-image pixel centers to unit rays.

    This mirrors the design convention for cuRobo's LiDAR path:
    azimuth(u) = u * 2*pi / W - pi, row 0 is max elevation, and row H - 1 is
    min elevation. For H == 1, the scan is planar and min/max elevation must
    be identical.
    """
    min_elev, max_elev = _as_angle_range(elevation_range_rad)
    if image_height == 1:
        if not math.isclose(min_elev, max_elev, abs_tol=1e-7):
            raise ValueError("planar H == 1 LiDAR requires a fixed elevation")
        elevation = torch.full_like(v, min_elev, dtype=torch.float32)
    else:
        v_float = v.to(dtype=torch.float32)
        elevation = max_elev - v_float * (max_elev - min_elev) / float(image_height - 1)

    azimuth = u.to(dtype=torch.float32) * (2.0 * math.pi / float(image_width)) - math.pi
    cos_elev = torch.cos(elevation)
    ray = torch.stack(
        (
            torch.cos(azimuth) * cos_elev,
            torch.sin(azimuth) * cos_elev,
            torch.sin(elevation),
        ),
        dim=-1,
    )
    return ray / ray.norm(dim=-1, keepdim=True).clamp(min=1e-7)


def _lidar_point_to_pixel(
    points_lidar: torch.Tensor,
    *,
    image_height: int,
    image_width: int,
    elevation_range_rad: tuple[float, float],
    valid_range_m: tuple[float, float] = (0.0, float("inf")),
    elevation_tolerance_rad: float = 1e-6,
    planar_elevation_tolerance_rad: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project LiDAR-frame points to floating pixel coordinates.

    Returns ``(u, v, range, valid)``. Azimuth wraps into ``[0, W)`` so the
    +pi/-pi boundary maps back to pixel column 0.
    """
    min_elev, max_elev = _as_angle_range(elevation_range_rad)
    points_lidar = points_lidar.to(dtype=torch.float32)
    range_m = points_lidar.norm(dim=-1)
    xy_norm = points_lidar[..., :2].norm(dim=-1)
    elevation = torch.atan2(points_lidar[..., 2], xy_norm)
    azimuth = torch.atan2(points_lidar[..., 1], points_lidar[..., 0])
    u = torch.remainder(
        (azimuth + math.pi) * (float(image_width) / (2.0 * math.pi)),
        image_width,
    )
    u = torch.where(u >= float(image_width) - 1e-5, torch.zeros_like(u), u)

    if image_height == 1:
        if not math.isclose(min_elev, max_elev, abs_tol=1e-7):
            raise ValueError("planar H == 1 LiDAR requires a fixed elevation")
        v = torch.zeros_like(u)
        elevation_valid = torch.abs(elevation - min_elev) <= planar_elevation_tolerance_rad
    else:
        v = (max_elev - elevation) * (float(image_height - 1) / (max_elev - min_elev))
        elevation_valid = (elevation >= min_elev - elevation_tolerance_rad) & (
            elevation <= max_elev + elevation_tolerance_rad
        )

    min_range, max_range = valid_range_m
    range_valid = (range_m >= min_range) & (range_m <= max_range)
    valid = torch.isfinite(range_m) & (range_m > 0.0) & range_valid & elevation_valid
    return u, v, range_m, valid


def test_lidar_projection_roundtrip_3d_asymmetric_elevation():
    """Pixel rays should project back to the same LiDAR image coordinates."""
    image_height = 5
    image_width = 16
    elevation_range = (-0.4, 0.7)
    u = torch.tensor([0.0, 3.0, 8.0, 15.0], dtype=torch.float32)
    v = torch.tensor([0.0, 1.0, 3.0, 4.0], dtype=torch.float32)

    rays = _lidar_pixel_to_ray(
        u,
        v,
        image_height=image_height,
        image_width=image_width,
        elevation_range_rad=elevation_range,
    )
    u_projected, v_projected, range_projected, valid = _lidar_point_to_pixel(
        rays * 2.5,
        image_height=image_height,
        image_width=image_width,
        elevation_range_rad=elevation_range,
        valid_range_m=(0.1, 10.0),
    )

    assert valid.all()
    torch.testing.assert_close(range_projected, torch.full_like(range_projected, 2.5))
    torch.testing.assert_close(u_projected, u, atol=1e-5, rtol=0.0)
    torch.testing.assert_close(v_projected, v, atol=1e-5, rtol=0.0)


def test_lidar_pixel_to_ray_elevation_row_order():
    """Row 0 is max elevation and the last row is min elevation."""
    rays = _lidar_pixel_to_ray(
        torch.zeros(2),
        torch.tensor([0.0, 4.0]),
        image_height=5,
        image_width=8,
        elevation_range_rad=(-0.25, 0.5),
    )
    elevations = torch.atan2(rays[:, 2], rays[:, :2].norm(dim=-1))

    torch.testing.assert_close(elevations[0], torch.tensor(0.5), atol=1e-6, rtol=0.0)
    torch.testing.assert_close(elevations[1], torch.tensor(-0.25), atol=1e-6, rtol=0.0)


def test_lidar_azimuth_wrap_projects_pi_boundary_to_column_zero():
    """The +pi/-pi azimuth seam should wrap to the first image column."""
    image_width = 32
    points = torch.tensor(
        [
            [-2.0, 0.0, 0.0],
            [-2.0, -1e-6, 0.0],
            [-2.0, 1e-6, 0.0],
        ],
        dtype=torch.float32,
    )
    u, v, _, valid = _lidar_point_to_pixel(
        points,
        image_height=3,
        image_width=image_width,
        elevation_range_rad=(-0.1, 0.1),
    )

    assert valid.all()
    assert torch.all((u >= 0.0) & (u < image_width))
    torch.testing.assert_close(v, torch.ones_like(v), atol=1e-5, rtol=0.0)
    assert min(u[0].item(), image_width - u[0].item()) < 1e-5
    assert min(u[1].item(), image_width - u[1].item()) < 1e-4
    assert min(u[2].item(), image_width - u[2].item()) < 1e-4


def test_lidar_planar_2d_projection_uses_fixed_elevation():
    """H == 1 should avoid H - 1 math and reject off-plane projections."""
    image_height = 1
    image_width = 12
    fixed_elevation = 0.2
    u = torch.tensor([0.0, 3.0, 6.0, 9.0], dtype=torch.float32)
    v = torch.zeros_like(u)

    rays = _lidar_pixel_to_ray(
        u,
        v,
        image_height=image_height,
        image_width=image_width,
        elevation_range_rad=(fixed_elevation, fixed_elevation),
    )
    elevations = torch.atan2(rays[:, 2], rays[:, :2].norm(dim=-1))
    torch.testing.assert_close(
        elevations,
        torch.full_like(elevations, fixed_elevation),
        atol=1e-6,
        rtol=0.0,
    )

    u_projected, v_projected, _, valid = _lidar_point_to_pixel(
        rays * 1.7,
        image_height=image_height,
        image_width=image_width,
        elevation_range_rad=(fixed_elevation, fixed_elevation),
    )
    assert valid.all()
    torch.testing.assert_close(u_projected, u, atol=1e-5, rtol=0.0)
    torch.testing.assert_close(v_projected, v, atol=1e-6, rtol=0.0)

    off_plane = rays.clone()
    off_plane[:, 2] += 0.05
    _, _, _, off_plane_valid = _lidar_point_to_pixel(
        off_plane,
        image_height=image_height,
        image_width=image_width,
        elevation_range_rad=(fixed_elevation, fixed_elevation),
    )
    assert not off_plane_valid.any()


def test_lidar_planar_2d_rejects_non_fixed_elevation_range():
    """A single-row scan cannot represent a vertical FOV interval."""
    with pytest.raises(ValueError, match="fixed elevation"):
        _lidar_pixel_to_ray(
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            image_height=1,
            image_width=8,
            elevation_range_rad=(-0.1, 0.1),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_lidar_integrator_allocates_blocks_from_planar_scan():
    """Smoke-test the end-to-end LiDAR integration path on CUDA."""
    init_warp()
    device = "cuda:0"
    cfg = BlockSparseTSDFIntegratorCfg(
        voxel_size=0.05,
        origin=torch.zeros(3, dtype=torch.float32),
        truncation_distance=0.1,
        grid_shape=(64, 64, 64),
        max_blocks=4096,
        block_size=4,
        image_height=4,
        image_width=4,
        num_cameras=1,
        lidar_num_sensors=1,
        lidar_image_height=1,
        lidar_image_width=32,
        device=device,
    )
    integrator = BlockSparseTSDFIntegrator(cfg)

    ranges = torch.full((1, 1, 32), 1.0, dtype=torch.float32, device=device)
    rgb = torch.zeros((1, 1, 32, 3), dtype=torch.uint8, device=device)
    rgb[..., 0] = 255
    observation = LidarObservation(
        range_image=ranges,
        rgb_image=rgb,
        pose=Pose(
            position=torch.zeros((1, 3), dtype=torch.float32, device=device),
            quaternion=torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]],
                dtype=torch.float32,
                device=device,
            ),
        ),
        valid_range_m=torch.tensor([[0.1, 3.0]], dtype=torch.float32, device=device),
        elevation_range_rad=torch.zeros((1, 2), dtype=torch.float32, device=device),
    )

    integrator.integrate(lidar_observation=observation)
    stats = integrator.get_stats(scan_pool=True)

    assert stats["frame_count"] == 1
    assert stats["num_allocated"] > 0
    assert stats["last_lidar_integration"]["num_visible_blocks"] > 0

    with pytest.raises(ValueError, match="fixed elevation"):
        _lidar_point_to_pixel(
            torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
            image_height=1,
            image_width=8,
            elevation_range_rad=(-0.1, 0.1),
        )
