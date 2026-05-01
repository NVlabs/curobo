# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Generated-observation correctness tests for the high-level Mapper API."""

from __future__ import annotations

import pytest
import torch

from curobo._src.perception.mapper.mapper import Mapper
from curobo._src.perception.mapper.mapper_cfg import MapperCfg
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose
from curobo._src.util.warp import init_warp


IMAGE_H = 32
IMAGE_W = 40
VOXEL_SIZE = 0.02
PLANE_Z = 1.0


@pytest.fixture(scope="module")
def warp_init():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for block-sparse mapper integration tests")
    init_warp()
    return True


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for block-sparse mapper integration tests")
    return "cuda:0"


def _intrinsics(
    device: str,
    image_height: int = IMAGE_H,
    image_width: int = IMAGE_W,
    focal: float = 80.0,
    cx: float | None = None,
    cy: float | None = None,
) -> torch.Tensor:
    if cx is None:
        cx = image_width / 2.0
    if cy is None:
        cy = image_height / 2.0
    return torch.tensor(
        [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )


def _identity_pose(device: str, num_cameras: int = 1) -> Pose:
    position = torch.zeros((num_cameras, 3), dtype=torch.float32, device=device)
    quaternion = torch.zeros((num_cameras, 4), dtype=torch.float32, device=device)
    quaternion[:, 0] = 1.0
    return Pose(position=position, quaternion=quaternion)


def _make_mapper(
    device: str,
    *,
    image_height: int = IMAGE_H,
    image_width: int = IMAGE_W,
    num_cameras: int = 1,
    feature_dim: int = 0,
    feature_integration_kernel: str = "auto",
    feature_channels_per_thread: int = 8,
    support_capacity: int = 8,
    decay_factor: float = 1.0,
    frustum_decay_factor: float = 1.0,
    profile_kernel_timings: bool = False,
) -> Mapper:
    feature_grid_kwargs = {}
    if feature_dim > 0:
        feature_grid_kwargs = {
            "feature_grid_height": 7,
            "feature_grid_width": 11,
        }
    return Mapper(
        MapperCfg(
            extent_meters_xyz=(1.0, 0.8, 0.8),
            extent_esdf_meters_xyz=(1.0, 0.8, 0.8),
            voxel_size=VOXEL_SIZE,
            esdf_voxel_size=0.04,
            grid_center=torch.tensor([0.0, 0.0, PLANE_Z], dtype=torch.float32),
            truncation_distance=0.04,
            depth_minimum_distance=0.1,
            depth_maximum_distance=3.0,
            decay_factor=decay_factor,
            frustum_decay_factor=frustum_decay_factor,
            image_height=image_height,
            image_width=image_width,
            num_cameras=num_cameras,
            block_size=2,
            max_support_pixels_per_block_camera=support_capacity,
            feature_dim=feature_dim,
            **feature_grid_kwargs,
            feature_channels_per_thread=feature_channels_per_thread,
            feature_integration_kernel=feature_integration_kernel,
            profile_integration_kernel_timings=profile_kernel_timings,
            device=device,
        )
    )


def _observation(
    *,
    device: str,
    depth: torch.Tensor,
    rgb: torch.Tensor,
    intrinsics: torch.Tensor | None = None,
    feature_grid: torch.Tensor | None = None,
    pose: Pose | None = None,
) -> CameraObservation:
    if depth.ndim == 2:
        depth = depth.unsqueeze(0)
    if rgb.ndim == 3:
        rgb = rgb.unsqueeze(0)
    if intrinsics is None:
        intrinsics = _intrinsics(device, depth.shape[-2], depth.shape[-1])
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0)
    if pose is None:
        pose = _identity_pose(device, depth.shape[0])
    return CameraObservation(
        depth_image=depth,
        rgb_image=rgb,
        pose=pose,
        intrinsics=intrinsics,
        feature_grid=feature_grid,
    )


def _constant_plane_observation(
    device: str,
    *,
    image_height: int = IMAGE_H,
    image_width: int = IMAGE_W,
    rgb_value: tuple[int, int, int] = (128, 128, 128),
    feature_grid: torch.Tensor | None = None,
    num_cameras: int = 1,
) -> CameraObservation:
    depth = torch.full(
        (num_cameras, image_height, image_width),
        PLANE_Z,
        dtype=torch.float32,
        device=device,
    )
    rgb = torch.empty(
        (num_cameras, image_height, image_width, 3),
        dtype=torch.uint8,
        device=device,
    )
    rgb[..., 0] = rgb_value[0]
    rgb[..., 1] = rgb_value[1]
    rgb[..., 2] = rgb_value[2]
    return _observation(
        device=device,
        depth=depth,
        rgb=rgb,
        feature_grid=feature_grid,
        pose=_identity_pose(device, num_cameras),
        intrinsics=_intrinsics(device, image_height, image_width).view(1, 3, 3)
        .expand(num_cameras, 3, 3)
        .contiguous(),
    )


def _render_identity_plane_depth(
    device: str,
    *,
    normal: torch.Tensor,
    point: torch.Tensor,
    image_height: int = IMAGE_H,
    image_width: int = IMAGE_W,
) -> torch.Tensor:
    intr = _intrinsics(device, image_height, image_width)
    u = torch.arange(image_width, dtype=torch.float32, device=device)
    v = torch.arange(image_height, dtype=torch.float32, device=device)
    uu, vv = torch.meshgrid(u, v, indexing="xy")
    x_norm = (uu - intr[0, 2]) / intr[0, 0]
    y_norm = (vv - intr[1, 2]) / intr[1, 1]
    rays = torch.stack(
        (x_norm, y_norm, torch.ones_like(x_norm)),
        dim=-1,
    )
    normal = normal.to(device=device, dtype=torch.float32)
    normal = normal / normal.norm().clamp(min=1e-6)
    point = point.to(device=device, dtype=torch.float32)
    numerator = torch.dot(normal, point)
    denominator = (rays * normal.view(1, 1, 3)).sum(dim=-1).clamp(min=1e-4)
    return (numerator / denominator).contiguous()


def _constant_feature_grid(
    device: str,
    feature_vector: torch.Tensor,
    *,
    num_cameras: int = 1,
    feature_height: int = 7,
    feature_width: int = 11,
) -> torch.Tensor:
    feature_vector = feature_vector.to(device=device, dtype=torch.float16)
    return (
        feature_vector.view(1, 1, 1, -1)
        .expand(num_cameras, feature_height, feature_width, feature_vector.numel())
        .contiguous()
    )


def _spatial_feature_grid(
    device: str,
    *,
    feature_dim: int,
    feature_height: int = 7,
    feature_width: int = 11,
    num_cameras: int = 1,
) -> torch.Tensor:
    gy = torch.arange(feature_height, dtype=torch.float32, device=device).view(
        feature_height, 1
    )
    gx = torch.arange(feature_width, dtype=torch.float32, device=device).view(
        1, feature_width
    )
    gx_norm = gx / float(max(feature_width - 1, 1))
    gy_norm = gy / float(max(feature_height - 1, 1))
    channels = []
    for ch in range(feature_dim):
        if ch % 4 == 0:
            values = gx_norm.expand(feature_height, feature_width)
        elif ch % 4 == 1:
            values = gy_norm.expand(feature_height, feature_width)
        elif ch % 4 == 2:
            values = 0.5 * (gx_norm + gy_norm)
        else:
            values = torch.full(
                (feature_height, feature_width),
                -0.25 + 0.05 * ch,
                dtype=torch.float32,
                device=device,
            )
        channels.append(values)
    grid = torch.stack(channels, dim=-1).to(torch.float16)
    return grid.view(1, feature_height, feature_width, feature_dim).expand(
        num_cameras, feature_height, feature_width, feature_dim
    ).contiguous()


def _active_rgb(mapper: Mapper) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = mapper.tsdf.data
    n = int(data.num_allocated.item())
    block_rgb = data.block_rgb[:n].float()
    active = block_rgb[:, 3] > 0
    pool_idx = torch.arange(n, dtype=torch.int64, device=block_rgb.device)[active]
    normalized = block_rgb[active, :3] / block_rgb[active, 3:4].clamp(min=1e-6)
    return pool_idx, normalized, block_rgb[active, 3]


def _active_features(mapper: Mapper) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    data = mapper.tsdf.data
    n = int(data.num_allocated.item())
    weights = data.block_feature_weight[:n].float()
    active = weights > 0
    pool_idx = torch.arange(n, dtype=torch.int64, device=weights.device)[active]
    normalized = data.block_features[:n].float()[active] / weights[active].view(-1, 1)
    return pool_idx, normalized, weights[active]


def _sort_active_features_by_coord(mapper: Mapper) -> tuple[torch.Tensor, torch.Tensor]:
    pool_idx, normalized, _ = _active_features(mapper)
    coords = mapper.tsdf.data.block_coords.view(-1, 3)[pool_idx].long()
    sort_key = coords[:, 0] * (2**40) + coords[:, 1] * (2**20) + coords[:, 2]
    order = sort_key.argsort()
    return coords[order], normalized[order]


def _expected_rgb_from_support(
    mapper: Mapper,
    rgb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    voxel_integrator = mapper.integrator._tsdf_integrator._integrator
    n_visible = int(voxel_integrator.visible_count.item())
    max_pool = int(mapper.tsdf.data.num_allocated.item())
    expected_sum = torch.zeros((max_pool, 3), dtype=torch.float32, device=rgb.device)
    expected_weight = torch.zeros(max_pool, dtype=torch.float32, device=rgb.device)
    for vis_idx in range(n_visible):
        pool_idx = int(voxel_integrator.pool_indices[vis_idx].item())
        if pool_idx < 0:
            continue
        for cam_i in range(rgb.shape[0]):
            count = int(voxel_integrator.support_counts[vis_idx, cam_i].item())
            count = min(count, mapper.config.max_support_pixels_per_block_camera)
            if count <= 0:
                continue
            pixels = voxel_integrator.support_pixels[vis_idx, cam_i, :count].long()
            py = pixels // mapper.config.image_width
            px = pixels - py * mapper.config.image_width
            valid = (
                (py >= 0)
                & (py < mapper.config.image_height)
                & (px >= 0)
                & (px < mapper.config.image_width)
            )
            if valid.any():
                values = rgb[cam_i, py[valid], px[valid]].float() / 255.0
                expected_sum[pool_idx] += values.sum(dim=0)
                expected_weight[pool_idx] += float(values.shape[0])
    return expected_sum, expected_weight


def _expected_features_from_support(
    mapper: Mapper,
    feature_grid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    voxel_integrator = mapper.integrator._tsdf_integrator._integrator
    n_visible = int(voxel_integrator.visible_count.item())
    max_pool = int(mapper.tsdf.data.num_allocated.item())
    feature_dim = feature_grid.shape[-1]
    expected_sum = torch.zeros(
        (max_pool, feature_dim),
        dtype=torch.float32,
        device=feature_grid.device,
    )
    expected_weight = torch.zeros(max_pool, dtype=torch.float32, device=feature_grid.device)
    feature_h = feature_grid.shape[1]
    feature_w = feature_grid.shape[2]
    for vis_idx in range(n_visible):
        pool_idx = int(voxel_integrator.pool_indices[vis_idx].item())
        if pool_idx < 0:
            continue
        for cam_i in range(feature_grid.shape[0]):
            count = int(voxel_integrator.support_counts[vis_idx, cam_i].item())
            count = min(count, mapper.config.max_support_pixels_per_block_camera)
            if count <= 0:
                continue
            pixels = voxel_integrator.support_pixels[vis_idx, cam_i, :count].long()
            py = pixels // mapper.config.image_width
            px = pixels - py * mapper.config.image_width
            valid = (
                (py >= 0)
                & (py < mapper.config.image_height)
                & (px >= 0)
                & (px < mapper.config.image_width)
            )
            if valid.any():
                gy = ((py[valid] * feature_h) // mapper.config.image_height).clamp(
                    min=0, max=feature_h - 1
                )
                gx = ((px[valid] * feature_w) // mapper.config.image_width).clamp(
                    min=0, max=feature_w - 1
                )
                values = feature_grid[cam_i, gy, gx].float()
                expected_sum[pool_idx] += values.sum(dim=0)
                expected_weight[pool_idx] += float(values.shape[0])
    return expected_sum, expected_weight


def _sample_esdf_nearest(mapper: Mapper, points: torch.Tensor) -> torch.Tensor:
    grid = mapper.compute_esdf()
    field = grid.feature_tensor
    origin = torch.tensor(grid.pose[:3], dtype=torch.float32, device=points.device)
    dims = torch.tensor(grid.dims, dtype=torch.float32, device=points.device)
    idx = torch.round((points - origin + 0.5 * dims) / grid.voxel_size).long()
    idx[:, 0].clamp_(0, field.shape[0] - 1)
    idx[:, 1].clamp_(0, field.shape[1] - 1)
    idx[:, 2].clamp_(0, field.shape[2] - 1)
    return field[idx[:, 0], idx[:, 1], idx[:, 2]].float()


def test_plane_surface_and_esdf_distance_are_voxel_accurate(warp_init, device):
    mapper = _make_mapper(device)
    obs = _constant_plane_observation(device, rgb_value=(128, 128, 128))
    mapper.integrate(obs)

    stats = mapper.get_stats(scan_pool=False)
    assert stats["last_integration"]["num_visible_blocks"] > 0
    assert stats["last_integration_kernel_timings_ms"] == {}

    surface = mapper.extract_occupied_voxels(surface_only=True, sdf_threshold=0.04)
    assert len(surface) > 0
    assert torch.isfinite(surface.centers).all()
    median_z = surface.centers[:, 2].median()
    assert torch.abs(median_z - PLANE_Z) <= 2.0 * VOXEL_SIZE

    points = torch.tensor(
        [
            [0.0, 0.0, PLANE_Z],
            [0.0, 0.0, PLANE_Z - 0.12],
            [0.0, 0.0, PLANE_Z + 0.12],
        ],
        dtype=torch.float32,
        device=device,
    )
    distances = _sample_esdf_nearest(mapper, points)
    assert distances[0] <= 2.0 * VOXEL_SIZE
    torch.testing.assert_close(
        distances[1:],
        torch.full((2,), 0.12, dtype=torch.float32, device=device),
        atol=0.05,
        rtol=0.0,
    )


def test_tilted_plane_surface_matches_analytic_plane(warp_init, device):
    normal = torch.tensor([0.25, -0.10, 1.0], dtype=torch.float32, device=device)
    normal = normal / normal.norm().clamp(min=1e-6)
    point = torch.tensor([0.0, 0.0, PLANE_Z], dtype=torch.float32, device=device)
    depth = _render_identity_plane_depth(device, normal=normal, point=point)
    rgb = torch.full((IMAGE_H, IMAGE_W, 3), 127, dtype=torch.uint8, device=device)

    mapper = _make_mapper(device)
    mapper.integrate(_observation(device=device, depth=depth, rgb=rgb))
    surface = mapper.extract_occupied_voxels(surface_only=True, sdf_threshold=0.04)

    assert len(surface) > 0
    plane_distance = torch.abs((surface.centers - point).matmul(normal))
    assert plane_distance.median() <= 2.0 * VOXEL_SIZE
    assert torch.quantile(plane_distance, 0.90) <= 4.0 * VOXEL_SIZE


def test_constant_rgb_accumulates_across_frames(warp_init, device):
    mapper = _make_mapper(device)
    mapper.integrate(_constant_plane_observation(device, rgb_value=(255, 0, 0)))
    pool_idx, rgb_first, weights_first = _active_rgb(mapper)
    assert pool_idx.numel() > 0
    expected_red = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32, device=device)
    torch.testing.assert_close(
        rgb_first,
        expected_red.view(1, 3).expand_as(rgb_first),
        atol=0.03,
        rtol=0.0,
    )

    mapper.integrate(_constant_plane_observation(device, rgb_value=(0, 0, 255)))
    _, rgb_second, weights_second = _active_rgb(mapper)
    expected_purple = torch.tensor([0.5, 0.0, 0.5], dtype=torch.float32, device=device)
    torch.testing.assert_close(
        rgb_second,
        expected_purple.view(1, 3).expand_as(rgb_second),
        atol=0.03,
        rtol=0.0,
    )
    assert weights_second.sum() > weights_first.sum()


def test_gradient_rgb_matches_stored_support_pixel_reference(warp_init, device):
    mapper = _make_mapper(device, support_capacity=8)
    x = torch.arange(IMAGE_W, dtype=torch.float32, device=device).view(1, IMAGE_W)
    y = torch.arange(IMAGE_H, dtype=torch.float32, device=device).view(IMAGE_H, 1)
    rgb = torch.empty((IMAGE_H, IMAGE_W, 3), dtype=torch.uint8, device=device)
    rgb[..., 0] = torch.round(x.expand(IMAGE_H, IMAGE_W) * 255.0 / (IMAGE_W - 1)).to(
        torch.uint8
    )
    rgb[..., 1] = torch.round(y.expand(IMAGE_H, IMAGE_W) * 255.0 / (IMAGE_H - 1)).to(
        torch.uint8
    )
    rgb[..., 2] = 64
    depth = torch.full((IMAGE_H, IMAGE_W), PLANE_Z, dtype=torch.float32, device=device)

    mapper.integrate(_observation(device=device, depth=depth, rgb=rgb))
    expected_sum, expected_weight = _expected_rgb_from_support(mapper, rgb.unsqueeze(0))
    pool_idx, normalized, _ = _active_rgb(mapper)
    expected_active = expected_sum[pool_idx] / expected_weight[pool_idx].view(-1, 1)

    assert (expected_weight[pool_idx] > 0).all()
    torch.testing.assert_close(normalized, expected_active, atol=0.03, rtol=0.0)


@pytest.mark.parametrize("feature_kernel", ["grouped", "tiled"])
def test_constant_features_include_trailing_channels(warp_init, device, feature_kernel):
    feature_vector = torch.tensor(
        [-0.75, -0.25, 0.0, 0.25, 0.50, 0.75, 1.0],
        dtype=torch.float32,
        device=device,
    )
    feature_grid = _constant_feature_grid(device, feature_vector)
    mapper = _make_mapper(
        device,
        feature_dim=feature_vector.numel(),
        feature_integration_kernel=feature_kernel,
        feature_channels_per_thread=5,
    )

    mapper.integrate(
        _constant_plane_observation(device, rgb_value=(0, 0, 0), feature_grid=feature_grid)
    )
    _, normalized, weights = _active_features(mapper)

    assert weights.numel() > 0
    torch.testing.assert_close(
        normalized,
        feature_vector.view(1, -1).expand_as(normalized),
        atol=0.03,
        rtol=0.0,
    )


@pytest.mark.parametrize("feature_kernel", ["grouped", "tiled"])
def test_spatial_features_match_stored_support_pixel_reference(
    warp_init,
    device,
    feature_kernel,
):
    feature_dim = 9
    feature_grid = _spatial_feature_grid(device, feature_dim=feature_dim)
    mapper = _make_mapper(
        device,
        feature_dim=feature_dim,
        feature_integration_kernel=feature_kernel,
        support_capacity=8,
    )

    mapper.integrate(
        _constant_plane_observation(device, rgb_value=(0, 0, 0), feature_grid=feature_grid)
    )
    expected_sum, expected_weight = _expected_features_from_support(mapper, feature_grid)
    pool_idx, normalized, _ = _active_features(mapper)
    expected_active = expected_sum[pool_idx] / expected_weight[pool_idx].view(-1, 1)

    assert (expected_weight[pool_idx] > 0).all()
    torch.testing.assert_close(normalized, expected_active, atol=0.04, rtol=0.0)


def test_grouped_tiled_feature_outputs_match_for_same_scene(warp_init, device):
    feature_dim = 9
    feature_vector = torch.linspace(
        -1.0,
        1.0,
        feature_dim,
        dtype=torch.float32,
        device=device,
    )
    feature_grid = _constant_feature_grid(device, feature_vector)
    obs = _constant_plane_observation(
        device,
        rgb_value=(0, 0, 0),
        feature_grid=feature_grid,
    )
    grouped = _make_mapper(device, feature_dim=feature_dim, feature_integration_kernel="grouped")
    tiled = _make_mapper(device, feature_dim=feature_dim, feature_integration_kernel="tiled")

    grouped.integrate(obs)
    tiled.integrate(obs)
    grouped_coords, grouped_features = _sort_active_features_by_coord(grouped)
    tiled_coords, tiled_features = _sort_active_features_by_coord(tiled)

    assert grouped_coords.shape == tiled_coords.shape
    torch.testing.assert_close(grouped_coords, tiled_coords)
    torch.testing.assert_close(grouped_features, tiled_features, atol=0.04, rtol=0.0)


def test_time_decay_reduces_tsdf_rgb_and_feature_weights(warp_init, device):
    feature_vector = torch.tensor([0.2, -0.1, 0.8], dtype=torch.float32, device=device)
    feature_grid = _constant_feature_grid(device, feature_vector)
    mapper = _make_mapper(
        device,
        feature_dim=feature_vector.numel(),
        decay_factor=0.5,
        frustum_decay_factor=1.0,
    )
    mapper.integrate(
        _constant_plane_observation(device, rgb_value=(64, 128, 192), feature_grid=feature_grid)
    )
    n = int(mapper.tsdf.data.num_allocated.item())
    before_tsdf = mapper.tsdf.data.block_data[:n, :, 1].float().sum()
    before_rgb = mapper.tsdf.data.block_rgb[:n, 3].float().sum()
    before_feature = mapper.tsdf.data.block_feature_weight[:n].float().sum()

    empty_depth = torch.zeros((IMAGE_H, IMAGE_W), dtype=torch.float32, device=device)
    empty_rgb = torch.zeros((IMAGE_H, IMAGE_W, 3), dtype=torch.uint8, device=device)
    mapper.integrate(
        _observation(
            device=device,
            depth=empty_depth,
            rgb=empty_rgb,
            feature_grid=feature_grid,
        )
    )
    after_tsdf = mapper.tsdf.data.block_data[:n, :, 1].float().sum()
    after_rgb = mapper.tsdf.data.block_rgb[:n, 3].float().sum()
    after_feature = mapper.tsdf.data.block_feature_weight[:n].float().sum()

    assert before_tsdf > 0
    expected_decay = torch.tensor(0.5, dtype=torch.float32, device=device)
    torch.testing.assert_close(
        after_tsdf / before_tsdf,
        expected_decay,
        atol=0.08,
        rtol=0.0,
    )
    torch.testing.assert_close(
        after_rgb / before_rgb,
        expected_decay,
        atol=0.08,
        rtol=0.0,
    )
    torch.testing.assert_close(
        after_feature / before_feature,
        expected_decay,
        atol=0.08,
        rtol=0.0,
    )


def test_clear_region_removes_stale_rgb_and_features(warp_init, device):
    feature_a = torch.tensor([1.0, 0.0, -0.5], dtype=torch.float32, device=device)
    feature_b = torch.tensor([-0.5, 0.5, 1.0], dtype=torch.float32, device=device)
    mapper = _make_mapper(device, feature_dim=feature_a.numel())
    mapper.integrate(
        _constant_plane_observation(
            device,
            rgb_value=(255, 0, 0),
            feature_grid=_constant_feature_grid(device, feature_a),
        )
    )

    n_cleared = mapper.clear_region(
        torch.tensor([-1.0, -1.0, 0.5], dtype=torch.float32, device=device),
        torch.tensor([1.0, 1.0, 1.5], dtype=torch.float32, device=device),
    )
    assert n_cleared > 0
    n = int(mapper.tsdf.data.num_allocated.item())
    assert mapper.tsdf.data.block_rgb[:n, 3].float().sum() == 0
    assert mapper.tsdf.data.block_feature_weight[:n].float().sum() == 0

    mapper.integrate(
        _constant_plane_observation(
            device,
            rgb_value=(0, 0, 255),
            feature_grid=_constant_feature_grid(device, feature_b),
        )
    )
    _, rgb_normalized, _ = _active_rgb(mapper)
    _, feature_normalized, _ = _active_features(mapper)
    expected_blue = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
    torch.testing.assert_close(
        rgb_normalized,
        expected_blue.view(1, 3).expand_as(rgb_normalized),
        atol=0.03,
        rtol=0.0,
    )
    torch.testing.assert_close(
        feature_normalized,
        feature_b.view(1, -1).expand_as(feature_normalized),
        atol=0.03,
        rtol=0.0,
    )


def test_stats_report_last_integration_and_kernel_timings(warp_init, device):
    mapper = _make_mapper(device, profile_kernel_timings=True)
    mapper.integrate(_constant_plane_observation(device))
    stats = mapper.get_stats(scan_pool=False)

    assert stats["frame_count"] == 1
    assert stats["last_integration"]["num_visible_blocks"] > 0
    assert stats["last_integration"]["support_overflow_count"] >= 0
    assert stats["last_integration"]["profile_kernel_timings"] is True
    timings = stats["last_integration_kernel_timings_ms"]
    assert timings
    assert all(isinstance(value, float) and value >= 0.0 for value in timings.values())

    empty_depth = torch.zeros((IMAGE_H, IMAGE_W), dtype=torch.float32, device=device)
    empty_rgb = torch.zeros((IMAGE_H, IMAGE_W, 3), dtype=torch.uint8, device=device)
    mapper.integrate(_observation(device=device, depth=empty_depth, rgb=empty_rgb))
    stats = mapper.get_stats(scan_pool=False)
    assert stats["frame_count"] == 2
    assert stats["last_integration"]["num_visible_blocks"] == 0


def test_support_capacity_overflow_matches_stored_support_reference(warp_init, device):
    feature_vector = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32, device=device)
    feature_grid = _constant_feature_grid(device, feature_vector)
    mapper = _make_mapper(
        device,
        feature_dim=feature_vector.numel(),
        support_capacity=1,
        feature_integration_kernel="grouped",
    )
    obs = _constant_plane_observation(
        device,
        rgb_value=(32, 160, 224),
        feature_grid=feature_grid,
    )
    mapper.integrate(obs)

    stats = mapper.get_stats(scan_pool=False)
    assert stats["last_integration"]["support_overflow_count"] > 0

    expected_rgb_sum, expected_rgb_weight = _expected_rgb_from_support(mapper, obs.rgb_image)
    rgb_pool_idx, rgb_normalized, _ = _active_rgb(mapper)
    torch.testing.assert_close(
        rgb_normalized,
        expected_rgb_sum[rgb_pool_idx] / expected_rgb_weight[rgb_pool_idx].view(-1, 1),
        atol=0.03,
        rtol=0.0,
    )

    expected_feature_sum, expected_feature_weight = _expected_features_from_support(
        mapper,
        feature_grid,
    )
    feature_pool_idx, feature_normalized, _ = _active_features(mapper)
    torch.testing.assert_close(
        feature_normalized,
        expected_feature_sum[feature_pool_idx]
        / expected_feature_weight[feature_pool_idx].view(-1, 1),
        atol=0.03,
        rtol=0.0,
    )


def test_two_camera_mapper_averages_rgb_and_features(warp_init, device):
    feature_dim = 3
    feature_grid = torch.empty((2, 7, 11, feature_dim), dtype=torch.float16, device=device)
    feature_a = torch.tensor([1.0, 0.0, -0.5], dtype=torch.float16, device=device)
    feature_b = torch.tensor([-1.0, 0.5, 0.5], dtype=torch.float16, device=device)
    feature_grid[0] = feature_a.view(1, 1, -1)
    feature_grid[1] = feature_b.view(1, 1, -1)

    depth = torch.full((2, IMAGE_H, IMAGE_W), PLANE_Z, dtype=torch.float32, device=device)
    rgb = torch.zeros((2, IMAGE_H, IMAGE_W, 3), dtype=torch.uint8, device=device)
    rgb[0, ..., 0] = 255
    rgb[1, ..., 2] = 255
    obs = _observation(
        device=device,
        depth=depth,
        rgb=rgb,
        feature_grid=feature_grid,
        pose=_identity_pose(device, 2),
        intrinsics=_intrinsics(device).view(1, 3, 3).expand(2, 3, 3).contiguous(),
    )
    mapper = _make_mapper(
        device,
        num_cameras=2,
        feature_dim=feature_dim,
        feature_integration_kernel="tiled",
    )

    mapper.integrate(obs)
    _, rgb_normalized, _ = _active_rgb(mapper)
    _, feature_normalized, _ = _active_features(mapper)

    expected_rgb = torch.tensor([0.5, 0.0, 0.5], dtype=torch.float32, device=device)
    expected_feature = ((feature_a.float() + feature_b.float()) * 0.5).view(1, -1)
    torch.testing.assert_close(
        rgb_normalized,
        expected_rgb.view(1, 3).expand_as(rgb_normalized),
        atol=0.03,
        rtol=0.0,
    )
    torch.testing.assert_close(
        feature_normalized,
        expected_feature.expand_as(feature_normalized),
        atol=0.03,
        rtol=0.0,
    )
