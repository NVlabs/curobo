# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for VoxelGrid collision checking kernels.

Tests the sphere-vs-voxel collision pipeline at the kernel level:
VoxelGrid → VoxelData → VoxelDataWarp → sphere_obstacle_collision_kernel.
"""

# Standard Library
from importlib import import_module
from typing import Any

# Third Party
import pytest
import torch
import warp as wp

# CuRobo
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.wp_collision_common import (
    accumulate_collision,
    apply_collision_activation,
    load_sphere_query,
)
from curobo._src.geom.collision.wp_collision_kernel import sphere_obstacle_collision_kernel
from curobo._src.geom.collision.wp_sweep_collision_kernel import (
    swept_sphere_obstacle_collision_kernel,
)
from curobo._src.geom.data import OBSTACLE_SDF_MODULES
from curobo._src.geom.data.data_voxel import VoxelData
from curobo._src.geom.types import SceneCfg, VoxelGrid
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.warp import get_warp_device_stream

# =============================================================================
# Register compute_local_sdf overloads in THIS module's scope for standalone testing.
# This mirrors how wp_sweep_collision_kernel.py registers overloads.
# =============================================================================
_test_compute_local_sdf = None
_test_compute_local_sdf_with_grad = None
_test_is_obs_enabled = None
_test_load_obstacle_transform = None

for _module_path in OBSTACLE_SDF_MODULES:
    _data_module = import_module(_module_path)
    _test_is_obs_enabled = wp.func(getattr(_data_module, "is_obs_enabled"), module=__name__)
    _test_load_obstacle_transform = wp.func(
        getattr(_data_module, "load_obstacle_transform"), module=__name__
    )
    _test_compute_local_sdf = wp.func(
        getattr(_data_module, "compute_local_sdf"), module=__name__
    )
    _test_compute_local_sdf_with_grad = wp.func(
        getattr(_data_module, "compute_local_sdf_with_grad"), module=__name__
    )

del _module_path, _data_module


@wp.kernel(enable_backward=False)
def _test_sdf_only_kernel(
    obs_set: Any,
    query_points: wp.array(dtype=wp.vec3),
    env_idx: wp.int32,
    obs_local_idx: wp.int32,
    out_sdf: wp.array(dtype=wp.float32),
):
    """Minimal kernel: calls compute_local_sdf only, no sweep, no collision activation."""
    tid = wp.tid()
    if tid >= query_points.shape[0]:
        return
    local_pt = query_points[tid]
    sdf = _test_compute_local_sdf(obs_set, env_idx, obs_local_idx, local_pt)
    out_sdf[tid] = sdf


@wp.kernel(enable_backward=False)
def _test_sdf_with_grad_kernel(
    obs_set: Any,
    query_points: wp.array(dtype=wp.vec3),
    env_idx: wp.int32,
    obs_local_idx: wp.int32,
    out_sdf: wp.array(dtype=wp.float32),
):
    """Minimal kernel: calls compute_local_sdf_with_grad, extracts sdf only."""
    tid = wp.tid()
    if tid >= query_points.shape[0]:
        return
    local_pt = query_points[tid]
    result = _test_compute_local_sdf_with_grad(obs_set, env_idx, obs_local_idx, local_pt)
    out_sdf[tid] = result[0]


@wp.kernel(enable_backward=False)
def _test_both_sdf_kernel(
    obs_set: Any,
    query_points: wp.array(dtype=wp.vec3),
    env_idx: wp.int32,
    obs_local_idx: wp.int32,
    out_sdf_only: wp.array(dtype=wp.float32),
    out_sdf_grad: wp.array(dtype=wp.float32),
):
    """Kernel that calls BOTH compute_local_sdf and compute_local_sdf_with_grad.

    Reproduces the pattern used in the sweep kernel where both are called
    for the same point.
    """
    tid = wp.tid()
    if tid >= query_points.shape[0]:
        return
    local_pt = query_points[tid]

    sdf = _test_compute_local_sdf(obs_set, env_idx, obs_local_idx, local_pt)
    out_sdf_only[tid] = sdf

    if sdf < 0.0:
        result = _test_compute_local_sdf_with_grad(obs_set, env_idx, obs_local_idx, local_pt)
        out_sdf_grad[tid] = result[0]


_TEST_SWEEP_STEPS = wp.constant(3)


@wp.kernel(enable_backward=False)
def _test_sweep_pattern_kernel(
    obs_set: Any,
    query_points: wp.array(dtype=wp.vec3),
    env_idx: wp.int32,
    obs_local_idx: wp.int32,
    out_sdf: wp.array(dtype=wp.float32),
):
    """Single-loop sweep pattern: compute_local_sdf inside a for loop."""
    tid = wp.tid()
    if tid >= query_points.shape[0]:
        return

    local_pt = query_points[tid]

    result = _test_compute_local_sdf_with_grad(obs_set, env_idx, obs_local_idx, local_pt)
    sdf_current = result[0]

    cost_sum = wp.float32(0.0)
    jump = wp.float32(0.0)
    half_dist = wp.float32(0.1)

    for _ in range(_TEST_SWEEP_STEPS):
        if jump >= half_dist:
            break
        t = 1.0 - 0.5 * jump / half_dist
        interp_pt = t * local_pt

        sdf = _test_compute_local_sdf(obs_set, env_idx, obs_local_idx, interp_pt)
        penetration = -sdf + 0.01

        if penetration > 0.0:
            sdf_result = _test_compute_local_sdf_with_grad(obs_set, env_idx, obs_local_idx, interp_pt)
            cost_sum += penetration
            jump += penetration
        else:
            jump += wp.max(-penetration, 0.01)

    out_sdf[tid] = cost_sum


@wp.kernel(enable_backward=False)
def _test_dual_sweep_kernel(
    obs_set: Any,
    query_points: wp.array(dtype=wp.vec3),
    prev_points: wp.array(dtype=wp.vec3),
    next_points: wp.array(dtype=wp.vec3),
    env_idx: wp.int32,
    obs_local_idx: wp.int32,
    out_cost: wp.array(dtype=wp.float32),
    out_grad: wp.array(dtype=wp.float32),
):
    """Two loops with compute_local_sdf + compute_local_sdf_with_grad."""
    tid = wp.tid()
    if tid >= query_points.shape[0]:
        return

    local_current = query_points[tid]
    local_prev = prev_points[tid]
    local_next = next_points[tid]

    inv_t = _test_load_obstacle_transform(obs_set, env_idx, obs_local_idx)

    radius_adjusted = wp.float32(0.02)
    eta = wp.float32(0.02)

    cost_sum = wp.float32(0.0)
    grad_sum_local = wp.vec3(0.0, 0.0, 0.0)

    sdf_result = _test_compute_local_sdf_with_grad(obs_set, env_idx, obs_local_idx, local_current)
    penetration = -sdf_result[0] + radius_adjusted

    if penetration > 0.0:
        cost_sum += penetration
        grad_sum_local += wp.vec3(sdf_result[1], sdf_result[2], sdf_result[3])

    # Sweep toward prev (loop 1)
    half_dist = wp.length(local_prev - local_current) * 0.5
    inv_half_dist = 1.0 / wp.max(half_dist, 0.001)
    jump = wp.float32(0.0)
    for _ in range(_TEST_SWEEP_STEPS):
        if jump >= half_dist:
            break
        t = 1.0 - 0.5 * jump * inv_half_dist
        local_pt = t * local_current + (1.0 - t) * local_prev

        sdf = _test_compute_local_sdf(obs_set, env_idx, obs_local_idx, local_pt)
        penetration = -sdf + radius_adjusted

        if penetration > 0.0:
            sdf_result = _test_compute_local_sdf_with_grad(obs_set, env_idx, obs_local_idx, local_pt)
            cost_sum += penetration
            grad_sum_local += wp.vec3(sdf_result[1], sdf_result[2], sdf_result[3])
            jump += penetration
        else:
            if -penetration >= 1000.0:
                jump += radius_adjusted
            else:
                jump += wp.max(-penetration, radius_adjusted)

    # Sweep toward next (loop 2)
    half_dist = wp.length(local_next - local_current) * 0.5
    inv_half_dist = 1.0 / wp.max(half_dist, 0.001)
    jump = wp.float32(0.0)
    for _ in range(_TEST_SWEEP_STEPS):
        if jump >= half_dist:
            break
        t = 1.0 - 0.5 * jump * inv_half_dist
        local_pt = t * local_current + (1.0 - t) * local_next

        sdf = _test_compute_local_sdf(obs_set, env_idx, obs_local_idx, local_pt)
        penetration = -sdf + radius_adjusted

        if penetration > 0.0:
            sdf_result = _test_compute_local_sdf_with_grad(obs_set, env_idx, obs_local_idx, local_pt)
            cost_sum += penetration
            grad_sum_local += wp.vec3(sdf_result[1], sdf_result[2], sdf_result[3])
            jump += penetration
        else:
            if -penetration >= 1000.0:
                jump += radius_adjusted
            else:
                jump += wp.max(-penetration, radius_adjusted)

    out_cost[tid] = cost_sum
    out_grad[tid] = wp.length(grad_sum_local)


@wp.kernel(enable_backward=False)
def _test_full_sweep_kernel(
    obs_set: Any,
    spheres: wp.array(dtype=wp.vec4),
    weight: wp.array(dtype=wp.float32),
    activation_distance: wp.array(dtype=wp.float32),
    env_query_idx: wp.array(dtype=wp.int32),
    distance: wp.array(dtype=wp.float32),
    gradient: wp.array(dtype=wp.float32),
    batch_size: wp.int32,
    horizon: wp.int32,
    num_spheres: wp.int32,
    max_n_obs: wp.int32,
    use_multi_env: wp.uint8,
):
    """Exact replica of swept_sphere_obstacle_collision_kernel using compute_local_sdf.

    Identical signature, identical logic, identical function calls.
    Only difference: registered in test module instead of sweep kernel module.
    """
    tid = wp.tid()

    sph_flat_idx = tid / max_n_obs
    obs_local_idx = tid - sph_flat_idx * max_n_obs

    total_spheres = batch_size * horizon * num_spheres
    if sph_flat_idx >= total_spheres:
        return

    b_idx = sph_flat_idx / (horizon * num_spheres)
    h_idx = (sph_flat_idx - b_idx * horizon * num_spheres) / num_spheres

    env_idx = wp.int32(0)
    if use_multi_env == wp.uint8(1):
        env_idx = env_query_idx[b_idx]

    if not _test_is_obs_enabled(obs_set, env_idx, obs_local_idx):
        return

    eta = activation_distance[0]
    w = weight[0]

    query = load_sphere_query(spheres, sph_flat_idx, eta)
    if query.radius < 0.0:
        return
    radius_adjusted = query.radius_adjusted

    inv_t = _test_load_obstacle_transform(obs_set, env_idx, obs_local_idx)

    cost_sum = wp.float32(0.0)
    grad_sum_local = wp.vec3(0.0, 0.0, 0.0)

    local_current = wp.transform_point(inv_t, query.center)

    sdf_result = _test_compute_local_sdf_with_grad(obs_set, env_idx, obs_local_idx, local_current)
    penetration = -sdf_result[0] + radius_adjusted

    if penetration > 0.0:
        activation_result = apply_collision_activation(penetration, eta)
        cost_sum += activation_result[0]
        grad_sum_local += activation_result[1] * wp.vec3(sdf_result[1], sdf_result[2], sdf_result[3])

    if h_idx > 0 and _TEST_SWEEP_STEPS > 0:
        prev_sphere = spheres[sph_flat_idx - num_spheres]
        local_prev = wp.transform_point(
            inv_t, wp.vec3(prev_sphere[0], prev_sphere[1], prev_sphere[2])
        )
        half_dist = wp.length(local_prev - local_current) * 0.5
        inv_half_dist = 1.0 / wp.max(half_dist, 0.001)
        jump = wp.float32(0.0)
        for _ in range(_TEST_SWEEP_STEPS):
            if jump >= half_dist:
                break
            t = 1.0 - 0.5 * jump * inv_half_dist
            local_pt = t * local_current + (1.0 - t) * local_prev

            sdf = _test_compute_local_sdf(obs_set, env_idx, obs_local_idx, local_pt)
            penetration = -sdf + radius_adjusted

            if penetration > 0.0:
                sdf_result = _test_compute_local_sdf_with_grad(obs_set, env_idx, obs_local_idx, local_pt)
                activation_result = apply_collision_activation(penetration, eta)
                cost_sum += activation_result[0]
                grad_sum_local += activation_result[1] * wp.vec3(sdf_result[1], sdf_result[2], sdf_result[3])
                jump += penetration
            else:
                if -penetration >= 1000.0:
                    jump += radius_adjusted
                else:
                    jump += wp.max(-penetration, radius_adjusted)

    if h_idx < horizon - 1 and _TEST_SWEEP_STEPS > 0:
        next_sphere = spheres[sph_flat_idx + num_spheres]
        local_next = wp.transform_point(
            inv_t, wp.vec3(next_sphere[0], next_sphere[1], next_sphere[2])
        )
        half_dist = wp.length(local_next - local_current) * 0.5
        inv_half_dist = 1.0 / wp.max(half_dist, 0.001)
        jump = wp.float32(0.0)
        for _ in range(_TEST_SWEEP_STEPS):
            if jump >= half_dist:
                break
            t = 1.0 - 0.5 * jump * inv_half_dist
            local_pt = t * local_current + (1.0 - t) * local_next

            sdf = _test_compute_local_sdf(obs_set, env_idx, obs_local_idx, local_pt)
            penetration = -sdf + radius_adjusted

            if penetration > 0.0:
                sdf_result = _test_compute_local_sdf_with_grad(obs_set, env_idx, obs_local_idx, local_pt)
                activation_result = apply_collision_activation(penetration, eta)
                cost_sum += activation_result[0]
                grad_sum_local += activation_result[1] * wp.vec3(sdf_result[1], sdf_result[2], sdf_result[3])
                jump += penetration
            else:
                if -penetration >= 1000.0:
                    jump += radius_adjusted
                else:
                    jump += wp.max(-penetration, radius_adjusted)

    if cost_sum > 0.0:
        fwd_t = wp.transform_inverse(inv_t)
        grad_world = wp.transform_vector(fwd_t, grad_sum_local)
        accumulate_collision(sph_flat_idx, w * cost_sum, w * grad_world, distance, gradient)


def _make_empty_esdf(
    dims=(0.5, 0.5, 0.5),
    voxel_size=0.02,
    center=(0.0, 0.0, 0.0),
    fill_value=1.0,
    device="cuda:0",
) -> VoxelGrid:
    """All-free-space ESDF grid."""
    nx = round(dims[0] / voxel_size)
    ny = round(dims[1] / voxel_size)
    nz = round(dims[2] / voxel_size)
    feature = torch.full((nx, ny, nz), fill_value, dtype=torch.float16, device=device)
    return VoxelGrid(
        name="test_esdf",
        pose=[center[0], center[1], center[2], 1.0, 0.0, 0.0, 0.0],
        dims=list(dims),
        voxel_size=voxel_size,
        feature_tensor=feature,
        feature_dtype=torch.float16,
    )


def _make_box_esdf(
    grid_dims=(0.5, 0.5, 0.5),
    voxel_size=0.02,
    grid_center=(0.0, 0.0, 0.0),
    box_center=(0.0, 0.0, 0.0),
    box_half=(0.05, 0.05, 0.05),
    device="cuda:0",
) -> VoxelGrid:
    """ESDF grid with a box obstacle (negative inside, positive outside)."""
    nx = round(grid_dims[0] / voxel_size)
    ny = round(grid_dims[1] / voxel_size)
    nz = round(grid_dims[2] / voxel_size)

    ix = torch.arange(nx, device=device, dtype=torch.float32)
    iy = torch.arange(ny, device=device, dtype=torch.float32)
    iz = torch.arange(nz, device=device, dtype=torch.float32)
    gx, gy, gz = torch.meshgrid(ix, iy, iz, indexing="ij")

    wx = grid_center[0] + (gx - (nx - 1) / 2.0) * voxel_size
    wy = grid_center[1] + (gy - (ny - 1) / 2.0) * voxel_size
    wz = grid_center[2] + (gz - (nz - 1) / 2.0) * voxel_size

    dx = (wx - box_center[0]).abs() - box_half[0]
    dy = (wy - box_center[1]).abs() - box_half[1]
    dz = (wz - box_center[2]).abs() - box_half[2]

    outside = torch.sqrt(dx.clamp(min=0) ** 2 + dy.clamp(min=0) ** 2 + dz.clamp(min=0) ** 2)
    inside = torch.stack([dx, dy, dz], dim=-1).max(dim=-1).values.clamp(max=0)
    sdf = outside + inside

    return VoxelGrid(
        name="test_esdf_box",
        pose=[grid_center[0], grid_center[1], grid_center[2], 1.0, 0.0, 0.0, 0.0],
        dims=list(grid_dims),
        voxel_size=voxel_size,
        feature_tensor=sdf.to(torch.float16),
        feature_dtype=torch.float16,
    )


def _launch_collision(
    voxel_data: VoxelData,
    spheres: torch.Tensor,
    activation_distance: float = 0.02,
    weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch sphere_obstacle_collision_kernel against VoxelDataWarp.

    Args:
        voxel_data: VoxelData with loaded grid(s).
        spheres: Query spheres (batch, horizon, num_spheres, 4).
        activation_distance: Collision activation distance.
        weight: Collision cost weight.

    Returns:
        (distance, gradient) tensors.
    """
    device_cfg = DeviceCfg(device=spheres.device, dtype=torch.float32)

    b, h, n, _ = spheres.shape
    buffer = CollisionBuffer.from_shape(spheres.shape, device_cfg)
    buffer.zero_()

    device, stream = get_warp_device_stream(spheres)
    data_wp = voxel_data.to_warp()
    spheres_wp = wp.from_torch(spheres.detach().view(-1, 4), dtype=wp.vec4)
    env_idx = torch.zeros(b, dtype=torch.int32, device=spheres.device)
    env_idx_wp = wp.from_torch(env_idx, dtype=wp.int32)
    weight_t = torch.tensor([weight], dtype=torch.float32, device=spheres.device)
    eta_t = torch.tensor([activation_distance], dtype=torch.float32, device=spheres.device)
    weight_wp = wp.from_torch(weight_t)
    eta_wp = wp.from_torch(eta_t)
    out_cost_wp = wp.from_torch(buffer.distance.detach().view(-1))
    out_grad_wp = wp.from_torch(buffer.gradient.detach().view(-1), dtype=wp.float32)

    max_n = voxel_data.max_n
    wp.launch(
        kernel=sphere_obstacle_collision_kernel,
        dim=b * h * n * max_n,
        inputs=[
            data_wp, spheres_wp, weight_wp, eta_wp, env_idx_wp,
            out_cost_wp, out_grad_wp,
            b, h, n, max_n, wp.uint8(0),
        ],
        stream=stream,
        device=device,
    )
    wp.synchronize_device(device)

    return buffer.distance, buffer.gradient


def _launch_swept_collision(
    voxel_data: VoxelData,
    spheres: torch.Tensor,
    activation_distance: float = 0.02,
    weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch swept_sphere_obstacle_collision_kernel against VoxelDataWarp.

    The swept kernel interpolates between adjacent trajectory timesteps
    (horizon dimension) to detect collisions along the swept path.

    Args:
        voxel_data: VoxelData with loaded grid(s).
        spheres: Query spheres (batch, horizon, num_spheres, 4).
        activation_distance: Collision activation distance.
        weight: Collision cost weight.

    Returns:
        (distance, gradient) tensors.
    """
    device_cfg = DeviceCfg(device=spheres.device, dtype=torch.float32)

    b, h, n, _ = spheres.shape
    buffer = CollisionBuffer.from_shape(spheres.shape, device_cfg)
    buffer.zero_()

    device, stream = get_warp_device_stream(spheres)
    data_wp = voxel_data.to_warp()
    spheres_wp = wp.from_torch(spheres.detach().view(-1, 4), dtype=wp.vec4)
    env_idx = torch.zeros(b, dtype=torch.int32, device=spheres.device)
    env_idx_wp = wp.from_torch(env_idx, dtype=wp.int32)
    weight_t = torch.tensor([weight], dtype=torch.float32, device=spheres.device)
    eta_t = torch.tensor([activation_distance], dtype=torch.float32, device=spheres.device)
    weight_wp = wp.from_torch(weight_t)
    eta_wp = wp.from_torch(eta_t)
    out_cost_wp = wp.from_torch(buffer.distance.detach().view(-1))
    out_grad_wp = wp.from_torch(buffer.gradient.detach().view(-1), dtype=wp.float32)

    max_n = voxel_data.max_n
    wp.launch(
        kernel=swept_sphere_obstacle_collision_kernel,
        dim=b * h * n * max_n,
        inputs=[
            data_wp, spheres_wp, weight_wp, eta_wp, env_idx_wp,
            out_cost_wp, out_grad_wp,
            b, h, n, max_n, wp.uint8(0),
        ],
        stream=stream,
        device=device,
        block_dim=128,
    )
    wp.synchronize_device(device)

    return buffer.distance, buffer.gradient


# ── Fixtures ──

@pytest.fixture(scope="module")
def device_cfg():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)


# ── VoxelData creation tests ──

class TestVoxelDataCreation:
    """Test VoxelData creation from synthetic VoxelGrids."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_from_scene_cfg(self, device_cfg):
        """Test VoxelData.from_scene_cfg with a single VoxelGrid."""
        grid = _make_empty_esdf()
        scene = SceneCfg(voxel=[grid])
        voxel_data = VoxelData.from_scene_cfg(scene, device_cfg)

        assert voxel_data is not None
        assert voxel_data.max_n == 1
        assert voxel_data.num_envs == 1
        assert int(voxel_data.count[0].item()) == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_from_voxel_grids(self, device_cfg):
        """Test VoxelData.create_from_voxel_grids directly."""
        grid = _make_empty_esdf()
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        assert voxel_data is not None
        assert voxel_data.features.shape[0] == 1  # num_envs
        assert voxel_data.features.shape[1] == 1  # max_n
        assert voxel_data.enable[0, 0].item() == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_feature_tensor_is_referenced(self, device_cfg):
        """Test that create_from_voxel_grids references (not copies) feature tensor."""
        grid = _make_empty_esdf()
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)
        n_voxels = grid.feature_tensor.numel()
        assert voxel_data.features.shape[2] == n_voxels

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_params_contain_grid_dimensions(self, device_cfg):
        """Test that params store voxel counts and voxel_size."""
        grid = _make_empty_esdf(dims=(0.5, 0.5, 0.5), voxel_size=0.02)
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)
        params = voxel_data.params[0, 0].cpu()
        assert params[0].item() == pytest.approx(25.0, abs=1.0)  # nx = 0.5/0.02
        assert params[1].item() == pytest.approx(25.0, abs=1.0)  # ny
        assert params[2].item() == pytest.approx(25.0, abs=1.0)  # nz
        assert params[3].item() == pytest.approx(0.02, abs=1e-5)  # voxel_size


# ── VoxelDataWarp conversion tests ──

class TestVoxelDataWarp:
    """Test VoxelData → VoxelDataWarp conversion."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_to_warp_returns_struct(self, device_cfg):
        """Test to_warp returns a VoxelDataWarp struct."""
        grid = _make_empty_esdf()
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)
        warp_data = voxel_data.to_warp()
        assert warp_data is not None
        assert hasattr(warp_data, "features")
        assert hasattr(warp_data, "params")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_warp_fields_populated(self, device_cfg):
        """Test VoxelDataWarp has populated fields."""
        grid = _make_empty_esdf()
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)
        warp_data = voxel_data.to_warp()
        assert warp_data.max_n == 1
        assert warp_data.num_envs == 1
        assert warp_data.n_voxels_per_layer == grid.feature_tensor.numel()


# ── Kernel launch tests (collision queries) ──

class TestVoxelCollisionKernelEmpty:
    """Test sphere_obstacle_collision_kernel against an empty (all-free) voxel grid."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_sphere_in_free_space_zero_cost(self, device_cfg):
        """A sphere far inside free space should have zero collision cost."""
        grid = _make_empty_esdf(dims=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0))
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        # Sphere at grid center, radius 0.01; all ESDF values are +1.0
        spheres = torch.tensor(
            [[[[0.0, 0.0, 0.0, 0.01]]]], dtype=torch.float32, device="cuda:0",
        )
        dist, grad = _launch_collision(voxel_data, spheres)

        assert dist.shape == (1, 1, 1)
        assert dist.item() == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_multiple_spheres_in_free_space(self, device_cfg):
        """Multiple spheres in free space should all have zero cost."""
        grid = _make_empty_esdf(dims=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0))
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        spheres = torch.tensor(
            [[
                [[0.1, 0.0, 0.0, 0.01]],
                [[-0.1, 0.0, 0.0, 0.01]],
                [[0.0, 0.1, 0.0, 0.01]],
            ]],
            dtype=torch.float32, device="cuda:0",
        )  # (1, 3, 1, 4)
        dist, grad = _launch_collision(voxel_data, spheres)

        assert dist.shape == (1, 3, 1)
        assert (dist == 0.0).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_sphere_outside_grid_zero_cost(self, device_cfg):
        """A sphere outside the grid bounds should have zero cost (default_val is large positive)."""
        grid = _make_empty_esdf(dims=(0.2, 0.2, 0.2), center=(0.0, 0.0, 0.0))
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        spheres = torch.tensor(
            [[[[5.0, 5.0, 5.0, 0.01]]]], dtype=torch.float32, device="cuda:0",
        )
        dist, grad = _launch_collision(voxel_data, spheres)
        assert dist.item() == pytest.approx(0.0, abs=1e-5)


class TestVoxelCollisionKernelBox:
    """Test sphere_obstacle_collision_kernel against an ESDF with a box obstacle."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_sphere_inside_box_has_cost(self, device_cfg):
        """A sphere at the center of the box obstacle should have non-zero cost."""
        grid = _make_box_esdf(
            grid_dims=(0.5, 0.5, 0.5), voxel_size=0.01,
            grid_center=(0.0, 0.0, 0.0),
            box_center=(0.0, 0.0, 0.0), box_half=(0.05, 0.05, 0.05),
        )
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        # Sphere at box center, radius 0.01
        spheres = torch.tensor(
            [[[[0.0, 0.0, 0.0, 0.01]]]], dtype=torch.float32, device="cuda:0",
        )
        dist, grad = _launch_collision(voxel_data, spheres, activation_distance=0.02)

        assert dist.item() > 0.0, "Sphere inside box should have positive collision cost"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_sphere_far_from_box_zero_cost(self, device_cfg):
        """A sphere far from the box obstacle should have zero cost."""
        grid = _make_box_esdf(
            grid_dims=(0.5, 0.5, 0.5), voxel_size=0.01,
            grid_center=(0.0, 0.0, 0.0),
            box_center=(0.0, 0.0, 0.0), box_half=(0.05, 0.05, 0.05),
        )
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        # Sphere far from box (0.2m away, activation_distance=0.02)
        spheres = torch.tensor(
            [[[[0.2, 0.2, 0.2, 0.01]]]], dtype=torch.float32, device="cuda:0",
        )
        dist, grad = _launch_collision(voxel_data, spheres, activation_distance=0.02)

        assert dist.item() == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_sphere_near_surface_has_cost(self, device_cfg):
        """A sphere near (within activation distance of) the box surface should have cost."""
        grid = _make_box_esdf(
            grid_dims=(0.5, 0.5, 0.5), voxel_size=0.01,
            grid_center=(0.0, 0.0, 0.0),
            box_center=(0.0, 0.0, 0.0), box_half=(0.05, 0.05, 0.05),
        )
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        # Sphere just touching box surface: box_half=0.05, sphere at 0.05 with r=0.02
        # penetration = -sdf + (r + eta) = -0.0 + (0.02 + 0.02) = 0.04
        spheres = torch.tensor(
            [[[[0.05, 0.0, 0.0, 0.02]]]], dtype=torch.float32, device="cuda:0",
        )
        dist, grad = _launch_collision(voxel_data, spheres, activation_distance=0.02)

        assert dist.item() > 0.0, "Sphere at box surface should have cost"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_gradient_nonzero_at_surface(self, device_cfg):
        """Gradient should be non-zero for a sphere penetrating the box."""
        grid = _make_box_esdf(
            grid_dims=(0.5, 0.5, 0.5), voxel_size=0.01,
            grid_center=(0.0, 0.0, 0.0),
            box_center=(0.0, 0.0, 0.0), box_half=(0.05, 0.05, 0.05),
        )
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        # Sphere inside the box
        spheres = torch.tensor(
            [[[[0.02, 0.0, 0.0, 0.01]]]], dtype=torch.float32, device="cuda:0",
        )
        dist, grad = _launch_collision(voxel_data, spheres, activation_distance=0.02)

        grad_xyz = grad[0, 0, 0, :3]
        assert grad_xyz.abs().sum().item() > 0.0, "Gradient should be non-zero inside the box"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_inside_deeper_has_higher_cost(self, device_cfg):
        """A sphere deeper inside the box should have higher cost than one at the surface."""
        grid = _make_box_esdf(
            grid_dims=(0.5, 0.5, 0.5), voxel_size=0.01,
            grid_center=(0.0, 0.0, 0.0),
            box_center=(0.0, 0.0, 0.0), box_half=(0.05, 0.05, 0.05),
        )
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        # Sphere at center (deep inside)
        spheres_deep = torch.tensor(
            [[[[0.0, 0.0, 0.0, 0.01]]]], dtype=torch.float32, device="cuda:0",
        )
        # Sphere at edge (barely inside)
        spheres_edge = torch.tensor(
            [[[[0.04, 0.0, 0.0, 0.01]]]], dtype=torch.float32, device="cuda:0",
        )

        dist_deep, _ = _launch_collision(voxel_data, spheres_deep, activation_distance=0.02)
        dist_edge, _ = _launch_collision(voxel_data, spheres_edge, activation_distance=0.02)

        assert dist_deep.item() > dist_edge.item(), (
            f"Deeper sphere ({dist_deep.item():.4f}) should have higher cost "
            f"than edge sphere ({dist_edge.item():.4f})"
        )


class TestVoxelCollisionKernelBatch:
    """Test batched sphere queries against VoxelGrid."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_batch_of_spheres(self, device_cfg):
        """Test batch dimension with multiple spheres."""
        grid = _make_box_esdf(
            grid_dims=(0.5, 0.5, 0.5), voxel_size=0.01,
            grid_center=(0.0, 0.0, 0.0),
            box_center=(0.0, 0.0, 0.0), box_half=(0.05, 0.05, 0.05),
        )
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        # batch=2, horizon=1, num_spheres=3
        spheres = torch.tensor([
            [[[0.0, 0.0, 0.0, 0.01], [0.2, 0.0, 0.0, 0.01], [0.04, 0.0, 0.0, 0.01]]],
            [[[0.2, 0.2, 0.2, 0.01], [0.0, 0.0, 0.0, 0.01], [-0.2, 0.0, 0.0, 0.01]]],
        ], dtype=torch.float32, device="cuda:0")  # (2, 1, 3, 4)

        dist, grad = _launch_collision(voxel_data, spheres, activation_distance=0.02)

        assert dist.shape == (2, 1, 3)
        # batch 0: sphere 0 inside, sphere 1 far, sphere 2 near edge
        assert dist[0, 0, 0].item() > 0.0   # inside box
        assert dist[0, 0, 1].item() == pytest.approx(0.0, abs=1e-5)  # far away
        # batch 1: sphere 0 far, sphere 1 inside, sphere 2 far
        assert dist[1, 0, 0].item() == pytest.approx(0.0, abs=1e-5)  # far away
        assert dist[1, 0, 1].item() > 0.0   # inside box

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_horizon_dimension(self, device_cfg):
        """Test horizon dimension (trajectory timesteps)."""
        grid = _make_box_esdf(
            grid_dims=(0.5, 0.5, 0.5), voxel_size=0.01,
            grid_center=(0.0, 0.0, 0.0),
            box_center=(0.0, 0.0, 0.0), box_half=(0.05, 0.05, 0.05),
        )
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        # batch=1, horizon=4, num_spheres=1; moving from inside to outside
        spheres = torch.tensor([
            [
                [[0.0, 0.0, 0.0, 0.01]],   # inside
                [[0.03, 0.0, 0.0, 0.01]],   # inside (near edge)
                [[0.06, 0.0, 0.0, 0.01]],   # just outside
                [[0.2, 0.0, 0.0, 0.01]],    # far outside
            ],
        ], dtype=torch.float32, device="cuda:0")  # (1, 4, 1, 4)

        dist, grad = _launch_collision(voxel_data, spheres, activation_distance=0.02)

        assert dist.shape == (1, 4, 1)
        assert dist[0, 0, 0].item() > 0.0   # inside box
        assert dist[0, 3, 0].item() == pytest.approx(0.0, abs=1e-5)  # far outside


class TestVoxelCollisionUpdateFeatures:
    """Test updating VoxelData features and re-querying."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_update_features_inplace(self, device_cfg):
        """Start with empty grid, update features in-place, verify collision appears."""
        voxel_size = 0.01
        grid_dims = (0.5, 0.5, 0.5)
        grid = _make_empty_esdf(dims=grid_dims, voxel_size=voxel_size, center=(0.0, 0.0, 0.0))
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        spheres = torch.tensor(
            [[[[0.0, 0.0, 0.0, 0.01]]]], dtype=torch.float32, device="cuda:0",
        )
        dist_before, _ = _launch_collision(voxel_data, spheres, activation_distance=0.02)
        assert dist_before.item() == pytest.approx(0.0, abs=1e-5)

        box_grid = _make_box_esdf(
            grid_dims=grid_dims, voxel_size=voxel_size,
            grid_center=(0.0, 0.0, 0.0),
            box_center=(0.0, 0.0, 0.0), box_half=(0.05, 0.05, 0.05),
        )
        # Directly copy new features into the backing tensor (same layout)
        grid.feature_tensor.copy_(box_grid.feature_tensor)

        dist_after, _ = _launch_collision(voxel_data, spheres, activation_distance=0.02)
        assert dist_after.item() > 0.0, "Collision should appear after feature update"


# ── Swept sphere collision tests ──

class TestSweptVoxelCollisionFreeSpace:
    """Test swept_sphere_obstacle_collision_kernel against an empty voxel grid."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_swept_horizon1_no_sweep(self, device_cfg):
        """horizon=1 disables sweep loops; should work like the static kernel."""
        grid = _make_empty_esdf(dims=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0))
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        spheres = torch.tensor(
            [[[[0.0, 0.0, 0.0, 0.01]]]], dtype=torch.float32, device="cuda:0",
        )
        dist, grad = _launch_swept_collision(voxel_data, spheres)
        assert dist.shape == (1, 1, 1)
        assert dist.item() == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_swept_horizon2_minimal_sweep(self, device_cfg):
        """horizon=2 enables one sweep each direction."""
        grid = _make_empty_esdf(dims=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0))
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        spheres = torch.tensor([
            [
                [[-0.1, 0.0, 0.0, 0.01]],
                [[0.1, 0.0, 0.0, 0.01]],
            ],
        ], dtype=torch.float32, device="cuda:0")

        dist, grad = _launch_swept_collision(voxel_data, spheres)
        assert dist.shape == (1, 2, 1)
        assert (dist == 0.0).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_swept_free_space_zero_cost(self, device_cfg):
        """Swept spheres through free space should have zero cost."""
        grid = _make_empty_esdf(dims=(1.0, 1.0, 1.0), center=(0.0, 0.0, 0.0))
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        spheres = torch.tensor([
            [
                [[-0.3, 0.0, 0.0, 0.01]],
                [[-0.1, 0.0, 0.0, 0.01]],
                [[0.1, 0.0, 0.0, 0.01]],
                [[0.3, 0.0, 0.0, 0.01]],
            ],
        ], dtype=torch.float32, device="cuda:0")

        dist, grad = _launch_swept_collision(voxel_data, spheres)
        assert dist.shape == (1, 4, 1)
        assert (dist == 0.0).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_swept_outside_grid_zero_cost(self, device_cfg):
        """Swept spheres outside the grid should have zero cost."""
        grid = _make_empty_esdf(dims=(0.2, 0.2, 0.2), center=(0.0, 0.0, 0.0))
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        spheres = torch.tensor([
            [
                [[5.0, 0.0, 0.0, 0.01]],
                [[5.1, 0.0, 0.0, 0.01]],
            ],
        ], dtype=torch.float32, device="cuda:0")

        dist, grad = _launch_swept_collision(voxel_data, spheres)
        assert (dist == 0.0).all()


class TestSweptVoxelCollisionBox:
    """Test swept_sphere_obstacle_collision_kernel against an ESDF with a box obstacle."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_swept_through_box_has_cost(self, device_cfg):
        """A trajectory that sweeps through a box obstacle should have cost at collision timesteps."""
        grid = _make_box_esdf(
            grid_dims=(0.5, 0.5, 0.5), voxel_size=0.01,
            grid_center=(0.0, 0.0, 0.0),
            box_center=(0.0, 0.0, 0.0), box_half=(0.05, 0.05, 0.05),
        )
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        # Trajectory: far left → inside box → inside box → far right
        spheres = torch.tensor([
            [
                [[-0.2, 0.0, 0.0, 0.01]],  # far from box
                [[0.0, 0.0, 0.0, 0.01]],    # inside box
                [[0.0, 0.0, 0.0, 0.01]],    # inside box
                [[0.2, 0.0, 0.0, 0.01]],    # far from box
            ],
        ], dtype=torch.float32, device="cuda:0")

        dist, grad = _launch_swept_collision(voxel_data, spheres, activation_distance=0.02)

        assert dist.shape == (1, 4, 1)
        # Timesteps 1 and 2 are inside the box → should have cost
        assert dist[0, 1, 0].item() > 0.0, "Sphere inside box should have swept cost"
        assert dist[0, 2, 0].item() > 0.0, "Sphere inside box should have swept cost"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_swept_detects_intermediate_collision(self, device_cfg):
        """Swept collision should detect a box between two free-space timesteps.

        If t=0 is before the box and t=1 is after it, the sweep interpolation
        should still detect the obstacle along the path.
        """
        grid = _make_box_esdf(
            grid_dims=(0.5, 0.5, 0.5), voxel_size=0.01,
            grid_center=(0.0, 0.0, 0.0),
            box_center=(0.0, 0.0, 0.0), box_half=(0.05, 0.05, 0.05),
        )
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        # Both endpoints are in free space, but the line between them
        # passes through the box along the x-axis
        spheres = torch.tensor([
            [
                [[-0.1, 0.0, 0.0, 0.01]],  # free space (before box)
                [[0.1, 0.0, 0.0, 0.01]],    # free space (after box)
            ],
        ], dtype=torch.float32, device="cuda:0")

        dist_swept, _ = _launch_swept_collision(voxel_data, spheres, activation_distance=0.02)

        # Non-swept collision at same endpoints should be zero
        dist_static, _ = _launch_collision(voxel_data, spheres, activation_distance=0.02)
        assert dist_static[0, 0, 0].item() == pytest.approx(0.0, abs=1e-5)
        assert dist_static[0, 1, 0].item() == pytest.approx(0.0, abs=1e-5)

        # Swept collision should detect the box along the interpolated path
        swept_total = dist_swept.sum().item()
        assert swept_total > 0.0, (
            "Swept collision should detect box between free-space endpoints"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_swept_stationary_matches_static(self, device_cfg):
        """A stationary trajectory (all timesteps identical) should give cost
        consistent with static collision: both non-zero when inside the box.
        """
        grid = _make_box_esdf(
            grid_dims=(0.5, 0.5, 0.5), voxel_size=0.01,
            grid_center=(0.0, 0.0, 0.0),
            box_center=(0.0, 0.0, 0.0), box_half=(0.05, 0.05, 0.05),
        )
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        # All timesteps at box center
        spheres = torch.tensor([
            [
                [[0.0, 0.0, 0.0, 0.01]],
                [[0.0, 0.0, 0.0, 0.01]],
                [[0.0, 0.0, 0.0, 0.01]],
            ],
        ], dtype=torch.float32, device="cuda:0")

        dist_swept, _ = _launch_swept_collision(voxel_data, spheres, activation_distance=0.02)
        dist_static, _ = _launch_collision(voxel_data, spheres, activation_distance=0.02)

        # Both should be non-zero at every timestep
        for t in range(3):
            assert dist_swept[0, t, 0].item() > 0.0, f"Swept cost should be >0 at t={t}"
            assert dist_static[0, t, 0].item() > 0.0, f"Static cost should be >0 at t={t}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_swept_gradient_nonzero_on_collision(self, device_cfg):
        """Gradient should be non-zero when swept path intersects box."""
        grid = _make_box_esdf(
            grid_dims=(0.5, 0.5, 0.5), voxel_size=0.01,
            grid_center=(0.0, 0.0, 0.0),
            box_center=(0.0, 0.0, 0.0), box_half=(0.05, 0.05, 0.05),
        )
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        # Sphere at t=1 is off-center inside the box (gradient is zero at exact center)
        spheres = torch.tensor([
            [
                [[-0.1, 0.0, 0.0, 0.01]],
                [[0.02, 0.0, 0.0, 0.01]],   # inside box, off-center
                [[0.1, 0.0, 0.0, 0.01]],
            ],
        ], dtype=torch.float32, device="cuda:0")

        dist, grad = _launch_swept_collision(voxel_data, spheres, activation_distance=0.02)

        grad_xyz = grad[0, 1, 0, :3]
        assert grad_xyz.abs().sum().item() > 0.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_swept_far_from_box_zero_cost(self, device_cfg):
        """A trajectory entirely far from the box should have zero swept cost."""
        grid = _make_box_esdf(
            grid_dims=(0.5, 0.5, 0.5), voxel_size=0.01,
            grid_center=(0.0, 0.0, 0.0),
            box_center=(0.0, 0.0, 0.0), box_half=(0.05, 0.05, 0.05),
        )
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        spheres = torch.tensor([
            [
                [[0.15, 0.15, 0.15, 0.01]],
                [[0.16, 0.15, 0.15, 0.01]],
                [[0.17, 0.15, 0.15, 0.01]],
            ],
        ], dtype=torch.float32, device="cuda:0")

        dist, grad = _launch_swept_collision(voxel_data, spheres, activation_distance=0.02)
        assert (dist == 0.0).all()


class TestSweptVoxelCollisionBatch:
    """Test batched swept sphere collision queries."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_batch_swept_collision(self, device_cfg):
        """Test swept collision with batch dimension: one trajectory collides, one doesn't."""
        grid = _make_box_esdf(
            grid_dims=(0.5, 0.5, 0.5), voxel_size=0.01,
            grid_center=(0.0, 0.0, 0.0),
            box_center=(0.0, 0.0, 0.0), box_half=(0.05, 0.05, 0.05),
        )
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        # batch=2, horizon=3, num_spheres=1
        # Batch 0: sweeps through the box
        # Batch 1: stays far from the box
        spheres = torch.tensor([
            [
                [[-0.1, 0.0, 0.0, 0.01]],
                [[0.0, 0.0, 0.0, 0.01]],    # inside box
                [[0.1, 0.0, 0.0, 0.01]],
            ],
            [
                [[0.2, 0.2, 0.0, 0.01]],
                [[0.2, 0.2, 0.01, 0.01]],
                [[0.2, 0.2, 0.02, 0.01]],
            ],
        ], dtype=torch.float32, device="cuda:0")  # (2, 3, 1, 4)

        dist, grad = _launch_swept_collision(voxel_data, spheres, activation_distance=0.02)

        assert dist.shape == (2, 3, 1)
        # Batch 0 should have some cost (trajectory passes through box)
        assert dist[0].sum().item() > 0.0
        # Batch 1 should have zero cost (entirely free space)
        assert dist[1].sum().item() == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_multi_sphere_swept(self, device_cfg):
        """Test swept collision with multiple spheres per timestep."""
        grid = _make_box_esdf(
            grid_dims=(0.5, 0.5, 0.5), voxel_size=0.01,
            grid_center=(0.0, 0.0, 0.0),
            box_center=(0.0, 0.0, 0.0), box_half=(0.05, 0.05, 0.05),
        )
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        # batch=1, horizon=3, num_spheres=2
        # Sphere 0: sweeps through the box
        # Sphere 1: stays far away
        spheres = torch.tensor([
            [
                [[-0.1, 0.0, 0.0, 0.01], [0.2, 0.2, 0.0, 0.01]],
                [[0.0, 0.0, 0.0, 0.01], [0.2, 0.2, 0.01, 0.01]],
                [[0.1, 0.0, 0.0, 0.01], [0.2, 0.2, 0.02, 0.01]],
            ],
        ], dtype=torch.float32, device="cuda:0")  # (1, 3, 2, 4)

        dist, grad = _launch_swept_collision(voxel_data, spheres, activation_distance=0.02)

        assert dist.shape == (1, 3, 2)
        # Sphere 0 should have cost at the middle timestep
        assert dist[0, 1, 0].item() > 0.0
        # Sphere 1 should have zero cost everywhere
        assert dist[0, :, 1].sum().item() == pytest.approx(0.0, abs=1e-5)


# ── Standalone SDF function tests (isolating compute_local_sdf from sweep kernel) ──


class TestStandaloneComputeLocalSdf:
    """Test compute_local_sdf for VoxelDataWarp in a minimal kernel.

    This isolates whether the crash is in compute_local_sdf itself (Warp
    codegen / float16 access issue) vs an interaction with the sweep kernel.
    """

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_sdf_with_grad_kernel_free_space(self, device_cfg):
        """compute_local_sdf_with_grad should work (baseline)."""
        grid = _make_empty_esdf(fill_value=0.5)
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        pts = torch.tensor(
            [[0.0, 0.0, 0.0], [0.01, 0.01, 0.01], [-0.05, 0.05, 0.0]],
            dtype=torch.float32, device="cuda:0",
        )
        pts_wp = wp.from_torch(pts, dtype=wp.vec3)
        out = torch.zeros(pts.shape[0], dtype=torch.float32, device="cuda:0")
        out_wp = wp.from_torch(out)

        device, stream = get_warp_device_stream(pts)
        data_wp = voxel_data.to_warp()

        wp.launch(
            kernel=_test_sdf_with_grad_kernel,
            dim=pts.shape[0],
            inputs=[data_wp, pts_wp, wp.int32(0), wp.int32(0), out_wp],
            stream=stream,
            device=device,
        )
        wp.synchronize_device(device)

        for i in range(pts.shape[0]):
            assert out[i].item() > 0.0, f"Point {i}: expected positive SDF in free space"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_sdf_only_kernel_free_space(self, device_cfg):
        """compute_local_sdf (returns float32 only): does this crash?"""
        grid = _make_empty_esdf(fill_value=0.5)
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        pts = torch.tensor(
            [[0.0, 0.0, 0.0], [0.01, 0.01, 0.01], [-0.05, 0.05, 0.0]],
            dtype=torch.float32, device="cuda:0",
        )
        pts_wp = wp.from_torch(pts, dtype=wp.vec3)
        out = torch.zeros(pts.shape[0], dtype=torch.float32, device="cuda:0")
        out_wp = wp.from_torch(out)

        device, stream = get_warp_device_stream(pts)
        data_wp = voxel_data.to_warp()

        wp.launch(
            kernel=_test_sdf_only_kernel,
            dim=pts.shape[0],
            inputs=[data_wp, pts_wp, wp.int32(0), wp.int32(0), out_wp],
            stream=stream,
            device=device,
        )
        wp.synchronize_device(device)

        for i in range(pts.shape[0]):
            assert out[i].item() > 0.0, f"Point {i}: expected positive SDF in free space"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_sdf_only_vs_with_grad_agree(self, device_cfg):
        """Both functions should return the same SDF values."""
        grid = _make_box_esdf()
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        pts = torch.tensor(
            [
                [0.0, 0.0, 0.0],      # inside box
                [0.1, 0.0, 0.0],       # outside box
                [0.05, 0.05, 0.0],     # near box surface
                [-0.1, -0.1, 0.0],     # far outside
            ],
            dtype=torch.float32, device="cuda:0",
        )
        pts_wp = wp.from_torch(pts, dtype=wp.vec3)
        device, stream = get_warp_device_stream(pts)
        data_wp = voxel_data.to_warp()

        out_sdf_only = torch.zeros(pts.shape[0], dtype=torch.float32, device="cuda:0")
        out_with_grad = torch.zeros(pts.shape[0], dtype=torch.float32, device="cuda:0")

        wp.launch(
            kernel=_test_sdf_only_kernel,
            dim=pts.shape[0],
            inputs=[data_wp, pts_wp, wp.int32(0), wp.int32(0),
                    wp.from_torch(out_sdf_only)],
            stream=stream, device=device,
        )
        wp.launch(
            kernel=_test_sdf_with_grad_kernel,
            dim=pts.shape[0],
            inputs=[data_wp, pts_wp, wp.int32(0), wp.int32(0),
                    wp.from_torch(out_with_grad)],
            stream=stream, device=device,
        )
        wp.synchronize_device(device)

        for i in range(pts.shape[0]):
            assert out_sdf_only[i].item() == pytest.approx(
                out_with_grad[i].item(), abs=1e-4
            ), f"Point {i}: sdf_only={out_sdf_only[i].item()} vs with_grad={out_with_grad[i].item()}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_sdf_only_kernel_many_points(self, device_cfg):
        """Stress test with many query points to catch intermittent crashes."""
        grid = _make_box_esdf()
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        pts = torch.randn(1024, 3, dtype=torch.float32, device="cuda:0") * 0.2
        pts_wp = wp.from_torch(pts, dtype=wp.vec3)
        out = torch.zeros(1024, dtype=torch.float32, device="cuda:0")
        out_wp = wp.from_torch(out)

        device, stream = get_warp_device_stream(pts)
        data_wp = voxel_data.to_warp()

        wp.launch(
            kernel=_test_sdf_only_kernel,
            dim=1024,
            inputs=[data_wp, pts_wp, wp.int32(0), wp.int32(0), out_wp],
            stream=stream, device=device,
        )
        wp.synchronize_device(device)

        assert not torch.isnan(out).any(), "NaN in SDF output"
        assert not torch.isinf(out).any(), "Inf in SDF output"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_both_sdf_functions_in_one_kernel(self, device_cfg):
        """Call BOTH compute_local_sdf and compute_local_sdf_with_grad in one kernel.

        This reproduces the original sweep kernel pattern where both functions
        are called for the same obs_set.
        """
        grid = _make_box_esdf()
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        pts = torch.tensor(
            [
                [0.0, 0.0, 0.0],      # inside box (sdf < 0)
                [0.1, 0.0, 0.0],       # outside box
                [0.05, 0.05, 0.0],     # near surface
                [-0.1, -0.1, 0.0],     # far outside
            ],
            dtype=torch.float32, device="cuda:0",
        )
        pts_wp = wp.from_torch(pts, dtype=wp.vec3)
        out_sdf_only = torch.zeros(pts.shape[0], dtype=torch.float32, device="cuda:0")
        out_sdf_grad = torch.zeros(pts.shape[0], dtype=torch.float32, device="cuda:0")

        device, stream = get_warp_device_stream(pts)
        data_wp = voxel_data.to_warp()

        wp.launch(
            kernel=_test_both_sdf_kernel,
            dim=pts.shape[0],
            inputs=[
                data_wp, pts_wp, wp.int32(0), wp.int32(0),
                wp.from_torch(out_sdf_only), wp.from_torch(out_sdf_grad),
            ],
            stream=stream, device=device,
        )
        wp.synchronize_device(device)

        assert not torch.isnan(out_sdf_only).any(), "NaN in sdf_only output"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_both_sdf_functions_many_points(self, device_cfg):
        """Stress test: both functions in one kernel with 1024 random points."""
        grid = _make_box_esdf()
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        pts = torch.randn(1024, 3, dtype=torch.float32, device="cuda:0") * 0.2
        pts_wp = wp.from_torch(pts, dtype=wp.vec3)
        out_sdf_only = torch.zeros(1024, dtype=torch.float32, device="cuda:0")
        out_sdf_grad = torch.zeros(1024, dtype=torch.float32, device="cuda:0")

        device, stream = get_warp_device_stream(pts)
        data_wp = voxel_data.to_warp()

        wp.launch(
            kernel=_test_both_sdf_kernel,
            dim=1024,
            inputs=[
                data_wp, pts_wp, wp.int32(0), wp.int32(0),
                wp.from_torch(out_sdf_only), wp.from_torch(out_sdf_grad),
            ],
            stream=stream, device=device,
        )
        wp.synchronize_device(device)

        assert not torch.isnan(out_sdf_only).any(), "NaN in sdf_only output"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_sweep_pattern_kernel(self, device_cfg):
        """Single-loop pattern: compute_local_sdf in a for loop."""
        grid = _make_box_esdf()
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        pts = torch.randn(1024, 3, dtype=torch.float32, device="cuda:0") * 0.2
        pts_wp = wp.from_torch(pts, dtype=wp.vec3)
        out = torch.zeros(1024, dtype=torch.float32, device="cuda:0")
        out_wp = wp.from_torch(out)

        device, stream = get_warp_device_stream(pts)
        data_wp = voxel_data.to_warp()

        wp.launch(
            kernel=_test_sweep_pattern_kernel,
            dim=1024,
            inputs=[data_wp, pts_wp, wp.int32(0), wp.int32(0), out_wp],
            stream=stream, device=device,
        )
        wp.synchronize_device(device)

        assert not torch.isnan(out).any(), "NaN in single-loop sweep pattern"

    @pytest.mark.skip(
        reason="Warp overload resolution bug: using both compute_local_sdf and "
        "compute_local_sdf_with_grad for VoxelDataWarp in a complex kernel "
        "produces invalid codegen (9 ld.local with 0 st.local in PTX). "
        "Crashes CUDA context and poisons subsequent tests. "
        "Workaround: use only compute_local_sdf_with_grad.",
    )
    def test_full_sweep_replica_kernel(self, device_cfg):
        """Reproduces Warp overload resolution bug with compute_local_sdf in sweep loops.

        Uses both compute_local_sdf and compute_local_sdf_with_grad in a kernel
        matching the full sweep kernel complexity. This triggers a Warp codegen
        bug where ld.local instructions are emitted without corresponding
        st.local, causing out-of-bounds local memory reads.
        """
        grid = _make_empty_esdf(fill_value=0.5)
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        spheres = torch.tensor([
            [
                [[-0.1, 0.0, 0.0, 0.01]],
                [[0.1, 0.0, 0.0, 0.01]],
            ],
        ], dtype=torch.float32, device="cuda:0")  # (1, 2, 1, 4)

        b, h, n, _ = spheres.shape
        buffer = CollisionBuffer.from_shape(spheres.shape, device_cfg)
        buffer.zero_()

        device, stream = get_warp_device_stream(spheres)
        data_wp = voxel_data.to_warp()
        spheres_wp = wp.from_torch(spheres.detach().view(-1, 4), dtype=wp.vec4)
        env_idx = torch.zeros(b, dtype=torch.int32, device=spheres.device)
        env_idx_wp = wp.from_torch(env_idx, dtype=wp.int32)
        weight_t = torch.tensor([1.0], dtype=torch.float32, device=spheres.device)
        eta_t = torch.tensor([0.02], dtype=torch.float32, device=spheres.device)

        max_n = voxel_data.max_n
        wp.launch(
            kernel=_test_full_sweep_kernel,
            dim=b * h * n * max_n,
            inputs=[
                data_wp, spheres_wp,
                wp.from_torch(weight_t), wp.from_torch(eta_t), env_idx_wp,
                wp.from_torch(buffer.distance.detach().view(-1)),
                wp.from_torch(buffer.gradient.detach().view(-1), dtype=wp.float32),
                b, h, n, max_n, wp.uint8(0),
            ],
            stream=stream, device=device,
            block_dim=128,
        )
        wp.synchronize_device(device)

        assert (buffer.distance == 0.0).all(), "Expected zero cost in free space"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_dual_sweep_kernel(self, device_cfg):
        """TWO sweep loops (prev + next): simplified sweep pattern."""
        grid = _make_box_esdf()
        voxel_data = VoxelData.create_from_voxel_grids([grid], device_cfg)

        n = 1024
        pts = torch.randn(n, 3, dtype=torch.float32, device="cuda:0") * 0.1
        prev_pts = pts + torch.randn(n, 3, dtype=torch.float32, device="cuda:0") * 0.05
        next_pts = pts + torch.randn(n, 3, dtype=torch.float32, device="cuda:0") * 0.05

        out_cost = torch.zeros(n, dtype=torch.float32, device="cuda:0")
        out_grad = torch.zeros(n, dtype=torch.float32, device="cuda:0")

        device, stream = get_warp_device_stream(pts)
        data_wp = voxel_data.to_warp()

        wp.launch(
            kernel=_test_dual_sweep_kernel,
            dim=n,
            inputs=[
                data_wp,
                wp.from_torch(pts, dtype=wp.vec3),
                wp.from_torch(prev_pts, dtype=wp.vec3),
                wp.from_torch(next_pts, dtype=wp.vec3),
                wp.int32(0), wp.int32(0),
                wp.from_torch(out_cost),
                wp.from_torch(out_grad),
            ],
            stream=stream, device=device,
        )
        wp.synchronize_device(device)

        assert not torch.isnan(out_cost).any(), "NaN in dual-sweep cost"
        assert not torch.isinf(out_cost).any(), "Inf in dual-sweep cost"
