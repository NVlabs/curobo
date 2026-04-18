# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Block-sparse TSDF raycasting kernels for rendering.

This module provides GPU kernels for raycasting through block-sparse TSDF
to render depth, normal, and color images. Uses fixed-step raymarching
with linear interpolation for zero-crossing detection (KinectFusion style).

Features:
- Depth rendering with sub-voxel precision
- Surface normal computation from SDF gradient
- RGB color rendering from TSDF color storage
- Efficient hash-based block lookup

Reference:
- "KinectFusion: Real-time Dense Surface Mapping and Tracking" (Newcombe et al.)
"""

import warp as wp

from curobo._src.perception.mapper.kernel.warp_types import BlockSparseTSDFWarp
from curobo._src.perception.mapper.kernel.wp_coord import (
    voxel_to_world_corner,
    world_to_block_coords,
)
from curobo._src.perception.mapper.kernel.wp_hash import hash_lookup
from curobo._src.perception.mapper.kernel.wp_raycast_common import (
    compute_gradient,
    sample_rgb,
    sample_tsdf_trilinear,
)

# =============================================================================
# Helper Functions
# =============================================================================

MIN_STEP_SCALE = wp.constant(0.5)
BLOCK_SKIP_EPSILON = wp.constant(0.001)  # Small offset to ensure we enter the next block
HIT_REFINE_ITERATIONS = wp.constant(10)  # Bisection iterations for hit refinement


@wp.func
def refine_hit_bisection(
    tsdf: BlockSparseTSDFWarp,
    cam_pos: wp.vec3,
    ray_world: wp.vec3,
    t_lo: wp.float32,
    t_hi: wp.float32,
    sdf_lo: wp.float32,
    sdf_hi: wp.float32,
    minimum_tsdf_weight: wp.float32,
) -> wp.float32:
    """Refine hit position using bisection for smooth depth values.

    Given a bracket [t_lo, t_hi] where SDF changes sign (sdf_lo > 0, sdf_hi < 0),
    use bisection to find the accurate zero-crossing.

    Args:
        tsdf: BlockSparseTSDFWarp struct.
        cam_pos: Camera position.
        ray_world: Ray direction in world frame.
        t_lo, t_hi: Bracket containing the zero-crossing.
        sdf_lo, sdf_hi: SDF values at bracket endpoints.
        minimum_tsdf_weight: Minimum weight for valid voxel.

    Returns:
        Refined t value at the zero-crossing.
    """
    # Start with linear interpolation estimate
    denom = sdf_lo - sdf_hi
    if wp.abs(denom) < 1e-6:
        return (t_lo + t_hi) * 0.5

    t_mid = t_lo + sdf_lo * (t_hi - t_lo) / denom

    # Bisection refinement for accuracy
    for _ in range(HIT_REFINE_ITERATIONS):
        pos_mid = cam_pos + ray_world * t_mid
        result = sample_tsdf_trilinear(tsdf, pos_mid, minimum_tsdf_weight)
        sdf_mid = result[0]

        if result[1] < 0.5:
            # Invalid sample, fall back to current estimate
            break

        if wp.abs(sdf_mid) < 1e-6:
            # Close enough to surface
            break

        if sdf_mid > 0.0:
            # Zero-crossing is between t_mid and t_hi
            t_lo = t_mid
            sdf_lo = sdf_mid
        else:
            # Zero-crossing is between t_lo and t_mid
            t_hi = t_mid
            sdf_hi = sdf_mid

        # Update estimate with linear interpolation in new bracket
        denom = sdf_lo - sdf_hi
        if wp.abs(denom) < 1e-6:
            t_mid = (t_lo + t_hi) * 0.5
        else:
            t_mid = t_lo + sdf_lo * (t_hi - t_lo) / denom

    return t_mid


@wp.func
def quat_from_wxyz_array(q: wp.array(dtype=wp.float32)) -> wp.quat:
    """Convert wxyz quaternion array to warp quaternion (xyzw internal)."""
    return wp.quat(q[1], q[2], q[3], q[0])


@wp.func
def vec3_from_array(v: wp.array(dtype=wp.float32)) -> wp.vec3:
    """Convert float array to vec3."""
    return wp.vec3(v[0], v[1], v[2])


# =============================================================================
# Block-Level Acceleration Functions
# =============================================================================
# world_to_block_coords is imported from wp_coord


@wp.func
def block_to_world_bounds(
    bx: int,
    by: int,
    bz: int,
    origin: wp.vec3,
    voxel_size: float,
    block_size: int,
    grid_W: int,
    grid_H: int,
    grid_D: int,
) -> wp.mat22f:
    """Get world-space AABB bounds for a block.

    Args:
        bx, by, bz: Block coordinates.
        origin: Grid origin.
        voxel_size: Voxel size in meters.
        block_size: Voxels per block edge.
        grid_W, grid_H, grid_D: Grid dimensions for center-origin convention.

    Returns:
        mat22f where row 0 = (min_x, min_y), row 1 = (max_x, max_y).
        Z bounds returned via separate calculation.
    """
    block_world_size = wp.float32(block_size) * voxel_size

    # Block min corner using common function
    voxel_min = wp.vec3i(bx * block_size, by * block_size, bz * block_size)
    block_min = voxel_to_world_corner(voxel_min, origin, voxel_size, grid_W, grid_H, grid_D)

    min_x = block_min[0]
    min_y = block_min[1]
    max_x = min_x + block_world_size
    max_y = min_y + block_world_size

    # Pack into mat22f (we'll handle z separately)
    # Using mat22f to return 4 values, actual bounds need 6
    return wp.mat22f(min_x, min_y, max_x, max_y)


@wp.func
def ray_block_exit_t(
    ray_origin: wp.vec3,
    ray_dir: wp.vec3,
    bx: int,
    by: int,
    bz: int,
    origin: wp.vec3,
    voxel_size: float,
    block_size: int,
    grid_W: int,
    grid_H: int,
    grid_D: int,
) -> float:
    """Compute the t parameter where ray exits a block.

    Uses slab method for ray-AABB intersection.

    Args:
        ray_origin: Ray origin in world frame.
        ray_dir: Normalized ray direction.
        bx, by, bz: Block coordinates.
        origin: Grid origin.
        voxel_size: Voxel size in meters.
        block_size: Voxels per block edge.
        grid_W, grid_H, grid_D: Grid dimensions.

    Returns:
        t parameter where ray exits the block (t_max).
    """
    block_world_size = wp.float32(block_size) * voxel_size

    # Block min corner using common function
    voxel_min = wp.vec3i(bx * block_size, by * block_size, bz * block_size)
    block_min = voxel_to_world_corner(voxel_min, origin, voxel_size, grid_W, grid_H, grid_D)

    block_min_x = block_min[0]
    block_min_y = block_min[1]
    block_min_z = block_min[2]

    block_max_x = block_min_x + block_world_size
    block_max_y = block_min_y + block_world_size
    block_max_z = block_min_z + block_world_size

    # Slab method for ray-AABB intersection
    # Handle division by zero with large values
    inv_dir_x = 1.0 / ray_dir[0] if wp.abs(ray_dir[0]) > 1e-8 else 1e10 * wp.sign(ray_dir[0])
    inv_dir_y = 1.0 / ray_dir[1] if wp.abs(ray_dir[1]) > 1e-8 else 1e10 * wp.sign(ray_dir[1])
    inv_dir_z = 1.0 / ray_dir[2] if wp.abs(ray_dir[2]) > 1e-8 else 1e10 * wp.sign(ray_dir[2])

    # T values for each slab
    t1_x = (block_min_x - ray_origin[0]) * inv_dir_x
    t2_x = (block_max_x - ray_origin[0]) * inv_dir_x
    t1_y = (block_min_y - ray_origin[1]) * inv_dir_y
    t2_y = (block_max_y - ray_origin[1]) * inv_dir_y
    t1_z = (block_min_z - ray_origin[2]) * inv_dir_z
    t2_z = (block_max_z - ray_origin[2]) * inv_dir_z

    # Find exit t (minimum of the max t values for each axis)
    t_max_x = wp.max(t1_x, t2_x)
    t_max_y = wp.max(t1_y, t2_y)
    t_max_z = wp.max(t1_z, t2_z)

    t_exit = wp.min(wp.min(t_max_x, t_max_y), t_max_z)

    return t_exit


# =============================================================================
# Raycast Kernel - Depth + Normals
# =============================================================================


@wp.kernel
def raycast_block_sparse_kernel(
    # Camera parameters
    intrinsics: wp.array2d(dtype=wp.float32),  # (3, 3)
    cam_position: wp.array(dtype=wp.float32),  # (3,)
    cam_quaternion: wp.array(dtype=wp.float32),  # (4,) wxyz
    # Block-sparse TSDF (struct)
    tsdf: BlockSparseTSDFWarp,
    # Ray marching parameters
    depth_minimum_distance: float,
    depth_maximum_distance: float,
    minimum_tsdf_weight: float,
    # Output (per pixel)
    hit_points: wp.array2d(dtype=wp.float32),  # [H*W, 3] world coords
    hit_normals: wp.array2d(dtype=wp.float32),  # [H*W, 3] surface normals
    hit_depths: wp.array(dtype=wp.float32),  # [H*W] depth values
    hit_mask: wp.array(dtype=wp.uint8),  # [H*W] valid hit
    # Image dimensions
    img_H: int,
    img_W: int,
):
    """Raycast block-sparse TSDF to find surface intersections.

    Uses fixed-step raymarching with linear interpolation for zero-crossing.
    Samples from combined (dynamic + static) SDF channels.

    Args:
        intrinsics: [3, 3] camera intrinsic matrix.
        cam_position: [3] camera position in world frame.
        cam_quaternion: [4] camera orientation (wxyz) in world frame.
        tsdf: BlockSparseTSDFWarp struct containing all TSDF data.
        depth_minimum_distance, depth_maximum_distance: Valid depth range in meters.
        minimum_tsdf_weight: Minimum weight for observed dynamic voxel.
        hit_points: [H*W, 3] output hit positions in world frame.
        hit_normals: [H*W, 3] output surface normals.
        hit_depths: [H*W] output depth values.
        hit_mask: [H*W] output validity mask.
        img_H, img_W: Image dimensions.
    """
    tid = wp.tid()

    # Default output: clear all buffers so "invalid" pixels never carry
    # stale data from previous frames (buffers are reused across calls).
    hit_points[tid, 0] = 0.0
    hit_points[tid, 1] = 0.0
    hit_points[tid, 2] = 0.0
    hit_normals[tid, 0] = 0.0
    hit_normals[tid, 1] = 0.0
    hit_normals[tid, 2] = 0.0
    hit_depths[tid] = 0.0
    hit_mask[tid] = wp.uint8(0)

    # Get pixel coordinates
    px = tid % img_W
    py = tid // img_W

    if py >= img_H:
        return

    # Get camera parameters
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    cam_pos = vec3_from_array(cam_position)
    cam_quat = quat_from_wxyz_array(cam_quaternion)

    # Compute ray direction in camera frame
    ray_cam_x = (wp.float32(px) + 0.5 - cx) / fx
    ray_cam_y = (wp.float32(py) + 0.5 - cy) / fy
    ray_cam_z = 1.0

    # Normalize ray direction
    ray_len = wp.sqrt(ray_cam_x * ray_cam_x + ray_cam_y * ray_cam_y + ray_cam_z * ray_cam_z)
    ray_cam = wp.vec3(ray_cam_x / ray_len, ray_cam_y / ray_len, ray_cam_z / ray_len)

    # Transform ray to world frame
    ray_world = wp.quat_rotate(cam_quat, ray_cam)

    # Initialize raymarching
    step_size = tsdf.voxel_size * MIN_STEP_SCALE
    max_steps = wp.int32((depth_maximum_distance - depth_minimum_distance) / step_size) + 1
    max_steps = wp.min(max_steps, 10000)  # Safety limit

    t = float(depth_minimum_distance)
    prev_sdf = float(1e10)
    prev_t = float(depth_minimum_distance)

    hit_found = bool(False)
    hit_t = float(0.0)

    # Fixed-step raymarching
    for step in range(max_steps):
        if t > depth_maximum_distance:
            break

        # Current position along ray
        pos = cam_pos + ray_world * t

        # Sample combined SDF from block-sparse storage
        sdf_result = sample_tsdf_trilinear(tsdf, pos, minimum_tsdf_weight)
        sdf = sdf_result[0]
        valid = sdf_result[1]

        if valid < 0.5:
            # Unobserved region - continue with fixed step
            t += step_size
            continue

        # Check for sign change (zero-crossing)
        if prev_sdf < 1e9 and prev_sdf > 0.0 and sdf < 0.0:
            # Found surface - refine with bisection for smooth depth
            hit_t = refine_hit_bisection(
                tsdf, cam_pos, ray_world,
                prev_t, t, prev_sdf, sdf,
                minimum_tsdf_weight,
            )

            hit_found = True
            break

        prev_sdf = sdf
        prev_t = t
        t += step_size

    if not hit_found:
        return

    # Compute hit position
    hit_pos = cam_pos + ray_world * hit_t

    # Compute surface normal from combined SDF gradient
    normal = compute_gradient(tsdf, hit_pos, minimum_tsdf_weight)

    # Convert hit distance to z-depth
    z_depth = hit_t * ray_cam_z

    # Store results
    hit_points[tid, 0] = hit_pos[0]
    hit_points[tid, 1] = hit_pos[1]
    hit_points[tid, 2] = hit_pos[2]
    hit_normals[tid, 0] = normal[0]
    hit_normals[tid, 1] = normal[1]
    hit_normals[tid, 2] = normal[2]
    hit_depths[tid] = z_depth
    hit_mask[tid] = wp.uint8(1)


# =============================================================================
# Raycast Kernel - Depth + Normals + Color
# =============================================================================


@wp.kernel
def raycast_block_sparse_color_kernel(
    # Camera parameters
    intrinsics: wp.array2d(dtype=wp.float32),  # (3, 3)
    cam_position: wp.array(dtype=wp.float32),  # (3,)
    cam_quaternion: wp.array(dtype=wp.float32),  # (4,) wxyz
    # Block-sparse TSDF (struct)
    tsdf: BlockSparseTSDFWarp,
    # Ray marching parameters
    depth_minimum_distance: float,
    depth_maximum_distance: float,
    minimum_tsdf_weight: float,
    # Output (per pixel)
    hit_points: wp.array2d(dtype=wp.float32),  # [H*W, 3] world coords
    hit_normals: wp.array2d(dtype=wp.float32),  # [H*W, 3] surface normals
    hit_colors: wp.array2d(dtype=wp.uint8),  # [H*W, 3] RGB colors
    hit_depths: wp.array(dtype=wp.float32),  # [H*W] depth values
    hit_mask: wp.array(dtype=wp.uint8),  # [H*W] valid hit
    # Image dimensions
    img_H: int,
    img_W: int,
):
    """Raycast block-sparse TSDF with color output.

    Samples from combined (dynamic + static) SDF channels.

    Args:
        intrinsics: [3, 3] camera intrinsic matrix.
        cam_position: [3] camera position in world frame.
        cam_quaternion: [4] camera orientation (wxyz) in world frame.
        tsdf: BlockSparseTSDFWarp struct containing all TSDF data.
        depth_minimum_distance, depth_maximum_distance: Valid depth range.
        minimum_tsdf_weight: Minimum weight for observed dynamic voxel.
        hit_points: [H*W, 3] output hit positions.
        hit_normals: [H*W, 3] output surface normals.
        hit_colors: [H*W, 3] output RGB colors.
        hit_depths: [H*W] output depth values.
        hit_mask: [H*W] output validity mask.
        img_H, img_W: Image dimensions.
    """
    tid = wp.tid()

    # Default output: clear all buffers so "invalid" pixels never carry
    # stale data from previous frames (buffers are reused across calls).
    hit_points[tid, 0] = 0.0
    hit_points[tid, 1] = 0.0
    hit_points[tid, 2] = 0.0
    hit_normals[tid, 0] = 0.0
    hit_normals[tid, 1] = 0.0
    hit_normals[tid, 2] = 0.0
    hit_colors[tid, 0] = wp.uint8(0)
    hit_colors[tid, 1] = wp.uint8(0)
    hit_colors[tid, 2] = wp.uint8(0)
    hit_depths[tid] = 0.0
    hit_mask[tid] = wp.uint8(0)

    # Get pixel coordinates
    px = tid % img_W
    py = tid // img_W

    if py >= img_H:
        return

    # Get camera parameters
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    cam_pos = vec3_from_array(cam_position)
    cam_quat = quat_from_wxyz_array(cam_quaternion)


    # Compute ray direction in camera frame
    ray_cam_x = (wp.float32(px) + 0.5 - cx) / fx
    ray_cam_y = (wp.float32(py) + 0.5 - cy) / fy
    ray_cam_z = 1.0

    # Normalize ray direction
    ray_len = wp.sqrt(ray_cam_x * ray_cam_x + ray_cam_y * ray_cam_y + ray_cam_z * ray_cam_z)
    ray_cam = wp.vec3(ray_cam_x / ray_len, ray_cam_y / ray_len, ray_cam_z / ray_len)

    # Transform ray to world frame
    ray_world = wp.quat_rotate(cam_quat, ray_cam)

    # Initialize raymarching
    step_size = tsdf.voxel_size * MIN_STEP_SCALE
    max_steps = wp.int32((depth_maximum_distance - depth_minimum_distance) / step_size) + 1
    max_steps = wp.min(max_steps, 200000)

    t = float(depth_minimum_distance)
    prev_sdf = float(1e10)
    prev_t = float(depth_minimum_distance)

    hit_found = bool(False)
    hit_t = float(0.0)

    # Fixed-step raymarching
    for step in range(max_steps):
        if t > depth_maximum_distance:
            break

        pos = cam_pos + ray_world * t

        sdf_result = sample_tsdf_trilinear(tsdf, pos, minimum_tsdf_weight)
        sdf = sdf_result[0]
        valid = sdf_result[1]

        if valid < 0.5:
            t += step_size
            continue

        if prev_sdf < 1e9 and prev_sdf > 0.0 and sdf < 0.0:
            # Found surface - refine with bisection for smooth depth
            hit_t = refine_hit_bisection(
                tsdf, cam_pos, ray_world,
                prev_t, t, prev_sdf, sdf,
                minimum_tsdf_weight,
            )

            hit_found = True
            break

        prev_sdf = sdf
        prev_t = t
        t += step_size

    if not hit_found:
        return

    # Compute hit position
    hit_pos = cam_pos + ray_world * hit_t

    # Compute surface normal
    normal = compute_gradient(tsdf, hit_pos, minimum_tsdf_weight)

    # Sample color at hit point (per-block average)
    color = sample_rgb(tsdf, hit_pos)

    # Convert hit distance to z-depth
    z_depth = hit_t * ray_cam_z

    # Store results
    hit_points[tid, 0] = hit_pos[0]
    hit_points[tid, 1] = hit_pos[1]
    hit_points[tid, 2] = hit_pos[2]
    hit_normals[tid, 0] = normal[0]
    hit_normals[tid, 1] = normal[1]
    hit_normals[tid, 2] = normal[2]
    hit_colors[tid, 0] = wp.uint8(color[0])
    hit_colors[tid, 1] = wp.uint8(color[1])
    hit_colors[tid, 2] = wp.uint8(color[2])
    hit_depths[tid] = z_depth
    hit_mask[tid] = wp.uint8(1)


# =============================================================================
# Block-Accelerated Raycast Kernels
# =============================================================================


@wp.kernel
def raycast_block_sparse_accelerated_kernel(
    # Camera parameters
    intrinsics: wp.array2d(dtype=wp.float32),  # (3, 3)
    cam_position: wp.array(dtype=wp.float32),  # (3,)
    cam_quaternion: wp.array(dtype=wp.float32),  # (4,) wxyz
    # Block-sparse TSDF (struct)
    tsdf: BlockSparseTSDFWarp,
    # Ray marching parameters
    depth_minimum_distance: float,
    depth_maximum_distance: float,
    minimum_tsdf_weight: float,
    # Output (per pixel)
    hit_points: wp.array2d(dtype=wp.float32),  # [H*W, 3] world coords
    hit_normals: wp.array2d(dtype=wp.float32),  # [H*W, 3] surface normals
    hit_depths: wp.array(dtype=wp.float32),  # [H*W] depth values
    hit_mask: wp.array(dtype=wp.uint8),  # [H*W] valid hit
    # Image dimensions
    img_H: int,
    img_W: int,
):
    """Block-accelerated raycast through block-sparse TSDF.

    This kernel skips unallocated blocks entirely, only performing fine-grained
    stepping within allocated blocks. Samples from combined (dynamic + static) channels.

    Args:
        intrinsics: [3, 3] camera intrinsic matrix.
        cam_position: [3] camera position in world frame.
        cam_quaternion: [4] camera orientation (wxyz) in world frame.
        tsdf: BlockSparseTSDFWarp struct containing all TSDF data.
        depth_minimum_distance, depth_maximum_distance: Valid depth range.
        minimum_tsdf_weight: Minimum weight for observed dynamic voxel.
        hit_points, hit_normals, hit_depths, hit_mask: Output buffers.
        img_H, img_W: Image dimensions.
    """
    tid = wp.tid()

    # Default output: clear all buffers so "invalid" pixels never carry
    # stale data from previous frames (buffers are reused across calls).
    hit_points[tid, 0] = 0.0
    hit_points[tid, 1] = 0.0
    hit_points[tid, 2] = 0.0
    hit_normals[tid, 0] = 0.0
    hit_normals[tid, 1] = 0.0
    hit_normals[tid, 2] = 0.0
    hit_depths[tid] = 0.0
    hit_mask[tid] = wp.uint8(0)

    px = tid % img_W
    py = tid // img_W

    if py >= img_H:
        return

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    cam_pos = vec3_from_array(cam_position)
    cam_quat = quat_from_wxyz_array(cam_quaternion)

    ray_cam_x = (wp.float32(px) + 0.5 - cx) / fx
    ray_cam_y = (wp.float32(py) + 0.5 - cy) / fy
    ray_cam_z = 1.0

    ray_len = wp.sqrt(ray_cam_x * ray_cam_x + ray_cam_y * ray_cam_y + ray_cam_z * ray_cam_z)
    ray_cam = wp.vec3(ray_cam_x / ray_len, ray_cam_y / ray_len, ray_cam_z / ray_len)
    ray_world = wp.quat_rotate(cam_quat, ray_cam)

    step_size = tsdf.voxel_size * MIN_STEP_SCALE

    t = float(depth_minimum_distance)
    prev_sdf = float(1e10)
    prev_t = float(depth_minimum_distance)

    hit_found = bool(False)
    hit_t = float(0.0)

    curr_bx = wp.int32(-999999)
    curr_by = wp.int32(-999999)
    curr_bz = wp.int32(-999999)
    curr_block_allocated = bool(False)
    curr_block_exit_t = float(0.0)

    max_iterations = 10000

    for iteration in range(max_iterations):
        if t > depth_maximum_distance:
            #print("t > depth_maximum_distance")
            break

        pos = cam_pos + ray_world * t

        block_coords = world_to_block_coords(
            pos, tsdf.origin, tsdf.voxel_size, tsdf.block_size, tsdf.grid_W, tsdf.grid_H, tsdf.grid_D
        )
        bx = block_coords[0]
        by = block_coords[1]
        bz = block_coords[2]

        if bx != curr_bx or by != curr_by or bz != curr_bz:
            curr_bx = bx
            curr_by = by
            curr_bz = bz

            block_idx = hash_lookup(tsdf.hash_table, bx, by, bz, tsdf.hash_capacity)
            curr_block_allocated = block_idx >= 0

            curr_block_exit_t = ray_block_exit_t(
                cam_pos, ray_world, bx, by, bz,
                tsdf.origin, tsdf.voxel_size, tsdf.block_size,
                tsdf.grid_W, tsdf.grid_H, tsdf.grid_D
            )

        if not curr_block_allocated:
            t = curr_block_exit_t + step_size #* BLOCK_SKIP_EPSILON
            prev_sdf = 1e10
            continue

        sdf_result = sample_tsdf_trilinear(tsdf, pos, minimum_tsdf_weight)
        sdf = sdf_result[0]
        valid = sdf_result[1]

        if valid < 0.5:
            t += step_size
            continue

        if prev_sdf < 1e9 and prev_sdf > 0.0 and sdf < 0.0:
            # Found surface - refine with bisection for smooth depth
            hit_t = refine_hit_bisection(
                tsdf, cam_pos, ray_world,
                prev_t, t, prev_sdf, sdf,
                minimum_tsdf_weight,
            )
            hit_found = True
            break

        prev_sdf = sdf
        prev_t = t
        t += step_size

    if not hit_found:
        return

    hit_pos = cam_pos + ray_world * hit_t
    normal = compute_gradient(tsdf, hit_pos, minimum_tsdf_weight)
    z_depth = hit_t * ray_cam_z

    hit_points[tid, 0] = hit_pos[0]
    hit_points[tid, 1] = hit_pos[1]
    hit_points[tid, 2] = hit_pos[2]
    hit_normals[tid, 0] = normal[0]
    hit_normals[tid, 1] = normal[1]
    hit_normals[tid, 2] = normal[2]
    hit_depths[tid] = z_depth
    hit_mask[tid] = wp.uint8(1)


@wp.kernel
def raycast_block_sparse_accelerated_color_kernel(
    # Camera parameters
    intrinsics: wp.array2d(dtype=wp.float32),  # (3, 3)
    cam_position: wp.array(dtype=wp.float32),  # (3,)
    cam_quaternion: wp.array(dtype=wp.float32),  # (4,) wxyz
    # Block-sparse TSDF (struct)
    tsdf: BlockSparseTSDFWarp,
    # Ray marching parameters
    depth_minimum_distance: float,
    depth_maximum_distance: float,
    minimum_tsdf_weight: float,
    # Output (per pixel)
    hit_points: wp.array2d(dtype=wp.float32),  # [H*W, 3] world coords
    hit_normals: wp.array2d(dtype=wp.float32),  # [H*W, 3] surface normals
    hit_colors: wp.array2d(dtype=wp.uint8),  # [H*W, 3] RGB colors
    hit_depths: wp.array(dtype=wp.float32),  # [H*W] depth values
    hit_mask: wp.array(dtype=wp.uint8),  # [H*W] valid hit
    # Image dimensions
    img_H: int,
    img_W: int,
):
    """Block-accelerated raycast with color output.

    Samples from combined (dynamic + static) SDF channels.

    Args:
        intrinsics: [3, 3] camera intrinsic matrix.
        cam_position: [3] camera position in world frame.
        cam_quaternion: [4] camera orientation (wxyz) in world frame.
        tsdf: BlockSparseTSDFWarp struct containing all TSDF data.
        depth_minimum_distance, depth_maximum_distance: Valid depth range.
        minimum_tsdf_weight: Minimum weight for observed dynamic voxel.
        hit_points, hit_normals, hit_colors, hit_depths, hit_mask: Output buffers.
        img_H, img_W: Image dimensions.
    """
    tid = wp.tid()

    # Default output: clear all buffers so "invalid" pixels never carry
    # stale data from previous frames (buffers are reused across calls).
    hit_points[tid, 0] = 0.0
    hit_points[tid, 1] = 0.0
    hit_points[tid, 2] = 0.0
    hit_normals[tid, 0] = 0.0
    hit_normals[tid, 1] = 0.0
    hit_normals[tid, 2] = 0.0
    hit_colors[tid, 0] = wp.uint8(0)
    hit_colors[tid, 1] = wp.uint8(0)
    hit_colors[tid, 2] = wp.uint8(0)
    hit_depths[tid] = 0.0
    hit_mask[tid] = wp.uint8(0)

    px = tid % img_W
    py = tid // img_W

    if py >= img_H:
        return

    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    cam_pos = vec3_from_array(cam_position)
    cam_quat = quat_from_wxyz_array(cam_quaternion)

    ray_cam_x = (wp.float32(px) + 0.5 - cx) / fx
    ray_cam_y = (wp.float32(py) + 0.5 - cy) / fy
    ray_cam_z = 1.0

    ray_len = wp.sqrt(ray_cam_x * ray_cam_x + ray_cam_y * ray_cam_y + ray_cam_z * ray_cam_z)
    ray_cam = wp.vec3(ray_cam_x / ray_len, ray_cam_y / ray_len, ray_cam_z / ray_len)
    ray_world = wp.quat_rotate(cam_quat, ray_cam)

    step_size = tsdf.voxel_size * MIN_STEP_SCALE

    t = float(depth_minimum_distance)
    prev_sdf = float(1e10)
    prev_t = float(depth_minimum_distance)

    hit_found = bool(False)
    hit_t = float(0.0)

    curr_bx = wp.int32(-999999)
    curr_by = wp.int32(-999999)
    curr_bz = wp.int32(-999999)
    curr_block_allocated = bool(False)
    curr_block_exit_t = float(0.0)

    max_iterations = 10000

    for iteration in range(max_iterations):
        if t > depth_maximum_distance:
            break

        pos = cam_pos + ray_world * t

        block_coords = world_to_block_coords(
            pos, tsdf.origin, tsdf.voxel_size, tsdf.block_size, tsdf.grid_W, tsdf.grid_H, tsdf.grid_D
        )
        bx = block_coords[0]
        by = block_coords[1]
        bz = block_coords[2]

        if bx != curr_bx or by != curr_by or bz != curr_bz:
            curr_bx = bx
            curr_by = by
            curr_bz = bz

            block_idx = hash_lookup(tsdf.hash_table, bx, by, bz, tsdf.hash_capacity)
            curr_block_allocated = block_idx >= 0

            curr_block_exit_t = ray_block_exit_t(
                cam_pos, ray_world, bx, by, bz,
                tsdf.origin, tsdf.voxel_size, tsdf.block_size,
                tsdf.grid_W, tsdf.grid_H, tsdf.grid_D
            )

        if not curr_block_allocated:
            t = curr_block_exit_t + step_size #* BLOCK_SKIP_EPSILON
            prev_sdf = 1e10
            continue

        sdf_result = sample_tsdf_trilinear(tsdf, pos, minimum_tsdf_weight)
        sdf = sdf_result[0]
        valid = sdf_result[1]

        if valid < 0.5:
            t += step_size
            continue

        if prev_sdf < 1e9 and prev_sdf > 0.0 and sdf < 0.0:
            # Found surface - refine with bisection for smooth depth
            hit_t = refine_hit_bisection(
                tsdf, cam_pos, ray_world,
                prev_t, t, prev_sdf, sdf,
                minimum_tsdf_weight,
            )
            hit_found = True
            break

        prev_sdf = sdf
        prev_t = t
        t += step_size

    if not hit_found:
        return

    hit_pos = cam_pos + ray_world * hit_t
    normal = compute_gradient(tsdf, hit_pos, minimum_tsdf_weight)
    color = sample_rgb(tsdf, hit_pos)

    # Convert hit distance to z-depth
    z_depth = hit_t * ray_cam_z

    # Store results
    hit_points[tid, 0] = hit_pos[0]
    hit_points[tid, 1] = hit_pos[1]
    hit_points[tid, 2] = hit_pos[2]
    hit_normals[tid, 0] = normal[0]
    hit_normals[tid, 1] = normal[1]
    hit_normals[tid, 2] = normal[2]
    hit_colors[tid, 0] = wp.uint8(color[0])
    hit_colors[tid, 1] = wp.uint8(color[1])
    hit_colors[tid, 2] = wp.uint8(color[2])
    hit_depths[tid] = z_depth
    hit_mask[tid] = wp.uint8(1)

