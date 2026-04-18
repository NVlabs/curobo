# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Fused GPU depth filtering using Warp.

Single-kernel implementation that combines:
1. Range filtering (min/max depth)
2. Flying pixel detection (depth discontinuities)
3. Bilateral filtering (edge-preserving smoothing)

All kernels operate strictly on batched ``(B, H, W)`` tensors. Single-image
callers of :class:`FilterDepth` must explicitly promote their input with
``depth.unsqueeze(0)`` and index ``filtered[0]`` on return.

This is more efficient than separate kernels due to:
- Single memory read per pixel
- Reduced kernel launch overhead
- Better cache utilization

The kernels are launched directly by :class:`FilterDepth` in
``curobo._src.perception.filter_depth``.
"""

import warp as wp


@wp.func
def compute_flying_pixel_threshold(threshold: float) -> float:
    """Convert threshold (0-1) to relative gradient tolerance.

    Args:
        threshold: Filter aggressiveness 0.0 (permissive) to 1.0 (aggressive).

    Returns:
        Tolerance value for relative gradient comparison.
    """
    # Exponential interpolation:
    # threshold=0.0: 0.08 (8%, permissive)
    # threshold=0.5: 0.02 (2%, balanced)
    # threshold=1.0: 0.005 (0.5%, aggressive)
    max_tol = 0.08
    min_tol = 0.005
    # tolerance = max_tol * (min_tol / max_tol) ^ threshold
    # = max_tol * exp(threshold * log(min_tol / max_tol))
    return max_tol * wp.exp(threshold * wp.log(min_tol / max_tol))


@wp.func
def is_flying_pixel(
    depth_center: float,
    d_left: float,
    d_right: float,
    d_up: float,
    d_down: float,
    tolerance: float,
) -> bool:
    """Check if pixel is a flying pixel based on 4-connected neighbors.

    A flying pixel has a large depth gradient relative to its depth value.

    Args:
        depth_center: Depth at current pixel.
        d_left, d_right, d_up, d_down: Depths at 4-connected neighbors.
        tolerance: Relative gradient tolerance (from threshold).

    Returns:
        True if this is a flying pixel (should be filtered out).
    """
    # Compute absolute differences
    diff_left = wp.abs(depth_center - d_left)
    diff_right = wp.abs(depth_center - d_right)
    diff_up = wp.abs(depth_center - d_up)
    diff_down = wp.abs(depth_center - d_down)

    # Maximum difference
    max_diff = wp.max(wp.max(diff_left, diff_right), wp.max(diff_up, diff_down))

    # Adaptive threshold based on depth
    adaptive_threshold = tolerance * depth_center

    return max_diff > adaptive_threshold


@wp.kernel
def filter_depth_fused_kernel(
    depth_in: wp.array3d(dtype=wp.float32),
    depth_out: wp.array3d(dtype=wp.float32),
    valid_mask_out: wp.array3d(dtype=wp.uint8),
    # Range filter params
    depth_minimum_distance: float,
    depth_maximum_distance: float,
    # Flying pixel params
    enable_flying_pixel: wp.int32,
    flying_tolerance: float,
    # Bilateral params
    enable_bilateral: wp.int32,
    bilateral_radius: wp.int32,  # (kernel_size - 1) / 2
    sigma_spatial_sq2: float,  # 2 * sigma_spatial^2
    sigma_depth_sq2: float,  # 2 * sigma_depth^2
):
    """Fused depth filtering kernel over a batch of depth images.

    Applies range filter, flying pixel detection, and bilateral smoothing
    in a single pass for maximum efficiency. Neighbor access never crosses
    the batch axis, so images in the batch are filtered independently.

    Args:
        depth_in: Input depth images (B, H, W).
        depth_out: Output filtered depth images (B, H, W).
        valid_mask_out: Output validity masks (B, H, W), 1=valid, 0=invalid.
        depth_minimum_distance, depth_maximum_distance: Valid depth range.
        enable_flying_pixel: 1 to enable flying pixel filter, 0 to disable.
        flying_tolerance: Relative gradient tolerance for flying pixels.
        enable_bilateral: 1 to enable bilateral filter, 0 to disable.
        bilateral_radius: Half-width of bilateral kernel.
        sigma_spatial_sq2: 2 * sigma_spatial^2 for spatial Gaussian.
        sigma_depth_sq2: 2 * sigma_depth^2 for depth Gaussian.
    """
    b, i, j = wp.tid()
    B = depth_in.shape[0]
    H = depth_in.shape[1]
    W = depth_in.shape[2]

    if b >= B or i >= H or j >= W:
        return

    # Read center depth
    d_center = depth_in[b, i, j]

    # --- Stage 1: Range filter ---
    if d_center < depth_minimum_distance or d_center > depth_maximum_distance or not wp.isfinite(d_center):
        depth_out[b, i, j] = 0.0
        valid_mask_out[b, i, j] = wp.uint8(0)
        return

    # --- Stage 2: Flying pixel detection ---
    if enable_flying_pixel != 0:
        # Read 4-connected neighbors (with boundary clamping)
        i_up = wp.max(i - 1, 0)
        i_down = wp.min(i + 1, H - 1)
        j_left = wp.max(j - 1, 0)
        j_right = wp.min(j + 1, W - 1)

        d_left = depth_in[b, i, j_left]
        d_right = depth_in[b, i, j_right]
        d_up = depth_in[b, i_up, j]
        d_down = depth_in[b, i_down, j]

        # Replace invalid neighbors with center depth (don't trigger false positives)
        if d_left < depth_minimum_distance or d_left > depth_maximum_distance:
            d_left = d_center
        if d_right < depth_minimum_distance or d_right > depth_maximum_distance:
            d_right = d_center
        if d_up < depth_minimum_distance or d_up > depth_maximum_distance:
            d_up = d_center
        if d_down < depth_minimum_distance or d_down > depth_maximum_distance:
            d_down = d_center

        if is_flying_pixel(d_center, d_left, d_right, d_up, d_down, flying_tolerance):
            depth_out[b, i, j] = 0.0
            valid_mask_out[b, i, j] = wp.uint8(0)
            return

    # --- Stage 3: Bilateral filter ---
    if enable_bilateral != 0:
        sum_val = float(0.0)
        sum_weight = float(0.0)

        # Iterate over kernel neighborhood
        for di in range(-bilateral_radius, bilateral_radius + 1):
            for dj in range(-bilateral_radius, bilateral_radius + 1):
                ni = i + di
                nj = j + dj

                # Boundary check
                if ni < 0 or ni >= H or nj < 0 or nj >= W:
                    continue

                d_neighbor = depth_in[b, ni, nj]

                # Skip invalid neighbors
                if d_neighbor < depth_minimum_distance or d_neighbor > depth_maximum_distance:
                    continue

                # Spatial weight: exp(-dist^2 / (2 * sigma_spatial^2))
                spatial_dist_sq = float(di * di + dj * dj)
                w_spatial = wp.exp(-spatial_dist_sq / sigma_spatial_sq2)

                # Depth weight: exp(-depth_diff^2 / (2 * sigma_depth^2))
                depth_diff = d_neighbor - d_center
                w_depth = wp.exp(-(depth_diff * depth_diff) / sigma_depth_sq2)

                # Combined weight
                w = w_spatial * w_depth
                sum_val += d_neighbor * w
                sum_weight += w

        # Compute filtered depth
        if sum_weight > 1e-8:
            depth_out[b, i, j] = sum_val / sum_weight
        else:
            depth_out[b, i, j] = d_center
    else:
        # No bilateral, just pass through
        depth_out[b, i, j] = d_center

    valid_mask_out[b, i, j] = wp.uint8(1)


# =============================================================================
# Separable bilateral filter (for large kernels)
# =============================================================================


@wp.kernel
def bilateral_filter_separable_h_kernel(
    depth_in: wp.array3d(dtype=wp.float32),
    depth_out: wp.array3d(dtype=wp.float32),
    radius: wp.int32,
    sigma_spatial_sq2: float,
    sigma_depth_sq2: float,
    depth_minimum_distance: float,
    depth_maximum_distance: float,
):
    """Horizontal pass of separable bilateral filter approximation.

    Operates on a batch of depth images (B, H, W). Bilateral filter is not
    truly separable, but this approximation works well in practice and is
    ~2x faster for large kernels.
    """
    b, i, j = wp.tid()
    B = depth_in.shape[0]
    H = depth_in.shape[1]
    W = depth_in.shape[2]

    if b >= B or i >= H or j >= W:
        return

    d_center = depth_in[b, i, j]
    if d_center < depth_minimum_distance or d_center > depth_maximum_distance:
        depth_out[b, i, j] = 0.0
        return

    sum_val = float(0.0)
    sum_weight = float(0.0)

    for dj in range(-radius, radius + 1):
        nj = j + dj
        if nj < 0 or nj >= W:
            continue

        d_neighbor = depth_in[b, i, nj]
        if d_neighbor < depth_minimum_distance or d_neighbor > depth_maximum_distance:
            continue

        w_spatial = wp.exp(-float(dj * dj) / sigma_spatial_sq2)
        depth_diff = d_neighbor - d_center
        w_depth = wp.exp(-(depth_diff * depth_diff) / sigma_depth_sq2)

        w = w_spatial * w_depth
        sum_val += d_neighbor * w
        sum_weight += w

    if sum_weight > 1e-8:
        depth_out[b, i, j] = sum_val / sum_weight
    else:
        depth_out[b, i, j] = d_center


@wp.kernel
def bilateral_filter_separable_v_kernel(
    depth_in: wp.array3d(dtype=wp.float32),
    depth_out: wp.array3d(dtype=wp.float32),
    radius: wp.int32,
    sigma_spatial_sq2: float,
    sigma_depth_sq2: float,
    depth_minimum_distance: float,
    depth_maximum_distance: float,
):
    """Vertical pass of separable bilateral filter approximation (batched)."""
    b, i, j = wp.tid()
    B = depth_in.shape[0]
    H = depth_in.shape[1]
    W = depth_in.shape[2]

    if b >= B or i >= H or j >= W:
        return

    d_center = depth_in[b, i, j]
    if d_center < depth_minimum_distance or d_center > depth_maximum_distance:
        depth_out[b, i, j] = 0.0
        return

    sum_val = float(0.0)
    sum_weight = float(0.0)

    for di in range(-radius, radius + 1):
        ni = i + di
        if ni < 0 or ni >= H:
            continue

        d_neighbor = depth_in[b, ni, j]
        if d_neighbor < depth_minimum_distance or d_neighbor > depth_maximum_distance:
            continue

        w_spatial = wp.exp(-float(di * di) / sigma_spatial_sq2)
        depth_diff = d_neighbor - d_center
        w_depth = wp.exp(-(depth_diff * depth_diff) / sigma_depth_sq2)

        w = w_spatial * w_depth
        sum_val += d_neighbor * w
        sum_weight += w

    if sum_weight > 1e-8:
        depth_out[b, i, j] = sum_val / sum_weight
    else:
        depth_out[b, i, j] = d_center
