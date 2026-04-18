# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Utility functions for pose detection."""

from typing import Optional, Tuple

import torch

from curobo._src.geom.transform import matrix_to_quaternion
from curobo._src.types.camera import CameraObservation
from curobo._src.util.torch_util import get_profiler_decorator, get_torch_jit_decorator


def extract_observed_points(
    camera_obs: CameraObservation,
    min_depth: float = 0.1,
) -> torch.Tensor:
    """Extract and filter observed points from camera observation.

    Args:
        camera_obs: Camera observation with depth and optional segmentation.
        min_depth: Minimum valid depth in meters.

    Returns:
        observed_points: [N, 3] filtered points in world frame.
    """
    pointcloud_full = camera_obs.get_pointcloud(project_to_pose=True)

    # Filter by segmentation mask if available
    if camera_obs.image_segmentation is not None:
        mask = camera_obs.image_segmentation > 0
        points_flat = pointcloud_full.view(-1, 3)
        mask_flat = mask.view(-1)
        observed_points = points_flat[mask_flat]
    else:
        observed_points = pointcloud_full.view(-1, 3)

    # Remove invalid points (NaN, inf, depth < min_depth)
    valid_mask = torch.isfinite(observed_points).all(dim=1)
    valid_mask &= observed_points[:, 2].abs() > min_depth
    observed_points = observed_points[valid_mask]

    return observed_points


@get_torch_jit_decorator()
def omega_to_quaternion(omega: torch.Tensor) -> torch.Tensor:
    """Convert rotation vector to quaternion (wxyz format).

    Args:
        omega: [3] rotation vector (axis * angle).

    Returns:
        quaternion: [4] quaternion in wxyz format.
    """
    theta = omega.norm()
    half_theta = theta * 0.5

    # sin(θ/2)/θ → 0.5 as θ → 0 (Taylor series), branchless with clamp
    sinc_coeff = torch.sin(half_theta) / theta.clamp(min=1e-10)
    w = torch.cos(half_theta)
    xyz = omega * sinc_coeff

    return torch.cat([w.unsqueeze(0), xyz])


def huber_loss(residuals: torch.Tensor, delta: float = 0.02) -> torch.Tensor:
    """Huber loss (robust M-estimator).

    Quadratic for small errors, linear for large errors.
    This downweights outliers for more robust estimation.

    Args:
        residuals: [N] or [N, d] residual values.
        delta: Threshold for switching from quadratic to linear (meters).

    Returns:
        losses: [N] or [N, d] Huber losses.
    """
    abs_res = torch.abs(residuals)
    quadratic = 0.5 * residuals**2
    linear = delta * (abs_res - 0.5 * delta)
    return torch.where(abs_res < delta, quadratic, linear)


def find_nearest_neighbors(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    distance_threshold: float = float("inf"),
) -> torch.Tensor:
    """Find nearest neighbor in target for each source point.

    Args:
        source_points: [N, 3] source points.
        target_points: [M, 3] target points.
        distance_threshold: Maximum distance for valid correspondence.

    Returns:
        indices: [N] indices into target_points (-1 if no match within threshold).
    """
    # Compute pairwise distances [N, M]
    dists = torch.cdist(source_points.unsqueeze(0), target_points.unsqueeze(0)).squeeze(0)

    # Find nearest neighbor for each source point
    min_dists, indices = dists.min(dim=1)

    # Invalidate correspondences beyond threshold
    if distance_threshold < float("inf"):
        indices[min_dists > distance_threshold] = -1

    return indices


def resample_points(
    points: torch.Tensor,
    target_count: int,
    device: torch.device = None,
) -> torch.Tensor:
    """Resample point cloud to exactly target_count points.

    Args:
        points: [N, 3] input points.
        target_count: Desired number of points.
        device: Device for the output tensor.

    Returns:
        resampled_points: [target_count, 3] points.
    """
    n_points = len(points)

    if device is None:
        device = points.device

    if n_points >= target_count:
        # Downsample: random selection
        indices = torch.randperm(n_points, device=device)[:target_count]
        return points[indices]
    else:
        # Upsample: repeat points
        repeat_factor = (target_count // n_points) + 1
        points_repeated = points.repeat(repeat_factor, 1)
        indices = torch.randperm(len(points_repeated), device=device)[:target_count]
        return points_repeated[indices]



def compute_pose_point_to_plane_svd(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    target_normals: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    use_huber: bool = False,
    huber_delta: float = 0.02,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute optimal rigid transform using point-to-plane ICP with SVD.

    Uses the projection approach:
    1. Project point differences onto normal directions
    2. Create modified target points along normals
    3. Solve with standard SVD (optionally weighted for robust estimation)

    Args:
        source_points: [N, 3] source points.
        target_points: [N, 3] target points (corresponding).
        target_normals: [N, 3] target surface normals (unit vectors).
        weights: [N] optional weights per point (for robust estimation).
        use_huber: If True, compute Huber weights from distances.
        huber_delta: Huber loss threshold (meters).

    Returns:
        position: [3] translation vector.
        quaternion: [4] quaternion in wxyz format.
    """
    # Compute point-to-plane distances
    diff = target_points - source_points  # [N, 3]
    distances = (diff * target_normals).sum(dim=1, keepdim=True)  # [N, 1]

    # Compute Huber weights if requested
    if use_huber:
        # Huber weight: w(r) = 1 if |r| < δ, else w(r) = δ/|r|
        abs_dist = distances.abs().squeeze()
        huber_weights = torch.where(
            abs_dist < huber_delta, torch.ones_like(abs_dist), huber_delta / (abs_dist + 1e-10)
        )
        if weights is not None:
            weights = huber_weights * weights
        else:
            weights = huber_weights

    # Create projected target points:
    # target_proj = source + distance * normal
    target_projected = source_points + distances * target_normals  # [N, 3]

    # Solve weighted point-to-point ICP with SVD
    if weights is not None:
        # Weighted mean
        weights_normalized = weights / (weights.sum() + 1e-10)
        source_mean = (source_points * weights_normalized.unsqueeze(1)).sum(dim=0, keepdim=True)
        target_proj_mean = (target_projected * weights_normalized.unsqueeze(1)).sum(
            dim=0, keepdim=True
        )

        source_centered = source_points - source_mean
        target_proj_centered = target_projected - target_proj_mean

        # Weighted cross-covariance: H = target^T @ W @ source
        source_weighted = source_centered * weights.unsqueeze(1)
        H = target_proj_centered.T @ source_weighted
    else:
        # Unweighted (standard)
        source_mean = source_points.mean(dim=0, keepdim=True)
        target_proj_mean = target_projected.mean(dim=0, keepdim=True)

        source_centered = source_points - source_mean
        target_proj_centered = target_projected - target_proj_mean

        H = target_proj_centered.T @ source_centered

    # SVD: H = U @ S @ V^T
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.T

    # Compute rotation: R = U @ V^T
    R = U @ V.T

    # Handle reflection (det(R) should be 1, not -1)
    if torch.det(R) < 0:
        V[:, -1] *= -1
        R = U @ V.T

    # Compute translation: t = target_proj_mean - R @ source_mean
    t = target_proj_mean.T - R @ source_mean.T
    position = t.squeeze()  # [3]

    # Convert rotation matrix to quaternion (wxyz format)
    quaternion = matrix_to_quaternion(R.unsqueeze(0)).squeeze(0)  # [4]

    return position, quaternion


@get_profiler_decorator("pose_estimation/compute_transform_cholesky")
@get_torch_jit_decorator()
def compute_pose_point_to_plane_cholesky(
    source_points: torch.Tensor,
    target_points: torch.Tensor,
    target_normals: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    use_huber: bool = False,
    huber_delta: float = 0.02,
    damping: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute rigid transform using Cholesky-based point-to-plane ICP.

    Uses the small-angle approximation to linearize the rotation, resulting in a
    linear least squares problem solved via Cholesky decomposition. More
    numerically stable than normal equations which square the condition number.

    The linearization approximates R ≈ I + [ω]× for small rotation vector ω,
    giving a linear residual: r_i = ω · (s_i × n_i) + t · n_i + (s_i - p_i) · n_i

    Args:
        source_points: [N, 3] source points.
        target_points: [N, 3] target points (corresponding).
        target_normals: [N, 3] target surface normals (unit vectors).
        weights: [N] optional weights per point (for robust estimation).
        use_huber: If True, compute Huber weights from point-to-plane distances.
        huber_delta: Huber loss threshold (meters).

    Returns:
        position: [3] translation vector.
        quaternion: [4] quaternion in wxyz format.
    """
    device = source_points.device
    dtype = source_points.dtype

    # Compute point-to-plane distances
    diff = target_points - source_points  # [N, 3]
    distances = (diff * target_normals).sum(dim=1)  # [N]

    # Compute Huber weights if requested (matches SVD version behavior)
    if use_huber:
        # Huber weight: w(r) = 1 if |r| < δ, else w(r) = δ/|r|
        abs_dist = distances.abs()
        huber_weights = torch.where(
            abs_dist < huber_delta,
            torch.ones_like(abs_dist),
            huber_delta / (abs_dist + 1e-10),
        )
        if weights is not None:
            weights = huber_weights * weights
        else:
            weights = huber_weights

    # Build Jacobian: J[i] = [s_i × n_i | n_i] (rotation part | translation part)
    # Each row corresponds to the linearized point-to-plane constraint
    cross = torch.cross(source_points, target_normals, dim=1)  # [N, 3]

    if weights is not None:
        sqrt_w = weights.sqrt().unsqueeze(1)  # [N, 1]
        target_normals = target_normals * sqrt_w
        cross = cross * sqrt_w
        distances = distances * sqrt_w.squeeze(1)

    J = torch.cat([cross, target_normals], dim=1)  # [N, 6]
    b = distances  # [N]

    # Solve least squares: min ||J @ x - b||^2
    # Use lstsq which uses QR/SVD internally - much more numerically stable
    # than normal equations (J^T J) which square the condition number
    #x = torch.linalg.lstsq(J, b.unsqueeze(1)).solution.squeeze(1)  # [6]

    JtJ = J.T @ J
    Jtb = J.T @ b
    JtJ = JtJ + damping * torch.eye(6, device=device, dtype=dtype)
    L, info = torch.linalg.cholesky_ex(JtJ)
    if info == 0:
        x = torch.cholesky_solve(Jtb.unsqueeze(1), L).squeeze(1)
    else:
        x = torch.linalg.lstsq(J, b.unsqueeze(1)).solution.squeeze(1)

    # Extract rotation vector and translation
    omega = x[:3]
    position = x[3:]

    # Convert rotation vector to quaternion directly
    quaternion = omega_to_quaternion(omega)

    return position, quaternion
