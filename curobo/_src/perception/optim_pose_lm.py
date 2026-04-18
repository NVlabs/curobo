# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Levenberg-Marquardt utilities for SE(3) pose optimization.

Shared utilities for LM-based pose refinement used by:
- SDFPoseDetector (mesh SDF-based)
- PoseRefinerRaycast (TSDF raycast-based)

Key functions:
- compute_predicted_reduction: Compute predicted cost reduction from linear model
- trust_region_update: Trust region update with step acceptance/rejection
- solve_lm_step: Solve LM linear system via Cholesky
"""

from __future__ import annotations

from typing import Tuple

import torch

from curobo._src.util.torch_util import get_profiler_decorator, get_torch_jit_decorator


@get_profiler_decorator("optim_pose_lm/compute_predicted_reduction")
@get_torch_jit_decorator(dynamic=False)
def compute_predicted_reduction(
    delta: torch.Tensor,
    Jtr: torch.Tensor,
    JtJ: torch.Tensor,
) -> torch.Tensor:
    """Compute predicted cost reduction from linear model.

    For cost = 0.5 * r^T @ r, the predicted reduction is:
        pred = -delta^T @ Jtr - 0.5 * delta^T @ JtJ @ delta

    Args:
        delta: [6] pose update vector.
        Jtr: [6] J^T @ r.
        JtJ: [6, 6] J^T @ J.

    Returns:
        Scalar predicted reduction.
    """
    term1 = -torch.dot(delta, Jtr)
    term2 = -0.5 * torch.dot(delta, JtJ @ delta)
    return term1 + term2


@get_profiler_decorator("optim_pose_lm/trust_region_update")
@get_torch_jit_decorator(dynamic=False)
def trust_region_update(
    cand_n_valid: torch.Tensor,
    sum_sq_residuals: torch.Tensor,
    cand_JtJ: torch.Tensor,
    cand_Jtr: torch.Tensor,
    cand_position: torch.Tensor,
    cand_quaternion: torch.Tensor,
    best_error: torch.Tensor,
    best_sum_sq: torch.Tensor,
    best_n_valid: torch.Tensor,
    best_JtJ: torch.Tensor,
    best_Jtr: torch.Tensor,
    best_position: torch.Tensor,
    best_quaternion: torch.Tensor,
    pred_reduction: torch.Tensor,
    lambda_damping: torch.Tensor,
    n_total_valid: torch.Tensor,
    min_valid_ratio: float,
    rho_min: float,
    lambda_factor: float,
    lambda_min: float,
    lambda_max: float,
    inf_tensor: torch.Tensor,
    minimum_valid_count: int = 10,

) -> Tuple[
    torch.Tensor,  # new_best_position
    torch.Tensor,  # new_best_quaternion
    torch.Tensor,  # new_best_error
    torch.Tensor,  # new_best_sum_sq
    torch.Tensor,  # new_best_n_valid
    torch.Tensor,  # new_best_JtJ
    torch.Tensor,  # new_best_Jtr
    torch.Tensor,  # new_lambda
]:
    """Trust region update for LM optimization.

    Computes candidate error, trust ratio, accept/reject decision, and
    selects new state values based on acceptance.

    Note: The trust ratio is computed using sum_sq (sum of squared residuals)
    to match the scale of pred_reduction, which is also in sum_sq units.

    Args:
        cand_n_valid: Number of valid correspondences at candidate pose.
        sum_sq_residuals: Sum of squared residuals at candidate pose.
        cand_JtJ: [6, 6] J^T @ J at candidate pose.
        cand_Jtr: [6] J^T @ r at candidate pose.
        cand_position: Candidate position.
        cand_quaternion: Candidate quaternion.
        best_error: Current best RMS error (for reporting).
        best_sum_sq: Current best sum of squared residuals (for trust ratio).
        best_n_valid: Current best n_valid.
        best_JtJ: [6, 6] J^T @ J at current best pose.
        best_Jtr: [6] J^T @ r at current best pose.
        best_position: Current best position.
        best_quaternion: Current best quaternion.
        pred_reduction: Predicted cost reduction from linear model (in sum_sq units).
        lambda_damping: Current LM damping parameter.
        n_total_valid: Total valid points/pixels for ratio computation.
        min_valid_ratio: Minimum ratio of valid correspondences.
        rho_min: Minimum trust ratio for acceptance.
        lambda_factor: Factor for lambda adaptation.
        lambda_min: Minimum lambda value.
        lambda_max: Maximum lambda value.
        inf_tensor: Pre-allocated infinity tensor (for CUDA graph compatibility).

    Returns:
        Tuple of updated state values including new_best_sum_sq.
    """
    # Check if candidate has enough valid correspondences
    has_enough_valid = cand_n_valid > minimum_valid_count

    # Compute candidate error (inf if not enough valid) - for reporting
    cand_error = torch.where(
        has_enough_valid,
        torch.sqrt(sum_sq_residuals / (cand_n_valid + 1e-8)),
        inf_tensor,
    )

    # Compute trust ratio using sum_sq (same units as pred_reduction)
    # pred_reduction is computed as: -delta^T @ Jtr - 0.5 * delta^T @ JtJ @ delta
    # which is the predicted reduction in 0.5 * ||r||^2 = 0.5 * sum_sq
    # So actual_reduction should also use sum_sq for consistent scaling
    actual_reduction = (best_sum_sq - sum_sq_residuals)
    trust_ratio = actual_reduction / (pred_reduction + 1e-8)


    # Accept if trust_ratio >= rho_min AND enough valid correspondences
    step_accepted = torch.logical_and(trust_ratio >= 0, has_enough_valid)


    # Update lambda based on acceptance
    new_lambda = torch.where(
        step_accepted,
        lambda_damping / lambda_factor,  # Decrease on accept
        lambda_damping * lambda_factor,  # Increase on reject
    )
    new_lambda = torch.clamp(new_lambda, lambda_min, lambda_max)

    # Select state values based on acceptance
    new_best_position = torch.where(step_accepted, cand_position, best_position)
    new_best_quaternion = torch.where(step_accepted, cand_quaternion, best_quaternion)
    new_best_error = torch.where(step_accepted, cand_error, best_error)
    new_best_sum_sq = torch.where(step_accepted, sum_sq_residuals, best_sum_sq)
    new_best_n_valid = torch.where(step_accepted, cand_n_valid, best_n_valid)

    # For JtJ/Jtr, broadcast scalar condition
    accept_mask_2d = step_accepted.view(1, 1).expand(6, 6)
    new_best_JtJ = torch.where(accept_mask_2d, cand_JtJ, best_JtJ)
    accept_mask_1d = step_accepted.view(1).expand(6)
    new_best_Jtr = torch.where(accept_mask_1d, cand_Jtr, best_Jtr)

    return (
        new_best_position,
        new_best_quaternion,
        new_best_error,
        new_best_sum_sq,
        new_best_n_valid,
        new_best_JtJ,
        new_best_Jtr,
        new_lambda,
    )


@get_profiler_decorator("optim_pose_lm/solve_lm_step")
@get_torch_jit_decorator(dynamic=False)
def solve_lm_step(
    JtJ: torch.Tensor,
    Jtr: torch.Tensor,
    lambda_damping: torch.Tensor,
    eye6: torch.Tensor,
) -> torch.Tensor:
    """Solve LM step via Cholesky.

    Solves (J^T @ J + lambda * I) @ delta = -J^T @ r.

    Args:
        JtJ: [6, 6] J^T @ J matrix.
        Jtr: [6] J^T @ r vector.
        lambda_damping: Scalar LM damping parameter.
        eye6: [6, 6] identity matrix for damping.

    Returns:
        delta: [6] pose update vector.
    """
    A = JtJ + lambda_damping * eye6
    L, _ = torch.linalg.cholesky_ex(A)
    delta = torch.cholesky_solve((-Jtr).unsqueeze(1), L).squeeze(1)
    return delta

