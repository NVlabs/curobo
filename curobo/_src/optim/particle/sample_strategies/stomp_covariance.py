# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""STOMP covariance matrix computation."""

# Third Party
from typing import Tuple

import torch
import torch.autograd.profiler as profiler

# CuRobo


@profiler.record_function("particle_opt_utils/get_stomp_cov")
def get_stomp_cov(
    horizon: int,
    zero_out_boundary: bool = True,
    stencil_type: str = "3point",  # "3point", "5point", "7point"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the covariance matrix following STOMP motion planner

    Args:
        horizon: Time horizon for trajectory sampling
        zero_out_boundary: Whether to zero out boundary conditions
        stencil_type: Type of finite difference stencil to use
    """
    # Select finite difference coefficients based on stencil type
    if stencil_type == "3point":
        # 3-point stencil (2nd-order accurate): [1, -2, 1]
        fd_coeffs = torch.tensor([0.0, 0.0, 1.0, -2.0, 1.0, 0.0, 0.0], dtype=torch.float32)
    elif stencil_type == "5point":
        # 5-point stencil (4th-order accurate): [-1/12, 4/3, -5/2, 4/3, -1/12]
        fd_coeffs = torch.tensor(
            [0.0, -1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12, 0.0], dtype=torch.float32
        )
    elif stencil_type == "7point":
        # 7-point stencil (6th-order accurate)
        fd_coeffs = torch.tensor(
            [1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90], dtype=torch.float32
        )
    else:
        raise ValueError(f"Unknown stencil_type: {stencil_type}")

    # Create finite difference matrix using vectorized operations
    A = torch.zeros((horizon, horizon), dtype=torch.float32, device="cpu")

    # For both modes, use standard finite difference approach
    # The original "acc" mode had a bug with negative indexing
    for offset_idx, coeff in enumerate(fd_coeffs):
        if coeff != 0:
            offset = offset_idx - 3  # Convert to actual offset (-3 to 3)

            if offset >= 0:
                # Upper diagonal
                diag_length = horizon - offset
                row_indices = torch.arange(diag_length)
                col_indices = row_indices + offset
            else:
                # Lower diagonal
                diag_length = horizon + offset
                row_indices = torch.arange(-offset, horizon)
                col_indices = torch.arange(diag_length)

            # Boundary clamping for robustness
            col_indices = torch.clamp(col_indices, 0, horizon - 1)
            A[row_indices, col_indices] = coeff

    # Compute covariance matrix: M = (A^T * A)^(-1)
    R = torch.matmul(A.T, A)
    M = torch.inverse(R)

    # Zero out first and last rows/columns to ensure no noise at boundaries
    if zero_out_boundary:
        M[0, :] = 0.0
        M[:, 0] = 0.0
        M[horizon - 1, :] = 0.0
        M[:, horizon - 1] = 0.0

        # Add small diagonal terms for numerical stability at boundaries
        M[0, 0] = 1e-8
        M[horizon - 1, horizon - 1] = 1e-8

    # Create normalized versions
    scaled_M = (1 / horizon) * M / (torch.max(torch.abs(M), dim=1)[0].unsqueeze(0) + 1e-8)
    cov = M / (torch.max(torch.abs(M)) + 1e-8)

    # Ensure symmetry for numerical stability
    cov = (cov + cov.T) / 2

    # Compute Cholesky decomposition if matrix is positive definite
    try:
        if (cov == cov.T).all() and (torch.linalg.eigvals(cov).real >= 0).all():
            scale_tril = torch.linalg.cholesky(cov)
        else:
            scale_tril = cov
    except RuntimeError:
        # Fallback if Cholesky decomposition fails due to boundary modifications
        scale_tril = cov

    return cov, scale_tril, scaled_M
