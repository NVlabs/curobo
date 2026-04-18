# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# Standard Library
from dataclasses import dataclass

# CuRobo
from curobo._src.solver.solver_base_result import BaseSolverResult


@dataclass
class IKSolverResult(BaseSolverResult):
    """Result specific to the Inverse Kinematics solver."""

    # Inherits fields: success, solution, js_solution, position_error, rotation_error,
    # goalset_index, solve_time, debug_info, optimized_seeds
    # Add any IK-specific result fields here if needed in the future.
    # Example: seed: Optional[torch.Tensor] = None # Could store the specific seed used for this solution
    pass

