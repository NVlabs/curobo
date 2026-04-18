# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trajectory optimization module.

This module provides collision-aware trajectory generation with optimization.

Example:
    ```python
    from curobo import TrajectoryOptimizer, TrajectoryOptimizerCfg

    config = TrajectoryOptimizerCfg.create(robot="franka.yml")
    trajopt = TrajectoryOptimizer(config)
    result = trajopt.solve_pose(goal_tool_poses=target_poses)
    ```
"""

from curobo._src.solver.solver_trajopt import TrajOptSolver as TrajectoryOptimizer
from curobo._src.solver.solver_trajopt_cfg import (
    TrajOptSolverCfg as TrajectoryOptimizerCfg,
)
from curobo._src.solver.solver_trajopt_result import (
    TrajOptSolverResult as TrajectoryOptimizerResult,
)

__all__ = [
    "TrajectoryOptimizer",
    "TrajectoryOptimizerCfg",
    "TrajectoryOptimizerResult",
]
