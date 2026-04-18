# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inverse kinematics solver module.

This module provides collision-aware inverse kinematics solving with optimization.

Example:
    ```python
    from curobo import InverseKinematics, InverseKinematicsCfg

    config = InverseKinematicsCfg.create(
        robot="franka.yml",
        num_seeds=12,
    )
    ik = InverseKinematics(config)
    result = ik.solve_pose(goal_tool_poses=target_poses)
    ```
"""

from curobo._src.solver.solver_ik import IKSolver as InverseKinematics
from curobo._src.solver.solver_ik_cfg import IKSolverCfg as InverseKinematicsCfg
from curobo._src.solver.solver_ik_result import IKSolverResult as InverseKinematicsResult

__all__ = [
    "InverseKinematics",
    "InverseKinematicsCfg",
    "InverseKinematicsResult",
]
