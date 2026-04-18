# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Solver module for inverse kinematics, trajectory optimization, and MPC.

This module provides optimization-based solvers for robot motion planning:
- IKSolver: Inverse kinematics solver
- TrajOptSolver: Trajectory optimization solver
- MPCSolver: Model predictive control solver
"""

from curobo._src.solver.manager_goal import GoalManager
from curobo._src.solver.manager_seed import SeedManager

# Seed IK Solver
from curobo._src.solver.seed_ik import (
    SeedIKSolver,
    SeedIKSolverCfg,
)
from curobo._src.solver.solve_mode import SolveMode, SolveModeInput, parse_solve_mode
from curobo._src.solver.solve_state import (
    MotionPlanSolveState,
    SolveState,
)
from curobo._src.solver.solver_base_result import BaseSolverResult
from curobo._src.solver.solver_core import SolverCore
from curobo._src.solver.solver_core_cfg import SolverCoreCfg
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.solver.solver_ik_result import IKSolverResult
from curobo._src.solver.solver_mpc import MPCSolver
from curobo._src.solver.solver_mpc_cfg import MPCSolverCfg
from curobo._src.solver.solver_mpc_result import MPCSolverResult
from curobo._src.solver.solver_trajopt import TrajOptSolver
from curobo._src.solver.solver_trajopt_cfg import TrajOptSolverCfg
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult

__all__ = [
    # Result base class
    "BaseSolverResult",
    # SolverCore (new composition-based infrastructure)
    "SolverCore",
    "SolverCoreCfg",
    # IK Solver
    "IKSolver",
    "IKSolverCfg",
    "IKSolverResult",
    # TrajOpt Solver
    "TrajOptSolver",
    "TrajOptSolverCfg",
    "TrajOptSolverResult",
    # MPC Solver
    "MPCSolver",
    "MPCSolverCfg",
    "MPCSolverResult",
    # Solve mode and state
    "SolveMode",
    "SolveModeInput",
    "parse_solve_mode",
    "SolveState",
    "MotionPlanSolveState",
    # Managers
    "SeedManager",
    "GoalManager",
    # Seed IK Solver
    "SeedIKSolver",
    "SeedIKSolverCfg",
]
