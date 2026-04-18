# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Solve mode enum for optimization-based solvers."""

from enum import Enum
from typing import Literal, Union


class SolveMode(Enum):
    """Solve mode for optimization problems.

    This enum defines the batch mode for solving optimization problems:
    - SINGLE: Solve a single problem (batch_size=1)
    - BATCH: Solve multiple problems with shared environment
    - MULTI_ENV: Solve multiple problems with per-problem environments
    """

    SINGLE = "single"
    BATCH = "batch"
    MULTI_ENV = "multi_env"


# Type alias for solve mode input (accepts enum or string)
SolveModeInput = Union[SolveMode, Literal["single", "batch", "multi_env"]]


def parse_solve_mode(mode: SolveModeInput) -> SolveMode:
    """Parse solve mode input to SolveMode enum.

    Args:
        mode: Solve mode as enum or string.

    Returns:
        SolveMode enum value.
    """
    if isinstance(mode, SolveMode):
        return mode
    return SolveMode(mode)

