# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

# Standard Library
from enum import Enum


class ControlSpace(Enum):
    POSITION = 0
    VELOCITY = 1
    ACCELERATION = 2
    BSPLINE_3 = 3
    BSPLINE_4 = 4
    BSPLINE_5 = 5

    @staticmethod
    def bspline_types():
        return [ControlSpace.BSPLINE_3, ControlSpace.BSPLINE_4, ControlSpace.BSPLINE_5]

    @staticmethod
    def position_types():
        return [ControlSpace.POSITION] + ControlSpace.bspline_types()

    @staticmethod
    def spline_degree(control_space: ControlSpace):
        if control_space in ControlSpace.bspline_types():
            if control_space == ControlSpace.BSPLINE_3:
                return 3
            if control_space == ControlSpace.BSPLINE_4:
                return 4
            if control_space == ControlSpace.BSPLINE_5:
                return 5
        return 0

    @staticmethod
    def spline_total_knots(control_space: ControlSpace, action_knots: int) -> int:
        if control_space not in ControlSpace.bspline_types():
            return action_knots
        return action_knots + ControlSpace.spline_degree(control_space) + 1

    @staticmethod
    def spline_total_interpolation_steps(
        control_space: ControlSpace, action_knots: int, interpolation_steps: int
    ) -> int:
        return (
            ControlSpace.spline_total_knots(control_space, action_knots) * interpolation_steps + 1
        )
