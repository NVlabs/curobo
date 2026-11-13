# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared geometry of a fixed B-spline optimization grid."""

from __future__ import annotations

import math
from dataclasses import dataclass

from curobo._src.types.control_space import ControlSpace
from curobo._src.util.logging import log_and_raise

MIN_TRAJOPT_KNOTS = 12
MAX_SPLINE_KNOTS = 1024


def validate_trajopt_knots(num_knots: int) -> None:
    """Require an initialization-time control count supported by trajectory optimization."""
    if (
        isinstance(num_knots, bool)
        or not isinstance(num_knots, int)
        or not MIN_TRAJOPT_KNOTS <= num_knots <= MAX_SPLINE_KNOTS
    ):
        log_and_raise(
            f"num_knots must be an integer in [{MIN_TRAJOPT_KNOTS}, {MAX_SPLINE_KNOTS}]"
        )


@dataclass(frozen=True)
class SplineLayout:
    """Geometry shared by spline optimization, reference construction, and dense output."""

    #: Number of optimized action controls, excluding boundary support.
    num_knots: int
    #: B-spline degree and boundary convention.
    control_space: ControlSpace = ControlSpace.BSPLINE_3
    #: Native samples per spline span.
    interpolation_steps: int = 4

    @property
    def degree(self) -> int:
        """Degree of the configured B-spline basis."""
        return ControlSpace.spline_degree(self.control_space)

    @property
    def native_horizon(self) -> int:
        """Number of native states, including the final endpoint."""
        return ControlSpace.spline_total_interpolation_steps(
            self.control_space, self.num_knots, self.interpolation_steps
        )

    @property
    def padding(self) -> int:
        """Native intervals reserved for each stationary boundary span."""
        return self.interpolation_steps

    @property
    def reference_intervals(self) -> int:
        """Active native intervals between the stationary boundary spans."""
        return self.native_horizon - 2 * self.padding - 1

    def dense_horizon(self, maximum_dt: float, interpolation_dt: float) -> int:
        """Capacity for dense output over the configured native-timestep range."""
        if (
            not math.isfinite(maximum_dt)
            or maximum_dt <= 0
            or not math.isfinite(interpolation_dt)
            or interpolation_dt <= 0
        ):
            log_and_raise("dense sampling timesteps must be finite and positive")
        steps = math.ceil(maximum_dt * self.interpolation_steps / interpolation_dt) + 1
        return ControlSpace.spline_total_interpolation_steps(
            self.control_space, self.num_knots, steps
        )

    def waypoint_indices(self, num_waypoints: int) -> list[int]:
        """Native target indices, excluding the separately supplied start pose."""
        return reference_sample_indices(self.native_horizon, self.padding, num_waypoints)


def reference_sample_indices(native_horizon: int, padding: int, num_waypoints: int) -> list[int]:
    """Indices of retained targets on a padded native reference grid."""
    intervals = native_horizon - 2 * padding - 1
    if padding < 1 or not 1 <= num_waypoints <= intervals:
        log_and_raise(
            f"reference grid cannot preserve {num_waypoints} targets; "
            f"it supports 1 to {intervals} with the configured native horizon and padding"
        )
    return [padding + round(intervals * i / num_waypoints) for i in range(1, num_waypoints + 1)]
