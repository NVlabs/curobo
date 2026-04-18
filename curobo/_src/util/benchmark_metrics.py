# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Standard Library
from dataclasses import dataclass
from typing import List, Optional, Sequence

# Third Party
import numpy as np


def percent_true(arr: Sequence) -> float:
    """Return the percentage of truthy / nonzero elements in a sequence.

    Args:
        arr: Boolean or numerical sequence.

    Returns:
        Percentage in [0, 100]. Returns 0 for an empty sequence.
    """
    if len(arr) == 0:
        return 0.0
    return 100.0 * np.count_nonzero(arr) / len(arr)


@dataclass
class Statistic:
    """Descriptive statistics for a list of values."""

    mean: float
    std: float
    median: float
    percent_25: float
    percent_75: float
    percent_98: float
    min: float
    max: float

    @classmethod
    def from_list(cls, lst: Sequence[float]) -> "Statistic":
        """Create a :class:`Statistic` from a list of floats.

        Values equal to ``np.inf`` are filtered out before computation.

        Args:
            lst: Input values.

        Returns:
            Computed statistics (all zeros when *lst* is empty).
        """
        if not lst:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        filtered = [v for v in lst if v < np.inf]
        if not filtered:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        return cls(
            mean=float(np.mean(filtered)),
            std=float(np.std(filtered)),
            median=float(np.median(filtered)),
            percent_25=float(np.percentile(filtered, 25)),
            percent_75=float(np.percentile(filtered, 75)),
            percent_98=float(np.percentile(filtered, 98)),
            min=float(np.min(filtered)),
            max=float(np.max(filtered)),
        )

    def __str__(self) -> str:
        return (
            f"mean: {self.mean:2.3f} \u00b1 {self.std:2.3f}"
            f"  median: {self.median:2.3f}"
            f"  75%: {self.percent_75:2.3f}"
            f"  98%: {self.percent_98:2.3f}"
        )


@dataclass
class CuroboMetrics:
    """Per-trajectory evaluation metrics for cuRobo benchmarks."""

    # Trajectory status
    skip: bool = True
    success: bool = False
    collision: bool = True
    joint_limit_violation: bool = True
    self_collision: bool = True
    physical_violation: bool = True
    payload_success: bool = False
    perception_success: bool = False
    perception_interpolated_success: bool = False

    # Errors
    position_error: float = np.inf
    orientation_error: float = np.inf

    # Path lengths
    eef_position_path_length: float = np.inf
    eef_orientation_path_length: float = np.inf
    cspace_path_length: float = 0.0
    trajectory_length: int = 1

    # Timing
    attempts: int = 1
    motion_time: float = np.inf
    solve_time: float = np.inf
    time: float = np.inf
    perception_time: float = 0.0

    # Dynamics
    jerk: float = np.inf
    energy: float = 0.0
    torque: float = 0.0
    power: float = 0.0
    work: float = 0.0
    peak_power: float = 0.0


@dataclass
class CuroboGroupMetrics:
    """Aggregate metrics over a group of cuRobo trajectories."""

    # Counts / rates
    group_size: int = 0
    success: float = 0.0
    skips: int = 0
    env_collision_rate: float = 0.0
    self_collision_rate: float = 0.0
    joint_violation_rate: float = 0.0
    physical_violation_rate: float = 0.0
    payload_success: float = 0.0
    perception_success: float = 0.0
    perception_interpolated_success: float = 0.0

    # Accuracy rates
    within_one_cm_rate: float = 0.0
    within_five_cm_rate: float = 0.0
    within_fifteen_deg_rate: float = 0.0
    within_thirty_deg_rate: float = 0.0

    # Statistics (populated by from_list)
    eef_position_path_length: Optional[Statistic] = None
    eef_orientation_path_length: Optional[Statistic] = None
    cspace_path_length: Optional[Statistic] = None
    attempts: Optional[Statistic] = None
    position_error: Optional[Statistic] = None
    orientation_error: Optional[Statistic] = None
    motion_time: Optional[Statistic] = None
    solve_time: Optional[Statistic] = None
    solve_time_per_step: Optional[Statistic] = None
    time: Optional[Statistic] = None
    perception_time: Optional[Statistic] = None
    jerk: Optional[Statistic] = None
    energy: Optional[Statistic] = None
    torque: Optional[Statistic] = None
    power: Optional[Statistic] = None
    work: Optional[Statistic] = None
    peak_power: Optional[Statistic] = None

    @classmethod
    def from_list(cls, group: List[CuroboMetrics]) -> "CuroboGroupMetrics":
        """Build group-level metrics from a list of per-trajectory metrics."""
        unskipped = [m for m in group if not m.skip]
        successes = [m for m in unskipped if m.success]

        return cls(
            # Counts / rates
            group_size=len(group),
            success=percent_true([m.success for m in group]),
            skips=len([m for m in group if m.skip]),
            env_collision_rate=percent_true([m.collision for m in unskipped]),
            self_collision_rate=percent_true([m.self_collision for m in unskipped]),
            joint_violation_rate=percent_true(
                [m.joint_limit_violation for m in unskipped]
            ),
            physical_violation_rate=percent_true(
                [m.physical_violation for m in unskipped]
            ),
            payload_success=percent_true([m.payload_success for m in group]),
            perception_success=percent_true(
                [m.perception_success for m in group]
            ),
            perception_interpolated_success=percent_true(
                [m.perception_interpolated_success for m in group]
            ),
            # Accuracy rates
            within_one_cm_rate=percent_true(
                [m.position_error < 1 for m in unskipped]
            ),
            within_five_cm_rate=percent_true(
                [m.position_error < 5 for m in unskipped]
            ),
            within_fifteen_deg_rate=percent_true(
                [m.orientation_error < 15 for m in unskipped]
            ),
            within_thirty_deg_rate=percent_true(
                [m.orientation_error < 30 for m in unskipped]
            ),
            # Statistics
            eef_position_path_length=Statistic.from_list(
                [m.eef_position_path_length for m in successes]
            ),
            eef_orientation_path_length=Statistic.from_list(
                [m.eef_orientation_path_length for m in successes]
            ),
            cspace_path_length=Statistic.from_list(
                [m.cspace_path_length for m in successes]
            ),
            attempts=Statistic.from_list([m.attempts for m in successes]),
            position_error=Statistic.from_list(
                [m.position_error for m in successes]
            ),
            orientation_error=Statistic.from_list(
                [m.orientation_error for m in successes]
            ),
            motion_time=Statistic.from_list([m.motion_time for m in successes]),
            solve_time=Statistic.from_list([m.solve_time for m in successes]),
            solve_time_per_step=Statistic.from_list(
                [m.solve_time / m.trajectory_length for m in successes]
            ),
            time=Statistic.from_list([m.time for m in successes]),
            perception_time=Statistic.from_list(
                [m.perception_time for m in successes]
            ),
            jerk=Statistic.from_list([m.jerk for m in successes]),
            energy=Statistic.from_list([m.energy for m in successes]),
            torque=Statistic.from_list([m.torque for m in successes]),
            power=Statistic.from_list([m.power for m in successes]),
            work=Statistic.from_list([m.work for m in successes]),
            peak_power=Statistic.from_list([m.peak_power for m in successes]),
        )

    def print_summary(self) -> None:
        """Print a human-readable summary to stdout."""
        print(f"Total problems: {self.group_size}")
        print(f"# Skips (Hard Failures): {self.skips}")
        print(f"% Success: {self.success:4.2f}")
        print(f"% Within 1cm: {self.within_one_cm_rate:4.2f}")
        print(f"% Within 5cm: {self.within_five_cm_rate:4.2f}")
        print(f"% Within 15deg: {self.within_fifteen_deg_rate:4.2f}")
        print(f"% Within 30deg: {self.within_thirty_deg_rate:4.2f}")
        print(f"% With Environment Collision: {self.env_collision_rate:4.2f}")
        print(f"% With Self Collision: {self.self_collision_rate:4.2f}")
        print(f"% With Joint Limit Violations: {self.joint_violation_rate:4.2f}")
        print(f"% With Physical Violations: {self.physical_violation_rate:4.2f}")
        print(f"Eef Position Path Length: {self.eef_position_path_length}")
        print(f"Eef Orientation Path Length: {self.eef_orientation_path_length}")
        print(f"Motion Time: {self.motion_time}")
        print(f"Solve Time: {self.solve_time}")
        print(f"Solve Time Per Step: {self.solve_time_per_step}")
