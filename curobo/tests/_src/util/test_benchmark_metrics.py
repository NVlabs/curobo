# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Standard Library

# Third Party
import numpy as np
import pytest


class TestBenchmarkMetrics:
    def test_curobo_metrics_creation(self):
        """Test CuroboMetrics dataclass creation."""
        try:
            # CuRobo
            from curobo._src.util.benchmark_metrics import CuroboMetrics

            metrics = CuroboMetrics(
                success=True,
                time=1.5,
                cspace_path_length=10.5,
                perception_success=True,
                jerk=0.5,
            )
            assert metrics.success is True
            assert metrics.time == 1.5
            assert metrics.cspace_path_length == 10.5
            assert metrics.perception_success is True
            assert metrics.jerk == 0.5
        except ImportError:
            pytest.skip("robometrics not installed")

    def test_curobo_metrics_defaults(self):
        """Test CuroboMetrics default values."""
        try:
            # CuRobo
            from curobo._src.util.benchmark_metrics import CuroboMetrics

            metrics = CuroboMetrics(success=True)
            assert metrics.time == np.inf
            assert metrics.cspace_path_length == 0.0
            assert metrics.perception_success is False
            assert metrics.jerk == np.inf
            assert metrics.energy == 0.0
            assert metrics.torque == 0.0
            assert metrics.power == 0.0
            assert metrics.work == 0.0
            assert metrics.peak_power == 0.0
        except ImportError:
            pytest.skip("robometrics not installed")

    def test_curobo_group_metrics_from_list(self):
        """Test CuroboGroupMetrics.from_list method."""
        try:
            # CuRobo
            from curobo._src.util.benchmark_metrics import CuroboGroupMetrics, CuroboMetrics

            metrics_list = [
                CuroboMetrics(
                    success=True,
                    skip=False,
                    time=1.0,
                    cspace_path_length=10.0,
                    perception_success=True,
                    jerk=0.5,
                    energy=1.0,
                    torque=0.5,
                    power=2.0,
                    peak_power=3.0,
                    work=1.5,
                    eef_position_path_length=5.0,
                    eef_orientation_path_length=2.0,
                    position_error=0.01,
                    orientation_error=0.02,
                    motion_time=1.5,
                    solve_time=0.1,
                ),
                CuroboMetrics(
                    success=True,
                    skip=False,
                    time=2.0,
                    cspace_path_length=20.0,
                    perception_success=False,
                    jerk=1.0,
                    energy=2.0,
                    torque=1.0,
                    power=3.0,
                    peak_power=4.0,
                    work=2.5,
                    eef_position_path_length=6.0,
                    eef_orientation_path_length=2.5,
                    position_error=0.015,
                    orientation_error=0.025,
                    motion_time=2.5,
                    solve_time=0.15,
                ),
                CuroboMetrics(
                    success=False,
                    skip=False,
                    time=3.0,
                    perception_success=False,
                    jerk=1.5,
                    eef_position_path_length=4.0,
                    eef_orientation_path_length=1.8,
                    position_error=0.1,
                    orientation_error=0.1,
                    motion_time=3.5,
                    solve_time=0.2,
                ),
            ]

            group = CuroboGroupMetrics.from_list(metrics_list)
            assert group is not None
            # Check that statistics were computed for successes
            assert group.time is not None
            assert group.cspace_path_length is not None
        except ImportError:
            pytest.skip("robometrics not installed")

    def test_curobo_group_metrics_with_skipped(self):
        """Test CuroboGroupMetrics handles skipped metrics."""
        try:
            # CuRobo
            from curobo._src.util.benchmark_metrics import CuroboGroupMetrics, CuroboMetrics

            metrics_list = [
                CuroboMetrics(
                    success=True,
                    skip=False,
                    time=1.0,
                    cspace_path_length=10.0,
                    jerk=0.5,
                    perception_time=0.1,
                    energy=1.0,
                    torque=0.5,
                    power=2.0,
                    peak_power=3.0,
                    work=1.5,
                    eef_position_path_length=5.0,
                    eef_orientation_path_length=2.0,
                    position_error=0.01,
                    orientation_error=0.02,
                    motion_time=1.5,
                    solve_time=0.1,
                ),
                CuroboMetrics(
                    success=True,
                    skip=True,
                    time=2.0,
                    cspace_path_length=15.0,
                    jerk=0.6,
                    perception_time=0.15,
                    energy=1.5,
                    torque=0.6,
                    power=2.5,
                    peak_power=3.5,
                    work=2.0,
                    eef_position_path_length=6.0,
                    eef_orientation_path_length=2.5,
                    position_error=0.015,
                    orientation_error=0.025,
                    motion_time=2.5,
                    solve_time=0.15,
                ),  # Should be filtered out
                CuroboMetrics(
                    success=True,
                    skip=False,
                    time=3.0,
                    cspace_path_length=12.0,
                    jerk=0.55,
                    perception_time=0.12,
                    energy=1.2,
                    torque=0.52,
                    power=2.2,
                    peak_power=3.2,
                    work=1.7,
                    eef_position_path_length=4.5,
                    eef_orientation_path_length=1.8,
                    position_error=0.012,
                    orientation_error=0.022,
                    motion_time=3.5,
                    solve_time=0.12,
                ),
            ]

            group = CuroboGroupMetrics.from_list(metrics_list)
            assert group is not None
        except ImportError:
            pytest.skip("robometrics not installed")

    def test_curobo_group_metrics_perception_success_rate(self):
        """Test perception success rate calculation."""
        try:
            # CuRobo
            from curobo._src.util.benchmark_metrics import CuroboGroupMetrics, CuroboMetrics

            metrics_list = [
                CuroboMetrics(
                    success=True,
                    skip=False,
                    perception_success=True,
                    time=1.0,
                    cspace_path_length=10.0,
                    jerk=0.5,
                    perception_time=0.1,
                    energy=1.0,
                    torque=0.5,
                    power=2.0,
                    peak_power=3.0,
                    work=1.5,
                    eef_position_path_length=5.0,
                    eef_orientation_path_length=2.0,
                    position_error=0.01,
                    orientation_error=0.02,
                    motion_time=1.5,
                    solve_time=0.1,
                ),
                CuroboMetrics(
                    success=True,
                    skip=False,
                    perception_success=True,
                    time=1.1,
                    cspace_path_length=11.0,
                    jerk=0.52,
                    perception_time=0.11,
                    energy=1.1,
                    torque=0.52,
                    power=2.1,
                    peak_power=3.1,
                    work=1.6,
                    eef_position_path_length=5.5,
                    eef_orientation_path_length=2.1,
                    position_error=0.011,
                    orientation_error=0.021,
                    motion_time=1.6,
                    solve_time=0.11,
                ),
                CuroboMetrics(
                    success=True,
                    skip=False,
                    perception_success=False,
                    time=0.9,
                    cspace_path_length=9.5,
                    jerk=0.48,
                    perception_time=0.09,
                    energy=0.9,
                    torque=0.48,
                    power=1.9,
                    peak_power=2.9,
                    work=1.4,
                    eef_position_path_length=4.8,
                    eef_orientation_path_length=1.9,
                    position_error=0.009,
                    orientation_error=0.019,
                    motion_time=1.4,
                    solve_time=0.09,
                ),
                CuroboMetrics(
                    success=True,
                    skip=False,
                    perception_success=False,
                    time=1.05,
                    cspace_path_length=10.2,
                    jerk=0.51,
                    perception_time=0.105,
                    energy=1.05,
                    torque=0.51,
                    power=2.05,
                    peak_power=3.05,
                    work=1.55,
                    eef_position_path_length=5.2,
                    eef_orientation_path_length=2.0,
                    position_error=0.01,
                    orientation_error=0.02,
                    motion_time=1.55,
                    solve_time=0.105,
                ),
            ]

            group = CuroboGroupMetrics.from_list(metrics_list)
            # Should be 50% (2 out of 4)
            assert group.perception_success == 50.0
        except ImportError:
            pytest.skip("robometrics not installed")

