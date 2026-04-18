# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for using graph planner output as batch trajectory seeds.

Verifies that ``find_path`` with ``B*S`` flat queries produces
``interpolated_waypoints`` of shape ``(B*S, horizon, dof)`` that can be
reshaped to ``(B, S, horizon, dof)`` for the trajectory optimization
seed manager.
"""

import pytest
import torch

from curobo._src.graph_planner.graph_planner_prm import PRMGraphPlanner, TrajInterpolationType
from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg


def _get_planner():
    return PRMGraphPlanner(
        PRMGraphPlannerCfg.create(robot="franka.yml", scene_model="collision_test.yml")
    )


@pytest.fixture(scope="module")
def planner():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    p = _get_planner()
    p.warmup(max_batch_size=16)
    return p


class TestGraphPlannerBatchSeedShape:
    """Verify interpolated_waypoints shape for batch seed usage."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_interpolated_waypoints_shape_flat_batch(self, planner):
        """find_path with N queries returns interpolated_waypoints of shape (N, H, D)."""
        N = 8
        horizon = 20
        samples = planner.sampling_strategy.generate_feasible_action_samples(100)

        x_start = samples[:N]
        x_goal = samples[50 : 50 + N]

        result = planner.find_path(
            x_start, x_goal,
            interpolate_waypoints=True,
            interpolation_steps=horizon,
            interpolation_type=TrajInterpolationType.LINEAR,
            validate_interpolated_trajectory=False,
        )

        assert result.success.shape == (N,)
        assert result.interpolated_waypoints.shape == (N, horizon, planner.action_dim)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_interpolated_waypoints_reshape_to_batch_seeds(self, planner):
        """Waypoints from B*S queries can be reshaped to (B, S, H, D)."""
        B, S = 4, 2
        N = B * S
        horizon = 16
        samples = planner.sampling_strategy.generate_feasible_action_samples(100)

        x_start = samples[0:1].repeat(N, 1)
        x_goal = samples[50 : 50 + N]

        result = planner.find_path(
            x_start, x_goal,
            interpolate_waypoints=True,
            interpolation_steps=horizon,
            interpolation_type=TrajInterpolationType.LINEAR,
            validate_interpolated_trajectory=False,
        )

        assert result.interpolated_waypoints.shape == (N, horizon, planner.action_dim)

        reshaped = result.interpolated_waypoints.view(B, S, horizon, planner.action_dim)
        assert reshaped.shape == (B, S, horizon, planner.action_dim)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_all_failed_returns_none_waypoints(self, planner):
        """When all queries fail, interpolated_waypoints is None."""
        horizon = 16
        samples = planner.sampling_strategy.generate_feasible_action_samples(100)

        x_start = samples[:2]
        x_goal = torch.full_like(x_start, 999.0)

        result = planner.find_path(
            x_start, x_goal,
            interpolate_waypoints=True,
            interpolation_steps=horizon,
            interpolation_type=TrajInterpolationType.LINEAR,
            validate_interpolated_trajectory=False,
        )

        assert not result.success.any()
        assert result.interpolated_waypoints is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_successful_queries_have_full_shape(self, planner):
        """When some queries succeed, waypoints shape is (N, H, D), not filtered."""
        N = 4
        horizon = 16
        samples = planner.sampling_strategy.generate_feasible_action_samples(100)

        x_start = samples[:N]
        x_goal = samples[50 : 50 + N]

        result = planner.find_path(
            x_start, x_goal,
            interpolate_waypoints=True,
            interpolation_steps=horizon,
            interpolation_type=TrajInterpolationType.LINEAR,
            validate_interpolated_trajectory=False,
        )

        if result.success.any():
            assert result.interpolated_waypoints.shape == (N, horizon, planner.action_dim)
            for i in range(N):
                if not result.success[i]:
                    assert (result.interpolated_waypoints[i] == 0).all()
        else:
            assert result.interpolated_waypoints is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_repeated_start_different_goals(self, planner):
        """Same start repeated B*S times with different goals (batch planner pattern)."""
        B, S = 2, 4
        N = B * S
        horizon = 16
        samples = planner.sampling_strategy.generate_feasible_action_samples(100)

        start = samples[0:1]
        x_start = start.repeat(N, 1)
        x_goal = samples[50 : 50 + N]

        result = planner.find_path(
            x_start, x_goal,
            interpolate_waypoints=True,
            interpolation_steps=horizon,
            interpolation_type=TrajInterpolationType.LINEAR,
            validate_interpolated_trajectory=False,
        )

        assert result.interpolated_waypoints.shape == (N, horizon, planner.action_dim)

        reshaped = result.interpolated_waypoints.view(B, S, horizon, planner.action_dim)
        assert reshaped.shape == (B, S, horizon, planner.action_dim)

        if result.success.any():
            success_reshaped = result.success.view(B, S)
            for b in range(B):
                for s in range(S):
                    if success_reshaped[b, s]:
                        assert not (reshaped[b, s] == 0).all()
