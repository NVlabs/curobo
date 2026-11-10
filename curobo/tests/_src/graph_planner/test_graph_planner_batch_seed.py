# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for using graph planner output as batch trajectory seeds.

Verifies that ``find_path`` with ``B*S`` flat queries produces
``interpolated_waypoints`` of shape ``(B*S, horizon, dof)`` that can be
reshaped to ``(B, S, horizon, dof)`` for the trajectory optimization
seed manager.
"""

from unittest.mock import patch

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

    @pytest.mark.parametrize("invalid_index", [0, 1, 2])
    @pytest.mark.parametrize("invalid_endpoint", ["start", "goal", "both"])
    @pytest.mark.parametrize("invalid_kind", ["joint_limit", "collision"])
    @pytest.mark.parametrize("interpolate", [False, True])
    def test_mixed_queries_build_shared_roadmap(
        self,
        planner: PRMGraphPlanner,
        invalid_index: int,
        invalid_endpoint: str,
        invalid_kind: str,
        interpolate: bool,
    ) -> None:
        """Valid pairs share a roadmap and retain their original result positions."""
        planner.reset_buffer()
        starts = planner.default_joint_state.position.reshape(1, -1).repeat(3, 1)
        starts[:, 0] += starts.new_tensor([-0.12, 0.0, 0.12])
        goals = starts.clone()
        goals[:, 1] += starts.new_tensor([0.08, 0.10, 0.12])
        expected_lengths = planner.distance_calculator.calculate_weighted_distance(starts, goals)
        assert (expected_lengths > planner.config.cspace_similarity_threshold).all()
        alpha = torch.linspace(0, 1, 32, device=starts.device).view(1, -1, 1)
        paths = starts[:, None] + alpha * (goals - starts)[:, None]
        assert planner.check_samples_feasibility(paths.reshape(-1, planner.action_dim)).all()

        invalid_state = torch.full_like(starts[0], 999.0)
        if invalid_kind == "collision":
            invalid_state = starts.new_tensor([2.8, 1.3, 0.0, -2.0, 1.4, 3.1, -0.8])
            assert (invalid_state >= planner.action_bound_lows).all()
            assert (invalid_state <= planner.action_bound_highs).all()
        assert not planner.check_samples_feasibility(invalid_state.unsqueeze(0)).any()
        if invalid_endpoint in ("start", "both"):
            starts[invalid_index] = invalid_state
        if invalid_endpoint in ("goal", "both"):
            goals[invalid_index] = invalid_state
        expected_success = torch.ones(3, device=starts.device, dtype=torch.bool)
        expected_success[invalid_index] = False

        with patch.object(
            planner.graph_constructor,
            "initialize_terminal_graph_connections",
            wraps=planner.graph_constructor.initialize_terminal_graph_connections,
        ) as insert_terminals:
            result = planner.find_path(
                starts,
                goals,
                interpolate_waypoints=interpolate,
                interpolation_steps=32,
                validate_interpolated_trajectory=True,
            )

        insert_terminals.assert_called_once()
        inserted_starts, inserted_goals = insert_terminals.call_args.args[:2]
        torch.testing.assert_close(inserted_starts, starts[expected_success])
        torch.testing.assert_close(inserted_goals, goals[expected_success])
        nodes = planner.node_manager.valid_node_buffer[:, : planner.action_dim]
        for endpoint in torch.cat((inserted_starts, inserted_goals)):
            assert torch.isclose(nodes, endpoint).all(dim=-1).any()

        torch.testing.assert_close(result.success, expected_success)
        torch.testing.assert_close(
            result.path_length[expected_success], expected_lengths[expected_success]
        )
        assert torch.isinf(result.path_length[invalid_index])
        assert result.plan_waypoints[invalid_index] is None
        assert not result.valid_query
        assert str(invalid_index) in result.debug_info
        for index in torch.where(expected_success)[0].tolist():
            path = result.plan_waypoints[index].reshape(-1, planner.action_dim)
            torch.testing.assert_close(path[0], starts[index])
            torch.testing.assert_close(path[-1], goals[index])
            assert planner.check_samples_feasibility(path).all()

        if interpolate:
            assert result.interpolated_waypoints.shape == (3, 32, planner.action_dim)
            torch.testing.assert_close(
                result.interpolated_waypoints[expected_success, 0], starts[expected_success]
            )
            torch.testing.assert_close(
                result.interpolated_waypoints[expected_success, -1], goals[expected_success]
            )
            assert (result.interpolated_waypoints[invalid_index] == 0).all()
        else:
            assert result.interpolated_waypoints is None

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
    def test_infeasible_query_does_not_reject_feasible_batch_query(self, planner):
        """An infeasible endpoint only rejects its own batch query."""
        horizon = 16
        feasible_state = planner.sampling_strategy.generate_feasible_action_samples(1)
        x_start = feasible_state.repeat(2, 1)
        x_goal = torch.cat((feasible_state, torch.full_like(feasible_state, 999.0)))

        result = planner.find_path(
            x_start,
            x_goal,
            interpolate_waypoints=True,
            interpolation_steps=horizon,
            interpolation_type=TrajInterpolationType.LINEAR,
            validate_interpolated_trajectory=False,
        )

        assert result.success.tolist() == [True, False]
        assert result.interpolated_waypoints.shape == (2, horizon, planner.action_dim)
        assert not (result.interpolated_waypoints[0] == 0).all()
        assert (result.interpolated_waypoints[1] == 0).all()

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
