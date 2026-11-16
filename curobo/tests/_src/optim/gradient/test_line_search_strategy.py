# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for cached-baseline line-search selection."""

from __future__ import annotations

import pytest
import torch

from curobo._src.optim.components.debug_recorder import DebugRecorder
from curobo._src.optim.gradient.conjugate_gradient import ConjugateGradientOptCfg
from curobo._src.optim.gradient.lbfgs import LBFGSOptCfg
from curobo._src.optim.gradient.line_search_context import LineSearchContext
from curobo._src.optim.gradient.line_search_strategy import (
    GreedyLineSearchStrategy,
    LineSearchStatus,
)
from curobo._src.optim.optimization_iteration_state import OptimizationIterationState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.config_io import load_yaml
from curobo.content import get_task_configs_path


@pytest.mark.parametrize("config_type", [LBFGSOptCfg, ConjugateGradientOptCfg])
def test_backtracking_is_opt_in(config_type: type) -> None:
    """Generic optimizers leave backtracking disabled unless requested by MPC."""
    assert not config_type().line_search_backtracking
    assert config_type(line_search_backtracking=True).line_search_backtracking


@pytest.mark.parametrize("profile", ["lbfgs_mpc.yml", "lbfgs_retarget_mpc.yml"])
def test_mpc_profiles_enable_backtracking(profile: str) -> None:
    """Both shipped MPC profiles explicitly select backtracking."""
    config = load_yaml(str(get_task_configs_path() / "mpc" / profile))
    assert config["optimizer"]["line_search_backtracking"] is True


def _make_state(batch_size: int, scales: torch.Tensor) -> OptimizationIterationState:
    """Create a one-dimensional state with a valid cached baseline."""
    device = scales.device
    action = torch.zeros((batch_size, 1, 1), device=device)
    gradient = -torch.ones_like(action)
    return OptimizationIterationState(
        action=action.clone(),
        cost=torch.full((batch_size,), 10.0, device=device),
        gradient=gradient.clone(),
        exploration_action=action.clone(),
        exploration_gradient=gradient.clone(),
        exploration_cost=torch.full((batch_size,), 10.0, device=device),
        step_direction=torch.ones_like(action),
        best_action=action.clone(),
        best_cost=torch.full((batch_size,), 10.0, device=device),
        best_iteration=torch.zeros(batch_size, dtype=torch.int32, device=device),
        current_iteration=torch.zeros(batch_size, dtype=torch.int32, device=device),
        converged=torch.zeros(batch_size, dtype=torch.uint8, device=device),
    )


def _make_context(
    batch_size: int,
    compute_costs_and_gradients,
    *,
    line_search_scale: list[float] | None = None,
) -> LineSearchContext:
    """Create a line-search context for four positive probes."""
    if line_search_scale is None:
        line_search_scale = [0.1, 0.5, 0.7, 1.0]
    return LineSearchContext(
        device_cfg=DeviceCfg(device="cuda", dtype=torch.float32),
        line_search_scale=line_search_scale,
        line_search_c_1=1.0e-4,
        line_search_c_2=0.9,
        num_problems=batch_size,
        opt_dim=1,
        action_horizon=1,
        action_dim=1,
        use_cuda_kernel_line_search=False,
        line_search_backtracking=True,
        compute_costs_and_gradients=compute_costs_and_gradients,
        convergence_iteration=10,
        cost_delta_threshold=0.0,
        cost_relative_threshold=0.0,
    )


def test_lbfgs_config_rejects_nonpositive_probe() -> None:
    """Reject a zero-valued line-search probe."""
    with pytest.raises(ValueError):
        LBFGSOptCfg(line_search_scale=[0.0, 0.1, 0.5, 1.0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_debug_trace_records_line_search_outcomes() -> None:
    """Expose the per-iteration fields needed for acceptance-rate analysis."""
    device = torch.device("cuda")
    state = _make_state(
        1,
        torch.tensor([[0.1, 0.5, 0.7, 1.0]], device=device),
    )
    state.line_search_index = torch.tensor([2], dtype=torch.int32, device=device)
    state.line_search_status = torch.tensor(
        [LineSearchStatus.GREEDY], dtype=torch.int32, device=device
    )
    state.curvature_update_valid = torch.tensor([False], device=device)
    recorder = DebugRecorder()

    recorder.record(state, action_horizon=1, action_dim=1)
    trace = recorder.get_trace()

    assert trace["debug_line_search_index"][0].item() == 2
    assert trace["debug_line_search_status"][0].item() == LineSearchStatus.GREEDY
    assert not trace["debug_curvature_update_valid"][0].item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_greedy_reuses_baseline_without_evaluating_zero_probe() -> None:
    """Keep the cached point when every positive Greedy probe is worse."""
    device = torch.device("cuda")
    scales = torch.tensor([[0.1, 0.5, 0.7, 1.0]], device=device)
    evaluated_points = None

    def evaluate(x_set: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal evaluated_points
        evaluated_points = x_set.detach().clone()
        return 10.0 + x_set.square(), 2.0 * x_set

    state = _make_state(1, scales)
    result = GreedyLineSearchStrategy().search(
        state,
        _make_context(1, evaluate),
    )

    assert evaluated_points is not None
    assert torch.all(evaluated_points > 0.0)
    assert result.line_search_index.item() == -1
    assert result.line_search_status.item() == LineSearchStatus.BACKTRACK
    torch.testing.assert_close(result.exploration_action, state.exploration_action)
    torch.testing.assert_close(result.exploration_gradient, state.exploration_gradient)
    torch.testing.assert_close(result.exploration_cost, state.exploration_cost)
