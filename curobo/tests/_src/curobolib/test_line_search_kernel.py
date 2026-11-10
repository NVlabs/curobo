# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the fused CUDA Wolfe line-search kernel."""

from __future__ import annotations

import pytest
import torch

from curobo._src.optim.gradient.line_search_context import LineSearchContext
from curobo._src.optim.gradient.line_search_strategy import (
    ApproxWolfeLineSearchStrategy,
    ArmijoLineSearchStrategy,
    BaseWolfeLineSearchStrategy,
    LineSearchStatus,
    LineSearchStrategy,
    StrongWolfeLineSearchStrategy,
    WolfeLineSearchStrategy,
)
from curobo._src.optim.optimization_iteration_state import OptimizationIterationState
from curobo._src.types.device_cfg import DeviceCfg


@pytest.mark.parametrize("use_cuda", [False, True])
def test_failed_search_retries_smaller_candidates(use_cuda: bool) -> None:
    """A rejected batch retains its point and retries a smaller parallel probe fan."""
    state = _make_iteration_state(2, 7)
    state.action.fill_(1.0)
    state.exploration_action.copy_(state.action)
    stiffness = torch.tensor([1.0e6, 1.0], device="cuda").view(2, 1, 1)
    state.exploration_gradient.copy_(stiffness * state.action)
    state.gradient.copy_(state.exploration_gradient)
    state.exploration_cost.copy_((0.5 * stiffness * state.action.square()).sum((1, 2)))
    state.cost.copy_(state.exploration_cost)
    state.best_cost.copy_(state.cost)
    state.best_action.copy_(state.action)
    state.step_direction.copy_(-0.001 * state.gradient)
    baseline = state.exploration_action.clone()
    original_direction = state.step_direction.clone()

    def evaluate(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (0.5 * stiffness * points.square()).sum(-1, keepdim=True), stiffness * points

    context = LineSearchContext(
        device_cfg=DeviceCfg(),
        line_search_scale=[0.01, 0.5, 1.0],
        line_search_c_1=1.0e-5,
        line_search_c_2=0.9,
        num_problems=2,
        opt_dim=7,
        action_horizon=1,
        action_dim=7,
        use_cuda_kernel_line_search=use_cuda,
        compute_costs_and_gradients=evaluate,
        line_search_backtracking=True,
        convergence_iteration=10,
        cost_delta_threshold=0.0,
        cost_relative_threshold=0.0,
    )
    strategy = ApproxWolfeLineSearchStrategy()
    strategy.update_num_problems(2, context)
    result = strategy.search(state, context)
    torch.testing.assert_close(result.exploration_action[0], baseline[0])
    torch.testing.assert_close(result.step_direction, original_direction)
    torch.testing.assert_close(context.beta, torch.tensor([0.1, 1.0], device="cuda"))
    assert not result.curvature_update_valid[0].item()
    assert result.curvature_update_valid[1].item()
    assert result.line_search_index[0].item() == -1
    # The next probe fan can now reach the stiff quadratic's minimizer.
    result = strategy.search(result, context)
    torch.testing.assert_close(result.exploration_action[0], torch.zeros_like(baseline[0]))
    assert result.curvature_update_valid[0].item()
    torch.testing.assert_close(context.beta, torch.ones_like(context.beta))


@pytest.mark.parametrize("use_cuda", [False, True])
@pytest.mark.parametrize("invalid_cost", [float("inf"), float("nan")])
def test_nonfinite_probe_fan_preserves_baseline(use_cuda: bool, invalid_cost: float) -> None:
    """Nonfinite trial costs shrink beta without changing the accepted baseline."""
    state = _make_iteration_state(2, 7)

    def evaluate(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.full_like(points[..., :1], invalid_cost), torch.ones_like(points)

    context = LineSearchContext(
        device_cfg=DeviceCfg(),
        line_search_scale=[0.01, 0.5, 1.0],
        line_search_c_1=1.0e-5,
        line_search_c_2=0.9,
        num_problems=2,
        opt_dim=7,
        action_horizon=1,
        action_dim=7,
        use_cuda_kernel_line_search=use_cuda,
        compute_costs_and_gradients=evaluate,
        line_search_backtracking=True,
        convergence_iteration=10,
        cost_delta_threshold=0.0,
        cost_relative_threshold=0.0,
    )
    strategy = ApproxWolfeLineSearchStrategy()
    strategy.update_num_problems(2, context)
    pointer = context.beta.data_ptr()
    for attempt in range(3):
        state = strategy.search(state, context)
        torch.testing.assert_close(state.exploration_action, torch.zeros_like(state.action))
        torch.testing.assert_close(state.exploration_cost, torch.full_like(state.cost, 10.0))
        torch.testing.assert_close(state.step_direction, torch.ones_like(state.step_direction))
        torch.testing.assert_close(
            context.beta, torch.full_like(context.beta, 0.1 ** (attempt + 1))
        )
        assert not state.curvature_update_valid.any().item()
    assert context.beta.data_ptr() == pointer


def _make_iteration_state(
    batch_size: int,
    opt_dim: int,
) -> OptimizationIterationState:
    """Create the mutable buffers required by the fused line-search operation."""
    device = torch.device("cuda")
    action = torch.zeros((batch_size, 1, opt_dim), device=device)
    baseline_gradient = torch.full_like(action, -1.0 / opt_dim)
    return OptimizationIterationState(
        action=action.clone(),
        cost=torch.full((batch_size,), 10.0, device=device),
        gradient=baseline_gradient.clone(),
        exploration_action=action.clone(),
        exploration_gradient=baseline_gradient.clone(),
        exploration_cost=torch.full((batch_size,), 10.0, device=device),
        step_direction=torch.ones_like(action),
        best_action=action.clone(),
        best_cost=torch.full((batch_size,), 1_000.0, device=device),
        best_iteration=torch.zeros(batch_size, dtype=torch.int32, device=device),
        current_iteration=torch.zeros(batch_size, dtype=torch.int32, device=device),
        converged=torch.zeros(batch_size, dtype=torch.uint8, device=device),
        line_search_index=torch.full((batch_size,), -1, dtype=torch.int32, device=device),
        line_search_status=torch.zeros(batch_size, dtype=torch.int32, device=device),
        curvature_update_valid=torch.zeros(batch_size, dtype=torch.bool, device=device),
    )


@pytest.mark.parametrize("use_cuda", [False, True])
@pytest.mark.parametrize(
    "strategy_type", [ApproxWolfeLineSearchStrategy, ArmijoLineSearchStrategy]
)
def test_disabled_backtracking_does_not_access_beta(
    use_cuda: bool, strategy_type: type[LineSearchStrategy]
) -> None:
    """A poisoned beta cannot affect the fixed-fan candidate or selection kernels."""
    state = _make_iteration_state(2, 7)

    def evaluate(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return 10.0 - points.mean(-1, keepdim=True), torch.full_like(points, -1.0 / 7)

    context = LineSearchContext(
        device_cfg=DeviceCfg(),
        line_search_scale=[0.1, 0.5, 1.0],
        line_search_c_1=1.0e-5,
        line_search_c_2=0.9,
        num_problems=2,
        opt_dim=7,
        action_horizon=1,
        action_dim=7,
        use_cuda_kernel_line_search=use_cuda,
        compute_costs_and_gradients=evaluate,
        convergence_iteration=10,
        cost_delta_threshold=0.0,
        cost_relative_threshold=0.0,
    )
    context.beta.fill_(float("nan"))
    strategy = strategy_type()
    strategy.update_num_problems(2, context)
    result = strategy.search(state, context)
    torch.testing.assert_close(result.exploration_action, torch.ones_like(state.action))
    assert torch.isfinite(result.exploration_cost).all().item()
    assert torch.isnan(context.beta).all().item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    "strategy_type",
    [WolfeLineSearchStrategy, StrongWolfeLineSearchStrategy],
)
@pytest.mark.parametrize(
    ("opt_dim", "n_linesearch"),
    [(7, 3), (224, 4), (1024, 4), (1025, 4), (3137, 7)],
)
def test_fused_line_search_matches_tensor_reference(
    opt_dim: int,
    n_linesearch: int,
    strategy_type: type[BaseWolfeLineSearchStrategy],
) -> None:
    """Match fused and tensor Wolfe selection above and below one CUDA block."""
    batch_size = 3
    device_cfg = DeviceCfg(device=torch.device("cuda"), dtype=torch.float32)
    line_search_scale = torch.linspace(0.01, 1.0, n_linesearch).tolist()
    context = LineSearchContext(
        device_cfg=device_cfg,
        line_search_scale=line_search_scale,
        line_search_c_1=1.0e-4,
        line_search_c_2=0.9,
        num_problems=batch_size,
        opt_dim=opt_dim,
        action_horizon=1,
        action_dim=opt_dim,
        use_cuda_kernel_line_search=True,
        compute_costs_and_gradients=lambda _: None,
        convergence_iteration=10,
        cost_delta_threshold=1.0e-6,
        cost_relative_threshold=1.0e-3,
    )

    generator = torch.Generator(device=device_cfg.device).manual_seed(1234 + opt_dim)
    search_action = torch.randn(
        (batch_size, n_linesearch, opt_dim), generator=generator, device=device_cfg.device
    )
    directional_derivatives = torch.linspace(-100.0, 20.0, n_linesearch, device=device_cfg.device)
    search_gradient = (
        directional_derivatives.view(1, n_linesearch, 1)
        .expand(batch_size, n_linesearch, opt_dim)
        .contiguous()
        / opt_dim
    )
    search_cost = (
        torch.linspace(10.0, 7.0, n_linesearch, device=device_cfg.device)
        .view(1, n_linesearch, 1)
        .expand(batch_size, n_linesearch, 1)
        .contiguous()
    )
    step_direction = torch.ones((batch_size, 1, opt_dim), device=device_cfg.device)

    cuda_strategy = strategy_type()
    cuda_strategy.update_num_problems(batch_size, context)
    scale_tensor = context.get_line_search_scale()
    cuda_state = _make_iteration_state(batch_size, opt_dim)
    cuda_result = cuda_strategy._cuda_kernel_wolfe_search(
        search_action,
        search_cost,
        search_gradient,
        step_direction,
        cuda_state,
        context,
        scale_tensor,
    )

    tensor_strategy = strategy_type()
    tensor_state = _make_iteration_state(batch_size, opt_dim)
    tensor_result = tensor_strategy._torch_wolfe_search(
        search_action,
        step_direction,
        search_cost,
        search_gradient,
        tensor_state,
        scale_tensor,
        context,
        tensor_strategy._compute_curvature_condition,
        tensor_strategy._handle_no_valid_step,
    )

    torch.testing.assert_close(cuda_result.selected_state.cost, tensor_result.selected_state.cost)
    torch.testing.assert_close(
        cuda_result.selected_state.action, tensor_result.selected_state.action
    )
    torch.testing.assert_close(
        cuda_result.selected_state.gradient, tensor_result.selected_state.gradient
    )
    torch.testing.assert_close(
        cuda_result.exploration_state.action, tensor_result.exploration_state.action
    )
    torch.testing.assert_close(
        cuda_result.exploration_state.gradient, tensor_result.exploration_state.gradient
    )
    torch.testing.assert_close(cuda_state.best_action, cuda_result.selected_state.action)
    torch.testing.assert_close(cuda_state.best_cost, cuda_result.selected_state.cost)
    assert torch.equal(cuda_state.current_iteration, torch.ones_like(cuda_state.current_iteration))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_approx_wolfe_matches_tensor_reference() -> None:
    """Match cached-baseline Wolfe, Armijo, and fixed-fallback selection."""
    batch_size = 4
    opt_dim = 224
    device = torch.device("cuda")
    device_cfg = DeviceCfg(device=device, dtype=torch.float32)
    context = LineSearchContext(
        device_cfg=device_cfg,
        line_search_scale=[0.1, 0.5, 0.7, 1.0],
        line_search_backtracking=False,
        line_search_c_1=1.0e-4,
        line_search_c_2=0.9,
        num_problems=batch_size,
        opt_dim=opt_dim,
        action_horizon=1,
        action_dim=opt_dim,
        use_cuda_kernel_line_search=True,
        compute_costs_and_gradients=lambda _: None,
        convergence_iteration=10,
        cost_delta_threshold=0.0,
        cost_relative_threshold=0.0,
    )
    scales = context.get_line_search_scale()
    costs = torch.tensor(
        [
            [9.9, 9.5, 9.4, 9.2],
            [9.99, 9.8, 10.2, 11.0],
            [11.0, 10.5, 10.2, 10.7],
            [11.0, 10.9, 10.8, 10.7],
        ],
        device=device,
    ).unsqueeze(-1)
    derivatives = torch.tensor(
        [
            [-1.0, -0.8, -0.7, -0.6],
            [-1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0],
        ],
        device=device,
    )
    gradients = derivatives.unsqueeze(-1).expand(-1, -1, opt_dim).contiguous() / opt_dim
    actions = torch.randn((batch_size, 4, opt_dim), device=device)
    direction = torch.ones((batch_size, 1, opt_dim), device=device)

    cuda_strategy = ApproxWolfeLineSearchStrategy()
    cuda_strategy.update_num_problems(batch_size, context)
    # CUDA Graph input refresh copies the initial iteration state's false value
    # into this aliased output buffer before replay.  The kernel must overwrite
    # it instead of relying on the allocation-time default.
    cuda_strategy._output_buffers[-1].fill_(False)
    cuda_state = _make_iteration_state(batch_size, opt_dim)
    cuda_result = cuda_strategy._cuda_kernel_wolfe_search(
        actions,
        costs,
        gradients,
        direction,
        cuda_state,
        context,
        scales,
    )

    tensor_strategy = ApproxWolfeLineSearchStrategy()
    tensor_state = _make_iteration_state(batch_size, opt_dim)
    tensor_result = tensor_strategy._torch_wolfe_search(
        actions,
        direction,
        costs,
        gradients,
        tensor_state,
        scales,
        context,
        tensor_strategy._compute_curvature_condition,
        tensor_strategy._handle_no_valid_step,
    )

    torch.testing.assert_close(cuda_result.selected_state.cost, tensor_result.selected_state.cost)
    torch.testing.assert_close(
        cuda_result.exploration_state.cost, tensor_result.exploration_state.cost
    )
    torch.testing.assert_close(
        cuda_result.exploration_state.action, tensor_result.exploration_state.action
    )
    assert torch.equal(cuda_result.line_search_index, tensor_result.line_search_index)
    assert torch.equal(cuda_result.line_search_status, tensor_result.line_search_status)
    assert torch.equal(
        cuda_result.line_search_status,
        torch.tensor(
            [
                LineSearchStatus.WOLFE,
                LineSearchStatus.ARMIJO,
                LineSearchStatus.FALLBACK,
                LineSearchStatus.FALLBACK,
            ],
            dtype=torch.int32,
            device=device,
        ),
    )
    assert torch.equal(
        cuda_result.curvature_update_valid,
        tensor_result.curvature_update_valid,
    )
