#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for the fused CUDA L-BFGS optimizer."""

from __future__ import annotations

# Standard Library
from dataclasses import replace
from typing import Any

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.curobolib.backends.cuda_core_backend.optimization_config import (
    LBFGSCompactLaunchCfg,
)
from curobo._src.curobolib.cuda_ops.optimization import LBFGSCompactCu, LBFGScu
from curobo._src.optim.components.action_bounds import ActionBounds
from curobo._src.optim.components.quasi_newton_buffers import QuasiNewtonBuffers
from curobo._src.optim.gradient.lbfgs import LBFGSOpt, LBFGSOptCfg
from curobo._src.rollout.metrics import (
    CostCollection,
    CostsAndConstraints,
    RolloutMetrics,
    RolloutResult,
)
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.sampling.sample_buffer import SampleBuffer


@pytest.mark.parametrize("compact", [False, True])
def test_backtracking_preserves_direction_and_history_in_graph(compact: bool) -> None:
    """Rejected problems retain every quasi-Newton buffer during graph replay."""
    device_cfg = DeviceCfg()
    batch, dimension, history = 2, 64, 16
    properties = torch.cuda.get_device_properties(device_cfg.device)
    layout = LBFGSCompactLaunchCfg.calculate_config(
        batch, dimension, history, properties.shared_memory_per_block_optin
    )
    assert layout is not None
    _, shared_matrices, workspace = layout
    buffers = QuasiNewtonBuffers(device_cfg, history)
    buffers.resize(batch, dimension, compact, workspace)
    step = torch.full((batch, dimension), 123.0, device="cuda")
    q = torch.full_like(step, 0.02)
    gradient = 2.0 * q.unsqueeze(1)
    status = torch.tensor([5, 1], device="cuda", dtype=torch.int32)
    allowed = torch.tensor([False, True], device="cuda")
    step_max = torch.ones(dimension, device="cuda")

    def solve() -> None:
        if compact:
            LBFGSCompactCu.apply(
                step, buffers.rho, buffers.y, buffers.s, buffers.compact_inverse_r,
                buffers.compact_matrix_workspace, q, gradient, buffers.x_0,
                buffers.grad_0, buffers.history_head, buffers.history_count,
                step_max, 0.01, shared_matrices, True, allowed, True, status,
            )
        else:
            LBFGScu.apply(
                step, buffers.rho, buffers.y, buffers.s, q, gradient, buffers.x_0,
                buffers.grad_0, buffers.history_head, buffers.history_count,
                0.01, True, True, True, step_max, True, allowed, status,
            )

    solve()  # Compile before capture.
    buffers.clear()
    buffers.x_0.zero_()
    buffers.grad_0.zero_()
    step.fill_(123.0)
    preserved = {name: getattr(buffers, name).clone() for name in (
        "rho", "s", "y", "x_0", "grad_0", "history_head", "history_count"
    )}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        solve()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(step[0], torch.full_like(step[0], 123.0), rtol=0, atol=0)
    for name, original in preserved.items():
        actual = getattr(buffers, name)
        if name in ("rho", "s", "y"):
            torch.testing.assert_close(actual[:, 0], original[:, 0], rtol=0, atol=0)
        else:
            torch.testing.assert_close(actual[0], original[0], rtol=0, atol=0)
    assert buffers.history_count[1].item() == 1
    torch.testing.assert_close(step[1], -q[1], rtol=2.e-3, atol=1.e-5)
    if compact:
        torch.testing.assert_close(buffers.compact_inverse_r[0], torch.eye(history, device="cuda"))


def pytest_generate_tests(metafunc):
    """Generate test parameters dynamically based on CLI options.

    This enables different parametrization strategies:
    - Default mode (pytest): history=[1, 10, 31, 64, 128].
    - Full mode (pytest --full-params): history=[1..128].

    Args:
        metafunc: pytest metafunc object for the current test.
    """
    if "lbfgs_history_range" in metafunc.fixturenames:
        # Check if --full-params flag is set
        use_full_params = metafunc.config.getoption("--full-params", default=False)

        if use_full_params:
            # Full supported range.
            history_values = list(range(1, 129))
        else:
            # Quick range: representative subset for fast CI/default runs
            history_values = [1, 10, 31, 64, 128]

        metafunc.parametrize("lbfgs_history_range", history_values)


def cost_fn(state):
    costs = torch.sum((10.0 - state) ** 2, dim=-1).unsqueeze(-1)
    return costs


def _reference_update_buffers(
    q: torch.Tensor,
    grad_q: torch.Tensor,
    s_buffer: torch.Tensor,
    y_buffer: torch.Tensor,
    rho_buffer: torch.Tensor,
    x_0: torch.Tensor,
    grad_0: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Append only valid curvature pairs to the test-only PyTorch history."""
    gradient = grad_q.transpose(-1, -2)
    position = q.unsqueeze(-1)
    y = gradient - grad_0
    s = position - x_0
    curvature = y.transpose(-1, -2) @ s
    valid = torch.isfinite(curvature) & (curvature > 0.0)
    rho = torch.where(valid, 1.0 / curvature, torch.zeros_like(curvature))

    valid_history = valid.unsqueeze(0)
    s_buffer.copy_(torch.where(valid_history, torch.roll(s_buffer, -1, dims=0), s_buffer))
    y_buffer.copy_(torch.where(valid_history, torch.roll(y_buffer, -1, dims=0), y_buffer))
    rho_buffer.copy_(torch.where(valid_history, torch.roll(rho_buffer, -1, dims=0), rho_buffer))
    s_buffer[-1].copy_(torch.where(valid, s, s_buffer[-1]))
    y_buffer[-1].copy_(torch.where(valid, y, y_buffer[-1]))
    rho_buffer[-1].copy_(torch.where(valid, rho, rho_buffer[-1]))
    x_0.copy_(position)
    grad_0.copy_(gradient)
    return s_buffer, y_buffer, rho_buffer, x_0, grad_0


def _reference_step_direction(
    rho_buffer: torch.Tensor,
    y_buffer: torch.Tensor,
    s_buffer: torch.Tensor,
    grad_q: torch.Tensor,
    history: int,
    epsilon: float,
    use_geometric_scaling: bool = True,
) -> torch.Tensor:
    """Compute a test-only PyTorch L-BFGS two-loop reference."""
    gradient = grad_q.transpose(-1, -2)
    work = gradient.detach().clone()
    alpha = torch.zeros_like(rho_buffer)
    rho_s = rho_buffer * s_buffer.transpose(-1, -2)
    for index in range(history - 1, -1, -1):
        alpha[index] = rho_s[index] @ work
        work = work - alpha[index] * y_buffer[index]

    sty = s_buffer[-1].transpose(-1, -2) @ y_buffer[-1]
    yty = y_buffer[-1].transpose(-1, -2) @ y_buffer[-1]
    if use_geometric_scaling:
        sts = s_buffer[-1].transpose(-1, -2) @ s_buffer[-1]
        candidate_scale = torch.sqrt(sts / yty)
    else:
        candidate_scale = sty / yty
    valid_scale = (
        torch.isfinite(candidate_scale)
        & (candidate_scale > 0.0)
        & torch.isfinite(rho_buffer[-1])
        & (rho_buffer[-1] > 0.0)
    )
    gamma = torch.where(valid_scale, candidate_scale, torch.full_like(candidate_scale, epsilon))
    result = gamma * work
    rho_y = rho_buffer * y_buffer.transpose(-1, -2)
    for index in range(history):
        result = result + (alpha[index] - rho_y[index] @ result) * s_buffer[index]
    return -result


class MockRollout:
    """Standalone mock rollout (no base class). Satisfies the Rollout protocol."""

    def __init__(
        self,
        num_dof: int = 7,
        action_horizon: int = 10,
        batch_size: int = 10,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.device_cfg = DeviceCfg(device=device, dtype=dtype)
        self.sum_horizon = True
        self.sampler_seed = 1
        self._action_dim = num_dof
        self._action_horizon = action_horizon
        self._batch_size = None
        self._tensor_args = self.device_cfg
        self._action_bound_lows = torch.ones(num_dof, device=device, dtype=dtype) * -1.0
        self._action_bound_highs = torch.ones(num_dof, device=device, dtype=dtype) * 1.0
        self.start_state = None
        self.act_sample_gen = SampleBuffer.create_halton_sample_buffer(
            ndims=num_dof,
            device_cfg=self.device_cfg,
            up_bounds=self._action_bound_highs,
            low_bounds=self._action_bound_lows,
            seed=self.sampler_seed,
        )

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def action_horizon(self) -> int:
        return self._action_horizon

    @property
    def action_bound_lows(self) -> torch.Tensor:
        return self._action_bound_lows

    @property
    def action_bound_highs(self) -> torch.Tensor:
        return self._action_bound_highs

    @property
    def action_bounds(self) -> torch.Tensor:
        return torch.stack([self._action_bound_lows, self._action_bound_highs])

    @property
    def state_bounds(self) -> torch.Tensor:
        return self.action_bounds

    @property
    def horizon(self) -> int:
        return self._action_horizon

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @property
    def dt(self) -> float:
        return 1.0

    def evaluate_action(self, act_seq, **kwargs):
        batch_size = act_seq.shape[0]
        if batch_size != self._batch_size:
            self.update_batch_size(batch_size)
        state = self._compute_state_from_action_impl(act_seq)
        cc = self._compute_costs_and_constraints_impl(state, **kwargs)
        return RolloutResult(actions=act_seq, state=state, costs_and_constraints=cc)

    def compute_metrics_from_state(self, state, **kwargs):
        cc = self._compute_costs_and_constraints_impl(state, **kwargs)
        convergence = cc.get_sum_cost(sum_horizon=False)
        return RolloutMetrics(
            costs_and_constraints=cc,
            feasible=cc.get_feasible(),
            state=state,
            convergence=convergence,
        )

    def compute_metrics_from_action(self, act_seq, **kwargs):
        batch_size = act_seq.shape[0]
        self.update_batch_size(batch_size)
        state = self._compute_state_from_action_impl(act_seq)
        metrics = self.compute_metrics_from_state(state, **kwargs)
        metrics.actions = act_seq
        return metrics

    def update_batch_size(self, batch_size):
        if self._batch_size is None or self._batch_size != batch_size:
            self._batch_size = batch_size

    def update_params(self, **kwargs):
        return True

    def update_dt(self, dt, **kwargs):
        return True

    def reset(self, reset_problem_ids=None, **kwargs):
        return True

    def reset_shape(self):
        return True

    def reset_seed(self):
        self.act_sample_gen.reset()

    def reset_cuda_graph(self):
        return self.reset_shape()

    def sample_random_actions(self, n=0, bounded=True):
        return self.act_sample_gen.get_samples(n, bounded=bounded)

    def get_initial_action(self, use_random=True, use_zero=False, **kwargs):
        num_samples = self._batch_size or 1
        if use_random:
            n_samples = num_samples * self._action_horizon
            init_action = self.sample_random_actions(n=n_samples, bounded=True)
        elif use_zero:
            init_action = torch.zeros(
                (num_samples, self._action_horizon, self._action_dim),
                **self.device_cfg.as_torch_dict(),
            )
        else:
            init_action = torch.zeros(
                (num_samples, self._action_horizon, self._action_dim),
                **self.device_cfg.as_torch_dict(),
            )
        return init_action.view(num_samples, self._action_horizon, self._action_dim)

    def get_all_cost_components(self):
        return {}

    def filter_robot_state(self, state):
        return state

    def _compute_state_from_action_impl(self, act_seq):
        return act_seq

    def _compute_costs_and_constraints_impl(self, state, **kwargs):
        costs = cost_fn(state)
        costs = CostCollection(values=[costs], names=["cost"], weights=[1.0], sq_weights=[1.0])
        return CostsAndConstraints(costs=costs)


@pytest.fixture
def optimizer_setup():
    """Set up environment for LBFGS optimization tests.

    Returns:
        Dictionary containing test setup parameters and objects.
    """
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    # Define test parameters
    num_dof = 7
    action_horizon = 28
    batch_size = 4

    # Create rollout function
    rollout_fn = MockRollout(
        num_dof=num_dof,
        action_horizon=action_horizon,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )

    return {
        "device": device,
        "dtype": dtype,
        "num_dof": num_dof,
        "action_horizon": action_horizon,
        "batch_size": batch_size,
        "rollout_fn": rollout_fn,
        "step_scale": 0.98,
        "return_best_action": True,
    }


def create_optimizer(
    rollout_fn,
    batch_size,
    use_cuda_kernel_line_search=True,
    history=27,
    num_iters=50,
):
    """Create an LBFGS optimizer with the specified configuration.

    Args:
        rollout_fn: Rollout function for optimization.
        batch_size: Batch size for optimization.
        use_cuda_kernel_line_search: Whether to use the fused line-search kernel.
        history: Number of L-BFGS history entries.
        num_iters: Number of optimizer iterations.

    Returns:
        LBFGSOpt: Configured optimizer.
    """
    # Create optimizer configuration
    config = LBFGSOptCfg(
        num_problems=batch_size,
        num_iters=num_iters,
        history=history,
        step_scale=0.98,
        use_cuda_kernel_shared_buffers=True,
        use_cuda_kernel_line_search=use_cuda_kernel_line_search,
        stable_mode=True,
        solver_type="lbfgs",
        line_search_scale=[0.01, 0.1, 0.5, 1.0],
    )

    # Create optimizer with 2 rollout instances (default for LBFGS)
    optimizer = LBFGSOpt(config, [rollout_fn, rollout_fn])
    optimizer.update_num_problems(batch_size)
    return optimizer


def test_optimize_without_timing_matches_timed_solve(optimizer_setup):
    """The capture-safe entry point must preserve normal L-BFGS semantics."""
    timed_optimizer = create_optimizer(
        optimizer_setup["rollout_fn"],
        optimizer_setup["batch_size"],
        history=3,
        num_iters=10,
    )
    untimed_optimizer = create_optimizer(
        optimizer_setup["rollout_fn"],
        optimizer_setup["batch_size"],
        history=3,
        num_iters=10,
    )
    initial_action = torch.linspace(
        -2.0,
        2.0,
        steps=(
            optimizer_setup["batch_size"]
            * optimizer_setup["action_horizon"]
            * optimizer_setup["num_dof"]
        ),
        device=optimizer_setup["device"],
        dtype=optimizer_setup["dtype"],
    ).view(
        optimizer_setup["batch_size"],
        optimizer_setup["action_horizon"],
        optimizer_setup["num_dof"],
    )
    timed_optimizer.reinitialize(initial_action)
    untimed_optimizer.reinitialize(initial_action)

    timed_result = timed_optimizer.optimize(initial_action)
    untimed_result = untimed_optimizer.optimize_without_timing(initial_action)

    torch.testing.assert_close(untimed_result, timed_result)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_50_then_25_iterations_reuse_cuda_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replay one 25-step graph twice initially and once while retaining finetune history."""
    config = LBFGSOptCfg(
        num_problems=2, num_iters=50, inner_iters=25, history=7,
        fixed_iters=True, step_scale=0.05,
    )
    graph_rollout = MockRollout(num_dof=7, action_horizon=4, batch_size=2)
    eager_rollout = MockRollout(num_dof=7, action_horizon=4, batch_size=2)
    optimizer = LBFGSOpt(config, [graph_rollout, graph_rollout], use_cuda_graph=True)
    eager = LBFGSOpt(replace(config), [eager_rollout, eager_rollout], use_cuda_graph=False)
    seed = torch.linspace(-0.3, 0.3, 56, device="cuda", dtype=torch.float32).view(2, 4, 7)
    optimizer.reinitialize(seed)
    optimizer.optimize(seed)
    executor = optimizer._core._executors["_opt_iters"]
    graph = executor._graph
    assert graph is not None
    original_replay = executor.replay
    replay_count = 0

    def count_replay(*args: Any, **kwargs: Any) -> Any:
        nonlocal replay_count
        replay_count += 1
        return original_replay(*args, **kwargs)

    monkeypatch.setattr(executor, "replay", count_replay)
    graph_seed, eager_seed = seed.clone(), seed.clone()
    for iterations, retain_history, expected_replays in (
        (50, False, 2), (25, True, 1), (50, False, 2)
    ):
        if not retain_history:
            graph_seed, eager_seed = seed.clone(), seed.clone()
        optimizer.update_niters(iterations)
        eager.update_niters(iterations)
        optimizer.reinitialize(graph_seed, clear_optimizer_state=not retain_history)
        eager.reinitialize(eager_seed, clear_optimizer_state=not retain_history)
        before = replay_count
        graph_seed = optimizer.optimize(graph_seed).clone()
        eager_seed = eager.optimize(eager_seed).clone()
        assert replay_count - before == expected_replays
        assert optimizer._core._executors["_opt_iters"] is executor
        assert executor._graph is graph
        assert optimizer.config.inner_iters == 25
        torch.testing.assert_close(graph_seed, eager_seed, rtol=3e-4, atol=1e-5)


@pytest.mark.parametrize("use_cuda_kernel_line_search", [True, False])
def test_lbfgs_optimization_cuda_kernel(
    optimizer_setup,
    lbfgs_history_range,
    use_cuda_kernel_line_search,
):
    """Test that LBFGS optimizer can minimize a simple quadratic function.

    Tests with different history sizes:
    - Quick mode tests history=[1, 10, 31, 64, 128].
    - Full mode tests every supported history from 1 through 128.

    Args:
        optimizer_setup: pytest fixture providing test setup.
        lbfgs_history_range: History size for LBFGS (dynamic based on --full-params).
        use_cuda_kernel_line_search: Whether to use CUDA kernel for line search.
    """
    history = lbfgs_history_range
    initial_action = None
    for i in range(2):
        optimizer_with_cuda = create_optimizer(
            optimizer_setup["rollout_fn"],
            optimizer_setup["batch_size"],
            use_cuda_kernel_line_search=use_cuda_kernel_line_search,
            history=history,
            num_iters=10,
        )

        # Create a random starting point
        if initial_action is None:
            initial_action = (
                torch.randn(
                    optimizer_setup["batch_size"],
                    optimizer_setup["action_horizon"],
                    optimizer_setup["num_dof"],
                    device=optimizer_setup["device"],
                    dtype=optimizer_setup["dtype"],
                )
                * 10.0
            )

        # Create copies of the initial action for both optimizers
        action_cuda = initial_action.clone()

        # Run optimization with CUDA kernel
        optimizer_with_cuda.reinitialize(action_cuda)
        result_cuda = optimizer_with_cuda.optimize(action_cuda).clone()

        # Check that both optimizers reduced the cost
        initial_cost = cost_fn(initial_action).mean().item()

        final_cost_cuda = cost_fn(result_cuda).mean().item()
        print(f"Iteration {i}")
        print(f"Initial cost: {initial_cost}")
        print(f"Final cost (CUDA kernel): {final_cost_cuda}")

        assert final_cost_cuda < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("opt_dim", "batch_size"),
    [(7, 17), (31, 2), (1024, 2), (1025, 2), (1568, 2), (4097, 2), (8193, 2)],
)
def test_lbfgs_step_matches_tensor_reference(opt_dim: int, batch_size: int):
    """Match the tensor recursion across warp, block, register, and workspace paths."""
    torch.manual_seed(1000 + opt_dim)
    history = 7
    device = torch.device("cuda")

    step = torch.zeros((batch_size, opt_dim), device=device)
    rho = torch.zeros((history, batch_size, 1, 1), device=device)
    y = torch.zeros((history, batch_size, opt_dim, 1), device=device)
    s = torch.zeros_like(y)
    x_0 = torch.zeros((batch_size, opt_dim, 1), device=device)
    grad_0 = torch.zeros_like(x_0)
    history_head = torch.zeros((batch_size,), dtype=torch.int32, device=device)
    history_count = torch.zeros_like(history_head)

    ref_rho = torch.zeros_like(rho)
    ref_y = torch.zeros_like(y)
    ref_s = torch.zeros_like(s)
    ref_x_0 = torch.zeros_like(x_0)
    ref_grad_0 = torch.zeros_like(grad_0)

    weights = torch.linspace(0.5, 1.5, opt_dim, device=device).view(1, 1, opt_dim)
    q = 0.1 * torch.randn((batch_size, opt_dim), device=device)
    for _ in range(history + 2):
        q.add_(0.01 * torch.randn_like(q))
        gradient = weights * q.unsqueeze(1)
        cuda_step = LBFGScu.apply(
            step,
            rho,
            y,
            s,
            q,
            gradient,
            x_0,
            grad_0,
            history_head,
            history_count,
            0.01,
            True,
            True,
        ).clone()
        ref_s, ref_y, ref_rho, ref_x_0, ref_grad_0 = _reference_update_buffers(
            q,
            gradient,
            ref_s,
            ref_y,
            ref_rho,
            ref_x_0,
            ref_grad_0,
        )
        reference_step = _reference_step_direction(
            ref_rho,
            ref_y,
            ref_s,
            gradient,
            history,
            0.01,
        ).squeeze(-1)
        torch.testing.assert_close(cuda_step, reference_step, atol=2.0e-6, rtol=2.0e-5)

    assert torch.all(history_count == history)
    assert torch.all(history_head == 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_curvature_mask_advances_reference_without_inserting_pair() -> None:
    """Advance every reference point while gating individual history updates."""
    device = torch.device("cuda")
    history = 7
    batch_size = 2
    opt_dim = 8
    position = torch.ones((batch_size, opt_dim), device=device)
    gradient = torch.ones((batch_size, 1, opt_dim), device=device)
    reference_position = torch.zeros((batch_size, opt_dim, 1), device=device)
    reference_gradient = torch.zeros_like(reference_position)
    history_head = torch.zeros(batch_size, dtype=torch.int32, device=device)
    history_count = torch.zeros_like(history_head)
    curvature_update_valid = torch.tensor([True, False], dtype=torch.bool, device=device)

    LBFGScu.apply(
        torch.zeros_like(position),
        torch.zeros((history, batch_size, 1, 1), device=device),
        torch.zeros((history, batch_size, opt_dim, 1), device=device),
        torch.zeros((history, batch_size, opt_dim, 1), device=device),
        position,
        gradient,
        reference_position,
        reference_gradient,
        history_head,
        history_count,
        0.01,
        True,
        True,
        True,
        torch.ones(opt_dim, device=device),
        False,
        curvature_update_valid,
    )

    assert torch.equal(history_count, torch.tensor([1, 0], dtype=torch.int32, device=device))
    assert torch.equal(history_head, torch.tensor([1, 0], dtype=torch.int32, device=device))
    torch.testing.assert_close(reference_position, position.unsqueeze(-1))
    torch.testing.assert_close(reference_gradient, gradient.transpose(-1, -2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("use_shared_buffers", [False, True])
def test_lbfgs_ignores_inactive_per_problem_history(use_shared_buffers: bool) -> None:
    """Ignore stale circular slots after independent problem resets."""
    torch.manual_seed(1701)
    device = torch.device("cuda")
    history = 7
    batch_size = 4
    opt_dim = 16
    history_count = torch.tensor([0, 1, 3, 7], dtype=torch.int32, device=device)
    history_head = torch.tensor([0, 4, 1, 3], dtype=torch.int32, device=device)

    clean_s = torch.zeros((history, batch_size, opt_dim, 1), device=device)
    clean_y = torch.zeros_like(clean_s)
    clean_rho = torch.zeros((history, batch_size, 1, 1), device=device)
    stale_s = torch.randn_like(clean_s)
    stale_y = stale_s * 1.25
    stale_rho = 1.0 / torch.sum(stale_s * stale_y, dim=-2, keepdim=True)
    for problem in range(batch_size):
        count = int(history_count[problem].item())
        head = int(history_head[problem].item())
        for active_offset in range(count):
            slot = (head - count + active_offset) % history
            value = torch.randn((opt_dim, 1), device=device)
            gradient_delta = value * (0.75 + 0.1 * active_offset)
            reciprocal_curvature = 1.0 / torch.sum(value * gradient_delta)
            clean_s[slot, problem].copy_(value)
            clean_y[slot, problem].copy_(gradient_delta)
            clean_rho[slot, problem].copy_(reciprocal_curvature)
            stale_s[slot, problem].copy_(value)
            stale_y[slot, problem].copy_(gradient_delta)
            stale_rho[slot, problem].copy_(reciprocal_curvature)

    position = torch.randn((batch_size, opt_dim), device=device)
    gradient = torch.randn((batch_size, 1, opt_dim), device=device)
    step_max = torch.ones(opt_dim, device=device)
    curvature_update_valid = torch.ones(batch_size, dtype=torch.bool, device=device)

    def run_step(
        rho: torch.Tensor,
        y: torch.Tensor,
        s: torch.Tensor,
    ) -> torch.Tensor:
        return LBFGScu.apply(
            torch.zeros_like(position),
            rho,
            y,
            s,
            position,
            gradient,
            position.unsqueeze(-1).clone(),
            gradient.transpose(-1, -2).clone(),
            history_head.clone(),
            history_count.clone(),
            0.01,
            True,
            use_shared_buffers,
            True,
            step_max,
            False,
            curvature_update_valid,
        ).clone()

    expected = run_step(clean_rho, clean_y, clean_s)
    actual = run_step(stale_rho, stale_y, stale_s)
    torch.testing.assert_close(actual, expected, atol=2.0e-6, rtol=2.0e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compact_pair_eligibility_and_fallback_scale_guard() -> None:
    """Keep circular eligibility except for extreme fallback curvature scales."""
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device)[0] < 8:
        pytest.skip("Compact L-BFGS requires TF32 tensor cores")

    history = 32
    batch_size = 4
    opt_dim = 112
    properties = torch.cuda.get_device_properties(device)
    shared_memory = int(
        getattr(properties, "shared_memory_per_block_optin", 0)
        or properties.shared_memory_per_block
    )
    compact_config = LBFGSCompactLaunchCfg.calculate_config(
        batch_size,
        opt_dim,
        history,
        shared_memory,
    )
    assert compact_config is not None
    _, shared_matrix_count, workspace_elements = compact_config
    position = torch.ones((batch_size, opt_dim), device=device)
    gradient = torch.ones((batch_size, 1, opt_dim), device=device)
    position[2].zero_()
    position[2, 0] = 1.0
    gradient[2].zero_()
    gradient[2, 0, 0] = 1.0e-3
    gradient[2, 0, 1] = 1.0
    position[3].fill_(1.0e-3)
    gradient[3].fill_(1.0e-7)
    reference_position = torch.zeros((batch_size, opt_dim, 1), device=device)
    reference_gradient = torch.zeros_like(reference_position)
    history_head = torch.zeros(batch_size, dtype=torch.int32, device=device)
    history_count = torch.zeros_like(history_head)
    history_head[3] = 1
    history_count[3] = 1
    curvature_update_valid = torch.ones(batch_size, dtype=torch.bool, device=device)
    line_search_status = torch.tensor([1, 4, 1, 4], dtype=torch.int32, device=device)
    rho_buffer = torch.ones((history, batch_size, 1, 1), device=device)
    rho_buffer[0, 3] = 1.0 / opt_dim
    y_buffer = torch.zeros((history, batch_size, opt_dim, 1), device=device)
    s_buffer = torch.zeros_like(y_buffer)
    y_buffer[0, 3] = 1.0
    s_buffer[0, 3] = 1.0
    inverse_r = torch.eye(history, device=device).expand(batch_size, -1, -1).clone()
    inverse_r[3, 0, 0] = 1.0 / opt_dim
    workspace_shape = (batch_size, workspace_elements) if workspace_elements > 0 else (1,)

    step = LBFGSCompactCu.apply(
        torch.zeros_like(position),
        rho_buffer,
        y_buffer,
        s_buffer,
        inverse_r,
        torch.empty(workspace_shape, device=device),
        position,
        gradient,
        reference_position,
        reference_gradient,
        history_head,
        history_count,
        torch.ones(opt_dim, device=device),
        0.01,
        shared_matrix_count,
        False,
        curvature_update_valid,
        True,
        line_search_status,
    ).clone()

    assert torch.equal(
        history_count,
        torch.tensor([1, 1, 1, 1], dtype=torch.int32, device=device),
    )
    assert torch.equal(
        history_head,
        torch.tensor([1, 1, 1, 1], dtype=torch.int32, device=device),
    )
    torch.testing.assert_close(step[1], -gradient[1, 0])
    torch.testing.assert_close(reference_position, position.unsqueeze(-1))
    torch.testing.assert_close(reference_gradient, gradient.transpose(-1, -2))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("history", "opt_dim"),
    [
        (1, 7),
        (7, 7),
        (15, 65),
        (16, 8),
        (16, 63),
        (16, 64),
        (16, 264),
        (28, 112),
        (32, 8),
        (32, 64),
        (127, 48),
        (128, 48),
    ],
)
@pytest.mark.parametrize("use_geometric_scaling", [True, False])
def test_compact_step_matches_two_loop_during_history_startup(
    history: int,
    opt_dim: int,
    use_geometric_scaling: bool,
):
    """Match compact and two-loop directions while history fills and wraps."""
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device)[0] < 8:
        pytest.skip("Compact L-BFGS requires TF32 tensor cores")

    torch.manual_seed(9012)
    batch_size = 4
    properties = torch.cuda.get_device_properties(device)
    shared_memory = int(
        getattr(properties, "shared_memory_per_block_optin", 0)
        or properties.shared_memory_per_block
    )
    compact_config = LBFGSCompactLaunchCfg.calculate_config(
        batch_size,
        opt_dim,
        history,
        shared_memory,
    )
    assert compact_config is not None
    _, shared_matrix_count, workspace_elements = compact_config

    two_loop_step = torch.zeros((batch_size, opt_dim), device=device)
    two_loop_rho = torch.zeros((history, batch_size, 1, 1), device=device)
    two_loop_y = torch.zeros((history, batch_size, opt_dim, 1), device=device)
    two_loop_s = torch.zeros_like(two_loop_y)
    two_loop_x = torch.zeros((batch_size, opt_dim, 1), device=device)
    two_loop_gradient = torch.zeros_like(two_loop_x)
    two_loop_head = torch.zeros((batch_size,), dtype=torch.int32, device=device)
    two_loop_count = torch.zeros_like(two_loop_head)

    compact_step = torch.zeros_like(two_loop_step)
    compact_rho = torch.ones_like(two_loop_rho)
    compact_y = torch.zeros_like(two_loop_y)
    compact_s = torch.zeros_like(two_loop_s)
    compact_x = torch.zeros_like(two_loop_x)
    compact_gradient = torch.zeros_like(two_loop_gradient)
    compact_head = torch.zeros_like(two_loop_head)
    compact_count = torch.zeros_like(two_loop_count)
    compact_inverse_r = torch.eye(history, device=device).expand(batch_size, -1, -1).clone()
    workspace_shape = (batch_size, workspace_elements) if workspace_elements > 0 else (1,)
    compact_workspace = torch.empty(workspace_shape, device=device)
    scale_step = opt_dim > 256
    step_max_value = 0.001 if scale_step else 1.0
    step_max = torch.full((opt_dim,), step_max_value, device=device)
    line_search_status = torch.zeros(batch_size, dtype=torch.int32, device=device)

    q = torch.zeros((batch_size, opt_dim), device=device)
    weights = torch.linspace(0.5, 1.5, opt_dim, device=device).view(1, 1, opt_dim)
    for iteration in range(history + 2):
        q.add_(0.01 * torch.randn_like(q))
        gradient = weights * q.unsqueeze(1)
        curvature_update_valid = torch.full(
            (batch_size,), iteration != history, dtype=torch.bool, device=device
        )
        line_search_status.fill_(4 if iteration == history else 0)
        expected = LBFGScu.apply(
            two_loop_step,
            two_loop_rho,
            two_loop_y,
            two_loop_s,
            q,
            gradient,
            two_loop_x,
            two_loop_gradient,
            two_loop_head,
            two_loop_count,
            0.01,
            True,
            True,
            use_geometric_scaling,
            step_max,
            scale_step,
            curvature_update_valid,
        ).clone()
        actual = LBFGSCompactCu.apply(
            compact_step,
            compact_rho,
            compact_y,
            compact_s,
            compact_inverse_r,
            compact_workspace,
            q,
            gradient,
            compact_x,
            compact_gradient,
            compact_head,
            compact_count,
            step_max,
            0.01,
            shared_matrix_count,
            scale_step,
            curvature_update_valid,
            use_geometric_scaling,
            line_search_status,
        ).clone()
        torch.testing.assert_close(actual, expected, atol=2.5e-3, rtol=3.0e-3)

    torch.testing.assert_close(compact_s, two_loop_s)
    torch.testing.assert_close(compact_y, two_loop_y)
    torch.testing.assert_close(compact_x, two_loop_x)
    torch.testing.assert_close(compact_gradient, two_loop_gradient)
    torch.testing.assert_close(compact_head, two_loop_head)
    torch.testing.assert_close(compact_count, two_loop_count)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compact_operator_auto_selection_and_two_loop_flag():
    """Select compact for benchmarked shapes and retain explicit fallback."""
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Compact L-BFGS requires TF32 tensor cores")

    rollout = MockRollout(num_dof=7, action_horizon=16, batch_size=2)
    compact_config = LBFGSOptCfg(
        num_problems=2,
        num_iters=10,
        history=32,
        step_scale=0.98,
    )
    compact_optimizer = LBFGSOpt(compact_config, [rollout, rollout])
    assert compact_config.use_compact_operator
    assert compact_optimizer._compact_shared_matrix_count is not None
    assert compact_optimizer._qn.compact_inverse_r is not None

    two_loop_config = LBFGSOptCfg(
        num_problems=2,
        num_iters=10,
        history=32,
        step_scale=0.98,
        use_compact_operator=False,
    )
    two_loop_optimizer = LBFGSOpt(two_loop_config, [rollout, rollout])
    assert two_loop_optimizer._compact_shared_matrix_count is None
    assert two_loop_optimizer._qn.compact_inverse_r is None

    padded_dimension_rollout = MockRollout(num_dof=7, action_horizon=10, batch_size=2)
    padded_dimension_optimizer = LBFGSOpt(
        LBFGSOptCfg(
            num_problems=2,
            num_iters=10,
            history=32,
            step_scale=0.98,
        ),
        [padded_dimension_rollout, padded_dimension_rollout],
    )
    assert padded_dimension_optimizer.opt_dim == 70
    assert padded_dimension_optimizer._compact_shared_matrix_count is not None

    short_history_rollout = MockRollout(num_dof=7, action_horizon=1, batch_size=2)
    short_history_optimizer = LBFGSOpt(
        LBFGSOptCfg(
            num_problems=2,
            num_iters=10,
            history=7,
            step_scale=0.98,
        ),
        [short_history_rollout, short_history_rollout],
    )
    assert short_history_optimizer.opt_dim == 7
    assert short_history_optimizer._compact_shared_matrix_count is None
    assert short_history_optimizer._qn.compact_inverse_r is None

    large_dimension_rollout = MockRollout(num_dof=257, action_horizon=1, batch_size=2)
    large_dimension_optimizer = LBFGSOpt(
        LBFGSOptCfg(
            num_problems=2,
            num_iters=10,
            history=16,
            step_scale=0.98,
        ),
        [large_dimension_rollout, large_dimension_rollout],
    )
    assert large_dimension_optimizer.opt_dim == 257
    assert large_dimension_optimizer._compact_shared_matrix_count is None
    assert large_dimension_optimizer._qn.compact_inverse_r is None

    conventional_optimizer = LBFGSOpt(
        LBFGSOptCfg(
            num_problems=2,
            num_iters=10,
            history=32,
            step_scale=0.98,
            use_geometric_scaling=False,
        ),
        [rollout, rollout],
    )
    assert conventional_optimizer._compact_shared_matrix_count is not None
    assert conventional_optimizer._qn.compact_inverse_r is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(
    ("history", "opt_dim"),
    [
        (11, 64),
        (12, 64),
        (64, 64),
        (65, 64),
        (29, 112),
        (30, 112),
        (64, 112),
        (32, 128),
    ],
)
def test_compact_operator_auto_selection_thresholds(history: int, opt_dim: int):
    """Apply performance thresholds only after checking shared-memory fit."""
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device)[0] < 8:
        pytest.skip("Compact L-BFGS requires TF32 tensor cores")

    properties = torch.cuda.get_device_properties(device)
    shared_memory = int(
        getattr(properties, "shared_memory_per_block_optin", 0)
        or properties.shared_memory_per_block
    )
    layout_fits = (
        LBFGSCompactLaunchCfg.calculate_config(2, opt_dim, history, shared_memory) is not None
    )
    in_history_range = 12 <= history <= 64
    in_dimension_range = opt_dim <= 64 or (history >= 30 and opt_dim <= 112)
    expect_compact = layout_fits and in_history_range and in_dimension_range
    rollout = MockRollout(num_dof=opt_dim, action_horizon=1, batch_size=2)
    optimizer = LBFGSOpt(
        LBFGSOptCfg(num_problems=2, num_iters=10, history=history),
        [rollout, rollout],
    )

    assert (optimizer._compact_shared_matrix_count is not None) is expect_compact
    assert (optimizer._qn.compact_inverse_r is not None) is expect_compact


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_history_128_selects_two_loop_for_all_problem_shapes():
    """Use circular two-loop above the compact performance range."""
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("Compact L-BFGS requires TF32 tensor cores")

    compact_rollout = MockRollout(num_dof=6, action_horizon=8, batch_size=2)
    compact_optimizer = LBFGSOpt(
        LBFGSOptCfg(num_problems=2, num_iters=10, history=128),
        [compact_rollout, compact_rollout],
    )
    assert compact_optimizer.opt_dim == 48
    assert compact_optimizer._compact_shared_matrix_count is None
    assert compact_optimizer._qn.compact_inverse_r is None

    fallback_rollout = MockRollout(num_dof=1, action_horizon=1025, batch_size=2)
    fallback_optimizer = LBFGSOpt(
        LBFGSOptCfg(num_problems=2, num_iters=10, history=128),
        [fallback_rollout, fallback_rollout],
    )
    assert fallback_optimizer.opt_dim == 1025
    assert fallback_optimizer._compact_shared_matrix_count is None
    assert fallback_optimizer._qn.compact_inverse_r is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize(("opt_dim", "use_shared_buffers"), [(8, True), (112, False)])
def test_two_loop_history_128_launch_paths(opt_dim: int, use_shared_buffers: bool):
    """Launch history 128 with shared and global circular history storage."""
    torch.manual_seed(1280 + opt_dim)
    device = torch.device("cuda")
    history = 128
    batch_size = 2
    q = 0.01 * torch.randn((batch_size, opt_dim), device=device)
    weights = torch.linspace(0.5, 1.5, opt_dim, device=device).view(1, 1, opt_dim)
    gradient = weights * q.unsqueeze(1)
    step = LBFGScu.apply(
        torch.zeros_like(q),
        torch.zeros((history, batch_size, 1, 1), device=device),
        torch.zeros((history, batch_size, opt_dim, 1), device=device),
        torch.zeros((history, batch_size, opt_dim, 1), device=device),
        q,
        gradient,
        torch.zeros((batch_size, opt_dim, 1), device=device),
        torch.zeros((batch_size, opt_dim, 1), device=device),
        torch.zeros((batch_size,), dtype=torch.int32, device=device),
        torch.zeros((batch_size,), dtype=torch.int32, device=device),
        0.01,
        True,
        use_shared_buffers,
        True,
    )

    torch.testing.assert_close(step, -q, atol=2.0e-6, rtol=2.0e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fused_step_scaling_matches_amax_reference():
    """Fuse the flat horizon step limit without changing L-BFGS history updates."""
    torch.manual_seed(2001)
    device = torch.device("cuda")
    batch_size = 17
    action_horizon = 10
    action_dim = 7
    opt_dim = action_horizon * action_dim
    history = 7

    lows = -torch.linspace(0.5, 1.1, action_dim, device=device)
    highs = torch.linspace(0.7, 1.3, action_dim, device=device)
    bounds = ActionBounds(lows, highs, action_horizon, step_scale=0.05)
    torch.testing.assert_close(
        bounds.horizon_step_max,
        bounds.step_max.repeat(action_horizon),
    )
    assert bounds.horizon_step_max.is_contiguous()

    rho = torch.zeros((history, batch_size, 1, 1), device=device)
    y = torch.zeros((history, batch_size, opt_dim, 1), device=device)
    s = torch.zeros_like(y)
    x_0 = torch.zeros((batch_size, opt_dim, 1), device=device)
    grad_0 = torch.zeros_like(x_0)
    history_head = torch.zeros((batch_size,), dtype=torch.int32, device=device)
    history_count = torch.zeros_like(history_head)

    scaled_rho = rho.clone()
    scaled_y = y.clone()
    scaled_s = s.clone()
    scaled_x_0 = x_0.clone()
    scaled_grad_0 = grad_0.clone()
    scaled_head = history_head.clone()
    scaled_count = history_count.clone()

    q = 0.2 * torch.randn((batch_size, opt_dim), device=device)
    weights = torch.linspace(0.5, 1.5, opt_dim, device=device).view(1, 1, opt_dim)
    gradient = weights * q.unsqueeze(1)
    raw_step = LBFGScu.apply(
        torch.zeros((batch_size, opt_dim), device=device),
        rho,
        y,
        s,
        q,
        gradient,
        x_0,
        grad_0,
        history_head,
        history_count,
        0.01,
        True,
        True,
        True,
    ).clone()
    fused_step = LBFGScu.apply(
        torch.zeros((batch_size, opt_dim), device=device),
        scaled_rho,
        scaled_y,
        scaled_s,
        q,
        gradient,
        scaled_x_0,
        scaled_grad_0,
        scaled_head,
        scaled_count,
        0.01,
        True,
        True,
        True,
        bounds.horizon_step_max,
        True,
    ).clone()

    normalizer = torch.amax(
        torch.abs(raw_step) / bounds.horizon_step_max.view(1, opt_dim),
        dim=-1,
        keepdim=True,
    ).clamp(min=1.0)
    torch.testing.assert_close(fused_step, raw_step / normalizer, atol=2.0e-6, rtol=2.0e-5)
    torch.testing.assert_close(scaled_rho, rho)
    torch.testing.assert_close(scaled_y, y)
    torch.testing.assert_close(scaled_s, s)
    torch.testing.assert_close(scaled_x_0, x_0)
    torch.testing.assert_close(scaled_grad_0, grad_0)
    torch.testing.assert_close(scaled_head, history_head)
    torch.testing.assert_close(scaled_count, history_count)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_optimizer_core_step_scaling_uses_flat_horizon_limit():
    """Expose the same amax step scaling for non-fused optimizer directions."""
    rollout = MockRollout(num_dof=7, action_horizon=10, batch_size=2)
    config = LBFGSOptCfg(num_problems=2, num_iters=10, history=7, step_scale=0.05)
    optimizer = LBFGSOpt(config, [rollout, rollout])
    direction = torch.randn((2, 10, 7), device="cuda")

    scaled = optimizer._core.scale_step_direction(direction)
    flat_direction = direction.view(2, -1)
    step_max = optimizer.action_horizon_step_max.view(1, -1)
    normalizer = torch.amax(torch.abs(flat_direction) / step_max, dim=-1, keepdim=True)
    expected = (flat_direction / normalizer.clamp(min=1.0)).view_as(direction)

    torch.testing.assert_close(scaled, expected)
    torch.testing.assert_close(
        optimizer.action_horizon_step_max,
        optimizer.action_step_max.repeat(optimizer.action_horizon),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_scaling_specializations_match_tensor_reference():
    """Match geometric-default and conventional scaling kernel specializations."""
    device = torch.device("cuda")
    history = 7
    batch_size = 1
    opt_dim = 4
    q = torch.tensor([[2.0, 0.0, 0.0, 0.0]], device=device)
    gradient = torch.tensor([[[1.0, 1.0, 1.0, 0.0]]], device=device)
    outputs = []

    for use_geometric_scaling in (True, False):
        step = torch.zeros((batch_size, opt_dim), device=device)
        rho = torch.zeros((history, batch_size, 1, 1), device=device)
        y = torch.zeros((history, batch_size, opt_dim, 1), device=device)
        s = torch.zeros_like(y)
        x_0 = torch.zeros((batch_size, opt_dim, 1), device=device)
        grad_0 = torch.zeros_like(x_0)
        grad_0[:, 2] = 1.0
        history_head = torch.zeros((batch_size,), dtype=torch.int32, device=device)
        history_count = torch.zeros_like(history_head)

        ref_rho = torch.zeros_like(rho)
        ref_y = torch.zeros_like(y)
        ref_s = torch.zeros_like(s)
        ref_x_0 = torch.zeros_like(x_0)
        ref_grad_0 = grad_0.clone()

        cuda_step = LBFGScu.apply(
            step,
            rho,
            y,
            s,
            q,
            gradient,
            x_0,
            grad_0,
            history_head,
            history_count,
            0.01,
            True,
            True,
            use_geometric_scaling,
        ).clone()
        ref_s, ref_y, ref_rho, ref_x_0, ref_grad_0 = _reference_update_buffers(
            q,
            gradient,
            ref_s,
            ref_y,
            ref_rho,
            ref_x_0,
            ref_grad_0,
        )
        reference_step = _reference_step_direction(
            ref_rho,
            ref_y,
            ref_s,
            gradient,
            history,
            0.01,
            use_geometric_scaling,
        ).squeeze(-1)
        torch.testing.assert_close(cuda_step, reference_step, atol=2.0e-6, rtol=2.0e-5)
        outputs.append(cuda_step)

    assert not torch.allclose(outputs[0], outputs[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_invalid_pair_does_not_modify_full_ring():
    """Reject invalid curvature without overwriting or advancing a full history ring."""
    device = torch.device("cuda")
    history = 2
    batch_size = 1
    opt_dim = 2
    step = torch.zeros((batch_size, opt_dim), device=device)
    rho = torch.zeros((history, batch_size, 1, 1), device=device)
    y = torch.zeros((history, batch_size, opt_dim, 1), device=device)
    s = torch.zeros_like(y)
    x_0 = torch.zeros((batch_size, opt_dim, 1), device=device)
    grad_0 = torch.zeros_like(x_0)
    history_head = torch.zeros((batch_size,), dtype=torch.int32, device=device)
    history_count = torch.zeros_like(history_head)

    invalid_step = LBFGScu.apply(
        step,
        rho,
        y,
        s,
        torch.tensor([[1.0, 0.0]], device=device),
        torch.tensor([[[-1.0, 0.0]]], device=device),
        x_0,
        grad_0,
        history_head,
        history_count,
        0.01,
        True,
        True,
        True,
    ).clone()
    torch.testing.assert_close(
        invalid_step,
        torch.tensor([[0.01, 0.0]], device=device),
    )
    assert history_head.item() == 0
    assert history_count.item() == 0
    assert torch.count_nonzero(rho) == 0

    for position, gradient_value in ((2.0, 0.0), (3.0, 1.0)):
        LBFGScu.apply(
            step,
            rho,
            y,
            s,
            torch.tensor([[position, 0.0]], device=device),
            torch.tensor([[[gradient_value, 0.0]]], device=device),
            x_0,
            grad_0,
            history_head,
            history_count,
            0.01,
            True,
            True,
            True,
        )

    assert history_head.item() == 0
    assert history_count.item() == history
    rho_before = rho.clone()
    y_before = y.clone()
    s_before = s.clone()

    LBFGScu.apply(
        step,
        rho,
        y,
        s,
        torch.tensor([[4.0, 0.0]], device=device),
        torch.tensor([[[0.0, 0.0]]], device=device),
        x_0,
        grad_0,
        history_head,
        history_count,
        0.01,
        True,
        True,
        True,
    )
    torch.testing.assert_close(rho, rho_before)
    torch.testing.assert_close(y, y_before)
    torch.testing.assert_close(s, s_before)
    assert history_head.item() == 0
    assert history_count.item() == history
    torch.testing.assert_close(x_0.squeeze(-1), torch.tensor([[4.0, 0.0]], device=device))
    torch.testing.assert_close(grad_0.squeeze(-1), torch.tensor([[0.0, 0.0]], device=device))

    LBFGScu.apply(
        step,
        rho,
        y,
        s,
        torch.tensor([[5.0, 0.0]], device=device),
        torch.tensor([[[2.0, 0.0]]], device=device),
        x_0,
        grad_0,
        history_head,
        history_count,
        0.01,
        True,
        True,
        True,
    )
    assert history_head.item() == 1
    assert history_count.item() == history
    torch.testing.assert_close(s[0, 0].squeeze(-1), torch.tensor([1.0, 0.0], device=device))
    torch.testing.assert_close(y[0, 0].squeeze(-1), torch.tensor([2.0, 0.0], device=device))
    torch.testing.assert_close(rho[0, 0].squeeze(), torch.tensor(0.5, device=device))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_quasi_newton_clear_resets_circular_state():
    """Clear the circular indices for all problems or a selected subset."""
    buffers = QuasiNewtonBuffers(DeviceCfg(), history=7)
    buffers.resize(num_problems=3, opt_dim=16)
    buffers.history_head.copy_(torch.tensor([1, 2, 3], dtype=torch.int32, device="cuda"))
    buffers.history_count.fill_(7)

    mask = torch.tensor([True, False, True], dtype=torch.bool, device="cuda")
    buffers.clear(mask)
    torch.testing.assert_close(
        buffers.history_head,
        torch.tensor([0, 2, 0], dtype=torch.int32, device="cuda"),
    )
    torch.testing.assert_close(
        buffers.history_count,
        torch.tensor([0, 7, 0], dtype=torch.int32, device="cuda"),
    )

    buffers.clear()
    assert torch.count_nonzero(buffers.history_head) == 0
    assert torch.count_nonzero(buffers.history_count) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_quasi_newton_clear_resets_compact_state():
    """Reset compact inverse and virtual inactive curvature entries in place."""
    history = 28
    buffers = QuasiNewtonBuffers(DeviceCfg(), history=history)
    buffers.resize(
        num_problems=3,
        opt_dim=112,
        use_compact_operator=True,
        compact_workspace_elements=1024,
    )
    buffers.compact_inverse_r.mul_(3.0)
    buffers.rho.zero_()

    mask = torch.tensor([True, False, True], dtype=torch.bool, device="cuda")
    buffers.clear(mask)
    identity = torch.eye(history, device="cuda")
    torch.testing.assert_close(buffers.compact_inverse_r[0], identity)
    torch.testing.assert_close(buffers.compact_inverse_r[2], identity)
    torch.testing.assert_close(buffers.compact_inverse_r[1], 3.0 * identity)
    assert torch.all(buffers.rho[:, mask] == 1.0)
    assert torch.all(buffers.rho[:, ~mask] == 0.0)

    buffers.clear()
    torch.testing.assert_close(
        buffers.compact_inverse_r,
        identity.expand_as(buffers.compact_inverse_r),
    )
    assert torch.all(buffers.rho == 1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_quasi_newton_shift_rebuilds_compact_inverse():
    """Rebuild compact curvature state after an MPC horizon shift."""
    torch.manual_seed(83)
    history = 16
    buffers = QuasiNewtonBuffers(DeviceCfg(), history=history)
    buffers.resize(
        num_problems=1,
        opt_dim=16,
        use_compact_operator=True,
    )
    active_history = 2
    history_s = torch.randn((active_history, 16), device="cuda")
    buffers.s[:active_history, 0, :, 0].copy_(history_s)
    buffers.y[:active_history, 0, :, 0].copy_(history_s)
    buffers.history_head.fill_(active_history)
    buffers.history_count.fill_(active_history)

    buffers.shift(shift_steps=1, action_dim=8)
    buffers.rebuild_compact_state()

    shifted_s = buffers.s[:active_history, 0, :, 0]
    shifted_y = buffers.y[:active_history, 0, :, 0]
    upper = torch.triu(shifted_s @ shifted_y.transpose(0, 1))
    expected_inverse = torch.linalg.inv(upper)
    torch.testing.assert_close(
        buffers.compact_inverse_r[0, :active_history, :active_history],
        expected_inverse,
    )
    torch.testing.assert_close(
        buffers.rho[:active_history, 0, 0, 0],
        1.0 / torch.diagonal(upper),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_history_can_exceed_problem_dimension():
    """Retain configured history even when it exceeds the optimization dimension."""
    rollout = MockRollout(num_dof=2, action_horizon=2, batch_size=2)
    config = LBFGSOptCfg(num_problems=2, num_iters=10, history=7)
    optimizer = LBFGSOpt(config, [rollout, rollout])

    assert optimizer._core.config.history == 7
    assert optimizer._qn.history == 7
    assert optimizer._qn.y.shape[0] == 7
    assert optimizer._core.config.use_geometric_scaling


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_large_problem_uses_fused_step_direction():
    """Use the circular CUDA direction kernel beyond 1024 variables."""
    rollout = MockRollout(num_dof=1, action_horizon=1025, batch_size=2)
    config = LBFGSOptCfg(num_problems=2, num_iters=10, history=7)
    optimizer = LBFGSOpt(config, [rollout, rollout])

    assert optimizer._qn.step_q_buffer.shape == (2, 1025)


@pytest.mark.parametrize("history", [0, 129])
def test_unsupported_history_fails_configuration(history: int):
    """Reject history capacities outside the fused CUDA kernel contract."""
    with pytest.raises(ValueError, match="history must be between 1 and 128"):
        LBFGSOptCfg(history=history)


def test_cpu_configuration_is_rejected():
    """Reject L-BFGS configurations that cannot use the fused CUDA kernel."""
    with pytest.raises(ValueError, match="requires a CUDA device"):
        LBFGSOptCfg(device_cfg=DeviceCfg(device="cpu"))
