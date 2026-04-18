#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for the LBFGS optimizer comparing CUDA and non-CUDA implementations."""

from __future__ import annotations

# Third Party
import numpy as np
import pytest
import torch

# CuRobo
from curobo._src.optim.gradient.lbfgs import LBFGSOpt, LBFGSOptCfg
from curobo._src.optim.optimization_iteration_state import OptimizationIterationState
from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints, RolloutMetrics, RolloutResult
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.sampling.sample_buffer import SampleBuffer


def pytest_generate_tests(metafunc):
    """Generate test parameters dynamically based on CLI options.

    This enables different parametrization strategies:
    - Default mode (pytest): history=[1, 10, 31] for fast CI (12 test variants)
    - Full mode (pytest --full-params): history=[1..31] for comprehensive testing (124 variants)

    Args:
        metafunc: pytest metafunc object for the current test.
    """
    if "lbfgs_history_range" in metafunc.fixturenames:
        # Check if --full-params flag is set
        use_full_params = metafunc.config.getoption("--full-params", default=False)

        if use_full_params:
            # Full range: 1 to 31 for complete parameter sweep
            history_values = list(range(1, 32))
        else:
            # Quick range: representative subset for fast CI/default runs
            history_values = [1, 10, 31]

        metafunc.parametrize("lbfgs_history_range", history_values)


def cost_fn(state):
    costs = torch.sum((10.0 - state) ** 2, dim=-1).unsqueeze(-1)
    return costs


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
            ndims=num_dof, device_cfg=self.device_cfg,
            up_bounds=self._action_bound_highs, low_bounds=self._action_bound_lows,
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
        return RolloutMetrics(costs_and_constraints=cc, feasible=cc.get_feasible(), state=state, convergence=convergence)

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
                (num_samples, self._action_horizon, self._action_dim), **self.device_cfg.as_torch_dict())
        else:
            init_action = torch.zeros(
                (num_samples, self._action_horizon, self._action_dim), **self.device_cfg.as_torch_dict())
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
    np.random.seed(0)

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
    use_cuda_kernel_step_direction=True,
    use_cuda_kernel_line_search=True,
    history=27,
    num_iters=50,
):
    """Create an LBFGS optimizer with the specified configuration.

    Args:
        rollout_fn: Rollout function for optimization.
        batch_size: Batch size for optimization.
        use_cuda_kernel: Whether to use the CUDA kernel implementation.

    Returns:
        LBFGSOpt: Configured optimizer.
    """
    # Create optimizer configuration
    config = LBFGSOptCfg(
        num_problems=batch_size,
        num_iters=num_iters,
        history=history,
        step_scale=0.98,
        use_cuda_kernel_step_direction=use_cuda_kernel_step_direction,
        use_cuda_kernel_shared_buffers=True,
        use_cuda_kernel_line_search=use_cuda_kernel_line_search,
        stable_mode=True,
        solver_type="lbfgs",
        line_search_scale=[0, 0.1, 0.5, 1.0],
    )

    # Create optimizer with 2 rollout instances (default for LBFGS)
    optimizer = LBFGSOpt(config, [rollout_fn, rollout_fn])
    optimizer.update_num_problems(batch_size)
    return optimizer


@pytest.mark.parametrize("use_cuda_kernel_step_direction", [True, False])
@pytest.mark.parametrize("use_cuda_kernel_line_search", [True, False])
def test_lbfgs_optimization_cuda_kernel(
    optimizer_setup,
    lbfgs_history_range,
    use_cuda_kernel_step_direction,
    use_cuda_kernel_line_search,
):
    """Test that LBFGS optimizer can minimize a simple quadratic function.

    Tests with different history sizes:
    - Quick mode (default): Tests history=[1, 10, 31] for fast CI (12 variants)
    - Full mode (pytest --full-params): Tests history=[1..31] for comprehensive validation (124 variants)

    Args:
        optimizer_setup: pytest fixture providing test setup.
        lbfgs_history_range: History size for LBFGS (dynamic based on --full-params).
        use_cuda_kernel_step_direction: Whether to use CUDA kernel for step direction.
        use_cuda_kernel_line_search: Whether to use CUDA kernel for line search.
    """
    history = lbfgs_history_range
    initial_action = None
    for i in range(2):
        optimizer_with_cuda = create_optimizer(
            optimizer_setup["rollout_fn"],
            optimizer_setup["batch_size"],
            use_cuda_kernel_step_direction=use_cuda_kernel_step_direction,
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


def test_lbfgs_optimization_kernel_vs_no_kernel(optimizer_setup):
    """Test that LBFGS optimizer can minimize a simple quadratic function.

    Args:
        optimizer_setup: pytest fixture providing test setup.
    """
    initial_action = None
    for i in range(2):
        # Create two optimizers: one with CUDA kernel and one without
        optimizer_with_cuda = create_optimizer(
            optimizer_setup["rollout_fn"],
            optimizer_setup["batch_size"],
            use_cuda_kernel_step_direction=True,
            use_cuda_kernel_line_search=True,
            num_iters=10,
        )
        optimizer_without_cuda = create_optimizer(
            optimizer_setup["rollout_fn"],
            optimizer_setup["batch_size"],
            use_cuda_kernel_step_direction=False,
            use_cuda_kernel_line_search=False,
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
        action_no_cuda = initial_action.clone()

        # Run optimization with CUDA kernel
        optimizer_with_cuda.reinitialize(action_cuda)
        result_cuda = optimizer_with_cuda.optimize(action_cuda).clone()

        # Run optimization without CUDA kernel
        optimizer_without_cuda.reinitialize(action_no_cuda)
        result_no_cuda = optimizer_without_cuda.optimize(action_no_cuda).clone()

        # Check that both optimizers reduced the cost
        initial_cost = cost_fn(initial_action).mean().item()

        final_cost_cuda = cost_fn(result_cuda).mean().item()
        final_cost_no_cuda = cost_fn(result_no_cuda).mean().item()
        print(f"Iteration {i}")
        print(f"Initial cost: {initial_cost}")
        print(f"Final cost (CUDA kernel): {final_cost_cuda}")
        print(f"Final cost (without CUDA kernel): {final_cost_no_cuda}")
        # return

        # Both optimizers should reduce the cost
        assert final_cost_cuda < initial_cost
        assert final_cost_no_cuda < initial_cost
        assert np.abs(final_cost_cuda - final_cost_no_cuda) < 1e-3


def test_step_direction_consistency(optimizer_setup):
    """Test that the step direction computation is consistent between CUDA and non-CUDA implementations.

    Args:
        optimizer_setup: pytest fixture providing test setup.
    """
    # Create two optimizers: one with CUDA kernel and one without
    optimizer_with_cuda = create_optimizer(
        optimizer_setup["rollout_fn"],
        optimizer_setup["batch_size"],
        use_cuda_kernel_step_direction=True,
        use_cuda_kernel_line_search=True,
    )
    optimizer_without_cuda = create_optimizer(
        optimizer_setup["rollout_fn"],
        optimizer_setup["batch_size"],
        use_cuda_kernel_step_direction=False,
        use_cuda_kernel_line_search=False,
    )

    # Create a random action and gradient
    action = torch.randn(
        optimizer_setup["batch_size"],
        optimizer_setup["action_horizon"],
        optimizer_setup["num_dof"],
        device=optimizer_setup["device"],
        dtype=optimizer_setup["dtype"],
    )
    gradient = torch.randn(
        optimizer_setup["batch_size"],
        optimizer_setup["action_horizon"],
        optimizer_setup["num_dof"],
        device=optimizer_setup["device"],
        dtype=optimizer_setup["dtype"],
    )

    # Reinitialize optimizers
    optimizer_with_cuda.reinitialize(action)
    optimizer_without_cuda.reinitialize(action)

    # Create iteration state
    iteration_state = OptimizationIterationState(
        action=action,
        gradient=gradient,
        cost=None,
        exploration_action=action,
        exploration_gradient=gradient,
        exploration_cost=None,
    )

    # Compute step direction with CUDA kernel
    optimizer_with_cuda._update_buffers(
        action.view(-1, optimizer_with_cuda.opt_dim), gradient.view(-1, 1, optimizer_with_cuda.opt_dim)
    )
    step_dir_cuda = optimizer_with_cuda._get_step_direction_impl(iteration_state)

    # Compute step direction without CUDA kernel
    optimizer_without_cuda._update_buffers(
        action.view(-1, optimizer_without_cuda.opt_dim),
        gradient.view(-1, 1, optimizer_without_cuda.opt_dim),
    )
    step_dir_no_cuda = optimizer_without_cuda._get_step_direction_impl(iteration_state)

    # Check that the directions are similar (may not be exactly equal due to numerical differences)
    # We check that the cosine similarity between the directions is close to 1
    step_dir_cuda_flat = step_dir_cuda.view(optimizer_setup["batch_size"], -1)
    step_dir_no_cuda_flat = step_dir_no_cuda.view(optimizer_setup["batch_size"], -1)

    # Normalize
    step_dir_cuda_norm = step_dir_cuda_flat / (
        torch.norm(step_dir_cuda_flat, dim=1, keepdim=True) + 1e-8
    )
    step_dir_no_cuda_norm = step_dir_no_cuda_flat / (
        torch.norm(step_dir_no_cuda_flat, dim=1, keepdim=True) + 1e-8
    )

    # Compute cosine similarity
    cosine_similarity = torch.sum(step_dir_cuda_norm * step_dir_no_cuda_norm, dim=1)

    # Average cosine similarity should be close to 1 if directions are similar
    avg_cosine_similarity = cosine_similarity.mean().item()
    print(
        f"Average cosine similarity between CUDA and non-CUDA step directions: {avg_cosine_similarity}"
    )

    # Allow for some numerical differences
    assert avg_cosine_similarity >= 0.9


def test_convergence_comparison(optimizer_setup):
    """Compare convergence speed and final solution quality between CUDA and non-CUDA implementations.

    Args:
        optimizer_setup: pytest fixture providing test setup.
    """
    # Create two optimizers with more iterations for thorough convergence
    config_cuda = LBFGSOptCfg(
        num_problems=optimizer_setup["batch_size"],
        num_iters=100,
        history=15,
        step_scale=optimizer_setup["step_scale"],
        use_cuda_kernel_step_direction=True,
        use_cuda_kernel_shared_buffers=True,
        stable_mode=True,
        solver_type="lbfgs",
    )

    config_no_cuda = LBFGSOptCfg(
        num_problems=optimizer_setup["batch_size"],
        num_iters=100,
        history=15,
        step_scale=optimizer_setup["step_scale"],
        use_cuda_kernel_step_direction=False,
        use_cuda_kernel_shared_buffers=True,
        stable_mode=True,
        solver_type="lbfgs",
    )

    rollout_fn = optimizer_setup["rollout_fn"]
    optimizer_with_cuda = LBFGSOpt(config_cuda, [rollout_fn, rollout_fn])
    optimizer_without_cuda = LBFGSOpt(config_no_cuda, [rollout_fn, rollout_fn])
    optimizer_with_cuda.update_num_problems(optimizer_setup["batch_size"])
    optimizer_without_cuda.update_num_problems(optimizer_setup["batch_size"])

    # Create a random but more challenging starting point
    # Using a larger magnitude makes the optimization problem harder
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
    action_no_cuda = initial_action.clone()

    # Run optimization with CUDA kernel and measure time
    optimizer_with_cuda.reinitialize(action_cuda)
    start_time_cuda = torch.cuda.Event(enable_timing=True)
    end_time_cuda = torch.cuda.Event(enable_timing=True)

    start_time_cuda.record()
    result_cuda = optimizer_with_cuda.optimize(action_cuda)
    end_time_cuda.record()
    torch.cuda.synchronize()
    time_cuda = start_time_cuda.elapsed_time(end_time_cuda)

    # Run optimization without CUDA kernel and measure time
    optimizer_without_cuda.reinitialize(action_no_cuda)
    start_time_no_cuda = torch.cuda.Event(enable_timing=True)
    end_time_no_cuda = torch.cuda.Event(enable_timing=True)

    start_time_no_cuda.record()
    result_no_cuda = optimizer_without_cuda.optimize(action_no_cuda)
    end_time_no_cuda.record()
    torch.cuda.synchronize()
    time_no_cuda = start_time_no_cuda.elapsed_time(end_time_no_cuda)

    # Get final costs
    final_cost_cuda = torch.sum(result_cuda**2, dim=-1).mean().item()
    final_cost_no_cuda = torch.sum(result_no_cuda**2, dim=-1).mean().item()

    print(f"Optimization time (CUDA kernel): {time_cuda} ms")
    print(f"Optimization time (without CUDA kernel): {time_no_cuda} ms")
    print(f"Final cost (CUDA kernel): {final_cost_cuda}")
    print(f"Final cost (without CUDA kernel): {final_cost_no_cuda}")

    # Both results should be close in terms of final cost
    # We use a relative tolerance to account for different magnitudes
    assert final_cost_cuda == final_cost_no_cuda
