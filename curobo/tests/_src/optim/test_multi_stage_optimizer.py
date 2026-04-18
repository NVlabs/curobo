#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for the MultiStageOptimizer."""

from __future__ import annotations

# Third Party
import numpy as np
import pytest
import torch

# CuRobo
from curobo._src.optim.gradient.gradient_descent import GradientDescentOpt, GradientDescentOptCfg
from curobo._src.optim.gradient.lbfgs import LBFGSOpt, LBFGSOptCfg
from curobo._src.optim.multi_stage_optimizer import MultiStageOptimizer
from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints, RolloutMetrics, RolloutResult
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.sampling.sample_buffer import SampleBuffer


def cost_fn(state):
    """Simple quadratic cost function: sum((10.0 - state)^2)."""
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
def device():
    """Get the device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def dtype():
    """Get the dtype for testing."""
    return torch.float32


@pytest.fixture
def optimizer_setup(device, dtype):
    """Set up environment for multi-stage optimizer tests."""
    # Set random seed for reproducibility
    torch.manual_seed(0)
    np.random.seed(0)

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
    }


class TestMultiStageOptimizer:
    """Test cases for MultiStageOptimizer."""

    def test_init_single_optimizer(self, optimizer_setup):
        """Test initialization with a single optimizer."""
        # Create a single optimizer
        config = GradientDescentOptCfg(
            num_problems=optimizer_setup["batch_size"],
            num_iters=10,
            solver_type="gradient_descent",
            device_cfg=DeviceCfg(
                device=optimizer_setup["device"], dtype=optimizer_setup["dtype"]
            ),
        )
        optimizer1 = GradientDescentOpt(config, [optimizer_setup["rollout_fn"]])
        optimizer1.update_num_problems(optimizer_setup["batch_size"])

        # Create multi-stage optimizer
        multi_opt = MultiStageOptimizer([optimizer1])

        assert len(multi_opt.optimizers) == 1
        assert multi_opt.action_horizon == optimizer1.action_horizon
        assert multi_opt.action_dim == optimizer1.action_dim

    def test_init_two_optimizers(self, optimizer_setup):
        """Test initialization with two optimizers."""
        rollout_fn = optimizer_setup["rollout_fn"]

        # Create two optimizers
        config1 = GradientDescentOptCfg(
            num_problems=optimizer_setup["batch_size"],
            num_iters=5,
            solver_type="gradient_descent",
            device_cfg=DeviceCfg(
                device=optimizer_setup["device"], dtype=optimizer_setup["dtype"]
            ),
        )
        optimizer1 = GradientDescentOpt(config1, [rollout_fn])
        optimizer1.update_num_problems(optimizer_setup["batch_size"])

        config2 = LBFGSOptCfg(
            num_problems=optimizer_setup["batch_size"],
            num_iters=10,
            history=10,
            solver_type="lbfgs",
            device_cfg=DeviceCfg(
                device=optimizer_setup["device"], dtype=optimizer_setup["dtype"]
            ),
        )
        optimizer2 = LBFGSOpt(config2, [rollout_fn, rollout_fn])
        optimizer2.update_num_problems(optimizer_setup["batch_size"])

        # Create multi-stage optimizer
        multi_opt = MultiStageOptimizer([optimizer1, optimizer2])

        assert len(multi_opt.optimizers) == 2
        assert multi_opt.action_horizon == optimizer2.action_horizon
        assert multi_opt.action_dim == optimizer2.action_dim

    def test_optimize_single_stage(self, optimizer_setup):
        """Test optimization with a single stage."""
        config = GradientDescentOptCfg(
            num_problems=optimizer_setup["batch_size"],
            num_iters=20,
            solver_type="gradient_descent",
            device_cfg=DeviceCfg(
                device=optimizer_setup["device"], dtype=optimizer_setup["dtype"]
            ),
        )
        optimizer = GradientDescentOpt(config, [optimizer_setup["rollout_fn"]])
        optimizer.update_num_problems(optimizer_setup["batch_size"])

        multi_opt = MultiStageOptimizer([optimizer])

        # Create initial action
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

        # Run optimization
        result = multi_opt.optimize(initial_action.clone())

        # Check that cost was reduced
        initial_cost = cost_fn(initial_action).mean().item()
        final_cost = cost_fn(result).mean().item()

        print(f"Initial cost: {initial_cost}")
        print(f"Final cost: {final_cost}")

        assert final_cost < initial_cost

    def test_optimize_two_stages(self, optimizer_setup):
        """Test optimization with two stages."""
        rollout_fn = optimizer_setup["rollout_fn"]

        # Stage 1: Gradient descent for quick initial progress
        config1 = GradientDescentOptCfg(
            num_problems=optimizer_setup["batch_size"],
            num_iters=10,
            solver_type="gradient_descent",
            device_cfg=DeviceCfg(
                device=optimizer_setup["device"], dtype=optimizer_setup["dtype"]
            ),
        )
        optimizer1 = GradientDescentOpt(config1, [rollout_fn])
        optimizer1.update_num_problems(optimizer_setup["batch_size"])

        # Stage 2: LBFGS for refinement
        config2 = LBFGSOptCfg(
            num_problems=optimizer_setup["batch_size"],
            num_iters=10,
            history=10,
            solver_type="lbfgs",
            use_cuda_kernel_step_direction=False,
            device_cfg=DeviceCfg(
                device=optimizer_setup["device"], dtype=optimizer_setup["dtype"]
            ),
        )
        optimizer2 = LBFGSOpt(config2, [rollout_fn, rollout_fn])
        optimizer2.update_num_problems(optimizer_setup["batch_size"])

        multi_opt = MultiStageOptimizer([optimizer1, optimizer2])

        # Create initial action
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

        # Run optimization
        result = multi_opt.optimize(initial_action.clone())

        # Check that cost was reduced
        initial_cost = cost_fn(initial_action).mean().item()
        final_cost = cost_fn(result).mean().item()

        print(f"Initial cost: {initial_cost}")
        print(f"Final cost after two stages: {final_cost}")

        assert final_cost < initial_cost
        assert final_cost < 50.0  # Should make significant progress

    def test_reinitialize(self, optimizer_setup):
        """Test that reinitialize works correctly."""
        config = GradientDescentOptCfg(
            num_problems=optimizer_setup["batch_size"],
            num_iters=10,
            solver_type="gradient_descent",
            device_cfg=DeviceCfg(
                device=optimizer_setup["device"], dtype=optimizer_setup["dtype"]
            ),
        )
        optimizer = GradientDescentOpt(config, [optimizer_setup["rollout_fn"]])
        optimizer.update_num_problems(optimizer_setup["batch_size"])

        multi_opt = MultiStageOptimizer([optimizer])

        # Create initial action for reinitialize
        initial_action = torch.zeros(
            optimizer_setup["batch_size"],
            optimizer_setup["action_horizon"],
            optimizer_setup["num_dof"],
            device=optimizer_setup["device"],
            dtype=optimizer_setup["dtype"],
        )

        # Reinitialize should not raise errors
        multi_opt.reinitialize(initial_action)
        multi_opt.reinitialize(initial_action, clear_optimizer_state=False)

    def test_update_num_problems(self, optimizer_setup):
        """Test updating the number of problems."""
        config = GradientDescentOptCfg(
            num_problems=optimizer_setup["batch_size"],
            num_iters=10,
            solver_type="gradient_descent",
            device_cfg=DeviceCfg(
                device=optimizer_setup["device"], dtype=optimizer_setup["dtype"]
            ),
        )
        optimizer = GradientDescentOpt(config, [optimizer_setup["rollout_fn"]])
        optimizer.update_num_problems(optimizer_setup["batch_size"])

        multi_opt = MultiStageOptimizer([optimizer])

        # Update number of problems
        new_num_problems = optimizer_setup["batch_size"] + 2
        multi_opt.update_num_problems(new_num_problems)

        # Check that it was updated in the underlying optimizer
        assert optimizer.config.num_problems == new_num_problems

    def test_get_all_rollout_instances(self, optimizer_setup):
        """Test getting all rollout instances."""
        rollout_fn = optimizer_setup["rollout_fn"]

        config1 = GradientDescentOptCfg(
            num_problems=optimizer_setup["batch_size"],
            num_iters=5,
            solver_type="gradient_descent",
            device_cfg=DeviceCfg(
                device=optimizer_setup["device"], dtype=optimizer_setup["dtype"]
            ),
        )
        optimizer1 = GradientDescentOpt(config1, [rollout_fn])

        config2 = LBFGSOptCfg(
            num_problems=optimizer_setup["batch_size"],
            num_iters=5,
            history=5,
            solver_type="lbfgs",
            device_cfg=DeviceCfg(
                device=optimizer_setup["device"], dtype=optimizer_setup["dtype"]
            ),
        )
        optimizer2 = LBFGSOpt(config2, [rollout_fn, rollout_fn])

        multi_opt = MultiStageOptimizer([optimizer1, optimizer2])

        rollout_instances = multi_opt.get_all_rollout_instances()

        # Should have rollouts from both optimizers
        assert len(rollout_instances) >= 2

    def test_get_recorded_trace(self, optimizer_setup):
        """Test getting the recorded trace."""
        config = GradientDescentOptCfg(
            num_problems=optimizer_setup["batch_size"],
            num_iters=5,
            solver_type="gradient_descent",
            device_cfg=DeviceCfg(
                device=optimizer_setup["device"], dtype=optimizer_setup["dtype"]
            ),
        )
        optimizer = GradientDescentOpt(config, [optimizer_setup["rollout_fn"]])

        multi_opt = MultiStageOptimizer([optimizer])

        trace = multi_opt.get_recorded_trace()

        # Should have debug keys
        assert "debug" in trace
        assert "debug_cost" in trace
        assert isinstance(trace["debug"], list)
        assert isinstance(trace["debug_cost"], list)

    def test_action_horizon_property(self, optimizer_setup):
        """Test the action_horizon property."""
        config = GradientDescentOptCfg(
            num_problems=optimizer_setup["batch_size"],
            num_iters=10,
            solver_type="gradient_descent",
            device_cfg=DeviceCfg(
                device=optimizer_setup["device"], dtype=optimizer_setup["dtype"]
            ),
        )
        optimizer = GradientDescentOpt(config, [optimizer_setup["rollout_fn"]])
        optimizer.update_num_problems(optimizer_setup["batch_size"])

        multi_opt = MultiStageOptimizer([optimizer])

        assert multi_opt.action_horizon == optimizer_setup["action_horizon"]

    def test_action_dim_property(self, optimizer_setup):
        """Test the action_dim property."""
        config = GradientDescentOptCfg(
            num_problems=optimizer_setup["batch_size"],
            num_iters=10,
            solver_type="gradient_descent",
            device_cfg=DeviceCfg(
                device=optimizer_setup["device"], dtype=optimizer_setup["dtype"]
            ),
        )
        optimizer = GradientDescentOpt(config, [optimizer_setup["rollout_fn"]])
        optimizer.update_num_problems(optimizer_setup["batch_size"])

        multi_opt = MultiStageOptimizer([optimizer])

        assert multi_opt.action_dim == optimizer_setup["num_dof"]

    def test_outer_iters_property(self, optimizer_setup):
        """Test the outer_iters property."""
        config = GradientDescentOptCfg(
            num_problems=optimizer_setup["batch_size"],
            num_iters=10,
            solver_type="gradient_descent",
            device_cfg=DeviceCfg(
                device=optimizer_setup["device"], dtype=optimizer_setup["dtype"]
            ),
        )
        optimizer = GradientDescentOpt(config, [optimizer_setup["rollout_fn"]])

        multi_opt = MultiStageOptimizer([optimizer])

        # Multi-stage optimizer should return 1 for outer_iters
        assert multi_opt.outer_iters == 1

    def test_solver_names_property(self, optimizer_setup):
        """Test the solver_names property."""
        rollout_fn = optimizer_setup["rollout_fn"]

        config1 = GradientDescentOptCfg(
            num_problems=optimizer_setup["batch_size"],
            num_iters=5,
            solver_type="gradient_descent",
            solver_name="gd",
            device_cfg=DeviceCfg(
                device=optimizer_setup["device"], dtype=optimizer_setup["dtype"]
            ),
        )
        optimizer1 = GradientDescentOpt(config1, [rollout_fn])

        config2 = LBFGSOptCfg(
            num_problems=optimizer_setup["batch_size"],
            num_iters=5,
            history=5,
            solver_type="lbfgs",
            solver_name="lbfgs",
            device_cfg=DeviceCfg(
                device=optimizer_setup["device"], dtype=optimizer_setup["dtype"]
            ),
        )
        optimizer2 = LBFGSOpt(config2, [rollout_fn, rollout_fn])

        multi_opt = MultiStageOptimizer([optimizer1, optimizer2])

        solver_names = multi_opt.solver_names
        assert len(solver_names) == 2
        assert "gd" in solver_names
        assert "lbfgs" in solver_names

    def test_reset_cuda_graph(self, optimizer_setup):
        """Test reset_cuda_graph method."""
        config = GradientDescentOptCfg(
            num_problems=optimizer_setup["batch_size"],
            num_iters=5,
            solver_type="gradient_descent",
            device_cfg=DeviceCfg(
                device=optimizer_setup["device"], dtype=optimizer_setup["dtype"]
            ),
        )
        optimizer = GradientDescentOpt(config, [optimizer_setup["rollout_fn"]])

        multi_opt = MultiStageOptimizer([optimizer])

        # Should not raise errors
        multi_opt.reset_cuda_graph()

    def test_reset_shape(self, optimizer_setup):
        """Test reset_shape method."""
        config = GradientDescentOptCfg(
            num_problems=optimizer_setup["batch_size"],
            num_iters=5,
            solver_type="gradient_descent",
            device_cfg=DeviceCfg(
                device=optimizer_setup["device"], dtype=optimizer_setup["dtype"]
            ),
        )
        optimizer = GradientDescentOpt(config, [optimizer_setup["rollout_fn"]])

        multi_opt = MultiStageOptimizer([optimizer])

        # Should not raise errors
        multi_opt.reset_shape()

    def test_reset_seed(self, optimizer_setup):
        """Test reset_seed method."""
        config = GradientDescentOptCfg(
            num_problems=optimizer_setup["batch_size"],
            num_iters=5,
            solver_type="gradient_descent",
            device_cfg=DeviceCfg(
                device=optimizer_setup["device"], dtype=optimizer_setup["dtype"]
            ),
        )
        optimizer = GradientDescentOpt(config, [optimizer_setup["rollout_fn"]])

        multi_opt = MultiStageOptimizer([optimizer])

        # Should not raise errors
        multi_opt.reset_seed()

    def test_sequential_improvement(self, optimizer_setup):
        """Test that each stage improves the solution."""
        rollout_fn = optimizer_setup["rollout_fn"]

        # Create two stages
        config1 = GradientDescentOptCfg(
            num_problems=optimizer_setup["batch_size"],
            num_iters=5,
            solver_type="gradient_descent",
            device_cfg=DeviceCfg(
                device=optimizer_setup["device"], dtype=optimizer_setup["dtype"]
            ),
        )
        optimizer1 = GradientDescentOpt(config1, [rollout_fn])
        optimizer1.update_num_problems(optimizer_setup["batch_size"])

        config2 = GradientDescentOptCfg(
            num_problems=optimizer_setup["batch_size"],
            num_iters=10,
            solver_type="gradient_descent",
            device_cfg=DeviceCfg(
                device=optimizer_setup["device"], dtype=optimizer_setup["dtype"]
            ),
        )
        optimizer2 = GradientDescentOpt(config2, [rollout_fn])
        optimizer2.update_num_problems(optimizer_setup["batch_size"])

        multi_opt = MultiStageOptimizer([optimizer1, optimizer2])

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

        # Run first stage only
        result1 = optimizer1.optimize(initial_action.clone())
        cost1 = cost_fn(result1).mean().item()

        # Run both stages
        result2 = multi_opt.optimize(initial_action.clone())
        cost2 = cost_fn(result2).mean().item()

        initial_cost = cost_fn(initial_action).mean().item()

        print(f"Initial cost: {initial_cost}")
        print(f"Cost after stage 1: {cost1}")
        print(f"Cost after both stages: {cost2}")

        # Both should improve, and two stages should be at least as good as one
        assert cost1 < initial_cost
        assert cost2 <= cost1 * 1.1  # Allow small tolerance

