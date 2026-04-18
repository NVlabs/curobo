#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for OptimizerCudaGraphMixin and _graphable_methods capability declaration."""

from __future__ import annotations

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.optim.gradient.gradient_descent import GradientDescentOpt, GradientDescentOptCfg
from curobo._src.optim.gradient.lbfgs import LBFGSOptCfg
from curobo._src.optim.optim_factory import create_optimization_config, create_optimizer
from curobo._src.optim.particle.mppi import MPPI, MPPICfg
from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints, RolloutMetrics, RolloutResult
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.sampling.sample_buffer import SampleBuffer


def cost_fn(state):
    """Simple quadratic cost function."""
    costs = torch.sum((10.0 - state) ** 2, dim=-1).unsqueeze(-1)
    return costs


class MockRollout:
    """Standalone mock rollout function for testing (no base class)."""

    def __init__(
        self,
        num_dof: int = 7,
        action_horizon: int = 10,
        batch_size: int = 4,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize the mock rollout function."""
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
def device():
    """Get the device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def dtype():
    """Get the dtype for testing."""
    return torch.float32


@pytest.fixture
def device_cfg(device, dtype):
    """Create a DeviceCfg."""
    return DeviceCfg(device=device, dtype=dtype)


@pytest.fixture
def rollout_fn(device, dtype):
    """Create a mock rollout function."""
    return MockRollout(num_dof=7, action_horizon=10, batch_size=4, device=device, dtype=dtype)


class TestGraphableMethodsDeclaration:
    """Test the _graphable_methods capability declaration pattern."""

    def test_mppi_graphable_methods(self, device_cfg, rollout_fn):
        """Test that MPPI optimizer uses particle base graphable methods."""
        config = MPPICfg(
            num_iters=5,
            num_particles=10,
            solver_type="mppi",
            device_cfg=device_cfg,
        )
        optimizer = MPPI(config, [rollout_fn])

        # MPPI should inherit particle opt's graphable methods
        assert optimizer._graphable_methods == {"_opt_iters"}

    def test_gradient_descent_graphable_methods(self, device_cfg, rollout_fn):
        """Test that standalone GradientDescentOpt declares _opt_iters as graphable."""
        config = GradientDescentOptCfg(
            num_iters=5,
            solver_type="gradient_descent",
            device_cfg=device_cfg,
        )
        optimizer = GradientDescentOpt(config, [rollout_fn])

        assert "_opt_iters" in optimizer._graphable_methods


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestOptimizerCudaGraph:
    """Test CUDA graph executor creation for standalone optimizers."""

    def test_mixin_creates_executors_based_on_graphable_methods(self, device_cfg, rollout_fn):
        """Test that mixin only creates executors for declared graphable methods."""
        config = MPPICfg(
            num_iters=5,
            num_particles=10,
            solver_type="mppi",
            device_cfg=device_cfg,
        )

        # Create optimizer with CUDA graph mixin
        optimizer = create_optimizer(config, [rollout_fn], use_cuda_graph=True)

        # MPPI's _graphable_methods is {"_opt_iters"} only
        # So only _opt_iters executor should be created
        assert "_opt_iters" in optimizer._executors
        assert "_prepare_initial_iteration_state" not in optimizer._executors

    def test_standalone_gd_creates_opt_iters_executor(self, device_cfg, rollout_fn):
        """Test that standalone GD with use_cuda_graph creates _opt_iters executor."""
        config = GradientDescentOptCfg(
            num_iters=5,
            solver_type="gradient_descent",
            device_cfg=device_cfg,
        )

        optimizer = create_optimizer(config, [rollout_fn], use_cuda_graph=True)

        assert "_opt_iters" in optimizer._executors

    def test_mixin_reset_cuda_graph(self, device_cfg, rollout_fn):
        """Test that reset_cuda_graph resets all executors."""
        config = GradientDescentOptCfg(
            num_iters=5,
            solver_type="gradient_descent",
            device_cfg=device_cfg,
        )

        optimizer = create_optimizer(config, [rollout_fn], use_cuda_graph=True)
        optimizer.update_num_problems(4)

        # Run optimization to record graphs
        initial_action = torch.randn(
            4, rollout_fn.action_horizon, rollout_fn.action_dim,
            device=device_cfg.device, dtype=device_cfg.dtype
        )
        optimizer.optimize(initial_action)

        # Reset should not raise errors
        optimizer.reset_cuda_graph()

    def test_mixin_can_optimize(self, device_cfg, rollout_fn):
        """Test that optimizer with mixin can successfully optimize."""
        config = GradientDescentOptCfg(
            num_iters=10,
            solver_type="gradient_descent",
            device_cfg=device_cfg,
        )

        optimizer = create_optimizer(config, [rollout_fn], use_cuda_graph=True)
        optimizer.update_num_problems(4)

        initial_action = torch.randn(
            4, rollout_fn.action_horizon, rollout_fn.action_dim,
            device=device_cfg.device, dtype=device_cfg.dtype
        ) * 5.0

        result = optimizer.optimize(initial_action)

        # Cost should be reduced
        initial_cost = cost_fn(initial_action).mean().item()
        final_cost = cost_fn(result).mean().item()

        assert final_cost < initial_cost

    def test_particle_optimizer_with_cuda_graph(self, device_cfg, rollout_fn):
        """Test MPPI with CUDA graph mixin."""
        config = MPPICfg(
            num_iters=5,
            num_particles=20,
            solver_type="mppi",
            device_cfg=device_cfg,
        )

        optimizer = create_optimizer(config, [rollout_fn], use_cuda_graph=True)
        optimizer.update_num_problems(4)

        initial_action = torch.randn(
            4, rollout_fn.action_horizon, rollout_fn.action_dim,
            device=device_cfg.device, dtype=device_cfg.dtype
        ) * 5.0

        result = optimizer.optimize(initial_action)

        # Should produce valid output
        assert result.shape == initial_action.shape
        assert torch.all(torch.isfinite(result))

    def test_lbfgs_with_cuda_graph(self, device_cfg, rollout_fn):
        """Test LBFGS with CUDA graph mixin."""
        config = LBFGSOptCfg(
            num_iters=10,
            history=5,
            solver_type="lbfgs",
            use_cuda_kernel_step_direction=False,
            device_cfg=device_cfg,
        )

        optimizer = create_optimizer(config, [rollout_fn, rollout_fn], use_cuda_graph=True)
        optimizer.update_num_problems(4)

        # LBFGS (via GradientOptBase) should have both executors
        assert "_opt_iters" in optimizer._executors
        assert "_prepare_initial_iteration_state" in optimizer._executors

        initial_action = torch.randn(
            4, rollout_fn.action_horizon, rollout_fn.action_dim,
            device=device_cfg.device, dtype=device_cfg.dtype
        ) * 5.0

        result = optimizer.optimize(initial_action)

        # Cost should be reduced
        initial_cost = cost_fn(initial_action).mean().item()
        final_cost = cost_fn(result).mean().item()

        assert final_cost < initial_cost


class TestFactoryWithCudaGraph:
    """Test optimizer factory with use_cuda_graph=True."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "solver_type",
        ["lbfgs", "gradient_descent", "conjugate_gradient", "lsr1", "mppi", "es"],
    )
    def test_factory_creates_cuda_graph_optimizer(self, device_cfg, rollout_fn, solver_type):
        """Test that factory correctly creates CUDA graph wrapped optimizers."""
        config_dict = {
            "solver_type": solver_type,
            "num_iters": 5,
        }

        # Add solver-specific parameters
        if solver_type in ["lbfgs", "lsr1"]:
            config_dict["history"] = 5
            config_dict["use_cuda_kernel_step_direction"] = False
            rollout_list = [rollout_fn, rollout_fn]
        elif solver_type == "conjugate_gradient":
            config_dict["cg_method"] = "FR"
            rollout_list = [rollout_fn]
        elif solver_type in ["mppi", "es"]:
            config_dict["num_particles"] = 20
            rollout_list = [rollout_fn]
            if solver_type == "es":
                config_dict["learning_rate"] = 0.1
        else:
            rollout_list = [rollout_fn]

        config = create_optimization_config(config_dict, device_cfg)
        optimizer = create_optimizer(config, rollout_list, use_cuda_graph=True)

        # Should have the mixin's _executors attribute
        assert hasattr(optimizer, "_executors")

        # Should have at least _opt_iters executor
        assert "_opt_iters" in optimizer._executors

        # Line-search gradient optimizers graph both _opt_iters and _prepare_initial_iteration_state
        if solver_type in ["lbfgs", "conjugate_gradient", "lsr1"]:
            assert "_prepare_initial_iteration_state" in optimizer._executors

        # Simple gradient descent and particle optimizers only graph _opt_iters
        if solver_type in ["gradient_descent", "mppi", "es"]:
            assert "_prepare_initial_iteration_state" not in optimizer._executors
