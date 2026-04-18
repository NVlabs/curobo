#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for the optimizer factory functions."""

from __future__ import annotations

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.optim.external.scipy_opt import ScipyOpt, ScipyOptCfg
from curobo._src.optim.external.torch_opt import TorchOpt, TorchOptCfg
from curobo._src.optim.gradient.conjugate_gradient import (
    ConjugateGradientOpt,
    ConjugateGradientOptCfg,
)
from curobo._src.optim.gradient.gradient_descent import (
    GradientDescentOpt,
    LineSearchGradientDescentOpt,
)
from curobo._src.optim.gradient.gradient_descent import GradientDescentOptCfg
from curobo._src.optim.gradient.lbfgs import LBFGSOpt, LBFGSOptCfg
from curobo._src.optim.gradient.lsr1 import LSR1Opt
from curobo._src.optim.optim_factory import create_optimization_config, create_optimizer
from curobo._src.optim.particle.evolution_strategies import (
    EvolutionStrategies,
    EvolutionStrategiesCfg,
)
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
def device_cfg(device, dtype):
    """Create a DeviceCfg."""
    return DeviceCfg(device=device, dtype=dtype)


@pytest.fixture
def rollout_fn(device, dtype):
    """Create a mock rollout function."""
    return MockRollout(num_dof=7, action_horizon=10, batch_size=4, device=device, dtype=dtype)


class TestCreateOptimizationConfig:
    """Test cases for create_optimization_config function."""

    @pytest.mark.parametrize(
        "solver_type,expected_config_class",
        [
            ("lbfgs", LBFGSOptCfg),
            ("gradient_descent", GradientDescentOptCfg),
            ("line_search_gradient_descent", GradientDescentOptCfg),
            ("conjugate_gradient", ConjugateGradientOptCfg),
            ("lsr1", LBFGSOptCfg),
            ("scipy", ScipyOptCfg),
            ("torch", TorchOptCfg),
            ("mppi", MPPICfg),
            ("es", EvolutionStrategiesCfg),
        ],
    )
    def test_create_config_for_solver_type(self, device_cfg, solver_type, expected_config_class):
        """Test creating config for different solver types."""
        config_dict = {
            "solver_type": solver_type,
            "num_iters": 10,
        }

        config = create_optimization_config(config_dict, device_cfg)

        assert isinstance(config, expected_config_class)
        assert config.solver_type == solver_type
        assert config.num_iters == 10

    def test_create_config_lbfgs_with_history(self, device_cfg):
        """Test creating LBFGS config with history parameter."""
        config_dict = {
            "solver_type": "lbfgs",
            "num_iters": 20,
            "history": 15,
        }

        config = create_optimization_config(config_dict, device_cfg)

        assert isinstance(config, LBFGSOptCfg)
        assert config.history == 15
        assert config.num_iters == 20

    def test_create_config_conjugate_gradient_with_method(self, device_cfg):
        """Test creating conjugate gradient config with cg_method."""
        config_dict = {
            "solver_type": "conjugate_gradient",
            "num_iters": 10,
            "cg_method": "PR",
            "max_beta": 5.0,
        }

        config = create_optimization_config(config_dict, device_cfg)

        assert isinstance(config, ConjugateGradientOptCfg)
        assert config.cg_method == "PR"
        assert config.max_beta == 5.0

    def test_create_config_mppi(self, device_cfg):
        """Test creating MPPI config."""
        config_dict = {
            "solver_type": "mppi",
            "num_iters": 10,
            "num_particles": 100,
        }

        config = create_optimization_config(config_dict, device_cfg)

        assert isinstance(config, MPPICfg)
        assert config.num_particles == 100

    def test_create_config_es(self, device_cfg):
        """Test creating Evolution Strategies config."""
        config_dict = {
            "solver_type": "es",
            "num_iters": 10,
            "num_particles": 50,
            "learning_rate": 0.1,
        }

        config = create_optimization_config(config_dict, device_cfg)

        assert isinstance(config, EvolutionStrategiesCfg)
        assert config.num_particles == 50
        assert config.learning_rate == 0.1

    def test_create_config_scipy(self, device_cfg):
        """Test creating scipy config."""
        config_dict = {
            "solver_type": "scipy",
            "num_iters": 100,
        }

        config = create_optimization_config(config_dict, device_cfg)

        assert isinstance(config, ScipyOptCfg)
        assert config.num_iters == 100

    def test_create_config_torch(self, device_cfg):
        """Test creating torch optimizer config."""
        config_dict = {
            "solver_type": "torch",
            "num_iters": 50,
            "torch_optim_name": "Adam",
            "torch_optim_kwargs": {"lr": 0.01},
        }

        config = create_optimization_config(config_dict, device_cfg)

        assert isinstance(config, TorchOptCfg)
        assert config.torch_optim_name == "Adam"
        assert config.torch_optim_kwargs["lr"] == 0.01

    def test_create_config_invalid_solver_type(self, device_cfg):
        """Test that invalid solver type raises error."""
        config_dict = {
            "solver_type": "invalid_solver",
            "num_iters": 10,
        }

        with pytest.raises(ValueError, match="Unsupported solver type"):
            create_optimization_config(config_dict, device_cfg)

    def test_create_config_preserves_device_cfg(self, device_cfg):
        """Test that tensor config is preserved."""
        config_dict = {
            "solver_type": "lbfgs",
            "num_iters": 10,
        }

        config = create_optimization_config(config_dict, device_cfg)

        assert config.device_cfg.device == device_cfg.device
        assert config.device_cfg.dtype == device_cfg.dtype


class TestCreateOptimizer:
    """Test cases for create_optimizer function."""

    @pytest.mark.parametrize(
        "solver_type,expected_optimizer_class",
        [
            ("lbfgs", LBFGSOpt),
            ("gradient_descent", GradientDescentOpt),
            ("line_search_gradient_descent", LineSearchGradientDescentOpt),
            ("conjugate_gradient", ConjugateGradientOpt),
            ("lsr1", LSR1Opt),
            ("scipy", ScipyOpt),
            ("torch", TorchOpt),
            ("mppi", MPPI),
            ("es", EvolutionStrategies),
        ],
    )
    def test_create_optimizer_for_solver_type(
        self, device_cfg, rollout_fn, solver_type, expected_optimizer_class
    ):
        """Test creating optimizer for different solver types."""
        # Create config
        config_dict = {
            "solver_type": solver_type,
            "num_iters": 5,
        }

        # Add solver-specific parameters
        if solver_type in ["lbfgs", "lsr1"]:
            config_dict["history"] = 5
            config_dict["use_cuda_kernel_step_direction"] = False
        elif solver_type == "conjugate_gradient":
            config_dict["cg_method"] = "FR"
        elif solver_type in ["mppi", "es"]:
            config_dict["num_particles"] = 20
            if solver_type == "es":
                config_dict["learning_rate"] = 0.1
        elif solver_type == "torch":
            config_dict["torch_optim_name"] = "Adam"
            config_dict["torch_optim_kwargs"] = {"lr": 0.01}

        config = create_optimization_config(config_dict, device_cfg)

        # Create optimizer
        if solver_type in ["lbfgs", "lsr1"]:
            rollout_list = [rollout_fn, rollout_fn]
        else:
            rollout_list = [rollout_fn]

        optimizer = create_optimizer(config, rollout_list)

        assert isinstance(optimizer, expected_optimizer_class)

    def test_create_optimizer_lbfgs(self, device_cfg, rollout_fn):
        """Test creating LBFGS optimizer."""
        config = LBFGSOptCfg(
            num_iters=10,
            history=10,
            solver_type="lbfgs",
            use_cuda_kernel_step_direction=False,
            device_cfg=device_cfg,
        )

        optimizer = create_optimizer(config, [rollout_fn, rollout_fn])

        assert isinstance(optimizer, LBFGSOpt)
        assert optimizer.config.history == 10

    def test_create_optimizer_gradient_descent(self, device_cfg, rollout_fn):
        """Test creating gradient descent optimizer."""
        config = GradientDescentOptCfg(
            num_iters=10,
            solver_type="gradient_descent",
            device_cfg=device_cfg,
        )

        optimizer = create_optimizer(config, [rollout_fn])

        assert isinstance(optimizer, GradientDescentOpt)

    def test_create_optimizer_conjugate_gradient(self, device_cfg, rollout_fn):
        """Test creating conjugate gradient optimizer."""
        config = ConjugateGradientOptCfg(
            num_iters=10,
            cg_method="FR",
            solver_type="conjugate_gradient",
            device_cfg=device_cfg,
        )

        optimizer = create_optimizer(config, [rollout_fn])

        assert isinstance(optimizer, ConjugateGradientOpt)

    def test_create_optimizer_lsr1(self, device_cfg, rollout_fn):
        """Test creating LSR1 optimizer."""
        config = LBFGSOptCfg(
            num_iters=10,
            history=10,
            solver_type="lsr1",
            use_cuda_kernel_step_direction=False,
            device_cfg=device_cfg,
        )

        optimizer = create_optimizer(config, [rollout_fn, rollout_fn])

        assert isinstance(optimizer, LSR1Opt)

    def test_create_optimizer_mppi(self, device_cfg, rollout_fn):
        """Test creating MPPI optimizer."""
        config = MPPICfg(
            num_iters=10,
            num_particles=20,
            solver_type="mppi",
            device_cfg=device_cfg,
        )

        optimizer = create_optimizer(config, [rollout_fn])

        assert isinstance(optimizer, MPPI)

    def test_create_optimizer_es(self, device_cfg, rollout_fn):
        """Test creating Evolution Strategies optimizer."""
        config = EvolutionStrategiesCfg(
            num_iters=10,
            num_particles=20,
            learning_rate=0.1,
            solver_type="es",
            device_cfg=device_cfg,
        )

        optimizer = create_optimizer(config, [rollout_fn])

        assert isinstance(optimizer, EvolutionStrategies)

    def test_create_optimizer_scipy(self, device_cfg, rollout_fn):
        """Test creating scipy optimizer."""
        config = ScipyOptCfg(
            num_iters=100,
            solver_type="scipy",
            device_cfg=device_cfg,
        )

        optimizer = create_optimizer(config, [rollout_fn])

        assert isinstance(optimizer, ScipyOpt)

    def test_create_optimizer_torch(self, device_cfg, rollout_fn):
        """Test creating torch optimizer."""
        config = TorchOptCfg(
            num_iters=50,
            solver_type="torch",
            torch_optim_name="Adam",
            torch_optim_kwargs={"lr": 0.01},
            device_cfg=device_cfg,
        )

        optimizer = create_optimizer(config, [rollout_fn])

        assert isinstance(optimizer, TorchOpt)

    def test_create_optimizer_invalid_rollout_type(self, device_cfg, rollout_fn):
        """Test that passing non-list rollout raises error."""
        config = GradientDescentOptCfg(
            num_iters=10,
            solver_type="gradient_descent",
            device_cfg=device_cfg,
        )

        with pytest.raises(ValueError, match="rollout must be a list"):
            create_optimizer(config, rollout_fn)  # Pass rollout_fn directly, not as list

    def test_create_optimizer_can_optimize(self, device_cfg, rollout_fn):
        """Test that created optimizer can actually optimize."""
        config_dict = {
            "solver_type": "gradient_descent",
            "num_iters": 10,
        }

        config = create_optimization_config(config_dict, device_cfg)
        optimizer = create_optimizer(config, [rollout_fn])

        # Create initial action
        initial_action = torch.randn(
            4, 10, 7, device=device_cfg.device, dtype=device_cfg.dtype
        ) * 10.0

        optimizer.update_num_problems(4)

        # Run optimization - should not raise errors
        result = optimizer.optimize(initial_action)

        # Result should have the same shape
        assert result.shape == initial_action.shape

        # Cost should be reduced
        initial_cost = cost_fn(initial_action).mean().item()
        final_cost = cost_fn(result).mean().item()

        assert final_cost < initial_cost


class TestFactoryIntegration:
    """Integration tests for the factory functions."""

    def test_end_to_end_workflow(self, device_cfg, rollout_fn):
        """Test complete workflow from config dict to optimization."""
        # Define config as dictionary (as would come from YAML/JSON)
        config_dict = {
            "solver_type": "lbfgs",
            "num_iters": 20,
            "history": 10,
            "use_cuda_kernel_step_direction": False,
        }

        # Create config object
        config = create_optimization_config(config_dict, device_cfg)

        # Create optimizer
        optimizer = create_optimizer(config, [rollout_fn, rollout_fn])

        # Update number of problems
        optimizer.update_num_problems(4)

        # Create initial action
        initial_action = torch.randn(
            4, 10, 7, device=device_cfg.device, dtype=device_cfg.dtype
        ) * 10.0

        # Optimize
        result = optimizer.optimize(initial_action)

        # Verify improvement
        initial_cost = cost_fn(initial_action).mean().item()
        final_cost = cost_fn(result).mean().item()

        assert final_cost < initial_cost

    @pytest.mark.parametrize(
        "solver_type",
        ["lbfgs", "gradient_descent", "conjugate_gradient", "lsr1", "mppi", "es"],
    )
    def test_all_gradient_solvers_can_optimize(self, device_cfg, rollout_fn, solver_type):
        """Test that all solver types can perform optimization."""
        config_dict = {
            "solver_type": solver_type,
            "num_iters": 10,
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

        # Create and run optimizer
        config = create_optimization_config(config_dict, device_cfg)
        optimizer = create_optimizer(config, rollout_list)
        optimizer.update_num_problems(4)

        initial_action = torch.randn(
            4, 10, 7, device=device_cfg.device, dtype=device_cfg.dtype
        ) * 5.0

        # Should not raise errors
        result = optimizer.optimize(initial_action)

        # Should have valid output
        assert result.shape == initial_action.shape
        assert torch.all(torch.isfinite(result))

