#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for MPPI and EvolutionStrategies optimizers."""

from __future__ import annotations

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.optim.particle.evolution_strategies import (
    EvolutionStrategies,
    EvolutionStrategiesCfg,
    calc_exp,
    compute_es_mean,
)
from curobo._src.optim.particle.mppi import (
    MPPI,
    BaseActionType,
    CovType,
    MPPICfg,
    SampleMode,
    jit_blend_cov,
    jit_blend_mean,
    jit_calculate_exp_util,
    jit_calculate_exp_util_from_costs,
    jit_compute_total_cost,
    jit_diag_a_cov_update,
    jit_mean_cov_diag_a,
)
from curobo._src.optim.particle.particle_opt_utils import SquashType
from curobo._src.optim.particle.sample_strategies.particle_sampler_cfg import ParticleSamplerCfg
from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints, RolloutMetrics, RolloutResult
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.sampling.sample_buffer import SampleBuffer


def quadratic_cost_fn(state):
    """Simple quadratic cost function for testing."""
    costs = torch.sum((state - 3.0) ** 2, dim=-1).unsqueeze(-1)
    return costs


class MockRolloutForParticleOpt:
    """Standalone mock rollout for particle optimizer testing (no base class)."""

    def __init__(
        self,
        num_dof: int = 3,
        action_horizon: int = 5,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.device_cfg = DeviceCfg(device=device, dtype=dtype)
        self.sum_horizon = True
        self.sampler_seed = 1
        self._action_dim = num_dof
        self._action_horizon = action_horizon
        self._tensor_args = self.device_cfg
        self._action_bound_lows = torch.ones(num_dof, device=device, dtype=dtype) * -5.0
        self._action_bound_highs = torch.ones(num_dof, device=device, dtype=dtype) * 5.0
        self._batch_size = None
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

    def get_initial_action(self, use_random=False, use_zero=False, **kwargs):
        return torch.zeros(
            1, self._action_horizon, self._action_dim, device=self._tensor_args.device, dtype=self._tensor_args.dtype
        )

    def evaluate_action(self, act_seq, **kwargs):
        batch_size = act_seq.shape[0]
        if self._batch_size != batch_size:
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

    def get_all_cost_components(self):
        return {}

    def filter_robot_state(self, state):
        return state

    def _compute_state_from_action_impl(self, act_seq):
        return act_seq

    def _compute_costs_and_constraints_impl(self, state, **kwargs):
        costs = quadratic_cost_fn(state)
        costs_collection = CostCollection(
            values=[costs], names=["cost"], weights=[1.0], sq_weights=[1.0]
        )

        return CostsAndConstraints(costs=costs_collection, constraints=None)


@pytest.fixture
def device():
    """Get device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def dtype():
    """Get dtype for testing."""
    return torch.float32


@pytest.fixture
def rollout_fn(device, dtype):
    """Create a mock rollout function."""
    return MockRolloutForParticleOpt(num_dof=3, action_horizon=5, device=device, dtype=dtype)


# Parameterize optimizer type
@pytest.fixture(params=["mppi", "es"])
def optimizer_type(request):
    """Parameterize optimizer type."""
    return request.param


@pytest.fixture
def optimizer_config(optimizer_type, device, dtype):
    """Create optimizer config based on type."""
    base_config = {
        "num_iters": 10,
        "solver_type": "particle",
        "device_cfg": DeviceCfg(device=device, dtype=dtype),
        "num_particles": 20,
        "sample_params": ParticleSamplerCfg(
            device_cfg=DeviceCfg(device=device, dtype=dtype)
        ),
    }

    if optimizer_type == "mppi":
        return MPPICfg(**base_config)
    else:  # es
        base_config["learning_rate"] = 0.1
        return EvolutionStrategiesCfg(**base_config)


@pytest.fixture
def optimizer_class(optimizer_type):
    """Get optimizer class based on type."""
    return MPPI if optimizer_type == "mppi" else EvolutionStrategies


class TestParallelOptimizerConfig:
    """Test cases for optimizer configuration."""

    def test_init_default_mppi(self, device, dtype):
        """Test default MPPI initialization."""
        cfg = MPPICfg(
            num_iters=10,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=10,
            sample_params=ParticleSamplerCfg(
                device_cfg=DeviceCfg(device=device, dtype=dtype)
            ),
        )

        assert cfg.num_iters == 10
        assert cfg.num_particles == 10
        assert cfg.base_action == BaseActionType.REPEAT
        assert cfg.step_size_mean == 0.9
        assert cfg.step_size_cov == 0.1
        assert cfg.squash_fn == SquashType.CLAMP
        assert cfg.cov_type == CovType.DIAG_A
        assert cfg.update_cov is True
        assert cfg.beta == 0.1
        assert cfg.alpha == 1.0
        assert cfg.kappa == 0.01

    def test_init_default_es(self, device, dtype):
        """Test default ES initialization."""
        cfg = EvolutionStrategiesCfg(
            num_iters=10,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=10,
            sample_params=ParticleSamplerCfg(
                device_cfg=DeviceCfg(device=device, dtype=dtype)
            ),
        )

        assert cfg.num_iters == 10
        assert cfg.num_particles == 10
        assert cfg.learning_rate == 0.1

    def test_init_custom_values(self, device, dtype):
        """Test initialization with custom values."""
        cfg = MPPICfg(
            num_iters=20,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=50,
            sample_params=ParticleSamplerCfg(
                device_cfg=DeviceCfg(device=device, dtype=dtype)
            ),
            base_action=BaseActionType.NULL,
            step_size_mean=0.5,
            step_size_cov=0.2,
            cov_type=CovType.SIGMA_I,
            beta=0.05,
            kappa=0.02,
        )

        assert cfg.base_action == BaseActionType.NULL
        assert cfg.step_size_mean == 0.5
        assert cfg.step_size_cov == 0.2
        assert cfg.cov_type == CovType.SIGMA_I
        assert cfg.beta == 0.05
        assert cfg.kappa == 0.02

    @pytest.mark.parametrize("cov_type", [CovType.SIGMA_I, CovType.DIAG_A])
    def test_cov_types(self, device, dtype, cov_type):
        """Test different covariance types."""
        cfg = MPPICfg(
            num_iters=10,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=10,
            sample_params=ParticleSamplerCfg(
                device_cfg=DeviceCfg(device=device, dtype=dtype)
            ),
            cov_type=cov_type,
        )
        assert cfg.cov_type == cov_type


class TestParallelOptimizers:
    """Test cases common to both MPPI and ES optimizers."""

    def test_init_basic(self, optimizer_class, optimizer_config, rollout_fn):
        """Test basic initialization."""
        optimizer = optimizer_class(optimizer_config, [rollout_fn])

        assert optimizer.config.num_iters == 10
        assert optimizer.config.num_particles == 20
        assert optimizer.mean_action is not None
        assert optimizer.cov_action is not None
        assert optimizer.scale_tril is not None

    @pytest.mark.parametrize("cov_type", [CovType.SIGMA_I, CovType.DIAG_A])
    def test_different_cov_types(self, optimizer_class, optimizer_type, device, dtype, rollout_fn, cov_type):
        """Test with different covariance types."""
        base_config = {
            "num_iters": 10,
            "solver_type": "particle",
            "device_cfg": DeviceCfg(device=device, dtype=dtype),
            "num_particles": 20,
            "sample_params": ParticleSamplerCfg(
                device_cfg=DeviceCfg(device=device, dtype=dtype)
            ),
            "cov_type": cov_type,
        }

        if optimizer_type == "es":
            base_config["learning_rate"] = 0.1
            cfg = EvolutionStrategiesCfg(**base_config)
        else:
            cfg = MPPICfg(**base_config)

        optimizer = optimizer_class(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        # Should initialize without errors
        assert optimizer.cov_action is not None
        assert optimizer.scale_tril is not None

    def test_optimize_converges(self, optimizer_class, optimizer_type, device, dtype, rollout_fn):
        """Test that optimization improves cost."""
        base_config = {
            "num_iters": 30 if optimizer_type == "es" else 20,  # ES needs more iterations
            "solver_type": "particle",
            "device_cfg": DeviceCfg(device=device, dtype=dtype),
            "num_particles": 50,
            "sample_params": ParticleSamplerCfg(
                device_cfg=DeviceCfg(device=device, dtype=dtype)
            ),
            "sample_mode": SampleMode.MEAN,
        }

        if optimizer_type == "es":
            base_config["learning_rate"] = 0.2  # Higher learning rate for ES
            base_config["cov_type"] = CovType.DIAG_A  # Now fixed!
            cfg = EvolutionStrategiesCfg(**base_config)
        else:
            cfg = MPPICfg(**base_config)

        optimizer = optimizer_class(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        # Start far from optimum (optimum is at 3.0)
        initial_action = torch.zeros(
            1, rollout_fn.action_horizon, rollout_fn.action_dim, device=device, dtype=dtype
        )

        result = optimizer.optimize(initial_action)

        # Check that result is closer to optimum (3.0) than initial (0.0)
        initial_cost = quadratic_cost_fn(initial_action).mean()
        final_cost = quadratic_cost_fn(result).mean()

        assert final_cost < initial_cost
        # Result should be closer to 3.0
        assert torch.abs(result.mean() - 3.0) < torch.abs(initial_action.mean() - 3.0)

    def test_update_num_problems(self, optimizer_class, optimizer_config, rollout_fn):
        """Test updating number of problems."""
        optimizer = optimizer_class(optimizer_config, [rollout_fn])
        optimizer.update_num_problems(3)

        assert optimizer.config.num_problems == 3
        assert optimizer.mean_action.shape[0] == 3
        assert optimizer.total_num_particles == 3 * 20

    def test_reset_distribution(self, optimizer_class, optimizer_config, device, rollout_fn):
        """Test reset distribution."""
        optimizer = optimizer_class(optimizer_config, [rollout_fn])
        optimizer.update_num_problems(1)

        # Modify the mean
        optimizer.mean_action[:] = 5.0

        # Reset should restore to initial mean
        optimizer.reset_distribution()

        # Should be reset
        assert torch.allclose(optimizer.mean_action, torch.zeros_like(optimizer.mean_action))

    def test_sample_actions(self, optimizer_class, optimizer_config, device, dtype, rollout_fn):
        """Test sampling actions."""
        optimizer = optimizer_class(optimizer_config, [rollout_fn])
        optimizer.update_num_problems(1)

        init_act = torch.zeros(
            1, rollout_fn.action_horizon, rollout_fn.action_dim, device=device, dtype=dtype
        )

        actions = optimizer.sample_actions(init_act)

        # Check shape
        assert actions.shape == (20, rollout_fn.action_horizon, rollout_fn.action_dim)
        # Actions should be within bounds
        assert torch.all(actions >= rollout_fn.action_bound_lows)
        assert torch.all(actions <= rollout_fn.action_bound_highs)

    @pytest.mark.parametrize("mode", [SampleMode.MEAN, SampleMode.BEST, SampleMode.SAMPLE])
    def test_sample_modes(self, optimizer_class, optimizer_type, device, dtype, rollout_fn, mode):
        """Test different sampling modes."""
        base_config = {
            "num_iters": 10,  # More iterations for stable results
            "solver_type": "particle",
            "device_cfg": DeviceCfg(device=device, dtype=dtype),
            "num_particles": 20,
            "sample_params": ParticleSamplerCfg(
                device_cfg=DeviceCfg(device=device, dtype=dtype)
            ),
            "sample_mode": mode,
            "cov_type": CovType.DIAG_A,  # Test with DIAG_A
        }

        if optimizer_type == "es":
            base_config["learning_rate"] = 0.1
            cfg = EvolutionStrategiesCfg(**base_config)
        else:
            cfg = MPPICfg(**base_config)

        optimizer = optimizer_class(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        initial_action = torch.zeros(
            1, rollout_fn.action_horizon, rollout_fn.action_dim, device=device, dtype=dtype
        )

        # Should not raise errors
        result = optimizer.optimize(initial_action)
        assert result.shape == initial_action.shape

    def test_reinitialize(self, optimizer_class, optimizer_config, device, dtype, rollout_fn):
        """Test optimizer reinitialize."""
        optimizer = optimizer_class(optimizer_config, [rollout_fn])
        optimizer.update_num_problems(1)

        # Modify the mean
        optimizer.mean_action[:] = 5.0

        # Create initial action for reinitialize
        initial_action = torch.zeros(
            1, rollout_fn.action_horizon, rollout_fn.action_dim, device=device, dtype=dtype
        )

        # Reinitialize should restore distribution
        optimizer.reinitialize(initial_action)

        # Mean should be reset
        assert torch.allclose(optimizer.mean_action, torch.zeros_like(optimizer.mean_action))

    def test_reset_seed(self, optimizer_class, optimizer_config, rollout_fn):
        """Test reset seed functionality."""
        optimizer = optimizer_class(optimizer_config, [rollout_fn])
        optimizer.update_num_problems(1)

        # Reset seed should not raise errors
        optimizer.reset_seed()

        # Sample library should be reset
        assert optimizer.sample_lib is not None

    def test_update_init_mean(self, optimizer_class, optimizer_config, device, dtype, rollout_fn):
        """Test updating initial mean."""
        optimizer = optimizer_class(optimizer_config, [rollout_fn])
        optimizer.update_num_problems(2)

        new_mean = torch.ones(2, rollout_fn.action_horizon, rollout_fn.action_dim, device=device, dtype=dtype) * 2.0
        optimizer.update_init_mean(new_mean)

        assert torch.allclose(optimizer.mean_action, new_mean)
        assert torch.allclose(optimizer.best_traj, new_mean)

    def test_properties(self, optimizer_class, optimizer_type, optimizer_config, rollout_fn):
        """Test optimizer properties."""
        optimizer = optimizer_class(optimizer_config, [rollout_fn])
        optimizer.update_num_problems(1)

        # Test various properties
        assert optimizer.full_scale_tril is not None
        assert optimizer.full_inv_cov is not None
        if optimizer_type == "mppi":
            assert optimizer.entropy is not None
            assert optimizer.squashed_mean is not None

    def test_multiple_problems(self, optimizer_class, optimizer_type, device, dtype, rollout_fn):
        """Test with multiple problems."""
        base_config = {
            "num_iters": 10,  # More iterations for stable results
            "solver_type": "particle",
            "device_cfg": DeviceCfg(device=device, dtype=dtype),
            "num_particles": 10,
            "sample_params": ParticleSamplerCfg(
                device_cfg=DeviceCfg(device=device, dtype=dtype)
            ),
            "cov_type": CovType.DIAG_A,  # Test with DIAG_A
        }

        if optimizer_type == "es":
            base_config["learning_rate"] = 0.1
            cfg = EvolutionStrategiesCfg(**base_config)
        else:
            cfg = MPPICfg(**base_config)

        optimizer = optimizer_class(cfg, [rollout_fn])
        optimizer.update_num_problems(3)

        initial_action = torch.zeros(
            3, rollout_fn.action_horizon, rollout_fn.action_dim, device=device, dtype=dtype
        )

        result = optimizer.optimize(initial_action)
        assert result.shape == (3, rollout_fn.action_horizon, rollout_fn.action_dim)


class TestMPPISpecific:
    """Test cases specific to MPPI optimizer."""

    def test_mppi_exp_util(self, device, dtype, rollout_fn):
        """Test MPPI exponential utility calculation."""
        cfg = MPPICfg(
            num_iters=10,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=20,
            sample_params=ParticleSamplerCfg(
                device_cfg=DeviceCfg(device=device, dtype=dtype)
            ),
        )

        optimizer = MPPI(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        costs = torch.randn(1, 20, 1, device=device, dtype=dtype).abs()
        w = optimizer._exp_util_from_costs(costs)

        # Weights should sum to 1
        assert torch.allclose(w.sum(dim=-1), torch.ones(1, device=device, dtype=dtype), atol=1e-5)


class TestESSpecific:
    """Test cases specific to ES optimizer."""

    def test_es_exp_util(self, device, dtype, rollout_fn):
        """Test ES exponential utility calculation (different from MPPI)."""
        cfg = EvolutionStrategiesCfg(
            num_iters=10,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=20,
            sample_params=ParticleSamplerCfg(
                device_cfg=DeviceCfg(device=device, dtype=dtype)
            ),
            learning_rate=0.1,
        )

        optimizer = EvolutionStrategies(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        costs = torch.randn(1, 20, 1, device=device, dtype=dtype).abs()
        w = optimizer._exp_util_from_costs(costs)

        # ES uses standardized weights (not softmax)
        # Mean should be close to 0, std close to 1
        assert torch.abs(w.mean()) < 1.0
        assert torch.abs(w.std() - 1.0) < 0.5

    def test_es_mean_computation(self, device, dtype, rollout_fn):
        """Test ES-specific mean computation."""
        cfg = EvolutionStrategiesCfg(
            num_iters=10,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=20,
            sample_params=ParticleSamplerCfg(
                device_cfg=DeviceCfg(device=device, dtype=dtype)
            ),
            learning_rate=0.1,
            cov_type=CovType.DIAG_A,
        )

        optimizer = EvolutionStrategies(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        # Test ES mean computation using normalized weights
        w = torch.randn(1, 20, 1, 1, device=device, dtype=dtype)
        # Normalize weights to simulate actual ES weights
        w = (w - w.mean()) / (w.std() + 1e-8)
        actions = torch.randn(1, 20, rollout_fn.action_horizon, rollout_fn.action_dim, device=device, dtype=dtype)

        # ES mean computation should not raise errors
        new_mean = optimizer._compute_mean(w, actions)

        assert new_mean.shape == (1, rollout_fn.action_horizon, rollout_fn.action_dim)

    def test_learning_rate(self, device, dtype, rollout_fn):
        """Test that learning rate is properly set."""
        cfg = EvolutionStrategiesCfg(
            num_iters=10,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=20,
            sample_params=ParticleSamplerCfg(
                device_cfg=DeviceCfg(device=device, dtype=dtype)
            ),
            learning_rate=0.2,
        )

        optimizer = EvolutionStrategies(cfg, [rollout_fn])

        assert hasattr(optimizer.config, 'learning_rate')
        assert optimizer.config.learning_rate == 0.2


class TestJITHelperFunctions:
    """Test JIT-compiled helper functions."""

    def test_jit_calculate_exp_util(self, device, dtype):
        """Test exponential utility calculation."""
        beta = 0.1
        costs = torch.tensor([[1.0, 2.0, 3.0]], device=device, dtype=dtype)

        weights = jit_calculate_exp_util(beta, costs)

        # Weights should sum to 1
        assert torch.allclose(weights.sum(dim=-1), torch.ones(1, device=device, dtype=dtype))
        # Lower cost should have higher weight
        assert weights[0, 0] > weights[0, 2]

    def test_jit_calculate_exp_util_from_costs(self, device, dtype):
        """Test exponential utility calculation from costs."""
        beta = 0.1
        costs = torch.randn(2, 10, 1, device=device, dtype=dtype).abs()
        gamma_seq = torch.ones(1, 1, device=device, dtype=dtype)

        weights = jit_calculate_exp_util_from_costs(costs, gamma_seq, beta)

        # Weights should sum to 1 for each problem
        assert torch.allclose(weights.sum(dim=-1), torch.ones(2, device=device, dtype=dtype))

    def test_jit_compute_total_cost(self, device, dtype):
        """Test total cost computation."""
        gamma_seq = torch.tensor([[1.0, 0.9, 0.81]], device=device, dtype=dtype)
        costs = torch.randn(2, 10, 3, device=device, dtype=dtype).abs()

        total_costs = jit_compute_total_cost(gamma_seq, costs)

        assert total_costs.shape == (2, 10)

    def test_jit_blend_mean(self, device, dtype):
        """Test mean blending."""
        old_mean = torch.ones(2, 5, 3, device=device, dtype=dtype)
        new_mean = torch.zeros(2, 5, 3, device=device, dtype=dtype)
        step_size = 0.5

        blended = jit_blend_mean(old_mean, new_mean, step_size)

        # Should be halfway between old and new
        assert torch.allclose(blended, torch.ones_like(blended) * 0.5)

    def test_jit_blend_cov(self, device, dtype):
        """Test covariance blending."""
        old_cov = torch.ones(2, 1, 3, device=device, dtype=dtype)
        new_cov = torch.zeros(2, 1, 3, device=device, dtype=dtype)
        step_size = 0.5
        kappa = 0.01

        blended = jit_blend_cov(old_cov, new_cov, step_size, kappa)

        # Should be blended with kappa added
        expected = torch.ones_like(blended) * 0.5 + kappa
        assert torch.allclose(blended, expected)

    def test_jit_diag_a_cov_update(self, device, dtype):
        """Test diagonal covariance update."""
        w = torch.randn(2, 10, 1, 1, device=device, dtype=dtype).abs()
        w = w / w.sum(dim=1, keepdim=True)  # Normalize
        actions = torch.randn(2, 10, 5, 3, device=device, dtype=dtype)
        mean_action = torch.randn(2, 5, 3, device=device, dtype=dtype)

        cov_update = jit_diag_a_cov_update(w, actions, mean_action)

        assert cov_update.shape == (2, 1, 3)
        assert torch.all(cov_update >= 0)  # Covariance should be non-negative

    def test_jit_mean_cov_diag_a(self, device, dtype):
        """Test combined mean and covariance update for diagonal case."""
        costs = torch.randn(2, 10, 1, device=device, dtype=dtype).abs()
        actions = torch.randn(2, 10, 5, 3, device=device, dtype=dtype)
        gamma_seq = torch.ones(1, 1, device=device, dtype=dtype)
        mean_action = torch.randn(2, 5, 3, device=device, dtype=dtype)
        cov_action = torch.ones(2, 1, 3, device=device, dtype=dtype)
        step_size_mean = 0.5
        step_size_cov = 0.5
        kappa = 0.01
        beta = 0.1

        new_mean, new_cov, new_tril = jit_mean_cov_diag_a(
            costs, actions, gamma_seq, mean_action, cov_action,
            step_size_mean, step_size_cov, kappa, beta
        )

        assert new_mean.shape == mean_action.shape
        assert new_cov.shape == cov_action.shape
        assert new_tril.shape == cov_action.shape
        assert torch.all(new_cov >= 0)  # Covariance should be non-negative

    def test_calc_exp(self, device, dtype):
        """Test ES calc_exp function."""
        total_costs = torch.randn(2, 10, device=device, dtype=dtype)

        w = calc_exp(total_costs)

        # Should be standardized (mean ~0, std ~1)
        assert torch.abs(w.mean(dim=-1)).max() < 1e-5
        assert torch.allclose(w.std(dim=-1), torch.ones(2, device=device, dtype=dtype), atol=1e-5)

    def test_compute_es_mean(self, device, dtype):
        """Test ES mean computation function."""
        w = torch.randn(2, 20, 1, 1, device=device, dtype=dtype)
        # Ensure weights have proper statistics for ES (standardized)
        w = (w - w.mean(dim=1, keepdim=True)) / (w.std(dim=1, keepdim=True) + 1e-8)
        actions = torch.randn(2, 20, 5, 3, device=device, dtype=dtype)
        mean_action = torch.randn(2, 5, 3, device=device, dtype=dtype)
        # full_inv_cov should be [num_problems, action_dim, action_dim] after fix
        full_inv_cov = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(2, -1, -1)
        num_particles = 20
        learning_rate = 0.1

        new_mean = compute_es_mean(w, actions, mean_action, full_inv_cov, num_particles, learning_rate)

        assert new_mean.shape == mean_action.shape

