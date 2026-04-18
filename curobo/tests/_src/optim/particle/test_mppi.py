#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for MPPI-specific functionality to improve coverage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.optim.particle.mppi import (
    MPPI,
    CovType,
    MPPICfg,
    SampleMode,
)
from curobo._src.optim.particle.sample_strategies.particle_sampler_cfg import ParticleSamplerCfg
from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints, RolloutMetrics, RolloutResult
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.sampling.sample_buffer import SampleBuffer


def quadratic_cost_fn(state):
    """Simple quadratic cost function for testing."""
    costs = torch.sum((state - 3.0) ** 2, dim=-1).unsqueeze(-1)
    return costs


@dataclass
class MockState:
    """Mock state object for testing."""

    position: torch.Tensor
    ee_position: Optional[torch.Tensor] = None

    def clone(self):
        """Clone the state object."""
        return MockState(
            position=self.position.clone(),
            ee_position=self.ee_position.clone() if self.ee_position is not None else None,
        )


class MockRolloutForMPPI:
    """Standalone mock rollout for MPPI testing (no base class)."""

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
            1,
            self._action_horizon,
            self._action_dim,
            device=self._tensor_args.device,
            dtype=self._tensor_args.dtype,
        )

    def evaluate_action(self, act_seq, **kwargs):
        batch_size = act_seq.shape[0]
        if self._batch_size != batch_size:
            self.update_batch_size(batch_size)
        state = self._compute_state_from_action_impl(act_seq)
        cc = self._compute_costs_and_constraints_impl(state, **kwargs)
        return RolloutResult(actions=act_seq, state=state, costs_and_constraints=cc)

    def compute_metrics_from_state(self, state, **kwargs):
        if isinstance(state, MockState):
            cc = self._compute_costs_and_constraints_impl(state, **kwargs)
        else:
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
        state = MockState(
            position=act_seq,
            ee_position=act_seq,
        )
        return state

    def _compute_costs_and_constraints_impl(self, state, **kwargs):
        if isinstance(state, MockState):
            costs = quadratic_cost_fn(state.position)
        else:
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
    return MockRolloutForMPPI(num_dof=3, action_horizon=5, device=device, dtype=dtype)


class TestMPPIConfigWithInitMean:
    """Test cases for MPPI config with init_mean provided (line 81)."""

    def test_init_with_init_mean(self, device, dtype):
        """Test MPPI config initialization with provided init_mean."""
        init_mean = torch.ones(1, 5, 3, device=device, dtype=dtype) * 2.0
        cfg = MPPICfg(
            num_iters=10,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=10,
            sample_params=ParticleSamplerCfg(device_cfg=DeviceCfg(device=device, dtype=dtype)),
            init_mean=init_mean,
        )

        assert cfg.init_mean is not None
        assert torch.allclose(cfg.init_mean, init_mean)


class TestMPPICovarianceTypes:
    """Test cases for different covariance types."""


    def test_sigma_i_compute_covariance(self, device, dtype, rollout_fn):
        """Test _compute_covariance with SIGMA_I type (lines 244-247)."""
        cfg = MPPICfg(
            num_iters=5,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=20,
            sample_params=ParticleSamplerCfg(device_cfg=DeviceCfg(device=device, dtype=dtype)),
            cov_type=CovType.SIGMA_I,
            update_cov=True,
        )

        optimizer = MPPI(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        # Create mock weights and actions
        w = torch.ones(1, 20, 1, 1, device=device, dtype=dtype) / 20
        actions = torch.randn(
            1, 20, rollout_fn.action_horizon, rollout_fn.action_dim, device=device, dtype=dtype
        )

        cov_update = optimizer._compute_covariance(w, actions)
        assert cov_update is not None

    def test_sigma_i_update_cov_scale(self, device, dtype, rollout_fn):
        """Test _update_cov_scale with SIGMA_I type (lines 278-279)."""
        cfg = MPPICfg(
            num_iters=5,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=10,
            sample_params=ParticleSamplerCfg(device_cfg=DeviceCfg(device=device, dtype=dtype)),
            cov_type=CovType.SIGMA_I,
            update_cov=True,
        )

        optimizer = MPPI(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        # Call _update_cov_scale with None (uses self.cov_action)
        optimizer._update_cov_scale(None)
        assert optimizer.scale_tril is not None

    def test_sigma_i_full_scale_tril(self, device, dtype, rollout_fn):
        """Test full_scale_tril property with SIGMA_I type (line 484)."""
        cfg = MPPICfg(
            num_iters=5,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=10,
            sample_params=ParticleSamplerCfg(device_cfg=DeviceCfg(device=device, dtype=dtype)),
            cov_type=CovType.SIGMA_I,
        )

        optimizer = MPPI(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        full_tril = optimizer.full_scale_tril
        assert full_tril is not None

    def test_sigma_i_full_inv_cov(self, device, dtype, rollout_fn):
        """Test full_inv_cov property with SIGMA_I type (line 564)."""
        cfg = MPPICfg(
            num_iters=5,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=10,
            sample_params=ParticleSamplerCfg(device_cfg=DeviceCfg(device=device, dtype=dtype)),
            cov_type=CovType.SIGMA_I,
        )

        optimizer = MPPI(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        inv_cov = optimizer.full_inv_cov
        assert inv_cov is not None


class TestMPPIUpdateCovDisabled:
    """Test cases when update_cov is False (lines 240, 276-277, 327-329)."""

    def test_no_cov_update_compute_covariance(self, device, dtype, rollout_fn):
        """Test _compute_covariance returns None when update_cov=False (line 240)."""
        cfg = MPPICfg(
            num_iters=5,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=20,
            sample_params=ParticleSamplerCfg(device_cfg=DeviceCfg(device=device, dtype=dtype)),
            cov_type=CovType.DIAG_A,
            update_cov=False,
        )

        optimizer = MPPI(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        w = torch.ones(1, 20, 1, 1, device=device, dtype=dtype) / 20
        actions = torch.randn(
            1, 20, rollout_fn.action_horizon, rollout_fn.action_dim, device=device, dtype=dtype
        )

        result = optimizer._compute_covariance(w, actions)
        assert result is None

    def test_no_cov_update_scale(self, device, dtype, rollout_fn):
        """Test _update_cov_scale returns early when update_cov=False (lines 276-277)."""
        cfg = MPPICfg(
            num_iters=5,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=10,
            sample_params=ParticleSamplerCfg(device_cfg=DeviceCfg(device=device, dtype=dtype)),
            cov_type=CovType.DIAG_A,
            update_cov=False,
        )

        optimizer = MPPI(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        # This should return early and not update scale_tril
        optimizer._update_cov_scale(None)

    def test_no_cov_update_in_distribution(self, device, dtype, rollout_fn):
        """Test _update_distribution when update_cov=False (lines 327-329)."""
        cfg = MPPICfg(
            num_iters=10,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=20,
            sample_params=ParticleSamplerCfg(device_cfg=DeviceCfg(device=device, dtype=dtype)),
            update_cov=False,
        )

        optimizer = MPPI(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        init_act = torch.zeros(
            1, rollout_fn.action_horizon, rollout_fn.action_dim, device=device, dtype=dtype
        )
        # Run optimization - this will trigger _update_distribution
        result = optimizer.optimize(init_act)
        assert result is not None


class TestMPPIResetWithProblemIds:
    """Test cases for reset_mean with reset_problem_ids (lines 405-413)."""

    def test_reset_mean_with_problem_ids(self, device, dtype, rollout_fn):
        """Test reset_mean with specific problem ids (lines 408-413)."""
        cfg = MPPICfg(
            num_iters=5,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=10,
            sample_params=ParticleSamplerCfg(device_cfg=DeviceCfg(device=device, dtype=dtype)),
            random_mean=False,
        )

        optimizer = MPPI(cfg, [rollout_fn])
        optimizer.update_num_problems(3)

        # Modify mean actions
        optimizer.mean_action[:] = 5.0

        # Reset only problem 1
        reset_ids = torch.tensor([1], device=device, dtype=torch.long)
        optimizer.reset_mean(reset_problem_ids=reset_ids)

        # Problem 1 should be reset, others should stay at 5.0
        assert torch.allclose(
            optimizer.mean_action[1], torch.zeros_like(optimizer.mean_action[1])
        )


class TestMPPISamplePerProblem:
    """Test cases for sample_per_problem=False (lines 616-627, 655-666)."""

    def test_sample_per_problem_false(self, device, dtype, rollout_fn):
        """Test MPPI with sample_per_problem=False (lines 616-627)."""
        cfg = MPPICfg(
            num_iters=5,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=10,
            sample_params=ParticleSamplerCfg(device_cfg=DeviceCfg(device=device, dtype=dtype)),
            sample_per_problem=False,
        )

        optimizer = MPPI(cfg, [rollout_fn])
        optimizer.update_num_problems(2)

        # Samples should be shared across problems
        assert optimizer._sample_set is not None

    def test_update_samples_non_fixed(self, device, dtype, rollout_fn):
        """Test update_samples with non-fixed samples (line 637)."""
        cfg = MPPICfg(
            num_iters=5,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=10,
            sample_params=ParticleSamplerCfg(
                device_cfg=DeviceCfg(device=device, dtype=dtype),
                fixed_samples=False,
            ),
            sample_per_problem=False,
        )

        optimizer = MPPI(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        # update_samples should use num_iters when fixed_samples=False
        optimizer.update_samples()
        assert optimizer._sample_set is not None


class TestMPPINullActFrac:
    """Test cases for null_act_frac which creates neg and null particles (lines 352-356)."""

    def test_null_act_frac(self, device, dtype, rollout_fn):
        """Test MPPI with null_act_frac > 0 (lines 352-356)."""
        cfg = MPPICfg(
            num_iters=5,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=20,  # Need more particles for null_act_frac to create neg/null
            sample_params=ParticleSamplerCfg(device_cfg=DeviceCfg(device=device, dtype=dtype)),
            null_act_frac=0.2,  # 20% of particles will be null/neg
        )

        optimizer = MPPI(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        # Should have some null and neg particles
        assert optimizer.null_per_problem >= 0
        assert optimizer.neg_per_problem >= 0

        init_act = torch.zeros(
            1, rollout_fn.action_horizon, rollout_fn.action_dim, device=device, dtype=dtype
        )
        actions = optimizer.sample_actions(init_act)
        assert actions is not None


class TestMPPIGetRollouts:
    """Test cases for get_rollouts method (line 180)."""

    def test_get_rollouts(self, device, dtype, rollout_fn):
        """Test get_rollouts method (line 180)."""
        cfg = MPPICfg(
            num_iters=5,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=10,
            sample_params=ParticleSamplerCfg(device_cfg=DeviceCfg(device=device, dtype=dtype)),
        )

        optimizer = MPPI(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        # Initially top_trajs is None
        result = optimizer.get_rollouts()
        assert result is None


class TestMPPIUpdateSeed:
    """Test cases for update_seed method (line 380)."""

    def test_update_seed_with_correct_shape(self, device, dtype, rollout_fn):
        """Test update_seed with correct shape."""
        cfg = MPPICfg(
            num_iters=5,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=10,
            sample_params=ParticleSamplerCfg(device_cfg=DeviceCfg(device=device, dtype=dtype)),
        )

        optimizer = MPPI(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        new_seed = torch.ones(
            1, rollout_fn.action_horizon, rollout_fn.action_dim, device=device, dtype=dtype
        )
        optimizer.update_seed(new_seed)

        assert torch.allclose(optimizer.mean_action, new_seed)

    def test_update_seed_with_wrong_shape_raises(self, device, dtype, rollout_fn):
        """Test update_seed with wrong shape raises error (line 380)."""
        cfg = MPPICfg(
            num_iters=5,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=10,
            sample_params=ParticleSamplerCfg(device_cfg=DeviceCfg(device=device, dtype=dtype)),
        )

        optimizer = MPPI(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        # 4D tensor should raise error
        wrong_seed = torch.ones(
            1, 1, rollout_fn.action_horizon, rollout_fn.action_dim, device=device, dtype=dtype
        )

        with pytest.raises(ValueError):
            optimizer.update_seed(wrong_seed)


class TestMPPIGetActionSeq:
    """Test cases for _get_action_seq method (lines 456-465)."""

    def test_get_action_seq_mean_mode(self, device, dtype, rollout_fn):
        """Test _get_action_seq with MEAN mode (line 455)."""
        cfg = MPPICfg(
            num_iters=5,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=10,
            sample_params=ParticleSamplerCfg(device_cfg=DeviceCfg(device=device, dtype=dtype)),
        )

        optimizer = MPPI(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        act_seq = optimizer._get_action_seq(SampleMode.MEAN)
        assert act_seq is not None
        assert torch.allclose(act_seq, optimizer.mean_action)

    def test_get_action_seq_best_mode(self, device, dtype, rollout_fn):
        """Test _get_action_seq with BEST mode (lines 462-463)."""
        cfg = MPPICfg(
            num_iters=5,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=10,
            sample_params=ParticleSamplerCfg(device_cfg=DeviceCfg(device=device, dtype=dtype)),
        )

        optimizer = MPPI(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        act_seq = optimizer._get_action_seq(SampleMode.BEST)
        assert act_seq is not None
        assert torch.allclose(act_seq, optimizer.best_traj)


class TestMPPIGenerateNoise:
    """Test cases for generate_noise method (lines 473-474)."""

    def test_generate_noise(self, device, dtype, rollout_fn):
        """Test generate_noise method (lines 473-474)."""
        cfg = MPPICfg(
            num_iters=5,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=10,
            sample_params=ParticleSamplerCfg(device_cfg=DeviceCfg(device=device, dtype=dtype)),
        )

        optimizer = MPPI(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        noise = optimizer.generate_noise(
            shape=torch.Size((10, rollout_fn.action_horizon)), base_seed=42
        )

        assert noise is not None
        assert noise.shape[0] == 10


class TestMPPIDirectMethodCalls:
    """Test cases for directly calling internal methods to hit coverage."""

    def test_exp_util(self, device, dtype, rollout_fn):
        """Test _exp_util method directly (lines 196-198)."""
        cfg = MPPICfg(
            num_iters=5,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=20,
            sample_params=ParticleSamplerCfg(device_cfg=DeviceCfg(device=device, dtype=dtype)),
        )

        optimizer = MPPI(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        total_costs = torch.randn(1, 20, device=device, dtype=dtype).abs()
        w = optimizer._exp_util(total_costs)

        assert w is not None
        assert torch.allclose(w.sum(dim=-1), torch.ones(1, device=device, dtype=dtype), atol=1e-5)

    def test_compute_mean(self, device, dtype, rollout_fn):
        """Test _compute_mean method directly (lines 206-208)."""
        cfg = MPPICfg(
            num_iters=5,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=20,
            sample_params=ParticleSamplerCfg(device_cfg=DeviceCfg(device=device, dtype=dtype)),
        )

        optimizer = MPPI(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        w = torch.ones(1, 20, 1, 1, device=device, dtype=dtype) / 20
        actions = torch.randn(
            1, 20, rollout_fn.action_horizon, rollout_fn.action_dim, device=device, dtype=dtype
        )

        new_mean = optimizer._compute_mean(w, actions)
        assert new_mean.shape == optimizer.mean_action.shape

    def test_compute_mean_covariance_sigma_i(self, device, dtype, rollout_fn):
        """Test _compute_mean_covariance with SIGMA_I (lines 230-234)."""
        cfg = MPPICfg(
            num_iters=5,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=20,
            sample_params=ParticleSamplerCfg(device_cfg=DeviceCfg(device=device, dtype=dtype)),
            cov_type=CovType.SIGMA_I,
            update_cov=True,
        )

        optimizer = MPPI(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        costs = torch.randn(1, 20, 1, device=device, dtype=dtype).abs()
        actions = torch.randn(
            1, 20, rollout_fn.action_horizon, rollout_fn.action_dim, device=device, dtype=dtype
        )

        new_mean, new_cov = optimizer._compute_mean_covariance(costs, actions)
        assert new_mean is not None
        assert new_cov is not None


class TestMPPISampleIteration:
    """Test cases for sample iteration (lines 341-343)."""

    def test_sample_iteration_increment(self, device, dtype, rollout_fn):
        """Test sample iteration incrementing (lines 341-343)."""
        cfg = MPPICfg(
            num_iters=3,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=10,
            sample_params=ParticleSamplerCfg(
                device_cfg=DeviceCfg(device=device, dtype=dtype),
                fixed_samples=False,
            ),
        )

        optimizer = MPPI(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        init_act = torch.zeros(
            1, rollout_fn.action_horizon, rollout_fn.action_dim, device=device, dtype=dtype
        )

        # Sample multiple times to trigger iteration
        for _ in range(5):
            optimizer.sample_actions(init_act)

        # Should have cycled through iterations
        assert optimizer._sample_iter is not None


class TestMPPISampleModeBest:
    """Test SampleMode.BEST during optimization (lines 305-307)."""

    def test_sample_mode_best_optimization(self, device, dtype, rollout_fn):
        """Test optimization with SampleMode.BEST (lines 305-307)."""
        cfg = MPPICfg(
            num_iters=10,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=20,
            sample_params=ParticleSamplerCfg(device_cfg=DeviceCfg(device=device, dtype=dtype)),
            sample_mode=SampleMode.BEST,
        )

        optimizer = MPPI(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        init_act = torch.zeros(
            1, rollout_fn.action_horizon, rollout_fn.action_dim, device=device, dtype=dtype
        )

        result = optimizer.optimize(init_act)
        assert result is not None
        # Best trajectory should be updated
        assert optimizer.best_traj is not None


class TestMPPIResetCovariance:
    """Test reset_covariance with various conditions."""

    def test_sigma_i_expand(self, device, dtype, rollout_fn):
        """Test reset_covariance SIGMA_I expand path (line 423)."""
        cfg = MPPICfg(
            num_iters=5,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=10,
            sample_params=ParticleSamplerCfg(device_cfg=DeviceCfg(device=device, dtype=dtype)),
            cov_type=CovType.SIGMA_I,
            init_cov=0.5,
        )

        optimizer = MPPI(cfg, [rollout_fn])
        # Update to multiple problems to trigger expand
        optimizer.update_num_problems(3)

        assert optimizer.cov_action.shape[0] == 3

    def test_diag_a_expand(self, device, dtype, rollout_fn):
        """Test reset_covariance DIAG_A expand path (line 439)."""
        cfg = MPPICfg(
            num_iters=5,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=10,
            sample_params=ParticleSamplerCfg(device_cfg=DeviceCfg(device=device, dtype=dtype)),
            cov_type=CovType.DIAG_A,
            init_cov=0.5,
        )

        optimizer = MPPI(cfg, [rollout_fn])
        optimizer.update_num_problems(3)

        assert optimizer.cov_action.shape[0] == 3
        assert optimizer.cov_action.shape[-1] == rollout_fn.action_dim


class TestMPPIConfigCreateDataDict:
    """Test MPPICfg.create_data_dict static method."""

    def test_create_data_dict_minimal(self, device, dtype):
        """Test create_data_dict with minimal config."""
        device_cfg = DeviceCfg(device=device, dtype=dtype)
        data_dict = {
            "num_iters": 10,
            "solver_type": "particle",
            "num_particles": 20,
        }

        result = MPPICfg.create_data_dict(data_dict, device_cfg)



class TestMPPIComputeTotalCost:
    """Test _compute_total_cost method."""

    def test_compute_total_cost(self, device, dtype, rollout_fn):
        """Test _compute_total_cost method (line 192)."""
        cfg = MPPICfg(
            num_iters=5,
            solver_type="particle",
            device_cfg=DeviceCfg(device=device, dtype=dtype),
            num_particles=10,
            sample_params=ParticleSamplerCfg(device_cfg=DeviceCfg(device=device, dtype=dtype)),
        )

        optimizer = MPPI(cfg, [rollout_fn])
        optimizer.update_num_problems(1)

        costs = torch.randn(1, 10, 5, device=device, dtype=dtype).abs()
        total_cost = optimizer._compute_total_cost(costs)

        assert total_cost is not None
        assert total_cost.shape == (1, 10)
