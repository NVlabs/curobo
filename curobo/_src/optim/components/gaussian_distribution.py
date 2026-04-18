# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gaussian distribution state and sampling infrastructure for particle optimizers.

Manages the mean, covariance, scale_tril, sample library, and pre-generated noise
shared by MPPI and Evolution Strategies. Weighting strategies (softmax vs. z-score)
remain in each optimizer.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

import torch
import torch.autograd.profiler as profiler

from curobo._src.optim.particle.sample_strategies.particle_sampler import MixedParticleSampler
from curobo._src.optim.particle.sample_strategies.particle_sampler_cfg import ParticleSamplerCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise
from curobo._src.util.tensor_util import copy_tensor
from curobo._src.util.torch_util import get_torch_jit_decorator


class CovType(Enum):
    """Covariance parameterization used by particle optimizers.

    Controls how the Gaussian distribution's covariance matrix is stored and
    updated during optimization.
    """

    SIGMA_I = "SIGMA_I"
    """Scalar covariance: cov = sigma * I, shared across all action dimensions."""

    DIAG_A = "DIAG_A"
    """Diagonal covariance: cov = diag(a_1, ..., a_d), per-dimension variance."""


class GaussianDistribution:
    """Manages Gaussian distribution state and sampling infrastructure for particle optimizers.

    Handles: mean, covariance, scale_tril, sample library, pre-generated noise.
    Does NOT handle: weighting, cost computation, action bounds (those stay in the optimizer).
    """

    def __init__(
        self,
        device_cfg: DeviceCfg,
        action_horizon: int,
        action_dim: int,
        cov_type: CovType,
        init_mean: torch.Tensor,
        init_cov: torch.Tensor,
        sample_params: ParticleSamplerCfg,
        random_mean: bool = False,
        seed: int = 0,
    ):
        self.device_cfg = device_cfg
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.cov_type = cov_type
        self.init_mean = init_mean
        self.init_cov = init_cov
        self.random_mean = random_mean
        self.seed = seed

        # Distribution state
        self.mean: Optional[torch.Tensor] = None
        self.cov: Optional[torch.Tensor] = None
        self.scale_tril: Optional[torch.Tensor] = None
        self.inv_cov: Optional[torch.Tensor] = None
        self.best_traj: Optional[torch.Tensor] = None

        # Sample library
        self.sample_lib = MixedParticleSampler(sample_params, action_horizon, action_dim)
        self.mean_lib = None
        if random_mean:
            self.mean_lib = MixedParticleSampler(
                ParticleSamplerCfg(
                    device_cfg=device_cfg,
                    fixed_samples=False,
                    seed=2567,
                    filter_coeffs=None,
                ),
                action_horizon,
                action_dim,
            )

        # Pre-generated sample sets
        self._sample_set: Optional[torch.Tensor] = None
        self._sample_iter: Optional[torch.Tensor] = None

        # Identity matrix for covariance operations
        self.I = torch.eye(action_dim, device=device_cfg.device, dtype=device_cfg.dtype)

    def reset_mean(self, num_problems: int, reset_problem_ids: Optional[torch.Tensor] = None):
        """Reset mean to initial value."""
        if reset_problem_ids is None:
            if self.random_mean:
                mean = self.mean_lib.get_samples([num_problems])
                self.update_mean(mean, num_problems)
            else:
                self.update_mean(self.init_mean, num_problems)
        else:
            if self.random_mean:
                new_mean = self.mean_lib.get_samples([reset_problem_ids.shape[0]])
            else:
                new_mean = self.init_mean
            current_mean = self.mean.clone()
            current_mean[reset_problem_ids, :, :] = new_mean
            self.update_mean(current_mean, num_problems)

    def reset_covariance(self, num_problems: int):
        """Reset covariance to initial value."""
        if self.cov_type == CovType.SIGMA_I:
            init_cov = self.init_cov.clone()
            if len(init_cov.shape) == 1:
                init_cov = init_cov.unsqueeze(0)
            if init_cov.shape[0] != num_problems:
                init_cov = init_cov.expand(num_problems, -1)
            if not copy_tensor(init_cov, self.cov):
                self.cov = init_cov.clone()
            self.inv_cov = 1.0 / self.cov
            a = torch.sqrt(self.cov)
            if not copy_tensor(a, self.scale_tril):
                self.scale_tril = a

        elif self.cov_type == CovType.DIAG_A:
            init_cov = self.init_cov.clone()
            if len(init_cov.shape) == 1:
                init_cov = init_cov.unsqueeze(-1).expand(-1, self.action_dim)
            if len(init_cov.shape) == 2 and init_cov.shape[-1] != self.action_dim:
                init_cov = init_cov.expand(-1, self.action_dim)
            init_cov = init_cov.unsqueeze(1)
            if init_cov.shape[0] != num_problems:
                init_cov = init_cov.repeat(num_problems, 1, 1)
            if not copy_tensor(init_cov.clone(), self.cov):
                self.cov = init_cov.clone()
            self.inv_cov = 1.0 / self.cov
            a = torch.sqrt(self.cov)
            if not copy_tensor(a, self.scale_tril):
                self.scale_tril = a
        else:
            log_and_raise(f"Unidentified covariance type: {self.cov_type}")

    def reset(self, num_problems: int, reset_problem_ids: Optional[torch.Tensor] = None):
        """Reset both mean and covariance."""
        self.reset_mean(num_problems, reset_problem_ids)
        self.reset_covariance(num_problems)

    def update_mean(self, new_mean: torch.Tensor, num_problems: int):
        """Update the distribution mean."""
        if new_mean.shape[0] != num_problems:
            new_mean = new_mean.expand(num_problems, -1, -1)
        if not copy_tensor(new_mean, self.mean):
            self.mean = new_mean.clone()
        if not copy_tensor(new_mean, self.best_traj):
            self.best_traj = new_mean.clone()

    def update_cov_scale(self, new_cov: torch.Tensor):
        """Update scale_tril from new covariance."""
        if self.cov_type in (CovType.SIGMA_I, CovType.DIAG_A):
            self.scale_tril.copy_(torch.sqrt(new_cov))
        else:
            log_and_raise(f"Unidentified covariance type: {self.cov_type}")

    def initialize_samples(
        self,
        num_problems: int,
        sampled_particles_per_problem: int,
        num_iters: int,
        fixed_samples: bool,
        sample_per_problem: bool,
    ):
        """Allocate pre-generated sample sets."""
        iters = 1 if fixed_samples else num_iters
        if sample_per_problem:
            s_set = (
                self.sample_lib.get_samples(
                    sample_shape=[sampled_particles_per_problem * num_problems * iters],
                    base_seed=self.seed,
                )
                .view(iters, num_problems, sampled_particles_per_problem,
                      self.action_horizon, self.action_dim)
                .clone()
            )
        else:
            s_set = self.sample_lib.get_samples(
                sample_shape=[iters * sampled_particles_per_problem],
                base_seed=self.seed,
            )
            s_set = s_set.view(
                iters, 1, sampled_particles_per_problem,
                self.action_horizon, self.action_dim,
            )
            s_set = s_set.repeat(1, num_problems, 1, 1, 1).clone()
        s_set[:, :, -1, :, :] = 0.0
        self._sample_set = s_set
        self._sample_iter = torch.zeros((1,), dtype=torch.long, device=self.device_cfg.device)

    def update_samples(
        self,
        num_problems: int,
        sampled_particles_per_problem: int,
        num_iters: int,
        fixed_samples: bool,
        sample_per_problem: bool,
    ):
        """Regenerate pre-generated samples (e.g., after seed reset)."""
        with profiler.record_function("gaussian_dist/update_samples"):
            iters = 1 if fixed_samples else num_iters
            if sample_per_problem:
                s_set = self.sample_lib.get_samples(
                    sample_shape=[sampled_particles_per_problem * num_problems * iters],
                    base_seed=self.seed,
                ).view(
                    iters, num_problems, sampled_particles_per_problem,
                    self.action_horizon, self.action_dim,
                )
            else:
                s_set = self.sample_lib.get_samples(
                    sample_shape=[iters * sampled_particles_per_problem],
                    base_seed=self.seed,
                )
                s_set = s_set.view(
                    iters, 1, sampled_particles_per_problem,
                    self.action_horizon, self.action_dim,
                )
                s_set = s_set.repeat(1, num_problems, 1, 1, 1)
            if self._sample_set is None:
                log_and_raise("sample set is None")
            if self._sample_set.shape != s_set.shape:
                log_and_raise("sample set shape mismatch")
            self._sample_set.copy_(s_set)
            self._sample_set[:, :, -1, :, :] = 0.0
            self._sample_iter[:] = 0

    def get_samples(self, num_iters: int, fixed_samples: bool) -> torch.Tensor:
        """Get noise samples for current iteration.

        Returns:
            Noise tensor [num_problems, sampled_particles_per_problem, action_horizon, action_dim].
        """
        if fixed_samples:
            return self._sample_set[0]
        delta = torch.index_select(self._sample_set, 0, self._sample_iter).squeeze(0)
        self._sample_iter[:] += 1
        self._sample_iter[:] = torch.where(
            self._sample_iter >= num_iters,
            torch.zeros_like(self._sample_iter),
            self._sample_iter,
        )
        return delta

    def generate_noise(self, shape, base_seed=None) -> torch.Tensor:
        """Generate correlated noisy samples."""
        return self.sample_lib.get_samples(sample_shape=shape, seed=base_seed)

    @property
    def full_scale_tril(self) -> torch.Tensor:
        """Scale tril expanded to [num_problems, 1, action_horizon, action_dim]."""
        if self.cov_type == CovType.SIGMA_I:
            return self.scale_tril.unsqueeze(-2).unsqueeze(-2).expand(
                -1, -1, self.action_horizon, -1
            )
        elif self.cov_type == CovType.DIAG_A:
            return self.scale_tril.unsqueeze(-2).expand(-1, -1, self.action_horizon, -1)
        else:
            log_and_raise(f"Unidentified covariance type: {self.cov_type}")

    @property
    def full_inv_cov(self) -> torch.Tensor:
        """Inverse covariance as full matrix [num_problems, action_dim, action_dim]."""
        if self.cov_type == CovType.SIGMA_I:
            return self.inv_cov * self.I
        elif self.cov_type == CovType.DIAG_A:
            if len(self.inv_cov.shape) == 3:
                return torch.diag_embed(self.inv_cov.squeeze(1))
            else:
                log_and_raise("full_inv_cov: Unexpected inv_cov shape for DIAG_A")
                return torch.diag_embed(self.inv_cov)
        else:
            log_and_raise(f"Unidentified covariance type: {self.cov_type}")

    def shift(self, shift_steps: int, repeat_last: bool):
        """Shift mean and best_traj for MPC warm start."""
        if shift_steps == 0:
            return
        if self.mean is not None:
            self.mean = _jit_shift_action_buffer(self.mean, shift_steps, repeat_last)
        if self.best_traj is not None:
            self.best_traj = _jit_shift_action_buffer(self.best_traj, shift_steps, repeat_last)

    def reset_seed(self):
        """Reset sample library seeds."""
        self.sample_lib.reset_seed()
        if self.mean_lib is not None:
            self.mean_lib.reset_seed()


@get_torch_jit_decorator()
def _jit_shift_action_buffer(
    buffer: torch.Tensor, shift_steps: int, repeat_last: bool
) -> torch.Tensor:
    """Shift action buffer left by shift_steps, filling the end."""
    shifted = torch.roll(buffer, -shift_steps, dims=-2)
    if repeat_last:
        shifted[..., -shift_steps:, :] = shifted[..., -shift_steps - 1 : -shift_steps, :]
    else:
        shifted[..., -shift_steps:, :] = 0.0
    return shifted
