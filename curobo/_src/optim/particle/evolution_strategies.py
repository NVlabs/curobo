# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Evolution Strategies (ES) optimizer for sampling-based trajectory optimization.

Draws action samples from a Gaussian distribution via ParticleOptCore, evaluates
costs through rollouts, and updates the distribution using z-score weighting and
natural gradient mean updates instead of the softmax weighting used by MPPI.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional

import torch
import torch.autograd.profiler as profiler

from curobo._src.optim.components.gaussian_distribution import CovType
from curobo._src.optim.components.particle_opt_core import ParticleOptCore
from curobo._src.optim.particle.mppi import (
    MPPICfg,
    jit_blend_cov,
    jit_blend_mean,
    jit_compute_total_cost,
    jit_diag_a_cov_update,
)
from curobo._src.optim.components.particle_opt_core import SampleMode
from curobo._src.optim.particle.particle_opt_utils import (
    SquashType,
    gaussian_entropy,
    scale_ctrl,
)
from curobo._src.rollout.metrics import RolloutResult
from curobo._src.rollout.rollout_protocol import Rollout
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise
from curobo._src.util.tensor_util import stable_topk
from curobo._src.util.torch_util import get_torch_jit_decorator


@dataclass
class EvolutionStrategiesCfg(MPPICfg):
    """ES configuration. Extends MPPICfg with learning_rate."""

    #: Step size for the natural gradient mean update. Controls how far the
    #: distribution mean moves along the estimated natural gradient direction
    #: each iteration.
    learning_rate: float = 0.1


class EvolutionStrategies:
    """Evolution Strategies optimizer using z-score weighting and natural gradient mean updates.

    Draws action samples from a Gaussian via ParticleOptCore, computes trajectory
    costs through rollouts, and updates the distribution using z-score-weighted
    sample statistics with a configurable learning rate.
    """

    @profiler.record_function("es/init")
    def __init__(
        self,
        config: EvolutionStrategiesCfg,
        rollout_list: List[Rollout],
        use_cuda_graph: bool = False,
    ):
        self._core = ParticleOptCore(
            config,
            rollout_list,
            update_distribution_fn=self._update_distribution,
            use_cuda_graph=use_cuda_graph,
        )
        self._core.update_num_problems(config.num_problems)
        self._core.finish_init()

    # -- ES-specific distribution update --

    @torch.no_grad()
    def _update_distribution(self, trajectories: RolloutResult):
        c = self._core
        costs = trajectories.costs_and_constraints.get_sum_cost_and_constraint(
            sum_horizon=True
        )
        costs = costs.view(c.config.num_problems, c.particles_per_problem, 1)
        actions = trajectories.actions.view(
            c.config.num_problems,
            c.particles_per_problem,
            c.action_horizon,
            c.action_dim,
        )

        # ES computes weights once upfront
        w = self._exp_util_from_costs(costs)

        if c.config.sample_mode == SampleMode.BEST:
            best_idx = torch.argmax(w, dim=-1)
            c._dist.best_traj.copy_(actions[c.problem_col, best_idx])

        if c.config.store_rollouts and c.visual_traj is not None:
            total_costs = self._compute_total_cost(costs)
            vis_seq = getattr(trajectories.state, c.visual_traj)
            top_values, top_idx = stable_topk(total_costs, k=20, dim=1)
            c.top_values = top_values
            c.top_idx = top_idx
            top_trajs = torch.index_select(vis_seq, 0, top_idx[0])
            for i in range(1, top_idx.shape[0]):
                trajs = torch.index_select(
                    vis_seq,
                    0,
                    top_idx[i] + (c.particles_per_problem * i),
                )
                top_trajs = torch.cat((top_trajs, trajs), dim=0)
            if c.top_trajs is None or top_trajs.shape != c.top_trajs:
                c.top_trajs = top_trajs
            else:
                c.top_trajs.copy_(top_trajs)

        if not c.config.update_cov:
            w = w.unsqueeze(-1).unsqueeze(-1)
            new_mean = self._compute_mean(w, actions)
        else:
            new_mean, new_cov = self._compute_mean_covariance(costs, actions)
            c._dist.cov.copy_(new_cov)

        c._dist.mean.copy_(new_mean)

    def _compute_total_cost(self, costs):
        return jit_compute_total_cost(self._core.gamma_seq, costs)

    def _exp_util(self, total_costs):
        """ES weighting: z-score normalization (not softmax)."""
        return calc_exp(total_costs)

    def _exp_util_from_costs(self, costs):
        total_costs = self._compute_total_cost(costs)
        return self._exp_util(total_costs)

    def _compute_mean(self, w, actions):
        """ES mean: natural gradient update."""
        c = self._core
        if c.config.cov_type not in [CovType.SIGMA_I, CovType.DIAG_A]:
            log_and_raise(
                f"ES _compute_mean: cov_type {c.config.cov_type} not supported"
            )
        new_mean = compute_es_mean(
            w,
            actions,
            c._dist.mean,
            c._dist.full_inv_cov,
            c.config.num_particles,
            c.config.learning_rate,
        )
        return jit_blend_mean(
            c._dist.mean, new_mean, c.config.step_size_mean
        )

    def _compute_mean_covariance(self, costs, actions):
        w = self._exp_util_from_costs(costs)
        w = w.unsqueeze(-1).unsqueeze(-1)
        new_mean = self._compute_mean(w, actions)
        new_cov = self._compute_covariance(w, actions)
        self._update_cov_scale(new_cov)
        return new_mean, new_cov

    def _compute_covariance(self, w, actions):
        c = self._core
        if not c.config.update_cov:
            return
        if c.config.cov_type == CovType.SIGMA_I:
            delta_actions = actions - c._dist.mean.unsqueeze(-3)
            weighted_delta = w * (delta_actions**2)
            cov_update = torch.mean(
                torch.sum(torch.sum(weighted_delta, dim=-2), dim=-1),
                dim=-1, keepdim=True,
            )
        elif c.config.cov_type == CovType.DIAG_A:
            cov_update = jit_diag_a_cov_update(w, actions, c._dist.mean)
        else:
            log_and_raise(f"Unidentified covariance type: {c.config.cov_type}")
        cov_update = jit_blend_cov(
            c._dist.cov, cov_update, c.config.step_size_cov, c.config.kappa,
        )
        return cov_update

    def _update_cov_scale(self, new_cov=None):
        c = self._core
        if new_cov is None:
            new_cov = c._dist.cov
        if not c.config.update_cov:
            return
        c._dist.update_cov_scale(new_cov)

    # -- Protocol: delegate to core --

    @property
    def config(self):
        return self._core.config

    @property
    def device_cfg(self):
        return self._core.device_cfg

    @property
    def opt_dt(self):
        return self._core.opt_dt

    @opt_dt.setter
    def opt_dt(self, value):
        self._core.opt_dt = value

    @property
    def use_cuda_graph(self):
        return self._core.use_cuda_graph

    @property
    def enabled(self):
        return self._core.enabled

    def enable(self):
        self._core.enable()

    def disable(self):
        self._core.disable()

    @property
    def action_horizon(self):
        return self._core.action_horizon

    @property
    def action_dim(self):
        return self._core.action_dim

    @property
    def opt_dim(self):
        return self._core.opt_dim

    @property
    def outer_iters(self):
        return self._core.outer_iters

    @property
    def horizon(self):
        return self._core.horizon

    @property
    def action_bound_lows(self):
        return self._core.action_bound_lows

    @property
    def action_bound_highs(self):
        return self._core.action_bound_highs

    @property
    def action_step_max(self):
        return self._core.action_step_max

    @property
    def action_horizon_bounds_lows(self):
        return self._core.action_horizon_bounds_lows

    @property
    def action_horizon_bounds_highs(self):
        return self._core.action_horizon_bounds_highs

    @property
    def solve_time(self):
        return self._core.solve_time

    @property
    def solver_names(self):
        return self._core.solver_names

    @property
    def rollout_fn(self):
        return self._core.rollout_fn

    @property
    def _rollout_list(self):
        return self._core._rollout_list

    @property
    def _graphable_methods(self):
        return self._core._graphable_methods

    @property
    def _executors(self):
        return self._core._executors

    @_executors.setter
    def _executors(self, value):
        self._core._executors = value

    # -- Backward-compat properties --

    @property
    def mean_action(self):
        return self._core._dist.mean

    @mean_action.setter
    def mean_action(self, value):
        self._core._dist.mean = value

    @property
    def best_traj(self):
        return self._core._dist.best_traj

    @best_traj.setter
    def best_traj(self, value):
        self._core._dist.best_traj = value

    @property
    def cov_action(self):
        return self._core._dist.cov

    @cov_action.setter
    def cov_action(self, value):
        self._core._dist.cov = value

    @property
    def scale_tril(self):
        return self._core._dist.scale_tril

    @scale_tril.setter
    def scale_tril(self, value):
        self._core._dist.scale_tril = value

    @property
    def inv_cov_action(self):
        return self._core._dist.inv_cov

    @property
    def _sample_set(self):
        return self._core._dist._sample_set

    @property
    def _sample_iter(self):
        return self._core._dist._sample_iter

    @property
    def full_scale_tril(self):
        return self._core._dist.full_scale_tril

    @property
    def full_inv_cov(self):
        return self._core._dist.full_inv_cov

    @property
    def sample_lib(self):
        return self._core._dist.sample_lib

    @property
    def particles_per_problem(self):
        return self._core.particles_per_problem

    @property
    def sampled_particles_per_problem(self):
        return self._core.sampled_particles_per_problem

    @property
    def null_per_problem(self):
        return self._core.null_per_problem

    @property
    def neg_per_problem(self):
        return self._core.neg_per_problem

    @property
    def total_num_particles(self):
        return self._core.total_num_particles

    @property
    def null_act_seqs(self):
        return self._core.null_act_seqs

    @property
    def problem_col(self):
        return self._core.problem_col

    @property
    def top_trajs(self):
        return self._core.top_trajs

    @property
    def gamma_seq(self):
        return self._core.gamma_seq

    # -- Delegated methods --

    def optimize(self, seed_action):
        """Run the full optimization loop and return the best action sequence."""
        return self._core.optimize(seed_action)

    def reinitialize(self, action, mask=None, clear_optimizer_state=True, reset_num_iters=False):
        """Reset optimizer state and seed with new actions before a fresh solve."""
        return self._core.reinitialize(action, mask, clear_optimizer_state, reset_num_iters)

    def shift(self, shift_steps=0):
        return self._core.shift(shift_steps)

    def _shift(self, shift_steps=0):
        return self._core._shift(shift_steps)

    def update_num_problems(self, num_problems):
        """Resize internal buffers to accommodate a new number of parallel problems."""
        return self._core.update_num_problems(num_problems)

    def update_rollout_params(self, goal):
        return self._core.update_rollout_params(goal)

    def update_goal_dt(self, goal):
        return self._core.update_goal_dt(goal)

    def get_all_rollout_instances(self):
        return self._core.get_all_rollout_instances()

    def compute_metrics(self, action):
        """Evaluate cost and constraint metrics for the given action sequence."""
        return self._core.compute_metrics(action)

    def reset_shape(self):
        return self._core.reset_shape()

    def reset_seed(self):
        return self._core.reset_seed()

    def reset_cuda_graph(self):
        return self._core.reset_cuda_graph()

    def get_recorded_trace(self):
        return self._core.get_recorded_trace()

    def update_solver_params(self, solver_params):
        return self._core.update_solver_params(solver_params)

    def update_niters(self, niters):
        return self._core.update_niters(niters)

    def debug_dump(self, file_path=""):
        return self._core.debug_dump(file_path)

    def update_seed(self, init_act):
        return self._core.update_seed(init_act)

    def update_init_mean(self, init_mean):
        return self._core._dist.update_mean(init_mean, self._core.config.num_problems)

    def get_rollouts(self):
        return self._core.get_rollouts()

    def reset_distribution(self, reset_problem_ids=None):
        return self._core.reset_distribution(reset_problem_ids)

    def reset_mean(self, reset_problem_ids=None):
        return self._core._dist.reset_mean(self._core.config.num_problems, reset_problem_ids)

    def reset_covariance(self, reset_problem_ids=None):
        return self._core._dist.reset_covariance(self._core.config.num_problems)

    def initialize_samples(self):
        return self._core.initialize_samples()

    def update_samples(self):
        return self._core.update_samples()

    def sample_actions(self, init_act):
        return self._core.sample_actions(init_act)

    def generate_noise(self, shape, base_seed=None):
        return self._core._dist.generate_noise(shape, base_seed)

    def _get_action_seq(self, mode):
        return self._core._get_action_seq(mode)


# ---------------------------------------------------------------------------
# JIT helper functions (ES-specific)
# ---------------------------------------------------------------------------


@get_torch_jit_decorator()
def calc_exp(total_costs):
    """Z-score normalization (ES weighting, not softmax)."""
    total_costs = -1.0 * total_costs
    w = (total_costs - torch.mean(total_costs, keepdim=True, dim=-1)) / torch.std(
        total_costs, keepdim=True, dim=-1
    )
    return w


@get_torch_jit_decorator()
def compute_es_mean(
    w, actions, mean_action, full_inv_cov, num_particles: int, learning_rate: float
):
    """Natural gradient mean update for ES."""
    std_w = torch.std(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
    a_og = (actions - mean_action.unsqueeze(1)) / std_w

    weighted_sum = torch.sum(w * a_og, dim=-3, keepdim=True)

    inv_cov_diag = torch.diagonal(full_inv_cov, dim1=1, dim2=2)
    inv_cov_diag = inv_cov_diag.unsqueeze(1).unsqueeze(1)
    weighted_seq = (weighted_sum * inv_cov_diag / num_particles).squeeze(1)

    new_mean = mean_action + learning_rate * weighted_seq
    return new_mean
