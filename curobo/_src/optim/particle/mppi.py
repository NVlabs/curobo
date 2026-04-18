# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Model Predictive Path Integral (MPPI) optimizer for sampling-based trajectory optimization.

Draws action samples from a Gaussian distribution via ParticleOptCore, evaluates
costs through rollouts, and updates the distribution mean and covariance using
softmax-weighted sample statistics. Supports diagonal covariance fast paths and
CUDA graph acceleration.
"""

from __future__ import annotations

# Standard Library
import math
from copy import deepcopy
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Dict, List, Optional

# Third Party
import torch
import torch.autograd.profiler as profiler

from curobo._src.optim.components.gaussian_distribution import CovType
from curobo._src.optim.components.particle_opt_core import ParticleOptCore
from curobo._src.optim.particle.particle_opt_utils import (
    SquashType,
    gaussian_entropy,
    scale_ctrl,
)
from curobo._src.optim.particle.sample_strategies.particle_sampler_cfg import (
    ParticleSamplerCfg,
)
from curobo._src.rollout.metrics import RolloutResult
from curobo._src.rollout.rollout_protocol import Rollout
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise
from curobo._src.util.tensor_util import stable_topk
from curobo._src.util.torch_util import get_torch_jit_decorator


from curobo._src.optim.components.particle_opt_core import SampleMode  # canonical location


class BaseActionType(Enum):
    """Strategy for initializing the base action during MPPI warm-start shifts.

    Controls what value is used to fill newly exposed time steps when the action
    buffer is shifted forward (e.g., in MPC).
    """

    REPEAT = "REPEAT"
    """Repeat the last action of the previous horizon into new time steps."""

    NULL = "NULL"
    """Fill new time steps with zero actions."""

    RANDOM = "RANDOM"
    """Fill new time steps with random samples from the distribution."""


@dataclass
class MPPICfg:
    """Flat configuration for MPPI optimizer. All fields in one place."""

    # General
    num_iters: int = 100
    solver_type: str = "mppi"
    solver_name: str = "mppi"
    device_cfg: DeviceCfg = DeviceCfg()
    store_debug: bool = False
    debug_info: Any = None
    num_problems: int = 1
    num_particles: Optional[int] = None
    sync_cuda_time: bool = True
    use_coo_sparse: bool = True
    step_scale: float = 1.0
    inner_iters: int = 1
    _num_rollout_instances: int = 1

    # Particle
    gamma: float = 1.0
    sample_mode: SampleMode = SampleMode.MEAN
    seed: int = 0
    store_rollouts: bool = False
    #: Fraction of particles that use the null (zero) action sequence. Ensures
    #: at least some samples explore the "do nothing" trajectory, which improves
    #: stability in MPC settings. Range [0, 1]; 0 disables null samples.
    null_act_frac: float = 0.0

    # MPPI-specific
    init_mean: Optional[torch.Tensor] = None
    init_cov: float = 0.5
    base_action: BaseActionType = BaseActionType.REPEAT
    #: Exponential moving average blending factor for the distribution mean
    #: update. At each iteration the new mean is
    #: ``(1 - step_size_mean) * old_mean + step_size_mean * weighted_mean``.
    #: Higher values make the mean track the current sample statistics faster.
    step_size_mean: float = 0.9
    #: Exponential moving average blending factor for the covariance update,
    #: analogous to :attr:`step_size_mean`. Lower values keep the covariance
    #: closer to the previous iteration, reducing exploration noise.
    step_size_cov: float = 0.1
    squash_fn: SquashType = SquashType.CLAMP
    cov_type: CovType = CovType.DIAG_A
    sample_params: Optional[ParticleSamplerCfg] = None
    update_cov: bool = True
    #: When True, the distribution mean is re-sampled randomly each iteration
    #: instead of being updated from the weighted sample statistics. Useful for
    #: highly multi-modal landscapes but generally hurts convergence.
    random_mean: bool = False
    #: MPPI temperature parameter (inverse). Controls the sharpness of the
    #: softmax weighting over trajectory costs: smaller values concentrate
    #: weight on the lowest-cost samples (greedy), larger values spread
    #: weight more uniformly (exploratory). This is the single most important
    #: tuning knob for MPPI convergence behavior.
    beta: float = 0.1
    #: Trade-off between control cost and state cost in the MPPI objective.
    #: ``alpha = 1.0`` uses only state cost; values below 1.0 blend in a
    #: control-effort penalty proportional to ``(1 - alpha)``.
    alpha: float = 1.0
    #: Minimum covariance floor added after each covariance update to prevent
    #: the distribution from collapsing to a point. Keeps exploration alive
    #: across iterations.
    kappa: float = 0.01
    sample_per_problem: bool = True

    def __post_init__(self):
        if self.num_particles is None:
            self.num_particles = 1
        if self.sample_params is None:
            self.sample_params = ParticleSamplerCfg(device_cfg=self.device_cfg)
        elif isinstance(self.sample_params, dict):
            self.sample_params = ParticleSamplerCfg(
                **self.sample_params, device_cfg=self.device_cfg
            )
        self.init_cov = self.device_cfg.to_device(self.init_cov).unsqueeze(0)
        if self.init_mean is not None:
            self.init_mean = self.device_cfg.to_device(self.init_mean).clone()
        if isinstance(self.cov_type, str):
            self.cov_type = CovType[self.cov_type]
        if isinstance(self.base_action, str):
            self.base_action = BaseActionType[self.base_action]
        if isinstance(self.squash_fn, str):
            self.squash_fn = SquashType[self.squash_fn]
        if isinstance(self.sample_mode, str):
            self.sample_mode = SampleMode[self.sample_mode]

    @property
    def num_rollout_instances(self):
        return self._num_rollout_instances

    @property
    def outer_iters(self):
        return math.ceil(self.num_iters / self.inner_iters)

    @classmethod
    def create_data_dict(cls, data_dict, device_cfg=DeviceCfg(), child_dict=None):
        if child_dict is None:
            child_dict = deepcopy(data_dict)
        child_dict["device_cfg"] = device_cfg
        if "num_particles" not in child_dict:
            child_dict["num_particles"] = None
        dataclass_field_names = {f.name for f in fields(cls)}
        for k in [k for k in child_dict if k not in dataclass_field_names]:
            child_dict.pop(k)
        return child_dict

    def update_niters(self, niters: int):
        self.num_iters = niters


class MPPI:
    """MPPI optimizer using softmax-weighted sample statistics to update the distribution.

    Draws action samples from a Gaussian via ParticleOptCore, computes trajectory
    costs through rollouts, and updates mean and covariance using softmax weights
    over the cost-ranked samples.
    """

    @profiler.record_function("mppi/init")
    def __init__(
        self,
        config: MPPICfg,
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

    # -- MPPI-specific distribution update --

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

        with profiler.record_function("mppi/get_best"):
            if c.config.sample_mode == SampleMode.BEST:
                w = self._exp_util_from_costs(costs)
                best_idx = torch.argmax(w, dim=-1)
                c._dist.best_traj.copy_(actions[c.problem_col, best_idx])

        with profiler.record_function("mppi/store_rollouts"):
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
            w = self._exp_util_from_costs(costs)
            w = w.unsqueeze(-1).unsqueeze(-1)
            new_mean = self._compute_mean(w, actions)
        else:
            new_mean, new_cov = self._compute_mean_covariance(costs, actions)
            c._dist.cov.copy_(new_cov)

        c._dist.mean.copy_(new_mean)

    def _compute_total_cost(self, costs):
        return jit_compute_total_cost(self._core.gamma_seq, costs)

    def _exp_util(self, total_costs):
        return jit_calculate_exp_util(self._core.config.beta, total_costs)

    def _exp_util_from_costs(self, costs):
        return jit_calculate_exp_util_from_costs(
            costs, self._core.gamma_seq, self._core.config.beta
        )

    def _compute_mean(self, w, actions):
        new_mean = torch.sum(w * actions, dim=-3)
        return jit_blend_mean(
            self._core._dist.mean, new_mean, self._core.config.step_size_mean
        )

    def _compute_mean_covariance(self, costs, actions):
        c = self._core
        if c.config.cov_type == CovType.DIAG_A:
            new_mean, new_cov, new_scale_tril = jit_mean_cov_diag_a(
                costs, actions, c.gamma_seq, c._dist.mean, c._dist.cov,
                c.config.step_size_mean, c.config.step_size_cov,
                c.config.kappa, c.config.beta,
            )
            c._dist.scale_tril.copy_(new_scale_tril)
        else:
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

    # -- Backward-compat properties delegating to GaussianDistribution --

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
    def entropy(self):
        return gaussian_entropy(L=self.full_scale_tril)

    @property
    def squashed_mean(self):
        original_shape = self._core._dist.mean.shape
        flattened = self._core._dist.mean.view(original_shape[0], -1)
        squashed = scale_ctrl(
            flattened,
            self.action_horizon_bounds_lows,
            self.action_horizon_bounds_highs,
            squash_fn=self.config.squash_fn,
        )
        return squashed.view(original_shape)

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
# JIT helper functions
# ---------------------------------------------------------------------------


@get_torch_jit_decorator()
def jit_calculate_exp_util(beta: float, total_costs):
    """Compute softmax importance weights from pre-aggregated total costs.

    Args:
        beta: temperature parameter; smaller values sharpen the distribution.
        total_costs: (num_problems, num_particles) aggregated scalar costs.

    Returns:
        w: (num_problems, num_particles) softmax weights summing to 1 per problem.
    """
    w = torch.softmax((-1.0 / beta) * total_costs, dim=-1)
    return w


@get_torch_jit_decorator()
def jit_calculate_exp_util_from_costs(costs, gamma_seq, beta: float):
    """Compute softmax importance weights from per-step costs with discounting.

    Applies the gamma discount sequence to per-step costs, sums over the
    horizon, and returns softmax weights.

    Args:
        costs: (num_problems, num_particles, horizon) per-step costs.
        gamma_seq: (1, 1, horizon) discount factors per time step.
        beta: temperature parameter; smaller values sharpen the distribution.

    Returns:
        w: (num_problems, num_particles) softmax weights summing to 1 per problem.
    """
    cost_seq = gamma_seq * costs
    cost_seq = torch.sum(cost_seq, dim=-1, keepdim=False) / gamma_seq[..., 0]
    w = torch.softmax((-1.0 / beta) * cost_seq, dim=-1)
    return w


@get_torch_jit_decorator()
def jit_compute_total_cost(gamma_seq, costs):
    cost_seq = gamma_seq * costs
    cost_seq = torch.sum(cost_seq, dim=-1, keepdim=False) / gamma_seq[..., 0]
    return cost_seq


@get_torch_jit_decorator()
def jit_diag_a_cov_update(w, actions, mean_action):
    """Compute a diagonal covariance update from weighted squared deviations.

    Args:
        w: (num_problems, num_particles, 1, 1) importance weights.
        actions: (num_problems, num_particles, action_horizon, action_dim)
            sampled action sequences.
        mean_action: (num_problems, action_horizon, action_dim) current mean.

    Returns:
        cov_update: (num_problems, 1, action_dim) per-dimension variance estimate.
    """
    delta_actions = actions - mean_action.unsqueeze(-3)
    weighted_delta = w * (delta_actions**2)
    cov_update = torch.mean(
        torch.sum(weighted_delta, dim=-3), dim=-2
    ).unsqueeze(-2)
    return cov_update


@get_torch_jit_decorator()
def jit_blend_cov(cov_action, cov_update, step_size_cov: float, kappa: float):
    """Blend the current covariance with a new estimate using exponential smoothing.

    Computes ``(1 - step_size_cov) * cov_action + step_size_cov * cov_update + kappa``
    where kappa is a small positive regularizer preventing covariance collapse.

    Args:
        cov_action: current covariance, shape depends on CovType (e.g.,
            (num_problems, 1, action_dim) for DIAG_A).
        cov_update: new covariance estimate from weighted samples, same shape.
        step_size_cov: blending weight in [0, 1] for the new estimate.
        kappa: additive regularization constant.

    Returns:
        new_cov: blended covariance, same shape as cov_action.
    """
    new_cov = (
        (1.0 - step_size_cov) * cov_action + step_size_cov * cov_update + kappa
    )
    return new_cov


@get_torch_jit_decorator()
def jit_blend_mean(mean_action, new_mean, step_size_mean: float):
    """Blend the current mean with a new weighted-sample mean.

    Computes ``(1 - step_size_mean) * mean_action + step_size_mean * new_mean``.

    Args:
        mean_action: (num_problems, action_horizon, action_dim) current mean.
        new_mean: (num_problems, action_horizon, action_dim) weighted sample mean.
        step_size_mean: blending weight in [0, 1] for the new estimate.

    Returns:
        mean_update: (num_problems, action_horizon, action_dim) blended mean.
    """
    mean_update = (
        (1.0 - step_size_mean) * mean_action + step_size_mean * new_mean
    )
    return mean_update


@get_torch_jit_decorator()
def jit_mean_cov_diag_a(
    costs, actions, gamma_seq, mean_action, cov_action,
    step_size_mean: float, step_size_cov: float, kappa: float, beta: float,
):
    """Fused mean and diagonal covariance update for the DIAG_A covariance type.

    Computes softmax weights from costs, updates the mean via weighted average
    and exponential blending, updates the diagonal covariance from weighted
    squared deviations, and returns the Cholesky factor (sqrt of diagonal cov).

    Args:
        costs: (num_problems, num_particles, horizon) per-step costs.
        actions: (num_problems, num_particles, action_horizon, action_dim)
            sampled action sequences.
        gamma_seq: (1, 1, horizon) discount factors per time step.
        mean_action: (num_problems, action_horizon, action_dim) current mean.
        cov_action: (num_problems, 1, action_dim) current diagonal covariance.
        step_size_mean: blending weight for the mean update.
        step_size_cov: blending weight for the covariance update.
        kappa: additive covariance regularization constant.
        beta: softmax temperature parameter.

    Returns:
        Tuple of (new_mean, new_cov, new_scale_tril) where new_mean is
        (num_problems, action_horizon, action_dim), new_cov is
        (num_problems, 1, action_dim), and new_scale_tril is sqrt(new_cov).
    """
    w = jit_calculate_exp_util_from_costs(costs, gamma_seq, beta)
    w = w.unsqueeze(-1).unsqueeze(-1)
    new_mean = torch.sum(w * actions, dim=-3)
    new_mean = jit_blend_mean(mean_action, new_mean, step_size_mean)
    cov_update = jit_diag_a_cov_update(w, actions, mean_action)
    new_cov = jit_blend_cov(cov_action, cov_update, step_size_cov, kappa)
    new_tril = torch.sqrt(new_cov)
    return new_mean, new_cov, new_tril
