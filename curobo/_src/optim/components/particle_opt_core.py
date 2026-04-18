# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared infrastructure for particle-based (sampling) optimizers.

Manages: GaussianDistribution, ActionBounds, DebugRecorder, rollouts, sampling,
particle counts, optimization loop, and all lifecycle methods.

The optimizer provides an update_distribution_fn(trajectories) callback that
implements the algorithm-specific weighting and mean/cov update.
Composed by MPPI and EvolutionStrategies.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.autograd.profiler as profiler

from curobo._src.optim.components.action_bounds import ActionBounds
from curobo._src.optim.components.debug_recorder import DebugRecorder
from curobo._src.optim.components.gaussian_distribution import (
    CovType,
    GaussianDistribution,
)
from curobo._src.optim.optimization_iteration_state import OptimizationIterationState
from enum import Enum


class SampleMode(Enum):
    """Mode for selecting the output action sequence from the distribution.

    Controls how the final action is extracted from the fitted Gaussian after
    each optimization pass.
    """

    MEAN = "MEAN"
    """Return the distribution mean as the action sequence."""

    BEST = "BEST"
    """Return the single best-cost trajectory seen during sampling."""

    SAMPLE = "SAMPLE"
    """Draw a fresh sample from the fitted distribution."""
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
from curobo._src.util.cuda_event_timer import CudaEventTimer
from curobo._src.util.cuda_graph_util import GraphExecutor, create_graph_executor
from curobo._src.util.logging import log_and_raise
from curobo._src.util.torch_util import get_torch_jit_decorator


class ParticleOptCore:
    """Shared particle optimizer infrastructure.

    Encapsulates everything except the distribution update strategy.
    The owning optimizer provides:
      - update_distribution_fn(trajectories): algorithm-specific weighting + mean/cov update
    """

    _graphable_methods: set = {"_opt_iters"}

    def __init__(
        self,
        config,
        rollout_list: List[Rollout],
        update_distribution_fn: Callable,
        use_cuda_graph: bool = False,
    ):
        """Initialize particle optimizer core with rollouts, distribution, and config.

        Sets up the Gaussian sampling distribution, action bounds, particle
        counts (sampled, null, negative), gamma cost-to-go sequence, and
        debug recorder. The owning optimizer must call
        :meth:`update_num_problems` then :meth:`finish_init` after assigning
        ``self._core``.

        Args:
            config: Particle optimizer configuration (device, iterations,
                covariance type, sample parameters, gamma, etc.).
            rollout_list: Rollout instances; ``[0]`` is the main rollout.
                Length must equal ``config.num_rollout_instances``.
            update_distribution_fn: ``fn(trajectory)`` callback that
                implements the algorithm-specific distribution update
                (e.g. MPPI exponential weighting or CMA-ES rank update).
            use_cuda_graph: If True, a CUDA graph executor is created for
                ``_opt_iters`` after :meth:`finish_init` is called.
        """
        if len(rollout_list) != config.num_rollout_instances:
            log_and_raise(
                f"num_rollout_instances {config.num_rollout_instances} "
                f"!= len(rollout_list) {len(rollout_list)}"
            )

        self.config = config
        self.device_cfg = config.device_cfg
        self.opt_dt = 0.0
        self.use_cuda_graph = use_cuda_graph
        self._enabled = True
        self._iteration_state: Optional[OptimizationIterationState] = None
        self._og_num_iters = config.num_iters

        # Callback
        self._update_distribution_fn = update_distribution_fn

        # Rollout
        self.rollout_fn = rollout_list[0]
        self._rollout_list = rollout_list

        # Set init_mean from rollout if not provided
        if config.init_mean is None:
            config.init_mean = self.rollout_fn.get_initial_action()

        # Components
        self._bounds = ActionBounds(
            self.rollout_fn.action_bound_lows,
            self.rollout_fn.action_bound_highs,
            self.action_horizon,
            config.step_scale,
        )
        self._debug = DebugRecorder() if config.store_debug else None

        self._dist = GaussianDistribution(
            device_cfg=config.device_cfg,
            action_horizon=self.action_horizon,
            action_dim=self.action_dim,
            cov_type=config.cov_type,
            init_mean=config.init_mean,
            init_cov=config.init_cov,
            sample_params=config.sample_params,
            random_mean=config.random_mean,
            seed=config.seed,
        )

        # Particle counts
        self._init_particle_counts(config.num_particles)
        self._init_null_act_seqs()

        # Gamma sequence for cost-to-go
        self.gamma_seq = torch.cumprod(
            torch.tensor(
                [1.0] + [config.gamma] * (self.rollout_fn.horizon - 1)
            ),
            dim=0,
        ).reshape(1, self.rollout_fn.horizon)
        self.gamma_seq = config.device_cfg.to_device(self.gamma_seq)

        # Problem index helper
        self.problem_col = None

        # Visualization state
        self.top_values = None
        self.top_idx = None
        self.top_trajs = None
        self.visual_traj = None

        # Sample iteration counter
        self._sample_iter_n = 0
        self.num_steps = 0

        # CUDA graph executors
        self._executors: Dict[str, Optional[GraphExecutor]] = {}

        # NOTE: update_num_problems is NOT called here. The owning optimizer
        # must call it after self._core is assigned so the update_distribution_fn
        # callback can access the core.

        self._deferred_cuda_graph = use_cuda_graph

    def finish_init(self):
        """Call after the owning optimizer has assigned self._core and called update_num_problems."""
        if self._deferred_cuda_graph:
            self._executors["_opt_iters"] = create_graph_executor(
                capture_fn=self._opt_iters,
                device=self.device_cfg.device,
                use_cuda_graph=True,
            )

    # -- Particle count management --

    def _init_particle_counts(self, num_particles_per_problem):
        self.null_per_problem = round(
            int(self.config.null_act_frac * num_particles_per_problem * 0.5)
        )
        self.neg_per_problem = (
            round(int(self.config.null_act_frac * num_particles_per_problem))
            - self.null_per_problem
        )
        self.sampled_particles_per_problem = (
            num_particles_per_problem
            - self.null_per_problem
            - self.neg_per_problem
        )
        self.particles_per_problem = num_particles_per_problem

    def _init_null_act_seqs(self):
        if self.null_per_problem > 0:
            self.null_act_seqs = torch.zeros(
                self.null_per_problem,
                self.action_horizon,
                self.action_dim,
                device=self.device_cfg.device,
                dtype=self.device_cfg.dtype,
            )

    # -- Properties --

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self):
        self._enabled = True

    def disable(self):
        self._enabled = False

    @property
    def action_horizon(self) -> int:
        return self.rollout_fn.action_horizon

    @property
    def action_dim(self) -> int:
        return self.rollout_fn.action_dim

    @property
    def opt_dim(self) -> int:
        return self.action_horizon * self.action_dim

    @property
    def outer_iters(self) -> int:
        return self.config.outer_iters

    @property
    def horizon(self):
        return self.rollout_fn.horizon

    @property
    def action_bound_lows(self):
        return self.rollout_fn.action_bound_lows

    @property
    def action_bound_highs(self):
        return self.rollout_fn.action_bound_highs

    @property
    def action_step_max(self):
        return self._bounds.step_max

    @property
    def action_horizon_bounds_lows(self):
        self._bounds.refresh(
            self.action_bound_lows, self.action_bound_highs, self.action_horizon
        )
        return self._bounds.horizon_lows

    @property
    def action_horizon_bounds_highs(self):
        self._bounds.refresh(
            self.action_bound_lows, self.action_bound_highs, self.action_horizon
        )
        return self._bounds.horizon_highs

    @property
    def solve_time(self) -> float:
        return self.opt_dt

    @property
    def solver_names(self):
        return [self.config.solver_name]

    # -- Core algorithm --

    def optimize(self, seed_action: torch.Tensor) -> torch.Tensor:
        """Run the full particle optimization loop and return the best action.

        Seeds the distribution mean, runs ``outer_iters`` rounds of
        ``_opt_iters`` (each containing ``inner_iters`` sample-evaluate-update
        cycles), and records wall-clock solve time via a CUDA event timer.

        Args:
            seed_action: Initial action tensor of shape
                ``(num_problems, action_horizon, action_dim)`` or a flat
                view ``(num_problems, action_horizon * action_dim)``.

        Returns:
            Tensor of shape ``(num_problems, action_horizon, action_dim)``
            containing the best action from the fitted distribution.
        """
        timer = CudaEventTimer().start()
        seed_action = seed_action.view(
            self.config.num_problems, self.action_horizon, self.action_dim
        )
        iteration_state = OptimizationIterationState(
            action=seed_action,
            exploration_action=seed_action,
        )

        for _ in range(self.outer_iters):
            iteration_state = self._dispatch("_opt_iters", iteration_state)

        self.opt_dt = timer.stop()
        return iteration_state.best_action

    def _dispatch(self, method_name: str, *args):
        executor = self._executors.get(method_name)
        if executor is not None and self.use_cuda_graph:
            return executor(*args)
        return getattr(self, method_name)(*args)

    def _opt_iters(
        self, iteration_state: OptimizationIterationState
    ) -> OptimizationIterationState:
        """Run ``inner_iters`` sample-evaluate-update cycles (CUDA-graphable).

        For each inner iteration: samples particle actions, evaluates them
        through the rollout, and calls the ``update_distribution_fn`` callback
        to refit the Gaussian. After all inner iterations, extracts the
        output action according to ``config.sample_mode``.

        Args:
            iteration_state: Current optimization state carrying the seed
                action (used to set the distribution mean).

        Returns:
            New :class:`OptimizationIterationState` with ``action`` and
            ``best_action`` set to the extracted action of shape
            ``(num_problems, action_horizon, action_dim)``.
        """
        init_act = iteration_state.action.view(
            self.config.num_problems, self.action_horizon, self.action_dim
        )
        self.update_seed(init_act)

        trajectory = None
        for _ in range(self.config.inner_iters):
            trajectory = self._generate_rollouts()

            with profiler.record_function("particle_opt/update_distribution"):
                self._update_distribution_fn(trajectory)

            if self._debug:
                current_action_seq = self._get_action_seq(
                    mode=self.config.sample_mode
                )
                costs = (
                    trajectory.costs_and_constraints.get_sum_cost_and_constraint(
                        sum_horizon=True
                    )
                )
                costs = costs.view(
                    self.config.num_problems, self.particles_per_problem
                )
                iteration_state.action = current_action_seq.view(
                    self.config.num_problems,
                    self.action_dim * self.action_horizon,
                )
                iteration_state.cost = (
                    torch.min(costs, dim=-1)[0].unsqueeze(-1).clone()
                )
                iteration_state.exploration_action = iteration_state.action
                iteration_state.exploration_cost = iteration_state.cost
                self._record_iteration_state(iteration_state)

        curr_action_seq = self._get_action_seq(mode=self.config.sample_mode)
        curr_costs = (
            trajectory.costs_and_constraints.get_sum_cost_and_constraint(
                sum_horizon=False
            )
        )
        output_state = OptimizationIterationState(
            action=curr_action_seq,
            cost=torch.min(torch.sum(curr_costs, dim=-1), dim=-1)[0]
            .unsqueeze(-1)
            .clone(),
            best_action=curr_action_seq,
        )
        output_state.best_cost = iteration_state.cost
        return output_state

    # -- Sampling --

    @torch.no_grad()
    def sample_actions(self, init_act):
        """Sample action sequences from the current Gaussian distribution.

        Draws ``sampled_particles_per_problem`` samples from the distribution,
        appends negated-mean and zero-action particles (controlled by
        ``config.null_act_frac``), and squashes the result to action bounds.

        Args:
            init_act: Unused (kept for interface compatibility). Sampling
                uses the distribution mean set by :meth:`update_seed`.

        Returns:
            Tensor of shape
            ``(total_num_particles, action_horizon, action_dim)`` where
            ``total_num_particles = num_problems * particles_per_problem``.
        """
        delta = self._dist.get_samples(
            self.config.num_iters, self.config.sample_params.fixed_samples
        )
        scaled_delta = delta * self._dist.full_scale_tril
        act_seq = self._dist.mean.unsqueeze(-3) + scaled_delta

        cat_list = [act_seq]
        if self.neg_per_problem > 0:
            neg_action = -1.0 * self._dist.mean
            neg_act_seqs = neg_action.unsqueeze(-3).expand(
                -1, self.neg_per_problem, -1, -1
            )
            cat_list.append(neg_act_seqs)
        if self.null_per_problem > 0:
            cat_list.append(
                self.null_act_seqs[: self.null_per_problem]
                .unsqueeze(0)
                .expand(self.config.num_problems, -1, -1, -1)
            )

        act_seq = torch.cat(cat_list, dim=-3)
        act_seq = act_seq.reshape(
            self.total_num_particles, self.action_horizon * self.action_dim
        )
        act_seq = scale_ctrl(
            act_seq,
            self.action_horizon_bounds_lows,
            self.action_horizon_bounds_highs,
            squash_fn=self.config.squash_fn,
        )
        act_seq = act_seq.reshape(
            self.total_num_particles, self.action_horizon, self.action_dim
        )
        return act_seq

    def _generate_rollouts(self) -> RolloutResult:
        act_seq = self.sample_actions(None)
        return self.rollout_fn.evaluate_action(act_seq)

    # -- Action selection --

    def _get_action_seq(self, mode: SampleMode):
        if mode == SampleMode.MEAN:
            return self._dist.mean
        elif mode == SampleMode.SAMPLE:
            delta = self._dist.generate_noise(
                shape=torch.Size(
                    (self.config.num_problems, self.action_horizon)
                ),
                base_seed=self.config.seed + 123 * self.num_steps,
            )
            return (
                self._dist.mean
                + delta * self._dist.full_scale_tril.squeeze(-3)
            )
        elif mode == SampleMode.BEST:
            return self._dist.best_traj
        else:
            log_and_raise(f"Unidentified sampling mode: {mode}")

    def update_seed(self, init_act):
        if len(init_act.shape) > 3:
            log_and_raise(
                "updating seed requires initial action to be of shape "
                "[num_problems, action_horizon, action_dim]"
            )
        self._dist.update_mean(init_act, self.config.num_problems)

    def get_rollouts(self):
        return self.top_trajs

    # -- Distribution management --

    def reset_distribution(self, reset_problem_ids=None):
        self._dist.reset_mean(self.config.num_problems, reset_problem_ids)
        self._dist.reset_covariance(self.config.num_problems)

    def initialize_samples(self):
        self._dist.initialize_samples(
            self.config.num_problems,
            self.sampled_particles_per_problem,
            self.config.num_iters,
            self.config.sample_params.fixed_samples,
            self.config.sample_per_problem,
        )

    def update_samples(self):
        self._dist.update_samples(
            self.config.num_problems,
            self.sampled_particles_per_problem,
            self.config.num_iters,
            self.config.sample_params.fixed_samples,
            self.config.sample_per_problem,
        )

    # -- Lifecycle --

    def reinitialize(
        self,
        action: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        clear_optimizer_state: bool = True,
        reset_num_iters: bool = False,
    ) -> None:
        """Reset the distribution and sampling state for a new optimization.

        Resets the Gaussian distribution mean and covariance to their initial
        values, clears the sample iteration counter, and regenerates sample
        indices. Debug history is cleared when no mask is provided.

        Args:
            action: Seed action tensor of shape
                ``(num_problems, action_horizon, action_dim)`` or flat.
                Currently used only to validate shape; the distribution
                mean is reset to ``config.init_mean``.
            mask: Optional boolean mask. Currently unused by particle
                optimizers (distribution reset is always global).
            clear_optimizer_state: Unused; accepted for protocol
                compatibility with gradient optimizers.
            reset_num_iters: If True, restore ``num_iters`` to its
                original configured value.
        """
        action = action.view(
            self.config.num_problems, self.action_horizon, self.action_dim
        )
        if mask is None and self._debug:
            self._debug.clear()
        if reset_num_iters:
            self.config.num_iters = self._og_num_iters

        self.reset_distribution()
        self._dist._sample_iter[:] = 0
        self._sample_iter_n = 0
        self.update_samples()

    def shift(self, shift_steps: int = 0) -> bool:
        """Shift the distribution for MPC warm-starting.

        Args:
            shift_steps: Number of timesteps to shift the mean forward.
                Zero is a no-op.

        Returns:
            True on success.
        """
        return self._shift(shift_steps)

    def _shift(self, shift_steps: int = 0) -> bool:
        if shift_steps == 0:
            return True
        self._dist.shift(
            shift_steps,
            self.config.base_action == self.config.base_action.__class__["REPEAT"],
        )
        return True

    def update_num_problems(self, num_problems: int):
        """Resize all internal buffers for a new batch size.

        Updates particle counts, reinitializes the sample matrix, resets
        the Gaussian distribution, and resizes all rollout instances.

        Args:
            num_problems: New number of independent problems (must be > 0).
        """
        assert num_problems > 0
        self.config.num_problems = num_problems
        self.total_num_particles = num_problems * self.config.num_particles

        self.problem_col = torch.arange(
            0,
            num_problems,
            step=1,
            dtype=torch.long,
            device=self.device_cfg.device,
        )

        self.initialize_samples()
        self.reset_distribution()

        for rollout in self._rollout_list:
            rollout.update_batch_size(
                batch_size=num_problems * self.config.num_particles
            )

    def update_rollout_params(self, goal):
        for rollout in self._rollout_list:
            rollout.update_params(goal, num_particles=self.config.num_particles)

    def update_goal_dt(self, goal):
        for rollout in self._rollout_list:
            rollout.update_goal_dt(goal)

    def get_all_rollout_instances(self) -> List[Rollout]:
        return self._rollout_list

    def compute_metrics(self, action: torch.Tensor):
        return self.rollout_fn.compute_metrics_from_action(action)

    def reset_shape(self):
        for rollout in self._rollout_list:
            rollout.reset_shape()

    def reset_seed(self) -> bool:
        self._dist.reset_seed()
        self.update_samples()
        return True

    def reset_cuda_graph(self):
        for executor in self._executors.values():
            if executor is not None:
                executor.reset()
        if hasattr(self.rollout_fn, "reset_cuda_graph"):
            self.rollout_fn.reset_cuda_graph()

    def get_recorded_trace(self) -> Dict[str, Any]:
        if self._debug:
            return self._debug.get_trace()
        return {"debug": [], "debug_cost": []}

    def update_solver_params(
        self, solver_params: Dict[str, Dict[str, Any]]
    ) -> bool:
        if self.config.solver_name not in solver_params:
            log_and_raise(
                f"Optimizer {self.config.solver_name} not found in "
                f"{solver_params}"
            )
        for param_name, param_value in solver_params[
            self.config.solver_name
        ].items():
            setattr(self.config, param_name, param_value)
        return True

    def update_niters(self, niters: int):
        self.config.update_niters(niters)

    def _record_iteration_state(self, iteration_state):
        if self._debug:
            self._debug.record(
                iteration_state, self.action_horizon, self.action_dim
            )

    def debug_dump(self, file_path: str = ""):
        pass
