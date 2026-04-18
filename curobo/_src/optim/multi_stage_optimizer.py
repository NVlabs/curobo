# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Multi-stage optimizer that chains several optimizers in sequence.

Runs each optimizer for its configured iterations, passing the result of one
stage as the seed for the next. Implements the Optimizer protocol so solvers
can use it interchangeably with single-stage optimizers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
import torch.autograd.profiler as profiler

from curobo._src.optim.optimization_iteration_state import OptimizationIterationState
from curobo._src.rollout.rollout_protocol import Rollout
from curobo._src.util.cuda_event_timer import CudaEventTimer
from curobo._src.util.logging import log_and_raise, log_info


class MultiStageOptimizer:
    """Chains multiple optimizers in sequence, seeding each stage from the previous result.

    Delegates lifecycle operations (reinitialize, shift, update_num_problems, etc.)
    to all contained optimizers, and exposes the final stage's configuration and
    action shape as its own.
    """

    def __init__(self, optimizers: List, rollout_list: Optional[List[Rollout]] = None):
        """Initialize the multi-stage optimizer with an ordered list of stages.

        The final optimizer in the list determines the exposed ``config``,
        ``action_horizon``, and ``action_dim``. When ``rollout_list`` is not
        provided, it falls back to ``optimizers[-1]._rollout_list`` so the
        multi-stage wrapper can be constructed without explicitly duplicating
        rollout references.

        Args:
            optimizers: Ordered list of optimizer instances. Each must
                implement the Optimizer protocol (optimize, reinitialize,
                shift, update_num_problems, etc.).
            rollout_list: Optional explicit rollout list. If None, uses
                ``optimizers[-1]._rollout_list`` as the fallback.
        """
        self.optimizers = optimizers
        if rollout_list is None:
            rollout_list = self.optimizers[-1]._rollout_list
        self.rollout_fn = rollout_list[0]
        self._rollout_list = rollout_list
        self.config = self.optimizers[-1].config
        self.device_cfg = self.config.device_cfg
        self.opt_dt = 0.0
        self._enabled = True

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
        return self.optimizers[-1].action_horizon

    @property
    def action_dim(self) -> int:
        return self.optimizers[-1].action_dim

    @property
    def opt_dim(self) -> int:
        return self.action_horizon * self.action_dim

    @property
    def outer_iters(self):
        return 1

    @property
    def solver_names(self):
        return [x.config.solver_name for x in self.optimizers]

    @property
    def solve_time(self) -> float:
        return self.opt_dt

    # -- Core --

    def optimize(self, seed_action: torch.Tensor) -> torch.Tensor:
        """Run all optimizer stages in sequence and return the final result.

        Each stage receives the best action from the previous stage as its
        seed. Disabled optimizers are skipped. Wall-clock time is recorded
        via a CUDA event timer.

        Args:
            seed_action: Initial action tensor of shape
                ``(num_problems, action_horizon, action_dim)`` or a flat
                view ``(num_problems, action_horizon * action_dim)``.

        Returns:
            Tensor of shape ``(num_problems, action_horizon, action_dim)``
            containing the best action from the final stage.
        """
        timer = CudaEventTimer().start()

        seed_action = seed_action.view(
            self.config.num_problems, self.action_horizon, self.action_dim
        )
        iteration_state = OptimizationIterationState(
            action=seed_action,
            exploration_action=seed_action,
        )

        iteration_state = self._opt_iters(iteration_state)

        self.opt_dt = timer.stop()
        return iteration_state.best_action

    def _opt_iters(
        self, iteration_state: OptimizationIterationState
    ) -> OptimizationIterationState:
        """Execute all optimizer stages sequentially, piping results forward.

        For each enabled optimizer stage, extracts the best action from the
        current state (falling back to ``action`` if ``best_action`` is not
        available), reshapes it for the stage's action dimensions, runs
        :meth:`optimize`, and wraps the result in a new iteration state.

        Args:
            iteration_state: Initial state carrying the seed action.

        Returns:
            :class:`OptimizationIterationState` with ``action`` and
            ``best_action`` set to the output of the final stage, shaped
            ``(num_problems, action_horizon, action_dim)``.
        """
        current_state = iteration_state
        for i, optimizer in enumerate(self.optimizers):
            log_info(
                f"Running optimizer {i + 1}/{len(self.optimizers)}: "
                f"{type(optimizer).__name__}"
            )
            if not optimizer.enabled:
                log_info(
                    f"Skipping optimizer {i + 1}/{len(self.optimizers)}: "
                    f"{type(optimizer).__name__}"
                )
                continue

            with profiler.record_function(f"MultiStageOptimizer/stage_{i}"):
                action = current_state.action
                if (
                    hasattr(current_state, "best_action")
                    and current_state.best_action is not None
                ):
                    action = current_state.best_action

                action = action.view(
                    optimizer.config.num_problems,
                    optimizer.action_horizon,
                    optimizer.action_dim,
                )

                optimized_action = optimizer.optimize(action)

                current_state = OptimizationIterationState(
                    action=optimized_action.view(
                        optimizer.config.num_problems,
                        optimizer.action_horizon,
                        optimizer.action_dim,
                    ),
                    best_action=optimized_action.view(
                        optimizer.config.num_problems,
                        optimizer.action_horizon,
                        optimizer.action_dim,
                    ),
                )
        return current_state

    # -- Lifecycle --

    def reinitialize(
        self,
        action: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        clear_optimizer_state: bool = True,
        reset_num_iters: bool = False,
    ):
        for optimizer in self.optimizers:
            if optimizer.enabled:
                optimizer.reinitialize(
                    action,
                    mask=mask,
                    clear_optimizer_state=clear_optimizer_state,
                    reset_num_iters=reset_num_iters,
                )

    def shift(self, shift_steps: int = 0) -> bool:
        return self._shift(shift_steps)

    def _shift(self, shift_steps: int = 0) -> bool:
        result = True
        for optimizer in self.optimizers:
            result = result and optimizer._shift(shift_steps)
        return result

    def update_num_problems(self, num_problems: int):
        for optimizer in self.optimizers:
            optimizer.update_num_problems(num_problems)

    def update_rollout_params(self, goal):
        for optimizer in self.optimizers:
            if optimizer.enabled:
                optimizer.update_rollout_params(goal)

    def update_goal_dt(self, goal):
        for optimizer in self.optimizers:
            optimizer.update_goal_dt(goal)

    def get_all_rollout_instances(self) -> List[Rollout]:
        rollout_list = []
        for optimizer in self.optimizers:
            rollout_list.extend(optimizer.get_all_rollout_instances())
        return rollout_list

    def compute_metrics(self, action: torch.Tensor):
        """Compute rollout metrics for an action sequence (always raises).

        Multi-stage optimizers do not support ``compute_metrics`` because the
        intermediate stages may use different rollout configurations and it is
        ambiguous which stage's rollout should evaluate the action. Callers
        should invoke ``compute_metrics`` directly on the desired single-stage
        optimizer instead.

        Raises:
            RuntimeError: Always, via :func:`log_and_raise`.
        """
        log_and_raise("compute_metric in multi_stage_optimizer")
        return self.rollout_fn.compute_metrics_from_action(action)

    def reset_shape(self):
        for optimizer in self.optimizers:
            optimizer.reset_shape()
        self.rollout_fn.reset_shape()

    def reset_seed(self):
        for optimizer in self.optimizers:
            optimizer.reset_seed()
        self.rollout_fn.reset_seed()

    def reset_cuda_graph(self):
        for optimizer in self.optimizers:
            if hasattr(optimizer, "reset_cuda_graph"):
                optimizer.reset_cuda_graph()

    def get_recorded_trace(self) -> Dict[str, Any]:
        trace = {"debug": [], "debug_cost": []}
        for optimizer in self.optimizers:
            opt_trace = optimizer.get_recorded_trace()
            trace["debug"].extend(opt_trace["debug"])
            trace["debug_cost"].extend(opt_trace["debug_cost"])
        return trace

    def update_solver_params(
        self, solver_params: Dict[str, Dict[str, Any]]
    ) -> bool:
        if not isinstance(list(solver_params.values())[0], dict):
            log_and_raise("solver_params must be a dictionary of dictionaries")
        for key in solver_params:
            if key not in self.solver_names:
                log_and_raise(
                    f"Optimizer {key} not found in {self.solver_names}"
                )
            else:
                self.optimizers[self.solver_names.index(key)].update_solver_params(
                    solver_params
                )
        return True

    def update_niters(self, niters: int):
        for optimizer in self.optimizers:
            optimizer.update_niters(niters)

    def debug_dump(self, file_path: str = ""):
        for optimizer in self.optimizers:
            if hasattr(optimizer, "debug_dump"):
                optimizer.debug_dump(file_path)
