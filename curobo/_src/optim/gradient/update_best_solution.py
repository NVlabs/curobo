# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch

from curobo._src.optim.optimization_iteration_state import OptimizationIterationState
from curobo._src.util.logging import log_and_raise
from curobo._src.util.torch_util import get_torch_jit_decorator


@get_torch_jit_decorator(only_valid_for_compile=True)
def update_best_solution(
    iteration_state: OptimizationIterationState,
    action_horizon: int,
    action_dim: int,
    cost_delta_threshold: float,
    cost_relative_threshold: float,
    convergence_iteration: int,
) -> OptimizationIterationState:
    if iteration_state.best_cost is None:
        log_and_raise("best_cost is None")
    if iteration_state.best_action is None:
        log_and_raise("best_action is None")
    if iteration_state.best_iteration is None:
        log_and_raise("best_iteration is None")
    if iteration_state.current_iteration is None:
        log_and_raise("current_iteration is None")
    if iteration_state.cost is None:
        log_and_raise("cost is None")
    if iteration_state.action is None:
        log_and_raise("action is None")
    if iteration_state.converged is None:
        log_and_raise("converged is None")

    cost = iteration_state.cost.detach()
    q = iteration_state.action.detach()
    q = q.view(-1, action_horizon, action_dim)

    # check best cost is better than previous best cost
    cost_delta = iteration_state.best_cost - cost

    cost_relative = cost_delta / (iteration_state.best_cost + 1e-8)

    # cost delta should be positive and greater than cost_delta_threshold
    mask = torch.logical_and(
        cost_delta > cost_delta_threshold, cost_relative > cost_relative_threshold
    )
    # update best cost, action, and iteration
    iteration_state.best_cost.copy_(torch.where(mask, cost, iteration_state.best_cost))
    mask = mask.view(mask.shape[0])
    mask_q = mask.unsqueeze(-1).unsqueeze(-1).expand(-1, action_horizon, action_dim)
    iteration_state.best_action.copy_(torch.where(mask_q, q, iteration_state.best_action))

    # Should also update convergence logic:
    iteration_state.current_iteration += 1
    iteration_state.best_iteration.copy_(
        torch.where(mask, iteration_state.current_iteration, iteration_state.best_iteration)
    )

    # check if converged
    # converged if best_iteration + convergence_iteration <= current_iteration
    converged = (
        iteration_state.best_iteration + convergence_iteration <= iteration_state.current_iteration
    )
    iteration_state.converged.copy_(converged)

    return iteration_state
