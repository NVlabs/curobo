# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Third Party
import torch
from torch.autograd import Function

from curobo._src.curobolib.backends import optimization as optimization_cu
from curobo._src.curobolib.cuda_ops.tensor_checks import (
    check_float32_tensors,
    check_int16_tensors,
    check_int32_tensors,
    check_uint8_tensors,
)
from curobo._src.optim.gradient.line_search_context import LineSearchContext
from curobo._src.optim.optimization_iteration_state import OptimizationIterationState

# CuRobo
from curobo._src.util.logging import log_and_raise


def wolfe_line_search(
    iteration_state: OptimizationIterationState,  # contains best, current, exploration, converged values.
    line_search_context: LineSearchContext,
    exploration_idx: torch.Tensor,
    selected_idx: torch.Tensor,
    search_cost: torch.Tensor,  # should come from line search function
    search_action: torch.Tensor,  # should come from line search function
    search_gradient: torch.Tensor,  # should come from line search function
    step_direction: torch.Tensor,  # should come from line search function
    strong_wolfe: bool,
    approx_wolfe: bool,
):
    """Perform Wolfe line search to find the optimal step size. Also updates the iteration state.

    Args:
        iteration_state: OptimizationIterationState containing best, current, exploration, converged values.
        exploration_idx: Indices of the exploration solutions. Shape: (num_problems, n_linesearch)
        selected_idx: Indices of the selected solutions. Shape: (num_problems, n_linesearch)
        search_cost: Cost at each step size. Shape: (num_problems, n_linesearch)
        search_action: Action at each step size. Shape: (num_problems, n_linesearch, action_dim)
        search_gradient: Gradient at each step size. Shape: (num_problems, n_linesearch, action_dim)
        step_direction: Search direction. Shape: (num_problems, action_dim)
        search_magnitudes: Step sizes to evaluate. Shape: (n_linesearch)
        line_search_context: LineSearchContext containing line search parameters.

    Returns:
        OptimizationIterationState: Updated iteration state with new best values.
        exploration_idx: Indices of the exploration solutions. Shape: (num_problems, n_linesearch)
        selected_idx: Indices of the selected solutions. Shape: (num_problems, n_linesearch)
    """
    num_problems = line_search_context.num_problems
    opt_dim = line_search_context.opt_dim
    action_dim = line_search_context.action_dim
    action_horizon = line_search_context.action_horizon
    n_linesearch = line_search_context.n_linesearch

    device = iteration_state.best_cost.device
    check_float32_tensors(
        device,
        best_cost=iteration_state.best_cost,
        best_action=iteration_state.best_action,
        exploration_cost=iteration_state.exploration_cost,
        exploration_action=iteration_state.exploration_action,
        exploration_gradient=iteration_state.exploration_gradient,
        cost=iteration_state.cost,
        action=iteration_state.action,
        gradient=iteration_state.gradient,
        search_cost=search_cost,
        search_action=search_action,
        search_gradient=search_gradient,
        step_direction=step_direction,
        line_search_scale=line_search_context.line_search_scale,
    )
    check_int16_tensors(
        device,
        best_iteration=iteration_state.best_iteration,
        current_iteration=iteration_state.current_iteration,
    )
    check_uint8_tensors(device, converged=iteration_state.converged)
    check_int32_tensors(
        device,
        exploration_idx=exploration_idx,
        selected_idx=selected_idx,
    )

    # check shapes of inputs:
    if iteration_state.best_cost.shape != (num_problems,):
        log_and_raise(
            f"best_cost must have shape ({num_problems}). Got {iteration_state.best_cost.shape}"
        )
    if iteration_state.best_action.shape != (num_problems, action_horizon, action_dim):
        log_and_raise(
            f"best_action must have shape ({num_problems}, {action_horizon}, {action_dim}). Got {iteration_state.best_action.shape}"
        )
    if iteration_state.best_iteration.shape != (num_problems,):
        log_and_raise(
            f"best_iteration must have shape ({num_problems}). Got {iteration_state.best_iteration.shape}"
        )
    if iteration_state.current_iteration.shape != (num_problems,):
        log_and_raise(
            f"current_iteration must have shape ({num_problems}). Got {iteration_state.current_iteration.shape}"
        )
    if iteration_state.converged.shape != (num_problems,):
        log_and_raise(
            f"converged must have shape ({num_problems}). Got {iteration_state.converged.shape}"
        )
    if iteration_state.exploration_cost.shape != (num_problems,):
        log_and_raise(
            f"exploration_cost must have shape ({num_problems}). Got {iteration_state.exploration_cost.shape}"
        )
    if iteration_state.exploration_action.shape != (num_problems, action_horizon, action_dim):
        log_and_raise(
            f"exploration_action must have shape ({num_problems}, {action_horizon}, {action_dim}). Got {iteration_state.exploration_action.shape}"
        )
    if iteration_state.exploration_gradient.shape != (num_problems, action_horizon, action_dim):
        log_and_raise(
            f"exploration_gradient must have shape ({num_problems}, {action_horizon}, {action_dim}). Got {iteration_state.exploration_gradient.shape}"
        )
    if iteration_state.cost.shape != (num_problems,):
        log_and_raise(f"cost must have shape ({num_problems}). Got {iteration_state.cost.shape}")
    if iteration_state.action.shape != (num_problems, action_horizon, action_dim):
        log_and_raise(
            f"action must have shape ({num_problems}, {action_horizon}, {action_dim}). Got {iteration_state.action.shape}"
        )
    if iteration_state.gradient.shape != (num_problems, action_horizon, action_dim):
        log_and_raise(
            f"gradient must have shape ({num_problems}, {1}, {opt_dim}). Got {iteration_state.gradient.shape}"
        )
    if exploration_idx.shape != (num_problems, n_linesearch):
        log_and_raise(
            f"exploration_idx must have shape ({num_problems}, {n_linesearch}). Got {exploration_idx.shape}"
        )
    if selected_idx.shape != (num_problems, n_linesearch):
        log_and_raise(
            f"selected_idx must have shape ({num_problems}, {n_linesearch}). Got {selected_idx.shape}"
        )
    if search_cost.shape != (num_problems, n_linesearch, 1):
        log_and_raise(
            f"search_cost must have shape ({num_problems}, {n_linesearch}, {1}). Got {search_cost.shape}"
        )
    if search_action.shape != (num_problems, n_linesearch, opt_dim):
        log_and_raise(
            f"search_action must have shape ({num_problems}, {n_linesearch}, {opt_dim}). Got {search_action.shape}"
        )
    if search_gradient.shape != (num_problems, n_linesearch, opt_dim):
        log_and_raise(
            f"search_gradient must have shape ({num_problems}, {n_linesearch}, {opt_dim}). Got {search_gradient.shape}"
        )
    if step_direction.shape != (num_problems, 1, opt_dim):
        log_and_raise(
            f"step_direction must have shape ({num_problems}, {1}, {opt_dim}). Got {step_direction.shape}"
        )
    if line_search_context.line_search_scale.shape != (1, n_linesearch, 1, 1):
        log_and_raise(
            f"line_search_scale must have shape ({1}, {n_linesearch}, {1}, {1}). Got {line_search_context.line_search_scale.shape}"
        )

    optimization_cu.launch_line_search(
        iteration_state.best_cost,
        iteration_state.best_action,
        iteration_state.best_iteration,
        iteration_state.current_iteration,
        iteration_state.converged,
        line_search_context.convergence_iteration,
        line_search_context.cost_delta_threshold,
        line_search_context.cost_relative_threshold,
        iteration_state.exploration_cost,
        iteration_state.exploration_action,
        iteration_state.exploration_gradient,
        exploration_idx.view(-1),
        iteration_state.cost,
        iteration_state.action,
        iteration_state.gradient,
        selected_idx.view(-1),
        search_cost,
        search_action,
        search_gradient,
        step_direction,
        line_search_context.line_search_scale,
        line_search_context.line_search_c_1,
        line_search_context.line_search_c_2,
        strong_wolfe,
        approx_wolfe,
        n_linesearch,
        opt_dim,
        num_problems,
    )
    return iteration_state, exploration_idx, selected_idx


class LBFGScu(Function):
    @staticmethod
    def forward(
        ctx,
        step_vec,
        rho_buffer,
        y_buffer,
        s_buffer,
        q,
        grad_q,
        x_0,
        grad_0,
        epsilon=0.1,
        stable_mode=False,
        use_shared_buffers=True,
    ):
        device = step_vec.device
        check_float32_tensors(
            device,
            step_vec=step_vec,
            rho_buffer=rho_buffer,
            y_buffer=y_buffer,
            s_buffer=s_buffer,
            q=q,
            grad_q=grad_q,
            x_0=x_0,
            grad_0=grad_0,
        )
        m, b, v_dim, _ = y_buffer.shape
        R = optimization_cu.launch_lbfgs_step(
            step_vec,  # .view(-1),
            rho_buffer,  # .view(-1),
            y_buffer,  # .view(-1),
            s_buffer,  # .view(-1),
            q,
            grad_q,  # .view(-1),
            x_0,
            grad_0,
            epsilon,
            b,
            m,
            v_dim,
            stable_mode,
            use_shared_buffers,
        )
        step_v = R[0].view(step_vec.shape)

        # ctx.save_for_backward(batch_spheres, robot_spheres, link_mats, link_sphere_map)
        return step_v

    @staticmethod
    def backward(ctx, grad_output):
        return (
            None,
            None,
            None,
            None,
            None,
            None,
        )
