#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Standard Library
from dataclasses import dataclass
from typing import Optional

# Third Party
import torch

# CuRobo
from curobo.opt.particle.parallel_mppi import (
    CovType,
    ParallelMPPI,
    ParallelMPPIConfig,
    Trajectory,
    jit_blend_mean,
)
from curobo.opt.particle.particle_opt_base import SampleMode
from curobo.util.torch_utils import get_torch_jit_decorator


@dataclass
class ParallelESConfig(ParallelMPPIConfig):
    learning_rate: float = 0.1


class ParallelES(ParallelMPPI, ParallelESConfig):
    def __init__(self, config: Optional[ParallelESConfig] = None):
        if config is not None:
            ParallelESConfig.__init__(self, **vars(config))
        ParallelMPPI.__init__(self)

    def _compute_mean(self, w, actions):
        if self.cov_type not in [CovType.SIGMA_I, CovType.DIAG_A]:
            raise NotImplementedError()
        new_mean = compute_es_mean(
            w, actions, self.mean_action, self.full_inv_cov, self.num_particles, self.learning_rate
        )
        # get the new means from here
        # use Evolutionary Strategy Mean here:
        new_mean = jit_blend_mean(self.mean_action, new_mean, self.step_size_mean)
        return new_mean

    def _exp_util_from_costs(self, costs):
        total_costs = self._compute_total_cost(costs)
        w = self._exp_util(total_costs)
        return w

    def _exp_util(self, total_costs):
        w = calc_exp(total_costs)
        return w

    def _compute_mean_covariance(self, costs, actions):
        w = self._exp_util_from_costs(costs)
        w = w.unsqueeze(-1).unsqueeze(-1)
        new_mean = self._compute_mean(w, actions)
        new_cov = self._compute_covariance(w, actions)
        self._update_cov_scale(new_cov)

        return new_mean, new_cov

    @torch.no_grad()
    def _update_distribution(self, trajectories: Trajectory):
        costs = trajectories.costs
        actions = trajectories.actions

        # Let's reshape to n_problems now:

        # first find the means before doing exponential utility:

        # Update best action
        if self.sample_mode == SampleMode.BEST:
            w = self._exp_util_from_costs(costs)

            best_idx = torch.argmax(w, dim=-1)
            self.best_traj.copy_(actions[self.problem_col, best_idx])

        if self.store_rollouts and self.visual_traj is not None:
            total_costs = self._compute_total_cost(costs)
            vis_seq = getattr(trajectories.state, self.visual_traj)
            top_values, top_idx = torch.topk(total_costs, 20, dim=1)
            self.top_values = top_values
            self.top_idx = top_idx
            top_trajs = torch.index_select(vis_seq, 0, top_idx[0])
            for i in range(1, top_idx.shape[0]):
                trajs = torch.index_select(
                    vis_seq, 0, top_idx[i] + (self.particles_per_problem * i)
                )
                top_trajs = torch.cat((top_trajs, trajs), dim=0)
            if self.top_trajs is None or top_trajs.shape != self.top_trajs:
                self.top_trajs = top_trajs
            else:
                self.top_trajs.copy_(top_trajs)

        if not self.update_cov:
            w = w.unsqueeze(-1).unsqueeze(-1)

            new_mean = self._compute_mean(w, actions)
        else:
            new_mean, new_cov = self._compute_mean_covariance(costs, actions)
            self.cov_action.copy_(new_cov)

        self.mean_action.copy_(new_mean)


@get_torch_jit_decorator()
def calc_exp(
    total_costs,
):
    total_costs = -1.0 * total_costs
    # total_costs[torch.abs(total_costs) < 5.0] == 0.0

    w = (total_costs - torch.mean(total_costs, keepdim=True, dim=-1)) / torch.std(
        total_costs, keepdim=True, dim=-1
    )
    return w


@get_torch_jit_decorator()
def compute_es_mean(
    w, actions, mean_action, full_inv_cov, num_particles: int, learning_rate: float
):
    std_w = torch.std(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
    # std_w = torch.sqrt(torch.sum(w - torch.mean(w, dim=[1,2,3], keepdim=True))/float(num_particles))
    a_og = (actions - mean_action.unsqueeze(1)) / std_w
    weighted_seq = (
        (torch.sum(w * a_og, dim=-3, keepdim=True)) @ (full_inv_cov / num_particles)
    ).squeeze(1)

    # weighted_seq[weighted_seq != weighted_seq] = 0.0

    # 0.01 is the learning rate:
    new_mean = mean_action + learning_rate * weighted_seq  # torch.clamp(weighted_seq, -1000, 1000)
    return new_mean
