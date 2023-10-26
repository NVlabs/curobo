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
from curobo.opt.particle.parallel_mppi import CovType, ParallelMPPI, ParallelMPPIConfig


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
        return new_mean

    def _exp_util(self, total_costs):
        w = calc_exp(total_costs)
        return w


@torch.jit.script
def calc_exp(
    total_costs,
):
    total_costs = -1.0 * total_costs
    # total_costs[torch.abs(total_costs) < 5.0] == 0.0

    w = (total_costs - torch.mean(total_costs, keepdim=True, dim=-1)) / torch.std(
        total_costs, keepdim=True, dim=-1
    )
    return w


@torch.jit.script
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
