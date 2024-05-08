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
from curobo.rollout.dynamics_model.kinematic_model import TimeTrajConfig
from curobo.util.torch_utils import get_torch_jit_decorator

# Local Folder
from .cost_base import CostBase, CostConfig


@dataclass
class StopCostConfig(CostConfig):
    max_limit: Optional[float] = None
    max_nlimit: Optional[float] = None
    dt_traj_params: Optional[TimeTrajConfig] = None
    horizon: int = 1

    def __post_init__(self):
        return super().__post_init__()


class StopCost(CostBase, StopCostConfig):
    def __init__(self, config: StopCostConfig):
        StopCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
        sum_matrix = torch.tril(
            torch.ones(
                (self.horizon, self.horizon),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
        ).T
        traj_dt = self.tensor_args.to_device(self.dt_traj_params.get_dt_array(self.horizon))
        if self.max_nlimit is not None:
            # every timestep max acceleration:
            sum_matrix = torch.tril(
                torch.ones(
                    (self.horizon, self.horizon),
                    device=self.tensor_args.device,
                    dtype=self.tensor_args.dtype,
                )
            ).T
            delta_vel = traj_dt * self.max_nlimit
            self.max_vel = (sum_matrix @ delta_vel).unsqueeze(-1)
        elif self.max_limit is not None:
            sum_matrix = torch.tril(
                torch.ones(
                    (self.horizon, self.horizon),
                    device=self.tensor_args.device,
                    dtype=self.tensor_args.dtype,
                )
            ).T
            delta_vel = torch.ones_like(traj_dt) * self.max_limit
            self.max_vel = (sum_matrix @ delta_vel).unsqueeze(-1)

    def forward(self, vels):
        cost = velocity_cost(vels, self.weight, self.max_vel)
        return cost


@get_torch_jit_decorator()
def velocity_cost(vels, weight, max_vel):
    vel_abs = torch.abs(vels)
    vel_abs = torch.nn.functional.relu(vel_abs - max_vel[: vels.shape[1]])
    cost = weight * (torch.sum(vel_abs**2, dim=-1))

    return cost
