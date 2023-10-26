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
from itertools import product

# Third Party
import torch

# Local Folder
from .cost_base import CostBase, CostConfig


@dataclass
class ManipulabilityCostConfig(CostConfig):
    use_joint_limits: bool = False

    def __post_init__(self):
        return super().__post_init__()


class ManipulabilityCost(CostBase, ManipulabilityCostConfig):
    def __init__(self, config: ManipulabilityCostConfig):
        ManipulabilityCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
        self.i_mat = torch.ones(
            (6, 1), device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )

        self.delta_vector = torch.zeros(
            (64, 1, 1, 6, 1), device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )
        x = [i for i in product(range(2), repeat=6)]
        self.delta_vector[:, 0, 0, :, 0] = torch.as_tensor(
            x, device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )
        self.delta_vector[self.delta_vector == 0] = -1.0

        if self.cost_fn is None:
            if self.use_joint_limits and self.joint_limits is not None:
                self.cost_fn = self.joint_limited_manipulability_delta
            else:
                self.cost_fn = self.manipulability

    def forward(self, jac_batch, q, qdot):
        b, h, n = q.shape
        if self.use_nn:
            q = q.view(b * h, n)
        score = self.cost_fn(q, jac_batch, qdot)
        if self.use_nn:
            score = score.view(b, h)
        score[score > self.hinge_value] = self.hinge_value
        score = (self.hinge_value / score) - 1
        cost = self.weight * score

        return cost

    def manipulability(self, q, jac_batch, qdot=None):
        with torch.cuda.amp.autocast(enabled=False):
            J_J_t = torch.matmul(jac_batch, jac_batch.transpose(-2, -1))
            score = torch.sqrt(torch.det(J_J_t))

        score[score != score] = 0.0

        return score

    def joint_limited_manipulability_delta(self, q, jac_batch, qdot=None):
        # q is [b,h,dof]
        q_low = q - self.joint_limits[:, 0]
        q_high = q - self.joint_limits[:, 1]

        d_h_1 = torch.square(self.joint_limits[:, 1] - self.joint_limits[:, 0]) * (q_low + q_high)
        d_h_2 = 4.0 * (torch.square(q_low) * torch.square(q_high))
        d_h = torch.div(d_h_1, d_h_2)

        dh_term = 1.0 / torch.sqrt(1 + torch.abs(d_h))
        f_ten = torch.tensor(1.0, **self.tensor_args)
        q_low = torch.abs(q_low)
        q_high = torch.abs(q_high)
        p_plus = torch.where(q_low > q_high, dh_term, f_ten).unsqueeze(-2)
        p_minus = torch.where(q_low > q_high, f_ten, dh_term).unsqueeze(-2)

        j_sign = torch.sign(jac_batch)
        l_delta = torch.sign(self.delta_vector) * j_sign

        L = torch.where(l_delta < 0.0, p_minus, p_plus)

        with torch.cuda.amp.autocast(enabled=False):
            w_J = L * jac_batch
            J_J_t = torch.matmul(w_J, w_J.transpose(-2, -1))
            score = torch.sqrt(torch.det(J_J_t))

        # get actual score:
        min_score = torch.min(score, dim=0)[0]
        max_score = torch.max(score, dim=0)[0]
        score = min_score / max_score
        score[score != score] = 0.0
        return score
