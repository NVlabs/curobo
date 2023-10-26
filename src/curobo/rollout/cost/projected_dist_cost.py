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
"""
Distance cost projected into the null-space of the Jacobian
"""

# Standard Library
from dataclasses import dataclass
from enum import Enum

# Third Party
import torch

# Local Folder
from .dist_cost import DistCost, DistCostConfig


class ProjType(Enum):
    IDENTITY = 0
    PSEUDO_INVERSE = 1


@dataclass
class ProjectedDistCostConfig(DistCostConfig):
    eps: float = 1e-4
    proj_type: ProjType = ProjType.IDENTITY

    def __post_init__(self):
        return super().__post_init__()


class ProjectedDistCost(DistCost, ProjectedDistCostConfig):
    def __init__(self, config: ProjectedDistCostConfig):
        ProjectedDistCostConfig.__init__(self, **vars(config))
        DistCost.__init__(self)
        self.I = torch.eye(self.dof, device=self.tensor_args.device, dtype=self.tensor_args.dtype)
        self.task_I = torch.eye(6, device=self.tensor_args.device, dtype=self.tensor_args.dtype)

    def forward(self, disp_vec, jac_batch=None):
        disp_vec = self.vec_weight * disp_vec

        if self.proj_type == ProjType.PSEUDO_INVERSE:
            disp_vec_projected = self.get_pinv_null_disp(disp_vec, jac_batch)
        elif self.proj_type == ProjType.IDENTITY:
            disp_vec_projected = disp_vec

        return super().forward(disp_vec_projected)

    def get_pinv_null_disp(self, disp_vec, jac_batch):
        jac_batch_t = jac_batch.transpose(-2, -1)

        J_J_t = torch.matmul(jac_batch, jac_batch_t)

        J_pinv = jac_batch_t @ torch.inverse(J_J_t + self.eps * self.task_I.expand_as(J_J_t))

        J_pinv_J = torch.matmul(J_pinv, jac_batch)

        null_proj = self.I.expand_as(J_pinv_J) - J_pinv_J

        null_disp = torch.matmul(null_proj, disp_vec.unsqueeze(-1)).squeeze(-1)
        return null_disp
