# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Standard Library
from enum import Enum

# Third Party
import torch

# CuRobo


class SquashType(Enum):
    CLAMP = 0
    CLAMP_RESCALE = 1
    TANH = 2
    IDENTITY = 3


def scale_ctrl(ctrl, action_lows, action_highs, squash_fn: SquashType = SquashType.CLAMP):
    if len(ctrl.shape) == 1:
        ctrl = ctrl.unsqueeze(0).unsqueeze(-1)
    act_half_range = (action_highs - action_lows) / 2.0
    act_mid_range = (action_highs + action_lows) / 2.0
    if squash_fn == SquashType.CLAMP:
        # ctrl = torch.clamp(ctrl, action_lows[0], action_highs[0])
        ctrl = torch.max(torch.min(ctrl, action_highs), action_lows)
        return ctrl
    elif squash_fn == SquashType.CLAMP_RESCALE:
        ctrl = torch.clamp(ctrl, -1.0, 1.0)
    elif squash_fn == SquashType.TANH:
        ctrl = torch.tanh(ctrl)
    elif squash_fn == SquashType.IDENTITY:
        return ctrl
    return act_mid_range.unsqueeze(0) + ctrl * act_half_range.unsqueeze(0)


########################
## Gaussian Utilities ##
########################


def gaussian_entropy(cov=None, L=None):  # , cov_type="full"):
    """Entropy of multivariate gaussian given either covariance
    or cholesky decomposition of covariance

    """
    if cov is not None:
        inp_device = cov.device
        cov_logdet = torch.log(torch.det(cov))
        # print(np.linalg.det(cov.cpu().numpy()))
        # print(torch.det(cov))
        N = cov.shape[0]

    else:
        inp_device = L.device
        cov_logdet = 2.0 * torch.sum(torch.log(torch.diagonal(L)))
        N = L.shape[0]
    # if cov_type == "diagonal":
    # cov_logdet =  np.sum(np.log(cov.diagonal()))
    # else:
    # cov_logdet = np.log(np.linalg.det(cov))

    term1 = 0.5 * cov_logdet
    # pi = torch.tensor([math.pi], device=inp_device)
    # pre-calculate 1.0 + torch.log(2.0*pi) = 2.837877066
    term2 = 0.5 * N * 2.837877066

    ent = term1 + term2
    return ent.to(inp_device)


# @get_torch_jit_decorator()
def cost_to_go(cost_seq, gamma_seq, only_first=False):
    # type: (Tensor, Tensor, bool) -> Tensor
    """Calculate (discounted) cost to go for given cost sequence"""
    # if torch.any(gamma_seq == 0):
    #     return cost_seq
    cost_seq = gamma_seq * cost_seq  # discounted cost sequence
    if only_first:
        cost_seq = torch.sum(cost_seq, dim=-1, keepdim=True) / gamma_seq[..., 0]
    else:
        # cost_seq = torch.cumsum(cost_seq[:, ::-1], axis=-1)[:, ::-1]  # cost to go (but scaled by [1 , gamma, gamma*2 and so on])
        cost_seq = torch.fliplr(
            torch.cumsum(torch.fliplr(cost_seq), dim=-1)
        )  # cost to go (but scaled by [1 , gamma, gamma*2 and so on])
        cost_seq /= gamma_seq  # un-scale it to get true discounted cost to go
    return cost_seq


############
##Cholesky##
############
def matrix_cholesky(A):
    L = torch.zeros_like(A)
    for i in range(A.shape[-1]):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s = s + L[i, k] * L[j, k]

            L[i, j] = torch.sqrt(A[i, i] - s) if (i == j) else (1.0 / L[j, j] * (A[i, j] - s))
    return L


# Batched Cholesky decomp
def batch_cholesky(A):
    L = torch.zeros_like(A)

    for i in range(A.shape[-1]):
        for j in range(i + 1):
            s = 0.0
            for k in range(j):
                s = s + L[..., i, k] * L[..., j, k]

            L[..., i, j] = (
                torch.sqrt(A[..., i, i] - s)
                if (i == j)
                else (1.0 / L[..., j, j] * (A[..., i, j] - s))
            )
    return L
