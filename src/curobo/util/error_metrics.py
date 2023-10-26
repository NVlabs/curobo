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
import math

# Third Party
import numpy as np
import torch


def rotation_error_quaternion(q_des, q):
    #
    sum_q = torch.norm(q_des + q)
    diff_q = torch.norm(q_des - q)
    # err = torch.minimum(sum_q, diff_q) / math.sqrt(2)
    err = np.minimum(sum_q.cpu().numpy(), diff_q.cpu().numpy()) / math.sqrt(2)
    return err


def rotation_error_matrix(r_des, r):
    #
    """
    px = torch.tensor([1.0,0.0,0.0],device=r_des.device).T
    py = torch.tensor([0.0,1.0,0.0],device=r_des.device).T
    pz = torch.tensor([0.0,0.0,1.0],device=r_des.device).T
    print(px.shape, r.shape)

    current_px = r * px
    current_py = r * py
    current_pz = r * pz

    des_px = r_des * px
    des_py = r_des * py
    des_pz = r_des * pz

    cost = torch.norm(current_px -  des_px) +  torch.norm(current_py -  des_py) + torch.norm(current_pz -  des_pz)
    return cost
    """
    rot_delta = r - r_des
    cost = 0.5 * torch.sum(torch.square(rot_delta), dim=-2)
    cost = torch.sum(cost, dim=-1)
    return cost
