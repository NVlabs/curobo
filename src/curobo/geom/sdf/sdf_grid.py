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
"""Module contains deprecated code for computing Signed Distance Field and it's gradient."""
# Third Party
import torch


# @get_torch_jit_decorator()
def lookup_distance(pt, dist_matrix_flat, num_voxels):
    """Lookup distance from distance matrix."""
    # flatten:
    ind_pt = (
        (pt[..., 0]) * (num_voxels[1] * num_voxels[2]) + pt[..., 1] * num_voxels[2] + pt[..., 2]
    )
    dist = dist_matrix_flat[ind_pt]
    return dist


# @get_torch_jit_decorator()
def compute_sdf_gradient(pt, dist_matrix_flat, num_voxels, dist):
    """Compute gradient of SDF."""
    grad_l = []
    for i in range(3):  # x,y,z
        pt_n = pt.clone()
        pt_p = pt.clone()
        pt_n[..., i] -= 1
        pt_p[..., i] += 1
        # get distance from minus 1 and plus 1 idx:
        pt_n[pt_n < 0] = 0
        # pt_n[pt_n>nu]
        # pt_p[pt_p > num_voxels] = num_voxels[i]
        d_n = lookup_distance(pt_n, dist_matrix_flat, num_voxels)
        d_p = lookup_distance(pt_p, dist_matrix_flat, num_voxels)
        mask = d_n < d_p
        dx = torch.where(mask, -1, 1)
        ds = torch.where(mask, d_n - dist, d_p - dist)
        g_d = ds / dx
        grad_l.append(g_d)
        # print(i, dist,  pt)
    g_pt = torch.stack(grad_l, dim=-1)
    # g_pt = g_pt/torch.linalg.norm(g_pt, dim=-1, keepdim=True)
    return g_pt


class SDFGrid(torch.autograd.Function):
    """Sdf grid torch function."""

    @staticmethod
    def forward(ctx, pt, dist_matrix_flat, num_voxels):
        # input = x_id,y_id,z_id
        pt = (pt).to(dtype=torch.int64)

        dist = lookup_distance(pt, dist_matrix_flat, num_voxels)
        ctx.save_for_backward(pt, dist_matrix_flat, num_voxels, dist)
        return dist.unsqueeze(-1)

    @staticmethod
    def backward(ctx, grad_output):
        pt, dist_matrix_flat, num_voxels, dist = ctx.saved_tensors
        grad_pt = grad_voxels = grad_matrix_flat = None
        if ctx.needs_input_grad[0]:
            pt = pt.to(dtype=torch.int64)
            g_pt = compute_sdf_gradient(pt, dist_matrix_flat, num_voxels, dist)
            # print(g_pt)
            grad_pt = grad_output * g_pt
        if ctx.needs_input_grad[1]:
            raise NotImplementedError("SDFGrid: Can't get gradient w.r.t. dist_matrix")
        if ctx.needs_input_grad[2]:
            raise NotImplementedError("SDFGrid: Can't get gradient w.r.t. num_voxels")
        return grad_pt, grad_matrix_flat, grad_voxels
