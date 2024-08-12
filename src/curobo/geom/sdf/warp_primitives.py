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
"""Warp-lang based world collision functions are implemented as torch autograd functions."""

# Third Party
import torch
import warp as wp

wp.set_module_options({"fast_math": False})

# CuRobo
from curobo.util.warp import warp_support_sdf_struct

# Check version of warp and import the supported SDF function.
if warp_support_sdf_struct():
    # Local Folder
    from .warp_sdf_fns import get_closest_pt_batch_env, get_swept_closest_pt_batch_env
else:
    # Local Folder
    from .warp_sdf_fns_deprecated import get_closest_pt_batch_env, get_swept_closest_pt_batch_env


class SdfMeshWarpPy(torch.autograd.Function):
    """Pytorch autograd function for computing signed distance between spheres and meshes."""

    @staticmethod
    def forward(
        ctx,
        query_spheres,
        out_cost,
        out_grad,
        sparsity_idx,
        weight,
        activation_distance,
        mesh_idx,
        mesh_pose_inverse,
        mesh_enable,
        n_env_mesh,
        max_dist,
        env_query_idx=None,
        return_loss=False,
        compute_esdf=False,
    ):
        b, h, n, _ = query_spheres.shape
        use_batch_env = True
        if env_query_idx is None:
            use_batch_env = False
            env_query_idx = n_env_mesh
        requires_grad = query_spheres.requires_grad
        wp.launch(
            kernel=get_closest_pt_batch_env,
            dim=b * h * n,
            inputs=[
                wp.from_torch(query_spheres.detach().view(-1, 4), dtype=wp.vec4),
                wp.from_torch(out_cost.view(-1)),
                wp.from_torch(out_grad.view(-1), dtype=wp.float32),
                wp.from_torch(sparsity_idx.view(-1), dtype=wp.uint8),
                wp.from_torch(weight),
                wp.from_torch(activation_distance),
                wp.from_torch(mesh_idx.view(-1), dtype=wp.uint64),
                wp.from_torch(mesh_pose_inverse.view(-1), dtype=wp.float32),
                wp.from_torch(mesh_enable.view(-1), dtype=wp.uint8),
                wp.from_torch(n_env_mesh.view(-1), dtype=wp.int32),
                wp.from_torch(max_dist, dtype=wp.float32),
                requires_grad,
                b,
                h,
                n,
                mesh_idx.shape[1],
                wp.from_torch(env_query_idx.view(-1), dtype=wp.int32),
                use_batch_env,
                compute_esdf,
            ],
            stream=wp.stream_from_torch(query_spheres.device),
        )
        ctx.return_loss = return_loss
        ctx.save_for_backward(out_grad)
        return out_cost

    @staticmethod
    def backward(ctx, grad_output):
        grad_sph = None
        if ctx.needs_input_grad[0]:
            (r,) = ctx.saved_tensors
            grad_sph = r
            if ctx.return_loss:
                grad_sph = r * grad_output.unsqueeze(-1)
        return (
            grad_sph,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class SweptSdfMeshWarpPy(torch.autograd.Function):
    """Compute signed distance between trajectory of spheres and meshes."""

    @staticmethod
    def forward(
        ctx,
        query_spheres,
        out_cost,
        out_grad,
        sparsity_idx,
        weight,
        activation_distance,
        speed_dt,
        mesh_idx,
        mesh_pose_inverse,
        mesh_enable,
        n_env_mesh,
        max_dist,
        sweep_steps=1,
        enable_speed_metric=False,
        env_query_idx=None,
        return_loss=False,
    ):
        b, h, n, _ = query_spheres.shape
        use_batch_env = True
        if env_query_idx is None:
            use_batch_env = False
            env_query_idx = n_env_mesh
        requires_grad = query_spheres.requires_grad

        wp.launch(
            kernel=get_swept_closest_pt_batch_env,
            dim=b * h * n,
            inputs=[
                wp.from_torch(query_spheres.detach().view(-1, 4), dtype=wp.vec4),
                wp.from_torch(out_cost.view(-1)),
                wp.from_torch(out_grad.view(-1), dtype=wp.float32),
                wp.from_torch(sparsity_idx.view(-1), dtype=wp.uint8),
                wp.from_torch(weight),
                wp.from_torch(activation_distance),
                wp.from_torch(speed_dt),
                wp.from_torch(mesh_idx.view(-1), dtype=wp.uint64),
                wp.from_torch(mesh_pose_inverse.view(-1), dtype=wp.float32),
                wp.from_torch(mesh_enable.view(-1), dtype=wp.uint8),
                wp.from_torch(n_env_mesh.view(-1), dtype=wp.int32),
                wp.from_torch(max_dist, dtype=wp.float32),
                requires_grad,
                b,
                h,
                n,
                mesh_idx.shape[1],
                sweep_steps,
                enable_speed_metric,
                wp.from_torch(env_query_idx.view(-1), dtype=wp.int32),
                use_batch_env,
            ],
            stream=wp.stream_from_torch(query_spheres.device),
        )
        ctx.return_loss = return_loss
        ctx.save_for_backward(out_grad)
        return out_cost

    @staticmethod
    def backward(ctx, grad_output):
        grad_sph = None
        if ctx.needs_input_grad[0]:
            (r,) = ctx.saved_tensors
            grad_sph = r
            if ctx.return_loss:
                grad_sph = grad_sph * grad_output.unsqueeze(-1)
        return (
            grad_sph,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
