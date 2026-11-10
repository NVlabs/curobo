# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Third Party
from typing import Optional

import torch

from curobo._src.curobolib.backends import geometry as geometry_cu
from curobo._src.curobolib.cuda_ops.tensor_checks import (
    check_float32_tensors,
    check_int16_tensors,
    check_int32_tensors,
    check_uint8_tensors,
)


class SelfCollisionDistance(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        robot_spheres: torch.Tensor,
        out_distance: torch.Tensor,
        out_vec: torch.Tensor,
        pair_distance: torch.Tensor,
        sparse_idx: torch.Tensor,
        weight: torch.Tensor,
        sphere_padding: torch.Tensor,
        pair_locations: torch.Tensor,
        max_threads_per_block: int,
        store_pair_distance: bool,
        return_loss: bool,
        sum_collisions: bool,
        per_sphere_max_collisions: bool,
    ):
        # get batch size
        ctx.set_materialize_grads(False)
        device = robot_spheres.device
        check_float32_tensors(
            device,
            robot_spheres=robot_spheres,
            out_distance=out_distance,
            out_vec=out_vec,
            pair_distance=pair_distance,
            weight=weight,
            sphere_padding=sphere_padding,
        )
        check_uint8_tensors(device, sparse_idx=sparse_idx)
        check_int16_tensors(device, pair_locations=pair_locations)
        b, h, num_spheres, _ = robot_spheres.shape
        num_collision_pairs = pair_locations.shape[0]

        geometry_cu.self_collision_distance(
            out_distance,
            out_vec,
            pair_distance,
            sparse_idx,
            robot_spheres,
            sphere_padding,
            weight,
            pair_locations,
            max_threads_per_block,
            b,
            h,
            num_spheres,
            num_collision_pairs,
            store_pair_distance,
            robot_spheres.requires_grad,
            sum_collisions,
            per_sphere_max_collisions,
        )

        ctx.return_loss = return_loss
        ctx.mark_non_differentiable(
            out_vec,
            sparse_idx,
            robot_spheres,
            sphere_padding,
            weight,
            pair_locations,
        )
        ctx.save_for_backward(out_vec, out_distance)
        return out_distance

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out_distance: Optional[torch.Tensor]):
        sphere_grad = None
        if grad_out_distance is not None:
            if ctx.needs_input_grad[0]:
                (g_vec, g_dist) = ctx.saved_tensors
                if ctx.return_loss:
                    g_vec = g_vec * grad_out_distance.unsqueeze(-1)
                sphere_grad = g_vec

        return (
            sphere_grad,
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


class SelfCollisionLinkAabbDistance(torch.autograd.Function):
    """Self-collision with world-space link AABBs and exact grouped sphere pairs."""

    @staticmethod
    def forward(
        ctx,
        robot_spheres: torch.Tensor,
        out_distance: torch.Tensor,
        out_vec: torch.Tensor,
        sparse_idx: torch.Tensor,
        weight: torch.Tensor,
        sphere_padding: torch.Tensor,
        pair_locations: torch.Tensor,
        link_sphere_indices: torch.Tensor,
        link_sphere_offsets: torch.Tensor,
        link_pairs: torch.Tensor,
        link_pair_offsets: torch.Tensor,
        max_threads_per_block: int,
        return_loss: bool,
        sum_collisions: bool,
        per_sphere_max_collisions: bool,
    ):
        """Run fixed-shape broadphase and exact self-collision evaluation."""
        ctx.set_materialize_grads(False)
        device = robot_spheres.device
        check_float32_tensors(
            device,
            robot_spheres=robot_spheres,
            out_distance=out_distance,
            out_vec=out_vec,
            weight=weight,
            sphere_padding=sphere_padding,
        )
        check_uint8_tensors(device, sparse_idx=sparse_idx)
        check_int16_tensors(
            device,
            pair_locations=pair_locations,
            link_sphere_indices=link_sphere_indices,
            link_pairs=link_pairs,
        )
        check_int32_tensors(
            device,
            link_sphere_offsets=link_sphere_offsets,
            link_pair_offsets=link_pair_offsets,
        )
        batch_size, horizon, num_spheres, _ = robot_spheres.shape
        num_collision_links = link_sphere_offsets.shape[0] - 1
        num_link_pairs = link_pairs.shape[0]

        geometry_cu.self_collision_distance_link_aabb(
            out_distance,
            out_vec,
            sparse_idx,
            robot_spheres,
            sphere_padding,
            weight,
            pair_locations,
            link_sphere_indices,
            link_sphere_offsets,
            link_pairs,
            link_pair_offsets,
            max_threads_per_block,
            batch_size,
            horizon,
            num_spheres,
            num_collision_links,
            num_link_pairs,
            robot_spheres.requires_grad,
            sum_collisions,
            per_sphere_max_collisions,
        )

        ctx.return_loss = return_loss
        ctx.mark_non_differentiable(
            out_vec,
            sparse_idx,
            robot_spheres,
            sphere_padding,
            weight,
            pair_locations,
            link_sphere_indices,
            link_sphere_offsets,
            link_pairs,
            link_pair_offsets,
        )
        ctx.save_for_backward(out_vec, out_distance)
        return out_distance

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out_distance: Optional[torch.Tensor]):
        """Return the exact sphere gradients precomputed during forward."""
        sphere_grad = None
        if grad_out_distance is not None and ctx.needs_input_grad[0]:
            sphere_grad, _ = ctx.saved_tensors
            if ctx.return_loss:
                sphere_grad = sphere_grad * grad_out_distance.unsqueeze(-1)

        return (
            sphere_grad,
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
