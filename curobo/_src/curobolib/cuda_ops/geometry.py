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
        block_batch_max_value: torch.Tensor,
        block_batch_max_index: torch.Tensor,
        num_blocks_per_batch: int,
        max_threads_per_block: int,
        store_pair_distance: bool,
        return_loss: bool,
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
            block_batch_max_value=block_batch_max_value,
        )
        check_uint8_tensors(device, sparse_idx=sparse_idx)
        check_int16_tensors(
            device,
            pair_locations=pair_locations,
            block_batch_max_index=block_batch_max_index,
        )
        b, h, num_spheres, _ = robot_spheres.shape
        num_collision_pairs = pair_locations.shape[0]

        geometry_cu.self_collision_distance(
            out_distance,
            out_vec,
            pair_distance,
            sparse_idx,
            robot_spheres,  # .view(-1, 4),
            sphere_padding,
            weight,
            pair_locations,
            block_batch_max_value,
            block_batch_max_index,
            num_blocks_per_batch,
            max_threads_per_block,
            b,
            h,
            num_spheres,
            num_collision_pairs,
            store_pair_distance,
            robot_spheres.requires_grad,
        )

        ctx.return_loss = return_loss
        ctx.mark_non_differentiable(
            out_vec,
            sparse_idx,
            robot_spheres,
            sphere_padding,
            weight,
            pair_locations,
            block_batch_max_value,
            block_batch_max_index,
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
                    g_vec = g_vec * grad_out_distance
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
            None,
        )

