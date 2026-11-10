# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""PyTorch autograd functions for Warp-based collision queries.

These unified wrappers query all registered obstacle types in a
scene and accumulate results into a single buffer. The generic collision kernels
handle type dispatch via Warp function overloading.

Autograd Functions:
- SphereObstacleCollision: Regular sphere collision (single timestep)
- SweptSphereObstacleCollision: Swept sphere collision (trajectory interpolation)
  with optional speed metric scaling for motion-aware collision
"""

from __future__ import annotations

# Standard Library
from typing import TYPE_CHECKING

# Third Party
import torch
import warp as wp

import curobo._src.runtime as curobo_runtime
from curobo._src.geom.collision.wp_collision_kernel import get_tiled_collision_kernel
from curobo._src.geom.collision.wp_sweep_collision_kernel import (
    get_tiled_swept_collision_kernel,
)
from curobo._src.geom.data.data_collision_layers import collision_layers_to_warp
from curobo._src.util.warp import get_warp_device_stream

if TYPE_CHECKING:
    from curobo._src.geom.collision.buffer_collision import CollisionBuffer
    from curobo._src.geom.data.data_scene import SceneData


def select_collision_block_dim(
    num_states: int,
    num_spheres: int,
    num_sms: int,
    is_swept: bool,
) -> int:
    """Choose a state-block width from sphere work and blocks available per SM."""
    if num_states < 1 or num_spheres < 1 or num_sms < 1:
        raise ValueError("num_states, num_spheres, and num_sms must be positive")

    blocks_per_sm = num_states / num_sms
    if num_spheres <= 64:
        block_dim = 64
    elif num_spheres <= 128:
        block_dim = 64 if blocks_per_sm > 8.0 else 128
    elif num_spheres <= 256:
        block_dim = 256 if blocks_per_sm <= 4.0 else 128
    elif num_spheres <= 512:
        if blocks_per_sm <= 2.0:
            block_dim = 512
        elif blocks_per_sm <= 4.0:
            block_dim = 256
        else:
            block_dim = 128
    elif blocks_per_sm <= 1.0:
        block_dim = 1024
    elif blocks_per_sm <= 2.0:
        block_dim = 512
    elif blocks_per_sm <= 8.0:
        block_dim = 256
    else:
        block_dim = 128

    if is_swept and num_sms > 32 and blocks_per_sm > 1.0:
        block_dim = min(block_dim, 256)
    if blocks_per_sm < 0.125:
        block_dim = max(block_dim, 256)
    return block_dim


class SphereObstacleCollision(torch.autograd.Function):
    """Compute sphere collision against all obstacle types in a scene.

    One tiled implementation handles summed and per-sphere outputs. Each block
    owns a rollout state and each lane traverses a strided subset of its spheres.
    """

    @staticmethod
    def forward(
        ctx,
        query_spheres: torch.Tensor,
        buffer: "CollisionBuffer",
        scene: "SceneData",
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        max_distance: torch.Tensor,
        env_query_idx: torch.Tensor,
        use_multi_env: bool,
        return_loss: bool,
        sum_distance: bool,
        sphere_collision_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: compute collision distance to all scene obstacles.

        Args:
            ctx: Autograd context used to retain the precomputed gradient.
            query_spheres: Sphere positions and radii (batch, horizon, num_spheres, 4).
            buffer: Collision buffer for output accumulation.
            scene: SceneData containing obstacle data.
            weight: Collision cost weight scalar tensor.
            activation_distance: Distance threshold for collision activation.
            max_distance: Maximum query distance for mesh SDF.
            env_query_idx: Environment index per batch element.
            use_multi_env: Whether to use batch-specific environments.
            return_loss: If True, backward uses grad_output for scaling.
            sum_distance: Sum collision costs across robot spheres.
            sphere_collision_mask: Shared uint32 masks [kinematics num_envs, num_spheres].

        Returns:
            Collision cost tensor with shape ``(batch, horizon, 1)`` when
            ``sum_distance`` is True, otherwise
            ``(batch, horizon, num_spheres)``.
        """
        b, h, n, _ = query_spheres.shape

        device, stream = get_warp_device_stream(query_spheres)
        spheres_wp = wp.from_torch(query_spheres.detach().view(-1, 4), dtype=wp.vec4)
        mask_wp = wp.from_torch(sphere_collision_mask.view(torch.int32), dtype=wp.uint32)
        env_idx_wp = wp.from_torch(env_query_idx.view(-1), dtype=wp.int32)
        weight_wp = wp.from_torch(weight)
        eta_wp = wp.from_torch(activation_distance)
        out_cost_wp = wp.from_torch(buffer.distance.detach().view(-1))
        out_grad_wp = wp.from_torch(buffer.gradient.detach().view(-1), dtype=wp.float32)
        use_multi_env_wp = wp.uint8(use_multi_env)
        sum_distance_wp = wp.uint8(sum_distance)

        valid_data = scene.get_valid_data()
        if not valid_data:
            buffer.zero_()
        else:
            actual_sms = torch.cuda.get_device_properties(
                query_spheres.device
            ).multi_processor_count
            num_sms = curobo_runtime.resolve_kernel_launch_sm_count(actual_sms)
            block_dim = select_collision_block_dim(b * h, n, num_sms, False)
            for data_idx, data in enumerate(valid_data):
                kernel = get_tiled_collision_kernel(data.max_n, h, n, block_dim)
                wp.launch_tiled(
                    kernel,
                    dim=(b * h,),
                    inputs=[
                        data.to_warp(),
                        spheres_wp,
                        weight_wp,
                        eta_wp,
                        env_idx_wp,
                        out_cost_wp,
                        out_grad_wp,
                        use_multi_env_wp,
                        wp.uint8(data_idx == 0),
                        sum_distance_wp,
                        collision_layers_to_warp(data.obstacle_layers),
                        mask_wp,
                    ],
                    stream=stream,
                    device=device,
                    block_dim=block_dim,
                )
        ctx.return_loss = return_loss
        ctx.save_for_backward(buffer.gradient)
        if sum_distance:
            return buffer.distance.view(-1)[: b * h].view(b, h, 1)
        return buffer.distance

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: return pre-computed gradient from buffer."""
        grad_sph = None
        if ctx.needs_input_grad[0]:
            (grad_buffer,) = ctx.saved_tensors
            grad_sph = grad_buffer
            if ctx.return_loss:
                grad_sph = grad_buffer * grad_output.unsqueeze(-1)
        return (
            grad_sph,  # query_spheres
            None,  # buffer
            None,  # scene
            None,  # weight
            None,  # activation_distance
            None,  # max_distance
            None,  # env_query_idx
            None,  # use_multi_env
            None,  # return_loss
            None,  # sum_distance
            None,  # sphere_collision_mask
        )


class SweptSphereObstacleCollision(torch.autograd.Function):
    """Compute swept sphere collision against all obstacle types in a scene.

    This unified function handles sweep interpolation between trajectory timesteps,
    accumulates results into a single buffer, and optionally applies speed metric
    scaling for motion-aware collision checking.
    """

    @staticmethod
    def forward(
        ctx,
        query_spheres: torch.Tensor,
        buffer: "CollisionBuffer",
        scene: "SceneData",
        weight: torch.Tensor,
        activation_distance: torch.Tensor,
        max_distance: torch.Tensor,
        speed_dt: torch.Tensor,
        enable_speed_metric: bool,
        env_query_idx: torch.Tensor,
        use_multi_env: bool,
        return_loss: bool,
        sum_distance: bool,
        sphere_collision_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: compute swept collision distance to all scene obstacles.

        Args:
            ctx: Autograd context used to retain the precomputed gradient.
            query_spheres: Sphere positions and radii (batch, horizon, num_spheres, 4).
            buffer: Collision buffer for output accumulation.
            scene: SceneData containing obstacle data.
            weight: Collision cost weight scalar tensor.
            activation_distance: Distance threshold for collision activation.
            max_distance: Maximum query distance for mesh SDF.
            speed_dt: Time delta between trajectory steps for speed metric.
            enable_speed_metric: Scale collision cost by sphere velocity.
            env_query_idx: Environment index per batch element.
            use_multi_env: Whether to use batch-specific environments.
            return_loss: If True, backward uses grad_output for scaling.
            sum_distance: Sum collision costs across robot spheres.
            sphere_collision_mask: Shared uint32 masks [kinematics num_envs, num_spheres].

        Returns:
            Collision cost tensor with shape ``(batch, horizon, 1)`` when
            ``sum_distance`` is True, otherwise
            ``(batch, horizon, num_spheres)``.
        """
        b, h, n, _ = query_spheres.shape

        device, stream = get_warp_device_stream(query_spheres)
        spheres_wp = wp.from_torch(query_spheres.detach().view(-1, 4), dtype=wp.vec4)
        mask_wp = wp.from_torch(sphere_collision_mask.view(torch.int32), dtype=wp.uint32)
        env_idx_wp = wp.from_torch(env_query_idx.view(-1), dtype=wp.int32)
        weight_wp = wp.from_torch(weight)
        eta_wp = wp.from_torch(activation_distance)
        speed_dt_wp = wp.from_torch(speed_dt)
        out_cost_wp = wp.from_torch(buffer.distance.detach().view(-1))
        out_grad_wp = wp.from_torch(buffer.gradient.detach().view(-1), dtype=wp.float32)
        use_multi_env_wp = wp.uint8(use_multi_env)
        sum_distance_wp = wp.uint8(sum_distance)

        valid_data = scene.get_valid_data()
        if not valid_data:
            buffer.zero_()
        else:
            actual_sms = torch.cuda.get_device_properties(
                query_spheres.device
            ).multi_processor_count
            num_sms = curobo_runtime.resolve_kernel_launch_sm_count(actual_sms)
            block_dim = select_collision_block_dim(b * h, n, num_sms, True)
            for data_idx, data in enumerate(valid_data):
                kernel = get_tiled_swept_collision_kernel(
                    data.max_n, enable_speed_metric, h, n, block_dim
                )
                wp.launch_tiled(
                    kernel,
                    dim=(b * h,),
                    inputs=[
                        data.to_warp(),
                        spheres_wp,
                        weight_wp,
                        eta_wp,
                        speed_dt_wp,
                        env_idx_wp,
                        out_cost_wp,
                        out_grad_wp,
                        use_multi_env_wp,
                        wp.uint8(data_idx == 0),
                        sum_distance_wp,
                        collision_layers_to_warp(data.obstacle_layers),
                        mask_wp,
                    ],
                    stream=stream,
                    device=device,
                    block_dim=block_dim,
                )
        ctx.return_loss = return_loss
        ctx.save_for_backward(buffer.gradient)
        if sum_distance:
            return buffer.distance.view(-1)[: b * h].view(b, h, 1)
        return buffer.distance

    @staticmethod
    def backward(ctx, grad_output):
        """Backward pass: return pre-computed gradient from buffer."""
        grad_sph = None
        if ctx.needs_input_grad[0]:
            (grad_buffer,) = ctx.saved_tensors
            grad_sph = grad_buffer
            if ctx.return_loss:
                grad_sph = grad_buffer * grad_output.unsqueeze(-1)
        return (
            grad_sph,  # query_spheres
            None,  # buffer
            None,  # scene
            None,  # weight
            None,  # activation_distance
            None,  # max_distance
            None,  # speed_dt
            None,  # enable_speed_metric
            None,  # env_query_idx
            None,  # use_multi_env
            None,  # return_loss
            None,  # sum_distance
            None,  # sphere_collision_mask
        )
