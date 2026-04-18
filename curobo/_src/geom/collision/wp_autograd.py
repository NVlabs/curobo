# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""PyTorch autograd functions for Warp-based collision queries.

These unified wrappers query all obstacle types (cuboids, meshes, voxels) in a
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

from curobo._src.geom.collision.wp_collision_kernel import sphere_obstacle_collision_kernel
from curobo._src.geom.collision.wp_speed_metric import apply_speed_metric
from curobo._src.geom.collision.wp_sweep_collision_kernel import (
    swept_sphere_obstacle_collision_kernel,
)
from curobo._src.util.warp import get_warp_device_stream

if TYPE_CHECKING:
    from curobo._src.geom.collision.buffer_collision import CollisionBuffer
    from curobo._src.geom.data.data_scene import SceneData


class SphereObstacleCollision(torch.autograd.Function):
    """Compute sphere collision against all obstacle types in a scene.

    This unified function queries cuboids, meshes, and voxels, accumulating
    results into a single buffer via atomic operations.
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
        return_loss: bool = False,
    ) -> torch.Tensor:
        """Forward pass: compute collision distance to all scene obstacles.

        Args:
            query_spheres: Sphere positions and radii (batch, horizon, num_spheres, 4).
            buffer: Collision buffer for output accumulation.
            scene: SceneData containing obstacle data.
            weight: Collision cost weight scalar tensor.
            activation_distance: Distance threshold for collision activation.
            max_distance: Maximum query distance for mesh SDF.
            env_query_idx: Environment index per batch element.
            use_multi_env: Whether to use batch-specific environments.
            return_loss: If True, backward uses grad_output for scaling.

        Returns:
            Collision distance/cost tensor (batch, horizon, num_spheres).
        """
        b, h, n, _ = query_spheres.shape

        # Zero buffer before accumulation
        buffer.zero_()

        device, stream = get_warp_device_stream(query_spheres)
        spheres_wp = wp.from_torch(query_spheres.detach().view(-1, 4), dtype=wp.vec4)
        env_idx_wp = wp.from_torch(env_query_idx.view(-1), dtype=wp.int32)
        weight_wp = wp.from_torch(weight)
        eta_wp = wp.from_torch(activation_distance)
        out_cost_wp = wp.from_torch(buffer.distance.detach().view(-1))
        out_grad_wp = wp.from_torch(buffer.gradient.detach().view(-1), dtype=wp.float32)
        use_multi_env_wp = wp.uint8(use_multi_env) #wp.uint8(1) if use_multi_env else wp.uint8(0)

        for data in scene.get_valid_data():
            max_n = data.max_n
            data_wp = data.to_warp()
            wp.launch(
                kernel=sphere_obstacle_collision_kernel,
                dim=b * h * n * max_n,
                inputs=[data_wp, spheres_wp, weight_wp, eta_wp, env_idx_wp, out_cost_wp, out_grad_wp, b, h, n, max_n, use_multi_env_wp],
                stream=stream,
                device=device,
            )

        ctx.return_loss = return_loss
        ctx.save_for_backward(buffer.gradient)
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
        return_loss: bool = False,
    ) -> torch.Tensor:
        """Forward pass: compute swept collision distance to all scene obstacles.

        Args:
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

        Returns:
            Collision distance/cost tensor (batch, horizon, num_spheres).
        """
        b, h, n, _ = query_spheres.shape

        # Zero buffer before accumulation
        buffer.zero_()

        device, stream = get_warp_device_stream(query_spheres)
        spheres_wp = wp.from_torch(query_spheres.detach().view(-1, 4), dtype=wp.vec4)
        env_idx_wp = wp.from_torch(env_query_idx.view(-1), dtype=wp.int32)
        weight_wp = wp.from_torch(weight)
        eta_wp = wp.from_torch(activation_distance)
        speed_dt_wp = wp.from_torch(speed_dt)
        out_cost_wp = wp.from_torch(buffer.distance.detach().view(-1))
        out_grad_wp = wp.from_torch(buffer.gradient.detach().view(-1), dtype=wp.float32)
        use_multi_env_wp = wp.uint8(use_multi_env)

        for data in scene.get_valid_data():
            max_n = data.max_n
            data_wp = data.to_warp()
            wp.launch(
                kernel=swept_sphere_obstacle_collision_kernel,
                dim=b * h * n * max_n,
                inputs=[
                    data_wp,
                    spheres_wp,
                    weight_wp,
                    eta_wp,
                    env_idx_wp,
                    out_cost_wp,
                    out_grad_wp,
                    b,
                    h,
                    n,
                    max_n,
                    use_multi_env_wp,
                ],
                stream=stream,
                device=device,
                block_dim=128,
            )


        # Apply speed metric once after all obstacles
        if enable_speed_metric:
            wp.launch(
                kernel=apply_speed_metric,
                dim=b * h * n,
                inputs=[
                    spheres_wp,
                    out_cost_wp,
                    out_grad_wp,
                    speed_dt_wp,
                    b,
                    h,
                    n,
                ],
                stream=stream,
                device=device,
            )

        ctx.return_loss = return_loss
        ctx.save_for_backward(buffer.gradient)
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
        )
