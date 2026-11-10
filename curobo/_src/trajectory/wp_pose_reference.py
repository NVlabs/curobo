# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluate a Cartesian reference at per-seed geometric progress on the GPU."""

from __future__ import annotations

import torch
import warp as wp

from curobo._src.util.warp import get_warp_device_stream, warp_kernel


@warp_kernel("sample_pose_reference_linear_v1")
def _sample_pose_reference(
    position: wp.array(dtype=wp.vec3),
    quaternion: wp.array(dtype=wp.vec4),
    progress: wp.array(dtype=wp.float32),
    routes: wp.array(dtype=wp.int32),
    out_position: wp.array(dtype=wp.vec3),
    out_quaternion: wp.array(dtype=wp.vec4),
    batch: wp.int32,
    samples: wp.int32,
    horizon: wp.int32,
    links: wp.int32,
    goals: wp.int32,
    padding: wp.int32,
):
    tid = wp.tid()
    width = links * goals
    if tid >= batch * samples * width:
        return
    lane = tid / (samples * width)
    sample = (tid / width) % samples
    item = tid % width
    active = horizon - 2 * padding
    sample_progress = progress[lane * samples + sample]
    # Validation rejects nonfinite progress; keep its address calculation in bounds first.
    if not wp.isfinite(sample_progress):
        sample_progress = 0.0
    coordinate = wp.clamp(sample_progress, 0.0, 1.0) * wp.float32(active - 1)
    interval = wp.min(wp.int32(coordinate), active - 2)
    t = coordinate - wp.float32(interval)
    a = 1.0 - t
    first = ((routes[lane] * horizon + interval + padding) * width) + item
    last = first + width
    out_position[tid] = a * position[first] + t * position[last]
    qa = quaternion[first]
    qb = wp.vec4(quaternion[last])
    if wp.dot(qa, qb) < 0.0:
        qb = -qb
    rotation = wp.quat_slerp(
        wp.quat(qa[1], qa[2], qa[3], qa[0]),
        wp.quat(qb[1], qb[2], qb[3], qb[0]),
        t,
    )
    out_quaternion[tid] = wp.vec4(rotation[3], rotation[0], rotation[1], rotation[2])


def sample_pose_reference(
    position: torch.Tensor,
    quaternion: torch.Tensor,
    progress: torch.Tensor,
    routes: torch.Tensor,
    out_position: torch.Tensor,
    out_quaternion: torch.Tensor,
    padding: int,
) -> None:
    """Write dense linear/SLERP targets ``[lanes,samples,links,goalset,3/4]`` in place."""
    batch, samples, links, goals, _ = out_position.shape
    device, stream = get_warp_device_stream(out_position)
    wp.launch(
        kernel=_sample_pose_reference,
        dim=batch * samples * links * goals,
        inputs=[
            wp.from_torch(position.detach().view(-1, 3), dtype=wp.vec3),
            wp.from_torch(quaternion.detach().view(-1, 4), dtype=wp.vec4),
            wp.from_torch(progress.view(-1), dtype=wp.float32),
            wp.from_torch(routes.view(-1), dtype=wp.int32),
            wp.from_torch(out_position.view(-1, 3), dtype=wp.vec3),
            wp.from_torch(out_quaternion.view(-1, 4), dtype=wp.vec4),
            batch,
            samples,
            position.shape[1],
            links,
            goals,
            padding,
        ],
        device=device,
        stream=stream,
        adjoint=False,
    )
