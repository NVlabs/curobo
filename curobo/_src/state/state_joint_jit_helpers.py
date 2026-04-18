# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""JIT helper functions for JointState operations."""
from __future__ import annotations

# Standard Library
from typing import Tuple, Union

# Third Party
import torch

# CuRobo
from curobo._src.util.tensor_util import clone_if_not_none, tensor_repeat_seeds
from curobo._src.util.torch_util import get_torch_jit_decorator


@get_torch_jit_decorator(slow_to_compile=True)
def jit_js_scale(
    vel: Union[None, torch.Tensor],
    acc: Union[None, torch.Tensor],
    jerk: Union[None, torch.Tensor],
    dt: torch.Tensor,
    new_dt: torch.Tensor,
):
    scale_dt = dt / new_dt

    if vel is not None:
        if len(vel.shape) == 2:
            scale_dt = scale_dt.view(-1, 1)
        if len(vel.shape) == 3:
            scale_dt = scale_dt.view(-1, 1, 1)
        vel = vel * scale_dt
    if acc is not None:
        acc = acc * scale_dt * scale_dt
    if jerk is not None:
        jerk = jerk * scale_dt * scale_dt * scale_dt
    return vel, acc, jerk


@get_torch_jit_decorator(slow_to_compile=True)
def jit_get_index(
    position: torch.Tensor,
    velocity: Union[torch.Tensor, None],
    acc: Union[torch.Tensor, None],
    jerk: Union[torch.Tensor, None],
    dt: Union[torch.Tensor, None],
    idx: torch.Tensor,
):
    position = position[idx]
    if velocity is not None:
        velocity = velocity[idx]
    if acc is not None:
        acc = acc[idx]
    if jerk is not None:
        jerk = jerk[idx]
    if dt is not None and len(dt) > 1:
        idx = idx.squeeze()
        if len(list(idx.shape)) == 0:
            dt = dt[idx : idx + 1]
        elif len(list(idx.shape)) == 1 and idx.shape[0] == 1:
            idx = idx[0]
            dt = dt[idx : idx + 1]
        else:
            dt = dt[idx]

    return position, velocity, acc, jerk, dt


def fn_get_index(
    position: torch.Tensor,
    velocity: Union[torch.Tensor, None],
    acc: Union[torch.Tensor, None],
    jerk: Union[torch.Tensor, None],
    dt: Union[torch.Tensor, None],
    idx: torch.Tensor,
):
    position = position[idx]
    if velocity is not None:
        velocity = velocity[idx]
    if acc is not None:
        acc = acc[idx]
    if jerk is not None:
        jerk = jerk[idx]
    if dt is not None and len(dt) > 1:
        if isinstance(idx, slice):
            dt = dt[idx]
        else:
            dt = dt[idx : idx + 1]

    return position, velocity, acc, jerk, dt


@get_torch_jit_decorator(slow_to_compile=True)
def jit_get_index_int(
    position: torch.Tensor,
    velocity: Union[torch.Tensor, None],
    acc: Union[torch.Tensor, None],
    jerk: Union[torch.Tensor, None],
    dt: Union[torch.Tensor, None],
    idx: int,
):
    position = position[idx]
    if velocity is not None:
        velocity = velocity[idx]
    if acc is not None:
        acc = acc[idx]
    if jerk is not None:
        jerk = jerk[idx]
    if dt is not None and len(dt) > 1:
        dt = dt[idx : idx + 1]

    return position, velocity, acc, jerk, dt


@get_torch_jit_decorator(slow_to_compile=True)
def jit_inplace_reindex(
    position: torch.Tensor,
    velocity: Union[torch.Tensor, None],
    acceleration: Union[torch.Tensor, None],
    jerk: Union[torch.Tensor, None],
    knot: Union[torch.Tensor, None],
    new_index: torch.Tensor,
):
    position = torch.index_select(position, -1, new_index)

    if velocity is not None:
        velocity = torch.index_select(velocity, -1, new_index)
    if acceleration is not None:
        acceleration = torch.index_select(acceleration, -1, new_index)

    if jerk is not None:
        jerk = torch.index_select(jerk, -1, new_index)
    if knot is not None:
        knot = torch.index_select(knot, -1, new_index)

    return position, velocity, acceleration, jerk, knot


@get_torch_jit_decorator(slow_to_compile=True)
def jit_joint_state_repeat_seeds(
    position: Union[torch.Tensor, None],
    velocity: Union[torch.Tensor, None],
    acceleration: Union[torch.Tensor, None],
    jerk: Union[torch.Tensor, None],
    dt: Union[torch.Tensor, None],
    num_seeds: int,
):
    if position is not None:
        position = tensor_repeat_seeds(position, num_seeds)
    if velocity is not None:
        velocity = tensor_repeat_seeds(velocity, num_seeds)
    if acceleration is not None:
        acceleration = tensor_repeat_seeds(acceleration, num_seeds)
    if jerk is not None:
        jerk = tensor_repeat_seeds(jerk, num_seeds)
    if dt is not None:
        dt = tensor_repeat_seeds(dt, num_seeds)
    return position, velocity, acceleration, jerk, dt


@get_torch_jit_decorator(slow_to_compile=True)
def jit_joint_state_copy(
    position: torch.Tensor,
    velocity: Union[torch.Tensor, None],
    acceleration: Union[torch.Tensor, None],
    jerk: Union[torch.Tensor, None],
    dt: Union[torch.Tensor, None],
    in_position: torch.Tensor,
    in_velocity: Union[torch.Tensor, None],
    in_acceleration: Union[torch.Tensor, None],
    in_jerk: Union[torch.Tensor, None],
    in_dt: Union[torch.Tensor, None],
) -> Tuple[
    torch.Tensor,
    Union[torch.Tensor, None],
    Union[torch.Tensor, None],
    Union[torch.Tensor, None],
    Union[torch.Tensor, None],
]:
    position.copy_(in_position)
    if velocity is not None and in_velocity is not None:
        velocity.copy_(in_velocity)
    if acceleration is not None and in_acceleration is not None:
        acceleration.copy_(in_acceleration)
    if jerk is not None and in_jerk is not None:
        jerk.copy_(in_jerk)
    if dt is not None and in_dt is not None:
        dt.copy_(in_dt)
    return position, velocity, acceleration, jerk, dt


@get_torch_jit_decorator(slow_to_compile=True)
def clone_state_jit(
    position: Union[torch.Tensor, None],
    velocity: Union[torch.Tensor, None],
    acceleration: Union[torch.Tensor, None],
    jerk: Union[torch.Tensor, None],
    dt: Union[torch.Tensor, None],
) -> Tuple[
    Union[torch.Tensor, None],
    Union[torch.Tensor, None],
    Union[torch.Tensor, None],
    Union[torch.Tensor, None],
    Union[torch.Tensor, None],
]:
    position_cl = clone_if_not_none(position)
    velocity_cl = clone_if_not_none(velocity)
    acceleration_cl = clone_if_not_none(acceleration)
    jerk_cl = clone_if_not_none(jerk)
    dt_cl = clone_if_not_none(dt)
    return position_cl, velocity_cl, acceleration_cl, jerk_cl, dt_cl


#@get_torch_jit_decorator()
def trim_trajectory_jit(
    position,
    velocity,
    acceleration,
    jerk,
    start_idx: int,
    end_idx: int,
):
    pos = position[..., start_idx:end_idx, :].clone()
    vel = acc = jerk_new = None
    if velocity is not None:
        vel = velocity[..., start_idx:end_idx, :].clone()
    if acceleration is not None:
        acc = acceleration[..., start_idx:end_idx, :].clone()
    if jerk is not None:
        jerk_new = jerk[..., start_idx:end_idx, :].clone()
    return pos, vel, acc, jerk_new

