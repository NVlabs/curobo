# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Trajectory operations on JointState objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

import torch

from curobo._src.util.logging import log_and_raise

from .state_joint_jit_helpers import trim_trajectory_jit

if TYPE_CHECKING:
    from .state_joint import JointState


def gather_joint_state_by_seed(joint_state: "JointState", idx: torch.Tensor) -> "JointState":
    """Gather joint states by seed index.

    Joint state is of shape (batch_size, num_seeds, horizon, dof).
    idx is a tensor of shape (batch_size, topk).
    Output is (batch_size, topk, horizon, dof).

    Args:
        joint_state: Joint state with shape (batch_size, num_seeds, horizon, dof).
        idx: Index tensor of shape (batch_size, topk).

    Returns:
        Gathered joint state.
    """
    from .state_joint import JointState

    dim = 1
    if idx.ndim != 2:
        log_and_raise("idx is not a 2D tensor")
    if idx.shape[0] != joint_state.position.shape[0]:
        log_and_raise("idx batch size does not match position batch size")
    if joint_state.position.ndim != 4:
        log_and_raise("position is not a 4D tensor (batch_size, num_seeds, horizon, dof)")

    # create a flat index tensor:
    batch_size = joint_state.shape[0]
    num_seeds = joint_state.shape[1]
    horizon = joint_state.shape[2]
    dof = joint_state.shape[3]
    topk_seeds = idx.shape[1]
    offset = torch.arange(batch_size, device=idx.device) * num_seeds
    offset = offset.view(-1, 1)
    flat_indices = (idx + offset).view(-1)

    # gather position:
    position = joint_state.position.view(batch_size * num_seeds, horizon, dof)[flat_indices].view(
        batch_size, topk_seeds, horizon, dof
    )
    # gather velocity:
    velocity = joint_state.velocity.view(batch_size * num_seeds, horizon, dof)[flat_indices].view(
        batch_size, topk_seeds, horizon, dof
    )
    # gather acceleration:
    acceleration = joint_state.acceleration.view(batch_size * num_seeds, horizon, dof)[flat_indices].view(
        batch_size, topk_seeds, horizon, dof
    )
    # gather jerk:
    jerk = joint_state.jerk.view(batch_size * num_seeds, horizon, dof)[flat_indices].view(
        batch_size, topk_seeds, horizon, dof
    )
    # gather dt:
    dt = joint_state.dt.view(batch_size * num_seeds, -1)[flat_indices].view(batch_size, topk_seeds)

    if joint_state.knot is not None:
        knots = joint_state.knot.shape[-2]
        knot = joint_state.knot.view(batch_size * num_seeds, knots, dof)[flat_indices].view(
            batch_size, topk_seeds, knots, dof
        )
    else:
        knot = None

    if joint_state.knot_dt is not None:
        knot_dt = joint_state.knot_dt.view(batch_size * num_seeds, -1)[flat_indices].view(
            batch_size, topk_seeds
        )
    else:
        knot_dt = None
    return JointState(
        position,
        velocity,
        acceleration,
        jerk=jerk,
        joint_names=joint_state.joint_names,
        dt=dt,
        knot=knot,
        knot_dt=knot_dt,
        control_space=joint_state.control_space,
    )


def copy_joint_state_only_index(
    target: "JointState", source: "JointState", idx: Union[int, torch.Tensor]
) -> "JointState":
    """Copy joint state data at specific index from source to target.

    Args:
        target: Target joint state (modified in-place).
        source: Source joint state.
        idx: Index to copy.

    Returns:
        Modified target state.
    """
    if target.position is not None:
        target.position[idx] = source.position[idx]
    if target.velocity is not None:
        target.velocity[idx] = source.velocity[idx]
    if target.acceleration is not None:
        target.acceleration[idx] = source.acceleration[idx]
    if target.jerk is not None:
        target.jerk[idx] = source.jerk[idx]
    if target.dt is not None:
        target.dt[idx] = source.dt[idx]
    return target


def copy_joint_state_at_index(
    target: "JointState", source: "JointState", idx: Union[int, List, torch.Tensor]
) -> None:
    """Copy source joint state to target at specific index.

    Args:
        target: Target joint state (modified in-place).
        source: Source joint state.
        idx: Index to copy to.
    """
    max_idx = 0
    if isinstance(idx, int):
        max_idx = idx
    elif isinstance(idx, List):
        max_idx = max(idx)
    elif isinstance(idx, torch.Tensor):
        max_idx = torch.max(idx)
    if target.position is not None:
        if max_idx >= target.position.shape[0]:
            raise ValueError(
                str(max_idx)
                + " index out of range, current state is of length "
                + str(target.position.shape[0])
            )
        target.position[idx] = source.position
    if target.velocity is not None:
        target.velocity[idx] = source.velocity
    if target.acceleration is not None:
        target.acceleration[idx] = source.acceleration
    if target.jerk is not None:
        target.jerk[idx] = source.jerk
    if target.dt is not None and source.dt is not None:
        target.dt[idx] = source.dt


def copy_joint_state_at_batch_seed_indices(
    target: "JointState",
    source: "JointState",
    batch_idx: torch.Tensor,
    seed_idx: torch.Tensor,
) -> "JointState":
    """Copy joint state at specific batch and seed indices.

    Args:
        target: Target joint state (modified in-place).
        source: Source joint state.
        batch_idx: Batch indices.
        seed_idx: Seed indices.

    Returns:
        Modified target state.
    """
    if target.position is not None:
        target.position[batch_idx, seed_idx] = source.position[batch_idx, seed_idx]
    if target.velocity is not None:
        target.velocity[batch_idx, seed_idx] = source.velocity[batch_idx, seed_idx]
    if target.acceleration is not None:
        target.acceleration[batch_idx, seed_idx] = source.acceleration[batch_idx, seed_idx]
    if target.jerk is not None:
        target.jerk[batch_idx, seed_idx] = source.jerk[batch_idx, seed_idx]
    if target.dt is not None and source.dt is not None:
        target.dt[batch_idx, seed_idx] = source.dt[batch_idx, seed_idx]
    if target.knot_dt is not None and source.knot_dt is not None:
        target.knot_dt[batch_idx, seed_idx] = source.knot_dt[batch_idx, seed_idx]
    if target.knot is not None and source.knot is not None:
        target.knot[batch_idx, seed_idx] = source.knot[batch_idx, seed_idx]
    return target


def get_joint_state_at_horizon_index(joint_state: "JointState", horizon_index: int) -> "JointState":
    """Get joint state at a specific horizon index.

    Args:
        joint_state: Joint state with horizon dimension.
        horizon_index: Index along the horizon dimension.

    Returns:
        Joint state at the specified horizon index.
    """
    from .state_joint import JointState

    if len(joint_state.position.shape) < 2:
        raise ValueError("JointState does not have horizon")
    return JointState(
        position=joint_state.position[..., horizon_index, :],
        velocity=joint_state.velocity[..., horizon_index, :] if joint_state.velocity is not None else None,
        acceleration=(
            joint_state.acceleration[..., horizon_index, :] if joint_state.acceleration is not None else None
        ),
        jerk=joint_state.jerk[..., horizon_index, :] if joint_state.jerk is not None else None,
        joint_names=joint_state.joint_names,
        device_cfg=joint_state.device_cfg,
        dt=joint_state.dt,
        control_space=joint_state.control_space,
        knot=joint_state.knot[..., horizon_index, :] if joint_state.knot is not None else None,
        knot_dt=joint_state.knot_dt,
    )


def trim_joint_state_trajectory(
    joint_state: "JointState", start_idx: int, end_idx: Optional[int] = None
) -> "JointState":
    """Trim joint state trajectory to specified range.

    Args:
        joint_state: Joint state with trajectory.
        start_idx: Start index.
        end_idx: End index (defaults to end of trajectory).

    Returns:
        Trimmed joint state.
    """
    from .state_joint import JointState

    if end_idx is None or end_idx == 0:
        end_idx = joint_state.position.shape[-2]
    if len(joint_state.position.shape) < 2:
        raise ValueError("JointState does not have horizon")
    pos, vel, acc, jerk = trim_trajectory_jit(
        joint_state.position, joint_state.velocity, joint_state.acceleration, joint_state.jerk, start_idx, end_idx
    )

    return JointState(
        pos,
        vel,
        acc,
        joint_names=joint_state.joint_names,
        jerk=jerk,
        device_cfg=joint_state.device_cfg,
        dt=joint_state.dt,
        control_space=joint_state.control_space,
    )


def index_joint_state_dof(joint_state: "JointState", idx: torch.Tensor) -> "JointState":
    """Index joint state along the DOF dimension.

    Args:
        joint_state: Joint state.
        idx: Index tensor for DOF selection.

    Returns:
        Indexed joint state.
    """
    from .state_joint import JointState

    velocity = acceleration = jerk = None
    new_index = idx
    knot = None
    position = torch.index_select(joint_state.position, -1, new_index)
    if joint_state.velocity is not None:
        velocity = torch.index_select(joint_state.velocity, -1, new_index)
    if joint_state.acceleration is not None:
        acceleration = torch.index_select(joint_state.acceleration, -1, new_index)
    if joint_state.jerk is not None:
        jerk = torch.index_select(joint_state.jerk, -1, new_index)
    if joint_state.knot is not None:
        knot = torch.index_select(joint_state.knot, -1, new_index)
    joint_names = [joint_state.joint_names[x] for x in new_index]
    return JointState(
        position=position,
        joint_names=joint_names,
        velocity=velocity,
        acceleration=acceleration,
        jerk=jerk,
        control_space=joint_state.control_space,
        knot=knot,
        knot_dt=joint_state.knot_dt,
    )

