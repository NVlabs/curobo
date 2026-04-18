# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Operations on JointState objects."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

import torch
import torch.autograd.profiler as profiler

from curobo._src.util.logging import log_and_raise
from curobo._src.util.tensor_util import fd_tensor

from .filter_coeff import FilterCoeff
from .state_joint_jit_helpers import (
    jit_inplace_reindex,
    jit_joint_state_repeat_seeds,
    jit_js_scale,
)

if TYPE_CHECKING:
    from .state_joint import JointState


def blend_joint_states(
    target: "JointState", new_state: "JointState", coeff: FilterCoeff
) -> "JointState":
    """Blend two joint states using filter coefficients.

    Args:
        target: Target joint state to blend into (modified in-place).
        new_state: New state to blend from.
        coeff: Filter coefficients for blending.

    Returns:
        The blended target state.
    """
    target.position[:] = (
        coeff.position * new_state.position + (1.0 - coeff.position) * target.position
    )
    target.velocity[:] = (
        coeff.velocity * new_state.velocity + (1.0 - coeff.velocity) * target.velocity
    )
    target.acceleration[:] = (
        coeff.acceleration * new_state.acceleration
        + (1.0 - coeff.acceleration) * target.acceleration
    )
    target.jerk[:] = coeff.jerk * new_state.jerk + (1.0 - coeff.jerk) * target.jerk
    return target


def joint_state_to_tensor(joint_state: "JointState") -> torch.Tensor:
    """Pack joint state into a single tensor.

    Args:
        joint_state: Joint state to pack.

    Returns:
        Tensor of shape [..., 4*dof] containing [position, velocity, acceleration, jerk].
    """
    velocity = joint_state.velocity
    acceleration = joint_state.acceleration
    jerk = joint_state.jerk
    if velocity is None:
        velocity = joint_state.position * 0.0
    if acceleration is None:
        acceleration = joint_state.position * 0.0
    if jerk is None:
        jerk = joint_state.position * 0.0
    state_tensor = torch.cat((joint_state.position, velocity, acceleration, jerk), dim=-1)
    return state_tensor


def stack_joint_states(js1: "JointState", js2: "JointState") -> "JointState":
    """Stack two joint states along the second-to-last dimension.

    Args:
        js1: First joint state.
        js2: Second joint state.

    Returns:
        Stacked joint state.
    """
    from .state_joint import JointState

    return JointState.from_state_tensor(
        torch.cat((joint_state_to_tensor(js1), joint_state_to_tensor(js2)), dim=-2),
        joint_names=js1.joint_names,
        dof=js1.position.shape[-1],
    )


@profiler.record_function("state_joint_ops/cat_joint_states")
def cat_joint_states(js1: "JointState", js2: "JointState", dim: int) -> "JointState":
    """Concatenate two joint states along a dimension.

    Args:
        js1: First joint state.
        js2: Second joint state.
        dim: Dimension to concatenate along.

    Returns:
        Concatenated joint state.
    """
    from curobo._src.util.tensor_util import clone_if_not_none

    from .state_joint import JointState

    position = velocity = acceleration = jerk = None
    new_joint_names = None
    if js1.position is not None and js2.position is not None:
        position = torch.cat((js1.position, js2.position), dim=dim)
    if js1.velocity is not None and js2.velocity is not None:
        velocity = torch.cat((js1.velocity, js2.velocity), dim=dim)

    if js1.acceleration is not None and js2.acceleration is not None:
        acceleration = torch.cat((js1.acceleration, js2.acceleration), dim=dim)

    if js1.jerk is not None and js2.jerk is not None:
        jerk = torch.cat((js1.jerk, js2.jerk), dim=dim)
    if dim == -1:
        new_joint_names = js1.joint_names + js2.joint_names
    else:
        new_joint_names = js1.joint_names
    return JointState(
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        jerk=jerk,
        joint_names=new_joint_names,
        dt=js1.dt,
        device_cfg=js1.device_cfg,
        knot=clone_if_not_none(js1.knot),
        knot_dt=clone_if_not_none(js1.knot_dt),
        control_space=js1.control_space,
    )


@profiler.record_function("state_joint_ops/repeat_joint_state")
def repeat_joint_state(joint_state: "JointState", repeat_input: List[int]) -> "JointState":
    """Repeat joint state along dimensions.

    Args:
        joint_state: Joint state to repeat.
        repeat_input: List of repeat counts for each dimension.

    Returns:
        Repeated joint state.
    """
    from curobo._src.util.tensor_util import clone_if_not_none

    from .state_joint import JointState

    repeat_position = joint_state.position.repeat(repeat_input)
    repeat_velocity = repeat_acceleration = repeat_jerk = None
    if joint_state.velocity is not None:
        repeat_velocity = joint_state.velocity.repeat(repeat_input)
    if joint_state.acceleration is not None:
        repeat_acceleration = joint_state.acceleration.repeat(repeat_input)
    if joint_state.jerk is not None:
        repeat_jerk = joint_state.jerk.repeat(repeat_input)
    return JointState(
        position=repeat_position,
        velocity=repeat_velocity,
        acceleration=repeat_acceleration,
        jerk=repeat_jerk,
        joint_names=joint_state.joint_names,
        dt=joint_state.dt,
        device_cfg=joint_state.device_cfg,
        knot=clone_if_not_none(joint_state.knot),
        knot_dt=clone_if_not_none(joint_state.knot_dt),
        control_space=joint_state.control_space,
    )


def repeat_joint_state_seeds(joint_state: "JointState", num_seeds: int) -> "JointState":
    """Repeat joint state for multiple seeds.

    Args:
        joint_state: Joint state to repeat.
        num_seeds: Number of seeds.

    Returns:
        Repeated joint state.
    """
    from .state_joint import JointState

    position, velocity, acceleration, jerk, dt = jit_joint_state_repeat_seeds(
        joint_state.position,
        joint_state.velocity,
        joint_state.acceleration,
        joint_state.jerk,
        joint_state.dt,
        num_seeds,
    )
    return JointState(
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        jerk=jerk,
        dt=dt,
        joint_names=joint_state.joint_names,
    )


def apply_kernel_to_joint_state(joint_state: "JointState", kernel_mat: torch.Tensor) -> "JointState":
    """Apply a kernel matrix to joint state.

    Args:
        joint_state: Joint state.
        kernel_mat: Kernel matrix to apply.

    Returns:
        Transformed joint state.
    """
    from .state_joint import JointState

    return JointState(
        position=kernel_mat @ joint_state.position,
        velocity=kernel_mat @ joint_state.velocity,
        acceleration=kernel_mat @ joint_state.acceleration,
        jerk=kernel_mat @ joint_state.jerk if joint_state.jerk is not None else None,
        dt=kernel_mat @ joint_state.dt if joint_state.dt is not None else None,
        joint_names=joint_state.joint_names,
    )


def scale_joint_state(joint_state: "JointState", dt: Union[float, torch.Tensor]) -> "JointState":
    """Scale joint state velocities by dt factor.

    Args:
        joint_state: Joint state.
        dt: Time scaling factor.

    Returns:
        Scaled joint state.
    """
    from .state_joint import JointState

    vel = acc = jerk = None
    knot_dt = None
    if joint_state.velocity is not None:
        vel = joint_state.velocity * dt
    if joint_state.acceleration is not None:
        acc = joint_state.acceleration * (dt**2)
    if joint_state.jerk is not None:
        jerk = joint_state.jerk * (dt**3)
    if joint_state.knot_dt is not None:
        knot_dt = joint_state.knot_dt
        log_and_raise("knot dt needs to be scaled")
    return JointState(
        joint_state.position,
        vel,
        acc,
        joint_state.joint_names,
        jerk,
        joint_state.device_cfg,
        knot=joint_state.knot,
        knot_dt=knot_dt,
    )


def scale_joint_state_by_dt(
    joint_state: "JointState", dt: torch.Tensor, new_dt: torch.Tensor
) -> "JointState":
    """Scale joint state from one dt to another.

    Args:
        joint_state: Joint state.
        dt: Current time step.
        new_dt: New time step.

    Returns:
        Scaled joint state.
    """
    from .state_joint import JointState

    vel, acc, jerk = jit_js_scale(joint_state.velocity, joint_state.acceleration, joint_state.jerk, dt, new_dt)
    knot_dt = None
    if joint_state.knot_dt is not None:
        knot_dt = joint_state.knot_dt * (new_dt / dt)
    return JointState(
        joint_state.position,
        vel,
        acc,
        joint_state.joint_names,
        jerk,
        joint_state.device_cfg,
        dt=new_dt,
        knot=joint_state.knot,
        knot_dt=knot_dt,
        control_space=joint_state.control_space,
    )


def scale_joint_state_time(joint_state: "JointState", new_dt: torch.Tensor) -> "JointState":
    """Scale joint state to a new time step using stored dt.

    Args:
        joint_state: Joint state with dt attribute.
        new_dt: New time step.

    Returns:
        Scaled joint state.
    """
    from .state_joint import JointState

    vel, acc, jerk = jit_js_scale(joint_state.velocity, joint_state.acceleration, joint_state.jerk, joint_state.dt, new_dt)

    return JointState(
        joint_state.position,
        vel,
        acc,
        joint_state.joint_names,
        jerk,
        joint_state.device_cfg,
        dt=new_dt,
        knot=joint_state.knot,
        knot_dt=joint_state.knot_dt,
        control_space=joint_state.control_space,
    )


def calculate_fd_from_position(
    joint_state: "JointState", dt: Optional[torch.Tensor] = None
) -> "JointState":
    """Calculate velocity, acceleration, jerk using finite differences.

    Args:
        joint_state: Joint state with position data.
        dt: Time step (uses joint_state.dt if not provided).

    Returns:
        Joint state with computed derivatives (modified in-place).
    """
    if joint_state.dt is None and dt is None:
        log_and_raise("dt is required")
    if dt is None:
        dt = joint_state.dt
    joint_state.velocity = fd_tensor(joint_state.position, dt)
    joint_state.acceleration = fd_tensor(joint_state.velocity, dt)
    joint_state.jerk = fd_tensor(joint_state.acceleration, dt)
    return joint_state


@profiler.record_function("state_joint_ops/reorder_joint_state")
def reorder_joint_state(joint_state: "JointState", ordered_joint_names: List[str]) -> "JointState":
    """Return joint state with reordered joint names.

    Args:
        joint_state: Joint state to reorder.
        ordered_joint_names: Desired order of joint names.

    Returns:
        New joint state with reordered joints.
    """
    new_js = joint_state.clone()
    reindex_joint_state_inplace(new_js, ordered_joint_names)
    return new_js


@profiler.record_function("state_joint_ops/reindex_joint_state_inplace")
def reindex_joint_state_inplace(joint_state: "JointState", joint_names: List[str]) -> None:
    """Reindex joint state in-place to match new joint order.

    Args:
        joint_state: Joint state to reindex (modified in-place).
        joint_names: New order of joint names.
    """
    if joint_state.joint_names is None:
        log_and_raise("joint names are not specified in JointState")
    # get index of joint names:
    new_index_l = [joint_state.joint_names.index(j) for j in joint_names]
    joint_state.joint_names = [joint_state.joint_names[x] for x in new_index_l]

    new_index = torch.as_tensor(new_index_l, device=joint_state.device_cfg.device, dtype=torch.long)

    joint_state.position, joint_state.velocity, joint_state.acceleration, joint_state.jerk, joint_state.knot = jit_inplace_reindex(
        joint_state.position,
        joint_state.velocity,
        joint_state.acceleration,
        joint_state.jerk,
        joint_state.knot,
        new_index,
    )


@profiler.record_function("state_joint_ops/augment_joint_state")
def augment_joint_state(
    joint_state: "JointState", joint_names: List[str], lock_joints: Optional["JointState"] = None
) -> "JointState":
    """Augment joint state with locked joints.

    Args:
        joint_state: Joint state to augment.
        joint_names: Full list of joint names.
        lock_joints: Joint state for locked joints.

    Returns:
        Augmented joint state.
    """
    if lock_joints is None:
        return reorder_joint_state(joint_state, joint_names)
    if joint_names is None or joint_state.joint_names is None:
        raise ValueError("joint_names can't be None")

    # if some joints are locked, we assume that these joints are not in joint_state.joint_names:
    if any(item in joint_state.joint_names for item in lock_joints.joint_names):
        raise ValueError("lock_joints is also listed in js.joint_names")

    # append the lock_joints to existing joint state:
    new_js = joint_state.clone()
    new_js = append_joints_to_state(new_js, lock_joints)
    new_js = reorder_joint_state(new_js, joint_names)
    return new_js


def append_joints_to_state(joint_state: "JointState", other_js: "JointState") -> "JointState":
    """Append joints from another state.

    Args:
        joint_state: Base joint state.
        other_js: Joint state to append.

    Returns:
        Combined joint state.
    """
    from .state_joint import JointState

    if other_js.joint_names is None or len(other_js.joint_names) == 0:
        log_and_raise("joint_names are required to append")

    current_shape = joint_state.position.shape
    extra_len = len(other_js.joint_names)
    current_js = joint_state
    one_dim = False
    if joint_state.knot is not None and other_js.knot is not None:
        if joint_state.knot.shape[:-1] == other_js.knot.shape and len(joint_state.knot.shape) == len(
            other_js.knot.shape
        ):
            current_js.knot = torch.cat((current_js.knot, other_js.position), dim=-1)
        log_and_raise("knot append needs to be implemented")
    # if joint state is of shape dof:
    if len(current_shape) == 1:
        current_js = joint_state.unsqueeze(0)
        one_dim = True
        current_shape = current_js.position.shape

    if current_shape[:-1] != other_js.position.shape:
        if len(other_js.position.shape) > 1 and other_js.position.shape[0] > 1:
            log_and_raise(
                "appending joints requires the new joints to have a shape matching current"
                + " batch size or have a batch size of 1."
            )

    if current_shape[:-1] == other_js.position.shape and len(current_shape) == len(
        other_js.position.shape
    ):
        log_and_raise("This is dead code that should never be hit")
        current_js.joint_names.extend(other_js.joint_names)
        current_js.position = torch.cat((current_js.position, other_js.position), dim=-1)
        new_js = current_js
        # This should also add values for velocity, acceleration and jerk.
        if current_js.velocity is not None:
            log_and_raise("Not implemented")
    elif len(current_shape) == 2:
        # repeat new joints
        if len(other_js.shape) == 1:
            other_js = other_js.unsqueeze(0)
        repeated_js = repeat_joint_state(other_js, [current_shape[0], 1])
        if current_js.velocity is not None:
            repeated_js.velocity = torch.zeros_like(repeated_js.position)
        if current_js.acceleration is not None:
            repeated_js.acceleration = torch.zeros_like(repeated_js.position)
        if current_js.jerk is not None:
            repeated_js.jerk = torch.zeros_like(repeated_js.position)

        new_js = cat_joint_states(current_js, repeated_js, dim=-1)
    elif len(current_shape) >= 3:
        # Handle 3D and higher dimensional cases (e.g., batch, seeds, horizon, dof)
        # Expand other_js to match the batch dimensions of current_js
        while len(other_js.shape) < len(current_shape):
            other_js = other_js.unsqueeze(0)

        # Build repeat shape: match all dimensions except last (dof), which stays at 1
        repeat_shape = list(current_shape[:-1]) + [1]
        repeated_js = repeat_joint_state(other_js, repeat_shape)

        if current_js.velocity is not None:
            repeated_js.velocity = torch.zeros_like(repeated_js.position)
        if current_js.acceleration is not None:
            repeated_js.acceleration = torch.zeros_like(repeated_js.position)
        if current_js.jerk is not None:
            repeated_js.jerk = torch.zeros_like(repeated_js.position)

        new_shape = list(current_shape)
        new_shape[-1] += len(other_js.joint_names)
        combined_joint_names = joint_state.joint_names + other_js.joint_names
        new_js = JointState.zeros(new_shape, current_js.device_cfg, joint_names=combined_joint_names)
        new_js.position[..., :-extra_len] = current_js.position
        if current_js.velocity is not None:
            new_js.velocity[..., :-extra_len] = current_js.velocity
        if current_js.acceleration is not None:
            new_js.acceleration[..., :-extra_len] = current_js.acceleration
        if current_js.jerk is not None:
            new_js.jerk[..., :-extra_len] = current_js.jerk

        new_js.position[..., -extra_len:] = other_js.position
        if other_js.velocity is not None:
            new_js.velocity[..., -extra_len:] = other_js.velocity
        if other_js.acceleration is not None:
            new_js.acceleration[..., -extra_len:] = other_js.acceleration
        if other_js.jerk is not None:
            new_js.jerk[..., -extra_len:] = other_js.jerk
        new_js.control_space = joint_state.control_space

    if one_dim:
        new_js = new_js.squeeze()
    if joint_state.dt is not None:
        new_js.dt = joint_state.dt
    if other_js.dt is not None:
        new_js.dt = other_js.dt
    return new_js

