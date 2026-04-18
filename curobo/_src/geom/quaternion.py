# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

# Standard Library
from typing import Optional

# Third Party
import torch

# CuRobo
from curobo._src.util.torch_util import get_torch_jit_decorator


@get_torch_jit_decorator()
def normalize_quaternion(in_quaternion: torch.Tensor) -> torch.Tensor:
    """Normalize quaternion to unit length.

    Args:
        in_quaternion: Input quaternion. Shape: [..., 4]

    Returns:
        Normalized quaternion. Shape: [..., 4]
    """
    norm = torch.sqrt(torch.sum(in_quaternion * in_quaternion, dim=-1, keepdim=True))
    # Avoid division by zero
    norm = torch.where(norm > 1e-7, norm, torch.ones_like(norm))
    normalized = in_quaternion / norm

    # Ensure positive w component for consistency
    w_sign = torch.sign(normalized[..., 0:1])
    w_sign = torch.where(w_sign == 0, 1.0, w_sign)
    normalized = normalized * w_sign
    return normalized


@get_torch_jit_decorator()
def quat_multiply(
    q1: torch.Tensor, q2: torch.Tensor, q_res: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Multiply two quaternions.

    Args:
        q1: First quaternion. Shape: [..., 4]
        q2: Second quaternion. Shape: [..., 4]
        q_res: Optional output tensor. Shape: [..., 4]

    Returns:
        Resulting quaternion. Shape: [..., 4]
    """
    # Create output tensor if not provided
    if q_res is None:
        q_res = torch.zeros_like(q1)

    # Extract components
    a_w = q1[..., 0]
    a_x = q1[..., 1]
    a_y = q1[..., 2]
    a_z = q1[..., 3]
    b_w = q2[..., 0]
    b_x = q2[..., 1]
    b_y = q2[..., 2]
    b_z = q2[..., 3]

    # Compute multiplication
    q_res[..., 0] = a_w * b_w - a_x * b_x - a_y * b_y - a_z * b_z
    q_res[..., 1] = a_w * b_x + b_w * a_x + a_y * b_z - b_y * a_z
    q_res[..., 2] = a_w * b_y + b_w * a_y + a_z * b_x - b_z * a_x
    q_res[..., 3] = a_w * b_z + b_w * a_z + a_x * b_y - b_x * a_y
    return q_res


@get_torch_jit_decorator()
def angular_distance_phi3(goal_quat: torch.Tensor, current_quat: torch.Tensor) -> torch.Tensor:
    """This function computes the angular distance metric phi_3 given two quaternions.

    See Huynh, Du Q. "Metrics for 3D rotations: Comparison and analysis." Journal of Mathematical
    Imaging and Vision 35 (2009): 155-164.

    Args:
        goal_quat: Goal quaternion. Shape: [..., 4]
        current_quat: Current quaternion. Shape: [..., 4]

    Returns:
        Angular distance in range [0,1]. Shape: [...]
    """
    # Ensure inputs are normalized
    goal_quat = normalize_quaternion(goal_quat)
    current_quat = normalize_quaternion(current_quat)

    # Compute dot product
    dot_prod = torch.sum(goal_quat * current_quat, dim=-1)
    # Clamp to valid range and take absolute value
    dot_prod = torch.clamp(torch.abs(dot_prod), min=0.0, max=1.0)
    # Compute angle
    angle = torch.acos(dot_prod)
    # Normalize to [0,1] range
    distance = angle / (torch.pi * 0.5)
    return distance


# @get_torch_jit_decorator()
def angular_distance_axis_angle(
    goal_quat: torch.Tensor, current_quat: torch.Tensor
) -> torch.Tensor:
    """Compute the axis-angle error between two quaternions.

    This function computes the rotation angle of the relative rotation between
    two quaternions. The result represents the magnitude of rotation needed to
    align the current quaternion with the goal quaternion.

    Args:
        goal_quat: Goal quaternion. Shape: [..., 4]
        current_quat: Current quaternion. Shape: [..., 4]

    Returns:
        Rotation angle in radians. Shape: [...]
    """
    # Ensure inputs are normalized
    goal_quat = normalize_quaternion(goal_quat)
    current_quat = normalize_quaternion(current_quat)

    # Compute relative rotation: q_rel = goal_quat * current_quat^(-1)
    current_conj = current_quat.clone()
    current_conj[..., 1:] *= -1.0
    quat_rel = quat_multiply(goal_quat, current_conj)

    # Extract scalar part (w)
    w = quat_rel[..., 0]
    vec_part = quat_rel[..., 1:]

    vec_norm = torch.norm(vec_part, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vec_norm, torch.abs(w))

    # For a unit quaternion [w, x, y, z], the rotation angle is 2*acos(|w|)
    # Clamp w to avoid numerical issues
    # w_clamped = torch.clamp(torch.abs(w), min=0.0, max=1.0)
    # angle = 2.0 * torch.acos(w_clamped)

    return angle
