# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Result type for :class:`~curobo._src.motion.motion_retargeter.MotionRetargeter`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from curobo._src.state.state_joint import JointState


@dataclass
class RetargetResult:
    """Result returned by :meth:`MotionRetargeter.solve_frame` and
    :meth:`MotionRetargeter.solve_sequence`.

    For ``solve_frame``, shapes have no time dimension.
    For ``solve_sequence``, ``joint_state`` is stacked over frames.
    """

    #: Final joint state. For ``solve_frame``: position shape
    #: ``(num_envs, num_dof)``. For ``solve_sequence``: position shape
    #: ``(num_envs, num_output_frames, num_dof)``.
    joint_state: JointState

    #: Full intermediate MPC trajectory (MPC mode only, None for IK mode).
    #: For ``solve_frame``: position shape
    #: ``(num_envs, num_intermediate_frames, num_dof)``.
    #: For ``solve_sequence``: position shape
    #: ``(num_envs, total_intermediate_frames, num_dof)``.
    trajectory: Optional[JointState] = None
