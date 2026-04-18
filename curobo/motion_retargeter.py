# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retargeter module for motion retargeting.

This module provides IK and MPC-based motion retargeting. Given per-frame
tool pose targets, the retargeter produces joint trajectories suitable for
humanoid robots.

Example:

.. code-block:: python

    from curobo.motion_retargeter import (
        MotionRetargeter, MotionRetargeterCfg, ToolPoseCriteria,
    )

    cfg = MotionRetargeterCfg.create(
        robot="unitree_g1_29dof_retarget.yml",
        tool_pose_criteria={
            "pelvis": ToolPoseCriteria.track_position_and_orientation(
                xyz=[1.0, 1.0, 1.0], rpy=[0.5, 0.5, 0.5],
            ),
            "left_hand": ToolPoseCriteria.track_position_and_orientation(
                xyz=[1.0, 1.0, 1.0], rpy=[0.3, 0.3, 0.3],
            ),
            "right_hand": ToolPoseCriteria.track_position_and_orientation(
                xyz=[1.0, 1.0, 1.0], rpy=[0.3, 0.3, 0.3],
            ),
        },
    )
    retargeter = MotionRetargeter(cfg)
    result = retargeter.solve_frame(goal_tool_pose)
"""

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.motion.motion_retargeter import MotionRetargeter
from curobo._src.motion.motion_retargeter_cfg import MotionRetargeterCfg
from curobo._src.motion.motion_retargeter_result import RetargetResult
from curobo._src.types.sequence_tool_pose import SequenceGoalToolPose
from curobo._src.types.tool_pose import GoalToolPose

__all__ = [
    "GoalToolPose",
    "MotionRetargeter",
    "MotionRetargeterCfg",
    "RetargetResult",
    "SequenceGoalToolPose",
    "ToolPoseCriteria",
]
