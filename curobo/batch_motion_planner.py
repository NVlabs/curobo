# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Batch motion planning module.

This module provides batch motion planning for solving multiple independent
planning problems in parallel with a single IK + trajectory optimization pass.

Example:
    ```python
    from curobo import BatchMotionPlanner, MotionPlannerCfg

    config = MotionPlannerCfg.create(
        robot="franka.yml",
        max_batch_size=16,
    )
    planner = BatchMotionPlanner(config)
    result = planner.plan_pose(goal_tool_poses, current_states)
    ```
"""

from curobo._src.motion.motion_planner_batch import BatchMotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg

__all__ = [
    "BatchMotionPlanner",
    "MotionPlannerCfg",
]
