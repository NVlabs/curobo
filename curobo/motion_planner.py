# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Motion planning module.

This module provides high-level motion planning combining trajectory optimization
and graph planning.

Example:
    .. code-block:: python

        from curobo import MotionPlanner, MotionPlannerCfg

        config = MotionPlannerCfg.create(
            robot="franka.yml",
            ik_cspace_rank_weight=5.0,
        )
        planner = MotionPlanner(config)
        result = planner.plan_pose(goal_tool_poses, current_state, outer_iterations=1)


    Follow up to 256 ordered Cartesian targets with a stationary start. Configure
    ``num_knots=64`` to retain 256 targets; the default 16 controls retain up to 72:

    .. code-block:: python

        # waypoints: GoalToolPose with shape [1, num_waypoints, num_links, 1, 3/4].
        result = planner.plan_pose_waypoints(
            waypoints, current_state,
            path_tolerance=(0.02, 0.174533),  # meters, radians
            final_tolerance=(0.005, 0.05),
            track_orientation=True,  # False tracks only positions, including the goal.
            outer_iterations=4,
        )


``outer_iterations`` is the maximum number of initial solve/check passes per trajectory
optimization. Normal pose planning defaults to one pass; waypoint planning defaults to four.
Each pass runs 50 optimizer steps by default and checks feasibility. Successful solves stop
continuation early; subsequent passes preserve the iterate and optimizer history.
The count applies separately to each direct or RRT-seeded solve, not to the whole planning call.
``max_attempts`` limits planning retries; ``finetune_attempts`` controls separate 25-step
refinement passes. The same pass-count contract applies to ``BatchMotionPlanner``.
"""

from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.motion.motion_planner_result import GraspPlanResult

__all__ = [
    "MotionPlanner",
    "MotionPlannerCfg",
    "GraspPlanResult",
]
