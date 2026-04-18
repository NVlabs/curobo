# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model predictive control module.

This module provides model predictive control for real-time trajectory tracking
with warm-start optimization and obstacle avoidance.

Example:
    ```python
    from curobo import ModelPredictiveControl, ModelPredictiveControlCfg
    from curobo.types import JointState, Pose

    # Setup MPC
    config = ModelPredictiveControlCfg.create(
        robot="franka.yml",
        scene_model="scene.yml",
        optimization_dt=0.02,
        interpolation_steps=4,
    )
    mpc = ModelPredictiveControl(config)

    # Setup problem with current state
    current_state = JointState.from_position([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])
    mpc.setup(current_state)

    # Update goal poses
    target_pose = Pose(position=[0.5, 0.0, 0.5], quaternion=[1, 0, 0, 0])
    mpc.update_goal_tool_poses({"ee_link": target_pose})

    # Control loop
    while not done:
        current_state = get_robot_state()
        result = mpc.optimize_action_sequence(current_state)
        for action in result.action_sequence:
            robot.execute(action)
    ```
"""

from curobo._src.solver.solver_mpc import MPCSolver as ModelPredictiveControl
from curobo._src.solver.solver_mpc_cfg import MPCSolverCfg as ModelPredictiveControlCfg
from curobo._src.solver.solver_mpc_result import (
    MPCSolverResult as ModelPredictiveControlResult,
)

__all__ = [
    "ModelPredictiveControl",
    "ModelPredictiveControlCfg",
    "ModelPredictiveControlResult",
]
