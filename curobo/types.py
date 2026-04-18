# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common data types module.

This module provides common data types used throughout CuRobo for representing
robot states, poses, camera observations, and tensor device configurations.

Example:
    ```python
    from curobo.types import JointState, Pose, CameraObservation, DeviceCfg

    # Create joint state
    joint_state = JointState.from_position([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0])

    # Create pose
    pose = Pose(
        position=[0.5, 0.0, 0.5],
        quaternion=[1, 0, 0, 0]  # w, x, y, z
    )

    # Create camera observation
    camera_obs = CameraObservation(
        depth_image=depth_tensor,
        intrinsics=camera_intrinsics,
        pose=camera_pose,
    )

    # Specify tensor device and dtype
    device_cfg = DeviceCfg(device="cuda:0", dtype=torch.float32)
    ```
"""

# State types
# Camera types
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState
from curobo._src.types.camera import CameraObservation

# Content path
from curobo._src.types.content_path import ContentPath

# Tensor configuration
from curobo._src.types.device_cfg import DeviceCfg

# Pose types
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose, ToolPose

__all__ = [
    "JointState",
    "RobotState",
    "Pose",
    "ToolPose",
    "GoalToolPose",
    "CameraObservation",
    "ContentPath",
    "DeviceCfg",
]
