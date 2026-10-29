# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Common data types module.

This module provides common data types used throughout CuRobo for representing
robot states, poses, camera observations, and tensor device configurations.

Example:
    ```python
    from curobo.types import JointState, Pose, CameraObservation, LidarObservation, DeviceCfg

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

# CuRobo
from curobo._src.cost.criteria_tool_relative_transform import ToolRelativeTransformCriteria
from curobo._src.cost.criteria_tool_velocity_limit import ToolVelocityLimitCriteria
from curobo._src.cost.tool_pose_criteria import PoseErrorFrame, ToolPoseCriteria
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState
from curobo._src.types.camera import CameraObservation
from curobo._src.types.content_path import ContentPath
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.lidar import LidarObservation
from curobo._src.types.pose import Pose
from curobo._src.types.target_tool_relative_transform import ToolRelativeTransformTarget
from curobo._src.types.tool_pose import GoalToolPose, ToolPose

__all__ = [
    "JointState",
    "RobotState",
    "Pose",
    "ToolPose",
    "GoalToolPose",
    "ToolRelativeTransformTarget",
    "ToolRelativeTransformCriteria",
    "ToolVelocityLimitCriteria",
    "ToolPoseCriteria",
    "PoseErrorFrame",
    "CameraObservation",
    "LidarObservation",
    "ContentPath",
    "DeviceCfg",
]
