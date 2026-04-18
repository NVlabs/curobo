# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Pose estimation for rigid objects and articulated robots from depth observations.

This module provides production-ready pose detection using point-to-plane ICP with Huber loss.
Experimental tracking implementations are available in the 'experimental' subfolder.
"""

from .detection_result import DetectionResult
from .geometry import ArticulatedRobotGeometry, RigidObjectGeometry
from .pose_detector import PoseDetector
from .pose_detector_cfg import DetectorCfg

__all__ = [
    # Geometry
    "RigidObjectGeometry",
    "ArticulatedRobotGeometry",
    # Object Detection
    "PoseDetector",
    "DetectorCfg",
    "DetectionResult",
]
