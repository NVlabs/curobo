# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Perception module for CuRobo.

This module provides perception-related functionality:
- Robot segmentation from depth images
- ESDF integration
- Pose estimation
- Depth filtering
"""

from curobo._src.perception.robot_segmenter import RobotSegmenter

__all__ = [
    "RobotSegmenter",
]
