# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Result dataclass for pose detection."""

from dataclasses import dataclass
from typing import Optional

from curobo._src.types.pose import Pose


@dataclass
class DetectionResult:
    """Result from pose detection."""

    pose: Pose  # Camera to object transform
    config: Optional[object]  # Joint configuration (for robots) or None (for rigid)
    confidence: float
    alignment_error: float
    n_iterations: int
    compute_time: float = 0.0
