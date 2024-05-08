#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

# Standard Library
from dataclasses import dataclass
from typing import List, Optional

# Third Party
import numpy as np
from robometrics.statistics import (
    Statistic,
    TrajectoryGroupMetrics,
    TrajectoryMetrics,
    percent_true,
)


@dataclass
class CuroboMetrics(TrajectoryMetrics):
    time: float = np.inf
    cspace_path_length: float = 0.0
    perception_success: bool = False
    perception_interpolated_success: bool = False
    jerk: float = np.inf


@dataclass
class CuroboGroupMetrics(TrajectoryGroupMetrics):
    time: float = np.inf
    cspace_path_length: Optional[Statistic] = None
    perception_success: float = 0.0
    perception_interpolated_success: float = 0.0
    jerk: float = np.inf

    @classmethod
    def from_list(cls, group: List[CuroboMetrics]):
        unskipped = [m for m in group if not m.skip]
        successes = [m for m in unskipped if m.success]
        data = super().from_list(group)
        data.time = Statistic.from_list([m.time for m in successes])
        data.cspace_path_length = Statistic.from_list([m.cspace_path_length for m in successes])
        data.perception_success = percent_true([m.perception_success for m in group])
        data.perception_interpolated_success = percent_true(
            [m.perception_interpolated_success for m in group]
        )
        data.jerk = Statistic.from_list([m.jerk for m in successes])

        return data
