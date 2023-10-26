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
from typing import List

# Third Party
import numpy as np
from robometrics.statistics import Statistic, TrajectoryGroupMetrics, TrajectoryMetrics


@dataclass
class CuroboMetrics(TrajectoryMetrics):
    time: float = np.inf


@dataclass
class CuroboGroupMetrics(TrajectoryGroupMetrics):
    time: float = np.inf

    @classmethod
    def from_list(cls, group: List[CuroboMetrics]):
        unskipped = [m for m in group if not m.skip]
        successes = [m for m in unskipped if m.success]
        data = super().from_list(group)
        data.time = Statistic.from_list([m.time for m in successes])
        return data
