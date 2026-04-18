# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Time-first sequence of goal tool poses for retargeting.

Stores goal poses as contiguous ``[num_frames, num_envs, num_links, num_goalset, 3/4]``
tensors (time-first) so that per-frame extraction ``position[t]`` yields a
contiguous slice. Each frame is returned as a :class:`GoalToolPose`
``[num_envs, 1, num_links, num_goalset, 3/4]`` (horizon=1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util.logging import log_and_raise


@dataclass
class SequenceGoalToolPose:
    """Time-series of goal tool poses for batch offline retargeting.

    Layout is ``(num_frames, num_envs, num_links, num_goalset, 3/4)``, time-first
    for per-frame contiguity. :meth:`get_frame` converts a single timestep
    to a :class:`GoalToolPose` with ``horizon=1``.
    """

    #: Link names matching dim 2 of position/quaternion.
    tool_frames: List[str]

    #: ``(num_frames, num_envs, num_links, num_goalset, 3)``.
    position: torch.Tensor

    #: ``(num_frames, num_envs, num_links, num_goalset, 4)`` wxyz convention.
    quaternion: torch.Tensor

    def __post_init__(self):
        if self.position.ndim != 5:
            log_and_raise(
                f"SequenceGoalToolPose position must be 5D "
                f"(num_frames, num_envs, num_links, num_goalset, 3), got shape {self.position.shape}"
            )
        if self.quaternion.ndim != 5:
            log_and_raise(
                f"SequenceGoalToolPose quaternion must be 5D "
                f"(num_frames, num_envs, num_links, num_goalset, 4), got shape {self.quaternion.shape}"
            )
        if self.position.shape[2] != len(self.tool_frames):
            log_and_raise(
                f"num_links dim ({self.position.shape[2]}) does not match "
                f"len(tool_frames) ({len(self.tool_frames)})"
            )

    @property
    def num_frames(self) -> int:
        return self.position.shape[0]

    @property
    def num_envs(self) -> int:
        return self.position.shape[1]

    @property
    def num_links(self) -> int:
        return self.position.shape[2]

    @property
    def num_goalset(self) -> int:
        return self.position.shape[3]

    def get_frame(self, t: int) -> GoalToolPose:
        """Return frame ``t`` as a GoalToolPose (view, no copy).

        Returns:
            GoalToolPose ``[num_envs, 1, num_links, num_goalset, 3/4]`` (horizon=1).
        """
        pos = self.position[t]
        quat = self.quaternion[t]
        return GoalToolPose(
            tool_frames=self.tool_frames,
            position=pos.unsqueeze(1),
            quaternion=quat.unsqueeze(1),
        )

    def clone(self) -> SequenceGoalToolPose:
        return SequenceGoalToolPose(
            tool_frames=self.tool_frames.copy(),
            position=self.position.clone(),
            quaternion=self.quaternion.clone(),
        )

    @property
    def device(self) -> torch.device:
        return self.position.device
