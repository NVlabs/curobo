# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tool pose types for FK output and goal specification.

ToolPose: 4D ``[batch, horizon, num_links, 3/4]``, FK output.
GoalToolPose: 5D ``[batch, horizon, num_links, num_goalset, 3/4]``, goal-side only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import torch

from curobo._src.types.pose import Pose
from curobo._src.util.logging import log_and_raise


@dataclass
class ToolPose(Sequence):
    """4D FK output: ``[batch, horizon, num_links, 3/4]``.

    No goalset dimension; this represents the current state of the robot,
    not a goal target.
    """

    #: Link/frame names, length must equal ``position.shape[2]``.
    tool_frames: List[str]

    #: Position tensor ``[batch, horizon, num_links, 3]``.
    position: torch.Tensor

    #: Quaternion tensor ``[batch, horizon, num_links, 4]`` (wxyz).
    quaternion: torch.Tensor

    def __post_init__(self):
        if self.position.ndim != 4:
            log_and_raise(
                f"ToolPose position must be 4D [B,H,L,3], got {self.position.shape}"
            )
        if self.quaternion.ndim != 4:
            log_and_raise(
                f"ToolPose quaternion must be 4D [B,H,L,4], got {self.quaternion.shape}"
            )
        if self.position.shape[2] != len(self.tool_frames):
            log_and_raise(
                f"num_links dim ({self.position.shape[2]}) != "
                f"len(tool_frames) ({len(self.tool_frames)})"
            )

    @property
    def batch_size(self) -> int:
        return self.position.shape[0]

    @property
    def horizon(self) -> int:
        return self.position.shape[1]

    @property
    def num_links(self) -> int:
        return self.position.shape[2]

    @property
    def shape(self):
        return self.position.shape

    @property
    def ndim(self):
        return self.position.ndim

    @property
    def device(self):
        return self.position.device

    def get_link_pose(self, link_name: str, make_contiguous: bool = False) -> Pose:
        """Extract a single link as a 2D Pose ``[B*H, 3/4]``."""
        if link_name not in self.tool_frames:
            log_and_raise(f"Link {link_name} not found in {self.tool_frames}")

        li = self.tool_frames.index(link_name)
        pos = self.position[:, :, li, :].reshape(-1, 3)
        quat = self.quaternion[:, :, li, :].reshape(-1, 4)

        if make_contiguous:
            pos = pos.contiguous()
            quat = quat.contiguous()
        return Pose(position=pos, quaternion=quat, name=link_name, normalize_rotation=False)

    def to_dict(self, make_contiguous: bool = True) -> Dict[str, Pose]:
        """Convert to dictionary mapping link names to 2D Poses."""
        return {
            name: self.get_link_pose(name, make_contiguous)
            for name in self.tool_frames
        }

    def copy_(self, other: ToolPose):
        self.tool_frames = other.tool_frames
        self.position.copy_(other.position)
        self.quaternion.copy_(other.quaternion)

    def requires_grad_(self, requires_grad: bool):
        self.position.requires_grad_(requires_grad)
        self.quaternion.requires_grad_(requires_grad)

    def clone(self) -> ToolPose:
        return ToolPose(
            tool_frames=self.tool_frames.copy(),
            position=self.position.clone(),
            quaternion=self.quaternion.clone(),
        )

    def detach(self) -> ToolPose:
        return ToolPose(
            tool_frames=self.tool_frames.copy(),
            position=self.position.detach(),
            quaternion=self.quaternion.detach(),
        )

    def contiguous(self) -> ToolPose:
        return ToolPose(
            tool_frames=self.tool_frames,
            position=self.position.contiguous(),
            quaternion=self.quaternion.contiguous(),
        )

    def __len__(self):
        return len(self.tool_frames)

    def __getitem__(self, idx: Union[int, str, torch.Tensor]) -> Union[Pose, ToolPose]:
        if isinstance(idx, str):
            return self.get_link_pose(idx)
        if isinstance(idx, int):
            return ToolPose(
                tool_frames=self.tool_frames,
                position=self.position[idx].unsqueeze(0),
                quaternion=self.quaternion[idx].unsqueeze(0),
            )
        return ToolPose(
            tool_frames=self.tool_frames,
            position=self.position[idx],
            quaternion=self.quaternion[idx],
        )

    def reorder_links(self, ordered_tool_frames: List[str]) -> ToolPose:
        """Reorder links. Returns a new ToolPose."""
        if not set(ordered_tool_frames).issubset(self.tool_frames):
            log_and_raise(
                f"Ordered link names {ordered_tool_frames} not a subset of {self.tool_frames}"
            )
        if self.tool_frames == ordered_tool_frames:
            return self

        indices = [self.tool_frames.index(name) for name in ordered_tool_frames]
        position = self.position[:, :, indices, :].contiguous()
        quaternion = self.quaternion[:, :, indices, :].contiguous()
        return ToolPose(
            tool_frames=ordered_tool_frames,
            position=position,
            quaternion=quaternion,
        )

    def as_goal(self, ordered_tool_frames: Optional[List[str]] = None) -> GoalToolPose:
        """Convert to GoalToolPose by adding a goalset dimension (num_goalset=1).

        Args:
            ordered_tool_frames: Optionally reorder/select a subset of links.

        Returns:
            GoalToolPose ``[B, H, L, 1, 3/4]``.
        """
        tp = self.reorder_links(ordered_tool_frames) if ordered_tool_frames else self
        return GoalToolPose(
            tool_frames=tp.tool_frames,
            position=tp.position.unsqueeze(3),
            quaternion=tp.quaternion.unsqueeze(3),
        )


@dataclass
class GoalToolPose(Sequence):
    """5D goal specification: ``[batch, horizon, num_links, num_goalset, 3/4]``.

    Used only on the goal/target side. horizon=1 for static goals,
    horizon>1 for per-timestep targets.
    """

    #: Link/frame names, length must equal ``position.shape[2]``.
    tool_frames: List[str]

    #: Position tensor ``[batch, horizon, num_links, num_goalset, 3]``.
    position: torch.Tensor

    #: Quaternion tensor ``[batch, horizon, num_links, num_goalset, 4]`` (wxyz).
    quaternion: torch.Tensor

    def __post_init__(self):
        if self.position.ndim != 5:
            log_and_raise(
                f"GoalToolPose position must be 5D [B,H,L,G,3], got {self.position.shape}"
            )
        if self.quaternion.ndim != 5:
            log_and_raise(
                f"GoalToolPose quaternion must be 5D [B,H,L,G,4], got {self.quaternion.shape}"
            )
        if self.position.shape[2] != len(self.tool_frames):
            log_and_raise(
                f"num_links dim ({self.position.shape[2]}) != "
                f"len(tool_frames) ({len(self.tool_frames)})"
            )

    @property
    def batch_size(self) -> int:
        return self.position.shape[0]

    @property
    def horizon(self) -> int:
        return self.position.shape[1]

    @property
    def num_links(self) -> int:
        return self.position.shape[2]

    @property
    def num_goalset(self) -> int:
        return self.position.shape[3]

    @property
    def shape(self):
        return self.position.shape

    @property
    def ndim(self):
        return self.position.ndim

    @property
    def device(self):
        return self.position.device

    @classmethod
    def from_poses(
        cls,
        pose_dict: Dict[str, Pose],
        ordered_tool_frames: Optional[List[str]] = None,
        num_goalset: int = 1,
    ) -> GoalToolPose:
        """Build from per-link Pose objects.

        Each Pose has position ``[batch, 3]`` or ``[batch * num_goalset, 3]``.

        Returns:
            GoalToolPose ``[batch, 1, num_links, num_goalset, 3/4]`` (horizon=1).
        """
        if not pose_dict:
            log_and_raise("pose_dict cannot be empty")

        frames = list(ordered_tool_frames) if ordered_tool_frames else list(pose_dict.keys())
        missing = set(frames) - set(pose_dict.keys())
        if missing:
            log_and_raise(f"Missing poses for links: {missing}")

        first = pose_dict[frames[0]]
        total_batch = first.position.shape[0]
        batch = total_batch // num_goalset

        positions = torch.stack(
            [pose_dict[l].position.view(batch, num_goalset, 3) for l in frames], dim=1
        )
        quaternions = torch.stack(
            [pose_dict[l].quaternion.view(batch, num_goalset, 4) for l in frames], dim=1
        )
        return cls(
            tool_frames=frames,
            position=positions.unsqueeze(1),
            quaternion=quaternions.unsqueeze(1),
        )

    def get_link_pose(self, link_name: str, make_contiguous: bool = False) -> Pose:
        """Extract a single link as a 2D Pose ``[B*H*G, 3/4]``."""
        if link_name not in self.tool_frames:
            log_and_raise(f"Link {link_name} not found in {self.tool_frames}")

        li = self.tool_frames.index(link_name)
        pos = self.position[:, :, li, :, :].reshape(-1, 3)
        quat = self.quaternion[:, :, li, :, :].reshape(-1, 4)

        if make_contiguous:
            pos = pos.contiguous()
            quat = quat.contiguous()
        return Pose(position=pos, quaternion=quat, name=link_name, normalize_rotation=False)

    def to_dict(self, make_contiguous: bool = True) -> Dict[str, Pose]:
        return {
            name: self.get_link_pose(name, make_contiguous)
            for name in self.tool_frames
        }

    def copy_(self, other: GoalToolPose):
        self.tool_frames = other.tool_frames
        self.position.copy_(other.position)
        self.quaternion.copy_(other.quaternion)

    def requires_grad_(self, requires_grad: bool):
        self.position.requires_grad_(requires_grad)
        self.quaternion.requires_grad_(requires_grad)

    def clone(self) -> GoalToolPose:
        return GoalToolPose(
            tool_frames=self.tool_frames.copy(),
            position=self.position.clone(),
            quaternion=self.quaternion.clone(),
        )

    def detach(self) -> GoalToolPose:
        return GoalToolPose(
            tool_frames=self.tool_frames.copy(),
            position=self.position.detach(),
            quaternion=self.quaternion.detach(),
        )

    def __len__(self):
        return len(self.tool_frames)

    def __getitem__(self, idx: Union[int, str, torch.Tensor]) -> Union[Pose, GoalToolPose]:
        if isinstance(idx, str):
            return self.get_link_pose(idx)
        if isinstance(idx, int):
            return GoalToolPose(
                tool_frames=self.tool_frames,
                position=self.position[idx].unsqueeze(0),
                quaternion=self.quaternion[idx].unsqueeze(0),
            )
        return GoalToolPose(
            tool_frames=self.tool_frames,
            position=self.position[idx],
            quaternion=self.quaternion[idx],
        )

    def reorder_links(self, ordered_tool_frames: List[str]) -> GoalToolPose:
        if not set(ordered_tool_frames).issubset(self.tool_frames):
            log_and_raise(
                f"Ordered link names {ordered_tool_frames} not a subset of {self.tool_frames}"
            )
        if self.tool_frames == ordered_tool_frames:
            return self

        indices = [self.tool_frames.index(name) for name in ordered_tool_frames]
        position = self.position[:, :, indices, :, :].contiguous()
        quaternion = self.quaternion[:, :, indices, :, :].contiguous()
        return GoalToolPose(
            tool_frames=ordered_tool_frames,
            position=position,
            quaternion=quaternion,
        )
