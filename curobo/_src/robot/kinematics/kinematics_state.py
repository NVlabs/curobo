# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import torch

from curobo._src.robot.types.collision_geometry import RobotCollisionGeometry
from curobo._src.types.tool_pose import ToolPose


@dataclass
class KinematicsState:
    """Kinematic state of robot.

    All tensor fields preserve the ``[batch, horizon, ...]`` shape from FK.
    No flattening is done; shapes flow end-to-end from input to output.
    """

    #: Link poses ``[batch, horizon, num_links, 3/4]``.
    tool_poses: Optional[ToolPose] = None

    #: Jacobians ``[batch, horizon, num_links, 6, dof]``.
    tool_jacobians: Optional[torch.Tensor] = None

    #: Collision spheres ``[batch, horizon, num_spheres, 4]`` (x, y, z, r).
    robot_spheres: Optional[torch.Tensor] = None

    #: Center of mass ``[batch, horizon, 4]`` (xyz=global CoM, w=total mass).
    robot_com: Optional[torch.Tensor] = None

    #: Static collision geometry descriptor.
    robot_collision_geometry: Optional[RobotCollisionGeometry] = None

    @property
    def tool_frames(self) -> List[str]:
        if self.tool_poses is not None:
            return self.tool_poses.tool_frames
        return []

    def get_link_spheres(self) -> torch.Tensor:
        """Collision spheres ``[batch, horizon, num_spheres, 4]``."""
        return self.robot_spheres

    def clone(self):
        return KinematicsState(
            robot_spheres=self.robot_spheres.clone() if self.robot_spheres is not None else None,
            tool_jacobians=self.tool_jacobians.clone() if self.tool_jacobians is not None else None,
            tool_poses=self.tool_poses.clone() if self.tool_poses is not None else None,
            robot_com=self.robot_com.clone() if self.robot_com is not None else None,
            robot_collision_geometry=(
                self.robot_collision_geometry.clone()
                if self.robot_collision_geometry is not None
                else None
            ),
        )

    def detach(self):
        return KinematicsState(
            robot_spheres=self.robot_spheres.detach() if self.robot_spheres is not None else None,
            tool_jacobians=(
                self.tool_jacobians.detach() if self.tool_jacobians is not None else None
            ),
            tool_poses=self.tool_poses.detach() if self.tool_poses is not None else None,
            robot_com=self.robot_com.detach() if self.robot_com is not None else None,
            robot_collision_geometry=(
                self.robot_collision_geometry.detach()
                if self.robot_collision_geometry is not None
                else None
            ),
        )

    def copy_(self, other: KinematicsState):
        if self.robot_spheres is not None:
            self.robot_spheres.copy_(other.robot_spheres)
        if self.tool_jacobians is not None:
            self.tool_jacobians.copy_(other.tool_jacobians)
        if self.tool_poses is not None:
            self.tool_poses.copy_(other.tool_poses)
        if self.robot_com is not None:
            self.robot_com.copy_(other.robot_com)
        return self

    def __len__(self):
        """Batch size (shape[0])."""
        if self.robot_spheres is not None:
            return self.robot_spheres.shape[0]
        if self.tool_poses is not None:
            return self.tool_poses.batch_size
        if self.tool_jacobians is not None:
            return self.tool_jacobians.shape[0]
        return 0

    def __getitem__(self, idx: Union[int, torch.Tensor]):
        """Index into the batch dimension."""
        return KinematicsState(
            robot_spheres=self.robot_spheres[idx] if self.robot_spheres is not None else None,
            tool_jacobians=self.tool_jacobians[idx] if self.tool_jacobians is not None else None,
            tool_poses=self.tool_poses[idx] if self.tool_poses is not None else None,
            robot_com=self.robot_com[idx] if self.robot_com is not None else None,
            robot_collision_geometry=self.robot_collision_geometry,
        )
