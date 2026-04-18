# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

# Third Party
import torch

# CuRobo
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise


@dataclass
class ToolPoseCriteria:
    """Criteria for a link pose.

    This class is used to define the nature of the cost between the current pose and the goal pose.
    This used as part of the goalset cost term.
    """

    #: Factor vector that scales each axis (x,y,z,roll,pitch,yaw) of the terminal position
    #: and orientation. This is multiplied with the weight.
    terminal_pose_axes_weight_factor: Optional[Union[torch.Tensor, List[float]]] = None

    #: Factor vector that scales each axis (x,y,z,roll,pitch,yaw) of the non-terminal position
    #: and orientation. This is multiplied with the weight.
    non_terminal_pose_axes_weight_factor: Optional[Union[torch.Tensor, List[float]]] = None

    #: Convergence tolerance for the terminal position and orientation. This should be of shape
    #: (2,). Position unit is meter and orientation unit is radian.
    terminal_pose_convergence_tolerance: Optional[Union[torch.Tensor, List[float]]] = None

    #: Convergence tolerance for the non-terminal position and orientation. This should be of shape
    #: (2,). Position unit is meter and orientation unit is radian.
    non_terminal_pose_convergence_tolerance: Optional[Union[torch.Tensor, List[float]]] = None

    #: If true, the distance is computed after projecting the current pose to the goal frame.
    project_distance_to_goal: Union[torch.Tensor, bool] = False

    device_cfg: DeviceCfg = DeviceCfg()

    def __post_init__(self):
        if self.terminal_pose_axes_weight_factor is None:
            self.terminal_pose_axes_weight_factor = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        elif len(self.terminal_pose_axes_weight_factor) != 6:
            log_and_raise(
                "terminal_pose_axes_weight_factor must be a list of 6 floats, "
                + f"got {self.terminal_pose_axes_weight_factor}"
            )

        if self.non_terminal_pose_axes_weight_factor is None:
            self.non_terminal_pose_axes_weight_factor = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        elif len(self.non_terminal_pose_axes_weight_factor) != 6:
            log_and_raise(
                "non_terminal_pose_axes_weight_factor must be a list of 6 floats, "
                + f"got {self.non_terminal_pose_axes_weight_factor}"
            )

        if self.terminal_pose_convergence_tolerance is None:
            self.terminal_pose_convergence_tolerance = [0.0, 0.0]
        elif len(self.terminal_pose_convergence_tolerance) != 2:
            log_and_raise(
                "terminal_pose_convergence_tolerance must be a list of 2 floats, "
                + f"got {self.terminal_pose_convergence_tolerance}"
            )

        if self.non_terminal_pose_convergence_tolerance is None:
            self.non_terminal_pose_convergence_tolerance = [0.0, 0.0]
        elif len(self.non_terminal_pose_convergence_tolerance) != 2:
            log_and_raise(
                "non_terminal_pose_convergence_tolerance must be a list of 2 floats, "
                + f"got {self.non_terminal_pose_convergence_tolerance}"
            )

        if not isinstance(self.project_distance_to_goal, torch.Tensor):
            if isinstance(self.project_distance_to_goal, bool):
                self.project_distance_to_goal = torch.tensor(
                    [self.project_distance_to_goal],
                    device=self.device_cfg.device,
                    dtype=torch.uint8,
                )
            else:
                log_and_raise(
                    "project_distance_to_goal must be a bool or a torch.Tensor, "
                    + f"got {self.project_distance_to_goal}"
                )

        # copy to device:
        self.terminal_pose_axes_weight_factor = self.device_cfg.to_device(self.terminal_pose_axes_weight_factor)
        self.non_terminal_pose_axes_weight_factor = self.device_cfg.to_device(
            self.non_terminal_pose_axes_weight_factor
        )
        self.terminal_pose_convergence_tolerance = self.device_cfg.to_device(
            self.terminal_pose_convergence_tolerance
        )
        self.non_terminal_pose_convergence_tolerance = self.device_cfg.to_device(
            self.non_terminal_pose_convergence_tolerance
        )

    def clone(self):
        return ToolPoseCriteria(
            terminal_pose_axes_weight_factor=self.terminal_pose_axes_weight_factor.clone(),
            non_terminal_pose_axes_weight_factor=self.non_terminal_pose_axes_weight_factor.clone(),
            terminal_pose_convergence_tolerance=self.terminal_pose_convergence_tolerance.clone(),
            non_terminal_pose_convergence_tolerance=self.non_terminal_pose_convergence_tolerance.clone(),
            project_distance_to_goal=self.project_distance_to_goal.clone(),
            device_cfg=self.device_cfg,
        )

    def copy_(self, other: ToolPoseCriteria):
        if self.device_cfg != other.device_cfg:
            log_and_raise(f"device_cfg mismatch: {self.device_cfg} != {other.device_cfg}")

        if other.terminal_pose_axes_weight_factor is not None:
            self.terminal_pose_axes_weight_factor.copy_(other.terminal_pose_axes_weight_factor)
        if other.non_terminal_pose_axes_weight_factor is not None:
            self.non_terminal_pose_axes_weight_factor.copy_(other.non_terminal_pose_axes_weight_factor)
        if other.terminal_pose_convergence_tolerance is not None:
            self.terminal_pose_convergence_tolerance.copy_(
                other.terminal_pose_convergence_tolerance
            )
        if other.non_terminal_pose_convergence_tolerance is not None:
            self.non_terminal_pose_convergence_tolerance.copy_(
                other.non_terminal_pose_convergence_tolerance
            )
        if other.project_distance_to_goal is not None:
            self.project_distance_to_goal[:] = other.project_distance_to_goal

    @staticmethod
    def track_position(xyz: List[float] = [1.0, 1.0, 1.0]):
        return ToolPoseCriteria(
            terminal_pose_axes_weight_factor=[xyz[0], xyz[1], xyz[2], 0.0, 0.0, 0.0],
            non_terminal_pose_axes_weight_factor=[xyz[0], xyz[1], xyz[2], 0.0, 0.0, 0.0],
        )

    @staticmethod
    def track_orientation(
        rpy: List[float] = [0.001, 0.001, 0.001], non_terminal_scale: float = 1.0
    ):
        return ToolPoseCriteria(
            terminal_pose_axes_weight_factor=[0.0, 0.0, 0.0, rpy[0], rpy[1], rpy[2]],
            non_terminal_pose_axes_weight_factor=[
                0.0,
                0.0,
                0.0,
                non_terminal_scale * rpy[0],
                non_terminal_scale * rpy[1],
                non_terminal_scale * rpy[2],
            ],
        )

    @staticmethod
    def track_position_and_orientation(
        xyz: List[float] = [1.0, 1.0, 1.0],
        rpy: List[float] = [1.0, 1.0, 1.0],
        non_terminal_scale: float = 0.1,
    ):
        return ToolPoseCriteria(
            terminal_pose_axes_weight_factor=[xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2]],
            non_terminal_pose_axes_weight_factor=[
                non_terminal_scale * xyz[0],
                non_terminal_scale * xyz[1],
                non_terminal_scale * xyz[2],
                non_terminal_scale * rpy[0],
                non_terminal_scale * rpy[1],
                non_terminal_scale * rpy[2],
            ],
        )
    @staticmethod
    def linear_motion(
        axis: str = "z",
        non_terminal_scale: float = 1.0,
        project_distance_to_goal: bool = True,
    ):
        axis_vector = [0.0, 0.0, 0.0]
        if axis == "x":
            axis_vector[0] = 1.0
        elif axis == "y":
            axis_vector[1] = 1.0
        elif axis == "z":
            axis_vector[2] = 1.0
        else:
            log_and_raise(f"Invalid axis: {axis}, must be 'x', 'y', or 'z'")
        return ToolPoseCriteria(
            terminal_pose_axes_weight_factor=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            non_terminal_pose_axes_weight_factor=[
                non_terminal_scale * (1 - axis_vector[0]),
                non_terminal_scale * (1 - axis_vector[1]),
                non_terminal_scale * (1 - axis_vector[2]),
                non_terminal_scale * 1.0    ,
                non_terminal_scale * 1.0,
                non_terminal_scale * 1.0,
            ],
            project_distance_to_goal=project_distance_to_goal,
        )

    @staticmethod
    def disabled():
        """Create criteria that disables pose tracking for this tool frame.

        Use this when you want to include a tool frame in the solver but not
        apply any pose cost to it.

        Returns:
            ToolPoseCriteria with all weight factors set to zero.
        """
        return ToolPoseCriteria(
            terminal_pose_axes_weight_factor=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            non_terminal_pose_axes_weight_factor=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )


@dataclass
class StackedToolPoseCriteria:
    """Stacked link pose criteria.

    This class is used to stack multiple link pose criteria.
    """

    tool_frames: List[str]

    #: Weight factor vector that scales each axis (x,y,z,roll,pitch,yaw) of the terminal position
    #: and orientation. Shape is (num_links, 6). This is multiplied with the weight.
    terminal_pose_axes_weight_factor: torch.Tensor

    #: Weight factor vector that scales each axis (x,y,z,roll,pitch,yaw) of the non-terminal position
    #: and orientation. Shape is (num_links, 6). This is multiplied with the weight.
    non_terminal_pose_axes_weight_factor: torch.Tensor

    #: Convergence tolerance for the terminal position and orientation. This should be of shape
    #: (num_links, 2). Position unit is meter and orientation unit is radian.
    terminal_pose_convergence_tolerance: torch.Tensor

    #: Convergence tolerance for the non-terminal position and orientation. This should be of shape
    #: (num_links, 2). Position unit is meter and orientation unit is radian.
    non_terminal_pose_convergence_tolerance: torch.Tensor

    #: If true, the distance is computed after projecting the current pose to the goal frame.
    #: Shape is (num_links,1).
    project_distance_to_goal: torch.Tensor

    device_cfg: DeviceCfg = DeviceCfg()

    _tool_pose_criteria: Optional[Dict[str, ToolPoseCriteria]] = None

    @staticmethod
    def from_tool_pose_criteria(tool_pose_criteria: Dict[str, ToolPoseCriteria]):
        tool_frames = list(tool_pose_criteria.keys())
        terminal_pose_axes_weight_factor = torch.stack(
            [tool_pose_criteria[link_name].terminal_pose_axes_weight_factor for link_name in tool_frames]
        )
        non_terminal_pose_axes_weight_factor = torch.stack(
            [
                tool_pose_criteria[link_name].non_terminal_pose_axes_weight_factor
                for link_name in tool_frames
            ]
        )
        terminal_pose_convergence_tolerance = torch.stack(
            [
                tool_pose_criteria[link_name].terminal_pose_convergence_tolerance
                for link_name in tool_frames
            ]
        )
        non_terminal_pose_convergence_tolerance = torch.stack(
            [
                tool_pose_criteria[link_name].non_terminal_pose_convergence_tolerance
                for link_name in tool_frames
            ]
        )
        project_distance_to_goal = torch.stack(
            [tool_pose_criteria[link_name].project_distance_to_goal for link_name in tool_frames]
        )
        return StackedToolPoseCriteria(
            tool_frames=tool_frames,
            terminal_pose_axes_weight_factor=terminal_pose_axes_weight_factor,
            non_terminal_pose_axes_weight_factor=non_terminal_pose_axes_weight_factor,
            terminal_pose_convergence_tolerance=terminal_pose_convergence_tolerance,
            non_terminal_pose_convergence_tolerance=non_terminal_pose_convergence_tolerance,
            project_distance_to_goal=project_distance_to_goal,
            device_cfg=tool_pose_criteria[list(tool_pose_criteria.keys())[0]].device_cfg,
            _tool_pose_criteria=tool_pose_criteria,
        )

    def __post_init__(self):
        num_links = len(self.tool_frames)
        # check shapes of all tensors:
        if self.terminal_pose_axes_weight_factor.shape != (num_links, 6):
            log_and_raise(
                f"terminal_pose_axes_weight_factor must be of shape (num_links, 6), got {self.terminal_pose_axes_weight_factor.shape}"
            )
        if self.non_terminal_pose_axes_weight_factor.shape != (num_links, 6):
            log_and_raise(
                f"non_terminal_pose_axes_weight_factor must be of shape (num_links, 6), got {self.non_terminal_pose_axes_weight_factor.shape}"
            )
        if self.terminal_pose_convergence_tolerance.shape != (num_links, 2):
            log_and_raise(
                f"terminal_pose_convergence_tolerance must be of shape (num_links, 2), got {self.terminal_pose_convergence_tolerance.shape}"
            )
        if self.non_terminal_pose_convergence_tolerance.shape != (num_links, 2):
            log_and_raise(
                f"non_terminal_pose_convergence_tolerance must be of shape (num_links, 2), got {self.non_terminal_pose_convergence_tolerance.shape}"
            )
        if self.project_distance_to_goal.shape != (num_links, 1):
            log_and_raise(
                f"project_distance_to_goal must be of shape (num_links,1), got {self.project_distance_to_goal.shape}"
            )

    def clone(self) -> StackedToolPoseCriteria:
        return StackedToolPoseCriteria(
            tool_frames=self.tool_frames,
            terminal_pose_axes_weight_factor=self.terminal_pose_axes_weight_factor.clone(),
            non_terminal_pose_axes_weight_factor=self.non_terminal_pose_axes_weight_factor.clone(),
            terminal_pose_convergence_tolerance=self.terminal_pose_convergence_tolerance.clone(),
            non_terminal_pose_convergence_tolerance=self.non_terminal_pose_convergence_tolerance.clone(),
            project_distance_to_goal=self.project_distance_to_goal.clone(),
            device_cfg=self.device_cfg,
            _tool_pose_criteria=self._tool_pose_criteria,
        )

    def update_tool_pose_criteria(self, tool_pose_criteria: Dict[str, ToolPoseCriteria]):
        for link_name in tool_pose_criteria.keys():
            if link_name not in self.tool_frames:
                log_and_raise(f"link_name {link_name} not found in tool_frames")
            self._tool_pose_criteria[link_name].copy_(tool_pose_criteria[link_name])
            self._update_criteria_in_stack(link_name, tool_pose_criteria[link_name])

    def _update_criteria_in_stack(self, link_name: str, tool_pose_criteria: ToolPoseCriteria):
        link_idx = self.tool_frames.index(link_name)
        if tool_pose_criteria.terminal_pose_axes_weight_factor is not None:
            self.terminal_pose_axes_weight_factor[link_idx, :] = (
                tool_pose_criteria.terminal_pose_axes_weight_factor
            )
        if tool_pose_criteria.non_terminal_pose_axes_weight_factor is not None:
            self.non_terminal_pose_axes_weight_factor[link_idx, :] = (
                tool_pose_criteria.non_terminal_pose_axes_weight_factor
            )
        if tool_pose_criteria.terminal_pose_convergence_tolerance is not None:
            self.terminal_pose_convergence_tolerance[link_idx, :] = (
                tool_pose_criteria.terminal_pose_convergence_tolerance
            )
        if tool_pose_criteria.non_terminal_pose_convergence_tolerance is not None:
            self.non_terminal_pose_convergence_tolerance[link_idx, :] = (
                tool_pose_criteria.non_terminal_pose_convergence_tolerance
            )
        if tool_pose_criteria.project_distance_to_goal is not None:
            self.project_distance_to_goal[link_idx, :] = tool_pose_criteria.project_distance_to_goal
