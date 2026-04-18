# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import Optional

# Third Party
import torch

# CuRobo
from curobo._src.types.device_cfg import DeviceCfg


@dataclass
class PoseCostMetric:
    hold_partial_pose: bool = False
    release_partial_pose: bool = False
    hold_vec_weight: Optional[torch.Tensor] = None
    reach_partial_pose: bool = False
    reach_full_pose: bool = False
    reach_vec_weight: Optional[torch.Tensor] = None
    offset_position: Optional[torch.Tensor] = None
    offset_rotation: Optional[torch.Tensor] = None
    offset_tstep_fraction: float = -1.0
    remove_offset_waypoint: bool = False
    include_link_pose: bool = False
    project_to_goal_frame: Optional[bool] = None

    def clone(self):
        return PoseCostMetric(
            hold_partial_pose=self.hold_partial_pose,
            release_partial_pose=self.release_partial_pose,
            hold_vec_weight=None if self.hold_vec_weight is None else self.hold_vec_weight.clone(),
            reach_partial_pose=self.reach_partial_pose,
            reach_full_pose=self.reach_full_pose,
            reach_vec_weight=(
                None if self.reach_vec_weight is None else self.reach_vec_weight.clone()
            ),
            offset_position=None if self.offset_position is None else self.offset_position.clone(),
            offset_rotation=None if self.offset_rotation is None else self.offset_rotation.clone(),
            offset_tstep_fraction=self.offset_tstep_fraction,
            remove_offset_waypoint=self.remove_offset_waypoint,
            include_link_pose=self.include_link_pose,
            project_to_goal_frame=self.project_to_goal_frame,
        )

    @classmethod
    def create_grasp_approach_metric(
        cls,
        offset_position: float = 0.1,
        linear_axis: int = 2,
        tstep_fraction: float = 0.8,
        project_to_goal_frame: Optional[bool] = None,
        device_cfg: DeviceCfg = DeviceCfg(),
    ) -> PoseCostMetric:
        """Enables moving to a pregrasp and then locked orientation movement to final grasp.

        Since this is added as a cost, the trajectory will not reach the exact offset, instead it
        will try to take a blended path to the final grasp without stopping at the offset.

        Args:
            offset_position: offset in meters.
            linear_axis: specifies the x y or z axis.
            tstep_fraction:  specifies the timestep fraction to start activating this term.
            project_to_goal_frame: compute distance w.r.t. to goal frame instead of robot base
                frame. If None, it will use value set in PoseCostConfig.
            device_cfg: cuda device.

        Returns:
            cost metric.
        """
        hold_vec_weight = device_cfg.to_device([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        hold_vec_weight[3 + linear_axis] = 0.0
        offset_position_vec = device_cfg.to_device([0.0, 0.0, 0.0])
        offset_position_vec[linear_axis] = offset_position
        return cls(
            hold_partial_pose=True,
            hold_vec_weight=hold_vec_weight,
            offset_position=offset_position_vec,
            offset_tstep_fraction=tstep_fraction,
        )

    @classmethod
    def reset_metric(cls) -> PoseCostMetric:
        return PoseCostMetric(
            remove_offset_waypoint=True,
            reach_full_pose=True,
            release_partial_pose=True,
        )

    @classmethod
    def reach_position_metric(
        cls, device_cfg: DeviceCfg = DeviceCfg()
    ) -> PoseCostMetric:
        return PoseCostMetric(
            reach_partial_pose=True,
            reach_vec_weight=device_cfg.to_device([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        )
