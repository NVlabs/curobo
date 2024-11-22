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
from __future__ import annotations

# Standard Library
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

# Third Party
import torch

# CuRobo
from curobo.curobolib.geom import PoseError, PoseErrorDistance
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import OrientationError, Pose
from curobo.util.logger import log_error

# Local Folder
from .cost_base import CostBase, CostConfig


class PoseErrorType(Enum):
    SINGLE_GOAL = 0  #: Distance will be computed to a single goal pose
    BATCH_GOAL = 1  #: Distance will be computed pairwise between query batch and goal batch
    GOALSET = 2  #: Shortest Distance will be computed to a goal set
    BATCH_GOALSET = 3  #: Shortest Distance to a batch goal set


@dataclass
class PoseCostConfig(CostConfig):
    cost_type: PoseErrorType = PoseErrorType.BATCH_GOAL
    use_metric: bool = False
    project_distance: bool = True
    run_vec_weight: Optional[List[float]] = None
    use_projected_distance: bool = True
    offset_waypoint: List[float] = None
    offset_tstep_fraction: float = -1.0
    waypoint_horizon: int = 0

    def __post_init__(self):
        if self.run_vec_weight is not None:
            self.run_vec_weight = self.tensor_args.to_device(self.run_vec_weight)
        else:
            self.run_vec_weight = torch.ones(
                6, device=self.tensor_args.device, dtype=self.tensor_args.dtype
            )
        if self.vec_weight is None:
            self.vec_weight = torch.ones(
                6, device=self.tensor_args.device, dtype=self.tensor_args.dtype
            )
        if self.vec_convergence is None:
            self.vec_convergence = torch.zeros(
                2, device=self.tensor_args.device, dtype=self.tensor_args.dtype
            )
        if self.offset_waypoint is None:
            self.offset_waypoint = [0, 0, 0, 0, 0, 0]
        if self.run_weight is None:
            self.run_weight = 1
        self.offset_waypoint = self.tensor_args.to_device(self.offset_waypoint)
        if isinstance(self.offset_tstep_fraction, float):
            self.offset_tstep_fraction = self.tensor_args.to_device([self.offset_tstep_fraction])
        return super().__post_init__()


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
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ) -> PoseCostMetric:
        """Enables moving to a pregrasp and then locked orientation movement to final grasp.

        Since this is added as a cost, the trajectory will not reach the exact offset, instead it
        will try to take a blended path to the final grasp without stopping at the offset.

        Args:
            offset_position: offset in meters.
            linear_axis: specifies the x y or z axis.
            tstep_fraction:  specifies the timestep fraction to start activating this constraint.
            project_to_goal_frame: compute distance w.r.t. to goal frame instead of robot base
                frame. If None, it will use value set in PoseCostConfig.
            tensor_args: cuda device.

        Returns:
            cost metric.
        """
        hold_vec_weight = tensor_args.to_device([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        hold_vec_weight[3 + linear_axis] = 0.0
        offset_position_vec = tensor_args.to_device([0.0, 0.0, 0.0])
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


class PoseCost(CostBase, PoseCostConfig):
    def __init__(self, config: PoseCostConfig):
        PoseCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
        self.project_distance_tensor = torch.tensor(
            [self.project_distance],
            device=self.tensor_args.device,
            dtype=torch.uint8,
        )
        self.rot_weight = self.vec_weight[0:3]
        self.pos_weight = self.vec_weight[3:6]
        self._vec_convergence = self.tensor_args.to_device(self.vec_convergence)
        self._batch_size = 0

    def update_metric(self, metric: PoseCostMetric, update_offset_waypoint: bool = True):
        if metric.hold_partial_pose:
            if metric.hold_vec_weight is None:
                log_error("hold_vec_weight is required")
            self.hold_partial_pose(metric.hold_vec_weight)
        if metric.release_partial_pose:
            self.release_partial_pose()
        if metric.reach_partial_pose:
            if metric.reach_vec_weight is None:
                log_error("reach_vec_weight is required")
            self.reach_partial_pose(metric.reach_vec_weight)
        if metric.reach_full_pose:
            self.reach_full_pose()
        if metric.project_to_goal_frame is not None:
            self.project_distance_tensor[:] = metric.project_to_goal_frame
        else:
            self.project_distance_tensor[:] = self.project_distance
        if update_offset_waypoint:
            if metric.remove_offset_waypoint:
                self.remove_offset_waypoint()

            if metric.offset_position is not None or metric.offset_rotation is not None:
                self.update_offset_waypoint(
                    offset_position=metric.offset_position,
                    offset_rotation=metric.offset_rotation,
                    offset_tstep_fraction=metric.offset_tstep_fraction,
                )

    def hold_partial_pose(self, run_vec_weight: torch.Tensor):
        self.run_vec_weight.copy_(run_vec_weight)

    def release_partial_pose(self):
        self.run_vec_weight[:] = 0.0

    def reach_partial_pose(self, vec_weight: torch.Tensor):
        self.vec_weight[:] = vec_weight

    def reach_full_pose(self):
        self.vec_weight[:] = 1.0

    def update_offset_waypoint(
        self,
        offset_position: Optional[torch.Tensor] = None,
        offset_rotation: Optional[torch.Tensor] = None,
        offset_tstep_fraction: float = 0.75,
    ):
        if offset_position is not None:
            self.offset_waypoint[3:].copy_(offset_position)
        if offset_rotation is not None:
            self.offset_waypoint[:3].copy_(offset_rotation)
        self.offset_tstep_fraction[:] = offset_tstep_fraction
        if self.waypoint_horizon <= 0:
            log_error(
                "Updating offset waypoint requires PoseCostConfig.waypoint_horizon to be set."
            )
        self.update_run_weight(
            run_tstep_fraction=offset_tstep_fraction, horizon=self.waypoint_horizon
        )

    def remove_offset_waypoint(self):
        self.offset_tstep_fraction[:] = -1.0
        self.update_run_weight(horizon=self.waypoint_horizon)

    def update_run_weight(
        self,
        run_tstep_fraction: float = 0.0,
        run_weight: Optional[float] = None,
        horizon: Optional[int] = None,
    ):
        if horizon is None:
            horizon = self._horizon
        if horizon <= 1:
            return

        if run_weight is None:
            run_weight = self.run_weight

        active_steps = math.floor(horizon * run_tstep_fraction)
        self.initialize_run_weight_vec(horizon)
        self._run_weight_vec[:, :active_steps] = 0
        self._run_weight_vec[:, active_steps:-1] = run_weight

    def update_batch_size(self, batch_size, horizon):
        if batch_size != self._batch_size or horizon != self._horizon:
            self.out_distance = torch.zeros(
                (batch_size, horizon), device=self.tensor_args.device, dtype=self.tensor_args.dtype
            )
            self.out_position_distance = torch.zeros(
                (batch_size, horizon), device=self.tensor_args.device, dtype=self.tensor_args.dtype
            )
            self.out_rotation_distance = torch.zeros(
                (batch_size, horizon), device=self.tensor_args.device, dtype=self.tensor_args.dtype
            )
            self.out_idx = torch.zeros(
                (batch_size, horizon), device=self.tensor_args.device, dtype=torch.int32
            )
            self.out_p_vec = torch.zeros(
                (batch_size, horizon, 3),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
            self.out_q_vec = torch.zeros(
                (batch_size, horizon, 4),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
            self.out_p_grad = torch.zeros(
                (batch_size, horizon, 3),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
            self.out_q_grad = torch.zeros(
                (batch_size, horizon, 4),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
            self.initialize_run_weight_vec(horizon)
            if self.terminal and self.run_weight is not None and horizon > 1:
                self._run_weight_vec[:, :-1] = self.run_weight

            self._batch_size = batch_size
            self._horizon = horizon
            self.waypoint_horizon = horizon

    def initialize_run_weight_vec(self, horizon: Optional[int] = None):
        if horizon is None:
            horizon = self._horizon
        if self._run_weight_vec is None or self._run_weight_vec.shape[1] != horizon:
            self._run_weight_vec = torch.ones(
                (1, horizon), device=self.tensor_args.device, dtype=self.tensor_args.dtype
            )

    @property
    def goalset_index_buffer(self):
        return self.out_idx

    def _forward_goal_distribution(self, ee_pos_batch, ee_rot_batch, ee_goal_pos, ee_goal_rot):
        ee_goal_pos = ee_goal_pos.unsqueeze(1)
        ee_goal_pos = ee_goal_pos.unsqueeze(1)
        ee_goal_rot = ee_goal_rot.unsqueeze(1)
        ee_goal_rot = ee_goal_rot.unsqueeze(1)
        error, rot_error, pos_error = self.forward_single_goal(
            ee_pos_batch, ee_rot_batch, ee_goal_pos, ee_goal_rot
        )
        min_idx = torch.argmin(error[:, :, -1], dim=0)
        min_idx = min_idx.unsqueeze(1).expand(error.shape[1], error.shape[2])
        if len(min_idx.shape) == 2:
            min_idx = min_idx[0, 0]
        error = error[min_idx]
        rot_error = rot_error[min_idx]
        pos_error = pos_error[min_idx]
        return error, rot_error, pos_error, min_idx

    def _forward_single_goal(self, ee_pos_batch, ee_rot_batch, ee_goal_pos, ee_goal_rot):
        # b, h, _ = ee_pos_batch.shape
        d_g_ee = ee_pos_batch - ee_goal_pos

        position_err = torch.norm(self.pos_weight * d_g_ee, dim=-1)

        goal_dist = position_err  # .clone()
        rot_err = OrientationError.apply(ee_goal_rot, ee_rot_batch, ee_rot_batch.clone()).squeeze(
            -1
        )

        rot_err_c = rot_err.clone()
        goal_dist_c = goal_dist.clone()
        # clamp:
        if self.vec_convergence[1] > 0.0:
            position_err = torch.where(
                position_err > self.vec_convergence[1], position_err, position_err * 0.0
            )
        if self.vec_convergence[0] > 0.0:
            rot_err = torch.where(rot_err > self.vec_convergence[0], rot_err, rot_err * 0.0)

        # rot_err = torch.norm(goal_orient_vec, dim = -1)
        cost = self.weight[0] * rot_err + self.weight[1] * position_err

        # dimension should be bacth * traj_length
        return cost, rot_err_c, goal_dist_c

    def _forward_pytorch(self, ee_pos_batch, ee_rot_batch, ee_goal_pos, ee_goal_rot):
        if self.cost_type == PoseErrorType.SINGLE_GOAL:
            cost, r_err, g_dist = self.forward_single_goal(
                ee_pos_batch, ee_rot_batch, ee_goal_pos, ee_goal_rot
            )
        elif self.cost_type == PoseErrorType.BATCH_GOAL:
            cost, r_err, g_dist = self.forward_single_goal(
                ee_pos_batch, ee_rot_batch, ee_goal_pos, ee_goal_rot
            )
        else:
            cost, r_err, g_dist = self.forward_goal_distribution(
                ee_pos_batch, ee_rot_batch, ee_goal_pos, ee_goal_rot
            )
        if self.terminal and self.run_weight is not None:
            cost[:, :-1] *= self.run_weight
        return cost, r_err, g_dist

    def _update_cost_type(self, ee_goal_pos, ee_pos_batch, num_goals):
        d_g = len(ee_goal_pos.shape)
        b_sze = ee_goal_pos.shape[0]
        if d_g == 2 and b_sze == 1:  # [1, 3]
            self.cost_type = PoseErrorType.SINGLE_GOAL
        elif d_g == 2 and b_sze > 1:  # [b, 3]
            self.cost_type = PoseErrorType.BATCH_GOAL
        elif d_g == 3 and b_sze == 1:  # [1, goalset, 3]
            self.cost_type = PoseErrorType.GOALSET
        elif d_g == 3 and b_sze > 1:  # [b, goalset,3]
            self.cost_type = PoseErrorType.BATCH_GOALSET

    def forward_out_distance(
        self, ee_pos_batch, ee_rot_batch, goal: Goal, link_name: Optional[str] = None
    ):
        if link_name is None:
            goal_pose = goal.goal_pose
        else:
            goal_pose = goal.links_goal_pose[link_name]

        ee_goal_pos = goal_pose.position
        ee_goal_rot = goal_pose.quaternion
        num_goals = goal_pose.n_goalset
        self._update_cost_type(ee_goal_pos, ee_pos_batch, num_goals)

        b, h, _ = ee_pos_batch.shape

        self.update_batch_size(b, h)

        distance, g_dist, r_err, idx = PoseErrorDistance.apply(
            ee_pos_batch,  # .view(-1, 3).contiguous(),
            ee_goal_pos,
            ee_rot_batch,  # .view(-1, 4).contiguous(),
            ee_goal_rot,
            self.vec_weight,
            self.weight,
            self._vec_convergence,
            self._run_weight_vec,
            self.run_vec_weight,
            self.offset_waypoint,
            self.offset_tstep_fraction,
            goal.batch_pose_idx,
            self.project_distance_tensor,
            self.out_distance,
            self.out_position_distance,
            self.out_rotation_distance,
            self.out_p_vec,
            self.out_q_vec,
            self.out_idx,
            self.out_p_grad,
            self.out_q_grad,
            b,
            h,
            self.cost_type.value,
            num_goals,
            self.use_metric,
        )
        # print(self.out_idx.shape, self.out_idx[:,-1])
        # print(goal.batch_pose_idx.shape)
        cost = distance  # .view(b, h)#.clone()
        r_err = r_err  # .view(b, h)
        g_dist = g_dist  # .view(b, h)
        idx = idx  # .view(b, h)

        return cost, r_err, g_dist

    def forward(self, ee_pos_batch, ee_rot_batch, goal: Goal, link_name: Optional[str] = None):
        if link_name is None:
            goal_pose = goal.goal_pose
        else:
            goal_pose = goal.links_goal_pose[link_name]

        ee_goal_pos = goal_pose.position
        ee_goal_rot = goal_pose.quaternion
        num_goals = goal_pose.n_goalset
        self._update_cost_type(ee_goal_pos, ee_pos_batch, num_goals)
        # print(self.cost_type)
        b, h, _ = ee_pos_batch.shape
        self.update_batch_size(b, h)
        # return self.out_distance
        # print(b,h, ee_goal_pos.shape)

        distance = PoseError.apply(
            ee_pos_batch,
            ee_goal_pos,
            ee_rot_batch,  # .view(-1, 4).contiguous(),
            ee_goal_rot,
            self.vec_weight,
            self.weight,
            self._vec_convergence,
            self._run_weight_vec,
            self.run_vec_weight,
            self.offset_waypoint,
            self.offset_tstep_fraction,
            goal.batch_pose_idx,
            self.project_distance_tensor,
            self.out_distance,
            self.out_position_distance,
            self.out_rotation_distance,
            self.out_p_vec,
            self.out_q_vec,
            self.out_idx,
            self.out_p_grad,
            self.out_q_grad,
            b,
            h,
            self.cost_type.value,
            num_goals,
            self.use_metric,
            self.return_loss,
        )

        cost = distance
        # if link_name is None and cost.shape[0]==8:
        #    print(ee_pos_batch[...,-1].squeeze())
        # print(cost.shape)
        return cost

    def forward_pose(
        self,
        goal_pose: Pose,
        query_pose: Pose,
        batch_pose_idx: torch.Tensor,
        mode: PoseErrorType = PoseErrorType.BATCH_GOAL,
    ):
        if len(query_pose.position.shape) == 2:
            log_error("Query pose should be [batch, horizon, -1]")
        ee_goal_pos = goal_pose.position
        ee_goal_quat = goal_pose.quaternion
        self.cost_type = mode

        self.update_batch_size(query_pose.position.shape[0], query_pose.position.shape[1])
        b = query_pose.position.shape[0]
        h = query_pose.position.shape[1]
        num_goals = 1

        distance = PoseError.apply(
            query_pose.position,
            ee_goal_pos,
            query_pose.quaternion,
            ee_goal_quat,
            self.vec_weight,
            self.weight,
            self._vec_convergence,
            self._run_weight_vec,
            self.run_vec_weight,
            self.offset_waypoint,
            self.offset_tstep_fraction,
            batch_pose_idx,
            self.project_distance_tensor,
            self.out_distance,
            self.out_position_distance,
            self.out_rotation_distance,
            self.out_p_vec,
            self.out_q_vec,
            self.out_idx,
            self.out_p_grad,
            self.out_q_grad,
            b,
            h,
            self.cost_type.value,
            num_goals,
            self.use_metric,
            self.return_loss,
        )
        return distance
