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
from typing import Dict, List, Optional

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo.geom.sdf.world import WorldCollision
from curobo.rollout.cost.cost_base import CostConfig
from curobo.rollout.cost.dist_cost import DistCost, DistCostConfig
from curobo.rollout.cost.pose_cost import PoseCost, PoseCostConfig, PoseCostMetric
from curobo.rollout.cost.straight_line_cost import StraightLineCost
from curobo.rollout.cost.zero_cost import ZeroCost
from curobo.rollout.dynamics_model.kinematic_model import KinematicModelState
from curobo.rollout.rollout_base import Goal, RolloutMetrics
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.types.tensor import T_BValue_float, T_BValue_int
from curobo.util.helpers import list_idx_if_not_none
from curobo.util.logger import log_error, log_info, log_warn
from curobo.util.tensor_util import cat_max
from curobo.util.torch_utils import get_torch_jit_decorator

# Local Folder
from .arm_base import ArmBase, ArmBaseConfig, ArmCostConfig


@dataclass
class ArmReacherMetrics(RolloutMetrics):
    cspace_error: Optional[T_BValue_float] = None
    position_error: Optional[T_BValue_float] = None
    rotation_error: Optional[T_BValue_float] = None
    pose_error: Optional[T_BValue_float] = None
    goalset_index: Optional[T_BValue_int] = None
    null_space_error: Optional[T_BValue_float] = None

    def __getitem__(self, idx):
        d_list = [
            self.cost,
            self.constraint,
            self.feasible,
            self.state,
            self.cspace_error,
            self.position_error,
            self.rotation_error,
            self.pose_error,
            self.goalset_index,
            self.null_space_error,
        ]
        idx_vals = list_idx_if_not_none(d_list, idx)
        return ArmReacherMetrics(*idx_vals)

    def clone(self, clone_state=False):
        if clone_state:
            raise NotImplementedError()
        return ArmReacherMetrics(
            cost=None if self.cost is None else self.cost.clone(),
            constraint=None if self.constraint is None else self.constraint.clone(),
            feasible=None if self.feasible is None else self.feasible.clone(),
            state=None if self.state is None else self.state,
            cspace_error=None if self.cspace_error is None else self.cspace_error.clone(),
            position_error=None if self.position_error is None else self.position_error.clone(),
            rotation_error=None if self.rotation_error is None else self.rotation_error.clone(),
            pose_error=None if self.pose_error is None else self.pose_error.clone(),
            goalset_index=None if self.goalset_index is None else self.goalset_index.clone(),
            null_space_error=(
                None if self.null_space_error is None else self.null_space_error.clone()
            ),
        )


@dataclass
class ArmReacherCostConfig(ArmCostConfig):
    pose_cfg: Optional[PoseCostConfig] = None
    cspace_cfg: Optional[DistCostConfig] = None
    straight_line_cfg: Optional[CostConfig] = None
    zero_acc_cfg: Optional[CostConfig] = None
    zero_vel_cfg: Optional[CostConfig] = None
    zero_jerk_cfg: Optional[CostConfig] = None
    link_pose_cfg: Optional[PoseCostConfig] = None

    @staticmethod
    def _get_base_keys():
        base_k = ArmCostConfig._get_base_keys()
        # add new cost terms:
        new_k = {
            "pose_cfg": PoseCostConfig,
            "cspace_cfg": DistCostConfig,
            "straight_line_cfg": CostConfig,
            "zero_acc_cfg": CostConfig,
            "zero_vel_cfg": CostConfig,
            "zero_jerk_cfg": CostConfig,
            "link_pose_cfg": PoseCostConfig,
        }
        new_k.update(base_k)
        return new_k

    @staticmethod
    def from_dict(
        data_dict: Dict,
        robot_cfg: RobotConfig,
        world_coll_checker: Optional[WorldCollision] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ):
        k_list = ArmReacherCostConfig._get_base_keys()
        data = ArmCostConfig._get_formatted_dict(
            data_dict,
            k_list,
            robot_cfg,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
        )
        return ArmReacherCostConfig(**data)


@dataclass
class ArmReacherConfig(ArmBaseConfig):
    cost_cfg: ArmReacherCostConfig
    constraint_cfg: ArmReacherCostConfig
    convergence_cfg: ArmReacherCostConfig

    @staticmethod
    def cost_from_dict(
        cost_data_dict: Dict,
        robot_cfg: RobotConfig,
        world_coll_checker: Optional[WorldCollision] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ):
        return ArmReacherCostConfig.from_dict(
            cost_data_dict,
            robot_cfg,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
        )


@get_torch_jit_decorator()
def _compute_g_dist_jit(rot_err_norm, goal_dist):
    # goal_cost = goal_cost.view(cost.shape)
    # rot_err_norm = rot_err_norm.view(cost.shape)
    # goal_dist = goal_dist.view(cost.shape)
    g_dist = goal_dist.unsqueeze(-1) + 10.0 * rot_err_norm.unsqueeze(-1)
    return g_dist


class ArmReacher(ArmBase, ArmReacherConfig):
    """
    .. inheritance-diagram:: curobo.rollout.arm_reacher.ArmReacher
    """

    @profiler.record_function("arm_reacher/init")
    def __init__(self, config: Optional[ArmReacherConfig] = None):
        if config is not None:
            ArmReacherConfig.__init__(self, **vars(config))
        ArmBase.__init__(self)

        # self.goal_state = None
        # self.goal_ee_pos = None
        # self.goal_ee_rot = None
        # self.goal_ee_quat = None
        self._compute_g_dist = False
        self._n_goalset = 1

        if self.cost_cfg.cspace_cfg is not None:
            self.cost_cfg.cspace_cfg.dof = self.d_action
            # self.cost_cfg.cspace_cfg.update_vec_weight(self.dynamics_model.cspace_distance_weight)
            self.dist_cost = DistCost(self.cost_cfg.cspace_cfg)
        if self.cost_cfg.pose_cfg is not None:
            self.cost_cfg.pose_cfg.waypoint_horizon = self.horizon
            self.goal_cost = PoseCost(self.cost_cfg.pose_cfg)
            if self.cost_cfg.link_pose_cfg is None:
                log_info(
                    "Deprecated: Add link_pose_cfg to your rollout config. Using pose_cfg instead."
                )
                self.cost_cfg.link_pose_cfg = self.cost_cfg.pose_cfg
        self._link_pose_costs = {}

        if self.cost_cfg.link_pose_cfg is not None:
            for i in self.kinematics.link_names:
                if i != self.kinematics.ee_link:
                    self._link_pose_costs[i] = PoseCost(self.cost_cfg.link_pose_cfg)
        if self.cost_cfg.straight_line_cfg is not None:
            self.straight_line_cost = StraightLineCost(self.cost_cfg.straight_line_cfg)
        if self.cost_cfg.zero_vel_cfg is not None:
            self.zero_vel_cost = ZeroCost(self.cost_cfg.zero_vel_cfg)
            self._max_vel = self.state_bounds["velocity"][1]
            if self.zero_vel_cost.hinge_value is not None:
                self._compute_g_dist = True
        if self.cost_cfg.zero_acc_cfg is not None:
            self.zero_acc_cost = ZeroCost(self.cost_cfg.zero_acc_cfg)
            self._max_vel = self.state_bounds["velocity"][1]
            if self.zero_acc_cost.hinge_value is not None:
                self._compute_g_dist = True

        if self.cost_cfg.zero_jerk_cfg is not None:
            self.zero_jerk_cost = ZeroCost(self.cost_cfg.zero_jerk_cfg)
            self._max_vel = self.state_bounds["velocity"][1]
            if self.zero_jerk_cost.hinge_value is not None:
                self._compute_g_dist = True

        self.z_tensor = torch.tensor(
            0, device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )
        self._link_pose_convergence = {}

        if self.convergence_cfg.pose_cfg is not None:
            self.pose_convergence = PoseCost(self.convergence_cfg.pose_cfg)
            if self.convergence_cfg.link_pose_cfg is None:
                log_warn(
                    "Deprecated: Add link_pose_cfg to your rollout config. Using pose_cfg instead."
                )
                self.convergence_cfg.link_pose_cfg = self.convergence_cfg.pose_cfg

        if self.convergence_cfg.link_pose_cfg is not None:
            for i in self.kinematics.link_names:
                if i != self.kinematics.ee_link:
                    self._link_pose_convergence[i] = PoseCost(self.convergence_cfg.link_pose_cfg)
        if self.convergence_cfg.cspace_cfg is not None:
            self.convergence_cfg.cspace_cfg.dof = self.d_action
            self.cspace_convergence = DistCost(self.convergence_cfg.cspace_cfg)

        # check if g_dist is required in any of the cost terms:
        self.update_params(Goal(current_state=self._start_state))

    def cost_fn(self, state: KinematicModelState, action_batch=None):
        """
        Compute cost given that state dictionary and actions


        :class:`curobo.rollout.cost.PoseCost`
        :class:`curobo.rollout.cost.DistCost`

        """
        state_batch = state.state_seq
        with profiler.record_function("cost/base"):
            cost_list = super(ArmReacher, self).cost_fn(state, action_batch, return_list=True)
        ee_pos_batch, ee_quat_batch = state.ee_pos_seq, state.ee_quat_seq
        g_dist = None
        with profiler.record_function("cost/pose"):
            if (
                self._goal_buffer.goal_pose.position is not None
                and self.cost_cfg.pose_cfg is not None
                and self.goal_cost.enabled
            ):
                if self._compute_g_dist:
                    goal_cost, rot_err_norm, goal_dist = self.goal_cost.forward_out_distance(
                        ee_pos_batch,
                        ee_quat_batch,
                        self._goal_buffer,
                    )

                    g_dist = _compute_g_dist_jit(rot_err_norm, goal_dist)
                else:
                    goal_cost = self.goal_cost.forward(
                        ee_pos_batch, ee_quat_batch, self._goal_buffer
                    )
                cost_list.append(goal_cost)
        with profiler.record_function("cost/link_poses"):
            if self._goal_buffer.links_goal_pose is not None and self.cost_cfg.pose_cfg is not None:
                link_poses = state.link_pose

                for k in self._goal_buffer.links_goal_pose.keys():
                    if k != self.kinematics.ee_link:
                        current_fn = self._link_pose_costs[k]
                        if current_fn.enabled:
                            # get link pose
                            current_pose = link_poses[k].contiguous()
                            current_pos = current_pose.position
                            current_quat = current_pose.quaternion

                            c = current_fn.forward(current_pos, current_quat, self._goal_buffer, k)
                            cost_list.append(c)

        if (
            self._goal_buffer.goal_state is not None
            and self.cost_cfg.cspace_cfg is not None
            and self.dist_cost.enabled
        ):

            joint_cost = self.dist_cost.forward_target_idx(
                self._goal_buffer.goal_state.position,
                state_batch.position,
                self._goal_buffer.batch_goal_state_idx,
            )
            cost_list.append(joint_cost)
        if self.cost_cfg.straight_line_cfg is not None and self.straight_line_cost.enabled:
            st_cost = self.straight_line_cost.forward(ee_pos_batch)
            cost_list.append(st_cost)

        if (
            self.cost_cfg.zero_acc_cfg is not None
            and self.zero_acc_cost.enabled
            # and g_dist is not None
        ):
            z_acc = self.zero_acc_cost.forward(
                state_batch.acceleration,
                g_dist,
            )

            cost_list.append(z_acc)
        if self.cost_cfg.zero_jerk_cfg is not None and self.zero_jerk_cost.enabled:
            z_jerk = self.zero_jerk_cost.forward(
                state_batch.jerk,
                g_dist,
            )
            cost_list.append(z_jerk)

        if self.cost_cfg.zero_vel_cfg is not None and self.zero_vel_cost.enabled:
            z_vel = self.zero_vel_cost.forward(
                state_batch.velocity,
                g_dist,
            )
            cost_list.append(z_vel)
        with profiler.record_function("cat_sum"):
            if self.sum_horizon:
                cost = cat_sum_horizon_reacher(cost_list)
            else:
                cost = cat_sum_reacher(cost_list)

        return cost

    def convergence_fn(
        self, state: KinematicModelState, out_metrics: Optional[ArmReacherMetrics] = None
    ) -> ArmReacherMetrics:
        if out_metrics is None:
            out_metrics = ArmReacherMetrics()
        if not isinstance(out_metrics, ArmReacherMetrics):
            out_metrics = ArmReacherMetrics(**vars(out_metrics))
        out_metrics = super(ArmReacher, self).convergence_fn(state, out_metrics)

        # compute error with pose?
        if (
            self._goal_buffer.goal_pose.position is not None
            and self.convergence_cfg.pose_cfg is not None
        ):
            (
                out_metrics.pose_error,
                out_metrics.rotation_error,
                out_metrics.position_error,
            ) = self.pose_convergence.forward_out_distance(
                state.ee_pos_seq, state.ee_quat_seq, self._goal_buffer
            )
            out_metrics.goalset_index = self.pose_convergence.goalset_index_buffer  # .clone()
        if (
            self._goal_buffer.links_goal_pose is not None
            and self.convergence_cfg.pose_cfg is not None
        ):
            pose_error = [out_metrics.pose_error]
            position_error = [out_metrics.position_error]
            quat_error = [out_metrics.rotation_error]
            link_poses = state.link_pose

            for k in self._goal_buffer.links_goal_pose.keys():
                if k != self.kinematics.ee_link:
                    current_fn = self._link_pose_convergence[k]
                    if current_fn.enabled:
                        # get link pose
                        current_pos = link_poses[k].position.contiguous()
                        current_quat = link_poses[k].quaternion.contiguous()

                        pose_err, pos_err, quat_err = current_fn.forward_out_distance(
                            current_pos, current_quat, self._goal_buffer, k
                        )
                        pose_error.append(pose_err)
                        position_error.append(pos_err)
                        quat_error.append(quat_err)
            out_metrics.pose_error = cat_max(pose_error)
            out_metrics.rotation_error = cat_max(quat_error)
            out_metrics.position_error = cat_max(position_error)

        if (
            self._goal_buffer.goal_state is not None
            and self.convergence_cfg.cspace_cfg is not None
            and self.cspace_convergence.enabled
        ):
            _, out_metrics.cspace_error = self.cspace_convergence.forward_target_idx(
                self._goal_buffer.goal_state.position,
                state.state_seq.position,
                self._goal_buffer.batch_goal_state_idx,
                True,
            )

        if (
            self.convergence_cfg.null_space_cfg is not None
            and self.null_convergence.enabled
            and self._goal_buffer.batch_retract_state_idx is not None
        ):
            out_metrics.null_space_error = self.null_convergence.forward_target_idx(
                self._goal_buffer.retract_state,
                state.state_seq.position,
                self._goal_buffer.batch_retract_state_idx,
            )

        return out_metrics

    def update_params(
        self,
        goal: Goal,
    ):
        """
        Update params for the cost terms and dynamics model.

        """

        super(ArmReacher, self).update_params(goal)
        if goal.batch_pose_idx is not None:
            self._goal_idx_update = False
        if goal.goal_pose.position is not None:
            self.enable_cspace_cost(False)
        return True

    def enable_pose_cost(self, enable: bool = True):
        if enable:
            self.goal_cost.enable_cost()
        else:
            self.goal_cost.disable_cost()

    def enable_cspace_cost(self, enable: bool = True):
        if enable:
            self.dist_cost.enable_cost()
            self.cspace_convergence.enable_cost()
        else:
            self.dist_cost.disable_cost()
            self.cspace_convergence.disable_cost()

    def get_pose_costs(
        self,
        include_link_pose: bool = False,
        include_convergence: bool = True,
        only_convergence: bool = False,
    ):
        if only_convergence:
            return [self.pose_convergence]
        pose_costs = [self.goal_cost]
        if include_convergence:
            pose_costs += [self.pose_convergence]
        if include_link_pose:
            log_error("Not implemented yet")
        return pose_costs

    def update_pose_cost_metric(
        self,
        metric: PoseCostMetric,
    ):
        pose_costs = self.get_pose_costs(
            include_link_pose=metric.include_link_pose, include_convergence=False
        )
        for p in pose_costs:
            p.update_metric(metric, update_offset_waypoint=True)

        pose_costs = self.get_pose_costs(only_convergence=True)
        for p in pose_costs:
            p.update_metric(metric, update_offset_waypoint=False)


@get_torch_jit_decorator()
def cat_sum_reacher(tensor_list: List[torch.Tensor]):
    cat_tensor = torch.sum(torch.stack(tensor_list, dim=0), dim=0)
    return cat_tensor


@get_torch_jit_decorator()
def cat_sum_horizon_reacher(tensor_list: List[torch.Tensor]):
    cat_tensor = torch.sum(torch.stack(tensor_list, dim=0), dim=(0, -1))
    return cat_tensor
