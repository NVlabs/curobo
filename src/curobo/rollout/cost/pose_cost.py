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
from enum import Enum
from typing import List, Optional

# Third Party
import torch
from torch.autograd import Function

# CuRobo
from curobo.curobolib.geom import get_pose_distance, get_pose_distance_backward
from curobo.rollout.rollout_base import Goal
from curobo.types.math import OrientationError, Pose

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
    run_vec_weight: Optional[List[float]] = None

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
        return super().__post_init__()


@torch.jit.script
def backward_PoseError_jit(grad_g_dist, grad_out_distance, weight, g_vec):
    grad_vec = grad_g_dist + (grad_out_distance * weight)
    grad = 1.0 * (grad_vec).unsqueeze(-1) * g_vec
    return grad


# full method:
@torch.jit.script
def backward_full_PoseError_jit(
    grad_out_distance, grad_g_dist, grad_r_err, p_w, q_w, g_vec_p, g_vec_q
):
    p_grad = (grad_g_dist + (grad_out_distance * p_w)).unsqueeze(-1) * g_vec_p
    q_grad = (grad_r_err + (grad_out_distance * q_w)).unsqueeze(-1) * g_vec_q
    # p_grad = ((grad_out_distance * p_w)).unsqueeze(-1) * g_vec_p
    # q_grad = ((grad_out_distance * q_w)).unsqueeze(-1) * g_vec_q

    return p_grad, q_grad


class PoseErrorDistance(Function):
    @staticmethod
    def forward(
        ctx,
        current_position,
        goal_position,
        current_quat,
        goal_quat,
        vec_weight,
        weight,
        vec_convergence,
        run_weight,
        run_vec_weight,
        batch_pose_idx,
        out_distance,
        out_position_distance,
        out_rotation_distance,
        out_p_vec,
        out_r_vec,
        out_idx,
        out_p_grad,
        out_q_grad,
        batch_size,
        horizon,
        mode=PoseErrorType.BATCH_GOAL.value,
        num_goals=1,
        use_metric=False,
    ):
        # out_distance = current_position[..., 0].detach().clone() * 0.0
        # out_position_distance = out_distance.detach().clone()
        # out_rotation_distance = out_distance.detach().clone()
        # out_vec = (
        #    torch.cat((current_position.detach().clone(), current_quat.detach().clone()), dim=-1)
        #    * 0.0
        # )
        # out_idx = out_distance.clone().to(dtype=torch.long)

        (
            out_distance,
            out_position_distance,
            out_rotation_distance,
            out_p_vec,
            out_r_vec,
            out_idx,
        ) = get_pose_distance(
            out_distance,
            out_position_distance,
            out_rotation_distance,
            out_p_vec,
            out_r_vec,
            out_idx,
            current_position.contiguous(),
            goal_position,
            current_quat.contiguous(),
            goal_quat,
            vec_weight,
            weight,
            vec_convergence,
            run_weight,
            run_vec_weight,
            batch_pose_idx,
            batch_size,
            horizon,
            mode,
            num_goals,
            current_position.requires_grad,
            True,
            use_metric,
        )
        ctx.save_for_backward(out_p_vec, out_r_vec, weight, out_p_grad, out_q_grad)
        return out_distance, out_position_distance, out_rotation_distance, out_idx  # .view(-1,1)

    @staticmethod
    def backward(ctx, grad_out_distance, grad_g_dist, grad_r_err, grad_out_idx):
        (g_vec_p, g_vec_q, weight, out_grad_p, out_grad_q) = ctx.saved_tensors
        pos_grad = None
        quat_grad = None
        batch_size = g_vec_p.shape[0] * g_vec_p.shape[1]
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[2]:
            pos_grad, quat_grad = get_pose_distance_backward(
                out_grad_p,
                out_grad_q,
                grad_out_distance.contiguous(),
                grad_g_dist.contiguous(),
                grad_r_err.contiguous(),
                weight,
                g_vec_p,
                g_vec_q,
                batch_size,
                use_distance=True,
            )
            # pos_grad, quat_grad = backward_full_PoseError_jit(
            #    grad_out_distance,
            #    grad_g_dist, grad_r_err, p_w, q_w, g_vec_p, g_vec_q
            # )
        elif ctx.needs_input_grad[0]:
            pos_grad = backward_PoseError_jit(grad_g_dist, grad_out_distance, p_w, g_vec_p)
            # grad_vec = grad_g_dist + (grad_out_distance * weight[1])
            # pos_grad = 1.0 * (grad_vec).unsqueeze(-1) * g_vec[..., 4:]
        elif ctx.needs_input_grad[2]:
            quat_grad = backward_PoseError_jit(grad_r_err, grad_out_distance, q_w, g_vec_q)
            # grad_vec = grad_r_err + (grad_out_distance * weight[0])
            # quat_grad = 1.0 * (grad_vec).unsqueeze(-1) * g_vec[..., :4]
        return (
            pos_grad,
            None,
            quat_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class PoseLoss(Function):
    @staticmethod
    def forward(
        ctx,
        current_position,
        goal_position,
        current_quat,
        goal_quat,
        vec_weight,
        weight,
        vec_convergence,
        run_weight,
        run_vec_weight,
        batch_pose_idx,
        out_distance,
        out_position_distance,
        out_rotation_distance,
        out_p_vec,
        out_r_vec,
        out_idx,
        out_p_grad,
        out_q_grad,
        batch_size,
        horizon,
        mode=PoseErrorType.BATCH_GOAL.value,
        num_goals=1,
        use_metric=False,
    ):
        # out_distance = current_position[..., 0].detach().clone() * 0.0
        # out_position_distance = out_distance.detach().clone()
        # out_rotation_distance = out_distance.detach().clone()
        # out_vec = (
        #    torch.cat((current_position.detach().clone(), current_quat.detach().clone()), dim=-1)
        #    * 0.0
        # )
        # out_idx = out_distance.clone().to(dtype=torch.long)

        (
            out_distance,
            out_position_distance,
            out_rotation_distance,
            out_p_vec,
            out_r_vec,
            out_idx,
        ) = get_pose_distance(
            out_distance,
            out_position_distance,
            out_rotation_distance,
            out_p_vec,
            out_r_vec,
            out_idx,
            current_position.contiguous(),
            goal_position,
            current_quat.contiguous(),
            goal_quat,
            vec_weight,
            weight,
            vec_convergence,
            run_weight,
            run_vec_weight,
            batch_pose_idx,
            batch_size,
            horizon,
            mode,
            num_goals,
            current_position.requires_grad,
            False,
            use_metric,
        )
        ctx.save_for_backward(out_p_vec, out_r_vec)
        return out_distance

    @staticmethod
    def backward(ctx, grad_out_distance):  # , grad_g_dist, grad_r_err, grad_out_idx):
        pos_grad = None
        quat_grad = None
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[2]:
            (g_vec_p, g_vec_q) = ctx.saved_tensors
            pos_grad = g_vec_p * grad_out_distance.unsqueeze(1)
            quat_grad = g_vec_q * grad_out_distance.unsqueeze(1)
            pos_grad = pos_grad.unsqueeze(-2)
            quat_grad = quat_grad.unsqueeze(-2)
        elif ctx.needs_input_grad[0]:
            (g_vec_p, g_vec_q) = ctx.saved_tensors

            pos_grad = g_vec_p * grad_out_distance.unsqueeze(1)
            pos_grad = pos_grad.unsqueeze(-2)
        elif ctx.needs_input_grad[2]:
            (g_vec_p, g_vec_q) = ctx.saved_tensors

            quat_grad = g_vec_q * grad_out_distance.unsqueeze(1)
            quat_grad = quat_grad.unsqueeze(-2)

        return (
            pos_grad,
            None,
            quat_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class PoseError(Function):
    @staticmethod
    def forward(
        ctx,
        current_position,
        goal_position,
        current_quat,
        goal_quat,
        vec_weight,
        weight,
        vec_convergence,
        run_weight,
        run_vec_weight,
        batch_pose_idx,
        out_distance,
        out_position_distance,
        out_rotation_distance,
        out_p_vec,
        out_r_vec,
        out_idx,
        out_p_grad,
        out_q_grad,
        batch_size,
        horizon,
        mode=PoseErrorType.BATCH_GOAL.value,
        num_goals=1,
        use_metric=False,
    ):
        # out_distance = current_position[..., 0].detach().clone() * 0.0
        # out_position_distance = out_distance.detach().clone()
        # out_rotation_distance = out_distance.detach().clone()
        # out_vec = (
        #    torch.cat((current_position.detach().clone(), current_quat.detach().clone()), dim=-1)
        #    * 0.0
        # )
        # out_idx = out_distance.clone().to(dtype=torch.long)

        (
            out_distance,
            out_position_distance,
            out_rotation_distance,
            out_p_vec,
            out_r_vec,
            out_idx,
        ) = get_pose_distance(
            out_distance,
            out_position_distance,
            out_rotation_distance,
            out_p_vec,
            out_r_vec,
            out_idx,
            current_position.contiguous(),
            goal_position,
            current_quat.contiguous(),
            goal_quat,
            vec_weight,
            weight,
            vec_convergence,
            run_weight,
            run_vec_weight,
            batch_pose_idx,
            batch_size,
            horizon,
            mode,
            num_goals,
            current_position.requires_grad,
            False,
            use_metric,
        )
        ctx.save_for_backward(out_p_vec, out_r_vec)
        return out_distance

    @staticmethod
    def backward(ctx, grad_out_distance):  # , grad_g_dist, grad_r_err, grad_out_idx):
        pos_grad = None
        quat_grad = None
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[2]:
            (g_vec_p, g_vec_q) = ctx.saved_tensors

            pos_grad = g_vec_p
            quat_grad = g_vec_q
        elif ctx.needs_input_grad[0]:
            (g_vec_p, g_vec_q) = ctx.saved_tensors

            pos_grad = g_vec_p
        elif ctx.needs_input_grad[2]:
            (g_vec_p, g_vec_q) = ctx.saved_tensors

            quat_grad = g_vec_q
        return (
            pos_grad,
            None,
            quat_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class PoseCost(CostBase, PoseCostConfig):
    def __init__(self, config: PoseCostConfig):
        PoseCostConfig.__init__(self, **vars(config))
        CostBase.__init__(self)
        self.rot_weight = self.vec_weight[0:3]
        self.pos_weight = self.vec_weight[3:6]

        self._vec_convergence = self.tensor_args.to_device(self.vec_convergence)
        self._batch_size = 0
        self._horizon = 0

    def update_batch_size(self, batch_size, horizon):
        if batch_size != self._batch_size or horizon != self._horizon:
            # batch_size = b*h
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
            if self._run_weight_vec is None or self._run_weight_vec.shape[1] != horizon:
                self._run_weight_vec = torch.ones(
                    (1, horizon), device=self.tensor_args.device, dtype=self.tensor_args.dtype
                )
            if self.terminal and self.run_weight is not None:
                self._run_weight_vec[:, :-1] *= self.run_weight

            self._batch_size = batch_size
            self._horizon = horizon

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
        if d_g == 2 and b_sze == 1:  # 1, 3
            self.cost_type = PoseErrorType.SINGLE_GOAL
        elif d_g == 2 and b_sze == ee_pos_batch.shape[0]:  # b, 3
            self.cost_type = PoseErrorType.BATCH_GOAL
        elif d_g == 3:
            self.cost_type = PoseErrorType.GOALSET
        elif len(ee_goal_pos.shape) == 4 and b_sze == ee_pos_bath.shape[0]:
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
            goal.batch_pose_idx,
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

        b, h, _ = ee_pos_batch.shape
        self.update_batch_size(b, h)
        # return self.out_distance
        # print(b,h, ee_goal_pos.shape)
        if self.return_loss:
            distance = PoseLoss.apply(
                ee_pos_batch,
                ee_goal_pos,
                ee_rot_batch,  # .view(-1, 4).contiguous(),
                ee_goal_rot,
                self.vec_weight,
                self.weight,
                self._vec_convergence,
                self._run_weight_vec,
                self.run_vec_weight,
                goal.batch_pose_idx,
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
        else:
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
                goal.batch_pose_idx,
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

        cost = distance
        return cost

    def forward_pose(
        self,
        goal_pose: Pose,
        query_pose: Pose,
        batch_pose_idx: torch.Tensor,
        mode: PoseErrorType = PoseErrorType.BATCH_GOAL,
    ):
        ee_goal_pos = goal_pose.position
        ee_goal_quat = goal_pose.quaternion
        self.cost_type = mode

        self.update_batch_size(query_pose.position.shape[0], query_pose.position.shape[1])
        b = query_pose.position.shape[0]
        h = query_pose.position.shape[1]
        num_goals = 1
        if self.return_loss:
            distance = PoseLoss.apply(
                query_pose.position.unsqueeze(1),
                ee_goal_pos,
                query_pose.quaternion.unsqueeze(1),
                ee_goal_quat,
                self.vec_weight,
                self.weight,
                self._vec_convergence,
                self._run_weight_vec,
                self.run_vec_weight,
                batch_pose_idx,
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
        else:
            distance = PoseError.apply(
                query_pose.position.unsqueeze(1),
                ee_goal_pos,
                query_pose.quaternion.unsqueeze(1),
                ee_goal_quat,
                self.vec_weight,
                self.weight,
                self._vec_convergence,
                self._run_weight_vec,
                self.run_vec_weight,
                batch_pose_idx,
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
        return distance
