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
from typing import Optional

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo.types.robot import JointState
from curobo.types.tensor import T_DOF
from curobo.util.trajectory import calculate_dt


@dataclass
class TrajEvaluatorConfig:
    max_acc: float = 15.0
    max_jerk: float = 500.0
    cost_weight: float = 0.01
    min_dt: float = 0.001
    max_dt: float = 0.1


@torch.jit.script
def compute_path_length(vel, traj_dt, cspace_distance_weight):
    pl = torch.sum(
        torch.sum(torch.abs(vel) * traj_dt.unsqueeze(-1) * cspace_distance_weight, dim=-1), dim=-1
    )
    return pl


@torch.jit.script
def compute_path_length_cost(vel, cspace_distance_weight):
    pl = torch.sum(torch.sum(torch.abs(vel) * cspace_distance_weight, dim=-1), dim=-1)
    return pl


@torch.jit.script
def smooth_cost(abs_acc, abs_jerk, opt_dt):
    # acc = torch.max(torch.max(abs_acc, dim=-1)[0], dim=-1)[0]
    # jerk = torch.max(torch.max(abs_jerk, dim=-1)[0], dim=-1)[0]
    jerk = torch.mean(torch.max(abs_jerk, dim=-1)[0], dim=-1)
    mean_acc = torch.mean(torch.max(abs_acc, dim=-1)[0], dim=-1)  # [0]
    a = (jerk * 0.001) + 5.0 * opt_dt + (mean_acc * 0.01)

    return a


@torch.jit.script
def compute_smoothness(
    vel: torch.Tensor,
    acc: torch.Tensor,
    jerk: torch.Tensor,
    max_vel: torch.Tensor,
    max_acc: float,
    max_jerk: float,
    traj_dt: torch.Tensor,
    min_dt: float,
    max_dt: float,
):
    # compute scaled dt:
    # h = int(vel.shape[-2] / 2)

    max_v_arr = torch.max(torch.abs(vel), dim=-2)[0]  # output is batch, dof
    scale_dt = (max_v_arr) / (max_vel.view(1, max_v_arr.shape[-1]))  # batch,dof

    max_acc_arr = torch.max(torch.abs(acc), dim=-2)[0]  # batch, dof

    scale_dt_acc = torch.sqrt(torch.max(max_acc_arr / max_acc, dim=-1)[0])  # batch, 1

    max_jerk_arr = torch.max(torch.abs(jerk), dim=-2)[0]  # batch, dof

    scale_dt_jerk = torch.pow(torch.max(max_jerk_arr / max_jerk, dim=-1)[0], 1.0 / 3.0)  # batch, 1

    dt_score_vel = torch.max(scale_dt, dim=-1)[0]  # batch, 1

    dt_score = torch.maximum(dt_score_vel, scale_dt_acc)
    dt_score = torch.maximum(dt_score, scale_dt_jerk)

    # clamp dt score:

    dt_score = torch.clamp(dt_score, min_dt, max_dt)
    scale_dt = (1 / dt_score).view(-1, 1, 1)
    abs_acc = torch.abs(acc) * (scale_dt**2)
    # mean_acc_val = torch.max(torch.mean(abs_acc, dim=-1), dim=-1)[0]
    max_acc_val = torch.max(torch.max(abs_acc, dim=-1)[0], dim=-1)[0]
    abs_jerk = torch.abs(jerk) * scale_dt**3
    # calculate max mean jerk:
    # mean_jerk_val = torch.max(torch.mean(abs_jerk, dim=-1), dim=-1)[0]
    max_jerk_val = torch.max(torch.max(abs_jerk, dim=-1)[0], dim=-1)[0]
    acc_label = torch.logical_and(max_acc_val <= max_acc, max_jerk_val <= max_jerk)
    # print(max_acc_val, max_jerk_val, dt_score)
    return (acc_label, smooth_cost(abs_acc, abs_jerk, dt_score))


@torch.jit.script
def compute_smoothness_opt_dt(
    vel, acc, jerk, max_vel: torch.Tensor, max_acc: float, max_jerk: float, opt_dt: torch.Tensor
):
    abs_acc = torch.abs(acc)
    max_acc_val = torch.max(torch.max(abs_acc, dim=-1)[0], dim=-1)[0]
    abs_jerk = torch.abs(jerk)
    max_jerk_val = torch.max(torch.max(abs_jerk, dim=-1)[0], dim=-1)[0]

    acc_label = torch.logical_and(max_acc_val <= max_acc, max_jerk_val <= max_jerk)
    return acc_label, smooth_cost(abs_acc, abs_jerk, opt_dt)


class TrajEvaluator(TrajEvaluatorConfig):
    def __init__(self, config: Optional[TrajEvaluatorConfig] = None):
        if config is None:
            config = TrajEvaluatorConfig()
        super().__init__(**vars(config))

    def _compute_path_length(
        self, js: JointState, traj_dt: torch.Tensor, cspace_distance_weight: T_DOF
    ):
        path_length = compute_path_length(js.velocity, traj_dt, cspace_distance_weight)
        return path_length

    def _check_smoothness(self, js: JointState, traj_dt: torch.Tensor, max_vel: torch.Tensor):
        if js.jerk is None:
            js.jerk = (
                (torch.roll(js.acceleration, -1, -2) - js.acceleration)
                * (1 / traj_dt).unsqueeze(-1)
            )[..., :-1, :]

        acc_label, max_acc = compute_smoothness(
            js.velocity,
            js.acceleration,
            js.jerk,
            max_vel,
            self.max_acc,
            self.max_jerk,
            traj_dt,
            self.min_dt,
            self.max_dt,
        )
        return acc_label, max_acc

    @profiler.record_function("traj_evaluate_smoothness")
    def evaluate(
        self,
        js: JointState,
        traj_dt: torch.Tensor,
        cspace_distance_weight: T_DOF,
        max_vel: torch.Tensor,
    ):
        label, cost = self._check_smoothness(js, traj_dt, max_vel)
        pl_cost = self._compute_path_length(js, traj_dt, cspace_distance_weight)
        return label, pl_cost + self.cost_weight * cost

    @profiler.record_function("traj_evaluate_interpolated_smoothness")
    def evaluate_interpolated_smootheness(
        self,
        js: JointState,
        opt_dt: torch.Tensor,
        cspace_distance_weight: T_DOF,
        max_vel: torch.Tensor,
    ):
        label, cost = compute_smoothness_opt_dt(
            js.velocity, js.acceleration, js.jerk, max_vel, self.max_acc, self.max_jerk, opt_dt
        )
        label = torch.logical_and(label, opt_dt <= self.max_dt)
        pl_cost = compute_path_length_cost(js.velocity, cspace_distance_weight)
        return label, pl_cost + self.cost_weight * cost

    def evaluate_from_position(
        self,
        js: JointState,
        traj_dt: torch.Tensor,
        cspace_distance_weight: T_DOF,
        max_vel: torch.Tensor,
        skip_last_tstep: bool = False,
    ):
        js = js.calculate_fd_from_position(traj_dt)
        if skip_last_tstep:
            js.position = js.position[..., :-1, :]
            js.velocity = js.velocity[..., :-1, :]
            js.acceleration = js.acceleration[..., :-1, :]
            js.jerk = js.jerk[..., :-1, :]

        return self.evaluate(js, traj_dt, cspace_distance_weight, max_vel)

    def calculate_dt(self, js: JointState, max_vel: torch.Tensor, raw_dt: float):
        return calculate_dt(js.velocity, max_vel, raw_dt)
