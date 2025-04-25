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
"""This modules contains heuristics for scoring trajectories."""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import Optional, Tuple, Union

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.robot import JointState
from curobo.types.tensor import T_DOF
from curobo.util.logger import log_error
from curobo.util.torch_utils import get_torch_jit_decorator
from curobo.util.trajectory import calculate_dt


@dataclass
class TrajEvaluatorConfig:
    """Configurable Parameters for Trajectory Evaluator."""

    #: Maximum acceleration for each joint.
    max_acc: torch.Tensor

    #: Maximum jerk for each joint.
    max_jerk: torch.Tensor

    #: Minimum allowed time step for trajectory. Trajectories with time step less than this
    #: value will be rejected.
    min_dt: torch.Tensor

    #: Maximum allowed time step for trajectory. Trajectories with time step greater than this
    #: value will be rejected.
    max_dt: torch.Tensor

    #: Weight to scale smoothness cost, total cost = path length + cost_weight * smoothness cost.
    cost_weight: float = 0.01

    def __post_init__(self):
        """Checks if values are of correct type and converts them if possible."""
        if not isinstance(self.max_acc, torch.Tensor):
            log_error(
                "max_acc should be a torch.Tensor, this was changed recently, use "
                + "TrajEvaluatorConfig.from_basic() to create TrajEvaluatorConfig object"
            )
        if not isinstance(self.max_jerk, torch.Tensor):
            log_error(
                "max_jerk should be a torch.Tensor, this was changed recently, use "
                + "TrajEvaluatorConfig.from_basic() to create TrajEvaluatorConfig object"
            )
        if not isinstance(self.min_dt, torch.Tensor):
            self.min_dt = torch.as_tensor(
                self.min_dt, device=self.max_acc.device, dtype=self.max_acc.dtype
            )
        if not isinstance(self.max_dt, torch.Tensor):
            self.max_dt = torch.as_tensor(
                self.max_dt, device=self.max_acc.device, dtype=self.max_acc.dtype
            )

    @staticmethod
    def from_basic(
        dof: int,
        max_acc: float = 15.0,
        max_jerk: float = 500.0,
        cost_weight: float = 0.01,
        min_dt: float = 0.001,
        max_dt: float = 0.15,
        tensor_args: TensorDeviceType = TensorDeviceType(),
    ) -> TrajEvaluatorConfig:
        """Creates TrajEvaluatorConfig object from basic parameters.

        Args:
            dof: number of active joints in the robot.
            max_acc: maximum acceleration for all joints. Treats this as same for all joints.
            max_jerk: maximum jerk for all joints. Treats this as same for all joints.
            cost_weight: weight to scale smoothness cost.
            min_dt: minimum allowed time step between waypoints of a trajectory.
            max_dt: maximum allowed time step between waypoints of a trajectory.
            tensor_args: device and dtype for the tensors.

        Returns:
            TrajEvaluatorConfig: Configured Parameters for Trajectory Evaluator.
        """
        return TrajEvaluatorConfig(
            max_acc=torch.full((dof,), max_acc, device=tensor_args.device, dtype=tensor_args.dtype),
            max_jerk=torch.full(
                (dof,), max_jerk, device=tensor_args.device, dtype=tensor_args.dtype
            ),
            cost_weight=cost_weight,
            min_dt=torch.as_tensor(min_dt, device=tensor_args.device, dtype=tensor_args.dtype),
            max_dt=torch.as_tensor(max_dt, device=tensor_args.device, dtype=tensor_args.dtype),
        )


@get_torch_jit_decorator()
def compute_path_length(vel, traj_dt, cspace_distance_weight) -> torch.Tensor:
    """JIT compatible function to compute path length.

    Args:
        vel: joint space velocity tensor of shape (batch, horizon, dof).
        traj_dt: dt of trajectory tensor of shape (batch, horizon) or (1, horizon) or (1, 1).
        cspace_distance_weight: weight tensor of shape (dof).

    Returns:
        torch.Tensor: path length tensor of shape (batch).
    """
    pl = torch.sum(
        torch.sum(torch.abs(vel) * traj_dt.unsqueeze(-1) * cspace_distance_weight, dim=-1), dim=-1
    )
    return pl


@get_torch_jit_decorator()
def compute_path_length_cost(vel, cspace_distance_weight) -> torch.Tensor:
    """JIT compatible function to compute path length cost without considering time step.

    Args:
        vel: joint space velocity tensor of shape (batch, horizon, dof).
        cspace_distance_weight: weight tensor of shape (dof).

    Returns:
        torch.Tensor: path length cost tensor of shape (batch).
    """
    pl = torch.sum(torch.sum(torch.abs(vel) * cspace_distance_weight, dim=-1), dim=-1)
    return pl


@get_torch_jit_decorator()
def smooth_cost(abs_acc, abs_jerk, opt_dt) -> torch.Tensor:
    """JIT compatible function to compute smoothness cost.

    Args:
        abs_acc: absolute acceleration tensor of shape (batch, horizon, dof).
        abs_jerk: absolute jerk tensor of shape (batch, horizon, dof).
        opt_dt: optimal time step tensor of shape (batch).

    Returns:
        torch.Tensor: smoothness cost tensor of shape (batch).
    """
    jerk = torch.mean(torch.max(abs_jerk, dim=-1)[0], dim=-1)
    mean_acc = torch.mean(torch.max(abs_acc, dim=-1)[0], dim=-1)  # [0]
    a = (jerk * 0.001) + 10.0 * opt_dt + (mean_acc * 0.01)

    return a


@get_torch_jit_decorator()
def compute_smoothness(
    vel: torch.Tensor,
    acc: torch.Tensor,
    jerk: torch.Tensor,
    max_vel: torch.Tensor,
    max_acc: torch.Tensor,
    max_jerk: torch.Tensor,
    traj_dt: torch.Tensor,
    min_dt: float,
    max_dt: float,
    epsilon: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """JIT compatible function to compute smoothness.

    Args:
        vel: joint space velocity tensor of shape (batch, horizon, dof).
        acc: joint space acceleration tensor of shape (batch, horizon, dof).
        jerk: joint space jerk tensor of shape (batch, horizon, dof).
        max_vel: maximum velocity limits, used to find scaling factor for dt that pushes at least
            one joint to its limit, taking into account acceleration and jerk limits.
        max_acc: maximum acceleration limits, used to find scaling factor for dt that pushes at least
            one joint to its limit, taking into account velocity and jerk limits.
        max_jerk: maximum jerk limits, used to find scaling factor for dt that pushes at least
            one joint to its limit, taking into account velocity and acceleration limits.
        traj_dt: dt of trajectory tensor of shape (batch, horizon) or (1, horizon)
        min_dt: minimum delta time allowed between steps/waypoints in a trajectory.
        max_dt: maximum delta time allowed between steps/waypoints in a trajectory.
        epsilon: relaxes evaluation of velocity, acceleration, and jerk limits to allow for
            numerical errors.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: success label tensor of shape (batch) and
            smoothness cost tensor of shape (batch)
    """
    # compute scaled dt:

    max_v_arr = torch.max(torch.abs(vel), dim=-2)[0]  # output is batch, dof
    scale_dt = (max_v_arr) / (max_vel.view(1, max_v_arr.shape[-1]))  # batch,dof

    max_acc_arr = torch.max(torch.abs(acc), dim=-2)[0]  # batch, dof
    max_acc = max_acc.view(1, max_acc_arr.shape[-1])
    scale_dt_acc = torch.sqrt(torch.max(max_acc_arr / max_acc, dim=-1)[0])  # batch, 1

    max_jerk_arr = torch.max(torch.abs(jerk), dim=-2)[0]  # batch, dof

    max_jerk = max_jerk.view(1, max_jerk_arr.shape[-1])
    scale_dt_jerk = torch.pow(torch.max(max_jerk_arr / max_jerk, dim=-1)[0], 1.0 / 3.0)  # batch, 1

    dt_score_vel = torch.max(scale_dt, dim=-1)[0]  # batch, 1

    dt_score = torch.maximum(dt_score_vel, scale_dt_acc)
    dt_score = torch.maximum(dt_score, scale_dt_jerk)

    # clamp dt score:

    dt_score = torch.clamp(dt_score * (1 + epsilon), min_dt, max_dt)
    scale_dt = (1 / dt_score).view(-1, 1, 1)
    abs_acc = torch.abs(acc) * (scale_dt**2)
    # mean_acc_val = torch.max(torch.mean(abs_acc, dim=-1), dim=-1)[0]
    max_acc_val = torch.max(abs_acc, dim=-2)[0]  # batch x dof
    abs_jerk = torch.abs(jerk) * scale_dt**3
    # calculate max mean jerk:
    # mean_jerk_val = torch.max(torch.mean(abs_jerk, dim=-1), dim=-1)[0]
    # max_jerk_val = torch.max(torch.max(abs_jerk, dim=-1)[0], dim=-1)[0]
    max_jerk_val = torch.max(abs_jerk, dim=-2)[0]  # batch x dof
    acc_label = torch.logical_and(max_acc_val <= max_acc, max_jerk_val <= max_jerk)
    acc_label = torch.all(acc_label, dim=-1)
    return (acc_label, smooth_cost(abs_acc, abs_jerk, dt_score))


@get_torch_jit_decorator()
def compute_smoothness_opt_dt(
    vel,
    acc,
    jerk,
    max_vel: torch.Tensor,
    max_acc: torch.Tensor,
    max_jerk: torch.Tensor,
    opt_dt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """JIT compatible function to compute smoothness with pre-computed optimal time step.

    Args:
        vel: joint space velocity tensor of shape (batch, horizon, dof), not used in this
            implementation.
        acc: joint space acceleration tensor of shape (batch, horizon, dof).
        jerk: joint space jerk tensor of shape (batch, horizon, dof).
        max_vel: maximum velocity limit, not used in this implementation.
        max_acc: maximum acceleration limit, used to reject trajectories.
        max_jerk: maximum jerk limit, used to reject trajectories.
        opt_dt: optimal time step tensor of shape (batch).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: success label tensor of shape (batch) and
            smoothness cost tensor of shape (batch).
    """
    abs_acc = torch.abs(acc)
    abs_jerk = torch.abs(jerk)

    delta_acc = torch.min(torch.min(max_acc - abs_acc, dim=-1)[0], dim=-1)[0]

    max_jerk = max_jerk.view(1, abs_jerk.shape[-1])
    delta_jerk = torch.min(torch.min(max_jerk - abs_jerk, dim=-1)[0], dim=-1)[0]
    acc_label = torch.logical_and(delta_acc >= 0.0, delta_jerk >= 0.0)
    return acc_label, smooth_cost(abs_acc, abs_jerk, opt_dt)


class TrajEvaluator(TrajEvaluatorConfig):
    """Trajectory Evaluator class that uses heuristics to score trajectories."""

    def __init__(self, config: Optional[TrajEvaluatorConfig] = None):
        """Initializes the TrajEvaluator object.

        Args:
            config: Configurable parameters for Trajectory Evaluator.
        """
        if config is None:
            config = TrajEvaluatorConfig()
        super().__init__(**vars(config))

    def _compute_path_length(
        self, js: JointState, traj_dt: torch.Tensor, cspace_distance_weight: T_DOF
    ):
        """Compute path length from joint velocities across trajectory and dt between them.

        Args:
            js: joint state object with velocity tensor.
            traj_dt: time step tensor of shape (batch, horizon) or (1, horizon) or (1, 1).
            cspace_distance_weight: weight tensor of shape (dof).

        Returns:
            torch.Tensor: path length tensor of shape (batch).
        """
        path_length = compute_path_length(js.velocity, traj_dt, cspace_distance_weight)
        return path_length

    def _check_smoothness(
        self, js: JointState, traj_dt: torch.Tensor, max_vel: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Check smoothness of trajectory.

        Args:
            js: joint state object with velocity, acceleration and jerk tensors.
            traj_dt: time step tensor of shape (batch, horizon) or (1, horizon) or (1, 1).
            max_vel: maximum velocity limits, used to find scaling factor for dt that pushes at
                least one joint to its limit, taking into account acceleration and jerk limits.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: success label tensor of shape (batch) and
                smoothness cost tensor of shape (batch).
        """
        if js.jerk is None:
            js.jerk = (
                (torch.roll(js.acceleration, -1, -2) - js.acceleration)
                * (1 / traj_dt).unsqueeze(-1)
            )[..., :-1, :]

        smooth_success_label, smooth_cost = compute_smoothness(
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
        return smooth_success_label, smooth_cost

    @profiler.record_function("traj_evaluate_smoothness")
    def evaluate(
        self,
        js: JointState,
        traj_dt: torch.Tensor,
        cspace_distance_weight: T_DOF,
        max_vel: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate trajectory based on smoothness and path length.

        Args:
            js: joint state object with velocity, acceleration and jerk tensors.
            traj_dt: time step tensor of shape (batch, horizon) or (1, horizon) or (1, 1) or
                (batch, 1).
            cspace_distance_weight: weight tensor of shape (dof).
            max_vel: maximum velocity limits, used to find scaling factor for dt that pushes at
                least one joint to its limit, taking into account acceleration and jerk limits.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: success label tensor of shape (batch) and
                total cost tensor of shape (batch) where total cost = (path length +
                cost_weight * smoothness cost).
        """
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate trajectory based on smoothness and path length with pre-computed optimal dt.

        Args:
            js: joint state object with velocity, acceleration and jerk tensors.
            opt_dt: optimal time step tensor of shape (batch).
            cspace_distance_weight: weight tensor of shape (dof).
            max_vel: maximum velocity limits, used to find scaling factor for dt that pushes at
                least one joint to its limit, taking into account acceleration and jerk limits.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: success label tensor of shape (batch) and
                total cost tensor of shape (batch) where total cost = (path length +
                cost_weight * smoothness cost).
        """
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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate trajectory by first computing velocity, acceleration and jerk from position.

        Args:
            js: joint state object with position tensor.
            traj_dt: time step tensor of shape (batch, 1) or (1, 1).
            cspace_distance_weight: weight tensor of shape (dof).
            max_vel: maximum velocity limits, used to find scaling factor for dt that pushes at
                least one joint to its limit, taking into account acceleration and jerk limits.
            skip_last_tstep: flag to skip last time step in trajectory.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: success label tensor of shape (batch) and
                total cost tensor of shape (batch) where total cost = (path length +
                cost_weight * smoothness cost).
        """
        js = js.calculate_fd_from_position(traj_dt)
        if skip_last_tstep:
            js.position = js.position[..., :-1, :]
            js.velocity = js.velocity[..., :-1, :]
            js.acceleration = js.acceleration[..., :-1, :]
            js.jerk = js.jerk[..., :-1, :]

        return self.evaluate(js, traj_dt, cspace_distance_weight, max_vel)
