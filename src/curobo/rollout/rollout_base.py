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

# import torch
from __future__ import annotations

# Standard Library
from abc import abstractmethod, abstractproperty
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# Third Party
import torch

# CuRobo
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import CSpaceConfig, State
from curobo.types.tensor import (
    T_BDOF,
    T_DOF,
    T_BHDOF_float,
    T_BHValue_float,
    T_BValue_bool,
    T_BValue_float,
)
from curobo.util.helpers import list_idx_if_not_none
from curobo.util.logger import log_info
from curobo.util.sample_lib import HaltonGenerator
from curobo.util.tensor_util import copy_tensor
from curobo.util.torch_utils import get_torch_jit_decorator


@dataclass
class RolloutMetrics(Sequence):
    cost: Optional[T_BValue_float] = None
    constraint: Optional[T_BValue_float] = None
    feasible: Optional[T_BValue_bool] = None
    state: Optional[State] = None

    def __getitem__(self, idx):
        d_list = [self.cost, self.constraint, self.feasible, self.state]
        idx_vals = list_idx_if_not_none(d_list, idx)
        return RolloutMetrics(idx_vals[0], idx_vals[1], idx_vals[2], idx_vals[3])

    def __len__(self):
        if self.cost is not None:
            return self.cost.shape[0]
        else:
            return -1

    def clone(self, clone_state=False):
        if clone_state:
            raise NotImplementedError()
        return RolloutMetrics(
            cost=None if self.cost is None else self.cost.clone(),
            constraint=None if self.constraint is None else self.constraint.clone(),
            feasible=None if self.feasible is None else self.feasible.clone(),
            state=None if self.state is None else self.state,
        )


@dataclass
class Trajectory:
    actions: T_BHDOF_float
    costs: T_BHValue_float
    state: Optional[State] = None
    debug: Optional[dict] = None


@dataclass
class Goal(Sequence):
    """Goal data class used to update optimization target.

    #NOTE:
    We can parallelize Goal in two ways:
    1. Solve for current_state, pose pair in same environment
    2. Solve for current_state, pose pair in different environment
    For case (1), we use batch_pose_idx to find the memory address of the
    current_state, pose pair while keeping batch_world_idx = [0]
    For case (2), we add a batch_world_idx[0,1,2..].
    """

    name: str = "goal"
    goal_state: Optional[State] = None
    goal_pose: Pose = field(default_factory=Pose)
    links_goal_pose: Optional[Dict[str, Pose]] = None
    current_state: Optional[State] = None
    retract_state: Optional[T_DOF] = None
    batch: int = -1  # NOTE: add another variable for size of index tensors?
    # this should also contain a batch index tensor:
    batch_pose_idx: Optional[torch.Tensor] = None  # shape: [batch]
    batch_goal_state_idx: Optional[torch.Tensor] = None
    batch_retract_state_idx: Optional[torch.Tensor] = None
    batch_current_state_idx: Optional[torch.Tensor] = None  # shape: [batch]
    batch_enable_idx: Optional[torch.Tensor] = None  # shape: [batch, n]
    batch_world_idx: Optional[torch.Tensor] = None  # shape: [batch, n]
    update_batch_idx_buffers: bool = True
    n_goalset: int = 1  # NOTE: This currently does not get updated if goal_pose is updated later.

    def __getitem__(self, idx):
        d_list = [
            self.goal_state,
            self.goal_pose,
            self.current_state,
            self.retract_state,
            self.batch_pose_idx,
            self.batch_goal_state_idx,
            self.batch_retract_state_idx,
            self.batch_current_state_idx,
            self.batch_enable_idx,
            self.batch_world_idx,
        ]
        idx_vals = list_idx_if_not_none(d_list, idx)
        return Goal(
            name=self.name,
            batch=self.batch,
            n_goalset=self.n_goalset,
            goal_state=idx_vals[0],
            goal_pose=idx_vals[1],
            current_state=idx_vals[2],
            retract_state=idx_vals[3],
            batch_pose_idx=idx_vals[4],
            batch_goal_state_idx=idx_vals[5],
            batch_retract_state_idx=idx_vals[6],
            batch_current_state_idx=idx_vals[7],
            batch_enable_idx=idx_vals[8],
            batch_world_idx=idx_vals[9],
        )

    def __len__(self):
        return self.batch

    def __post_init__(self):
        self._update_batch_size()
        if self.goal_pose.position is not None:
            if self.batch_pose_idx is None:
                self.batch_pose_idx = torch.arange(
                    0, self.batch, 1, device=self.goal_pose.position.device, dtype=torch.int32
                ).unsqueeze(-1)
            self.n_goalset = self.goal_pose.n_goalset
        if self.current_state is not None:
            if self.batch_current_state_idx is None:
                self.batch_current_state_idx = torch.arange(
                    0,
                    self.current_state.position.shape[0],
                    1,
                    device=self.current_state.position.device,
                    dtype=torch.int32,
                ).unsqueeze(-1)
        if self.retract_state is not None:
            if self.batch_retract_state_idx is None:
                self.batch_retract_state_idx = torch.arange(
                    0,
                    self.retract_state.shape[0],
                    1,
                    device=self.retract_state.device,
                    dtype=torch.int32,
                ).unsqueeze(-1)

    def _update_batch_size(self):
        if self.goal_pose.position is not None:
            self.batch = self.goal_pose.batch
        elif self.goal_state is not None:
            self.batch = self.goal_state.position.shape[0]
        elif self.current_state is not None:
            self.batch = self.current_state.position.shape[0]

    def repeat_seeds(self, num_seeds: int):
        # across seeds, the data is the same, so could we just expand batch_idx
        goal_pose = goal_state = current_state = links_goal_pose = retract_state = None
        batch_enable_idx = batch_pose_idx = batch_world_idx = batch_current_state_idx = None
        batch_retract_state_idx = batch_goal_state_idx = None

        if self.links_goal_pose is not None:
            links_goal_pose = self.links_goal_pose
        if self.goal_pose is not None:
            goal_pose = self.goal_pose
        #    goal_pose = self.goal_pose.repeat_seeds(num_seeds)
        if self.goal_state is not None:
            goal_state = self.goal_state  # .repeat_seeds(num_seeds)
        if self.current_state is not None:
            current_state = self.current_state  # .repeat_seeds(num_seeds)
        if self.retract_state is not None:
            retract_state = self.retract_state
        # repeat seeds for indexing:
        if self.batch_pose_idx is not None:
            batch_pose_idx = self._tensor_repeat_seeds(self.batch_pose_idx, num_seeds)
        if self.batch_goal_state_idx is not None:
            batch_goal_state_idx = self._tensor_repeat_seeds(self.batch_goal_state_idx, num_seeds)
        if self.batch_retract_state_idx is not None:
            batch_retract_state_idx = self._tensor_repeat_seeds(
                self.batch_retract_state_idx, num_seeds
            )
        if self.batch_enable_idx is not None:
            batch_enable_idx = self._tensor_repeat_seeds(self.batch_enable_idx, num_seeds)
        if self.batch_world_idx is not None:
            batch_world_idx = self._tensor_repeat_seeds(self.batch_world_idx, num_seeds)
        if self.batch_current_state_idx is not None:
            batch_current_state_idx = self._tensor_repeat_seeds(
                self.batch_current_state_idx, num_seeds
            )

        return Goal(
            goal_state=goal_state,
            goal_pose=goal_pose,
            current_state=current_state,
            retract_state=retract_state,
            batch_pose_idx=batch_pose_idx,
            batch_world_idx=batch_world_idx,
            batch_enable_idx=batch_enable_idx,
            batch_current_state_idx=batch_current_state_idx,
            batch_retract_state_idx=batch_retract_state_idx,
            batch_goal_state_idx=batch_goal_state_idx,
            links_goal_pose=links_goal_pose,
        )

    def clone(self):
        return Goal(
            goal_state=self.goal_state,
            goal_pose=self.goal_pose,
            current_state=self.current_state,
            retract_state=self.retract_state,
            batch_pose_idx=self.batch_pose_idx,
            batch_world_idx=self.batch_world_idx,
            batch_enable_idx=self.batch_enable_idx,
            batch_current_state_idx=self.batch_current_state_idx,
            batch_retract_state_idx=self.batch_retract_state_idx,
            batch_goal_state_idx=self.batch_goal_state_idx,
            links_goal_pose=self.links_goal_pose,
            n_goalset=self.n_goalset,
        )

    def _tensor_repeat_seeds(self, tensor, num_seeds):
        return tensor_repeat_seeds(tensor, num_seeds)

    def apply_kernel(self, kernel_mat):
        # For each seed in optimization, we use kernel_mat to transform to many parallel goals
        # This can be modified to just multiply self.batch and update self.batch by the shape of self.batch
        # TODO: add other elements
        goal_pose = goal_state = current_state = links_goal_pose = None
        batch_enable_idx = batch_pose_idx = batch_world_idx = batch_current_state_idx = None
        batch_retract_state_idx = batch_goal_state_idx = None
        if self.links_goal_pose is not None:
            links_goal_pose = self.links_goal_pose
        if self.goal_pose is not None:
            goal_pose = self.goal_pose  # .apply_kernel(kernel_mat)
        if self.goal_state is not None:
            goal_state = self.goal_state  # .apply_kernel(kernel_mat)
        if self.current_state is not None:
            current_state = self.current_state  # .apply_kernel(kernel_mat)
        if self.batch_enable_idx is not None:
            batch_enable_idx = kernel_mat @ self.batch_enable_idx
        if self.batch_retract_state_idx is not None:
            batch_retract_state_idx = (
                kernel_mat @ self.batch_retract_state_idx.to(dtype=torch.float32)
            ).to(dtype=torch.int32)
        if self.batch_goal_state_idx is not None:
            batch_goal_state_idx = (
                kernel_mat @ self.batch_goal_state_idx.to(dtype=torch.float32)
            ).to(dtype=torch.int32)

        if self.batch_current_state_idx is not None:
            batch_current_state_idx = (
                kernel_mat @ self.batch_current_state_idx.to(dtype=torch.float32)
            ).to(dtype=torch.int32)
        if self.batch_pose_idx is not None:
            batch_pose_idx = (kernel_mat @ self.batch_pose_idx.to(dtype=torch.float32)).to(
                dtype=torch.int32
            )
        if self.batch_world_idx is not None:
            batch_world_idx = (kernel_mat @ self.batch_world_idx.to(dtype=torch.float32)).to(
                dtype=torch.int32
            )

        return Goal(
            goal_state=goal_state,
            goal_pose=goal_pose,
            current_state=current_state,
            batch_pose_idx=batch_pose_idx,
            batch_enable_idx=batch_enable_idx,
            batch_world_idx=batch_world_idx,
            batch_current_state_idx=batch_current_state_idx,
            batch_goal_state_idx=batch_goal_state_idx,
            batch_retract_state_idx=batch_retract_state_idx,
            links_goal_pose=links_goal_pose,
        )

    def to(self, tensor_args: TensorDeviceType):
        if self.goal_pose is not None:
            self.goal_pose = self.goal_pose.to(tensor_args)
        if self.goal_state is not None:
            self.goal_state = self.goal_state.to(**(tensor_args.as_torch_dict()))
        if self.current_state is not None:
            self.current_state = self.current_state.to(**(tensor_args.as_torch_dict()))
        return self

    def copy_(self, goal: Goal, update_idx_buffers: bool = True):
        """Copy data from another goal object.

        Args:
            goal (Goal): _description_

        Raises:
            NotImplementedError: _description_
            NotImplementedError: _description_

        Returns:
            _type_: _description_
        """

        self.goal_pose = self._copy_buffer(self.goal_pose, goal.goal_pose)
        self.goal_state = self._copy_buffer(self.goal_state, goal.goal_state)
        self.retract_state = self._copy_tensor(self.retract_state, goal.retract_state)
        self.current_state = self._copy_buffer(self.current_state, goal.current_state)
        if goal.links_goal_pose is not None:
            if self.links_goal_pose is None:
                self.links_goal_pose = goal.links_goal_pose
            else:
                for k in goal.links_goal_pose.keys():
                    self.links_goal_pose[k] = self._copy_buffer(
                        self.links_goal_pose[k], goal.links_goal_pose[k]
                    )
        self._update_batch_size()
        # copy pose indices as well?
        if goal.update_batch_idx_buffers and update_idx_buffers:
            self.batch_pose_idx = self._copy_tensor(self.batch_pose_idx, goal.batch_pose_idx)
            self.batch_enable_idx = self._copy_tensor(self.batch_enable_idx, goal.batch_enable_idx)
            self.batch_world_idx = self._copy_tensor(self.batch_world_idx, goal.batch_world_idx)
            self.batch_current_state_idx = self._copy_tensor(
                self.batch_current_state_idx, goal.batch_current_state_idx
            )
            self.batch_retract_state_idx = self._copy_tensor(
                self.batch_retract_state_idx, goal.batch_retract_state_idx
            )
            self.batch_goal_state_idx = self._copy_tensor(
                self.batch_goal_state_idx, goal.batch_goal_state_idx
            )

    def _copy_buffer(self, ref_buffer, buffer):
        if buffer is not None:
            if ref_buffer is not None:
                ref_buffer = ref_buffer.copy_(buffer)
            else:
                log_info("breaking reference")
                ref_buffer = buffer.clone()
        return ref_buffer

    def _copy_tensor(self, ref_buffer, buffer):
        if buffer is not None:
            if ref_buffer is not None and buffer.shape == ref_buffer.shape:
                if not copy_tensor(buffer, ref_buffer):
                    ref_buffer = buffer.clone()
            else:
                ref_buffer = buffer.clone()
        return ref_buffer

    def get_batch_goal_state(self):
        return self.goal_state[self.batch_pose_idx[:, 0]]

    def create_index_buffers(
        self,
        batch_size: int,
        batch_env: bool,
        batch_retract: bool,
        num_seeds: int,
        tensor_args: TensorDeviceType,
    ):
        new_goal = Goal.create_idx(batch_size, batch_env, batch_retract, num_seeds, tensor_args)
        new_goal.copy_(self, update_idx_buffers=False)
        return new_goal

    @classmethod
    def create_idx(
        cls,
        pose_batch_size: int,
        batch_env: bool,
        batch_retract: bool,
        num_seeds: int,
        tensor_args: TensorDeviceType,
    ):
        batch_pose_idx = torch.arange(
            0, pose_batch_size, 1, device=tensor_args.device, dtype=torch.int32
        ).unsqueeze(-1)
        if batch_env:
            batch_world_idx = batch_pose_idx.clone()
        else:
            batch_world_idx = 0 * batch_pose_idx
        if batch_retract:
            batch_retract_state_idx = batch_pose_idx.clone()
        else:
            batch_retract_state_idx = 0 * batch_pose_idx.clone()
        batch_currernt_state_idx = batch_pose_idx.clone()
        batch_goal_state_idx = batch_pose_idx.clone()

        g = Goal(
            batch_pose_idx=batch_pose_idx,
            batch_retract_state_idx=batch_retract_state_idx,
            batch_world_idx=batch_world_idx,
            batch_current_state_idx=batch_currernt_state_idx,
            batch_goal_state_idx=batch_goal_state_idx,
        )
        g_seeds = g.repeat_seeds(num_seeds)
        return g_seeds


@dataclass
class RolloutConfig:
    tensor_args: TensorDeviceType
    sum_horizon: bool = False
    sampler_seed: int = 1312


class RolloutBase:
    def __init__(self, config: Optional[RolloutConfig] = None):
        self.start_state = None
        self.batch_size = 1
        self._cuda_graph_valid = False
        self._metrics_cuda_graph_init = False
        self.cu_metrics_graph = None
        self._rollout_constraint_cuda_graph_init = False
        self.cu_rollout_constraint_graph = None
        if config is not None:
            self.tensor_args = config.tensor_args

    def _init_after_config_load(self):
        self.act_sample_gen = HaltonGenerator(
            self.d_action,
            self.tensor_args,
            up_bounds=self.action_bound_highs,
            low_bounds=self.action_bound_lows,
            seed=self.sampler_seed,
        )

    @abstractmethod
    def cost_fn(self, state: State):
        return

    @abstractmethod
    def constraint_fn(
        self, state: State, out_metrics: Optional[RolloutMetrics] = None
    ) -> RolloutMetrics:
        return

    @abstractmethod
    def convergence_fn(
        self, state: State, out_metrics: Optional[RolloutMetrics] = None
    ) -> RolloutMetrics:
        return

    def get_metrics(self, state: State):
        out_metrics = self.constraint_fn(state)
        out_metrics = self.convergence_fn(state, out_metrics)
        return out_metrics

    def get_metrics_cuda_graph(self, state: State):
        return self.get_metrics(state)

    def rollout_fn(self, act):
        pass

    def current_cost(self, current_state):
        pass

    @abstractmethod
    def update_params(self, goal: Goal):
        return

    def __call__(self, act: T_BHDOF_float) -> Trajectory:
        return self.rollout_fn(act)

    @abstractproperty
    def action_bounds(self):
        return self.tensor_args.to_device(
            torch.stack([self.action_bound_lows, self.action_bound_highs])
        )

    @abstractmethod
    def filter_robot_state(self, current_state: State) -> State:
        return current_state

    @abstractmethod
    def get_robot_command(
        self, current_state, act_seq, shift_steps: int = 1, state_idx: Optional[torch.Tensor] = None
    ):
        return act_seq

    def reset_seed(self):
        self.act_sample_gen.reset()

    def reset(self):
        return True

    @abstractproperty
    def d_action(self) -> int:
        raise NotImplementedError

    @abstractproperty
    def action_bound_lows(self):
        return 1

    @abstractproperty
    def action_bound_highs(self):
        return 1

    @abstractproperty
    def dt(self):
        return 0.1

    @property
    def horizon(self) -> int:
        raise NotImplementedError

    @property
    def action_horizon(self) -> int:
        return self.horizon

    def update_start_state(self, start_state: torch.Tensor):
        if self.start_state is None:
            self.start_state = start_state
        copy_tensor(start_state, self.start_state)

    @abstractmethod
    def get_init_action_seq(self):
        raise NotImplementedError

    @property
    def state_bounds(self) -> Dict[str, List[float]]:
        pass

    # sample random actions
    # @abstractmethod
    def sample_random_actions(self, n: int = 0):
        act_rand = self.act_sample_gen.get_samples(n, bounded=True)
        return act_rand

    # how to map act_seq to state?
    # rollout for feasibility?
    @abstractmethod
    def rollout_constraint(self, act_seq: torch.Tensor) -> RolloutMetrics:
        # get state by rolling out

        # get feasibility:
        pass

    def reset_cuda_graph(self):
        self._cuda_graph_valid = False
        self._metrics_cuda_graph_init = False
        if self.cu_metrics_graph is not None:
            self.cu_metrics_graph.reset()

        self._rollout_constraint_cuda_graph_init = False
        if self.cu_rollout_constraint_graph is not None:
            self.cu_rollout_constraint_graph.reset()
        self.reset_shape()

    def reset_shape(self):
        pass

    @property
    def cuda_graph_instance(self):
        return self._cuda_graph_valid

    @abstractmethod
    def get_action_from_state(self, state: State):
        pass

    @abstractmethod
    def get_state_from_action(
        self, start_state: State, act_seq: torch.Tensor, state_idx: Optional[torch.Tensor] = None
    ):
        pass

    @abstractproperty
    def cspace_config(self) -> CSpaceConfig:
        pass

    def get_full_dof_from_solution(self, q_js: JointState) -> JointState:
        return q_js

    def break_cuda_graph(self):
        self._cuda_graph_valid = False


@get_torch_jit_decorator()
def tensor_repeat_seeds(tensor, num_seeds: int):
    a = (
        tensor.view(tensor.shape[0], 1, tensor.shape[-1])
        .repeat(1, num_seeds, 1)
        .view(tensor.shape[0] * num_seeds, tensor.shape[-1])
    )
    return a
