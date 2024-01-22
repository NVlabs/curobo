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
import time
from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo.rollout.rollout_base import Goal, RolloutBase
from curobo.types.base import TensorDeviceType
from curobo.util.logger import log_info
from curobo.util.torch_utils import is_cuda_graph_available


@dataclass
class OptimizerConfig:
    d_action: int
    action_lows: List[float]
    action_highs: List[float]
    horizon: int
    n_iters: int
    rollout_fn: RolloutBase
    tensor_args: TensorDeviceType
    use_cuda_graph: bool
    store_debug: bool
    debug_info: Any
    cold_start_n_iters: int
    num_particles: Union[int, None]
    n_envs: int
    sync_cuda_time: bool
    use_coo_sparse: bool
    action_horizon: int

    def __post_init__(self):
        object.__setattr__(self, "action_highs", self.tensor_args.to_device(self.action_highs))
        object.__setattr__(self, "action_lows", self.tensor_args.to_device(self.action_lows))
        # check cuda graph version:
        if self.use_cuda_graph:
            self.use_cuda_graph = is_cuda_graph_available()
        if self.num_particles is None:
            self.num_particles = 1

    @staticmethod
    def create_data_dict(
        data_dict: Dict,
        rollout_fn: RolloutBase,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        child_dict: Optional[Dict] = None,
    ):
        if child_dict is None:
            child_dict = deepcopy(data_dict)
        child_dict["d_action"] = rollout_fn.d_action
        child_dict["action_lows"] = rollout_fn.action_bound_lows
        child_dict["action_highs"] = rollout_fn.action_bound_highs
        child_dict["rollout_fn"] = rollout_fn
        child_dict["tensor_args"] = tensor_args
        child_dict["horizon"] = rollout_fn.horizon
        child_dict["action_horizon"] = rollout_fn.action_horizon

        if "num_particles" not in child_dict:
            child_dict["num_particles"] = 1
        return child_dict


class Optimizer(OptimizerConfig):
    def __init__(self, config: Optional[OptimizerConfig] = None) -> None:
        if config is not None:
            super().__init__(**vars(config))
        self.opt_dt = 0.0
        self.COLD_START = True
        self.update_nenvs(self.n_envs)
        self._batch_goal = None
        self._rollout_list = None
        self.debug = []
        self.debug_cost = []

    def optimize(self, opt_tensor: torch.Tensor, shift_steps=0, n_iters=None) -> torch.Tensor:
        if self.COLD_START:
            n_iters = self.cold_start_n_iters
            self.COLD_START = False
        st_time = time.time()
        out = self._optimize(opt_tensor, shift_steps, n_iters)
        if self.sync_cuda_time:
            torch.cuda.synchronize()
        self.opt_dt = time.time() - st_time
        return out

    @abstractmethod
    def _optimize(self, opt_tensor: torch.Tensor, shift_steps=0, n_iters=None) -> torch.Tensor:
        pass

    def _shift(self, shift_steps=0):
        """
        Shift the variables in the solver to hotstart the next timestep
        """
        return

    def update_params(self, goal: Goal):
        with profiler.record_function("OptBase/batch_goal"):
            if self._batch_goal is not None:
                self._batch_goal.copy_(goal, update_idx_buffers=True)  # why True?
            else:
                self._batch_goal = goal.repeat_seeds(self.num_particles)
        self.rollout_fn.update_params(self._batch_goal)

    def reset(self):
        """
        Reset the controller
        """
        self.rollout_fn.reset()
        self.debug = []
        self.debug_cost = []
        # self.COLD_START = True

    def update_nenvs(self, n_envs):
        assert n_envs > 0
        self._update_env_kernel(n_envs, self.num_particles)
        self.n_envs = n_envs

    def _update_env_kernel(self, n_envs, num_particles):
        log_info(
            "Updating env kernel [n_envs: "
            + str(n_envs)
            + " , num_particles: "
            + str(num_particles)
            + " ]"
        )

        self.env_col = torch.arange(
            0, n_envs, step=1, dtype=torch.long, device=self.tensor_args.device
        )
        self.n_select_ = torch.tensor(
            [x * n_envs + x for x in range(n_envs)],
            device=self.tensor_args.device,
            dtype=torch.long,
        )

        # create sparse tensor:
        sparse_indices = []
        for i in range(n_envs):
            sparse_indices.extend([[i * num_particles + x, i] for x in range(num_particles)])

        sparse_values = torch.ones(len(sparse_indices))
        sparse_indices = torch.tensor(sparse_indices)
        if self.use_coo_sparse:
            self.env_kernel_ = torch.sparse_coo_tensor(
                sparse_indices.t(),
                sparse_values,
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
        else:
            self.env_kernel_ = torch.sparse_coo_tensor(
                sparse_indices.t(),
                sparse_values,
                device="cpu",
                dtype=self.tensor_args.dtype,
            )
            self.env_kernel_ = self.env_kernel_.to_dense().to(device=self.tensor_args.device)
        self._env_seeds = self.num_particles

    def get_nenv_tensor(self, x):
        """This function takes an input tensor of shape (n_env,....) and converts it into
        (n_particles,...)
        """

        # if x.shape[0] != self.n_envs:
        #    x_env = x.unsqueeze(0).repeat(self.n_envs, 1)
        # else:
        #    x_env = x

        # create a tensor
        nx_env = self.env_kernel_ @ x

        return nx_env

    def reset_seed(self):
        return True

    def reset_cuda_graph(self):
        if self.use_cuda_graph:
            self.cu_opt_init = False
        else:
            log_info("Cuda Graph was not enabled")
        self._batch_goal = None
        self.rollout_fn.reset_cuda_graph()

    def get_all_rollout_instances(self) -> List[RolloutBase]:
        if self._rollout_list is None:
            self._rollout_list = [self.rollout_fn]
        return self._rollout_list
