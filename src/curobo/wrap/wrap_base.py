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
from dataclasses import dataclass
from typing import Any, List, Optional

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo.opt.newton.newton_base import NewtonOptBase
from curobo.opt.opt_base import Optimizer
from curobo.opt.particle.particle_opt_base import ParticleOptBase
from curobo.rollout.rollout_base import Goal, RolloutBase, RolloutMetrics
from curobo.types.robot import State
from curobo.util.logger import log_info, log_warn


@dataclass
class WrapConfig:
    safety_rollout: RolloutBase
    optimizers: List[Optimizer]
    compute_metrics: bool
    use_cuda_graph_metrics: bool
    sync_cuda_time: bool

    def __post_init__(self):
        if self.use_cuda_graph_metrics:
            log_info(
                "Using cuda graph for metrics is experimental. If you encounter CUDA errors,\
                     please disable it."
            )


@dataclass
class WrapResult:
    action: State
    solve_time: float
    metrics: Optional[RolloutMetrics] = None
    debug: Any = None
    js_action: Optional[State] = None
    raw_action: Optional[torch.Tensor] = None

    def clone(self):
        return WrapResult(
            self.action.clone(), self.solve_time, self.metrics.clone(), debug=self.debug
        )


class WrapBase(WrapConfig):
    def __init__(self, config: Optional[WrapConfig] = None):
        if config is not None:
            WrapConfig.__init__(self, **vars(config))
        self.n_problems = 1
        self.opt_dt = 0
        self._rollout_list = None
        self._opt_rollouts = None
        self._init_solver = False

    def get_metrics(self, state: State, use_cuda_graph: bool = False) -> RolloutMetrics:
        if use_cuda_graph:
            return self.safety_rollout.get_metrics_cuda_graph(state)
        return self.safety_rollout.get_metrics(state)

    def optimize(self, act_seq: torch.Tensor, shift_steps: int = 0) -> torch.Tensor:
        for opt in self.optimizers:
            act_seq = opt.optimize(act_seq, shift_steps)
        return act_seq

    def get_debug_data(self):
        debug_list = []
        for opt in self.optimizers:
            debug_list.append(opt.debug)
        return debug_list

    def get_debug_cost(self):
        debug_list = []
        for opt in self.optimizers:
            debug_list.append(opt.debug_cost)
        return debug_list

    def update_nproblems(self, n_problems):
        if n_problems != self.n_problems:
            self.n_problems = n_problems
            for opt in self.optimizers:
                opt.update_nproblems(self.n_problems)

    def update_params(self, goal: Goal):
        with profiler.record_function("wrap_base/safety/update_params"):
            log_info("Updating safety params")
            self.safety_rollout.update_params(goal)
        log_info("Updating optimizer params")
        for opt in self.optimizers:
            opt.update_params(goal)

    def get_init_act(self):
        init_act = self.safety_rollout.get_init_action_seq()
        return init_act

    def reset(self):
        for opt in self.optimizers:
            opt.reset()
        self.safety_rollout.reset()

    def reset_seed(self):
        self.safety_rollout.reset_seed()
        for opt in self.optimizers:
            opt.reset_seed()

    def reset_cuda_graph(self):
        self.safety_rollout.reset_cuda_graph()
        for opt in self.optimizers:
            opt.reset_cuda_graph()
        self._init_solver = False

    def reset_shape(self):
        self.safety_rollout.reset_shape()
        for opt in self.optimizers:
            opt.reset_shape()
        self._init_solver = False

    @property
    def rollout_fn(self):
        return self.safety_rollout

    @property
    def tensor_args(self):
        return self.safety_rollout.tensor_args

    def solve(self, goal: Goal, seed: Optional[torch.Tensor] = None):
        metrics = None

        filtered_state = self.safety_rollout.filter_robot_state(goal.current_state)
        goal.current_state.copy_(filtered_state)
        self.update_params(goal)
        if seed is None:
            seed = self.get_init_act()
            log_info("getting random seed")
        else:
            seed = seed.detach().clone()
        start_time = time.time()
        if not self._init_solver:
            log_info("Solver was not initialized, warming up solver")
            for _ in range(2):
                act_seq = self.optimize(seed, shift_steps=0)
            self._init_solver = True
        act_seq = self.optimize(seed, shift_steps=0)
        self.opt_dt = time.time() - start_time

        act = self.safety_rollout.get_robot_command(
            filtered_state, act_seq, state_idx=goal.batch_current_state_idx
        )

        if self.compute_metrics:
            with profiler.record_function("wrap_base/compute_metrics"):
                metrics = self.get_metrics(
                    act, self.use_cuda_graph_metrics
                )  # TODO: use cuda graph for metrics

        result = WrapResult(
            action=act,
            solve_time=self.opt_dt,
            metrics=metrics,
            debug={"steps": self.get_debug_data(), "cost": self.get_debug_cost()},
            raw_action=act_seq,
        )
        return result

    @property
    def newton_optimizer(self) -> NewtonOptBase:
        return self.optimizers[-1]

    @property
    def particle_optimizer(self) -> ParticleOptBase:
        return self.optimizers[0]

    @property
    def joint_names(self):
        return self.rollout_fn.cspace_config.joint_names

    def _get_rollout_instances_from_optimizers(self) -> List[RolloutBase]:
        if self._opt_rollouts is None:
            self._opt_rollouts = []
            for i in self.optimizers:
                self._opt_rollouts.extend(i.get_all_rollout_instances())
        return self._opt_rollouts

    def get_all_rollout_instances(self) -> List[RolloutBase]:
        if self._rollout_list is None:
            self._rollout_list = [
                self.safety_rollout
            ] + self._get_rollout_instances_from_optimizers()
        return self._rollout_list
