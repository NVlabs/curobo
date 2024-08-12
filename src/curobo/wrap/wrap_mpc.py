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
import time
from typing import Optional

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo.rollout.rollout_base import Goal
from curobo.types.robot import State
from curobo.wrap.wrap_base import WrapBase, WrapConfig, WrapResult


class WrapMpc(WrapBase):
    def __init__(self, config: Optional[WrapConfig] = None):
        self._init_act_seq = None
        super().__init__(config)

    def update_init_seed(self, seed) -> bool:
        if self._init_act_seq is None:
            self._init_act_seq = seed.detach.clone()
        else:
            self._init_act_seq.copy_(seed)
        return True

    def solve(self, goal: Goal, seed: Optional[State] = None, shift_steps=1):
        if seed is None and self._init_act_seq is None:
            seed = self.get_init_act()
        elif self._init_act_seq is not None:
            seed = self._init_act_seq
        else:
            seed = seed.detach().clone()
        metrics = None

        start_time = time.time()
        filtered_state = self.safety_rollout.filter_robot_state(goal.current_state)
        goal.current_state.copy_(filtered_state)

        self.update_params(goal)
        if self.sync_cuda_time:
            torch.cuda.synchronize(device=self.tensor_args.device)
        # print("In: ", seed[0,:,0])
        start_time = time.time()
        with profiler.record_function("mpc/opt"):
            act_seq = self.optimize(seed, shift_steps=shift_steps)
        if self.sync_cuda_time:
            torch.cuda.synchronize(device=self.tensor_args.device)
        self.opt_dt = time.time() - start_time
        with profiler.record_function("mpc/filter"):
            act = self.safety_rollout.get_robot_command(
                filtered_state, act_seq, shift_steps=shift_steps
            )
        # print("Out: ", act_seq[0,:,0])
        self._init_act_seq = self._shift(act_seq, shift_steps=shift_steps)
        if self.compute_metrics:
            with profiler.record_function("mpc/get_metrics"):
                metrics = self.get_metrics(act)
        result = WrapResult(action=act, metrics=metrics, solve_time=self.opt_dt)
        return result

    def _shift(self, act_seq, shift_steps=1):
        act_seq = act_seq.roll(-shift_steps, 1)
        act_seq[:, -shift_steps:, :] = act_seq[:, -shift_steps - 1 : -shift_steps, :].clone()
        return act_seq

    def reset(self):
        self._init_act_seq = None
        return super().reset()

    def get_rollouts(self):
        return self.particle_optimizer.top_trajs
