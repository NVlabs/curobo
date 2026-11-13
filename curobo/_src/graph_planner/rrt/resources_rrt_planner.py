# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Own the robot, collision, sampling, and interpolation resources used by RRT."""

from __future__ import annotations

# Standard Library
from typing import List, Optional

# Third Party
import torch

# CuRobo
from curobo._src.geom.collision.collision_scene import SceneCollision
from curobo._src.graph_planner.config_graph_planner_resources import GraphPlannerResourcesCfg
from curobo._src.graph_planner.graph.connector_linear import LinearConnector
from curobo._src.graph_planner.rrt.rrt_sampling import UniformRRTSampler
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.rollout.rollout_robot import RobotRollout
from curobo._src.trajectory.trajectory_interpolation import TrajInterpolationType, linear_smooth
from curobo._src.util.cuda_graph_util import GraphExecutor, create_graph_executor
from curobo._src.util.logging import log_and_raise


class RRTPlannerResources:
    """Own the runtime resources composed into the RRT planner."""

    def __init__(
        self,
        config: GraphPlannerResourcesCfg,
        scene_collision_checker: Optional[SceneCollision] = None,
    ) -> None:
        self.config = config
        self.device_cfg = config.device_cfg
        self.scene_collision_checker = scene_collision_checker
        if self.scene_collision_checker is None and config.scene_collision_cfg is not None:
            from curobo._src.geom.collision.collision_scene import create_scene_collision

            self.scene_collision_checker = create_scene_collision(config.scene_collision_cfg)

        self.feasibility_rollout = RobotRollout(
            config.rollout_config,
            self.scene_collision_checker,
            use_cuda_graph=False,
        )
        self.auxiliary_rollout = RobotRollout(
            config.rollout_config,
            self.scene_collision_checker,
        )
        self._max_act_buffer = torch.zeros(
            (config.feasibility_buffer_size, 1, self.action_dim),
            device=self.device_cfg.device,
            dtype=self.device_cfg.dtype,
        )
        self._max_env_query_buffer = torch.zeros(
            (config.feasibility_buffer_size,),
            device=self.device_cfg.device,
            dtype=torch.int32,
        )
        self._max_goal_query_buffer = torch.zeros(
            (config.feasibility_buffer_size,),
            device=self.device_cfg.device,
            dtype=torch.int32,
        )
        self._sample_feasibility_executor: GraphExecutor = create_graph_executor(
            capture_fn=self._compute_sample_feasibility,
            device=self.device_cfg.device,
            use_cuda_graph=config.use_cuda_graph_for_rollout,
            clone_outputs=False,
        )
        self._sample_feasibility_env_executor: Optional[GraphExecutor] = None
        self._sample_feasibility_goal_executor: Optional[GraphExecutor] = None
        self._sample_feasibility_env_goal_executor: Optional[GraphExecutor] = None

        self._cspace_distance_weight = torch.ones(
            self.action_dim,
            device=self.device_cfg.device,
            dtype=self.device_cfg.dtype,
        )
        self.linear_connector = LinearConnector(config, self.device_cfg)
        self.linear_connector.set_dependencies(
            action_dim=self.action_dim,
            cspace_distance_weight=self.cspace_distance_weight,
            check_feasibility_fn=self._check_samples_feasibility_ephemeral,
        )
        self.uniform_sampler = UniformRRTSampler(
            self.action_bound_lows,
            self.action_bound_highs,
            config.sample_rejection_ratio,
            config.sampler_seed,
            self.device_cfg,
            self.check_samples_feasibility,
        )

    def _compute_sample_feasibility(self, action_samples: torch.Tensor) -> torch.Tensor:
        metrics = self.feasibility_rollout.compute_metrics_from_action(action_samples)
        return metrics.costs_and_constraints.get_feasible(
            include_all_hybrid=False,
            sum_horizon=True,
        )

    def _compute_sample_feasibility_with_env(
        self,
        action_samples: torch.Tensor,
        env_query_idx: torch.Tensor,
    ) -> torch.Tensor:
        metrics = self.feasibility_rollout.compute_metrics_from_action(
            action_samples,
            idxs_env_query=env_query_idx,
        )
        return metrics.costs_and_constraints.get_feasible(
            include_all_hybrid=False,
            sum_horizon=True,
        )

    def _compute_sample_feasibility_with_goal(
        self,
        action_samples: torch.Tensor,
        goal_query_idx: torch.Tensor,
    ) -> torch.Tensor:
        self.feasibility_rollout.route_goal_poses(goal_query_idx.view(-1, 1))
        return self._compute_sample_feasibility(action_samples)

    def _compute_sample_feasibility_with_env_and_goal(
        self,
        action_samples: torch.Tensor,
        env_query_idx: torch.Tensor,
        goal_query_idx: torch.Tensor,
    ) -> torch.Tensor:
        self.feasibility_rollout.route_goal_poses(goal_query_idx.view(-1, 1))
        return self._compute_sample_feasibility_with_env(action_samples, env_query_idx)

    def _check_samples_feasibility_ephemeral(
        self,
        action_samples: torch.Tensor,
        env_query_idx: Optional[torch.Tensor] = None,
        goal_query_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.check_samples_feasibility(
            action_samples,
            clone_output=False,
            env_query_idx=env_query_idx,
            goal_query_idx=goal_query_idx,
        )

    def check_samples_feasibility(
        self,
        action_samples: torch.Tensor,
        clone_output: bool = True,
        env_query_idx: Optional[torch.Tensor] = None,
        goal_query_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate exact feasibility with independent environment and goal routing."""
        if action_samples.ndim != 2:
            log_and_raise("action_samples must be a 2D tensor (batch_size, action_dim)")
        if env_query_idx is not None:
            if env_query_idx.shape != (action_samples.shape[0],):
                log_and_raise("env_query_idx must have shape (batch_size,)")
            if env_query_idx.dtype != torch.int32:
                log_and_raise("env_query_idx must use torch.int32")
            if env_query_idx.device != action_samples.device:
                log_and_raise("env_query_idx must share action_samples.device")
        if goal_query_idx is not None:
            if goal_query_idx.shape != (action_samples.shape[0],):
                log_and_raise("goal_query_idx must have shape (batch_size,)")
            if goal_query_idx.dtype != torch.int32:
                log_and_raise("goal_query_idx must use torch.int32")
            if goal_query_idx.device != action_samples.device:
                log_and_raise("goal_query_idx must share action_samples.device")

        action_samples = action_samples.unsqueeze(1)
        buffer_size = self.config.feasibility_buffer_size
        if action_samples.shape[0] == 0:
            return torch.empty(0, dtype=torch.bool, device=action_samples.device)

        if env_query_idx is None and goal_query_idx is None:
            executor = self._sample_feasibility_executor
            if not executor.is_initialized:
                executor.warmup(self._max_act_buffer)
            graph_input = executor.static_inputs[0]
            env_graph_input = None
            goal_graph_input = None
        elif env_query_idx is not None and goal_query_idx is None:
            if self._sample_feasibility_env_executor is None:
                self._sample_feasibility_env_executor = create_graph_executor(
                    capture_fn=self._compute_sample_feasibility_with_env,
                    device=self.device_cfg.device,
                    use_cuda_graph=self.config.use_cuda_graph_for_rollout,
                    clone_outputs=False,
                )
            executor = self._sample_feasibility_env_executor
            if not executor.is_initialized:
                executor.warmup(self._max_act_buffer, self._max_env_query_buffer)
            graph_input = executor.static_inputs[0]
            env_graph_input = executor.static_inputs[1]
            goal_graph_input = None
        elif env_query_idx is None:
            if self._sample_feasibility_goal_executor is None:
                self._sample_feasibility_goal_executor = create_graph_executor(
                    capture_fn=self._compute_sample_feasibility_with_goal,
                    device=self.device_cfg.device,
                    use_cuda_graph=self.config.use_cuda_graph_for_rollout,
                    clone_outputs=False,
                )
            executor = self._sample_feasibility_goal_executor
            if not executor.is_initialized:
                executor.warmup(self._max_act_buffer, self._max_goal_query_buffer)
            graph_input = executor.static_inputs[0]
            env_graph_input = None
            goal_graph_input = executor.static_inputs[1]
        else:
            if self._sample_feasibility_env_goal_executor is None:
                self._sample_feasibility_env_goal_executor = create_graph_executor(
                    capture_fn=self._compute_sample_feasibility_with_env_and_goal,
                    device=self.device_cfg.device,
                    use_cuda_graph=self.config.use_cuda_graph_for_rollout,
                    clone_outputs=False,
                )
            executor = self._sample_feasibility_env_goal_executor
            if not executor.is_initialized:
                executor.warmup(
                    self._max_act_buffer,
                    self._max_env_query_buffer,
                    self._max_goal_query_buffer,
                )
            graph_input = executor.static_inputs[0]
            env_graph_input = executor.static_inputs[1]
            goal_graph_input = executor.static_inputs[2]

        if action_samples.shape[0] <= buffer_size:
            graph_input[: action_samples.shape[0]].copy_(action_samples)
            if env_graph_input is not None:
                env_graph_input[: action_samples.shape[0]].copy_(env_query_idx)
            if goal_graph_input is not None:
                goal_graph_input[: action_samples.shape[0]].copy_(goal_query_idx)
            feasible = executor.replay()[: action_samples.shape[0]]
            return feasible.clone().squeeze() if clone_output else feasible.squeeze()

        feasible_chunks = []
        for start in range(0, action_samples.shape[0], buffer_size):
            end = min(start + buffer_size, action_samples.shape[0])
            chunk_size = end - start
            graph_input[:chunk_size].copy_(action_samples[start:end])
            if env_graph_input is not None:
                env_graph_input[:chunk_size].copy_(env_query_idx[start:end])
            if goal_graph_input is not None:
                goal_graph_input[:chunk_size].copy_(goal_query_idx[start:end])
            feasible_chunks.append(executor.replay()[:chunk_size].clone())
        return torch.cat(feasible_chunks).squeeze()

    def get_interpolated_trajectory(
        self,
        paths: List[torch.Tensor],
        success: torch.Tensor,
        interpolation_steps: int,
        interpolation_type: TrajInterpolationType,
    ) -> torch.Tensor:
        """Provide the generic cubic/quintic interpolation fallback."""
        supported_types = {
            TrajInterpolationType.LINEAR,
            TrajInterpolationType.CUBIC,
            TrajInterpolationType.QUINTIC,
        }
        if interpolation_type not in supported_types:
            log_and_raise(f"Unsupported interpolation type: {interpolation_type}")
        output = torch.zeros(
            (len(paths), interpolation_steps, self.action_dim),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        for batch_index in torch.where(success)[0].cpu().tolist():
            path = paths[batch_index].cpu().view(-1, self.action_dim).numpy()
            if path.shape[0] == 1:
                log_and_raise("RRT interpolation requires at least two waypoints")
            for action_index in range(path.shape[-1]):
                output[batch_index, :, action_index] = linear_smooth(
                    path[:, action_index],
                    y=None,
                    n=interpolation_steps,
                    kind=interpolation_type,
                    last_step=interpolation_steps,
                )
        return output.to(device=self.device_cfg.device, dtype=self.device_cfg.dtype)

    def reset_seed(self) -> None:
        """Reset only RRT-owned sampling state."""
        self.uniform_sampler.reset_seed()

    def reset_cuda_graph(self) -> None:
        """Release feasibility and auxiliary-rollout graph state."""
        self._sample_feasibility_executor.reset()
        if self._sample_feasibility_env_executor is not None:
            self._sample_feasibility_env_executor.reset()
            self._sample_feasibility_env_executor = None
        if self._sample_feasibility_goal_executor is not None:
            self._sample_feasibility_goal_executor.reset()
            self._sample_feasibility_goal_executor = None
        if self._sample_feasibility_env_goal_executor is not None:
            self._sample_feasibility_env_goal_executor.reset()
            self._sample_feasibility_env_goal_executor = None
        if hasattr(self.feasibility_rollout, "reset_cuda_graph"):
            self.feasibility_rollout.reset_cuda_graph()
        if hasattr(self.auxiliary_rollout, "reset_cuda_graph"):
            self.auxiliary_rollout.reset_cuda_graph()

    def get_all_rollout_instances(self) -> List[RobotRollout]:
        return [self.feasibility_rollout, self.auxiliary_rollout]

    @property
    def action_dim(self) -> int:
        return self.auxiliary_rollout.action_dim

    @property
    def kinematics(self) -> Kinematics:
        return self.auxiliary_rollout.transition_model.robot_model

    @property
    def joint_names(self) -> List[str]:
        return self.kinematics.joint_names

    @property
    def action_bound_lows(self) -> torch.Tensor:
        return self.auxiliary_rollout.action_bound_lows

    @property
    def action_bound_highs(self) -> torch.Tensor:
        return self.auxiliary_rollout.action_bound_highs

    @property
    def cspace_distance_weight(self) -> torch.Tensor:
        return self._cspace_distance_weight
