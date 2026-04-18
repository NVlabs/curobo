# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#


from __future__ import annotations

# Standard Library
import math
import random
from typing import List, Optional

# Third Party
import numpy as np
import torch
import torch.autograd.profiler as profiler

from curobo._src.geom.collision import SceneCollision, create_collision_checker
from curobo._src.graph_planner.graph.connector_linear import LinearConnector
from curobo._src.graph_planner.graph.constructor import GraphConstructor
from curobo._src.graph_planner.graph.node_distance import DistanceNeighborCalculator
from curobo._src.graph_planner.graph.node_manager import GraphNodeManager
from curobo._src.graph_planner.graph.node_sampling_strategy import NodeSamplingStrategy
from curobo._src.graph_planner.graph_planner_prm_cfg import PRMGraphPlannerCfg
from curobo._src.graph_planner.result import GraphPlannerResult
from curobo._src.graph_planner.search.path_finder_networkx import NetworkXPathFinder
from curobo._src.graph_planner.search.path_pruner import PathPruner

# CuRobo
from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsState
from curobo._src.rollout.rollout_robot import RobotRollout
from curobo._src.state.state_joint import JointState
from curobo._src.transition.robot_state_transition import RobotStateTransition
from curobo._src.util.cuda_event_timer import CudaEventTimer
from curobo._src.util.logging import log_and_raise, log_warn
from curobo._src.util.trajectory import TrajInterpolationType, linear_smooth


class PRMGraphPlanner:
    @profiler.record_function("graph_plan_base/init")
    def __init__(
        self,
        config: PRMGraphPlannerCfg,
        scene_collision_checker: Optional[SceneCollision] = None,
    ):
        self.config = config
        self.device_cfg = config.device_cfg
        self.scene_collision_checker = scene_collision_checker
        self._initialize_components()

    def _initialize_components(self):
        self._out_traj_state = None

        # initialize world collision checker:
        if self.scene_collision_checker is None:
            if self.config.scene_collision_cfg is not None:
                self.scene_collision_checker = create_collision_checker(
                    self.config.scene_collision_cfg
                )

        # initialize rollout function:
        self.feasibility_rollout = RobotRollout(
            self.config.rollout_config,
            self.scene_collision_checker,
            use_cuda_graph=self.config.use_cuda_graph_for_rollout,
        )
        self.auxiliary_rollout = RobotRollout(
            self.config.rollout_config, self.scene_collision_checker
        )

        # initialize buffers
        self._max_act_buffer = torch.zeros(
            (self.config.feasibility_buffer_size, 1, self.action_dim),
            device=self.device_cfg.device,
            dtype=self.device_cfg.dtype,
        )

        self.graph_path_finder = NetworkXPathFinder(seed=self.config.graph_path_finder_seed)

        # Initialize the path pruner
        self.path_pruner = PathPruner(self.config, self.device_cfg)

        # Initialize the linear connector
        self.linear_connector = LinearConnector(self.config, self.device_cfg)

        self._cspace_distance_weight = torch.ones(self.action_dim, device=self.device_cfg.device,
        dtype=self.device_cfg.dtype)

        # Initialize the distance neighbor calculator
        self.distance_calculator = DistanceNeighborCalculator(
            action_dim=self.action_dim,
            cspace_distance_weight=self.cspace_distance_weight,
            device_cfg=self.device_cfg,
        )

        # Initialize the node manager
        self.node_manager = GraphNodeManager(
            config=self.config,
            device_cfg=self.device_cfg,
            distance_calculator=self.distance_calculator,
            graph_path_finder=self.graph_path_finder,
            auxiliary_rollout=self.auxiliary_rollout,
        )

        # Initialize the sampling strategy
        self.sampling_strategy = NodeSamplingStrategy(
            config=self.config,
            action_lower_bounds=self.action_bound_lows,
            action_upper_bounds=self.action_bound_highs,
            cspace_distance_weight=self.cspace_distance_weight,
            action_dim=self.action_dim,
            check_feasibility_fn=self.check_samples_feasibility,
            device_cfg=self.device_cfg,
        )

        # Initialize the graph constructor
        self.graph_constructor = GraphConstructor(
            config=self.config,
            device_cfg=self.device_cfg,
            linear_connector=self.linear_connector,
            distance_calculator=self.distance_calculator,
            node_manager=self.node_manager,
            action_dim=self.action_dim,
            check_feasibility_fn=self.check_samples_feasibility,
        )

        # Set up dependencies for the components
        self._setup_dependencies()

    def _setup_dependencies(self):
        """Set up dependencies for all components that need them."""
        # Set up the path pruner
        self.path_pruner.set_dependencies(
            action_dim=self.action_dim,
            cspace_distance_weight=self.cspace_distance_weight,
            preallocated_node_buffer=self.node_manager.preallocated_node_buffer,
            steer_and_register_edges_fn=self.graph_constructor.steer_and_register_edges,
            find_path_for_index_pairs_fn=self._find_path_for_index_pairs,
        )

        # Set up the linear connector
        self.linear_connector.set_dependencies(
            action_dim=self.action_dim,
            cspace_distance_weight=self.cspace_distance_weight,
            check_feasibility_fn=self.check_samples_feasibility,
        )

    def check_samples_feasibility(self, action_samples):  # call feasibility here:
        # call mask samples here with a for loop:
        if len(action_samples.shape) != 2:
            log_and_raise("action_samples must be a 2D tensor (batch_size, action_dim)")

        feasible_mask = []
        action_samples = action_samples.unsqueeze(1)
        if self.config.feasibility_buffer_size < action_samples.shape[0]:
            for i in range(
                math.ceil(action_samples.shape[0] / self.config.feasibility_buffer_size)
            ):
                start = i * self.config.feasibility_buffer_size
                end = (i + 1) * self.config.feasibility_buffer_size
                if end > action_samples.shape[0]:
                    end = action_samples.shape[0]
                current_action_samples = action_samples[start:end, :]

                end_idx = current_action_samples.shape[0]
                self._max_act_buffer[:end_idx, :, :] = current_action_samples
                metrics_result = self.feasibility_rollout.compute_metrics_from_action(
                    self._max_act_buffer,
                )
                feasible = metrics_result.costs_and_constraints.get_feasible(
                    include_all_hybrid=False, sum_horizon=True
                )[:end_idx]
                feasible_mask.append(feasible)
        else:
            start = 0
            end = action_samples.shape[0]
            self._max_act_buffer[:end, :, :] = action_samples
            metrics_result = self.feasibility_rollout.compute_metrics_from_action(
                self._max_act_buffer,
            )
            feasible = metrics_result.costs_and_constraints.get_feasible(
                include_all_hybrid=False, sum_horizon=True
            )[:end]
            feasible_mask.append(feasible)
        feasible_mask = torch.cat(feasible_mask).squeeze()
        return feasible_mask

    @profiler.record_function("base_graph_planner/extend_roadmap_with_random_samples")
    def extend_roadmap_with_random_samples(
        self,
        num_samples: int,
        neighbors_per_node: int = 10,
    ):
        # get samples to search in:
        v_set = self.sampling_strategy.generate_feasible_samples(num_samples=num_samples)
        self.node_manager.add_nodes_to_buffer(v_set)

        sample_nodes = v_set
        self.graph_constructor.connect_nodes(sample_nodes, neighbors_per_node=neighbors_per_node)

    @profiler.record_function("base_graph_planner/extend_roadmap_with_ellipsoidal_samples")
    def extend_roadmap_with_ellipsoidal_samples(
        self,
        x_start: torch.Tensor,
        x_goal: torch.Tensor,
        max_sampling_radius: torch.Tensor,
        num_samples: int,
        neighbors_per_node: int = 5,
    ):
        # sample some points for vertex
        v_set = self.sampling_strategy.generate_feasible_samples_in_ellipsoid(
            x_start=x_start,
            x_goal=x_goal,
            num_samples=num_samples,
            max_sampling_radius=max_sampling_radius,
        )

        self.node_manager.add_nodes_to_buffer(v_set)

        sample_nodes = v_set
        self.graph_constructor.connect_nodes(sample_nodes, neighbors_per_node=neighbors_per_node)

    def _find_path_for_index_pairs(self, start_idx_list, goal_idx_list, return_length=False):
        if len(start_idx_list) != len(goal_idx_list):
            raise ValueError("Start and Goal idx length are not equal")
        path_list = []
        cmax_list = []
        for i in range(len(start_idx_list)):
            path = self.graph_path_finder.get_shortest_path(
                start_idx_list[i], goal_idx_list[i], return_length=return_length
            )
            if return_length:
                path_list.append(path[0])
                cmax_list.append(path[1])
            else:
                path_list.append(path)
        if return_length:
            return path_list, cmax_list
        return path_list

    def _check_paths_exist(
        self, start_idx_list: List[int], goal_idx_list: List[int], require_all_paths: bool = False
    ):
        if len(start_idx_list) != len(goal_idx_list):
            raise ValueError("Start and Goal idx length are not equal")
        path_label = []
        for i in range(len(start_idx_list)):
            path_label.append(
                self.graph_path_finder.path_exists(start_idx_list[i], goal_idx_list[i])
            )
        if require_all_paths:
            label = all(path_label)
        else:
            label = any(path_label)
        return label, path_label

    @profiler.record_function("base_graph_planner/find_path")
    def find_path(
        self,
        x_start: torch.Tensor,
        x_goal: torch.Tensor,
        interpolate_waypoints: bool = True,
        interpolation_steps: int = 100,
        interpolation_type: TrajInterpolationType = TrajInterpolationType.LINEAR,
        validate_interpolated_trajectory: bool = True,
    ) -> GraphPlannerResult:
        timer = CudaEventTimer().start()
        path_result = self._find_path_impl(x_start, x_goal)
        path_result.joint_names = self.joint_names
        if interpolate_waypoints and path_result.success.any():
            # Interpolate successful trajectories:
            path_result.interpolated_waypoints = self.get_interpolated_trajectory(
                path_result.plan_waypoints,
                path_result.success,
                interpolation_steps,
                interpolation_type,
            )
            if validate_interpolated_trajectory:
                # Validate interpolated trajectories:
                batch_size = x_start.shape[0]
                action_samples = path_result.interpolated_waypoints.reshape(
                    batch_size * interpolation_steps, self.action_dim
                )
                feasible_mask = self.check_samples_feasibility(action_samples)
                interpolated_valid = feasible_mask.reshape(batch_size, interpolation_steps).all(
                    dim=1
                )

                # make failed paths false:
                interpolated_valid[~path_result.success] = False
                path_result.success = interpolated_valid

        path_result.solve_time = timer.stop()
        return path_result

    def _find_path_impl(self, x_start: torch.Tensor, x_goal: torch.Tensor):
        """Find path from a batch of initial and goal configs

        Args:
            x_init ([type]): batch of start
            x_goal ([type]): batch of goal
            return_path_lengths (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: b, h, dof
        """
        if len(x_start.shape) != 2:
            log_and_raise("x_start must be a 2D tensor")
        if len(x_goal.shape) != 2:
            log_and_raise("x_goal must be a 2D tensor")
        if x_start.shape[0] != x_goal.shape[0]:
            log_and_raise("x_start and x_goal must have the same batch size")
        if x_start.shape[1] != self.action_dim:
            log_and_raise("x_start must have the same number of actions as the model")
        if x_goal.shape[1] != self.action_dim:
            log_and_raise("x_goal must have the same number of actions as the model")

        num_problems = x_start.shape[0]

        result = GraphPlannerResult(
            success=[False for x in range(num_problems)],
            path_length=self.device_cfg.to_device([np.inf for x in range(num_problems)]),
            plan_waypoints=[None for x in range(num_problems)],
        )
        result.success = torch.as_tensor(
            result.success, device=self.device_cfg.device, dtype=torch.bool
        )
        # check if start and goal are same, if so just return false
        if self.node_manager.n_nodes > (self.config.max_nodes * 0.75):
            self.reset_buffer()
        # add start and goal nodes to graph:
        node_set = torch.cat((x_start.unsqueeze(1), x_goal.unsqueeze(1)), dim=1)
        b, _, _ = node_set.shape
        node_set = node_set.view(b * 2, self.action_dim)
        # check if start and goal are in freespace:
        mask = self.check_samples_feasibility(node_set)
        if mask.all() != True:
            log_warn("Start or End state in collision")
            result.plan_waypoints = [None for _ in range(num_problems)]
            result.valid_query = False
            result.debug_info = "Start or End state in collision"
            return result

        # if start and goal are same, return True:
        linear_path_distance = self.distance_calculator.calculate_weighted_distance(x_start, x_goal)
        if torch.all(linear_path_distance < self.config.cspace_similarity_threshold):
            result.success[:] = True
            result.plan_waypoints = [
                torch.cat((x_start[i], x_goal[i]), dim=0) for i in range(num_problems)
            ]
            result.path_length[:] = linear_path_distance
            return result

        # node_set = self.add_nodes_to_roadmap(node_set, add_exact_node=True)

        start_nodes_in_roadmap, goal_nodes_in_roadmap = (
            self.graph_constructor.initialize_terminal_graph_connections(
                x_start, x_goal, self.default_joint_state
            )
        )

        batch_start_idx = (
            start_nodes_in_roadmap[..., self.action_dim].to(dtype=torch.int64, device="cpu").tolist()
        )
        batch_goal_idx = (
            goal_nodes_in_roadmap[..., self.action_dim].to(dtype=torch.int64, device="cpu").tolist()
        )
        path_exists, exist_label = self._check_paths_exist(
            batch_start_idx, batch_goal_idx, require_all_paths=True
        )
        k_nn = self.config.neighbors_per_node

        if path_exists:
            g_path, path_lengths = self._find_path_for_index_pairs(
                batch_start_idx, batch_goal_idx, return_length=True
            )
            len_max = max([len(g) for g in g_path])
            if len_max > 2:
                g_path, path_lengths = self.path_pruner.prune_path_with_shortcuts(
                    g_path, batch_start_idx, batch_goal_idx
                )
                len_max = max([len(g) for g in g_path])
                exist_label = [len(g) <= 3 for g in g_path]

            if len_max <= 2:
                for k in range(len(g_path)):
                    if len(g_path[k]) == 1:
                        g_path[k] = [g_path[k][0], g_path[k][0]]

                paths = self.node_manager.get_nodes_in_path(g_path)
                result.plan_waypoints = paths
                result.success = exist_label
                result.success = torch.as_tensor(
                    result.success, device=self.device_cfg.device, dtype=torch.bool
                )
                result.path_length = self.device_cfg.to_device(path_lengths)
                return result

        c_max_tensor = (
            self.distance_calculator.calculate_weighted_distance(x_start, x_goal).view(-1)
            * self.config.exploration_radius
        )

        n_nodes = self.config.new_nodes_per_iteration
        # find paths
        idx = 0
        finetune_iter = 0
        path_finding_iter = 0
        # print("Initial", path_exists, exist_label)
        while not path_exists or finetune_iter < (self.config.min_finetune_iterations):
            if all(exist_label):
                no_path_label = exist_label
            else:
                no_path_label = [not x for x in exist_label]
            # choose x_init, x_goal from the ones that don't have a path:
            no_path_idx = np.where(no_path_label)[0].tolist()
            idx = random.choice(no_path_idx)
            self.extend_roadmap_with_ellipsoidal_samples(
                x_start=x_start[idx],
                x_goal=x_goal[idx],
                num_samples=n_nodes,
                max_sampling_radius=c_max_tensor[idx],
                neighbors_per_node=k_nn,
            )
            path_finding_iter += 1
            path_exists, exist_label = self._check_paths_exist(
                batch_start_idx, batch_goal_idx, require_all_paths=True
            )

            if path_exists:
                g_path, path_lengths = self._find_path_for_index_pairs(
                    batch_start_idx, batch_goal_idx, return_length=True
                )
                len_max = max([len(g) for g in g_path])
                if len_max > 2:
                    g_path, path_lengths = self.path_pruner.prune_path_with_shortcuts(
                        g_path, batch_start_idx, batch_goal_idx
                    )
                    len_max = max([len(g) for g in g_path])
                exist_label = [len(g) <= 3 for g in g_path]
                finetune_iter += 1

                if len_max <= 2:
                    break
                c_max_tensor[:] = torch.as_tensor(
                    path_lengths, device=self.device_cfg.device, dtype=self.device_cfg.dtype
                )
            else:
                if path_finding_iter == 1:
                    n_nodes = self.config.new_nodes_per_iteration
                c_max_tensor[idx] = c_max_tensor[idx] * self.config.exploration_radius_growth_factor

            k_nn = int(self.config.neighbors_per_node_growth_factor * k_nn)
            n_nodes = int(self.config.new_nodes_per_iteration_growth_factor * n_nodes)
            if path_finding_iter >= self.config.max_path_finding_iterations:
                break
        path_exists, exist_label = self._check_paths_exist(
            batch_start_idx, batch_goal_idx, require_all_paths=True
        )

        if not path_exists:
            s_path = [None for _ in range(num_problems)]
            path_lengths = [np.inf for _ in range(num_problems)]
            if any(exist_label):
                # do shortcut for only possible paths:
                # get true indices:
                idx_list = np.where(exist_label)[0].tolist()
                batch_start_idx = [batch_start_idx[x] for x in idx_list]
                batch_goal_idx = [batch_goal_idx[x] for x in idx_list]
                path_list = self._find_path_for_index_pairs(batch_start_idx, batch_goal_idx)

                path_list, c_list = self.path_pruner.prune_path_with_shortcuts(
                    path_list, batch_start_idx, batch_goal_idx
                )

                # add this back
                for i, idx in enumerate(idx_list):
                    s_path[idx] = path_list[i]
                    path_lengths[idx] = c_list[i]
            g_path = [None for _ in range(num_problems)]

            # only take paths that are valid:
            for s_idx, s_value in enumerate(s_path):
                if exist_label[s_idx]:
                    g_path[s_idx] = s_value
            # g_path = s_path
        else:
            g_path, path_lengths = self._find_path_for_index_pairs(
                batch_start_idx, batch_goal_idx, return_length=True
            )

            len_max = max([len(g) for g in g_path])
            if len_max > 3:
                g_path, path_lengths = self.path_pruner.prune_path_with_shortcuts(
                    g_path, batch_start_idx, batch_goal_idx
                )
                len_max = max([len(g) for g in g_path])

        for k in range(len(g_path)):
            if g_path[k] is None:
                continue
            if len(g_path[k]) == 1:
                g_path[k] = [g_path[k][0], g_path[k][0]]
        paths = self.node_manager.get_nodes_in_path(g_path)
        result.plan_waypoints = paths  # will have None for invalid paths
        result.success = exist_label
        result.path_length = torch.as_tensor(
            path_lengths, device=self.device_cfg.device, dtype=self.device_cfg.dtype
        )
        result.success = torch.as_tensor(
            result.success, device=self.device_cfg.device, dtype=torch.bool
        )
        return result

    def get_interpolated_trajectory(
        self,
        paths: List[torch.Tensor],
        success: torch.Tensor,
        interpolation_steps: int,
        interpolation_type: TrajInterpolationType,
    ):
        supported_types = [
            TrajInterpolationType.LINEAR,
            TrajInterpolationType.CUBIC,
            TrajInterpolationType.QUINTIC,
        ]

        if interpolation_type not in supported_types:
            log_and_raise(
                "Unsupported interpolation type: {}".format(interpolation_type)
                + "Supported types: {}".format([t.value for t in supported_types])
            )

        interpolated_trajectory_cpu = torch.zeros(
            (len(paths), interpolation_steps, self.action_dim),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        # for loop over only success indices;
        success_idx = torch.where(success)[0].cpu().tolist()
        # print(success, success_idx, paths)
        for b in success_idx:
            current_kind = interpolation_type
            path_waypoints = paths[b].cpu().view(-1, self.action_dim).numpy()
            if path_waypoints.shape[0] == 1:
                log_and_raise("Something is wrong with the graph planner")

            for i_action in range(path_waypoints.shape[-1]):  # interpolate per joint
                interpolated_trajectory_cpu[b, :interpolation_steps, i_action] = linear_smooth(
                    path_waypoints[:, i_action],
                    y=None,
                    n=interpolation_steps,
                    kind=current_kind,
                    last_step=interpolation_steps,
                )

        interpolated_trajectory_gpu = interpolated_trajectory_cpu.to(
            device=self.device_cfg.device, dtype=self.device_cfg.dtype
        )
        return interpolated_trajectory_gpu

    def reset_buffer(self):
        self.node_manager.reset_buffer()
        self.graph_constructor.reset()

    def reset_seed(self):
        """Reset the random seed for the sampler."""
        self.sampling_strategy.reset_seed()
        self.graph_path_finder.reset_seed()

    def reset_cuda_graph(self):
        if hasattr(self.auxiliary_rollout, "reset_cuda_graph"):
            self.auxiliary_rollout.reset_cuda_graph()

    def get_all_rollout_instances(self) -> List[RobotRollout]:
        return [self.feasibility_rollout, self.auxiliary_rollout]

    def warmup(self, num_warmup_iterations: int = 10, max_batch_size: int = 4):
        actions = self.sampling_strategy.generate_feasible_action_samples(
            num_samples=num_warmup_iterations * 2 * max_batch_size
        )

        for i in range(num_warmup_iterations):
            start_x = actions[i : i + max_batch_size, :].clone()
            goal_x = actions[
                num_warmup_iterations + i : num_warmup_iterations + i + max_batch_size, :
            ].clone()
            self.find_path(
                x_start=start_x,
                x_goal=goal_x,
            )

            self.reset_buffer()
        self.reset_seed()

    @property
    def action_dim(self) -> int:
        """Dimensionality of the action space (robot degrees of freedom)."""
        return self.auxiliary_rollout.action_dim

    @property
    def kinematics(self) -> Kinematics:
        """Direct access to the CUDA robot model."""
        # Use auxiliary rollout's model as it's guaranteed to be standard RobotRollout
        return self.auxiliary_rollout.transition_model.robot_model

    @property
    def transition_model(self) -> RobotStateTransition:
        """Direct access to the transition model."""
        return self.auxiliary_rollout.transition_model

    def compute_kinematics(self, state: JointState) -> KinematicsState:
        """Computes forward kinematics for a given joint state."""
        return self.kinematics.compute_kinematics(state)

    @property
    def default_joint_state(self) -> JointState:
        """The default joint state."""
        # Return a JointState object
        return JointState.from_position(
            self.auxiliary_rollout.default_joint_state, joint_names=self.joint_names
        )

    @property
    def joint_names(self) -> List[str]:
        """Ordered list of joint names used in optimization."""
        return self.kinematics.joint_names

    @property
    def action_bound_lows(self) -> torch.Tensor:
        """Lower bounds for the action space."""
        return self.auxiliary_rollout.action_bound_lows

    @property
    def action_bound_highs(self) -> torch.Tensor:
        """Upper bounds for the action space."""
        return self.auxiliary_rollout.action_bound_highs

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the graph."""
        return self.node_manager.n_nodes

    @property
    def cspace_distance_weight(self) -> torch.Tensor:
        """Distance weight for the action space."""
        return self._cspace_distance_weight
