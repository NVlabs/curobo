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
import math
import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

# Third Party
import numpy as np
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo.geom.sdf.world import WorldCollision
from curobo.geom.types import WorldConfig
from curobo.graph.graph_nx import NetworkxGraph
from curobo.rollout.arm_base import ArmBase, ArmBaseConfig
from curobo.rollout.rollout_base import RolloutBase, RolloutMetrics
from curobo.types import tensor
from curobo.types.base import TensorDeviceType
from curobo.types.robot import JointState, RobotConfig, State
from curobo.util.logger import log_info, log_warn
from curobo.util.sample_lib import HaltonGenerator
from curobo.util.torch_utils import get_torch_jit_decorator
from curobo.util.trajectory import InterpolateType, get_interpolated_trajectory
from curobo.util_file import (
    get_robot_configs_path,
    get_task_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)


@dataclass
class GraphResult:
    success: tensor.T_BValue_bool
    start_q: tensor.T_BDOF
    goal_q: tensor.T_BDOF
    path_length: Optional[tensor.T_BValue_float] = None
    solve_time: float = 0.0
    plan: Optional[List[List[tensor.T_DOF]]] = None
    interpolated_plan: Optional[JointState] = None
    metrics: Optional[RolloutMetrics] = None
    valid_query: bool = True
    debug_info: Optional[Any] = None
    optimized_dt: Optional[torch.Tensor] = None
    path_buffer_last_tstep: Optional[List[int]] = None


@dataclass
class Graph:
    nodes: tensor.T_BDOF
    edges: tensor.T_BHDOF_float
    connectivity: tensor.T_BValue_int
    shortest_path_lengths: Optional[tensor.T_BValue_float] = None

    def set_shortest_path_lengths(self, shortest_path_lengths: tensor.T_BValue_float):
        self.shortest_path_lengths = shortest_path_lengths

    def get_node_distance(self):
        if self.shortest_path_lengths is not None:
            min_l = min(self.nodes.shape[0], self.shortest_path_lengths.shape[0])
            return torch.cat(
                (self.nodes[:min_l], self.shortest_path_lengths[:min_l].unsqueeze(1)), dim=-1
            )
        else:
            return None


@dataclass
class GraphConfig:
    max_nodes: int
    steer_delta_buffer: int
    sample_pts: int
    node_similarity_distance: float
    rejection_ratio: int
    k_nn: int
    c_max: float
    vertex_n: int
    graph_max_attempts: int
    graph_min_attempts: int
    init_nodes: int
    use_bias_node: bool
    dof: int
    bounds: torch.Tensor
    tensor_args: TensorDeviceType
    rollout_fn: RolloutBase
    safety_rollout_fn: RolloutBase
    max_buffer: int
    max_cg_buffer: int
    compute_metrics: bool
    interpolation_type: InterpolateType
    interpolation_steps: int
    seed: int
    use_cuda_graph_mask_samples: bool
    distance_weight: torch.Tensor
    bias_node: tensor.T_DOF
    interpolation_dt: float = 0.02
    interpolation_deviation: float = 0.05
    interpolation_acceleration_scale: float = 0.5

    @staticmethod
    def from_dict(
        graph_dict: Dict,
        tensor_args: TensorDeviceType,
        rollout_fn: RolloutBase,
        safety_rollout_fn: RolloutBase,
        use_cuda_graph: bool = True,
    ):
        graph_dict["dof"] = rollout_fn.d_action
        graph_dict["bounds"] = rollout_fn.action_bounds
        graph_dict["distance_weight"] = rollout_fn.cspace_config.cspace_distance_weight
        graph_dict["bias_node"] = rollout_fn.cspace_config.retract_config.view(1, -1)
        graph_dict["interpolation_type"] = InterpolateType(graph_dict["interpolation_type"])
        return GraphConfig(
            **graph_dict,
            tensor_args=tensor_args,
            rollout_fn=rollout_fn,
            safety_rollout_fn=safety_rollout_fn,
            use_cuda_graph_mask_samples=use_cuda_graph,
        )

    @staticmethod
    @profiler.record_function("graph_plan_config/load_from_robot_config")
    def load_from_robot_config(
        robot_cfg: Union[Union[str, Dict], RobotConfig],
        world_model: Optional[Union[Union[str, Dict], WorldConfig]] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        world_coll_checker: Optional[WorldCollision] = None,
        base_cfg_file: str = "base_cfg.yml",
        graph_file: str = "graph.yml",
        self_collision_check: bool = True,
        use_cuda_graph: bool = True,
        seed: Optional[int] = None,
    ):
        graph_data = load_yaml(join_path(get_task_configs_path(), graph_file))
        base_config_data = load_yaml(join_path(get_task_configs_path(), base_cfg_file))
        if isinstance(robot_cfg, str):
            robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_cfg))["robot_cfg"]
        if isinstance(world_model, str):
            world_model = load_yaml(join_path(get_world_configs_path(), world_model))
        if isinstance(robot_cfg, dict):
            robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
        if not self_collision_check:
            base_config_data["constraint"]["self_collision_cfg"]["weight"] = 0.0

        cfg = ArmBaseConfig.from_dict(
            robot_cfg,
            graph_data["model"],
            base_config_data["cost"],
            base_config_data["constraint"],
            base_config_data["convergence"],
            base_config_data["world_collision_checker_cfg"],
            world_model,
            world_coll_checker=world_coll_checker,
        )
        arm_base = ArmBase(cfg)

        if use_cuda_graph:
            cfg_cg = ArmBaseConfig.from_dict(
                robot_cfg,
                graph_data["model"],
                base_config_data["cost"],
                base_config_data["constraint"],
                base_config_data["convergence"],
                base_config_data["world_collision_checker_cfg"],
                world_model,
                world_coll_checker=world_coll_checker,
            )
            arm_base_cg_rollout = ArmBase(cfg_cg)
        else:
            arm_base_cg_rollout = arm_base
        if seed is not None:
            graph_data["graph"]["seed"] = seed
        graph_cfg = GraphConfig.from_dict(
            graph_data["graph"],
            tensor_args,
            arm_base_cg_rollout,
            arm_base,
            use_cuda_graph,
        )
        return graph_cfg


class GraphPlanBase(GraphConfig):
    @profiler.record_function("graph_plan_base/init")
    def __init__(self, config: Optional[GraphConfig] = None):
        if config is not None:
            super().__init__(**vars(config))
        self._rollout_list = None
        self._cu_act_buffer = None
        if self.use_cuda_graph_mask_samples:
            self._cu_act_buffer = torch.zeros(
                (self.max_cg_buffer, 1, self.dof),
                device=self.tensor_args.device,
                dtype=self.tensor_args.dtype,
            )
        self._valid_bias_node = False
        self._check_bias_node = self.use_bias_node
        self.steer_radius = self.node_similarity_distance
        self.xc_search = None
        self.i = 0
        self._valid_bias_node = False
        self._out_traj_state = None
        # validated graph is stored here:
        self.graph = NetworkxGraph()

        self.path = None

        self.cat_buffer = torch.as_tensor(
            [0.0, 0.0, 0.0], device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )
        self.delta_vec = torch.as_tensor(
            range(0, self.steer_delta_buffer),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        )
        self.path = torch.zeros(
            (self.max_nodes + 100, self.dof + 3),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        )
        self.sample_gen = HaltonGenerator(
            self.dof,
            self.tensor_args,
            up_bounds=self.rollout_fn.action_bound_highs,
            low_bounds=self.rollout_fn.action_bound_lows,
            seed=self.seed,
        )

        self.halton_samples, self.gauss_halton_samples = self._sample_pts()
        self.batch_mode = True

        self._rot_frame_col = torch.as_tensor(
            torch.eye(self.dof)[:, 0:1],
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        ).T

        self._max_joint_vel = (
            self.rollout_fn.state_bounds.velocity.view(2, self.dof)[1, :].reshape(1, 1, self.dof)
        ) - 0.02
        self._max_joint_acc = self.rollout_fn.state_bounds.acceleration[1, :] - 0.02
        self._max_joint_jerk = self.rollout_fn.state_bounds.jerk[1, :] - 0.02

        self._rollout_list = None

    def check_feasibility(self, x_set):
        mask = self.mask_samples(x_set)
        return mask.all(), mask

    def get_feasible_sample_set(self, x_samples):
        mask = self.mask_samples(x_samples)
        x_samples = x_samples[mask]
        return x_samples

    def mask_samples(self, x_samples):  # call feasibility here:
        if self.use_cuda_graph_mask_samples and x_samples.shape[0] <= self.max_cg_buffer:
            return self._mask_samples_cuda_graph(x_samples)
        else:
            return self._mask_samples(x_samples)

    @profiler.record_function("geometric_planner/cg_mask_samples")
    def _mask_samples_cuda_graph(self, x_samples):
        d = []
        if self.max_cg_buffer < x_samples.shape[0]:
            for i in range(math.ceil(x_samples.shape[0] / self.max_cg_buffer)):
                start = i * self.max_cg_buffer
                end = (i + 1) * self.max_cg_buffer
                feasible = self._cuda_graph_rollout_constraint(
                    x_samples[start:end, :].unsqueeze(1), use_batch_env=False
                )
                d.append(feasible)
        else:
            feasible = self._cuda_graph_rollout_constraint(
                x_samples.unsqueeze(1), use_batch_env=False
            )
            d.append(feasible)
        mask = torch.cat(d).squeeze()
        return mask

    @profiler.record_function("geometric_planner/mask_samples")
    def _mask_samples(self, x_samples):
        d = []
        if self.safety_rollout_fn.cuda_graph_instance:
            log_error("Cuda graph is using this rollout instance.")
        if self.max_buffer < x_samples.shape[0]:
            # c_samples = x_samples[:, 0:1] * 0.0
            for i in range(math.ceil(x_samples.shape[0] / self.max_buffer)):
                start = i * self.max_buffer
                end = (i + 1) * self.max_buffer
                metrics = self.safety_rollout_fn.rollout_constraint(
                    x_samples[start:end, :].unsqueeze(1), use_batch_env=False
                )
                d.append(metrics.feasible)
        else:
            metrics = self.safety_rollout_fn.rollout_constraint(
                x_samples.unsqueeze(1), use_batch_env=False
            )
            d.append(metrics.feasible)
        mask = torch.cat(d).squeeze()
        return mask

    def _cuda_graph_rollout_constraint(self, x_samples, use_batch_env=False):
        self._cu_act_buffer[: x_samples.shape[0]] = x_samples
        metrics = self.rollout_fn.rollout_constraint_cuda_graph(
            self._cu_act_buffer, use_batch_env=False
        )
        return metrics.feasible[: x_samples.shape[0]]

    def get_samples(self, n_samples: int, bounded: bool = True):
        return self._sample_pts(n_samples, bounded)[0]

    @profiler.record_function("geometric_planner/halton_samples")
    def _sample_pts(self, n_samples=None, bounded=False, unit_ball=False, seed=123):
        # sample in state space:
        if n_samples is None:
            n_samples = self.sample_pts
        if unit_ball:
            halton_samples = self.sample_gen.get_gaussian_samples(n_samples, variance=1.0)
            halton_samples = halton_samples / torch.norm(halton_samples, dim=-1, keepdim=True)
            if self.dof < 3:
                radius_samples = self.sample_gen.get_samples(n_samples, bounded=False)
                radius_samples = torch.clamp(radius_samples[:, 0:1], 0.0, 1.0)
                halton_samples = radius_samples * halton_samples
        else:
            halton_samples = self.sample_gen.get_samples(n_samples, bounded=bounded)
        return halton_samples, halton_samples

    def reset_buffer(self):
        # add a random node to graph:
        self.path = torch.zeros(
            (self.max_nodes, self.dof + 3),
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        )
        self.reset_graph()
        self.path *= 0.0
        self.i = 0
        self._valid_bias_node = False
        self._check_bias_node = self.use_bias_node

    @profiler.record_function("geometric_planner/sample_biased_nodes")
    def get_biased_vertex_set(self, x_start, x_goal, c_max=10.0, c_min=1, n=None, lazy=False):
        if n is None:
            n = self.vertex_n
        # get biased samples that are around x_start and x_goal
        # print(c_min.item(), c_max)

        unit_ball, _ = self._sample_pts(n_samples=n + int(n * self.rejection_ratio), unit_ball=True)
        # compute cost_to_go:
        x_samples = biased_vertex_projection_jit(
            x_start,
            x_goal,
            self.distance_weight,
            c_max,
            c_min,
            self.dof,
            self._rot_frame_col,
            unit_ball,
            self.bounds,
        )

        if False:  # non jit version:
            # rotate frame:
            C = self._compute_rotation_frame(
                x_start * self.distance_weight, x_goal * self.distance_weight
            )

            r = x_start * 0.0
            r[0] = c_max / 2.0
            r[1:] = (c_max**2 - c_min**2) / 2.0
            L = torch.diag(r)
            x_center = (x_start[..., : self.dof] + x_goal[..., : self.dof]) / 2.0
            x_samples = ((C @ L @ unit_ball.T).T) / self.distance_weight + x_center
            # clamp at joint angles:
            x_samples = torch.clamp(x_samples, self.bounds[0, :], self.bounds[1, :])

        if not lazy:
            x_search = self.get_feasible_sample_set(x_samples)
        else:
            x_search = x_samples
        xc_search = cat_xc_jit(x_search, n)
        # c_search = x_search[:, 0:1] * 0.0
        # xc_search = torch.cat((x_search, c_search), dim=1)[:n, :]
        return xc_search

    @profiler.record_function("geometric_planner/compute_rotation_frame")
    def _compute_rotation_frame(self, x_start, x_goal):
        return compute_rotation_frame_jit(x_start, x_goal, self._rot_frame_col)
        #: non jit version below
        a = ((x_goal - x_start) / torch.norm(x_start - x_goal)).unsqueeze(1)

        M = a @ self._rot_frame_col  # .T

        # with torch.cuda.amp.autocast(enabled=False):
        U, _, V = torch.svd(M, compute_uv=True, some=False)
        vec = a.flatten() * 0.0 + 1.0
        vec[-1] = torch.det(U) * torch.det(V)

        C = U @ torch.diag(vec) @ V.T
        return C

    @profiler.record_function("geometric_planner/sample_nodes")
    def get_new_vertex_set(self, n=None, lazy=False):
        if n is None:
            n = self.vertex_n
        # get a new seed value:
        # seed = random.randint(1, 1000)
        # generate new samples:
        x_samples, _ = self._sample_pts(
            n_samples=n + int(n * self.rejection_ratio),
            bounded=True,
        )
        if not lazy:
            x_search = self.get_feasible_sample_set(x_samples)
        else:
            x_search = x_samples
        xc_search = cat_xc_jit(x_search, n)
        # c_search = x_search[:, 0:1] * 0.0

        # xc_search = torch.cat((x_search, c_search), dim=1)[:n, :]
        return xc_search

    @torch.no_grad()
    def validate_graph(self):
        self._validate_graph()

    def get_graph_edges(self):
        """Return edges in the graph with start node and end node locations

        Returns:
            tensor
        """
        self.graph.update_graph()
        edge_list = self.graph.get_edges()
        edges = torch.as_tensor(
            edge_list, device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )

        # find start and end points for these edges:
        start_pts = self.path[edges[:, 0].long(), : self.dof].unsqueeze(1)
        end_pts = self.path[edges[:, 1].long(), : self.dof].unsqueeze(1)

        # first check the start and end points:
        node_edges = torch.cat((start_pts, end_pts), dim=1)
        return node_edges, edges

    def get_graph(self):
        node_edges, edge_connect = self.get_graph_edges()
        nodes = self.path[: self.i, : self.dof]
        return Graph(nodes=nodes, edges=node_edges, connectivity=edge_connect)

    def _validate_graph(self):
        self.graph.update_graph()
        edge_list = self.graph.get_edges()
        edges = torch.as_tensor(
            edge_list, device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )

        # find start and end points for these edges:
        start_pts = self.path[edges[:, 0].long(), : self.dof]
        end_pts = self.path[edges[:, 1].long(), : self.dof]

        # first check the start and end points:
        # get largest edge:
        dist = self._distance(start_pts, end_pts, norm=False)
        n = torch.ceil(torch.max(torch.abs(dist) / self.steer_radius)).item() + 1
        if n + 1 > self.delta_vec.shape[0]:
            print("error", n, self.delta_vec.shape)
        delta_vec = self.delta_vec[: int(n + 1)] / n

        #
        line_vec = (
            start_pts.unsqueeze(1)
            + delta_vec.unsqueeze(1) @ dist.unsqueeze(1) / self.distance_weight
        )
        b, h, _ = line_vec.shape
        print("Number of points to check: ", b * h)

        mask = self.mask_samples(line_vec.view(b * h, self.dof))
        mask = ~mask.view(b, h)
        # edge mask contains all edges that are valid for current world:
        edge_mask = ~torch.any(mask, dim=1)

        # add these to graph:
        new_edges = edges[edge_mask]  # .cpu().numpy()#.tolist()

        #
        node_mask = ~mask[:, 0]
        node_list = (
            torch.unique(
                torch.cat((edges[node_mask][:, 0].long(), edges[~mask[:, -1]][:, 1].long()))
            )
            .cpu()
            .tolist()
        )

        new_path = self.path[node_list]
        new_path[:, self.dof + 1] = torch.as_tensor(
            [x for x in range(new_path.shape[0])],
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype,
        )
        self.i = new_path.shape[0]  # + 1
        self.path[: self.i] = new_path

        reindex_edges = []
        if len(new_edges) > 0:
            # reindex edges:
            for e in range(len(new_edges)):
                st_idx = node_list.index(int(new_edges[e][0]))
                end_idx = node_list.index(int(new_edges[e][1]))
                reindex_edges.append([st_idx, end_idx])

            new_edges[:, 0:2] = torch.as_tensor(
                reindex_edges, device=self.tensor_args.device, dtype=self.tensor_args.dtype
            )
        else:
            print("ERROR")
            new_edges = edges
        d = self._distance(
            self.path[new_edges[:, 0].long(), : self.dof],
            self.path[new_edges[:, 1].long(), : self.dof],
        )
        new_edges[:, 2] = d
        new_edges = new_edges.detach().cpu().numpy().tolist()

        # compute path lengths:

        new_edges = [[int(x[0]), int(x[1]), x[2]] for x in new_edges]
        # self.i += 1
        self.path[self.i :] *= 0.0
        #

        self.graph.reset_graph()
        self.graph.add_edges(new_edges)
        self.graph.add_nodes(list(range(self.i)))
        self.graph.update_graph()
        print("Validated graph", len(new_edges), edges.shape)

    def _get_graph_shortest_path(self, start_node_idx, goal_node_idx, return_length=False):
        # st_time = time.time()
        path = self.graph.get_shortest_path(
            start_node_idx, goal_node_idx, return_length=return_length
        )
        # print('Graph search time: ',time.time() - st_time)
        return path

    def batch_get_graph_shortest_path(self, start_idx_list, goal_idx_list, return_length=False):
        if len(start_idx_list) != len(goal_idx_list):
            raise ValueError("Start and Goal idx length are not equal")
        path_list = []
        cmax_list = []
        for i in range(len(start_idx_list)):
            path = self._get_graph_shortest_path(
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

    @torch.no_grad()
    def batch_shortcut_path(self, g_path, start_idx, goal_idx):
        edge_set = []
        for k in range(len(g_path)):
            path = self.path[g_path[k]]
            for i in range(path.shape[0]):
                for j in range(i, path.shape[0]):
                    edge_set.append(
                        torch.cat((path[i : i + 1], path[j : j + 1]), dim=0).unsqueeze(0)
                    )
        edge_set = torch.cat(edge_set, dim=0)
        self.connect_nodes(edge_set=edge_set)
        s_path, c_max = self.batch_get_graph_shortest_path(start_idx, goal_idx, return_length=True)
        return s_path, c_max

    def get_node_idx(self, goal_state, exact=False) -> Optional[int]:
        goal_state = torch.as_tensor(
            goal_state, device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )
        dist = torch.norm(
            self._distance(goal_state, self.path[: self.i, : self.dof], norm=False), dim=-1
        )
        c_idx = torch.argmin(dist)
        if exact:
            if dist[c_idx] != 0.0:
                return None
            else:
                return c_idx.item()
        if dist[c_idx] <= self.node_similarity_distance:
            return c_idx.item()

    def get_path_lengths(self, goal_idx):
        path_lengths = self.graph.get_path_lengths(goal_idx)
        path_length = {
            "position": self.path[: self.i, : self.dof],
            "value": torch.as_tensor(
                path_lengths, device=self.tensor_args.device, dtype=self.tensor_args.dtype
            ),
        }
        return path_length

    def get_graph_shortest_path_lengths(self, goal_idx: int):
        graph = self.get_graph()
        shortest_paths = self.get_path_lengths(goal_idx)
        graph.set_shortest_path_lengths(shortest_paths["value"])
        return graph

    def path_exists(self, start_node_idx, goal_node_idx):
        return self.graph.path_exists(start_node_idx, goal_node_idx)

    def batch_path_exists(self, start_idx_list, goal_idx_list, all_paths=False):
        if len(start_idx_list) != len(goal_idx_list):
            raise ValueError("Start and Goal idx length are not equal")
        path_label = []
        for i in range(len(start_idx_list)):
            path_label.append(self.path_exists(start_idx_list[i], goal_idx_list[i]))
        if all_paths:
            label = all(path_label)
        else:
            label = any(path_label)
        return label, path_label

    @torch.no_grad()
    def find_paths(self, x_init, x_goal, interpolation_steps: Optional[int] = None) -> GraphResult:
        start_time = time.time()
        path = None
        try:
            path = self._find_paths(x_init, x_goal)
            path.success = torch.as_tensor(
                path.success, device=self.tensor_args.device, dtype=torch.bool
            )
            path.solve_time = time.time() - start_time

        except ValueError as e:
            log_info(e)
            self.reset_buffer()
            torch.cuda.empty_cache()
            success = torch.zeros(x_init.shape[0], device=self.tensor_args.device, dtype=torch.bool)
            path = GraphResult(success, x_init, x_goal)
            return path
        except RuntimeError as e:
            log_warn(e)
            self.reset_buffer()
            torch.cuda.empty_cache()
            success = torch.zeros(x_init.shape[0], device=self.tensor_args.device, dtype=torch.long)
            path = GraphResult(success, x_init, x_goal)
            return path
        if self.interpolation_type is not None and (torch.count_nonzero(path.success) > 0):
            (
                path.interpolated_plan,
                path.path_buffer_last_tstep,
                path.optimized_dt,
            ) = self.get_interpolated_trajectory(path.plan, interpolation_steps)
            # path.js_interpolated_plan = self.rollout_fn.get_full_dof_from_solution(
            #    path.interpolated_plan
            # )
            if self.compute_metrics:
                # compute metrics on interpolated plan:
                path.metrics = self.get_metrics(path.interpolated_plan)

                path.success = torch.logical_and(path.success, torch.all(path.metrics.feasible, 1))

        return path

    @abstractmethod
    def _find_paths(self, x_search, c_search, x_init) -> GraphResult:
        raise NotImplementedError

    def compute_path_length(self, path):
        # compute cost to go to next timestep:
        next_pt_path = path.roll(-1, dims=0)
        dist_vec = self._distance(next_pt_path, path)[:-1]
        path_length = torch.sum(dist_vec)
        return path_length

    def reset_graph(self):
        self.graph.reset_graph()

    @profiler.record_function("geometric_planner/compute_distance")
    def _distance(self, pt, batch_pts, norm=True):
        if norm:
            return compute_distance_norm_jit(pt, batch_pts, self.distance_weight)
        else:
            return compute_distance_jit(pt, batch_pts, self.distance_weight)

    def distance(self, pt, batch_pts, norm=True):
        return self._distance(pt, batch_pts, norm=norm)

    def _hybrid_nearest(self, sample_node, path, radius, k_n=10):
        # compute distance:
        dist = self._distance(sample_node[..., : self.dof], path[:, : self.dof])
        nodes = path[dist < radius]
        if nodes.shape[0] < k_n:
            _, idx = torch.topk(dist, k_n, largest=False)
            nodes = path[idx]  # , idx
        return nodes

    def _nearest(self, sample_point, current_graph):
        dist = self._distance(sample_point[..., : self.dof], current_graph[:, : self.dof])
        _, idx = torch.min(dist, 0)
        return current_graph[idx], idx

    def _k_nearest(self, sample_point, current_graph, k=10):
        dist = self._distance(sample_point[..., : self.dof], current_graph[:, : self.dof])
        # give the k nearest:
        # get_top_k(dist, k)
        _, idx = torch.topk(dist, k, largest=False)
        return current_graph[idx]  # , idx

    @profiler.record_function("geometric_planner/k_nearest")
    def _batch_k_nearest(self, sample_point, current_graph, k=10):
        dist = self._distance(
            sample_point[:, : self.dof].unsqueeze(1), current_graph[:, : self.dof]
        )
        # give the k nearest:
        # get_top_k(dist, k)
        _, idx = torch.topk(dist, k, largest=False, dim=-1)
        return current_graph[idx]  # , idx

    def _near(self, sample_point, current_graph, radius):
        dist = self._distance(sample_point[..., : self.dof], current_graph[:, : self.dof])
        nodes = current_graph[dist < radius]
        return nodes

    @profiler.record_function("geometric_planner/batch_steer_and_connect")
    def _batch_steer_and_connect(
        self,
        start_nodes,
        goal_nodes,
        add_steer_pts=-1,
        lazy=False,
        add_exact_node=False,
    ):
        """
            Connect node from start to goal where both are batched.
        Args:
            start_node ([type]): [description]
            goal_nodes ([type]): [description]
        """

        steer_nodes, _ = self._batch_steer(
            start_nodes,
            goal_nodes,
            add_steer_pts=add_steer_pts,
            lazy=lazy,
        )
        self._add_batch_edges_to_graph(
            steer_nodes, start_nodes, lazy=lazy, add_exact_node=add_exact_node
        )

    @profiler.record_function("geometric_planner/batch_steer")
    def _batch_steer(
        self,
        start_nodes,
        desired_nodes,
        steer_radius=None,
        add_steer_pts=-1,
        lazy=False,
    ):
        if lazy:
            extra_data = self.cat_buffer.unsqueeze(0).repeat(desired_nodes.shape[0], 1)
            current_node = torch.cat((desired_nodes, extra_data), dim=1)
            return current_node, True

        steer_radius = self.steer_radius if steer_radius is None else steer_radius
        dof = self.dof

        current_node = start_nodes

        g_vec = self._distance(
            start_nodes[..., :dof], desired_nodes[..., :dof], norm=False
        )  # .unsqueeze(0)

        n = torch.ceil(torch.max(torch.abs(g_vec) / steer_radius)).item() + 1

        delta_vec = self.delta_vec[: int(n + 1)] / n

        #
        line_vec = (
            start_nodes[..., :dof].unsqueeze(1)
            + delta_vec.unsqueeze(1) @ g_vec.unsqueeze(1) / self.distance_weight
        )
        b, h, dof = line_vec.shape
        line_vec = line_vec.view(b * h, dof)
        # print("Collision checks: ", b)
        # check along line vec:
        mask = self.mask_samples(line_vec)

        line_vec = line_vec.view(b, h, dof)
        # TODO: Make this cleaner..
        mask = mask.view(b, h).to(dtype=torch.int8)
        mask[mask == 0.0] = -1.0
        mask = mask * (delta_vec + 1.0)
        mask[mask < 0.0] = 1 / (mask[mask < 0.0])

        _, idx = torch.min(mask, dim=1)
        # idx will contain 1 when there is no collision.
        idx -= 1
        idx[idx == -1] = h - 1
        # idx contains the position of the first collision
        # if idx value is zero, then there is not path, so return the current node,
        # or you can just return line_vec[idx]
        if add_steer_pts > 0:
            raise NotImplementedError("Steer point addition is not implemented for batch mode")
        new_nodes = torch.diagonal(line_vec[:, idx], dim1=0, dim2=1).transpose(0, 1)
        edge_cost = self._distance(new_nodes[:, :dof], start_nodes[:, :dof])
        # current_node = new_node
        extra_data = self.cat_buffer.unsqueeze(0).repeat(new_nodes.shape[0], 1)
        extra_data[:, 2] = edge_cost
        current_node = torch.cat((new_nodes, extra_data), dim=1)
        return current_node, True

    @profiler.record_function("geometric_planner/add_edges_to_graph")
    def _add_batch_edges_to_graph(self, new_nodes, start_nodes, lazy=False, add_exact_node=False):
        # add new nodes to graph:
        node_set = self.add_nodes_to_graph(new_nodes[:, : self.dof], add_exact_node=add_exact_node)
        # now connect start nodes to new nodes:
        edge_list = []
        edge_distance = (
            self.distance(start_nodes[:, : self.dof], node_set[:, : self.dof])
            .to(device="cpu")
            .tolist()
        )
        start_idx_list = start_nodes[:, self.dof + 1].to(device="cpu", dtype=torch.int64).tolist()
        goal_idx_list = node_set[:, self.dof + 1].to(device="cpu", dtype=torch.int64).tolist()
        edge_list = [
            [start_idx_list[x], goal_idx_list[x], edge_distance[x]]
            for x in range(node_set.shape[0])
        ]
        self.graph.add_edges(edge_list)
        return True

    @profiler.record_function("geometric_planner/add_nodes")
    def add_nodes_to_graph(self, nodes, add_exact_node=False):
        # TODO: check if this and unique nodes fn can be merged
        # Check for duplicates in new nodes:
        dist_node = self.distance(nodes[:, : self.dof].unsqueeze(1), nodes[:, : self.dof])
        node_distance = self.node_similarity_distance
        if add_exact_node:
            node_distance = 0.0

        unique_nodes, n_inv = get_unique_nodes(dist_node, nodes, node_distance)

        node_set = self._add_unique_nodes_to_graph(unique_nodes, add_exact_node=add_exact_node)
        node_set = node_set[n_inv]
        return node_set

    @profiler.record_function("geometric_planner/add_unique_nodes")
    def _add_unique_nodes_to_graph(self, nodes, add_exact_node=False, skip_unique_check=False):
        if self.i > 0:  # and not skip_unique_check:
            dist, idx = torch.min(
                self.distance(nodes[:, : self.dof].unsqueeze(1), self.path[: self.i, : self.dof]),
                dim=-1,
            )
            node_distance = self.node_similarity_distance
            if add_exact_node:
                node_distance = 0.0
            flag = dist <= node_distance
            new_nodes = nodes[~flag]

            if self.path.shape[0] <= self.i + new_nodes.shape[0]:
                raise ValueError(
                    "reached max_nodes in graph, reduce graph attempts or increase max_nodes",
                    self.path.shape,
                    self.i,
                    new_nodes.shape,
                )
            self.path, node_set, i_new = add_new_nodes_jit(
                nodes, new_nodes, flag, self.cat_buffer, self.path, idx, self.i, self.dof
            )

        else:
            self.path, node_set, i_new = add_all_nodes_jit(
                nodes, self.cat_buffer, self.path, self.i, self.dof
            )

        self.i += i_new

        return node_set

    @profiler.record_function("geometric_planner/connect_nodes")
    def connect_nodes(
        self,
        x_set=None,
        connect_mode="knn",
        debug=False,
        lazy=False,
        add_exact_node=False,
        k_nn=10,
        edge_set=None,
    ):
        # connect the batch to the existing graph
        path = self.path
        dof = self.dof

        i = self.i
        if x_set is not None:
            if x_set.shape[0] == 0:
                log_info("no valid configuration found")
                return

            if connect_mode == "radius":
                raise NotImplementedError
                scale_radius = self.neighbour_radius * (np.log(i) / i) ** (1 / dof)
                nodes = self._near(sample_node, path[:i, :], radius=scale_radius)
                if nodes.shape[0] == 0:
                    nodes = self._k_nearest(sample_node, path[:i, :], k=k_n)
            elif connect_mode == "nearest":
                nodes = self._batch_k_nearest(x_set, path[:i, :], k=k_nn)[1:]
            elif connect_mode == "knn":
                # k_n = min(max(int(1 * 2.71828 * np.log(i)), k_nn), i)
                # print(k_n, self.i, k_nn)
                k_n = min(k_nn, i)

                nodes = self._batch_k_nearest(x_set, path[:i, :], k=k_n)
            elif connect_mode == "hybrid":
                k_n = min(max(int(1 * 2.71828 * np.log(i)), k_nn), i)
                nodes = self._batch_k_nearest(x_set, path[:i, :], k=k_n)
                print("Hybrid will default to knn")
            # you would end up with:
            # for each node in x_set, you would have n nodes to connect
            start_nodes = (
                x_set.unsqueeze(1)
                .repeat(1, nodes.shape[1], 1)
                .reshape(x_set.shape[0] * nodes.shape[1], -1)
            )
            goal_nodes = nodes.reshape(
                x_set.shape[0] * nodes.shape[1], -1
            )  # batch x k_n or batch x 1

            if edge_set is not None:
                # add 0th index to goal_node and 1st index to start
                goal_nodes = torch.cat((edge_set[:, 0], goal_nodes), dim=0)
                start_nodes = torch.cat((edge_set[:, 1, : self.dof], start_nodes), dim=0)
        elif edge_set is not None:
            goal_nodes = edge_set[:, 0]
            start_nodes = edge_set[:, 1, : self.dof]
        self._batch_steer_and_connect(
            goal_nodes, start_nodes, add_steer_pts=-1, lazy=lazy, add_exact_node=add_exact_node
        )

    def get_paths(self, path_list):
        paths = []
        for i in range(len(path_list)):
            paths.append(self.path[path_list[i], : self.dof])
        return paths

    # get interpolated trajectory
    def get_interpolated_trajectory(
        self, trajectory: List[tensor.T_HDOF_float], interpolation_steps: Optional[int] = None
    ):
        buffer = self.interpolation_steps
        if interpolation_steps is not None:
            buffer = interpolation_steps
        interpolation_type = self.interpolation_type
        if interpolation_type == InterpolateType.LINEAR_CUDA:
            log_warn(
                "LINEAR_CUDA interpolation not supported for GraphPlanner, switching to LINEAR"
            )
            interpolation_type = InterpolateType.LINEAR
        if (
            self._out_traj_state is None
            or self._out_traj_state.shape[0] != len(trajectory)
            or self._out_traj_state.shape[1] != buffer
        ):
            self._out_traj_state = JointState.from_position(
                torch.zeros(
                    (len(trajectory), buffer, trajectory[0].shape[-1]),
                    device=self.tensor_args.device,
                ),
                joint_names=self.rollout_fn.joint_names,
            )

        out_traj_state, last_tstep, opt_dt = get_interpolated_trajectory(
            trajectory,
            self._out_traj_state,
            interpolation_steps,
            self.interpolation_dt,
            self._max_joint_vel,
            self._max_joint_acc,  # * self.interpolation_acceleration_scale,
            self._max_joint_jerk,
            kind=self.interpolation_type,
            tensor_args=self.tensor_args,
            max_deviation=self.interpolation_deviation,
        )
        out_traj_state.joint_names = self.rollout_fn.joint_names

        return out_traj_state, last_tstep, opt_dt

    # validate plan
    def get_metrics(self, state: State):
        # compute metrics
        metrics = self.safety_rollout_fn.get_metrics(state)
        return metrics

    def reset_seed(self):
        self.safety_rollout_fn.reset_seed()
        self.sample_gen = HaltonGenerator(
            self.dof,
            self.tensor_args,
            up_bounds=self.safety_rollout_fn.action_bound_highs,
            low_bounds=self.safety_rollout_fn.action_bound_lows,
            seed=self.seed,
        )

    def reset_cuda_graph(self):
        self.rollout_fn.reset_cuda_graph()

    def get_all_rollout_instances(self) -> List[RolloutBase]:
        if self._rollout_list is None:
            self._rollout_list = [self.safety_rollout_fn, self.rollout_fn]
        return self._rollout_list

    def warmup(self, x_start: Optional[torch.Tensor] = None, x_goal: Optional[torch.Tensor] = None):
        pass


@get_torch_jit_decorator(dynamic=True)
def get_unique_nodes(dist_node: torch.Tensor, nodes: torch.Tensor, node_distance: float):
    node_flag = dist_node <= node_distance
    dist_node[node_flag] = 0.0
    dist_node[~node_flag] = 1.0
    _, idx = torch.min(dist_node, dim=-1)
    n_idx, n_inv = torch.unique(idx, return_inverse=True)

    #
    unique_nodes = nodes[n_idx]
    return unique_nodes, n_inv


@get_torch_jit_decorator(force_jit=True, dynamic=True)
def add_new_nodes_jit(
    nodes, new_nodes, flag, cat_buffer, path, idx, i: int, dof: int
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    new_idx = torch.as_tensor(
        [i + x for x in range(new_nodes.shape[0])],
        device=new_nodes.device,
        dtype=new_nodes.dtype,
    )

    old_node_idx = idx[flag]

    node_set = torch.cat((nodes, cat_buffer.unsqueeze(0).repeat(nodes.shape[0], 1)), dim=-1)
    # node_set[flag][:, self.dof + 1] = old_node_idx.to(dtype=node_set.dtype)
    node_set[flag, dof + 1] = old_node_idx.to(dtype=node_set.dtype)

    path[i : i + new_nodes.shape[0], :dof] = new_nodes
    path[i : i + new_nodes.shape[0], dof + 1] = new_idx
    node_temp = node_set[~flag]
    node_temp[:, dof + 1] = new_idx
    node_set[~flag] = node_temp
    return path, node_set, new_nodes.shape[0]


@get_torch_jit_decorator(force_jit=True, dynamic=True)
def add_all_nodes_jit(
    nodes, cat_buffer, path, i: int, dof: int
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    new_idx = torch.as_tensor(
        [i + x for x in range(nodes.shape[0])],
        device=nodes.device,
        dtype=nodes.dtype,
    )

    node_set = torch.cat((nodes, cat_buffer.unsqueeze(0).repeat(nodes.shape[0], 1)), dim=-1)

    path[i : i + nodes.shape[0], :dof] = nodes
    path[i : i + nodes.shape[0], dof + 1] = new_idx
    node_set[:, dof + 1] = new_idx
    return path, node_set, nodes.shape[0]


@get_torch_jit_decorator(force_jit=True, dynamic=True)
def compute_distance_norm_jit(pt, batch_pts, distance_weight):
    vec = (batch_pts - pt) * distance_weight
    dist = torch.norm(vec, dim=-1)
    return dist


@get_torch_jit_decorator(dynamic=True)
def compute_distance_jit(pt, batch_pts, distance_weight):
    vec = (batch_pts - pt) * distance_weight
    return vec


@get_torch_jit_decorator(dynamic=True)
def compute_rotation_frame_jit(
    x_start: torch.Tensor, x_goal: torch.Tensor, rot_frame_col: torch.Tensor
) -> torch.Tensor:
    a = ((x_goal - x_start) / torch.norm(x_start - x_goal)).unsqueeze(1)

    M = a @ rot_frame_col  # .T

    # with torch.cuda.amp.autocast(enabled=False):
    U, _, V = torch.svd(M, compute_uv=True, some=False)
    vec = a.flatten() * 0.0 + 1.0
    vec[-1] = torch.det(U) * torch.det(V)

    C = U @ torch.diag(vec) @ V.T
    return C


@get_torch_jit_decorator(force_jit=True, dynamic=True)
def biased_vertex_projection_jit(
    x_start,
    x_goal,
    distance_weight,
    c_max: float,
    c_min: float,
    dof: int,
    rot_frame_col: torch.Tensor,
    unit_ball: torch.Tensor,
    bounds: torch.Tensor,
) -> torch.Tensor:
    C = compute_rotation_frame_jit(
        x_start * distance_weight,
        x_goal * distance_weight,
        rot_frame_col,
    )

    r = x_start * 0.0
    r[0] = c_max / 2.0
    r[1:] = (c_max**2 - c_min**2) / 2.0
    L = torch.diag(r)
    x_center = (x_start[..., :dof] + x_goal[..., :dof]) / 2.0
    x_samples = ((C @ L @ unit_ball.T).T) / distance_weight + x_center
    # clamp at joint angles:
    x_samples = torch.clamp(x_samples, bounds[0, :], bounds[1, :]).contiguous()

    return x_samples


@get_torch_jit_decorator(force_jit=True, dynamic=True)
def cat_xc_jit(x, n: int):
    c = x[:, 0:1] * 0.0
    xc_search = torch.cat((x, c), dim=1)[:n, :]
    return xc_search
