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
import random
from cmath import inf
from typing import Optional

# Third Party
import numpy as np
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo.graph.graph_base import GraphConfig, GraphPlanBase, GraphResult
from curobo.util.logger import log_error, log_info, log_warn


class PRMStar(GraphPlanBase):
    def __init__(self, config: GraphConfig):
        super().__init__(config)

    @torch.no_grad()
    def _find_paths(self, x_init_batch, x_goal_batch, all_paths=False):
        if all_paths:
            return self._find_all_path(x_init_batch, x_goal_batch)
        else:
            return self._find_one_path(x_init_batch, x_goal_batch)

    @profiler.record_function("geometric_planner/prm/add_bias_graph")
    def _add_bias_graph(self, x_init_batch, x_goal_batch, node_set_batch, node_set):
        # if retract state is not in collision add it:
        if self._check_bias_node:
            bias_mask = self.mask_samples(self.bias_node)
            if bias_mask.all() == True:
                self._valid_bias_node = True
            else:
                log_warn("Bias node is not valid, not using bias node")
            self._check_bias_node = False

        if self._valid_bias_node:
            # add retract config to node set:
            start_retract = torch.cat(
                (
                    x_init_batch.unsqueeze(1),
                    self.bias_node.repeat(x_init_batch.shape[0], 1).unsqueeze(1),
                ),
                dim=1,
            )
            goal_retract = torch.cat(
                (
                    x_goal_batch.unsqueeze(1),
                    self.bias_node.repeat(x_init_batch.shape[0], 1).unsqueeze(1),
                ),
                dim=1,
            )
            retract_set = torch.cat((start_retract, goal_retract), dim=0)

            b_retract, _, _ = retract_set.shape
            retract_set = self.add_nodes_to_graph(
                retract_set.view(retract_set.shape[0] * 2, self.dof)
            )
            retract_set_batch = retract_set.view(b_retract, 2, retract_set.shape[-1])
            # create an edge set:
            #    connecting start to goal and also goal to start:
            edge_set = torch.cat(
                (
                    node_set_batch,
                    torch.flip(node_set_batch, dims=[1]),
                    retract_set_batch,
                    torch.flip(retract_set_batch, dims=[1]),
                ),
                dim=0,
            )
        else:
            edge_set = torch.cat(
                (
                    node_set_batch,
                    torch.flip(node_set_batch, dims=[1]),
                ),
                dim=0,
            )
        self.connect_nodes(
            node_set[:, : self.dof],
            edge_set=edge_set,
            k_nn=self.k_nn,
            connect_mode="knn",
            add_exact_node=False,
        )

    @torch.no_grad()
    def _find_one_path(self, x_init_batch, x_goal_batch):
        """Find path from a batch of initial and goal configs

        Args:
            x_init ([type]): batch of start
            x_goal ([type]): batch of goal
            return_path_lengths (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: b, h, dof
        """
        result = GraphResult(
            start_q=x_init_batch,
            goal_q=x_goal_batch,
            success=[False for x in range(x_init_batch.shape[0])],
            path_length=self.tensor_args.to_device([inf for x in range(x_init_batch.shape[0])]),
        )
        # check if start and goal are same, if so just return false
        if self.i > (self.max_nodes * 0.75):
            self.reset_buffer()
        # add start and goal nodes to graph:
        node_set = torch.cat((x_init_batch.unsqueeze(1), x_goal_batch.unsqueeze(1)), dim=1)

        b, _, dof = node_set.shape
        node_set = node_set.view(b * 2, dof)
        # check if start and goal are in freespace:
        mask = self.mask_samples(node_set)
        if mask.all() != True:
            log_warn("Start or End state in collision", exc_info=False)
            node_set_batch = node_set.view(b, 2, node_set.shape[-1])
            result.plan = [node_set_batch[i, :, : self.dof] for i in range(node_set_batch.shape[0])]
            result.valid_query = False
            result.debug_info = "Start or End state in collision"
            return result
        node_set = self.add_nodes_to_graph(node_set, add_exact_node=True)
        node_set_batch = node_set.view(b, 2, node_set.shape[-1])
        if (
            torch.min(
                torch.abs(node_set_batch[:, 0, self.dof + 1] - node_set_batch[:, 1, self.dof + 1])
            )
            == 0.0
        ):
            log_warn("WARNING: Start and Goal are same")
            result.success = [False for x in range(x_init_batch.shape[0])]
            result.plan = [node_set_batch[i, :, : self.dof] for i in range(node_set_batch.shape[0])]
            return result

        self._add_bias_graph(x_init_batch, x_goal_batch, node_set_batch, node_set)

        batch_start_idx = (
            node_set_batch[:, 0, self.dof + 1].to(dtype=torch.int64, device="cpu").tolist()
        )
        batch_goal_idx = (
            node_set_batch[:, 1, self.dof + 1].to(dtype=torch.int64, device="cpu").tolist()
        )

        graph_attempt = 0
        path_exists, exist_label = self.batch_path_exists(batch_start_idx, batch_goal_idx)
        k_nn = self.k_nn
        s_path = [[x, x] for x in batch_start_idx]
        c_max_all = [inf for _ in batch_start_idx]
        c_min = self.distance(x_init_batch, x_goal_batch).cpu().numpy()

        # NOTE: c_max is scaled by 10.0, this could be replaced by reading c_min
        c_max = np.ravel([self.c_max * c_min[i] for i in range(x_init_batch.shape[0])])
        if path_exists:
            idx_list = np.where(exist_label)[0].tolist()
            batch_start_ = [batch_start_idx[x] for x in idx_list]
            batch_goal_ = [batch_goal_idx[x] for x in idx_list]
            g_path, c_max_t = self.batch_get_graph_shortest_path(
                batch_start_, batch_goal_, return_length=True
            )

            len_min = min([len(g) for g in g_path])
            if len_min > 2:
                g_path, c_max_t = self.batch_shortcut_path(g_path, batch_start_, batch_goal_)
                len_min = min([len(g) for g in g_path])
            for i, idx in enumerate(idx_list):
                s_path[idx] = g_path[i]
                c_max[idx] = c_max_t[i]
                c_max_all[idx] = c_max_t[i]

            s_new_path = []

            # only take paths that are valid:
            for g_i, g_p in enumerate(s_path):
                if exist_label[g_i]:
                    s_new_path.append(g_p)
            s_path = s_new_path
            if len_min <= 2:
                paths = self.get_paths(s_path)
                result.plan = paths
                result.success = exist_label
                result.path_length = self.tensor_args.to_device(c_max_all)
                return result

        n_nodes = self.init_nodes
        # find paths
        idx = 0
        while not path_exists or graph_attempt <= (self.graph_min_attempts):
            no_path_label = exist_label
            if not any(exist_label):
                no_path_label = [not x for x in exist_label]
            no_path_idx = np.where(no_path_label)[0].tolist()
            idx = random.choice(no_path_idx)
            self.build_graph(
                x_start=x_init_batch[idx],
                x_goal=x_goal_batch[idx],
                bias_samples=True,
                k_nn=k_nn,
                c_max=c_max[idx],
                c_min=c_min[idx],
                number_of_nodes=n_nodes,
                lazy_nodes=False,
            )
            graph_attempt += 1

            path_exists, exist_label = self.batch_path_exists(batch_start_idx, batch_goal_idx)
            if path_exists:
                idx_list = np.where(exist_label)[0].tolist()
                batch_start_ = [batch_start_idx[x] for x in idx_list]
                batch_goal_ = [batch_goal_idx[x] for x in idx_list]

                g_path, c_max_ = self.batch_get_graph_shortest_path(
                    batch_start_, batch_goal_, return_length=True
                )
                len_min = min([len(g) for g in g_path])
                if len_min > 2:
                    g_path, c_max_ = self.batch_shortcut_path(g_path, batch_start_, batch_goal_)
                    len_min = min([len(g) for g in g_path])
                for i, idx in enumerate(idx_list):
                    c_max[idx] = c_max_[i]

                if len_min <= 2:
                    break
            else:
                if graph_attempt == 1:
                    n_nodes = self.vertex_n
                c_max[idx] += c_min[idx] * 0.05

            k_nn += int(0.1 * k_nn)
            n_nodes += int(0.1 * n_nodes)
            if graph_attempt > self.graph_max_attempts:
                break
        path_exists, exist_label = self.batch_path_exists(
            batch_start_idx, batch_goal_idx, all_paths=True
        )

        if not path_exists:
            s_path = [[x, x] for x in batch_start_idx]
            c_max = [inf for _ in batch_start_idx]
            if any(exist_label):
                # do shortcut for only possible paths:
                # get true indices:
                idx_list = np.where(exist_label)[0].tolist()
                batch_start_idx = [batch_start_idx[x] for x in idx_list]
                batch_goal_idx = [batch_goal_idx[x] for x in idx_list]
                path_list = self.batch_get_graph_shortest_path(batch_start_idx, batch_goal_idx)

                path_list, c_list = self.batch_shortcut_path(
                    path_list, batch_start_idx, batch_goal_idx
                )
                # add this back
                for i, idx in enumerate(idx_list):
                    s_path[idx] = path_list[i]
                    c_max[idx] = c_list[i]
            g_path = []

            # only take paths that are valid:
            for g_i, g_p in enumerate(s_path):
                if exist_label[g_i]:
                    g_path.append(g_p)
        else:
            g_path, c_max = self.batch_get_graph_shortest_path(
                batch_start_idx, batch_goal_idx, return_length=True
            )
            len_max = max([len(g) for g in g_path])
            if len_max > 3:
                g_path, c_max = self.batch_shortcut_path(g_path, batch_start_idx, batch_goal_idx)
                len_max = max([len(g) for g in g_path])
        paths = self.get_paths(g_path)
        result.plan = paths
        result.success = exist_label

        # Debugging check:
        # if torch.count_nonzero(torch.as_tensor(result.success)) != len(paths):
        #    log_warn("Error here")

        result.path_length = torch.as_tensor(
            c_max, device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )
        return result

    @torch.no_grad()
    def _find_all_path(self, x_init_batch, x_goal_batch):
        """Find path from a batch of initial and goal configs

        Args:
            x_init ([type]): batch of start
            x_goal ([type]): batch of goal
            return_path_lengths (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: b, h, dof
        """

        result = GraphResult(
            start_q=x_init_batch,
            goal_q=x_goal_batch,
            success=[False for x in range(x_init_batch.shape[0])],
            path_length=self.tensor_args.to_device([inf for x in range(x_init_batch.shape[0])]),
        )
        # check if start and goal are same, if so just return false
        if self.i > (self.max_nodes * 0.75):
            self.reset_buffer()
        # add start and goal nodes to graph:
        node_set = torch.cat((x_init_batch.unsqueeze(1), x_goal_batch.unsqueeze(1)), dim=1)

        b, _, dof = node_set.shape
        node_set = node_set.view(b * 2, dof)
        # check if start and goal are in freespace:
        mask = self.mask_samples(node_set)
        if mask.all() != True:
            log_warn("Start or End state in collision", exc_info=False)
            node_set_batch = node_set.view(b, 2, node_set.shape[-1])
            result.plan = [node_set_batch[i, :, : self.dof] for i in range(node_set_batch.shape[0])]
            result.valid_query = False
            result.debug_info = "Start or End state in collision"
            return result
        node_set = self.add_nodes_to_graph(node_set, add_exact_node=True)
        node_set_batch = node_set.view(b, 2, node_set.shape[-1])
        if (
            torch.min(
                torch.abs(node_set_batch[:, 0, self.dof + 1] - node_set_batch[:, 1, self.dof + 1])
            )
            == 0.0
        ):
            # print("WARNING: Start and Goal are same")
            result.plan = [node_set_batch[i, :, : self.dof] for i in range(node_set_batch.shape[0])]
            return result
        self._add_bias_graph(x_init_batch, x_goal_batch, node_set_batch, node_set)

        batch_start_idx = (
            node_set_batch[:, 0, self.dof + 1].to(dtype=torch.int64, device="cpu").tolist()
        )
        batch_goal_idx = (
            node_set_batch[:, 1, self.dof + 1].to(dtype=torch.int64, device="cpu").tolist()
        )

        graph_attempt = 0
        path_exists, exist_label = self.batch_path_exists(
            batch_start_idx, batch_goal_idx, all_paths=True
        )
        k_nn = self.k_nn

        if path_exists:
            g_path, c_max = self.batch_get_graph_shortest_path(
                batch_start_idx, batch_goal_idx, return_length=True
            )
            len_max = max([len(g) for g in g_path])
            if len_max > 2:
                g_path, c_max = self.batch_shortcut_path(g_path, batch_start_idx, batch_goal_idx)
                len_max = max([len(g) for g in g_path])
                exist_label = [len(g) <= 3 for g in g_path]

            if len_max <= 2:
                paths = self.get_paths(g_path)
                result.plan = paths
                result.success = exist_label
                result.path_length = self.tensor_args.to_device(c_max)
                return result

        c_min = self.distance(x_init_batch, x_goal_batch).cpu().numpy()
        c_max = np.ravel([self.c_max * c_min[i] for i in range(x_init_batch.shape[0])])

        n_nodes = self.init_nodes
        # find paths
        idx = 0
        # print("Initial", path_exists, exist_label)
        while not path_exists or graph_attempt < (self.graph_min_attempts):
            if all(exist_label):
                no_path_label = exist_label
            else:
                no_path_label = [not x for x in exist_label]
            # choose x_init, x_goal from the ones that don't have a path:
            no_path_idx = np.where(no_path_label)[0].tolist()
            idx = random.choice(no_path_idx)
            self.build_graph(
                x_start=x_init_batch[idx],
                x_goal=x_goal_batch[idx],
                bias_samples=True,
                k_nn=k_nn,
                c_max=c_max[idx],
                c_min=c_min[idx],
                number_of_nodes=n_nodes,
                lazy_nodes=False,
            )
            graph_attempt += 1
            path_exists, exist_label = self.batch_path_exists(
                batch_start_idx, batch_goal_idx, all_paths=True
            )

            if path_exists:
                g_path, c_max = self.batch_get_graph_shortest_path(
                    batch_start_idx, batch_goal_idx, return_length=True
                )
                len_max = max([len(g) for g in g_path])
                if len_max > 2:
                    g_path, c_max = self.batch_shortcut_path(
                        g_path, batch_start_idx, batch_goal_idx
                    )
                    len_max = max([len(g) for g in g_path])
                exist_label = [len(g) <= 3 for g in g_path]

                if len_max <= 2:
                    break
            else:
                if graph_attempt == 1:
                    n_nodes = self.vertex_n
                c_max[idx] += c_min[idx] * 0.05

            k_nn += int(0.1 * k_nn)
            n_nodes += int(0.1 * n_nodes)
            if graph_attempt > self.graph_max_attempts:
                break
        path_exists, exist_label = self.batch_path_exists(
            batch_start_idx, batch_goal_idx, all_paths=True
        )

        if not path_exists:
            s_path = [[x, x] for x in batch_start_idx]
            c_max = [inf for _ in batch_start_idx]
            if any(exist_label):
                # do shortcut for only possible paths:
                # get true indices:
                idx_list = np.where(exist_label)[0].tolist()
                batch_start_idx = [batch_start_idx[x] for x in idx_list]
                batch_goal_idx = [batch_goal_idx[x] for x in idx_list]
                path_list = self.batch_get_graph_shortest_path(batch_start_idx, batch_goal_idx)

                path_list, c_list = self.batch_shortcut_path(
                    path_list, batch_start_idx, batch_goal_idx
                )
                # add this back
                for i, idx in enumerate(idx_list):
                    s_path[idx] = path_list[i]
                    c_max[idx] = c_list[i]
            g_path = []

            # only take paths that are valid:
            for g_i, g_p in enumerate(s_path):
                if exist_label[g_i]:
                    g_path.append(g_p)
            # g_path = s_path
        else:
            g_path, c_max = self.batch_get_graph_shortest_path(
                batch_start_idx, batch_goal_idx, return_length=True
            )
            len_max = max([len(g) for g in g_path])
            if len_max > 3:
                g_path, c_max = self.batch_shortcut_path(g_path, batch_start_idx, batch_goal_idx)
                len_max = max([len(g) for g in g_path])
        paths = self.get_paths(g_path)
        result.plan = paths
        result.success = exist_label
        result.path_length = torch.as_tensor(
            c_max, device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )
        return result

    def build_graph(
        self,
        x_start=None,
        x_goal=None,
        number_of_nodes=None,
        lazy=False,
        bias_samples=False,
        k_nn=5,
        c_max=10,
        c_min=1,
        lazy_nodes=False,
    ):
        # get samples to search in:
        dof = self.dof
        path = self.path
        lazy_samples = lazy or lazy_nodes
        # add few nodes to path:
        if number_of_nodes is None:
            number_of_nodes = self.vertex_n
        if x_start is None or x_goal is None:
            log_warn("Start and goal is not given, not using biased sampling")
            bias_samples = False
        # sample some points for vertex
        if bias_samples:
            v_set = self.get_biased_vertex_set(
                x_start=x_start,
                x_goal=x_goal,
                n=number_of_nodes,
                lazy=lazy_samples,
                c_max=c_max,
                c_min=c_min,
            )
        else:
            v_set = self.get_new_vertex_set(n=number_of_nodes, lazy=lazy_samples)
        number_of_nodes = v_set.shape[0]
        if not lazy_samples:
            if self.i + number_of_nodes >= path.shape[0]:
                raise ValueError(
                    "Path memory buffer is too small", path.shape[0], self.i + number_of_nodes
                )
            path[self.i : self.i + number_of_nodes, : dof + 1] = v_set
            for i in range(number_of_nodes):
                path[self.i + i, dof + 1] = self.i + i

            self.i = self.i + number_of_nodes
        sample_nodes = v_set[:, : self.dof]
        self.connect_nodes(sample_nodes, lazy=lazy, k_nn=k_nn)

    def warmup(self, x_start: Optional[torch.Tensor] = None, x_goal: Optional[torch.Tensor] = None):
        for _ in range(3):
            self.build_graph(
                x_start=x_start.view(-1),
                x_goal=x_goal.view(-1),
                bias_samples=True,
                k_nn=self.k_nn,
            )
        super().warmup()
