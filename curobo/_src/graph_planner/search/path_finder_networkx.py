# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#


# Third Party
import random

import networkx as nx
import numpy as np
import torch
from torch import profiler


class NetworkXPathFinder:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.graph = nx.Graph(seed=self.seed)
        # maintain node buffer
        self.node_list = []
        # maintain edge buffer
        self.edge_list = []

    @profiler.record_function("networkx_path_finder/reset_graph")
    def reset_graph(self):
        self.graph.clear()
        self.edge_list = []
        self.node_list = []

    def reset_seed(self):
        """Reset seed to make planning deterministic.

        NOTE: This resets the global seed for numpy and random. This is not a good idea,
        but networkx currently does not allow for resetting it's seed for graphs.

        """
        np.random.seed(self.seed)
        random.seed(self.seed)

    @profiler.record_function("networkx_path_finder/add_node")
    def add_node(self, i):
        self.node_list.append(i)

    @profiler.record_function("networkx_path_finder/add_edges")
    def add_edges(self, edge_list):
        self.edge_list += edge_list

    @profiler.record_function("networkx_path_finder/add_nodes")
    def add_nodes(self, node_list):
        self.node_list += node_list

    @profiler.record_function("networkx_path_finder/add_edge")
    def add_edge(self, start_i, end_i, weight):
        self.edge_list.append([start_i, end_i, weight])

    @profiler.record_function("networkx_path_finder/update_graph")
    def update_graph(self):
        if len(self.edge_list) > 0:
            self.graph.add_weighted_edges_from(self.edge_list)
            self.edge_list = []
        if len(self.node_list) > 0:
            self.graph.add_nodes_from(self.node_list)
            self.node_list = []

    @profiler.record_function("networkx_path_finder/get_edges")
    def get_edges(self, attribue="weight"):
        edge_list = list(self.graph.edges.data("weight"))
        return edge_list

    @profiler.record_function("networkx_path_finder/path_exists")
    def path_exists(self, start_node_idx, goal_node_idx):
        self.update_graph()
        # check if nodes exist in the graph
        if self.graph.has_node(start_node_idx) and self.graph.has_node(goal_node_idx):
            return nx.has_path(self.graph, start_node_idx, goal_node_idx)
        else:
            return False

    @profiler.record_function("networkx_path_finder/get_shortest_path")
    def get_shortest_path(self, start_node_idx, goal_node_idx, return_length=False):
        self.update_graph()
        length, path = nx.bidirectional_dijkstra(
            self.graph, start_node_idx, goal_node_idx, weight="weight"
        )
        if return_length:
            return path, length
        return path

    @profiler.record_function("networkx_path_finder/get_path_lengths")
    def get_path_lengths(self, goal_node_idx):
        self.update_graph()
        path_length_dict = nx.shortest_path_length(
            self.graph, source=goal_node_idx, weight="weight"
        )
        dict_keys = list(path_length_dict.keys())
        max_n = self.graph.number_of_nodes()

        max_n = max(dict_keys) + 1

        path_lengths = [-1.0 for x in range(max_n)]
        for i in range(len(dict_keys)):
            k = dict_keys[i]
            # print(i,k, max_n)
            if k >= max_n:
                print(k, max_n)
                continue
            path_lengths[k] = path_length_dict[k]
        path_lengths = torch.as_tensor(path_lengths)
        return path_lengths.cpu().tolist()
