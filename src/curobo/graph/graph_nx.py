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


# Third Party
import networkx as nx
import torch


class NetworkxGraph(object):
    def __init__(self):
        self.graph = nx.Graph()
        # maintain node buffer
        self.node_list = []
        # maintain edge buffer
        self.edge_list = []

    def reset_graph(self):
        self.graph.clear()
        self.edge_list = []
        self.node_list = []

    def add_node(self, i):
        self.node_list.append(i)

    def add_edges(self, edge_list):
        self.edge_list += edge_list

    def add_nodes(self, node_list):
        self.node_list += node_list

    def add_edge(self, start_i, end_i, weight):
        self.edge_list.append([start_i, end_i, weight])

    def update_graph(self):
        if len(self.edge_list) > 0:
            self.graph.add_weighted_edges_from(self.edge_list)
            self.edge_list = []
        if len(self.node_list) > 0:
            self.graph.add_nodes_from(self.node_list)
            self.node_list = []

    def get_edges(self, attribue="weight"):
        edge_list = list(self.graph.edges.data("weight"))
        return edge_list

    def path_exists(self, start_node_idx, goal_node_idx):
        self.update_graph()
        # check if nodes exist in the graph
        if self.graph.has_node(start_node_idx) and self.graph.has_node(goal_node_idx):
            return nx.has_path(self.graph, start_node_idx, goal_node_idx)
        else:
            return False

    def get_shortest_path(self, start_node_idx, goal_node_idx, return_length=False):
        self.update_graph()
        length, path = nx.bidirectional_dijkstra(
            self.graph, start_node_idx, goal_node_idx, weight="weight"
        )
        if return_length:
            return path, length
        return path

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
