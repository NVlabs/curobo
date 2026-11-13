# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Tests for configuration-space linear connection."""

# Standard Library
from types import SimpleNamespace

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.graph_planner.graph.connector_linear import LinearConnector


def _make_connector(cuda_device_cfg, feasibility_fn, *, steer_buffer_size=16):
    config = SimpleNamespace(
        device_cfg=cuda_device_cfg,
        cspace_similarity_threshold=0.05,
        steer_buffer_size=steer_buffer_size,
    )
    connector = LinearConnector(config)
    connector.set_dependencies(
        action_dim=2,
        cspace_distance_weight=torch.ones(2, **cuda_device_cfg.as_torch_dict()),
        check_feasibility_fn=feasibility_fn,
    )
    return connector


def test_linear_connector_returns_configuration_result(cuda_device_cfg):
    def always_feasible(samples, env_query_idx=None):
        return torch.ones(samples.shape[0], dtype=torch.bool, device=samples.device)

    connector = _make_connector(cuda_device_cfg, always_feasible)
    start = torch.zeros((2, 2), **cuda_device_cfg.as_torch_dict())
    goal = torch.tensor([[0.4, 0.0], [0.0, -0.2]], **cuda_device_cfg.as_torch_dict())

    result = connector.connect(start, goal, cspace_resolution=0.1)

    torch.testing.assert_close(result.endpoint, goal, rtol=0.0, atol=0.0)
    assert result.reached.tolist() == [True, True]
    assert result.progressed.tolist() == [True, True]


def test_linear_connector_stops_before_first_infeasible_sample(cuda_device_cfg):
    def feasible_before_limit(samples, env_query_idx=None):
        return samples[:, 0] < 0.7

    connector = _make_connector(cuda_device_cfg, feasible_before_limit)
    start = torch.zeros((1, 2), **cuda_device_cfg.as_torch_dict())
    goal = torch.tensor([[1.0, 0.0]], **cuda_device_cfg.as_torch_dict())

    result = connector.connect(start, goal, cspace_resolution=0.25)

    torch.testing.assert_close(
        result.endpoint,
        torch.tensor([[0.5, 0.0]], **cuda_device_cfg.as_torch_dict()),
        rtol=0.0,
        atol=0.0,
    )
    assert result.reached.tolist() == [False]
    assert result.progressed.tolist() == [True]


def test_linear_connector_does_not_report_progress_for_identical_endpoints(cuda_device_cfg):
    def always_feasible(samples, env_query_idx=None):
        return torch.ones(samples.shape[0], dtype=torch.bool, device=samples.device)

    connector = _make_connector(cuda_device_cfg, always_feasible)
    configuration = torch.zeros((1, 2), **cuda_device_cfg.as_torch_dict())

    result = connector.connect(configuration, configuration)

    torch.testing.assert_close(result.endpoint, configuration, rtol=0.0, atol=0.0)
    assert result.reached.tolist() == [True]
    assert result.progressed.tolist() == [False]


def test_linear_connector_routes_each_edge_environment(cuda_device_cfg):
    observed_environment_indices = []

    def environment_feasibility(samples, env_query_idx=None):
        observed_environment_indices.append(env_query_idx.clone())
        return env_query_idx == 1

    connector = _make_connector(cuda_device_cfg, environment_feasibility)
    start = torch.zeros((2, 2), **cuda_device_cfg.as_torch_dict())
    goal = torch.tensor([[0.2, 0.0], [0.2, 0.0]], **cuda_device_cfg.as_torch_dict())
    environment_indices = torch.tensor([0, 1], device=cuda_device_cfg.device, dtype=torch.int32)

    result = connector.connect(
        start,
        goal,
        env_query_idx=environment_indices,
        cspace_resolution=0.1,
    )

    assert result.reached.tolist() == [False, True]
    assert result.progressed.tolist() == [False, True]
    assert len(observed_environment_indices) == 1
    assert observed_environment_indices[0].tolist() == [0, 0, 0, 1, 1, 1]


def test_linear_connector_rejects_insufficient_interpolation_capacity(cuda_device_cfg):
    def always_feasible(samples, env_query_idx=None):
        return torch.ones(samples.shape[0], dtype=torch.bool, device=samples.device)

    connector = _make_connector(cuda_device_cfg, always_feasible, steer_buffer_size=4)
    start = torch.zeros((1, 2), **cuda_device_cfg.as_torch_dict())
    goal = torch.tensor([[1.0, 0.0]], **cuda_device_cfg.as_torch_dict())

    with pytest.raises(ValueError, match="steer_buffer_size"):
        connector.connect(start, goal, cspace_resolution=0.1)


def test_linear_connector_rejects_graph_node_metadata(cuda_device_cfg):
    def always_feasible(samples, env_query_idx=None):
        return torch.ones(samples.shape[0], dtype=torch.bool, device=samples.device)

    connector = _make_connector(cuda_device_cfg, always_feasible)
    graph_nodes = torch.zeros((1, 3), **cuda_device_cfg.as_torch_dict())

    with pytest.raises(ValueError, match="action_dim"):
        connector.connect(graph_nodes, graph_nodes)
