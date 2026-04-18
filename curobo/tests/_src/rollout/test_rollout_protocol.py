# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Rollout protocol and re-exported data types."""

from __future__ import annotations

import pytest
import torch

from curobo._src.rollout.metrics import (
    CostCollection,
    CostsAndConstraints,
    RolloutMetrics,
    RolloutResult,
)
from curobo._src.rollout.rollout_protocol import Rollout
from curobo._src.rollout.rollout_rosenbrock import RosenbrockCfg, RosenbrockRollout
from curobo._src.types.device_cfg import DeviceCfg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def device_cfg():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return DeviceCfg(device="cuda:0")


@pytest.fixture
def rosenbrock(device_cfg):
    cfg = RosenbrockCfg(device_cfg=device_cfg)
    return RosenbrockRollout(cfg)


@pytest.fixture
def rosenbrock_cuda_graph(device_cfg):
    cfg = RosenbrockCfg(device_cfg=device_cfg)
    return RosenbrockRollout(cfg, use_cuda_graph=True)


# ---------------------------------------------------------------------------
# Protocol compliance: isinstance checks
# ---------------------------------------------------------------------------

class TestProtocolCompliance:
    """Verify concrete rollout classes satisfy the Rollout protocol."""

    def test_rosenbrock_satisfies_protocol(self, rosenbrock):
        assert isinstance(rosenbrock, Rollout)

    def test_rosenbrock_cuda_graph_satisfies_protocol(self, rosenbrock_cuda_graph):
        assert isinstance(rosenbrock_cuda_graph, Rollout)

    def test_robot_rollout_satisfies_protocol(self, device_cfg):
        """RobotRollout requires a full config with transition model.

        We check class-level methods/properties. Instance attributes like
        ``sum_horizon`` are set in ``__init__`` and verified via the
        integration tests that instantiate with real configs.
        """
        from curobo._src.rollout.rollout_robot import RobotRollout

        # Methods and descriptors visible on the class
        required_methods = [
            "action_dim", "action_horizon", "action_bound_lows",
            "action_bound_highs", "dt",
            "evaluate_action", "compute_metrics_from_state",
            "compute_metrics_from_action", "update_params",
            "update_batch_size", "update_dt", "reset",
            "reset_shape", "reset_seed",
        ]
        for attr in required_methods:
            assert hasattr(RobotRollout, attr), f"RobotRollout missing {attr}"
        # sum_horizon is an instance attribute set in __init__, not a class descriptor.
        # Protocol isinstance() checks work on instances (verified by integration tests).

    def test_plain_object_does_not_satisfy_protocol(self):
        assert not isinstance(object(), Rollout)

    def test_partial_class_does_not_satisfy_protocol(self):
        class Incomplete:
            @property
            def action_dim(self):
                return 2
        assert not isinstance(Incomplete(), Rollout)


# ---------------------------------------------------------------------------
# Re-exported data types
# ---------------------------------------------------------------------------

class TestReExports:
    """Verify data types are re-exported from rollout_protocol."""

    def test_rollout_result_reexport(self):
        from curobo._src.rollout.rollout_protocol import RolloutResult as RR
        assert RR is RolloutResult

    def test_rollout_metrics_reexport(self):
        from curobo._src.rollout.rollout_protocol import RolloutMetrics as RM
        assert RM is RolloutMetrics

    def test_costs_and_constraints_reexport(self):
        from curobo._src.rollout.rollout_protocol import CostsAndConstraints as CC
        assert CC is CostsAndConstraints

    def test_cost_collection_reexport(self):
        from curobo._src.rollout.rollout_protocol import CostCollection as CC
        assert CC is CostCollection


# ---------------------------------------------------------------------------
# Protocol methods: functional on a real instance
# ---------------------------------------------------------------------------

class TestProtocolMethods:
    """Verify protocol methods work on a concrete Rollout instance."""

    def test_evaluate_action(self, rosenbrock, device_cfg):
        act = torch.randn(2, 1, 2, device=device_cfg.device, dtype=device_cfg.dtype)
        result = rosenbrock.evaluate_action(act)
        assert isinstance(result, RolloutResult)
        assert result.costs_and_constraints is not None

    def test_compute_metrics_from_action(self, rosenbrock, device_cfg):
        act = torch.randn(2, 1, 2, device=device_cfg.device, dtype=device_cfg.dtype)
        metrics = rosenbrock.compute_metrics_from_action(act)
        assert isinstance(metrics, RolloutMetrics)
        assert metrics.feasible is not None

    def test_compute_metrics_from_state(self, rosenbrock, device_cfg):
        from curobo._src.state.state_joint import JointState
        pos = torch.randn(2, 1, 2, device=device_cfg.device, dtype=device_cfg.dtype)
        state = JointState.from_position(pos)
        metrics = rosenbrock.compute_metrics_from_state(state)
        assert isinstance(metrics, RolloutMetrics)

    def test_properties(self, rosenbrock):
        assert rosenbrock.action_dim == 2
        assert rosenbrock.action_horizon == 1
        assert isinstance(rosenbrock.action_bound_lows, torch.Tensor)
        assert isinstance(rosenbrock.action_bound_highs, torch.Tensor)
        assert isinstance(rosenbrock.dt, float)
        assert isinstance(rosenbrock.sum_horizon, bool)

    def test_lifecycle_methods(self, rosenbrock):
        assert rosenbrock.update_params() is True
        rosenbrock.update_batch_size(4)
        assert rosenbrock.batch_size == 4
        assert rosenbrock.reset() is True
        assert rosenbrock.reset_shape() is True
        rosenbrock.reset_seed()  # no return value to check
