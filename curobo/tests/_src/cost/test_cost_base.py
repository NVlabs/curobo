# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for BaseCost and BaseCostCfg."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.cost.cost_base import BaseCost
from curobo._src.cost.cost_base_cfg import BaseCostCfg
from curobo._src.types.device_cfg import DeviceCfg


@pytest.fixture(params=["cpu", "cuda:0"])
def device_cfg(request):
    """Create tensor configuration for both CPU and GPU."""
    device = request.param
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return DeviceCfg(device=torch.device(device))


class TestBaseCostCfg:
    """Test BaseCostCfg class."""

    def test_init_with_float_weight(self, device_cfg):
        """Test initialization with float weight."""
        cfg = BaseCostCfg(weight=1.0, device_cfg=device_cfg)
        assert isinstance(cfg.weight, torch.Tensor)
        assert cfg.weight.shape == (1,)
        assert cfg.weight[0] == 1.0
        assert cfg.weight.device == device_cfg.device

    def test_init_with_list_weight(self, device_cfg):
        """Test initialization with list weight."""
        cfg = BaseCostCfg(weight=[1.0, 2.0, 3.0], device_cfg=device_cfg)
        assert isinstance(cfg.weight, torch.Tensor)
        assert cfg.weight.shape == (3,)
        assert torch.allclose(
            cfg.weight,
            torch.tensor([1.0, 2.0, 3.0], device=device_cfg.device, dtype=device_cfg.dtype),
        )

    def test_init_with_tensor_weight(self, device_cfg):
        """Test initialization with tensor weight."""
        weight = torch.tensor([1.0, 2.0], device=device_cfg.device, dtype=device_cfg.dtype)
        cfg = BaseCostCfg(weight=weight, device_cfg=device_cfg)
        assert cfg.weight.device == device_cfg.device
        assert cfg.weight.dtype == device_cfg.dtype

    def test_init_with_int_weight_raises_error(self, device_cfg):
        """Test that int weight raises error."""
        with pytest.raises(Exception):  # log_and_raise raises exception
            BaseCostCfg(weight=1, device_cfg=device_cfg)

    def test_default_class_type(self, device_cfg):
        """Test default class type."""
        cfg = BaseCostCfg(weight=1.0, device_cfg=device_cfg)
        assert cfg.class_type == BaseCost

    def test_default_convert_to_binary(self, device_cfg):
        """Test default convert_to_binary."""
        cfg = BaseCostCfg(weight=1.0, device_cfg=device_cfg)
        assert cfg.convert_to_binary is False

    def test_default_use_grad_input(self, device_cfg):
        """Test default use_grad_input."""
        cfg = BaseCostCfg(weight=1.0, device_cfg=device_cfg)
        assert cfg.use_grad_input is False

    def test_clone(self, device_cfg):
        """Test cloning configuration."""
        original = BaseCostCfg(
            weight=[1.0, 2.0],
            device_cfg=device_cfg,
            convert_to_binary=True,
        )
        cloned = original.clone()

        assert torch.allclose(cloned.weight, original.weight)
        assert cloned.convert_to_binary == original.convert_to_binary

        # Verify deep copy
        cloned.weight[0] = 100.0
        assert original.weight[0] == 1.0


class TestBaseCost:
    """Test BaseCost class."""

    def test_init(self, device_cfg):
        """Test basic initialization."""
        cfg = BaseCostCfg(weight=1.0, device_cfg=device_cfg)
        cost = BaseCost(cfg)
        assert cost.config == cfg
        assert cost.device_cfg == device_cfg
        assert cost._batch_size == -1
        assert cost._horizon == -1
        assert cost._dt == 1

    def test_setup_batch_tensors(self, device_cfg):
        """Test setup_batch_tensors."""
        cfg = BaseCostCfg(weight=1.0, device_cfg=device_cfg)
        cost = BaseCost(cfg)
        cost.setup_batch_tensors(batch_size=4, horizon=10)
        assert cost._batch_size == 4
        assert cost._horizon == 10

    def test_setup_batch_tensors_no_change_on_same_values(self, device_cfg):
        """Test that setup_batch_tensors only updates when values change."""
        cfg = BaseCostCfg(weight=1.0, device_cfg=device_cfg)
        cost = BaseCost(cfg)
        cost.setup_batch_tensors(batch_size=4, horizon=10)

        # Store current values
        batch_size = cost._batch_size
        horizon = cost._horizon

        # Call again with same values
        cost.setup_batch_tensors(batch_size=4, horizon=10)

        # Values should remain unchanged
        assert cost._batch_size == batch_size
        assert cost._horizon == horizon

    def test_disable_cost(self, device_cfg):
        """Test disabling cost."""
        cfg = BaseCostCfg(weight=[1.0, 2.0], device_cfg=device_cfg)
        cost = BaseCost(cfg)

        assert cost.enabled is True
        cost.disable_cost()
        assert cost.enabled is False
        assert torch.all(cost._weight == 0.0)

    def test_disable_cost_idempotent(self, device_cfg):
        """Test that disabling cost multiple times is idempotent."""
        cfg = BaseCostCfg(weight=1.0, device_cfg=device_cfg)
        cost = BaseCost(cfg)
        cost.disable_cost()
        cost.disable_cost()  # Should not raise
        assert cost.enabled is False

    def test_enable_cost(self, device_cfg):
        """Test enabling cost after disabling."""
        cfg = BaseCostCfg(weight=[1.0, 2.0], device_cfg=device_cfg)
        cost = BaseCost(cfg)

        cost.disable_cost()
        assert cost.enabled is False

        cost.enable_cost()
        assert cost.enabled is True
        assert torch.allclose(cost._weight, cfg.weight)

    def test_enable_cost_idempotent(self, device_cfg):
        """Test that enabling cost multiple times is idempotent."""
        cfg = BaseCostCfg(weight=1.0, device_cfg=device_cfg)
        cost = BaseCost(cfg)
        cost.enable_cost()  # Already enabled
        cost.enable_cost()  # Should not raise
        assert cost.enabled is True

    def test_enable_cost_with_zero_weight_stays_disabled(self, device_cfg):
        """Test that enable_cost with zero weight stays disabled."""
        cfg = BaseCostCfg(weight=0.0, device_cfg=device_cfg)
        cost = BaseCost(cfg)
        assert cost.enabled is False  # Zero weight starts disabled

        cost.enable_cost()
        assert cost.enabled is False  # Still disabled because weight is zero

    def test_update_dt(self, device_cfg):
        """Test update_dt method."""
        cfg = BaseCostCfg(weight=1.0, device_cfg=device_cfg)
        cost = BaseCost(cfg)
        assert cost._dt == 1

        cost.update_dt(0.01)
        assert cost._dt == 0.01

    def test_update_dt_with_tensor(self, device_cfg):
        """Test update_dt with tensor."""
        cfg = BaseCostCfg(weight=1.0, device_cfg=device_cfg)
        cost = BaseCost(cfg)

        dt = torch.tensor(0.02, device=device_cfg.device, dtype=device_cfg.dtype)
        cost.update_dt(dt)
        assert cost._dt == dt

    def test_reset(self, device_cfg):
        """Test reset method (base implementation does nothing)."""
        cfg = BaseCostCfg(weight=1.0, device_cfg=device_cfg)
        cost = BaseCost(cfg)
        # Should not raise
        cost.reset()
        cost.reset(reset_problem_ids=torch.tensor([0, 1], device=device_cfg.device))

    def test_forward_returns_zeros(self, device_cfg):
        """Test that forward returns zeros (base implementation)."""
        cfg = BaseCostCfg(weight=1.0, device_cfg=device_cfg)
        cost = BaseCost(cfg)
        cost.setup_batch_tensors(batch_size=2, horizon=5)
        result = cost.forward()
        assert result.shape == (2, 5, 1)
        assert torch.all(result == 0.0)

    def test_enabled_property(self, device_cfg):
        """Test enabled property."""
        cfg = BaseCostCfg(weight=1.0, device_cfg=device_cfg)
        cost = BaseCost(cfg)
        assert cost.enabled is True

    def test_init_with_zero_weight_disables_cost(self, device_cfg):
        """Test that zero weight disables cost on init."""
        cfg = BaseCostCfg(weight=0.0, device_cfg=device_cfg)
        cost = BaseCost(cfg)
        assert cost.enabled is False

    def test_init_with_zero_vector_weight_disables_cost(self, device_cfg):
        """Test that zero vector weight disables cost on init."""
        cfg = BaseCostCfg(weight=[0.0, 0.0, 0.0], device_cfg=device_cfg)
        cost = BaseCost(cfg)
        assert cost.enabled is False

    def test_init_with_mixed_weight_keeps_enabled(self, device_cfg):
        """Test that mixed weight (some non-zero) keeps cost enabled."""
        cfg = BaseCostCfg(weight=[0.0, 1.0, 0.0], device_cfg=device_cfg)
        cost = BaseCost(cfg)
        assert cost.enabled is True

    def test_weight_copied_from_config(self, device_cfg):
        """Test that weight is copied from config."""
        cfg = BaseCostCfg(weight=[1.0, 2.0], device_cfg=device_cfg)
        cost = BaseCost(cfg)

        # Modify cost weight
        cost._weight[0] = 100.0

        # Config weight should be unchanged
        assert cfg.weight[0] == 1.0

