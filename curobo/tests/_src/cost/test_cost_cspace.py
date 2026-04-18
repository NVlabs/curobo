# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for CSpaceCostCfg and cspace cost classes."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.cost.cost_cspace_cfg import CSpaceCostCfg
from curobo._src.cost.cost_cspace_position import PositionCSpaceCost
from curobo._src.cost.cost_cspace_state import StateCSpaceCost
from curobo._src.cost.cost_cspace_type import CSpaceCostType
from curobo._src.robot.types.joint_limits import JointLimits
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg


@pytest.fixture(params=["cuda:0"])
def device_cfg(request):
    """Create tensor configuration for GPU only (warp requires CUDA)."""
    device = request.param
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return DeviceCfg(device=torch.device(device))


@pytest.fixture
def joint_limits(device_cfg):
    """Create valid joint limits for testing."""
    dof = 7
    joint_names = [f"joint_{i}" for i in range(dof)]
    position = torch.tensor(
        [[-3.14] * dof, [3.14] * dof], **device_cfg.as_torch_dict()
    )
    velocity = torch.tensor(
        [[-2.0] * dof, [2.0] * dof], **device_cfg.as_torch_dict()
    )
    acceleration = torch.tensor(
        [[-10.0] * dof, [10.0] * dof], **device_cfg.as_torch_dict()
    )
    jerk = torch.tensor(
        [[-100.0] * dof, [100.0] * dof], **device_cfg.as_torch_dict()
    )
    effort = torch.tensor(
        [[-50.0] * dof, [50.0] * dof], **device_cfg.as_torch_dict()
    )
    return JointLimits(
        joint_names=joint_names,
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        jerk=jerk,
        effort=effort,
        device_cfg=device_cfg,
    )


class TestCSpaceCostCfg:
    """Test CSpaceCostCfg class."""

    def test_position_cost_type_init(self, device_cfg, joint_limits):
        """Test initialization with POSITION cost type."""
        cfg = CSpaceCostCfg(
            weight=[1.0, 1.0],  # position, torque
            activation_distance=[0.01, 0.01],
            cost_type=CSpaceCostType.POSITION,
            dof=7,
            joint_limits=joint_limits,
            device_cfg=device_cfg,
        )
        assert cfg.cost_type == CSpaceCostType.POSITION
        assert cfg.class_type == PositionCSpaceCost

    def test_state_cost_type_init(self, device_cfg, joint_limits):
        """Test initialization with STATE cost type."""
        cfg = CSpaceCostCfg(
            weight=[1.0, 1.0, 1.0, 1.0, 1.0],  # position, velocity, acceleration, jerk, torque
            activation_distance=[0.01, 0.01, 0.01, 0.01, 0.01],
            cost_type=CSpaceCostType.STATE,
            dof=7,
            joint_limits=joint_limits,
            device_cfg=device_cfg,
        )
        assert cfg.cost_type == CSpaceCostType.STATE
        assert cfg.class_type == StateCSpaceCost

    def test_cost_type_from_string(self, device_cfg, joint_limits):
        """Test that cost_type can be initialized from string."""
        cfg = CSpaceCostCfg(
            weight=[1.0, 1.0],
            activation_distance=[0.01, 0.01],
            cost_type="POSITION",  # String instead of enum
            dof=7,
            joint_limits=joint_limits,
            device_cfg=device_cfg,
        )
        assert cfg.cost_type == CSpaceCostType.POSITION

    def test_none_cost_type_raises_error(self, device_cfg, joint_limits):
        """Test that None cost_type raises error."""
        with pytest.raises(Exception):
            CSpaceCostCfg(
                weight=[1.0, 1.0],
                activation_distance=[0.01, 0.01],
                cost_type=None,
                dof=7,
                joint_limits=joint_limits,
                device_cfg=device_cfg,
            )

    def test_position_wrong_weight_length_raises_error(self, device_cfg, joint_limits):
        """Test that wrong weight length raises error for POSITION."""
        with pytest.raises(Exception):
            CSpaceCostCfg(
                weight=[1.0, 1.0, 1.0],  # Should be 2 for POSITION
                activation_distance=[0.01, 0.01],
                cost_type=CSpaceCostType.POSITION,
                dof=7,
                joint_limits=joint_limits,
                device_cfg=device_cfg,
            )

    def test_position_wrong_activation_distance_length_raises_error(self, device_cfg, joint_limits):
        """Test that wrong activation_distance length raises error for POSITION."""
        with pytest.raises(Exception):
            CSpaceCostCfg(
                weight=[1.0, 1.0],
                activation_distance=[0.01, 0.01, 0.01],  # Should be 2 for POSITION
                cost_type=CSpaceCostType.POSITION,
                dof=7,
                joint_limits=joint_limits,
                device_cfg=device_cfg,
            )

    def test_state_wrong_weight_length_raises_error(self, device_cfg, joint_limits):
        """Test that wrong weight length raises error for STATE."""
        with pytest.raises(Exception):
            CSpaceCostCfg(
                weight=[1.0, 1.0, 1.0],  # Should be 5 for STATE
                activation_distance=[0.01, 0.01, 0.01, 0.01, 0.01],
                cost_type=CSpaceCostType.STATE,
                dof=7,
                joint_limits=joint_limits,
                device_cfg=device_cfg,
            )

    def test_state_with_regularization_weight(self, device_cfg, joint_limits):
        """Test STATE with squared_l2_regularization_weight."""
        cfg = CSpaceCostCfg(
            weight=[1.0, 1.0, 1.0, 1.0, 1.0],
            activation_distance=[0.01, 0.01, 0.01, 0.01, 0.01],
            cost_type=CSpaceCostType.STATE,
            dof=7,
            joint_limits=joint_limits,
            squared_l2_regularization_weight=[0.1, 0.1, 0.1, 0.1, 0.1],
            device_cfg=device_cfg,
        )
        assert cfg.squared_l2_regularization_weight is not None
        assert len(cfg.squared_l2_regularization_weight) == 5

    def test_cspace_target_weight_float(self, device_cfg, joint_limits):
        """Test cspace_target_weight as float."""
        cfg = CSpaceCostCfg(
            weight=[1.0, 1.0],
            activation_distance=[0.01, 0.01],
            cost_type=CSpaceCostType.POSITION,
            dof=7,
            joint_limits=joint_limits,
            cspace_target_weight=0.5,
            device_cfg=device_cfg,
        )
        assert isinstance(cfg.cspace_target_weight, torch.Tensor)
        assert cfg.cspace_target_weight[0] == 0.5

    def test_cspace_target_weight_list(self, device_cfg, joint_limits):
        """Test cspace_target_weight as list."""
        cfg = CSpaceCostCfg(
            weight=[1.0, 1.0],
            activation_distance=[0.01, 0.01],
            cost_type=CSpaceCostType.POSITION,
            dof=7,
            joint_limits=joint_limits,
            cspace_target_weight=[0.5],
            device_cfg=device_cfg,
        )
        assert isinstance(cfg.cspace_target_weight, torch.Tensor)
        assert cfg.cspace_target_weight[0] == 0.5

    def test_set_bounds(self, device_cfg, joint_limits):
        """Test set_bounds method."""
        cfg = CSpaceCostCfg(
            weight=[1.0, 1.0],
            activation_distance=[0.01, 0.01],
            cost_type=CSpaceCostType.POSITION,
            dof=7,
            device_cfg=device_cfg,
        )
        cfg.set_bounds(joint_limits)
        assert cfg.joint_limits is not None
        assert cfg.joint_limits.position.shape == (2, 7)

    def test_set_bounds_teleport_mode_converts_state_to_position(self, device_cfg, joint_limits):
        """Test that set_bounds with teleport_mode converts STATE to POSITION."""
        cfg = CSpaceCostCfg(
            weight=[1.0, 1.0, 1.0, 1.0, 1.0],
            activation_distance=[0.01, 0.01, 0.01, 0.01, 0.01],
            cost_type=CSpaceCostType.STATE,
            dof=7,
            device_cfg=device_cfg,
        )
        cfg.set_bounds(joint_limits, teleport_mode=True)
        assert cfg.cost_type == CSpaceCostType.POSITION
        assert cfg.class_type == PositionCSpaceCost
        assert len(cfg.weight) == 2  # Reduced from 5

    def test_update_dof(self, device_cfg, joint_limits):
        """Test update_dof method."""
        cfg = CSpaceCostCfg(
            weight=[1.0, 1.0],
            activation_distance=[0.01, 0.01],
            cost_type=CSpaceCostType.POSITION,
            dof=7,
            joint_limits=joint_limits,
            device_cfg=device_cfg,
        )
        # Update dof
        cfg.update_dof(10)
        assert cfg.dof == 10

    def test_activation_distance_float_raises_error(self, device_cfg, joint_limits):
        """Test that float activation_distance raises error."""
        with pytest.raises(ValueError):
            CSpaceCostCfg(
                weight=[1.0, 1.0],
                activation_distance=0.01,  # Should be list
                cost_type=CSpaceCostType.POSITION,
                dof=7,
                joint_limits=joint_limits,
                device_cfg=device_cfg,
            )


class TestPositionCSpaceCost:
    """Test PositionCSpaceCost class."""

    def test_init(self, device_cfg, joint_limits):
        """Test PositionCSpaceCost initialization."""
        cfg = CSpaceCostCfg(
            weight=[1.0, 1.0],
            activation_distance=[0.01, 0.01],
            cost_type=CSpaceCostType.POSITION,
            dof=7,
            joint_limits=joint_limits,
            device_cfg=device_cfg,
        )
        cost = PositionCSpaceCost(cfg)
        assert cost is not None
        assert cost.config == cfg

    def test_init_wrong_cost_type_raises_error(self, device_cfg, joint_limits):
        """Test that wrong cost type raises error."""
        cfg = CSpaceCostCfg(
            weight=[1.0, 1.0, 1.0, 1.0, 1.0],
            activation_distance=[0.01, 0.01, 0.01, 0.01, 0.01],
            cost_type=CSpaceCostType.STATE,  # Wrong type for PositionCSpaceCost
            dof=7,
            joint_limits=joint_limits,
            device_cfg=device_cfg,
        )
        with pytest.raises(Exception):
            PositionCSpaceCost(cfg)

    def test_setup_batch_tensors(self, device_cfg, joint_limits):
        """Test setup_batch_tensors."""
        cfg = CSpaceCostCfg(
            weight=[1.0, 1.0],
            activation_distance=[0.01, 0.01],
            cost_type=CSpaceCostType.POSITION,
            dof=7,
            joint_limits=joint_limits,
            device_cfg=device_cfg,
        )
        cost = PositionCSpaceCost(cfg)
        cost.setup_batch_tensors(batch=4, horizon=10)
        assert cost._batch_size == 4
        assert cost._horizon == 10
        assert cost._out_c_buffer.shape == (4, 10, 7)
        assert cost._out_gp_buffer.shape == (4, 10, 7)
        assert cost._out_gtau_buffer.shape == (4, 10, 7)

    def test_forward(self, device_cfg, joint_limits):
        """Test forward pass."""
        cfg = CSpaceCostCfg(
            weight=[1.0, 1.0],
            activation_distance=[0.01, 0.01],
            cost_type=CSpaceCostType.POSITION,
            dof=7,
            joint_limits=joint_limits,
            device_cfg=device_cfg,
        )
        cost = PositionCSpaceCost(cfg)
        batch_size = 4
        horizon = 10
        dof = 7
        cost.setup_batch_tensors(batch=batch_size, horizon=horizon)

        # Create valid joint state within bounds
        position = torch.zeros((batch_size, horizon, dof), **device_cfg.as_torch_dict())
        state = JointState(position=position)

        result = cost.forward(state)
        # Output is per-DOF cost: (batch_size, horizon, dof)
        assert result.shape == (batch_size, horizon, dof)

    def test_forward_with_out_of_bounds_position(self, device_cfg, joint_limits):
        """Test forward pass with out-of-bounds position."""
        cfg = CSpaceCostCfg(
            weight=[1.0, 1.0],
            activation_distance=[0.01, 0.01],
            cost_type=CSpaceCostType.POSITION,
            dof=7,
            joint_limits=joint_limits,
            device_cfg=device_cfg,
        )
        cost = PositionCSpaceCost(cfg)
        batch_size = 4
        horizon = 10
        dof = 7
        cost.setup_batch_tensors(batch=batch_size, horizon=horizon)

        # Create out-of-bounds position
        position = torch.full((batch_size, horizon, dof), 5.0, **device_cfg.as_torch_dict())
        state = JointState(position=position)

        result = cost.forward(state)
        # Output is per-DOF cost: (batch_size, horizon, dof)
        assert result.shape == (batch_size, horizon, dof)
        # Out of bounds should have non-zero cost
        assert torch.any(result > 0)


class TestStateCSpaceCost:
    """Test StateCSpaceCost class."""

    def test_init(self, device_cfg, joint_limits):
        """Test StateCSpaceCost initialization."""
        cfg = CSpaceCostCfg(
            weight=[1.0, 1.0, 1.0, 1.0, 1.0],
            activation_distance=[0.01, 0.01, 0.01, 0.01, 0.01],
            cost_type=CSpaceCostType.STATE,
            dof=7,
            joint_limits=joint_limits,
            device_cfg=device_cfg,
        )
        cost = StateCSpaceCost(cfg)
        assert cost is not None

    def test_init_wrong_cost_type_raises_error(self, device_cfg, joint_limits):
        """Test that wrong cost type raises error."""
        cfg = CSpaceCostCfg(
            weight=[1.0, 1.0],
            activation_distance=[0.01, 0.01],
            cost_type=CSpaceCostType.POSITION,  # Wrong type
            dof=7,
            joint_limits=joint_limits,
            device_cfg=device_cfg,
        )
        with pytest.raises(Exception):
            StateCSpaceCost(cfg)

    def test_setup_batch_tensors(self, device_cfg, joint_limits):
        """Test setup_batch_tensors."""
        cfg = CSpaceCostCfg(
            weight=[1.0, 1.0, 1.0, 1.0, 1.0],
            activation_distance=[0.01, 0.01, 0.01, 0.01, 0.01],
            cost_type=CSpaceCostType.STATE,
            dof=7,
            joint_limits=joint_limits,
            device_cfg=device_cfg,
        )
        cost = StateCSpaceCost(cfg)
        cost.setup_batch_tensors(batch=4, horizon=10)
        assert cost._batch_size == 4
        assert cost._horizon == 10
        assert cost._out_c_buffer.shape == (4, 10, 7)
        assert cost._out_gp_buffer.shape == (4, 10, 7)
        assert cost._out_gv_buffer.shape == (4, 10, 7)
        assert cost._out_ga_buffer.shape == (4, 10, 7)
        assert cost._out_gj_buffer.shape == (4, 10, 7)
        assert cost._out_gtau_buffer.shape == (4, 10, 7)

    def test_forward(self, device_cfg, joint_limits):
        """Test forward pass."""
        cfg = CSpaceCostCfg(
            weight=[1.0, 1.0, 1.0, 1.0, 1.0],
            activation_distance=[0.01, 0.01, 0.01, 0.01, 0.01],
            cost_type=CSpaceCostType.STATE,
            dof=7,
            joint_limits=joint_limits,
            device_cfg=device_cfg,
        )
        cost = StateCSpaceCost(cfg)
        batch_size = 4
        horizon = 10
        dof = 7
        cost.setup_batch_tensors(batch=batch_size, horizon=horizon)

        # Create valid joint state
        dt = torch.ones((batch_size,), **device_cfg.as_torch_dict()) * 0.01
        state = JointState(
            position=torch.zeros((batch_size, horizon, dof), **device_cfg.as_torch_dict()),
            velocity=torch.zeros((batch_size, horizon, dof), **device_cfg.as_torch_dict()),
            acceleration=torch.zeros((batch_size, horizon, dof), **device_cfg.as_torch_dict()),
            jerk=torch.zeros((batch_size, horizon, dof), **device_cfg.as_torch_dict()),
            dt=dt,
        )

        result = cost.forward(state)
        # Output is per-DOF cost: (batch_size, horizon, dof)
        assert result.shape == (batch_size, horizon, dof)

    def test_forward_with_regularization(self, device_cfg, joint_limits):
        """Test forward pass with regularization weights."""
        cfg = CSpaceCostCfg(
            weight=[1.0, 1.0, 1.0, 1.0, 1.0],
            activation_distance=[0.01, 0.01, 0.01, 0.01, 0.01],
            cost_type=CSpaceCostType.STATE,
            dof=7,
            joint_limits=joint_limits,
            squared_l2_regularization_weight=[0.1, 0.1, 0.1, 0.1, 0.1],
            device_cfg=device_cfg,
        )
        cost = StateCSpaceCost(cfg)
        batch_size = 4
        horizon = 10
        dof = 7
        cost.setup_batch_tensors(batch=batch_size, horizon=horizon)

        # Create valid joint state
        dt = torch.ones((batch_size,), **device_cfg.as_torch_dict()) * 0.01
        state = JointState(
            position=torch.zeros((batch_size, horizon, dof), **device_cfg.as_torch_dict()),
            velocity=torch.zeros((batch_size, horizon, dof), **device_cfg.as_torch_dict()),
            acceleration=torch.zeros((batch_size, horizon, dof), **device_cfg.as_torch_dict()),
            jerk=torch.zeros((batch_size, horizon, dof), **device_cfg.as_torch_dict()),
            dt=dt,
        )

        result = cost.forward(state)
        # Output is per-DOF cost: (batch_size, horizon, dof)
        assert result.shape == (batch_size, horizon, dof)


class TestCSpaceCostGradients:
    """Test gradient computation for cspace cost functions.

    Note: use_grad_input=True is required for proper gradient computation when
    the cost output is used in further computations (not just summed).
    """

    def test_position_cspace_cost_gradient(self, device_cfg, joint_limits):
        """Test gradient computation for PositionCSpaceCost w.r.t. position."""
        cfg = CSpaceCostCfg(
            weight=[1.0, 1.0],
            activation_distance=[0.01, 0.01],
            cost_type=CSpaceCostType.POSITION,
            dof=7,
            joint_limits=joint_limits,
            use_grad_input=True,  # Required for gradient computation
            device_cfg=device_cfg,
        )
        cost = PositionCSpaceCost(cfg)
        batch_size = 4
        horizon = 10
        dof = 7
        cost.setup_batch_tensors(batch=batch_size, horizon=horizon)

        # Create position with requires_grad=True
        position = torch.randn(
            (batch_size, horizon, dof), dtype=torch.float32, device=device_cfg.device, requires_grad=True
        )
        state = JointState(position=position)

        # Forward pass
        result = cost.forward(state)

        # Backward pass
        loss = result.sum()
        loss.backward()

        # Verify gradients exist and are finite
        assert position.grad is not None, "Gradient should not be None"
        assert position.grad.shape == position.shape, "Gradient shape mismatch"
        assert torch.isfinite(position.grad).all(), "Gradient contains non-finite values"

    def test_position_cspace_cost_gradcheck(self, device_cfg, joint_limits):
        """Test numerical gradient check for PositionCSpaceCost."""
        cfg = CSpaceCostCfg(
            weight=[1.0, 1.0],
            activation_distance=[0.01, 0.01],
            cost_type=CSpaceCostType.POSITION,
            dof=7,
            joint_limits=joint_limits,
            use_grad_input=True,  # Required for gradient computation
            device_cfg=device_cfg,
        )
        cost = PositionCSpaceCost(cfg)
        batch_size = 2
        horizon = 3
        dof = 7
        cost.setup_batch_tensors(batch=batch_size, horizon=horizon)

        # Define wrapper function for gradcheck
        def cost_fn(pos):
            state = JointState(position=pos)
            result = cost.forward(state)
            return result.sum()

        # Create position tensor (use float32 as requested)
        position = torch.randn(
            (batch_size, horizon, dof), dtype=torch.float32, device=device_cfg.device, requires_grad=True
        )

        # Numerical gradient check with relaxed tolerances for float32
        assert torch.autograd.gradcheck(
            cost_fn, position, eps=1e-3, atol=1e-2, rtol=1e-2, raise_exception=True
        ), "Gradient check failed for PositionCSpaceCost"

    def test_state_cspace_cost_gradient(self, device_cfg, joint_limits):
        """Test gradient computation for StateCSpaceCost."""
        cfg = CSpaceCostCfg(
            weight=[1.0, 1.0, 1.0, 1.0, 1.0],
            activation_distance=[0.01, 0.01, 0.01, 0.01, 0.01],
            cost_type=CSpaceCostType.STATE,
            dof=7,
            joint_limits=joint_limits,
            squared_l2_regularization_weight=[0.1, 0.1, 0.1, 0.1, 0.1],
            use_grad_input=True,  # Required for gradient computation
            device_cfg=device_cfg,
        )
        cost = StateCSpaceCost(cfg)
        batch_size = 4
        horizon = 10
        dof = 7
        cost.setup_batch_tensors(batch=batch_size, horizon=horizon)

        # Create state tensors with requires_grad=True
        position = torch.randn(
            (batch_size, horizon, dof), dtype=torch.float32, device=device_cfg.device, requires_grad=True
        )
        velocity = torch.randn(
            (batch_size, horizon, dof), dtype=torch.float32, device=device_cfg.device, requires_grad=True
        )
        acceleration = torch.randn(
            (batch_size, horizon, dof), dtype=torch.float32, device=device_cfg.device, requires_grad=True
        )
        jerk = torch.randn(
            (batch_size, horizon, dof), dtype=torch.float32, device=device_cfg.device, requires_grad=True
        )
        dt = torch.ones((batch_size,), **device_cfg.as_torch_dict()) * 0.01
        state = JointState(position=position, velocity=velocity, acceleration=acceleration, jerk=jerk, dt=dt)

        # Forward pass
        result = cost.forward(state)

        # Backward pass
        loss = result.sum()
        loss.backward()

        # Verify gradients exist and are finite
        assert position.grad is not None, "Position gradient should not be None"
        assert torch.isfinite(position.grad).all(), "Position gradient contains non-finite values"
        assert velocity.grad is not None, "Velocity gradient should not be None"
        assert torch.isfinite(velocity.grad).all(), "Velocity gradient contains non-finite values"

    def test_state_cspace_cost_gradcheck(self, device_cfg, joint_limits):
        """Test numerical gradient check for StateCSpaceCost."""
        cfg = CSpaceCostCfg(
            weight=[1.0, 1.0, 1.0, 1.0, 1.0],
            activation_distance=[0.01, 0.01, 0.01, 0.01, 0.01],
            cost_type=CSpaceCostType.STATE,
            dof=7,
            joint_limits=joint_limits,
            squared_l2_regularization_weight=[0.1, 0.1, 0.1, 0.1, 0.1],
            use_grad_input=True,  # Required for gradient computation
            device_cfg=device_cfg,
        )
        cost = StateCSpaceCost(cfg)
        batch_size = 2
        horizon = 3
        dof = 7
        cost.setup_batch_tensors(batch=batch_size, horizon=horizon)

        # Define wrapper function for gradcheck
        dt = torch.ones((batch_size,), **device_cfg.as_torch_dict()) * 0.01

        def cost_fn(pos):
            state = JointState(
                position=pos,
                velocity=torch.zeros_like(pos),
                acceleration=torch.zeros_like(pos),
                jerk=torch.zeros_like(pos),
                dt=dt,
            )
            result = cost.forward(state)
            return result.sum()

        # Create position tensor
        position = torch.randn(
            (batch_size, horizon, dof), dtype=torch.float32, device=device_cfg.device, requires_grad=True
        )

        # Numerical gradient check with relaxed tolerances for float32
        assert torch.autograd.gradcheck(
            cost_fn, position, eps=1e-3, atol=1e-2, rtol=1e-2, raise_exception=True
        ), "Gradient check failed for StateCSpaceCost"


class TestBaseCSpaceCostValidateInput:
    """Test validate_input method in BaseCSpaceCost (through subclasses)."""

    def test_validate_input_batch_size_mismatch(self, device_cfg, joint_limits):
        """Test that batch size mismatch raises error."""
        cfg = CSpaceCostCfg(
            weight=[1.0, 1.0],
            activation_distance=[0.01, 0.01],
            cost_type=CSpaceCostType.POSITION,
            dof=7,
            joint_limits=joint_limits,
            device_cfg=device_cfg,
        )
        cost = PositionCSpaceCost(cfg)
        cost.setup_batch_tensors(batch=4, horizon=10)

        # Create state with wrong batch size
        state = JointState(
            position=torch.zeros((8, 10, 7), **device_cfg.as_torch_dict()),  # Wrong batch
        )

        with pytest.raises(Exception):
            cost.forward(state)

    def test_validate_input_horizon_mismatch(self, device_cfg, joint_limits):
        """Test that horizon mismatch raises error."""
        cfg = CSpaceCostCfg(
            weight=[1.0, 1.0],
            activation_distance=[0.01, 0.01],
            cost_type=CSpaceCostType.POSITION,
            dof=7,
            joint_limits=joint_limits,
            device_cfg=device_cfg,
        )
        cost = PositionCSpaceCost(cfg)
        cost.setup_batch_tensors(batch=4, horizon=10)

        # Create state with wrong horizon
        state = JointState(
            position=torch.zeros((4, 20, 7), **device_cfg.as_torch_dict()),  # Wrong horizon
        )

        with pytest.raises(Exception):
            cost.forward(state)

    def test_validate_input_dof_mismatch(self, device_cfg, joint_limits):
        """Test that DOF mismatch raises error."""
        cfg = CSpaceCostCfg(
            weight=[1.0, 1.0],
            activation_distance=[0.01, 0.01],
            cost_type=CSpaceCostType.POSITION,
            dof=7,
            joint_limits=joint_limits,
            device_cfg=device_cfg,
        )
        cost = PositionCSpaceCost(cfg)
        cost.setup_batch_tensors(batch=4, horizon=10)

        # Create state with wrong DOF
        state = JointState(
            position=torch.zeros((4, 10, 10), **device_cfg.as_torch_dict()),  # Wrong DOF
        )

        with pytest.raises(Exception):
            cost.forward(state)
