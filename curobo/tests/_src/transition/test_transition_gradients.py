# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Gradient tests for RobotStateTransition and state transition functions.

Tests backward pass correctness:
1. No NaN values in gradients
2. Gradient accuracy via torch.autograd.gradcheck (float32)
"""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.state.state_joint import JointState
from curobo._src.transition.robot_state_transition import RobotStateTransition
from curobo._src.transition.robot_state_transition_cfg import (
    RobotStateTransitionCfg,
    TimeTrajCfg,
)
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util.state_filter import FilterCfg
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def franka_robot_cfg_dict():
    """Load Franka robot configuration dictionary."""
    robot_file = join_path(get_robot_configs_path(), "franka.yml")
    return load_yaml(robot_file)["robot_cfg"]


@pytest.fixture(scope="module")
def float32_device_cfg():
    """Create DeviceCfg with float32 for gradient checking."""
    return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)


@pytest.fixture(scope="module")
def franka_robot_cfg(franka_robot_cfg_dict, float32_device_cfg):
    """Create RobotCfg from Franka configuration with float32."""
    return RobotCfg.create({"robot_cfg": franka_robot_cfg_dict}, float32_device_cfg)


@pytest.fixture(scope="module")
def time_traj_cfg():
    """Create basic TimeTrajCfg."""
    return TimeTrajCfg(base_dt=0.02, base_ratio=1.0, max_dt=0.02)


@pytest.fixture(scope="module")
def state_filter_cfg(float32_device_cfg):
    """Create basic state filter config."""
    return FilterCfg.create(
        coeff_dict={},
        enable=False,
        dt=0.02,
        control_space=ControlSpace.POSITION,
        device_cfg=float32_device_cfg,
        teleport_mode=True,
    )


@pytest.fixture(scope="module")
def teleport_transition(franka_robot_cfg, time_traj_cfg, float32_device_cfg, state_filter_cfg):
    """Create RobotStateTransition with teleport mode."""
    cfg = RobotStateTransitionCfg(
        robot_config=franka_robot_cfg,
        dt_traj_params=time_traj_cfg,
        device_cfg=float32_device_cfg,
        control_space=ControlSpace.POSITION,
        teleport_mode=True,
        batch_size=2,
        horizon=10,
        state_filter_cfg=state_filter_cfg,
    )
    return RobotStateTransition(cfg)


@pytest.fixture(scope="module")
def clique_transition(franka_robot_cfg, time_traj_cfg, float32_device_cfg, state_filter_cfg):
    """Create RobotStateTransition with position clique mode."""
    cfg = RobotStateTransitionCfg(
        robot_config=franka_robot_cfg,
        dt_traj_params=time_traj_cfg,
        device_cfg=float32_device_cfg,
        control_space=ControlSpace.POSITION,
        teleport_mode=False,
        batch_size=2,
        horizon=14,  # Clique needs horizon >= 5
        state_filter_cfg=state_filter_cfg,
    )
    return RobotStateTransition(cfg)


@pytest.fixture(scope="module")
def acceleration_transition(franka_robot_cfg, time_traj_cfg, float32_device_cfg, state_filter_cfg):
    """Create RobotStateTransition with acceleration control space."""
    cfg = RobotStateTransitionCfg(
        robot_config=franka_robot_cfg,
        dt_traj_params=time_traj_cfg,
        device_cfg=float32_device_cfg,
        control_space=ControlSpace.ACCELERATION,
        batch_size=2,
        horizon=10,
        state_filter_cfg=state_filter_cfg,
    )
    return RobotStateTransition(cfg)


@pytest.fixture(scope="module")
def bspline_transition(franka_robot_cfg, time_traj_cfg, float32_device_cfg, state_filter_cfg):
    """Create RobotStateTransition with bspline control space."""
    cfg = RobotStateTransitionCfg(
        robot_config=franka_robot_cfg,
        dt_traj_params=time_traj_cfg,
        device_cfg=float32_device_cfg,
        control_space=ControlSpace.BSPLINE_4,
        n_knots=6,
        interpolation_steps=4,
        batch_size=2,
        state_filter_cfg=state_filter_cfg,
    )
    return RobotStateTransition(cfg)


# ============================================================================
# Helper Functions
# ============================================================================


def check_no_nan_gradients(tensor: torch.Tensor, name: str = "tensor"):
    """Assert that gradients exist and contain no NaN values."""
    assert tensor.grad is not None, f"{name} has no gradient"
    assert not torch.isnan(tensor.grad).any(), f"{name} gradient contains NaN values"
    assert not torch.isinf(tensor.grad).any(), f"{name} gradient contains Inf values"


def create_joint_state_with_grad(
    shape: tuple,
    device_cfg: DeviceCfg,
    requires_grad: bool = False,
) -> JointState:
    """Create a JointState with optional gradient tracking."""
    joint_state = JointState.zeros(shape, device_cfg)
    joint_state.position = torch.full(shape, fill_value=0.1, device=device_cfg.device, dtype=device_cfg.dtype, requires_grad=requires_grad)
    joint_state.velocity = torch.zeros_like(joint_state.position)
    joint_state.acceleration = torch.zeros_like(joint_state.position)
    joint_state.dt = torch.ones(shape[0], device=device_cfg.device, dtype=device_cfg.dtype)
    joint_state.jerk = torch.zeros_like(joint_state.position)
    return joint_state


# ============================================================================
# Test Classes - Backward Pass No NaN
# ============================================================================


class TestTeleportBackwardNoNaN:
    """Test backward pass for teleport mode produces no NaN gradients."""

    def test_backward_no_nan_simple(self, teleport_transition, float32_device_cfg):
        """Test backward pass with simple loss produces no NaN gradients."""
        transition = teleport_transition
        batch_size = 4

        start_state = JointState.zeros((1, transition.num_dof), float32_device_cfg)

        act_seq = torch.randn(
            batch_size, transition.action_horizon, transition.action_dim,
            device=float32_device_cfg.device, dtype=float32_device_cfg.dtype,
            requires_grad=True
        )

        result = transition.forward(start_state, act_seq, start_state_idx=None)

        # Compute loss and backward
        loss = result.joint_state.position.mean()
        loss.backward()

        check_no_nan_gradients(act_seq, "act_seq")

    def test_backward_no_nan_mse_loss(self, teleport_transition, float32_device_cfg):
        """Test backward pass with MSE loss produces no NaN gradients."""
        transition = teleport_transition
        batch_size = 4

        start_state = JointState.zeros((1, transition.num_dof), float32_device_cfg)

        act_seq = torch.randn(
            batch_size, transition.action_horizon, transition.action_dim,
            device=float32_device_cfg.device, dtype=float32_device_cfg.dtype,
            requires_grad=True
        )

        target = torch.zeros(
            batch_size, transition.horizon, transition.num_dof,
            device=float32_device_cfg.device, dtype=float32_device_cfg.dtype,
        )

        result = transition.forward(start_state, act_seq, start_state_idx=None)

        # MSE loss
        loss = torch.nn.functional.mse_loss(result.joint_state.position, target)
        loss.backward()

        check_no_nan_gradients(act_seq, "act_seq")


@pytest.mark.skip(reason="AccelerationTensorStepIdxKernel backward not implemented")
class TestAccelerationBackwardNoNaN:
    """Test backward pass for acceleration control space produces no NaN gradients."""

    def test_backward_no_nan(self, acceleration_transition, float32_device_cfg):
        """Test backward pass produces no NaN gradients."""
        transition = acceleration_transition
        batch_size = 4

        start_state = create_joint_state_with_grad(
            (1, transition.num_dof), float32_device_cfg
        )
        start_state_idx = torch.zeros(
            batch_size, device=float32_device_cfg.device, dtype=torch.int32
        )
        use_implicit_goal_state = torch.zeros(
            batch_size, device=float32_device_cfg.device, dtype=torch.uint8
        )

        act_seq = torch.randn(
            batch_size, transition.action_horizon, transition.action_dim,
            device=float32_device_cfg.device, dtype=float32_device_cfg.dtype,
            requires_grad=True
        )

        result = transition.forward(
            start_state, act_seq,
            start_state_idx=start_state_idx,
            use_implicit_goal_state=use_implicit_goal_state,
        )

        loss = result.joint_state.position.mean()
        loss.backward()

        check_no_nan_gradients(act_seq, "act_seq")

    def test_backward_no_nan_velocity_loss(self, acceleration_transition, float32_device_cfg):
        """Test backward with velocity loss produces no NaN gradients."""
        transition = acceleration_transition
        batch_size = 4

        start_state = create_joint_state_with_grad(
            (1, transition.num_dof), float32_device_cfg
        )
        start_state_idx = torch.zeros(
            batch_size, device=float32_device_cfg.device, dtype=torch.int32
        )
        use_implicit_goal_state = torch.zeros(
            batch_size, device=float32_device_cfg.device, dtype=torch.uint8
        )

        act_seq = torch.randn(
            batch_size, transition.action_horizon, transition.action_dim,
            device=float32_device_cfg.device, dtype=float32_device_cfg.dtype,
            requires_grad=True
        )

        result = transition.forward(
            start_state, act_seq,
            start_state_idx=start_state_idx,
            use_implicit_goal_state=use_implicit_goal_state,
        )

        # Loss on velocity
        loss = result.joint_state.velocity.pow(2).sum()
        loss.backward()

        check_no_nan_gradients(act_seq, "act_seq")


class TestCliqueBackwardNoNaN:
    """Test backward pass for position clique mode produces no NaN gradients."""

    @pytest.mark.skip(reason="Test produces Inf gradient values - numerical stability issue needs investigation")
    def test_backward_no_nan(self, clique_transition, float32_device_cfg):
        """Test backward pass produces no NaN gradients."""
        transition = clique_transition
        batch_size = 4
        horizon = transition.horizon

        start_state = create_joint_state_with_grad(
            (1, transition.num_dof), float32_device_cfg
        )
        start_state_idx = torch.zeros(
            batch_size, device=float32_device_cfg.device, dtype=torch.int32
        )

        # Goal state with dt
        goal_state = create_joint_state_with_grad(
            (1, horizon, transition.num_dof), float32_device_cfg
        )
        goal_state.dt = torch.ones(
            1, horizon, device=float32_device_cfg.device, dtype=float32_device_cfg.dtype
        )
        goal_state_idx = torch.zeros(
            batch_size, device=float32_device_cfg.device, dtype=torch.int32
        )
        use_implicit_goal_state = torch.zeros(
            1, horizon, device=float32_device_cfg.device, dtype=torch.uint8
        )

        act_seq = torch.full(
            (batch_size, transition.action_horizon, transition.action_dim),
            fill_value=0.05,
            device=float32_device_cfg.device, dtype=float32_device_cfg.dtype,
            requires_grad=True
        )

        result = transition.forward(
            start_state, act_seq,
            start_state_idx=start_state_idx,
            goal_state=goal_state,
            goal_state_idx=goal_state_idx,
            use_implicit_goal_state=use_implicit_goal_state,
        )

        loss = result.joint_state.position.mean()
        loss.backward()

        check_no_nan_gradients(act_seq, "act_seq")

    def test_backward_no_nan_combined_loss(self, clique_transition, float32_device_cfg):
        """Test backward with combined position/velocity/acceleration loss."""
        transition = clique_transition
        batch_size = 4
        horizon = transition.horizon

        start_state = create_joint_state_with_grad(
            (1, transition.num_dof), float32_device_cfg
        )
        start_state_idx = torch.zeros(
            batch_size, device=float32_device_cfg.device, dtype=torch.int32
        )

        goal_state = create_joint_state_with_grad(
            (1, horizon, transition.num_dof), float32_device_cfg
        )
        goal_state.dt = torch.ones(
            1, horizon, device=float32_device_cfg.device, dtype=float32_device_cfg.dtype
        )
        goal_state_idx = torch.zeros(
            batch_size, device=float32_device_cfg.device, dtype=torch.int32
        )
        use_implicit_goal_state = torch.zeros(
            1, horizon, device=float32_device_cfg.device, dtype=torch.uint8
        )

        act_seq = torch.randn(
            batch_size, transition.action_horizon, transition.action_dim,
            device=float32_device_cfg.device, dtype=float32_device_cfg.dtype,
            requires_grad=True
        )

        result = transition.forward(
            start_state, act_seq,
            start_state_idx=start_state_idx,
            goal_state=goal_state,
            goal_state_idx=goal_state_idx,
            use_implicit_goal_state=use_implicit_goal_state,
        )

        # Combined loss on position, velocity, acceleration
        loss = (
            result.joint_state.position.pow(2).sum()
            + result.joint_state.velocity.pow(2).sum()
            + result.joint_state.acceleration.pow(2).sum()
        )
        loss.backward()

        check_no_nan_gradients(act_seq, "act_seq")


class TestBSplineBackwardNoNaN:
    """Test backward pass for B-spline control space produces no NaN gradients."""

    def test_backward_no_nan(self, bspline_transition, float32_device_cfg):
        """Test backward pass produces no NaN gradients."""
        transition = bspline_transition
        batch_size = 4
        horizon = transition.horizon

        start_state = create_joint_state_with_grad(
            (1, transition.num_dof), float32_device_cfg
        )
        start_state_idx = torch.zeros(
            batch_size, device=float32_device_cfg.device, dtype=torch.int32
        )

        goal_state = create_joint_state_with_grad(
            (1, horizon, transition.num_dof), float32_device_cfg
        )
        goal_state.dt = torch.ones(
            1, horizon, device=float32_device_cfg.device, dtype=float32_device_cfg.dtype
        )
        goal_state_idx = torch.zeros(
            batch_size, device=float32_device_cfg.device, dtype=torch.int32
        )
        use_implicit_goal_state = torch.zeros(
            1, horizon, device=float32_device_cfg.device, dtype=torch.uint8
        )

        # B-spline uses n_knots for action horizon
        act_seq = torch.randn(
            batch_size, transition.n_knots, transition.action_dim,
            device=float32_device_cfg.device, dtype=float32_device_cfg.dtype,
            requires_grad=True
        )

        result = transition.forward(
            start_state, act_seq,
            start_state_idx=start_state_idx,
            goal_state=goal_state,
            goal_state_idx=goal_state_idx,
            use_implicit_goal_state=use_implicit_goal_state,
        )

        loss = result.joint_state.position.mean()
        loss.backward()

        check_no_nan_gradients(act_seq, "act_seq")


# ============================================================================
# Test Classes - Gradient Check (torch.autograd.gradcheck)
# ============================================================================


class TestTeleportGradcheck:
    """Test gradient accuracy for teleport mode using torch.autograd.gradcheck."""

    def test_gradcheck_teleport(self, teleport_transition, float32_device_cfg):
        """Test gradient accuracy with gradcheck."""
        transition = teleport_transition
        batch_size = 2

        start_state = JointState.zeros((1, transition.num_dof), float32_device_cfg)

        # Small input for numerical stability
        act_seq = torch.randn(
            batch_size, transition.action_horizon, transition.action_dim,
            device=float32_device_cfg.device, dtype=torch.float64,  # gradcheck needs float64
            requires_grad=True
        ) * 0.1

        def forward_fn(act):
            # Convert to float32 for forward, but return float64 for gradcheck
            result = transition.forward(
                start_state, act.float(), start_state_idx=None
            )
            return result.joint_state.position.double()

        # Use gradcheck with relaxed tolerances for CUDA kernels
        result = torch.autograd.gradcheck(
            forward_fn, (act_seq,),
            eps=1e-4,
            atol=1e-3,
            rtol=1e-2,
            raise_exception=False,
        )
        assert result, "Teleport gradcheck failed"


@pytest.mark.skip(reason="AccelerationTensorStepIdxKernel backward not implemented")
class TestAccelerationGradcheck:
    """Test gradient accuracy for acceleration control space."""

    def test_gradcheck_acceleration(self, acceleration_transition, float32_device_cfg):
        """Test gradient accuracy with gradcheck."""
        transition = acceleration_transition
        batch_size = 2

        start_state = create_joint_state_with_grad(
            (1, transition.num_dof), float32_device_cfg
        )
        start_state_idx = torch.zeros(
            batch_size, device=float32_device_cfg.device, dtype=torch.int32
        )
        use_implicit_goal_state = torch.zeros(
            batch_size, device=float32_device_cfg.device, dtype=torch.uint8
        )

        act_seq = torch.randn(
            batch_size, transition.action_horizon, transition.action_dim,
            device=float32_device_cfg.device, dtype=torch.float64,
            requires_grad=True
        ) * 0.1

        def forward_fn(act):
            result = transition.forward(
                start_state, act.float(),
                start_state_idx=start_state_idx,
                use_implicit_goal_state=use_implicit_goal_state,
            )
            return result.joint_state.position.double()

        result = torch.autograd.gradcheck(
            forward_fn, (act_seq,),
            eps=1e-4,
            atol=1e-3,
            rtol=1e-2,
            raise_exception=False,
        )
        assert result, "Acceleration gradcheck failed"


class TestCliqueGradcheck:
    """Test gradient accuracy for position clique mode."""

    def test_gradcheck_clique(self, clique_transition, float32_device_cfg):
        """Test gradient accuracy with gradcheck."""
        transition = clique_transition
        batch_size = 2
        horizon = transition.horizon

        start_state = create_joint_state_with_grad(
            (1, transition.num_dof), float32_device_cfg
        )
        start_state_idx = torch.zeros(
            batch_size, device=float32_device_cfg.device, dtype=torch.int32
        )

        goal_state = create_joint_state_with_grad(
            (1, horizon, transition.num_dof), float32_device_cfg
        )
        goal_state.dt = torch.ones(
            1, horizon, device=float32_device_cfg.device, dtype=float32_device_cfg.dtype
        )
        goal_state_idx = torch.zeros(
            batch_size, device=float32_device_cfg.device, dtype=torch.int32
        )
        use_implicit_goal_state = torch.zeros(
            1, horizon, device=float32_device_cfg.device, dtype=torch.uint8
        )

        act_seq = torch.randn(
            batch_size, transition.action_horizon, transition.action_dim,
            device=float32_device_cfg.device, dtype=torch.float64,
            requires_grad=True
        ) * 0.1

        def forward_fn(act):
            result = transition.forward(
                start_state, act.float(),
                start_state_idx=start_state_idx,
                goal_state=goal_state,
                goal_state_idx=goal_state_idx,
                use_implicit_goal_state=use_implicit_goal_state,
            )
            return result.joint_state.position.double()

        result = torch.autograd.gradcheck(
            forward_fn, (act_seq,),
            eps=1e-4,
            atol=1e-3,
            rtol=1e-2,
            raise_exception=False,
        )
        assert result, "Clique gradcheck failed"


class TestBSplineGradcheck:
    """Test gradient accuracy for B-spline control space."""

    def test_gradcheck_bspline(self, bspline_transition, float32_device_cfg):
        """Test gradient accuracy with gradcheck."""
        transition = bspline_transition
        batch_size = 2
        horizon = transition.horizon

        start_state = create_joint_state_with_grad(
            (1, transition.num_dof), float32_device_cfg
        )
        start_state_idx = torch.zeros(
            batch_size, device=float32_device_cfg.device, dtype=torch.int32
        )

        goal_state = create_joint_state_with_grad(
            (1, horizon, transition.num_dof), float32_device_cfg
        )
        goal_state.dt = torch.ones(
            1, horizon, device=float32_device_cfg.device, dtype=float32_device_cfg.dtype
        )
        goal_state_idx = torch.zeros(
            batch_size, device=float32_device_cfg.device, dtype=torch.int32
        )
        use_implicit_goal_state = torch.zeros(
            1, horizon, device=float32_device_cfg.device, dtype=torch.uint8
        )

        act_seq = torch.randn(
            batch_size, transition.n_knots, transition.action_dim,
            device=float32_device_cfg.device, dtype=torch.float64,
            requires_grad=True
        ) * 0.1

        def forward_fn(act):
            result = transition.forward(
                start_state, act.float(),
                start_state_idx=start_state_idx,
                goal_state=goal_state,
                goal_state_idx=goal_state_idx,
                use_implicit_goal_state=use_implicit_goal_state,
            )
            return result.joint_state.position.double()

        result = torch.autograd.gradcheck(
            forward_fn, (act_seq,),
            eps=1e-4,
            atol=1e-3,
            rtol=1e-2,
            raise_exception=False,
        )
        assert result, "B-spline gradcheck failed"


# ============================================================================
# Test Classes - Gradient Magnitude
# ============================================================================


class TestGradientMagnitude:
    """Test that gradients have reasonable magnitudes (not too small or large)."""

    def test_gradient_magnitude_teleport(self, teleport_transition, float32_device_cfg):
        """Test gradient magnitude is reasonable for teleport."""
        transition = teleport_transition
        batch_size = 4

        start_state = JointState.zeros((1, transition.num_dof), float32_device_cfg)

        act_seq = torch.randn(
            batch_size, transition.action_horizon, transition.action_dim,
            device=float32_device_cfg.device, dtype=float32_device_cfg.dtype,
            requires_grad=True
        )

        result = transition.forward(start_state, act_seq, start_state_idx=None)
        loss = result.joint_state.position.pow(2).mean()
        loss.backward()

        grad_norm = act_seq.grad.norm()
        assert grad_norm > 1e-10, "Gradient too small (near zero)"
        assert grad_norm < 1e10, "Gradient too large"

    @pytest.mark.skip(reason="AccelerationTensorStepIdxKernel backward not implemented")
    def test_gradient_magnitude_acceleration(self, acceleration_transition, float32_device_cfg):
        """Test gradient magnitude is reasonable for acceleration."""
        transition = acceleration_transition
        batch_size = 4

        start_state = create_joint_state_with_grad(
            (1, transition.num_dof), float32_device_cfg
        )
        start_state_idx = torch.zeros(
            batch_size, device=float32_device_cfg.device, dtype=torch.int32
        )
        use_implicit_goal_state = torch.zeros(
            batch_size, device=float32_device_cfg.device, dtype=torch.uint8
        )

        act_seq = torch.randn(
            batch_size, transition.action_horizon, transition.action_dim,
            device=float32_device_cfg.device, dtype=float32_device_cfg.dtype,
            requires_grad=True
        )

        result = transition.forward(
            start_state, act_seq,
            start_state_idx=start_state_idx,
            use_implicit_goal_state=use_implicit_goal_state,
        )
        loss = result.joint_state.position.pow(2).mean()
        loss.backward()

        grad_norm = act_seq.grad.norm()
        assert grad_norm > 1e-10, "Gradient too small (near zero)"
        assert grad_norm < 1e10, "Gradient too large"

    def test_gradient_magnitude_clique(self, clique_transition, float32_device_cfg):
        """Test gradient magnitude is reasonable for clique."""
        transition = clique_transition
        batch_size = 4
        horizon = transition.horizon

        start_state = create_joint_state_with_grad(
            (1, transition.num_dof), float32_device_cfg
        )
        start_state_idx = torch.zeros(
            batch_size, device=float32_device_cfg.device, dtype=torch.int32
        )

        goal_state = create_joint_state_with_grad(
            (1, horizon, transition.num_dof), float32_device_cfg
        )
        goal_state.dt = torch.ones(
            1, horizon, device=float32_device_cfg.device, dtype=float32_device_cfg.dtype
        )
        goal_state_idx = torch.zeros(
            batch_size, device=float32_device_cfg.device, dtype=torch.int32
        )
        use_implicit_goal_state = torch.zeros(
            1, horizon, device=float32_device_cfg.device, dtype=torch.uint8
        )

        act_seq = torch.randn(
            batch_size, transition.action_horizon, transition.action_dim,
            device=float32_device_cfg.device, dtype=float32_device_cfg.dtype,
            requires_grad=True
        )

        result = transition.forward(
            start_state, act_seq,
            start_state_idx=start_state_idx,
            goal_state=goal_state,
            goal_state_idx=goal_state_idx,
            use_implicit_goal_state=use_implicit_goal_state,
        )
        loss = result.joint_state.position.pow(2).mean()
        loss.backward()

        grad_norm = act_seq.grad.norm()
        assert grad_norm > 1e-10, "Gradient too small (near zero)"
        assert grad_norm < 1e10, "Gradient too large"


# ============================================================================
# Test Classes - Multiple Backward Passes
# ============================================================================


class TestMultipleBackwardPasses:
    """Test that multiple backward passes work correctly."""

    def test_multiple_backward_teleport(self, teleport_transition, float32_device_cfg):
        """Test multiple backward passes accumulate correctly."""
        transition = teleport_transition
        batch_size = 4

        start_state = JointState.zeros((1, transition.num_dof), float32_device_cfg)

        act_seq = torch.randn(
            batch_size, transition.action_horizon, transition.action_dim,
            device=float32_device_cfg.device, dtype=float32_device_cfg.dtype,
            requires_grad=True
        )

        # First forward/backward
        result1 = transition.forward(start_state, act_seq, start_state_idx=None)
        loss1 = result1.joint_state.position.mean()
        loss1.backward()
        grad1 = act_seq.grad.clone()

        # Zero grad and do second forward/backward
        act_seq.grad.zero_()
        result2 = transition.forward(start_state, act_seq, start_state_idx=None)
        loss2 = result2.joint_state.position.mean()
        loss2.backward()
        grad2 = act_seq.grad.clone()

        # Gradients should be the same for same input
        assert torch.allclose(grad1, grad2, atol=1e-6)

    def test_gradient_accumulation(self, teleport_transition, float32_device_cfg):
        """Test that gradients accumulate when not zeroed."""
        transition = teleport_transition
        batch_size = 4

        start_state = JointState.zeros((1, transition.num_dof), float32_device_cfg)

        act_seq = torch.randn(
            batch_size, transition.action_horizon, transition.action_dim,
            device=float32_device_cfg.device, dtype=float32_device_cfg.dtype,
            requires_grad=True
        )

        # First backward
        result1 = transition.forward(start_state, act_seq, start_state_idx=None)
        loss1 = result1.joint_state.position.mean()
        loss1.backward()
        grad1 = act_seq.grad.clone()

        # Second backward without zeroing - should accumulate
        result2 = transition.forward(start_state, act_seq, start_state_idx=None)
        loss2 = result2.joint_state.position.mean()
        loss2.backward()

        # Accumulated gradient should be 2x single gradient
        assert torch.allclose(act_seq.grad, grad1 * 2, atol=1e-6)

