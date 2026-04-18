# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Tests for error handling (log_and_raise) in transition module.

These tests verify that appropriate errors are raised when invalid inputs
are provided to the transition functions and classes.
"""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.state.state_joint import JointState
from curobo._src.transition.fns_state_transition import (
    StateFromAcceleration,
    StateFromBSplineKnot,
    StateFromPositionClique,
)
from curobo._src.transition.robot_state_transition import RobotStateTransition
from curobo._src.transition.robot_state_transition_cfg import (
    RobotStateTransitionCfg,
    TimeTrajCfg,
)
from curobo._src.types.control_space import ControlSpace
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
def franka_robot_cfg(franka_robot_cfg_dict, cuda_device_cfg):
    """Create RobotCfg from Franka configuration."""
    return RobotCfg.create({"robot_cfg": franka_robot_cfg_dict}, cuda_device_cfg)


@pytest.fixture(scope="module")
def time_traj_cfg():
    """Create basic TimeTrajCfg."""
    return TimeTrajCfg(base_dt=0.02, base_ratio=1.0, max_dt=0.02)


@pytest.fixture(scope="module")
def state_filter_cfg(cuda_device_cfg):
    """Create basic state filter config."""
    return FilterCfg.create(
        coeff_dict={},
        enable=False,
        dt=0.02,
        control_space=ControlSpace.POSITION,
        device_cfg=cuda_device_cfg,
        teleport_mode=True,
    )


@pytest.fixture(scope="module")
def teleport_transition_cfg(franka_robot_cfg, time_traj_cfg, cuda_device_cfg, state_filter_cfg):
    """Create RobotStateTransitionCfg with teleport mode (no dynamics)."""
    return RobotStateTransitionCfg(
        robot_config=franka_robot_cfg,
        dt_traj_params=time_traj_cfg,
        device_cfg=cuda_device_cfg,
        control_space=ControlSpace.POSITION,
        teleport_mode=True,
        batch_size=2,
        horizon=10,
        state_filter_cfg=state_filter_cfg,
    )


# ============================================================================
# Test Classes - RobotStateTransition Error Handling
# ============================================================================


class TestRobotStateTransitionVelocityControlSpace:
    """Test that VELOCITY control space raises error."""

    def test_velocity_control_space_raises_error(
        self, franka_robot_cfg, time_traj_cfg, cuda_device_cfg, state_filter_cfg
    ):
        """Test that creating RobotStateTransition with VELOCITY raises error."""
        cfg = RobotStateTransitionCfg(
            robot_config=franka_robot_cfg,
            dt_traj_params=time_traj_cfg,
            device_cfg=cuda_device_cfg,
            control_space=ControlSpace.VELOCITY,
            batch_size=2,
            horizon=10,
            state_filter_cfg=state_filter_cfg,
        )

        with pytest.raises(Exception) as exc_info:
            RobotStateTransition(cfg)

        assert "velocity" in str(exc_info.value).lower() or "not implemented" in str(exc_info.value).lower()


class TestRobotStateTransitionDynamicsErrors:
    """Test errors when dynamics methods are called without dynamics."""

    def test_update_link_mass_without_dynamics(self, teleport_transition_cfg):
        """Test update_link_mass raises error without dynamics."""
        transition = RobotStateTransition(teleport_transition_cfg)

        with pytest.raises(Exception) as exc_info:
            transition.update_link_mass("panda_link1", 1.0)

        assert "dynamics" in str(exc_info.value).lower()

    def test_update_link_inertial_without_dynamics(self, teleport_transition_cfg):
        """Test update_link_inertial raises error without dynamics."""
        transition = RobotStateTransition(teleport_transition_cfg)

        with pytest.raises(Exception) as exc_info:
            transition.update_link_inertial("panda_link1", mass=1.0)

        assert "dynamics" in str(exc_info.value).lower()

    def test_update_links_inertial_without_dynamics(self, teleport_transition_cfg):
        """Test update_links_inertial raises error without dynamics."""
        transition = RobotStateTransition(teleport_transition_cfg)

        with pytest.raises(Exception) as exc_info:
            transition.update_links_inertial({"panda_link1": {"mass": 1.0}})

        assert "dynamics" in str(exc_info.value).lower()


# ============================================================================
# Test Classes - StateFromAcceleration Error Handling
# ============================================================================


class TestStateFromAccelerationErrors:
    """Test error handling for StateFromAcceleration."""

    @pytest.fixture
    def acceleration_step_fn(self, cuda_device_cfg):
        """Create StateFromAcceleration instance."""
        batch_size = 2
        horizon = 10
        dof = 7
        dt_h = torch.ones(horizon, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype) * 0.02
        return StateFromAcceleration(cuda_device_cfg, dt_h, dof, batch_size, horizon)

    def test_forward_raises_without_start_state_idx(self, acceleration_step_fn, cuda_device_cfg):
        """Test forward raises error when start_state_idx is None."""
        batch_size = 2
        horizon = 10
        dof = 7

        start_state = JointState.zeros((1, dof), cuda_device_cfg)
        u_act = torch.randn(batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict())
        out_state_seq = JointState.zeros((batch_size, horizon, dof), cuda_device_cfg)

        with pytest.raises(Exception) as exc_info:
            acceleration_step_fn.forward(
                start_state, u_act, out_state_seq, start_state_idx=None
            )

        assert "start state index" in str(exc_info.value).lower() or "acceleration" in str(exc_info.value).lower()


# ============================================================================
# Test Classes - StateFromPositionClique Error Handling
# ============================================================================


class TestStateFromPositionCliqueErrors:
    """Test error handling for StateFromPositionClique."""

    @pytest.fixture
    def clique_step_fn(self, cuda_device_cfg):
        """Create StateFromPositionClique instance."""
        batch_size = 2
        horizon = 14
        dof = 7
        dt_h = torch.ones(horizon, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype) * 0.02
        return StateFromPositionClique(cuda_device_cfg, dt_h, dof, batch_size=batch_size, horizon=horizon)

    def test_forward_raises_without_start_state_idx(self, clique_step_fn, cuda_device_cfg):
        """Test forward raises error when start_state_idx is None."""
        batch_size = 2
        horizon = 14
        dof = 7
        action_horizon = horizon - 4

        start_state = JointState.zeros((1, dof), cuda_device_cfg)
        u_act = torch.randn(batch_size, action_horizon, dof, **cuda_device_cfg.as_torch_dict())
        out_state_seq = JointState.zeros((batch_size, horizon, dof), cuda_device_cfg)

        # Setup goal state with proper dt
        goal_state = JointState.zeros((1, horizon, dof), cuda_device_cfg)
        goal_state.dt = torch.ones(1, horizon, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        goal_state_idx = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)
        use_implicit_goal_state = torch.zeros(1, horizon, device=cuda_device_cfg.device, dtype=torch.uint8)

        with pytest.raises(Exception) as exc_info:
            clique_step_fn.forward(
                start_state, u_act, out_state_seq,
                start_state_idx=None,  # This should trigger the error
                goal_state=goal_state,
                goal_state_idx=goal_state_idx,
                use_implicit_goal_state=use_implicit_goal_state,
            )

        assert "start state index" in str(exc_info.value).lower() or "clique" in str(exc_info.value).lower()

    def test_forward_raises_on_dt_shape_mismatch(self, clique_step_fn, cuda_device_cfg):
        """Test forward raises error when goal_state.dt shape doesn't match position."""
        batch_size = 2
        horizon = 14
        dof = 7
        action_horizon = horizon - 4

        start_state = JointState.zeros((1, dof), cuda_device_cfg)
        u_act = torch.randn(batch_size, action_horizon, dof, **cuda_device_cfg.as_torch_dict())
        out_state_seq = JointState.zeros((batch_size, horizon, dof), cuda_device_cfg)
        start_state_idx = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)

        # Setup goal state with WRONG dt shape
        goal_state = JointState.zeros((1, horizon, dof), cuda_device_cfg)
        goal_state.dt = torch.ones(5, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)  # Wrong shape!
        goal_state_idx = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)
        use_implicit_goal_state = torch.zeros(1, horizon, device=cuda_device_cfg.device, dtype=torch.uint8)

        with pytest.raises(Exception) as exc_info:
            clique_step_fn.forward(
                start_state, u_act, out_state_seq,
                start_state_idx=start_state_idx,
                goal_state=goal_state,
                goal_state_idx=goal_state_idx,
                use_implicit_goal_state=use_implicit_goal_state,
            )

        assert "shape" in str(exc_info.value).lower()

    def test_forward_raises_on_use_implicit_goal_state_shape_mismatch(
        self, clique_step_fn, cuda_device_cfg
    ):
        """Test forward raises error when use_implicit_goal_state shape doesn't match."""
        batch_size = 2
        horizon = 14
        dof = 7
        action_horizon = horizon - 4

        start_state = JointState.zeros((1, dof), cuda_device_cfg)
        u_act = torch.randn(batch_size, action_horizon, dof, **cuda_device_cfg.as_torch_dict())
        out_state_seq = JointState.zeros((batch_size, horizon, dof), cuda_device_cfg)
        start_state_idx = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)

        # Setup goal state with proper dt
        goal_state = JointState.zeros((1, horizon, dof), cuda_device_cfg)
        goal_state.dt = torch.ones(1, horizon, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        goal_state_idx = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)
        # Wrong shape for use_implicit_goal_state!
        use_implicit_goal_state = torch.zeros(5, 3, device=cuda_device_cfg.device, dtype=torch.uint8)

        with pytest.raises(Exception) as exc_info:
            clique_step_fn.forward(
                start_state, u_act, out_state_seq,
                start_state_idx=start_state_idx,
                goal_state=goal_state,
                goal_state_idx=goal_state_idx,
                use_implicit_goal_state=use_implicit_goal_state,
            )

        assert "shape" in str(exc_info.value).lower()


# ============================================================================
# Test Classes - StateFromBSplineKnot Error Handling
# ============================================================================


class TestStateFromBSplineKnotErrors:
    """Test error handling for StateFromBSplineKnot."""

    @pytest.fixture
    def bspline_step_fn(self, cuda_device_cfg):
        """Create StateFromBSplineKnot instance."""
        batch_size = 2
        n_knots = 6
        dof = 7
        interpolation_steps = 4
        # horizon is computed based on bspline degree and knots
        horizon = (n_knots + 3) * interpolation_steps + 1  # For BSPLINE_4
        return StateFromBSplineKnot(
            cuda_device_cfg, dof,
            batch_size=batch_size, horizon=horizon,
            n_knots=n_knots, interpolation_steps=interpolation_steps,
            control_space=ControlSpace.BSPLINE_4,
            use_implicit_goal_state=True,
        )

    def test_forward_raises_without_start_state_idx(self, bspline_step_fn, cuda_device_cfg):
        """Test forward raises error when start_state_idx is None."""
        batch_size = 2
        n_knots = 6
        dof = 7
        horizon = bspline_step_fn.padded_horizon

        start_state = JointState.zeros((1, dof), cuda_device_cfg)
        start_state.jerk = torch.zeros_like(start_state.position)
        u_act = torch.randn(batch_size, n_knots, dof, **cuda_device_cfg.as_torch_dict())
        out_state_seq = JointState.zeros((batch_size, horizon, dof), cuda_device_cfg)
        out_state_seq.dt = torch.ones(batch_size, horizon, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        goal_state = JointState.zeros((1, horizon, dof), cuda_device_cfg)
        goal_state.jerk = torch.zeros_like(goal_state.position)
        goal_state.dt = torch.ones(1, horizon, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        goal_state_idx = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)
        use_implicit_goal_state = torch.zeros(1, horizon, device=cuda_device_cfg.device, dtype=torch.uint8)

        with pytest.raises(Exception) as exc_info:
            bspline_step_fn.forward(
                start_state, u_act, out_state_seq,
                start_state_idx=None,  # Should trigger error
                goal_state=goal_state,
                goal_state_idx=goal_state_idx,
                use_implicit_goal_state=use_implicit_goal_state,
            )

        assert "start state index" in str(exc_info.value).lower() or "bspline" in str(exc_info.value).lower()

    def test_forward_raises_without_goal_state_idx(self, bspline_step_fn, cuda_device_cfg):
        """Test forward raises error when goal_state_idx is None."""
        batch_size = 2
        n_knots = 6
        dof = 7
        horizon = bspline_step_fn.padded_horizon

        start_state = JointState.zeros((1, dof), cuda_device_cfg)
        start_state.jerk = torch.zeros_like(start_state.position)
        start_state_idx = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)
        u_act = torch.randn(batch_size, n_knots, dof, **cuda_device_cfg.as_torch_dict())
        out_state_seq = JointState.zeros((batch_size, horizon, dof), cuda_device_cfg)
        out_state_seq.dt = torch.ones(batch_size, horizon, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        goal_state = JointState.zeros((1, horizon, dof), cuda_device_cfg)
        goal_state.jerk = torch.zeros_like(goal_state.position)
        goal_state.dt = torch.ones(1, horizon, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        use_implicit_goal_state = torch.zeros(1, horizon, device=cuda_device_cfg.device, dtype=torch.uint8)

        with pytest.raises(Exception) as exc_info:
            bspline_step_fn.forward(
                start_state, u_act, out_state_seq,
                start_state_idx=start_state_idx,
                goal_state=goal_state,
                goal_state_idx=None,  # Should trigger error
                use_implicit_goal_state=use_implicit_goal_state,
            )

        assert "index" in str(exc_info.value).lower()

        bspline_step_fn.use_implicit_goal_state = False
        bspline_step_fn.forward(
            start_state, u_act, out_state_seq,
            start_state_idx=start_state_idx,
            goal_state=goal_state,
            goal_state_idx=None,
            use_implicit_goal_state=use_implicit_goal_state,
        )
        assert out_state_seq.position.shape == (batch_size, horizon, dof)
        bspline_step_fn.use_implicit_goal_state = True

    def test_forward_raises_without_goal_state_dt(self, bspline_step_fn, cuda_device_cfg):
        """Test forward raises error when goal_state.dt is None."""
        batch_size = 2
        n_knots = 6
        dof = 7
        horizon = bspline_step_fn.padded_horizon

        start_state = JointState.zeros((1, dof), cuda_device_cfg)
        start_state.jerk = torch.zeros_like(start_state.position)
        start_state_idx = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)
        u_act = torch.randn(batch_size, n_knots, dof, **cuda_device_cfg.as_torch_dict())
        out_state_seq = JointState.zeros((batch_size, horizon, dof), cuda_device_cfg)
        out_state_seq.dt = torch.ones(batch_size, horizon, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        goal_state = JointState.zeros((1, horizon, dof), cuda_device_cfg)
        goal_state.jerk = torch.zeros_like(goal_state.position)
        goal_state.dt = None  # Should trigger error
        goal_state_idx = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)
        use_implicit_goal_state = torch.zeros(1, horizon, device=cuda_device_cfg.device, dtype=torch.uint8)

        with pytest.raises(Exception) as exc_info:
            bspline_step_fn.forward(
                start_state, u_act, out_state_seq,
                start_state_idx=start_state_idx,
                goal_state=goal_state,
                goal_state_idx=goal_state_idx,
                use_implicit_goal_state=use_implicit_goal_state,
            )

        assert "dt" in str(exc_info.value).lower()

    def test_forward_raises_without_use_implicit_goal_state(self, bspline_step_fn, cuda_device_cfg):
        """Test forward raises error when use_implicit_goal_state is None."""
        batch_size = 2
        n_knots = 6
        dof = 7
        horizon = bspline_step_fn.padded_horizon

        start_state = JointState.zeros((1, dof), cuda_device_cfg)
        start_state.jerk = torch.zeros_like(start_state.position)
        start_state_idx = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)
        u_act = torch.randn(batch_size, n_knots, dof, **cuda_device_cfg.as_torch_dict())
        out_state_seq = JointState.zeros((batch_size, horizon, dof), cuda_device_cfg)
        out_state_seq.dt = torch.ones(batch_size, horizon, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        goal_state = JointState.zeros((1, horizon, dof), cuda_device_cfg)
        goal_state.jerk = torch.zeros_like(goal_state.position)
        goal_state.dt = torch.ones(1, horizon, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        goal_state_idx = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)

        with pytest.raises(Exception) as exc_info:
            bspline_step_fn.forward(
                start_state, u_act, out_state_seq,
                start_state_idx=start_state_idx,
                goal_state=goal_state,
                goal_state_idx=goal_state_idx,
                use_implicit_goal_state=None,  # Should trigger error
            )

        assert "use_implicit_goal_state" in str(exc_info.value).lower()

    def test_forward_raises_on_goal_state_idx_shape_mismatch(self, bspline_step_fn, cuda_device_cfg):
        """Test forward raises error when goal_state_idx.shape[0] != u_act.shape[0]."""
        batch_size = 2
        n_knots = 6
        dof = 7
        horizon = bspline_step_fn.padded_horizon

        start_state = JointState.zeros((1, dof), cuda_device_cfg)
        start_state.jerk = torch.zeros_like(start_state.position)
        start_state_idx = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)
        u_act = torch.randn(batch_size, n_knots, dof, **cuda_device_cfg.as_torch_dict())
        out_state_seq = JointState.zeros((batch_size, horizon, dof), cuda_device_cfg)
        out_state_seq.dt = torch.ones(batch_size, horizon, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        goal_state = JointState.zeros((1, horizon, dof), cuda_device_cfg)
        goal_state.jerk = torch.zeros_like(goal_state.position)
        goal_state.dt = torch.ones(1, horizon, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        # Wrong shape - should have batch_size elements
        goal_state_idx = torch.zeros(5, device=cuda_device_cfg.device, dtype=torch.int32)
        use_implicit_goal_state = torch.zeros(1, horizon, device=cuda_device_cfg.device, dtype=torch.uint8)

        with pytest.raises(Exception) as exc_info:
            bspline_step_fn.forward(
                start_state, u_act, out_state_seq,
                start_state_idx=start_state_idx,
                goal_state=goal_state,
                goal_state_idx=goal_state_idx,
                use_implicit_goal_state=use_implicit_goal_state,
            )

        assert "shape" in str(exc_info.value).lower()

    def test_forward_raises_on_use_implicit_goal_state_shape_mismatch(
        self, bspline_step_fn, cuda_device_cfg
    ):
        """Test forward raises error when use_implicit_goal_state.shape[0] != goal_state.shape[0]."""
        batch_size = 2
        n_knots = 6
        dof = 7
        horizon = bspline_step_fn.padded_horizon

        start_state = JointState.zeros((1, dof), cuda_device_cfg)
        start_state.jerk = torch.zeros_like(start_state.position)
        start_state_idx = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)
        u_act = torch.randn(batch_size, n_knots, dof, **cuda_device_cfg.as_torch_dict())
        out_state_seq = JointState.zeros((batch_size, horizon, dof), cuda_device_cfg)
        out_state_seq.dt = torch.ones(batch_size, horizon, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        goal_state = JointState.zeros((1, horizon, dof), cuda_device_cfg)
        goal_state.jerk = torch.zeros_like(goal_state.position)
        goal_state.dt = torch.ones(1, horizon, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        goal_state_idx = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)
        # Wrong shape - should have goal_state.shape[0] elements
        use_implicit_goal_state = torch.zeros(5, horizon, device=cuda_device_cfg.device, dtype=torch.uint8)

        with pytest.raises(Exception) as exc_info:
            bspline_step_fn.forward(
                start_state, u_act, out_state_seq,
                start_state_idx=start_state_idx,
                goal_state=goal_state,
                goal_state_idx=goal_state_idx,
                use_implicit_goal_state=use_implicit_goal_state,
            )

        assert "shape" in str(exc_info.value).lower()

    def test_forward_raises_without_start_jerk(self, bspline_step_fn, cuda_device_cfg):
        """Test forward raises error when start_state.jerk is None."""
        batch_size = 2
        n_knots = 6
        dof = 7
        horizon = bspline_step_fn.padded_horizon

        start_state = JointState.zeros((1, dof), cuda_device_cfg)
        start_state.jerk = None  # Should trigger error
        start_state_idx = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)
        u_act = torch.randn(batch_size, n_knots, dof, **cuda_device_cfg.as_torch_dict())
        out_state_seq = JointState.zeros((batch_size, horizon, dof), cuda_device_cfg)
        out_state_seq.dt = torch.ones(batch_size, horizon, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        goal_state = JointState.zeros((1, horizon, dof), cuda_device_cfg)
        goal_state.jerk = torch.zeros_like(goal_state.position)
        goal_state.dt = torch.ones(1, horizon, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        goal_state_idx = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)
        use_implicit_goal_state = torch.zeros(1, horizon, device=cuda_device_cfg.device, dtype=torch.uint8)

        with pytest.raises(Exception) as exc_info:
            bspline_step_fn.forward(
                start_state, u_act, out_state_seq,
                start_state_idx=start_state_idx,
                goal_state=goal_state,
                goal_state_idx=goal_state_idx,
                use_implicit_goal_state=use_implicit_goal_state,
            )

        assert "jerk" in str(exc_info.value).lower()

    def test_forward_raises_without_out_state_seq_dt(self, bspline_step_fn, cuda_device_cfg):
        """Test forward raises error when out_state_seq.dt is None."""
        batch_size = 2
        n_knots = 6
        dof = 7
        horizon = bspline_step_fn.padded_horizon

        start_state = JointState.zeros((1, dof), cuda_device_cfg)
        start_state.jerk = torch.zeros_like(start_state.position)
        start_state_idx = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)
        u_act = torch.randn(batch_size, n_knots, dof, **cuda_device_cfg.as_torch_dict())
        out_state_seq = JointState.zeros((batch_size, horizon, dof), cuda_device_cfg)
        out_state_seq.dt = None  # Should trigger error

        goal_state = JointState.zeros((1, horizon, dof), cuda_device_cfg)
        goal_state.jerk = torch.zeros_like(goal_state.position)
        goal_state.dt = torch.ones(1, horizon, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        goal_state_idx = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)
        use_implicit_goal_state = torch.zeros(1, horizon, device=cuda_device_cfg.device, dtype=torch.uint8)

        with pytest.raises(Exception) as exc_info:
            bspline_step_fn.forward(
                start_state, u_act, out_state_seq,
                start_state_idx=start_state_idx,
                goal_state=goal_state,
                goal_state_idx=goal_state_idx,
                use_implicit_goal_state=use_implicit_goal_state,
            )

        assert "dt" in str(exc_info.value).lower()

    def test_forward_raises_on_u_act_knots_mismatch(self, bspline_step_fn, cuda_device_cfg):
        """Test forward raises error when u_act.shape[1] != n_knots."""
        batch_size = 2
        dof = 7
        horizon = bspline_step_fn.padded_horizon
        wrong_n_knots = 3  # Should be 6

        start_state = JointState.zeros((1, dof), cuda_device_cfg)
        start_state.jerk = torch.zeros_like(start_state.position)
        start_state_idx = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)
        u_act = torch.randn(batch_size, wrong_n_knots, dof, **cuda_device_cfg.as_torch_dict())  # Wrong!
        out_state_seq = JointState.zeros((batch_size, horizon, dof), cuda_device_cfg)
        out_state_seq.dt = torch.ones(batch_size, horizon, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        goal_state = JointState.zeros((1, horizon, dof), cuda_device_cfg)
        goal_state.jerk = torch.zeros_like(goal_state.position)
        goal_state.dt = torch.ones(1, horizon, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        goal_state_idx = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)
        use_implicit_goal_state = torch.zeros(1, horizon, device=cuda_device_cfg.device, dtype=torch.uint8)

        with pytest.raises(Exception) as exc_info:
            bspline_step_fn.forward(
                start_state, u_act, out_state_seq,
                start_state_idx=start_state_idx,
                goal_state=goal_state,
                goal_state_idx=goal_state_idx,
                use_implicit_goal_state=use_implicit_goal_state,
            )

        assert "n_knots" in str(exc_info.value).lower()

    def test_forward_raises_on_horizon_mismatch(self, bspline_step_fn, cuda_device_cfg):
        """Test forward raises error when padded_horizon != out_state_seq.shape[1]."""
        batch_size = 2
        n_knots = 6
        dof = 7
        wrong_horizon = 10  # Should be padded_horizon

        start_state = JointState.zeros((1, dof), cuda_device_cfg)
        start_state.jerk = torch.zeros_like(start_state.position)
        start_state_idx = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)
        u_act = torch.randn(batch_size, n_knots, dof, **cuda_device_cfg.as_torch_dict())
        out_state_seq = JointState.zeros((batch_size, wrong_horizon, dof), cuda_device_cfg)  # Wrong!
        out_state_seq.dt = torch.ones(batch_size, wrong_horizon, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        goal_state = JointState.zeros((1, wrong_horizon, dof), cuda_device_cfg)
        goal_state.jerk = torch.zeros_like(goal_state.position)
        goal_state.dt = torch.ones(1, wrong_horizon, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        goal_state_idx = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)
        use_implicit_goal_state = torch.zeros(1, wrong_horizon, device=cuda_device_cfg.device, dtype=torch.uint8)

        with pytest.raises(Exception) as exc_info:
            bspline_step_fn.forward(
                start_state, u_act, out_state_seq,
                start_state_idx=start_state_idx,
                goal_state=goal_state,
                goal_state_idx=goal_state_idx,
                use_implicit_goal_state=use_implicit_goal_state,
            )

        assert "padded_horizon" in str(exc_info.value).lower() or "horizon" in str(exc_info.value).lower()


class TestStateFromBSplineKnotImplicitGoalErrors:
    """Test error handling for StateFromBSplineKnot with use_implicit_goal_state=True."""

    @pytest.fixture
    def bspline_implicit_goal_step_fn(self, cuda_device_cfg):
        """Create StateFromBSplineKnot with use_implicit_goal_state=True."""
        batch_size = 2
        n_knots = 6
        dof = 7
        interpolation_steps = 4
        horizon = (n_knots + 3) * interpolation_steps + 1
        return StateFromBSplineKnot(
            cuda_device_cfg, dof,
            batch_size=batch_size, horizon=horizon,
            n_knots=n_knots, interpolation_steps=interpolation_steps,
            use_implicit_goal_state=True,  # Enable implicit goal
            control_space=ControlSpace.BSPLINE_4
        )

    def test_forward_raises_without_goal_state_when_implicit(
        self, bspline_implicit_goal_step_fn, cuda_device_cfg
    ):
        """Test forward raises error when goal_state is None with implicit goal."""
        batch_size = 2
        n_knots = 6
        dof = 7
        horizon = bspline_implicit_goal_step_fn.padded_horizon

        start_state = JointState.zeros((1, dof), cuda_device_cfg)
        start_state.jerk = torch.zeros_like(start_state.position)
        start_state_idx = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)
        u_act = torch.randn(batch_size, n_knots, dof, **cuda_device_cfg.as_torch_dict())
        out_state_seq = JointState.zeros((batch_size, horizon, dof), cuda_device_cfg)
        out_state_seq.dt = torch.ones(batch_size, horizon, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        goal_state_idx = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)
        use_implicit_goal_state = torch.zeros(1, horizon, device=cuda_device_cfg.device, dtype=torch.uint8)

        with pytest.raises(Exception) as exc_info:
            bspline_implicit_goal_step_fn.forward(
                start_state, u_act, out_state_seq,
                start_state_idx=start_state_idx,
                goal_state=None,  # Should trigger error when use_implicit_goal_state=True
                goal_state_idx=goal_state_idx,
                use_implicit_goal_state=use_implicit_goal_state,
            )

        assert "goal state" in str(exc_info.value).lower()

    def test_forward_raises_without_goal_state_idx_when_implicit(
        self, bspline_implicit_goal_step_fn, cuda_device_cfg
    ):
        """Test forward raises error when goal_state_idx is None with implicit goal."""
        batch_size = 2
        n_knots = 6
        dof = 7
        horizon = bspline_implicit_goal_step_fn.padded_horizon

        start_state = JointState.zeros((1, dof), cuda_device_cfg)
        start_state.jerk = torch.zeros_like(start_state.position)
        start_state_idx = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)
        u_act = torch.randn(batch_size, n_knots, dof, **cuda_device_cfg.as_torch_dict())
        out_state_seq = JointState.zeros((batch_size, horizon, dof), cuda_device_cfg)
        out_state_seq.dt = torch.ones(batch_size, horizon, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        goal_state = JointState.zeros((1, horizon, dof), cuda_device_cfg)
        goal_state.jerk = torch.zeros_like(goal_state.position)
        goal_state.dt = torch.ones(1, horizon, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        use_implicit_goal_state = torch.zeros(1, horizon, device=cuda_device_cfg.device, dtype=torch.uint8)

        with pytest.raises(Exception) as exc_info:
            bspline_implicit_goal_step_fn.forward(
                start_state, u_act, out_state_seq,
                start_state_idx=start_state_idx,
                goal_state=goal_state,
                goal_state_idx=None,  # Should trigger error when use_implicit_goal_state=True
                use_implicit_goal_state=use_implicit_goal_state,
            )

        assert "goal state index" in str(exc_info.value).lower()

