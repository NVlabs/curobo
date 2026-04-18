# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for RobotStateTransition with full coverage of control spaces."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState
from curobo._src.transition.fns_state_transition import (
    StateFromAcceleration,
    StateFromBSplineKnot,
    StateFromPositionClique,
    StateFromPositionTeleport,
)
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
    """Create RobotStateTransitionCfg with teleport mode."""
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


@pytest.fixture(scope="module")
def position_clique_transition_cfg(
    franka_robot_cfg, time_traj_cfg, cuda_device_cfg, state_filter_cfg
):
    """Create RobotStateTransitionCfg with position clique mode."""
    return RobotStateTransitionCfg(
        robot_config=franka_robot_cfg,
        dt_traj_params=time_traj_cfg,
        device_cfg=cuda_device_cfg,
        control_space=ControlSpace.POSITION,
        teleport_mode=False,
        batch_size=2,
        horizon=14,  # Clique needs horizon >= 5, action_horizon = horizon - 4
        state_filter_cfg=state_filter_cfg,
    )


@pytest.fixture(scope="module")
def acceleration_transition_cfg(
    franka_robot_cfg, time_traj_cfg, cuda_device_cfg, state_filter_cfg
):
    """Create RobotStateTransitionCfg with acceleration control space."""
    return RobotStateTransitionCfg(
        robot_config=franka_robot_cfg,
        dt_traj_params=time_traj_cfg,
        device_cfg=cuda_device_cfg,
        control_space=ControlSpace.ACCELERATION,
        batch_size=2,
        horizon=10,
        state_filter_cfg=state_filter_cfg,
    )


@pytest.fixture(scope="module")
def bspline_transition_cfg(franka_robot_cfg, time_traj_cfg, cuda_device_cfg, state_filter_cfg):
    """Create RobotStateTransitionCfg with bspline control space."""
    return RobotStateTransitionCfg(
        robot_config=franka_robot_cfg,
        dt_traj_params=time_traj_cfg,
        device_cfg=cuda_device_cfg,
        control_space=ControlSpace.BSPLINE_4,
        n_knots=6,
        interpolation_steps=4,
        batch_size=2,
        state_filter_cfg=state_filter_cfg,
    )


# ============================================================================
# Helper Functions
# ============================================================================


def create_start_state_with_idx(
    transition: RobotStateTransition,
    batch_size: int,
    cuda_device_cfg: DeviceCfg,
):
    """Create start state and index tensor for forward pass.

    This mimics how rollout_arm.py sets up the start state.
    """
    num_dof = transition.num_dof

    # Create start state - typically shape (num_problems, num_dof) or (1, num_dof)
    # The index tensor selects which row to use for each batch element
    n_start_states = 1  # Usually 1 start state repeated for all batch elements
    start_state = JointState.zeros((n_start_states, num_dof), cuda_device_cfg)
    start_state.position = torch.randn_like(start_state.position) * 0.1

    # Index tensor - all batch elements use start state 0
    start_state_idx = torch.zeros(
        batch_size, device=cuda_device_cfg.device, dtype=torch.int32
    )

    return start_state, start_state_idx


def create_goal_state_with_idx(
    transition: RobotStateTransition,
    batch_size: int,
    cuda_device_cfg: DeviceCfg,
    horizon: int,
):
    """Create goal state and index tensor for clique/bspline forward pass.

    This mimics how rollout_arm.py sets up the goal state (seed_goal_js).
    """
    num_dof = transition.num_dof

    # Goal state - shape (n_goals, horizon, num_dof) with dt
    n_goals = 1
    goal_state = JointState.zeros((n_goals, horizon, num_dof), cuda_device_cfg)
    goal_state.position = torch.randn_like(goal_state.position) * 0.1

    # dt tensor must match shape (n_goals, horizon)
    goal_state.dt = torch.ones(
        n_goals, horizon, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
    ) * transition.dt

    # Index tensor - all batch elements use goal state 0
    goal_state_idx = torch.zeros(
        batch_size, device=cuda_device_cfg.device, dtype=torch.int32
    )

    # use_implicit_goal_state - shape (n_goals, horizon) matching goal_state shape
    use_implicit_goal_state = torch.zeros(
        n_goals, horizon, device=cuda_device_cfg.device, dtype=torch.uint8
    )

    return goal_state, goal_state_idx, use_implicit_goal_state


# ============================================================================
# Test Classes - Initialization
# ============================================================================


class TestRobotStateTransitionInitialization:
    """Test RobotStateTransition initialization."""

    def test_teleport_mode_initialization(self, teleport_transition_cfg):
        """Test initialization with teleport mode."""
        transition = RobotStateTransition(teleport_transition_cfg)

        assert transition.config is teleport_transition_cfg
        assert transition.batch_size == 2
        assert transition.horizon == 10
        assert transition.teleport_mode is True
        assert transition.control_space == ControlSpace.POSITION
        assert isinstance(transition._rollout_step_fn, StateFromPositionTeleport)
        assert isinstance(transition._cmd_step_fn, StateFromPositionTeleport)

    def test_position_clique_initialization(self, position_clique_transition_cfg):
        """Test initialization with position clique mode."""
        transition = RobotStateTransition(position_clique_transition_cfg)

        assert transition.teleport_mode is False
        assert transition.control_space == ControlSpace.POSITION
        assert isinstance(transition._rollout_step_fn, StateFromPositionClique)
        assert isinstance(transition._cmd_step_fn, StateFromPositionClique)

    def test_acceleration_initialization(self, acceleration_transition_cfg):
        """Test initialization with acceleration control space."""
        transition = RobotStateTransition(acceleration_transition_cfg)

        assert transition.control_space == ControlSpace.ACCELERATION
        assert isinstance(transition._rollout_step_fn, StateFromAcceleration)
        assert isinstance(transition._cmd_step_fn, StateFromAcceleration)

    def test_bspline_initialization(self, bspline_transition_cfg):
        """Test initialization with bspline control space."""
        transition = RobotStateTransition(bspline_transition_cfg)

        assert transition.control_space == ControlSpace.BSPLINE_4
        assert isinstance(transition._rollout_step_fn, StateFromBSplineKnot)
        assert isinstance(transition._cmd_step_fn, StateFromBSplineKnot)


# ============================================================================
# Test Classes - Forward with Different Control Spaces
# ============================================================================


class TestRobotStateTransitionForwardTeleport:
    """Test forward method with teleport mode (StateFromPositionTeleport)."""

    def test_forward_teleport_basic(self, teleport_transition_cfg, cuda_device_cfg):
        """Test forward with teleport mode - basic case without indices."""
        transition = RobotStateTransition(teleport_transition_cfg)
        batch_size = 4

        # Create start state (teleport doesn't require idx)
        start_state = JointState.zeros((1, transition.num_dof), cuda_device_cfg)

        # Create action sequence - teleport uses full horizon
        act_seq = torch.randn(
            batch_size, transition.action_horizon, transition.action_dim,
            **cuda_device_cfg.as_torch_dict()
        )

        # Forward pass - teleport mode doesn't require state_idx
        result = transition.forward(
            start_state,
            act_seq,
            start_state_idx=None,
        )

        assert isinstance(result, RobotState)
        assert result.joint_state.position.shape == (batch_size, transition.horizon, transition.num_dof)

    def test_forward_teleport_copies_action_to_position(self, teleport_transition_cfg, cuda_device_cfg):
        """Test that teleport mode copies action directly to position."""
        transition = RobotStateTransition(teleport_transition_cfg)
        batch_size = 2

        start_state = JointState.zeros((1, transition.num_dof), cuda_device_cfg)
        act_seq = torch.randn(
            batch_size, transition.action_horizon, transition.action_dim,
            **cuda_device_cfg.as_torch_dict()
        )

        result = transition.forward(start_state, act_seq, start_state_idx=None)

        # In teleport mode, position should equal action
        assert torch.allclose(result.joint_state.position, act_seq)


class TestRobotStateTransitionForwardAcceleration:
    """Test forward method with acceleration control space (StateFromAcceleration)."""

    def test_forward_acceleration_with_idx(self, acceleration_transition_cfg, cuda_device_cfg):
        """Test forward with acceleration control space using proper indices."""
        transition = RobotStateTransition(acceleration_transition_cfg)
        batch_size = 4

        # Create start state with index (required for acceleration kernel)
        start_state, start_state_idx = create_start_state_with_idx(
            transition, batch_size, cuda_device_cfg
        )

        # Create action sequence
        act_seq = torch.randn(
            batch_size, transition.action_horizon, transition.action_dim,
            **cuda_device_cfg.as_torch_dict()
        )
        use_implicit_goal_state = torch.zeros(
            batch_size, device=cuda_device_cfg.device, dtype=torch.uint8
        )

        # Forward pass with state index
        result = transition.forward(
            start_state,
            act_seq,
            start_state_idx=start_state_idx,
            use_implicit_goal_state=use_implicit_goal_state,
        )

        assert isinstance(result, RobotState)
        assert result.joint_state.position.shape == (batch_size, transition.horizon, transition.num_dof)
        assert result.joint_state.velocity is not None
        assert result.joint_state.acceleration is not None
        assert result.joint_state.jerk is not None

    def test_forward_acceleration_computes_derivatives(self, acceleration_transition_cfg, cuda_device_cfg):
        """Test that acceleration forward computes velocity/position from acceleration."""
        transition = RobotStateTransition(acceleration_transition_cfg)
        batch_size = 2

        start_state, start_state_idx = create_start_state_with_idx(
            transition, batch_size, cuda_device_cfg
        )
        # Zero initial velocity and acceleration
        start_state.velocity = torch.zeros_like(start_state.position)
        start_state.acceleration = torch.zeros_like(start_state.position)

        # Constant acceleration action
        act_seq = torch.ones(
            batch_size, transition.action_horizon, transition.action_dim,
            **cuda_device_cfg.as_torch_dict()
        ) * 0.1

        result = transition.forward(start_state, act_seq, start_state_idx=start_state_idx)

        # With constant acceleration, velocity should increase over time
        vel_diff = result.joint_state.velocity[:, -1, :] - result.joint_state.velocity[:, 0, :]
        assert torch.all(vel_diff > 0)  # Velocity should increase


class TestRobotStateTransitionForwardPositionClique:
    """Test forward method with position clique mode (StateFromPositionClique)."""

    def test_forward_clique_with_goal_state(self, position_clique_transition_cfg, cuda_device_cfg):
        """Test forward with position clique using goal state (like rollout_arm.py)."""
        transition = RobotStateTransition(position_clique_transition_cfg)
        batch_size = 4
        horizon = transition.horizon

        # Create start state with index
        start_state, start_state_idx = create_start_state_with_idx(
            transition, batch_size, cuda_device_cfg
        )

        # Create goal state with index (required for clique kernel)
        goal_state, goal_state_idx, use_implicit_goal_state = create_goal_state_with_idx(
            transition, batch_size, cuda_device_cfg, horizon
        )

        # Action horizon for clique is horizon - 4
        action_horizon = transition.action_horizon
        act_seq = torch.randn(
            batch_size, action_horizon, transition.action_dim,
            **cuda_device_cfg.as_torch_dict()
        )

        # Forward pass with all required parameters
        result = transition.forward(
            start_state,
            act_seq,
            start_state_idx=start_state_idx,
            goal_state=goal_state,
            goal_state_idx=goal_state_idx,
            use_implicit_goal_state=use_implicit_goal_state,
        )

        assert isinstance(result, RobotState)
        assert result.joint_state.position.shape == (batch_size, horizon, transition.num_dof)
        assert result.joint_state.velocity is not None
        assert result.joint_state.acceleration is not None
        assert result.joint_state.jerk is not None

    def test_forward_clique_computes_finite_differences(
        self, position_clique_transition_cfg, cuda_device_cfg
    ):
        """Test that clique forward computes velocity/acceleration from positions."""
        transition = RobotStateTransition(position_clique_transition_cfg)
        batch_size = 2
        horizon = transition.horizon

        start_state, start_state_idx = create_start_state_with_idx(
            transition, batch_size, cuda_device_cfg
        )
        goal_state, goal_state_idx, use_implicit_goal_state = create_goal_state_with_idx(
            transition, batch_size, cuda_device_cfg, horizon
        )

        # Linear position trajectory
        act_seq = torch.linspace(0, 1, transition.action_horizon, **cuda_device_cfg.as_torch_dict())
        act_seq = act_seq.view(1, -1, 1).expand(batch_size, -1, transition.action_dim).contiguous()

        result = transition.forward(
            start_state,
            act_seq,
            start_state_idx=start_state_idx,
            goal_state=goal_state,
            goal_state_idx=goal_state_idx,
            use_implicit_goal_state=use_implicit_goal_state,
        )

        # Velocity should be computed from position differences
        assert result.joint_state.velocity is not None
        assert not torch.all(result.joint_state.velocity == 0)


class TestRobotStateTransitionForwardBSpline:
    """Test forward method with B-spline control space (StateFromBSplineKnot)."""

    def test_forward_bspline_with_goal_state(self, bspline_transition_cfg, cuda_device_cfg):
        """Test forward with B-spline using goal state."""
        transition = RobotStateTransition(bspline_transition_cfg)
        batch_size = 4
        horizon = transition.horizon

        # Create start state with full state (position, velocity, acceleration, jerk)
        num_dof = transition.num_dof
        start_state = JointState.zeros((1, num_dof), cuda_device_cfg)
        start_state.position = torch.randn_like(start_state.position) * 0.1
        start_state.velocity = torch.zeros_like(start_state.position)
        start_state.acceleration = torch.zeros_like(start_state.position)
        start_state.jerk = torch.zeros_like(start_state.position)

        start_state_idx = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)

        # Create goal state with jerk (required for bspline)
        goal_state = JointState.zeros((1, horizon, num_dof), cuda_device_cfg)
        goal_state.position = torch.randn_like(goal_state.position) * 0.1
        goal_state.velocity = torch.zeros_like(goal_state.position)
        goal_state.acceleration = torch.zeros_like(goal_state.position)
        goal_state.jerk = torch.zeros_like(goal_state.position)
        goal_state.dt = torch.ones(
            1, horizon, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
        ) * transition.dt

        goal_state_idx = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)
        use_implicit_goal_state = torch.zeros(
            1, horizon, device=cuda_device_cfg.device, dtype=torch.uint8
        )

        # Action is knot positions
        n_knots = transition.n_knots
        act_seq = torch.randn(
            batch_size, n_knots, transition.action_dim,
            **cuda_device_cfg.as_torch_dict()
        ) * 0.1

        result = transition.forward(
            start_state,
            act_seq,
            start_state_idx=start_state_idx,
            goal_state=goal_state,
            goal_state_idx=goal_state_idx,
            use_implicit_goal_state=use_implicit_goal_state,
        )

        assert isinstance(result, RobotState)
        assert result.joint_state.position.shape == (batch_size, horizon, num_dof)

    def test_forward_bspline_smooth_trajectory(self, bspline_transition_cfg, cuda_device_cfg):
        """Test that B-spline produces smooth trajectories."""
        transition = RobotStateTransition(bspline_transition_cfg)
        batch_size = 1
        horizon = transition.horizon
        num_dof = transition.num_dof

        # Setup states
        start_state = JointState.zeros((1, num_dof), cuda_device_cfg)
        start_state.jerk = torch.zeros_like(start_state.position)
        start_state_idx = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)

        goal_state = JointState.zeros((1, horizon, num_dof), cuda_device_cfg)
        goal_state.jerk = torch.zeros_like(goal_state.position)
        goal_state.dt = torch.ones(
            1, horizon, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
        )

        goal_state_idx = torch.zeros(batch_size, device=cuda_device_cfg.device, dtype=torch.int32)
        use_implicit_goal_state = torch.zeros(
            1, horizon, device=cuda_device_cfg.device, dtype=torch.uint8
        )

        # Smooth knot sequence
        n_knots = transition.n_knots
        knots = torch.linspace(0, 1, n_knots, **cuda_device_cfg.as_torch_dict())
        act_seq = knots.view(1, -1, 1).expand(batch_size, -1, num_dof).contiguous()

        result = transition.forward(
            start_state,
            act_seq,
            start_state_idx=start_state_idx,
            goal_state=goal_state,
            goal_state_idx=goal_state_idx,
            use_implicit_goal_state=use_implicit_goal_state,
        )

        # B-spline should produce smooth output (bounded jerk).
        # With dt=0.02 and knots in [0,1], jerk ~ O(1/dt^3) ~ O(10^5),
        # so a smooth ramp should stay well below that.
        max_jerk = torch.abs(result.joint_state.jerk).max()
        assert max_jerk < 100000


# ============================================================================
# Test Classes - Properties
# ============================================================================


class TestRobotStateTransitionProperties:
    """Test RobotStateTransition properties."""

    def test_horizon_property(self, teleport_transition_cfg):
        """Test horizon property."""
        transition = RobotStateTransition(teleport_transition_cfg)
        assert transition.horizon == teleport_transition_cfg.horizon

    def test_n_knots_property(self, bspline_transition_cfg):
        """Test n_knots property for bspline."""
        transition = RobotStateTransition(bspline_transition_cfg)
        assert transition.n_knots == bspline_transition_cfg.n_knots

    def test_control_space_property(self, teleport_transition_cfg):
        """Test control_space property."""
        transition = RobotStateTransition(teleport_transition_cfg)
        assert transition.control_space == ControlSpace.POSITION

    def test_action_horizon_teleport(self, teleport_transition_cfg):
        """Test action_horizon for teleport mode."""
        transition = RobotStateTransition(teleport_transition_cfg)
        # Teleport: action_horizon == horizon
        assert transition.action_horizon == transition.horizon

    def test_action_horizon_clique(self, position_clique_transition_cfg):
        """Test action_horizon for clique mode."""
        transition = RobotStateTransition(position_clique_transition_cfg)
        # Clique: action_horizon == horizon - 4
        assert transition.action_horizon == transition.horizon - 4

    def test_action_horizon_bspline(self, bspline_transition_cfg):
        """Test action_horizon for bspline mode."""
        transition = RobotStateTransition(bspline_transition_cfg)
        # BSpline: action_horizon == n_knots
        assert transition.action_horizon == transition.n_knots

    def test_d_state_is_4_times_num_dof(self, teleport_transition_cfg):
        """Test that d_state is 4 * num_dof."""
        transition = RobotStateTransition(teleport_transition_cfg)
        assert transition.d_state == 4 * transition.num_dof


# ============================================================================
# Test Classes - Batch Size Updates
# ============================================================================


class TestRobotStateTransitionUpdateBatchSize:
    """Test batch size update functionality."""

    def test_update_batch_size(self, teleport_transition_cfg):
        """Test updating batch size."""
        transition = RobotStateTransition(teleport_transition_cfg)

        new_batch_size = 8
        transition.update_batch_size(new_batch_size)

        assert transition.batch_size == new_batch_size
        assert transition.state_seq.position.shape[0] == new_batch_size

    def test_update_batch_size_updates_step_fn(self, acceleration_transition_cfg):
        """Test that update_batch_size updates the step function."""
        transition = RobotStateTransition(acceleration_transition_cfg)

        new_batch_size = 16
        transition.update_batch_size(new_batch_size)

        assert transition._rollout_step_fn.batch_size == new_batch_size


# ============================================================================
# Test Classes - Action Bounds
# ============================================================================


class TestRobotStateTransitionActionBounds:
    """Test action bound properties."""

    def test_action_bound_lows_position(self, teleport_transition_cfg):
        """Test action_bound_lows for position control space."""
        transition = RobotStateTransition(teleport_transition_cfg)
        lows = transition.action_bound_lows

        assert lows is not None
        assert isinstance(lows, torch.Tensor)
        assert lows.shape[-1] == transition.num_dof

    def test_action_bound_highs_position(self, teleport_transition_cfg):
        """Test action_bound_highs for position control space."""
        transition = RobotStateTransition(teleport_transition_cfg)
        highs = transition.action_bound_highs

        assert highs is not None
        assert isinstance(highs, torch.Tensor)

    def test_action_bounds_acceleration(self, acceleration_transition_cfg):
        """Test action bounds for acceleration control space."""
        transition = RobotStateTransition(acceleration_transition_cfg)

        lows = transition.action_bound_lows
        highs = transition.action_bound_highs

        assert lows is not None
        assert highs is not None
        assert torch.all(lows < highs)


# ============================================================================
# Test Classes - Get Robot Command
# ============================================================================


class TestRobotStateTransitionGetRobotCommand:
    """Test get_robot_command method."""

    def test_get_robot_command_teleport(self, teleport_transition_cfg, cuda_device_cfg):
        """Test get_robot_command with teleport mode."""
        transition = RobotStateTransition(teleport_transition_cfg)
        batch_size = 2

        current_state = JointState.zeros((batch_size, transition.num_dof), cuda_device_cfg)
        act_seq = torch.randn(
            batch_size, transition.action_horizon, transition.action_dim,
            **cuda_device_cfg.as_torch_dict()
        )

        result = transition.get_robot_command(current_state, act_seq, shift_steps=1)

        assert result is not None
        assert result.position is not None

    def test_get_robot_command_multiple_shift_steps(self, teleport_transition_cfg, cuda_device_cfg):
        """Test get_robot_command with multiple shift steps."""
        transition = RobotStateTransition(teleport_transition_cfg)
        batch_size = 2

        current_state = JointState.zeros((batch_size, transition.num_dof), cuda_device_cfg)
        act_seq = torch.randn(
            batch_size, transition.action_horizon, transition.action_dim,
            **cuda_device_cfg.as_torch_dict()
        )

        result = transition.get_robot_command(current_state, act_seq, shift_steps=3)

        assert result is not None


# ============================================================================
# Test Classes - Dynamics
# ============================================================================


class TestRobotStateTransitionWithDynamics:
    """Test RobotStateTransition with dynamics enabled."""

    @pytest.fixture
    def dynamics_robot_cfg(self, franka_robot_cfg_dict, cuda_device_cfg):
        """Create RobotCfg with dynamics enabled."""
        cfg_dict = franka_robot_cfg_dict.copy()
        cfg_dict["load_dynamics"] = True
        return RobotCfg.create({"robot_cfg": cfg_dict}, cuda_device_cfg)

    @pytest.fixture
    def dynamics_transition_cfg(
        self, dynamics_robot_cfg, time_traj_cfg, cuda_device_cfg, state_filter_cfg
    ):
        """Create RobotStateTransitionCfg with dynamics."""
        return RobotStateTransitionCfg(
            robot_config=dynamics_robot_cfg,
            dt_traj_params=time_traj_cfg,
            device_cfg=cuda_device_cfg,
            control_space=ControlSpace.POSITION,
            teleport_mode=True,
            batch_size=2,
            horizon=10,
            state_filter_cfg=state_filter_cfg,
        )

    def test_dynamics_model_created(self, dynamics_transition_cfg):
        """Test that dynamics model is created when config has dynamics."""
        transition = RobotStateTransition(dynamics_transition_cfg)

        assert transition.robot_dynamics is not None
        assert transition.compute_inverse_dynamics is True

    def test_forward_with_dynamics_computes_torque(self, dynamics_transition_cfg, cuda_device_cfg):
        """Test that forward with dynamics computes joint torque."""
        transition = RobotStateTransition(dynamics_transition_cfg)
        batch_size = 2

        start_state = JointState.zeros((1, transition.num_dof), cuda_device_cfg)
        act_seq = torch.randn(
            batch_size, transition.action_horizon, transition.action_dim,
            **cuda_device_cfg.as_torch_dict()
        ) * 0.1

        result = transition.forward(start_state, act_seq, start_state_idx=None)

        assert result.joint_torque is not None
        # With dynamics, torque should be computed (not all zeros for non-trivial motion)


class TestRobotStateTransitionWithoutDynamics:
    """Test RobotStateTransition without dynamics."""

    def test_no_dynamics_model(self, teleport_transition_cfg):
        """Test that no dynamics model when not configured."""
        transition = RobotStateTransition(teleport_transition_cfg)

        assert transition.robot_dynamics is None
        assert transition.compute_inverse_dynamics is False


# ============================================================================
# Test Classes - Compute Augmented State
# ============================================================================


class TestRobotStateTransitionComputeAugmentedState:
    """Test compute_augmented_state method."""

    def test_compute_augmented_state_3d(self, teleport_transition_cfg, cuda_device_cfg):
        """Test compute_augmented_state with 3D input."""
        transition = RobotStateTransition(teleport_transition_cfg)

        batch_size = 2
        horizon = 5
        state_seq = JointState.zeros((batch_size, horizon, transition.num_dof), cuda_device_cfg)
        state_seq.position = torch.randn_like(state_seq.position) * 0.1

        result = transition.compute_augmented_state(state_seq)

        assert isinstance(result, RobotState)
        assert result.joint_state is not None
        assert result.cuda_robot_model_state is not None

    def test_compute_augmented_state_2d_adds_dimension(
        self, teleport_transition_cfg, cuda_device_cfg
    ):
        """Test compute_augmented_state with 2D input adds dimension."""
        transition = RobotStateTransition(teleport_transition_cfg)

        batch_size = 2
        state_seq = JointState.zeros((batch_size, transition.num_dof), cuda_device_cfg)
        state_seq.position = torch.randn_like(state_seq.position)

        result = transition.compute_augmented_state(state_seq)

        assert isinstance(result, RobotState)
        assert len(result.joint_state.shape) == 3
        assert result.joint_state.shape[1] == 1


# ============================================================================
# Test Classes - Get Action From State
# ============================================================================


class TestRobotStateTransitionGetActionFromState:
    """Test get_action_from_state method."""

    def test_get_action_position_control_space(self, teleport_transition_cfg, cuda_device_cfg):
        """Test get_action_from_state for position control space."""
        transition = RobotStateTransition(teleport_transition_cfg)

        position = torch.randn(1, transition.num_dof, **cuda_device_cfg.as_torch_dict())
        state = JointState.from_position(position)

        action = transition.get_action_from_state(state)

        assert torch.allclose(action, position)

    def test_get_action_acceleration_control_space(
        self, acceleration_transition_cfg, cuda_device_cfg
    ):
        """Test get_action_from_state for acceleration control space."""
        transition = RobotStateTransition(acceleration_transition_cfg)

        state = JointState.zeros((1, transition.num_dof), cuda_device_cfg)
        state.acceleration = torch.randn_like(state.position)

        action = transition.get_action_from_state(state)

        assert torch.allclose(action, state.acceleration)


# ============================================================================
# Test Classes - Init Action Mean
# ============================================================================


class TestRobotStateTransitionInitActionMean:
    """Test init_action_mean property and get_init_action_mean method."""

    def test_init_action_mean_position(self, teleport_transition_cfg):
        """Test init_action_mean for position control space."""
        transition = RobotStateTransition(teleport_transition_cfg)

        init_mean = transition.init_action_mean

        assert init_mean is not None
        assert init_mean.shape[0] == transition.action_horizon
        assert init_mean.shape[1] == transition.num_dof

    def test_init_action_mean_acceleration(self, acceleration_transition_cfg):
        """Test init_action_mean for acceleration control space returns zeros."""
        transition = RobotStateTransition(acceleration_transition_cfg)

        init_mean = transition.init_action_mean

        assert init_mean is not None
        assert torch.allclose(init_mean, torch.zeros_like(init_mean))

    def test_get_init_action_mean_equals_property(self, teleport_transition_cfg):
        """Test that get_init_action_mean returns same as property."""
        transition = RobotStateTransition(teleport_transition_cfg)

        prop_mean = transition.init_action_mean
        method_mean = transition.get_init_action_mean()

        assert torch.allclose(prop_mean, method_mean)

    def test_get_init_action_mean_velocity_control_space(
        self, franka_robot_cfg, time_traj_cfg, cuda_device_cfg, state_filter_cfg
    ):
        """Test get_init_action_mean for velocity control space returns zeros."""
        # Note: VELOCITY control space raises NotImplementedError in init
        # So we test with acceleration which has same logic
        transition = RobotStateTransition(
            RobotStateTransitionCfg(
                robot_config=franka_robot_cfg,
                dt_traj_params=time_traj_cfg,
                device_cfg=cuda_device_cfg,
                control_space=ControlSpace.ACCELERATION,
                batch_size=2,
                horizon=10,
                state_filter_cfg=state_filter_cfg,
            )
        )

        method_mean = transition.get_init_action_mean()
        assert torch.allclose(method_mean, torch.zeros_like(method_mean))


# ============================================================================
# Test Classes - Update Traj Dt
# ============================================================================


class TestRobotStateTransitionUpdateTrajDt:
    """Test update_traj_dt method."""

    def test_update_traj_dt_with_float(self, position_clique_transition_cfg):
        """Test update_traj_dt with float value."""
        transition = RobotStateTransition(position_clique_transition_cfg)

        original_dt = transition.dt
        new_dt = 0.05

        transition.update_traj_dt(new_dt)

        assert transition.config.dt_traj_params.base_dt == new_dt

    def test_update_traj_dt_with_base_dt(self, position_clique_transition_cfg):
        """Test update_traj_dt with base_dt parameter."""
        transition = RobotStateTransition(position_clique_transition_cfg)

        new_base_dt = 0.03
        transition.update_traj_dt(dt=None, base_dt=new_base_dt)

        assert transition.config.dt_traj_params.base_dt == new_base_dt

    def test_update_traj_dt_updates_step_functions(self, position_clique_transition_cfg):
        """Test that update_traj_dt updates both step functions."""
        transition = RobotStateTransition(position_clique_transition_cfg)

        new_dt = 0.04
        transition.update_traj_dt(new_dt)

        # Verify traj_dt tensor was updated
        assert transition.traj_dt[0].item() == pytest.approx(new_dt, rel=1e-5)


# ============================================================================
# Test Classes - Robot Cmd Tensor Step
# ============================================================================


class TestRobotStateTransitionRobotCmdTensorStep:
    """Test robot_cmd_tensor_step method."""

    def test_robot_cmd_tensor_step_sets_joint_names(self, teleport_transition_cfg, cuda_device_cfg):
        """Test robot_cmd_tensor_step sets joint names on output."""
        transition = RobotStateTransition(teleport_transition_cfg)
        batch_size = 2

        start_state = JointState.zeros((1, transition.num_dof), cuda_device_cfg)
        act_seq = torch.randn(
            batch_size, transition.action_horizon, transition.action_dim,
            **cuda_device_cfg.as_torch_dict()
        )

        transition.update_cmd_batch_size(batch_size)
        result = transition.robot_cmd_tensor_step(
            start_state, act_seq, transition._robot_cmd_state_seq, state_idx=None
        )

        assert result.joint_names == transition.joint_names


# ============================================================================
# Test Classes - Update Cmd Batch Size
# ============================================================================


class TestRobotStateTransitionUpdateCmdBatchSize:
    """Test update_cmd_batch_size method."""

    def test_update_cmd_batch_size(self, teleport_transition_cfg):
        """Test update_cmd_batch_size creates state sequence."""
        transition = RobotStateTransition(teleport_transition_cfg)

        new_batch_size = 8
        transition.update_cmd_batch_size(new_batch_size)

        assert transition._cmd_batch_size == new_batch_size
        assert transition._robot_cmd_state_seq.position.shape[0] == new_batch_size

    def test_update_cmd_batch_size_no_change(self, teleport_transition_cfg):
        """Test update_cmd_batch_size with same size does nothing."""
        transition = RobotStateTransition(teleport_transition_cfg)

        new_batch_size = 8
        transition.update_cmd_batch_size(new_batch_size)
        original_seq = transition._robot_cmd_state_seq

        transition.update_cmd_batch_size(new_batch_size)
        assert transition._robot_cmd_state_seq is original_seq


# ============================================================================
# Test Classes - Filter Robot State
# ============================================================================


class TestRobotStateTransitionFilterRobotState:
    """Test filter_robot_state method."""

    def test_filter_robot_state(self, teleport_transition_cfg, cuda_device_cfg):
        """Test filter_robot_state method."""
        transition = RobotStateTransition(teleport_transition_cfg)

        state = JointState.zeros((1, transition.num_dof), cuda_device_cfg)
        state.position = torch.randn_like(state.position)

        filtered = transition.filter_robot_state(state)

        assert filtered is not None
        assert filtered.position is not None


# ============================================================================
# Test Classes - Get Robot Command Extended
# ============================================================================


class TestRobotStateTransitionGetRobotCommandExtended:
    """Extended tests for get_robot_command method."""

    @pytest.fixture
    def return_full_buffer_cfg(
        self, franka_robot_cfg, time_traj_cfg, cuda_device_cfg, state_filter_cfg
    ):
        """Create config with return_full_act_buffer=True."""
        return RobotStateTransitionCfg(
            robot_config=franka_robot_cfg,
            dt_traj_params=time_traj_cfg,
            device_cfg=cuda_device_cfg,
            control_space=ControlSpace.POSITION,
            teleport_mode=True,
            batch_size=2,
            horizon=10,
            state_filter_cfg=state_filter_cfg,
            return_full_act_buffer=True,
        )

    def test_get_robot_command_full_buffer(self, return_full_buffer_cfg, cuda_device_cfg):
        """Test get_robot_command with return_full_act_buffer=True."""
        transition = RobotStateTransition(return_full_buffer_cfg)
        batch_size = 2

        current_state = JointState.zeros((1, transition.num_dof), cuda_device_cfg)
        act_seq = torch.randn(
            batch_size, transition.action_horizon, transition.action_dim,
            **cuda_device_cfg.as_torch_dict()
        )

        result = transition.get_robot_command(current_state, act_seq)

        assert result is not None
        assert result.position is not None
        assert result.joint_names == transition.joint_names

    def test_get_robot_command_acceleration_shift_1(
        self, acceleration_transition_cfg, cuda_device_cfg
    ):
        """Test get_robot_command with acceleration control space and shift_steps=1."""
        transition = RobotStateTransition(acceleration_transition_cfg)
        batch_size = 2

        current_state = JointState.zeros((batch_size, transition.num_dof), cuda_device_cfg)
        act_seq = torch.randn(
            batch_size, transition.action_horizon, transition.action_dim,
            **cuda_device_cfg.as_torch_dict()
        )

        result = transition.get_robot_command(current_state, act_seq, shift_steps=1)

        assert result is not None
        assert result.position is not None


# ============================================================================
# Test Classes - Properties Extended
# ============================================================================


class TestRobotStateTransitionPropertiesExtended:
    """Extended property tests."""

    def test_cspace_distance_weight(self, teleport_transition_cfg):
        """Test cspace_distance_weight property."""
        transition = RobotStateTransition(teleport_transition_cfg)
        weight = transition.cspace_distance_weight
        assert weight is not None

    def test_null_space_weight(self, teleport_transition_cfg):
        """Test null_space_weight property."""
        transition = RobotStateTransition(teleport_transition_cfg)
        weight = transition.null_space_weight
        assert weight is not None

    def test_null_space_maximum_distance(self, teleport_transition_cfg):
        """Test null_space_maximum_distance property."""
        transition = RobotStateTransition(teleport_transition_cfg)
        # May be None depending on config
        _ = transition.null_space_maximum_distance

    def test_max_acceleration(self, teleport_transition_cfg):
        """Test max_acceleration property."""
        transition = RobotStateTransition(teleport_transition_cfg)
        max_acc = transition.max_acceleration

        assert max_acc is not None
        assert isinstance(max_acc, torch.Tensor)
        assert max_acc.shape[-1] == transition.num_dof

    def test_max_jerk(self, teleport_transition_cfg):
        """Test max_jerk property."""
        transition = RobotStateTransition(teleport_transition_cfg)
        max_jerk = transition.max_jerk

        assert max_jerk is not None
        assert isinstance(max_jerk, torch.Tensor)
        assert max_jerk.shape[-1] == transition.num_dof

    def test_max_velocity(self, teleport_transition_cfg):
        """Test max_velocity property."""
        transition = RobotStateTransition(teleport_transition_cfg)
        max_vel = transition.max_velocity

        assert max_vel is not None
        assert isinstance(max_vel, torch.Tensor)
        assert max_vel.shape[-1] == transition.num_dof

    def test_state_finite_difference_mode(self, teleport_transition_cfg):
        """Test state_finite_difference_mode property."""
        transition = RobotStateTransition(teleport_transition_cfg)
        mode = transition.state_finite_difference_mode
        assert mode is not None

    def test_filter_robot_command_property(self, teleport_transition_cfg):
        """Test filter_robot_command property."""
        transition = RobotStateTransition(teleport_transition_cfg)
        # Default should be False
        assert transition.filter_robot_command is False

    def test_default_joint_position(self, teleport_transition_cfg):
        """Test default_joint_position property."""
        transition = RobotStateTransition(teleport_transition_cfg)
        default_position = transition.default_joint_position

        assert default_position is not None
        assert isinstance(default_position, torch.Tensor)
        assert default_position.shape[-1] == transition.num_dof

    def test_device_cfg(self, teleport_transition_cfg, cuda_device_cfg):
        """Test device_cfg property."""
        transition = RobotStateTransition(teleport_transition_cfg)
        assert transition.device_cfg.device == cuda_device_cfg.device
        assert transition.device_cfg.dtype == cuda_device_cfg.dtype

    def test_return_full_act_buffer(self, teleport_transition_cfg):
        """Test return_full_act_buffer property."""
        transition = RobotStateTransition(teleport_transition_cfg)
        # Check it returns a boolean
        assert isinstance(transition.return_full_act_buffer, bool)


# ============================================================================
# Test Classes - Get State From Action
# ============================================================================


class TestRobotStateTransitionGetStateFromAction:
    """Test get_state_from_action method."""

    def test_get_state_from_action(self, teleport_transition_cfg, cuda_device_cfg):
        """Test get_state_from_action returns joint state."""
        transition = RobotStateTransition(teleport_transition_cfg)
        batch_size = 2

        start_state = JointState.zeros((1, transition.num_dof), cuda_device_cfg)
        act_seq = torch.randn(
            batch_size, transition.action_horizon, transition.action_dim,
            **cuda_device_cfg.as_torch_dict()
        )

        result = transition.get_state_from_action(start_state, act_seq)

        assert result is not None
        assert result.position is not None
        assert result.position.shape[0] == batch_size

    def test_get_state_from_action_updates_cmd_batch_size(
        self, teleport_transition_cfg, cuda_device_cfg
    ):
        """Test get_state_from_action updates cmd batch size."""
        transition = RobotStateTransition(teleport_transition_cfg)
        batch_size = 8

        start_state = JointState.zeros((1, transition.num_dof), cuda_device_cfg)
        act_seq = torch.randn(
            batch_size, transition.action_horizon, transition.action_dim,
            **cuda_device_cfg.as_torch_dict()
        )

        transition.get_state_from_action(start_state, act_seq)

        assert transition._cmd_batch_size == batch_size


# ============================================================================
# Test Classes - Dynamics Methods
# ============================================================================


class TestRobotStateTransitionDynamicsMethods:
    """Test dynamics-related methods."""

    @pytest.fixture
    def dynamics_robot_cfg(self, franka_robot_cfg_dict, cuda_device_cfg):
        """Create RobotCfg with dynamics enabled."""
        cfg_dict = franka_robot_cfg_dict.copy()
        cfg_dict["load_dynamics"] = True
        return RobotCfg.create({"robot_cfg": cfg_dict}, cuda_device_cfg)

    @pytest.fixture
    def dynamics_transition_cfg(
        self, dynamics_robot_cfg, time_traj_cfg, cuda_device_cfg, state_filter_cfg
    ):
        """Create RobotStateTransitionCfg with dynamics."""
        return RobotStateTransitionCfg(
            robot_config=dynamics_robot_cfg,
            dt_traj_params=time_traj_cfg,
            device_cfg=cuda_device_cfg,
            control_space=ControlSpace.POSITION,
            teleport_mode=True,
            batch_size=2,
            horizon=10,
            state_filter_cfg=state_filter_cfg,
        )

    def test_update_link_mass_with_dynamics(self, dynamics_transition_cfg):
        """Test update_link_mass when dynamics is available."""
        transition = RobotStateTransition(dynamics_transition_cfg)

        # Should not raise - method is callable with dynamics
        try:
            transition.update_link_mass("panda_link3", 2.0)
        except Exception as e:
            # May fail if link name is invalid, but not due to missing dynamics
            assert "dynamics" not in str(e).lower()

    def test_update_link_inertial_with_dynamics(self, dynamics_transition_cfg):
        """Test update_link_inertial when dynamics is available."""
        transition = RobotStateTransition(dynamics_transition_cfg)

        try:
            transition.update_link_inertial("panda_link3", mass=2.0)
        except Exception as e:
            assert "dynamics" not in str(e).lower()

    def test_update_links_inertial_with_dynamics(self, dynamics_transition_cfg):
        """Test update_links_inertial when dynamics is available."""
        transition = RobotStateTransition(dynamics_transition_cfg)

        try:
            transition.update_links_inertial({"panda_link3": {"mass": 2.0}})
        except Exception as e:
            assert "dynamics" not in str(e).lower()

    def test_update_link_mass_raises_without_dynamics(self, teleport_transition_cfg):
        """Test update_link_mass raises without dynamics."""
        transition = RobotStateTransition(teleport_transition_cfg)

        with pytest.raises(Exception):
            transition.update_link_mass("panda_link1", 1.0)

    def test_update_link_inertial_raises_without_dynamics(self, teleport_transition_cfg):
        """Test update_link_inertial raises without dynamics."""
        transition = RobotStateTransition(teleport_transition_cfg)

        with pytest.raises(Exception):
            transition.update_link_inertial("panda_link1", mass=1.0)

    def test_update_links_inertial_raises_without_dynamics(self, teleport_transition_cfg):
        """Test update_links_inertial raises without dynamics."""
        transition = RobotStateTransition(teleport_transition_cfg)

        with pytest.raises(Exception):
            transition.update_links_inertial({"panda_link1": {"mass": 1.0}})


# ============================================================================
# Test Classes - Get Full DOF From Solution
# ============================================================================


class TestRobotStateTransitionGetFullDofFromSolution:
    """Test get_full_dof_from_solution method."""

    def test_get_full_dof_returns_joint_state(self, teleport_transition_cfg, cuda_device_cfg):
        """Test get_full_dof_from_solution returns JointState."""
        transition = RobotStateTransition(teleport_transition_cfg)

        q_js = JointState.zeros((1, transition.num_dof), cuda_device_cfg)
        q_js.position = torch.randn_like(q_js.position)
        q_js.joint_names = transition.joint_names

        result = transition.get_full_dof_from_solution(q_js)

        assert result is not None
        assert result.position is not None

    def test_get_full_dof_no_locked_joints_returns_same(
        self, teleport_transition_cfg, cuda_device_cfg
    ):
        """Test get_full_dof returns same when no locked joints."""
        transition = RobotStateTransition(teleport_transition_cfg)

        # If no locked joints, should return same state
        if transition.robot_model.lock_jointstate is None:
            q_js = JointState.zeros((1, transition.num_dof), cuda_device_cfg)
            q_js.position = torch.randn_like(q_js.position)
            q_js.joint_names = transition.joint_names

            result = transition.get_full_dof_from_solution(q_js)
            assert result is q_js


# ============================================================================
# Test Classes - State Bounds
# ============================================================================


class TestRobotStateTransitionStateBounds:
    """Test get_state_bounds method."""

    def test_get_state_bounds(self, teleport_transition_cfg):
        """Test get_state_bounds returns joint limits."""
        transition = RobotStateTransition(teleport_transition_cfg)

        bounds = transition.get_state_bounds()

        assert bounds is not None
        assert bounds.position is not None
        assert bounds.velocity is not None
        assert bounds.acceleration is not None

    def test_state_bounds_shape(self, teleport_transition_cfg):
        """Test state bounds have correct shape."""
        transition = RobotStateTransition(teleport_transition_cfg)

        bounds = transition.get_state_bounds()

        assert bounds.position.shape[-1] == transition.num_dof


# ============================================================================
# Test Classes - Joint Names and Limits
# ============================================================================


class TestRobotStateTransitionJointNamesLimits:
    """Test joint names and limits."""

    def test_joint_names(self, teleport_transition_cfg):
        """Test joint_names property."""
        transition = RobotStateTransition(teleport_transition_cfg)

        assert transition.joint_names is not None
        assert len(transition.joint_names) == transition.num_dof

    def test_joint_limits(self, teleport_transition_cfg):
        """Test joint_limits property."""
        transition = RobotStateTransition(teleport_transition_cfg)

        assert transition.joint_limits is not None
        assert transition.joint_limits.position is not None
