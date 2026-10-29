# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fit graph paths to uniformly timed B-spline controls used by TrajOpt.

Graph-planner waypoint indices do not represent trajectory time.  This module first
reparameterizes each fixed-size graph polyline by cumulative joint-space arc length,
then samples that geometry over the executable interval of TrajOpt's B-spline
rollout.  Endpoint samples outside that interval are held at the start and goal to
match the spline boundary convention.

The fitter first computes an affine, acceleration-regularized geometric fit,
then applies a fixed number of IRLS updates that reduce the uniform timestep
required by velocity, acceleration, and jerk limits. Runtime fitting is
allocation-free and captured as one fixed-shape CUDA graph.
"""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

# Third Party
import torch

# CuRobo
from curobo._src.curobolib.backends import trajectory as trajectory_cu
from curobo._src.curobolib.backends import trajectory_seed_fitter as seed_fitter_cu
from curobo._src.state.state_joint import JointState
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.cuda_graph_util import GraphExecutor

if TYPE_CHECKING:
    # CuRobo
    from curobo._src.transition.robot_state_transition import RobotStateTransition


@dataclass(frozen=True)
class BSplineTimeOptimalSeedFitterCfg:
    """Configuration for a fixed-shape time-optimized graph-path fit.

    Args:
        n_knots: Number of action knots consumed by TrajOpt.
        interpolation_steps: Number of rollout samples per padded B-spline knot.
        control_space: TrajOpt B-spline degree.
        rollout_horizon: Dense position horizon produced by the rollout kernel.
        num_lanes: Fixed number of independent graph paths fitted together.
        dof: Number of position dimensions per path.
        device_cfg: CUDA device and floating-point dtype.
        use_cuda_graph: Capture and replay the fixed-shape fit.
        regularization: Relative diagonal jitter applied to the normal matrix.
            The value is scaled by the mean position-normal diagonal.
        acceleration_regularization: Dimensionless acceleration penalty.  The
            acceleration normal is trace-normalized against the position normal,
            so this weight is insensitive to the kernel's derivative units. The
            default favors a smooth initialization over closely reproducing every
            piecewise-linear graph-path corner.
        trajectory_dt: Initial per-sample trajectory dt used for the boundary
            constraints.  TrajOpt also starts seed evaluation at ``1.0`` before
            estimating a trajectory-specific dt.
        irls_iterations: Fixed number of reweighted least-squares updates.
        timing_regularization: Strength of the normalized derivative objective.
        trust_region_regularization: Penalty on movement from accepted knots.
        irls_epsilon: Positive floor used when building IRLS weights.
        dt_epsilon: Safety margin applied to the reported required timestep.
    """

    n_knots: int
    interpolation_steps: int
    control_space: ControlSpace
    rollout_horizon: int
    num_lanes: int
    dof: int
    device_cfg: DeviceCfg
    use_cuda_graph: bool = True
    regularization: float = 1.0e-6
    acceleration_regularization: float = 5.0e-1
    trajectory_dt: float = 1.0
    irls_iterations: int = 2
    timing_regularization: float = 10.0
    trust_region_regularization: float = 5.0e-2
    irls_epsilon: float = 1.0e-6
    dt_epsilon: float = 1.0e-3

    def __post_init__(self) -> None:
        """Validate representation and fixed-shape constraints."""
        if self.control_space not in ControlSpace.bspline_types():
            raise ValueError("control_space must be a B-spline control space")
        if self.n_knots < 2:
            raise ValueError("n_knots must be at least 2")
        if self.interpolation_steps < 1:
            raise ValueError("interpolation_steps must be positive")
        if self.num_lanes < 1:
            raise ValueError("num_lanes must be positive")
        if self.dof < 1:
            raise ValueError("dof must be positive")
        if self.regularization < 0.0:
            raise ValueError("regularization must be non-negative")
        if self.acceleration_regularization < 0.0:
            raise ValueError("acceleration_regularization must be non-negative")
        if self.trajectory_dt <= 0.0:
            raise ValueError("trajectory_dt must be positive")
        if self.irls_iterations < 1:
            raise ValueError("irls_iterations must be positive")
        if self.timing_regularization < 0.0:
            raise ValueError("timing_regularization must be non-negative")
        if self.trust_region_regularization < 0.0:
            raise ValueError("trust_region_regularization must be non-negative")
        if self.irls_epsilon <= 0.0:
            raise ValueError("irls_epsilon must be positive")
        if self.dt_epsilon < 0.0:
            raise ValueError("dt_epsilon must be non-negative")

        expected_horizon = ControlSpace.spline_total_interpolation_steps(
            self.control_space,
            self.n_knots,
            self.interpolation_steps,
        )
        if self.rollout_horizon != expected_horizon:
            raise ValueError(
                f"rollout_horizon must be {expected_horizon} for "
                f"{self.control_space.name}, n_knots={self.n_knots}, and "
                f"interpolation_steps={self.interpolation_steps}; got "
                f"{self.rollout_horizon}"
            )
        if torch.device(self.device_cfg.device).type != "cuda":
            raise ValueError("BSplineTimeOptimalSeedFitter requires a CUDA device")
        if self.device_cfg.dtype != torch.float32:
            raise ValueError("BSplineTimeOptimalSeedFitter requires float32")


class BSplineTimeOptimalSeedFitter:
    """GPU time-optimized arc-length fitter for TrajOpt B-spline knots.

    An instance owns fixed-address buffers for exactly ``num_lanes`` paths.  Create
    another instance when the lane count changes.  Returned storage is owned by the
    fitter and is overwritten by the next call to :meth:`fit`; clone it when it must
    survive another fit. Each path is fitted independently of other batch lanes.
    """

    def __init__(
        self,
        config: BSplineTimeOptimalSeedFitterCfg,
        max_velocity: torch.Tensor,
        max_acceleration: torch.Tensor,
        max_jerk: torch.Tensor,
    ) -> None:
        """Create a fixed-shape fitter without compiling or capturing kernels."""
        self.config = config
        self.degree = ControlSpace.spline_degree(config.control_space)
        self.n_free_knots = config.n_knots - 1
        self._device = torch.device(config.device_cfg.device)
        tensor_args = {
            "device": self._device,
            "dtype": config.device_cfg.dtype,
        }

        lanes = config.num_lanes
        horizon = config.rollout_horizon
        dof = config.dof
        n_knots = config.n_knots

        expected_limit_shape = (dof,)
        limits = (max_velocity, max_acceleration, max_jerk)
        for name, value in zip(
            ("max_velocity", "max_acceleration", "max_jerk"),
            limits,
        ):
            if tuple(value.shape) != expected_limit_shape:
                raise ValueError(f"{name} must have shape {expected_limit_shape}")
            if not config.device_cfg.is_same_torch_device(value.device):
                raise ValueError(f"{name} must be on {config.device_cfg.device}")
            if value.dtype != torch.float32:
                raise ValueError(f"{name} must be float32")
            if torch.any(value <= 0.0):
                raise ValueError(f"{name} must be positive")
        self._max_velocity = max_velocity.detach().clone().contiguous()
        self._max_acceleration = max_acceleration.detach().clone().contiguous()
        self._max_jerk = max_jerk.detach().clone().contiguous()

        # Fixed kernel metadata and boundary values.
        self._lane_idx = torch.arange(lanes, device=self._device, dtype=torch.int32)
        self._use_implicit_goal = torch.ones(lanes, device=self._device, dtype=torch.uint8)
        self._trajectory_dt = torch.full((lanes,), config.trajectory_dt, **tensor_args)
        self._zero_controls = torch.zeros((lanes, n_knots, dof), **tensor_args)
        self._zero_state = torch.zeros((lanes, dof), **tensor_args)

        # Graph progress is uniform in joint-space arc length over the executable
        # spline interval.  The first/last interpolation interval is endpoint
        # padding imposed by the B-spline boundary convention.
        active_count = horizon - 2 * config.interpolation_steps
        if active_count < 2:
            raise ValueError("rollout horizon is too short for endpoint padding")
        self._arc_progress = torch.empty((horizon,), **tensor_args)
        self._arc_progress[: config.interpolation_steps].zero_()
        self._arc_progress[
            config.interpolation_steps : horizon - config.interpolation_steps
        ].copy_(torch.linspace(0.0, 1.0, active_count, **tensor_args))
        self._arc_progress[horizon - config.interpolation_steps :].fill_(1.0)

        self._arc_target = torch.empty((lanes, horizon, dof), **tensor_args)

        # Fixed work buffers used inside the captured fit.
        self._boundary_position = torch.empty((lanes, horizon, dof), **tensor_args)
        self._boundary_velocity = torch.empty_like(self._boundary_position)
        self._boundary_acceleration = torch.empty_like(self._boundary_position)
        self._boundary_jerk = torch.empty_like(self._boundary_position)
        self._boundary_dt = torch.empty((lanes,), **tensor_args)
        self._position_residual = torch.empty_like(self._boundary_position)
        self._fitted_knots = torch.empty((lanes, n_knots, dof), **tensor_args)

        # Fixed workspaces for derivative-limit IRLS refinement.
        self._current_knots = torch.empty_like(self._fitted_knots)
        self._candidate_knots = torch.empty_like(self._fitted_knots)
        self._current_position = torch.empty_like(self._boundary_position)
        self._current_velocity = torch.empty_like(self._boundary_position)
        self._current_acceleration = torch.empty_like(self._boundary_position)
        self._current_jerk = torch.empty_like(self._boundary_position)
        self._candidate_position = torch.empty_like(self._boundary_position)
        self._candidate_velocity = torch.empty_like(self._boundary_position)
        self._candidate_acceleration = torch.empty_like(self._boundary_position)
        self._candidate_jerk = torch.empty_like(self._boundary_position)
        self._current_dt = torch.empty((lanes,), **tensor_args)
        self._candidate_dt = torch.empty_like(self._current_dt)
        self._last_required_dt = torch.empty_like(self._current_dt)
        self._current_rollout_dt = torch.empty_like(self._boundary_dt)
        self._candidate_rollout_dt = torch.empty_like(self._boundary_dt)
        self._velocity_trace_workspace = torch.empty_like(self._boundary_position)
        self._acceleration_trace_workspace = torch.empty_like(self._boundary_position)
        self._jerk_trace_workspace = torch.empty_like(self._boundary_position)
        self._derivative_scale = torch.empty((lanes,), **tensor_args)

        self._basis_position: Optional[torch.Tensor] = None
        self._basis_velocity: Optional[torch.Tensor] = None
        self._basis_acceleration: Optional[torch.Tensor] = None
        self._basis_jerk: Optional[torch.Tensor] = None
        self._path_normal: Optional[torch.Tensor] = None
        self._path_normal_trace: Optional[torch.Tensor] = None
        self._path_diagonal_scale: Optional[torch.Tensor] = None
        self._path_normal_trace_value = 0.0
        self._path_diagonal_scale_value = 0.0

        self._normal_inverse: Optional[torch.Tensor] = None
        self._fit_matrix: Optional[torch.Tensor] = None
        self._start_position_coefficient: Optional[torch.Tensor] = None
        self._start_velocity_coefficient: Optional[torch.Tensor] = None
        self._start_acceleration_coefficient: Optional[torch.Tensor] = None
        self._start_jerk_coefficient: Optional[torch.Tensor] = None
        self._goal_position_coefficient: Optional[torch.Tensor] = None
        self._scaled_acceleration_regularization = 0.0
        self._executor = GraphExecutor(
            capture_fn=self._fit_fixed_shape,
            device=self._device,
            use_cuda_graph=config.use_cuda_graph,
            clone_outputs=False,
        )
        self._preparation_executors: dict[tuple, GraphExecutor] = {}

    @classmethod
    def from_transition_model(
        cls,
        transition_model: "RobotStateTransition",
        num_lanes: int,
        use_cuda_graph: bool = True,
        regularization: float = 1.0e-6,
        acceleration_regularization: float = 5.0e-1,
        irls_iterations: int = 2,
        timing_regularization: float = 10.0,
        trust_region_regularization: float = 5.0e-2,
        trajectory_dt: float = 1.0,
    ) -> "BSplineTimeOptimalSeedFitter":
        """Create a fitter whose representation matches a B-spline transition model.

        ``trajectory_dt`` must match the knot interval used when the fitted
        actions are rolled out. It scales the fixed start velocity and
        acceleration boundary knots in the same way as the solver's B-spline
        transition kernel.
        """
        config = BSplineTimeOptimalSeedFitterCfg(
            n_knots=transition_model.action_horizon,
            interpolation_steps=transition_model.interpolation_steps,
            control_space=transition_model.control_space,
            rollout_horizon=transition_model.horizon,
            num_lanes=num_lanes,
            dof=transition_model.action_dim,
            device_cfg=transition_model.device_cfg,
            use_cuda_graph=use_cuda_graph,
            regularization=regularization,
            acceleration_regularization=acceleration_regularization,
            irls_iterations=irls_iterations,
            timing_regularization=timing_regularization,
            trust_region_regularization=trust_region_regularization,
            trajectory_dt=trajectory_dt,
        )
        return cls(
            config,
            transition_model.max_velocity,
            transition_model.max_acceleration,
            transition_model.max_jerk,
        )

    @property
    def last_required_dt(self) -> torch.Tensor:
        """Return fixed-storage uniform dt scores from the most recent fit."""
        return self._last_required_dt

    @property
    def is_cuda_graph_initialized(self) -> bool:
        """Return whether the fixed-shape executor has been initialized."""
        return self._executor.is_initialized

    def _launch_forward(
        self,
        controls: torch.Tensor,
        start_position: torch.Tensor,
        start_velocity: torch.Tensor,
        start_acceleration: torch.Tensor,
        start_jerk: torch.Tensor,
        goal_position: torch.Tensor,
        out_position: torch.Tensor,
        out_velocity: torch.Tensor,
        out_acceleration: torch.Tensor,
        out_jerk: torch.Tensor,
        out_dt: torch.Tensor,
        lane_idx: torch.Tensor,
        use_implicit_goal: torch.Tensor,
        trajectory_dt: torch.Tensor,
    ) -> None:
        """Launch cuRobo's exact TrajOpt B-spline forward kernel."""
        trajectory_cu.launch_bspline_interpolation_forward_kernel(
            out_position,
            out_velocity,
            out_acceleration,
            out_jerk,
            out_dt,
            controls,
            start_position,
            start_velocity,
            start_acceleration,
            start_jerk,
            goal_position,
            self._zero_state[: goal_position.shape[0], : goal_position.shape[1]],
            self._zero_state[: goal_position.shape[0], : goal_position.shape[1]],
            self._zero_state[: goal_position.shape[0], : goal_position.shape[1]],
            lane_idx,
            lane_idx,
            trajectory_dt,
            use_implicit_goal,
            controls.shape[0],
            out_position.shape[1],
            controls.shape[2],
            controls.shape[1],
            self.degree,
        )

    def _launch_backward(
        self,
        out_gradient: torch.Tensor,
        position_gradient: torch.Tensor,
        velocity_gradient: torch.Tensor,
        acceleration_gradient: torch.Tensor,
        jerk_gradient: torch.Tensor,
        lane_idx: torch.Tensor,
        use_implicit_goal: torch.Tensor,
        trajectory_dt: torch.Tensor,
    ) -> None:
        """Launch cuRobo's exact TrajOpt B-spline backward kernel."""
        trajectory_cu.launch_bspline_interpolation_backward_kernel(
            out_gradient,
            position_gradient,
            velocity_gradient,
            acceleration_gradient,
            jerk_gradient,
            trajectory_dt,
            lane_idx,
            use_implicit_goal,
            position_gradient.shape[0],
            position_gradient.shape[1],
            position_gradient.shape[2],
            self.config.n_knots,
            self.degree,
            False,
        )

    def _resample_by_arc_length(self, graph_position: torch.Tensor) -> None:
        """Resample a fixed-size graph polyline onto the spline's active interval."""
        seed_fitter_cu.launch_resample_dense_kernel(
            graph_position,
            self._arc_progress,
            self._arc_target,
        )

    @torch.no_grad()
    def _initialize_normal_operator(self) -> None:
        """Build the position-plus-acceleration normal inverse with CUDA kernels."""
        if self._normal_inverse is not None:
            return

        free_knots = self.n_free_knots
        n_knots = self.config.n_knots
        horizon = self.config.rollout_horizon
        tensor_args = {
            "device": self._device,
            "dtype": self.config.device_cfg.dtype,
        }

        # Lane j is the j-th free-control impulse, so the position rollout is
        # column j of the exact B-spline design matrix A.
        controls = torch.zeros((free_knots, n_knots, 1), **tensor_args)
        diagonal_idx = torch.arange(free_knots, device=self._device)
        controls[diagonal_idx, diagonal_idx, 0] = 1.0
        zero_state = torch.zeros((free_knots, 1), **tensor_args)
        lane_idx = torch.arange(free_knots, device=self._device, dtype=torch.int32)
        use_implicit_goal = torch.ones(free_knots, device=self._device, dtype=torch.uint8)
        trajectory_dt = torch.full((free_knots,), self.config.trajectory_dt, **tensor_args)
        basis_position = torch.empty((free_knots, horizon, 1), **tensor_args)
        basis_velocity = torch.empty_like(basis_position)
        basis_acceleration = torch.empty_like(basis_position)
        basis_jerk = torch.empty_like(basis_position)
        basis_dt = torch.empty((free_knots,), **tensor_args)

        trajectory_cu.launch_bspline_interpolation_forward_kernel(
            basis_position,
            basis_velocity,
            basis_acceleration,
            basis_jerk,
            basis_dt,
            controls,
            zero_state,
            zero_state,
            zero_state,
            zero_state,
            zero_state,
            zero_state,
            zero_state,
            zero_state,
            lane_idx,
            lane_idx,
            trajectory_dt,
            use_implicit_goal,
            free_knots,
            horizon,
            1,
            n_knots,
            self.degree,
        )

        # Retain the exact design matrices for both the initial geometric fit
        # and the derivative-limit IRLS updates.
        self._basis_position = basis_position[:, :, 0].transpose(0, 1).contiguous()
        self._basis_velocity = basis_velocity[:, :, 0].transpose(0, 1).contiguous()
        self._basis_acceleration = basis_acceleration[:, :, 0].transpose(0, 1).contiguous()
        self._basis_jerk = basis_jerk[:, :, 0].transpose(0, 1).contiguous()
        self._path_normal = torch.sum(
            self._basis_position.unsqueeze(2) * self._basis_position.unsqueeze(1),
            dim=0,
        )
        self._path_normal = 0.5 * (self._path_normal + self._path_normal.transpose(0, 1))
        self._path_normal_trace = torch.diagonal(self._path_normal).sum()
        self._path_diagonal_scale = torch.diagonal(self._path_normal).mean()
        self._path_normal_trace_value = float(self._path_normal_trace.item())
        self._path_diagonal_scale_value = float(self._path_diagonal_scale.item())

        # Trace normalization makes the acceleration weight dimensionless and
        # stable across interpolation dt and B-spline degree.
        position_trace = torch.sum(torch.square(basis_position))
        acceleration_trace = torch.sum(torch.square(basis_acceleration))
        if self.config.acceleration_regularization > 0.0:
            acceleration_scale = float((position_trace / acceleration_trace).item())
            self._scaled_acceleration_regularization = (
                self.config.acceleration_regularization * acceleration_scale
            )
        else:
            self._scaled_acceleration_regularization = 0.0

        # Applying the exact B-spline backward kernel to every impulse rollout
        # builds A_p.T A_p + lambda A_a.T A_a without a second spline model.
        normal_columns = torch.empty((free_knots, n_knots, 1), **tensor_args)
        zero_gradient = torch.zeros_like(basis_position)
        scaled_basis_acceleration = basis_acceleration * self._scaled_acceleration_regularization
        trajectory_cu.launch_bspline_interpolation_backward_kernel(
            normal_columns,
            basis_position,
            zero_gradient,
            scaled_basis_acceleration,
            zero_gradient,
            trajectory_dt,
            lane_idx,
            use_implicit_goal,
            free_knots,
            horizon,
            1,
            n_knots,
            self.degree,
            False,
        )

        normal = normal_columns[:, :free_knots, 0].transpose(0, 1).contiguous()
        # Remove tiny reduction-order asymmetry before factorization.
        normal = 0.5 * (normal + normal.transpose(0, 1))
        if self.config.regularization > 0.0:
            diagonal_scale = torch.diagonal(self._path_normal).mean()
            normal = normal + (
                self.config.regularization * diagonal_scale * torch.eye(free_knots, **tensor_args)
            )
        cholesky = torch.linalg.cholesky(normal)
        self._normal_inverse = torch.cholesky_inverse(cholesky)

        # Fold the fixed inverse and boundary responses into the constants used
        # by the shared CUDA initializer. C++ exports these exact tensors.
        self._fit_matrix = torch.sum(
            self._normal_inverse.unsqueeze(2) * self._basis_position.transpose(0, 1).unsqueeze(0),
            dim=1,
        ).contiguous()
        boundary_controls = torch.zeros((1, n_knots, 1), **tensor_args)
        boundary_lane = torch.zeros(1, device=self._device, dtype=torch.int32)
        boundary_implicit_goal = torch.ones(1, device=self._device, dtype=torch.uint8)
        boundary_dt = torch.full((1,), self.config.trajectory_dt, **tensor_args)

        def boundary_response(
            start_position_value: float = 0.0,
            start_velocity_value: float = 0.0,
            start_acceleration_value: float = 0.0,
            start_jerk_value: float = 0.0,
            goal_position_value: float = 0.0,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            state = torch.empty((1, 1), **tensor_args)
            start_position = torch.full_like(state, start_position_value)
            start_velocity = torch.full_like(state, start_velocity_value)
            start_acceleration = torch.full_like(state, start_acceleration_value)
            start_jerk = torch.full_like(state, start_jerk_value)
            goal_position = torch.full_like(state, goal_position_value)
            position = torch.empty((1, horizon, 1), **tensor_args)
            velocity = torch.empty_like(position)
            acceleration = torch.empty_like(position)
            jerk = torch.empty_like(position)
            output_dt = torch.empty((1,), **tensor_args)
            self._launch_forward(
                boundary_controls,
                start_position,
                start_velocity,
                start_acceleration,
                start_jerk,
                goal_position,
                position,
                velocity,
                acceleration,
                jerk,
                output_dt,
                boundary_lane,
                boundary_implicit_goal,
                boundary_dt,
            )
            return position[0, :, 0], acceleration[0, :, 0]

        def boundary_coefficient(
            position: torch.Tensor,
            acceleration: torch.Tensor,
        ) -> torch.Tensor:
            rhs = torch.sum(self._basis_position * position.unsqueeze(1), dim=0)
            rhs += self._scaled_acceleration_regularization * torch.sum(
                self._basis_acceleration * acceleration.unsqueeze(1),
                dim=0,
            )
            return (-torch.sum(self._normal_inverse * rhs.unsqueeze(0), dim=1)).contiguous()

        position, acceleration = boundary_response(start_position_value=1.0)
        self._start_position_coefficient = boundary_coefficient(position, acceleration)
        position, acceleration = boundary_response(start_velocity_value=1.0)
        self._start_velocity_coefficient = boundary_coefficient(position, acceleration)
        position, acceleration = boundary_response(start_acceleration_value=1.0)
        self._start_acceleration_coefficient = boundary_coefficient(position, acceleration)
        position, acceleration = boundary_response(start_jerk_value=1.0)
        self._start_jerk_coefficient = boundary_coefficient(position, acceleration)
        position, acceleration = boundary_response(goal_position_value=1.0)
        self._goal_position_coefficient = boundary_coefficient(position, acceleration)

    def _required_dt(
        self,
        velocity: torch.Tensor,
        acceleration: torch.Tensor,
        jerk: torch.Tensor,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the per-lane uniform timestep required by all derivative limits."""
        if output is None:
            output = torch.empty(
                (velocity.shape[0],),
                device=velocity.device,
                dtype=velocity.dtype,
            )
        seed_fitter_cu.launch_required_dt_kernel(
            velocity,
            acceleration,
            jerk,
            self._max_velocity,
            self._max_acceleration,
            self._max_jerk,
            self.config.trajectory_dt,
            self.config.dt_epsilon,
            output,
        )
        return output

    def _rollout(
        self,
        controls: torch.Tensor,
        start_position: torch.Tensor,
        start_velocity: torch.Tensor,
        start_acceleration: torch.Tensor,
        start_jerk: torch.Tensor,
        goal_position: torch.Tensor,
        position: torch.Tensor,
        velocity: torch.Tensor,
        acceleration: torch.Tensor,
        jerk: torch.Tensor,
        out_dt: torch.Tensor,
    ) -> None:
        """Evaluate one fixed-shape knot batch at the configured knot timestep."""
        self._launch_forward(
            controls,
            start_position,
            start_velocity,
            start_acceleration,
            start_jerk,
            goal_position,
            position,
            velocity,
            acceleration,
            jerk,
            out_dt,
            self._lane_idx,
            self._use_implicit_goal,
            self._trajectory_dt,
        )

    def _solve_irls_step(
        self,
        current_knots: torch.Tensor,
        current_velocity: torch.Tensor,
        current_acceleration: torch.Tensor,
        goal_position: torch.Tensor,
    ) -> torch.Tensor:
        """Build and solve one p=6 IRLS quadratic surrogate."""
        seed_fitter_cu.launch_derivative_scale_kernel(
            current_velocity,
            current_acceleration,
            self._max_velocity,
            self._max_acceleration,
            self._max_jerk,
            self._basis_velocity,
            self._basis_acceleration,
            self._basis_jerk,
            self._path_normal_trace_value,
            self.config.timing_regularization,
            self.config.irls_epsilon,
            self._velocity_trace_workspace,
            self._acceleration_trace_workspace,
            self._jerk_trace_workspace,
            self._derivative_scale,
        )
        seed_fitter_cu.launch_solve_irls_kernel(
            self._position_residual,
            self._boundary_velocity,
            self._boundary_acceleration,
            self._boundary_jerk,
            current_velocity,
            current_acceleration,
            current_knots,
            goal_position,
            self._max_velocity,
            self._max_acceleration,
            self._max_jerk,
            self._basis_position,
            self._basis_velocity,
            self._basis_acceleration,
            self._basis_jerk,
            self._path_normal,
            self._derivative_scale,
            self.config.regularization,
            self.config.trust_region_regularization,
            self._path_diagonal_scale_value,
            self.config.irls_epsilon,
            self._candidate_knots,
        )
        return self._candidate_knots

    def _fit_fixed_shape(
        self,
        target_position: torch.Tensor,
        start_position: torch.Tensor,
        start_velocity: torch.Tensor,
        start_acceleration: torch.Tensor,
        start_jerk: torch.Tensor,
        goal_position: torch.Tensor,
    ) -> torch.Tensor:
        """Run the geometric initialization followed by accepted IRLS updates."""
        self._resample_by_arc_length(target_position)
        self._launch_forward(
            self._zero_controls,
            start_position,
            start_velocity,
            start_acceleration,
            start_jerk,
            goal_position,
            self._boundary_position,
            self._boundary_velocity,
            self._boundary_acceleration,
            self._boundary_jerk,
            self._boundary_dt,
            self._lane_idx,
            self._use_implicit_goal,
            self._trajectory_dt,
        )
        seed_fitter_cu.launch_initialize_fit_kernel(
            self._arc_target,
            self._boundary_position,
            start_position,
            start_velocity,
            start_acceleration,
            start_jerk,
            goal_position,
            self._fit_matrix,
            self._start_position_coefficient,
            self._start_velocity_coefficient,
            self._start_acceleration_coefficient,
            self._start_jerk_coefficient,
            self._goal_position_coefficient,
            self._position_residual,
            self._fitted_knots,
        )

        self._current_knots.copy_(self._fitted_knots)
        self._rollout(
            self._current_knots,
            start_position,
            start_velocity,
            start_acceleration,
            start_jerk,
            goal_position,
            self._current_position,
            self._current_velocity,
            self._current_acceleration,
            self._current_jerk,
            self._current_rollout_dt,
        )
        self._required_dt(
            self._current_velocity,
            self._current_acceleration,
            self._current_jerk,
            self._current_dt,
        )

        for _ in range(self.config.irls_iterations):
            self._solve_irls_step(
                self._current_knots,
                self._current_velocity,
                self._current_acceleration,
                goal_position,
            )
            self._rollout(
                self._candidate_knots,
                start_position,
                start_velocity,
                start_acceleration,
                start_jerk,
                goal_position,
                self._candidate_position,
                self._candidate_velocity,
                self._candidate_acceleration,
                self._candidate_jerk,
                self._candidate_rollout_dt,
            )
            self._required_dt(
                self._candidate_velocity,
                self._candidate_acceleration,
                self._candidate_jerk,
                self._candidate_dt,
            )
            seed_fitter_cu.launch_accept_candidate_kernel(
                self._candidate_knots,
                self._candidate_velocity,
                self._candidate_acceleration,
                self._candidate_jerk,
                self._candidate_dt,
                self._current_knots,
                self._current_velocity,
                self._current_acceleration,
                self._current_jerk,
                self._current_dt,
            )

        self._fitted_knots.copy_(self._current_knots)
        self._last_required_dt.copy_(self._current_dt)
        return self._fitted_knots

    def _validate_fit_inputs(
        self,
        target_position: torch.Tensor,
        start_state: JointState,
        goal_position: torch.Tensor,
    ) -> None:
        expected_target = (
            self.config.num_lanes,
            self.config.rollout_horizon,
            self.config.dof,
        )
        expected_state = (self.config.num_lanes, self.config.dof)
        if tuple(target_position.shape) != expected_target:
            raise ValueError(
                f"target_position must have shape {expected_target}; got "
                f"{tuple(target_position.shape)}"
            )
        if tuple(start_state.position.shape) != expected_state:
            raise ValueError(
                f"start_state.position must have shape {expected_state}; got "
                f"{tuple(start_state.position.shape)}"
            )
        if tuple(goal_position.shape) != expected_state:
            raise ValueError(
                f"goal_position must have shape {expected_state}; got {tuple(goal_position.shape)}"
            )
        tensors = [target_position, start_state.position, goal_position]
        tensors.extend(
            value
            for value in (start_state.velocity, start_state.acceleration, start_state.jerk)
            if value is not None
        )
        for tensor in tensors:
            if not self.config.device_cfg.is_same_torch_device(tensor.device):
                raise ValueError(f"all fitter inputs must be on {self._device}")
            if tensor.dtype != torch.float32:
                raise ValueError("all fitter inputs must be float32")
            if not tensor.is_contiguous():
                raise ValueError("all fitter inputs must be contiguous")
            if tensor is not target_position and tensor is not goal_position:
                if tuple(tensor.shape) != expected_state:
                    raise ValueError(
                        f"start-state tensors must have shape {expected_state}; got "
                        f"{tuple(tensor.shape)}"
                    )

    @torch.no_grad()
    def fit(
        self,
        target_position: torch.Tensor,
        start_state: JointState,
        goal_position: torch.Tensor,
    ) -> torch.Tensor:
        """Fit dense position paths and return fixed-shape TrajOpt action knots.

        Args:
            target_position: Dense graph paths with shape
                ``[num_lanes, rollout_horizon, dof]``.
            start_state: Start position and derivatives expanded to ``num_lanes``.
                Missing velocity, acceleration, or jerk tensors are treated as zero.
            goal_position: IK goal for each lane, shape ``[num_lanes, dof]``.

        Returns:
            Fitted action tensor with shape ``[num_lanes, n_knots, dof]``.
            Its final knot is exactly ``goal_position``.
        """
        self._validate_fit_inputs(target_position, start_state, goal_position)
        self._initialize_normal_operator()
        velocity = start_state.velocity
        acceleration = start_state.acceleration
        jerk = start_state.jerk
        if velocity is None:
            velocity = self._zero_state
        if acceleration is None:
            acceleration = self._zero_state
        if jerk is None:
            jerk = self._zero_state
        # When preparation owns capture, record these kernels into its graph.
        # Replaying a PyTorch CUDAGraph inside another capture is unsupported.
        if torch.cuda.is_current_stream_capturing():
            return self._fit_fixed_shape(
                target_position, start_state.position, velocity, acceleration,
                jerk, goal_position,
            )
        return self._executor(
            target_position,
            start_state.position,
            velocity,
            acceleration,
            jerk,
            goal_position,
            clone_outputs=False,
        )

    @torch.no_grad()
    def warmup(self) -> "BSplineTimeOptimalSeedFitter":
        """Build the normal operator and initialize the reusable CUDA graph."""
        self._initialize_normal_operator()
        self._executor.warmup(
            torch.zeros_like(self._boundary_position),
            self._zero_state,
            self._zero_state,
            self._zero_state,
            self._zero_state,
            self._zero_state,
        )
        return self

    def reset_cuda_graph(self) -> None:
        """Release the captured graph while retaining the normal operator."""
        for executor in self._preparation_executors.values():
            executor.reset()
        self._preparation_executors.clear()
        self._executor.reset()

    def destroy(self) -> None:
        """Release CUDA-graph resources owned by the fitter."""
        self.reset_cuda_graph()

    @torch.no_grad()
    def evaluate_position(
        self,
        knots: torch.Tensor,
        start_state: JointState,
        goal_position: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate knots with the exact rollout kernel for diagnostics and tests."""
        expected_knots = (
            self.config.num_lanes,
            self.config.n_knots,
            self.config.dof,
        )
        if tuple(knots.shape) != expected_knots:
            raise ValueError(f"knots must have shape {expected_knots}; got {tuple(knots.shape)}")
        # Reuse fit validation for device, dtype, contiguity, and boundary shapes.
        self._validate_fit_inputs(self._boundary_position, start_state, goal_position)
        velocity = start_state.velocity if start_state.velocity is not None else self._zero_state
        acceleration = (
            start_state.acceleration if start_state.acceleration is not None else self._zero_state
        )
        jerk = start_state.jerk if start_state.jerk is not None else self._zero_state
        position = torch.empty_like(self._boundary_position)
        velocity_out = torch.empty_like(position)
        acceleration_out = torch.empty_like(position)
        jerk_out = torch.empty_like(position)
        dt_out = torch.empty_like(self._boundary_dt)
        self._launch_forward(
            knots,
            start_state.position,
            velocity,
            acceleration,
            jerk,
            goal_position,
            position,
            velocity_out,
            acceleration_out,
            jerk_out,
            dt_out,
            self._lane_idx,
            self._use_implicit_goal,
            self._trajectory_dt,
        )
        return position

    @torch.no_grad()
    def evaluate_required_dt(
        self,
        knots: torch.Tensor,
        start_state: JointState,
        goal_position: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the uniform timestep required by a knot batch."""
        self._validate_fit_inputs(self._boundary_position, start_state, goal_position)
        expected_knots = (
            self.config.num_lanes,
            self.config.n_knots,
            self.config.dof,
        )
        if tuple(knots.shape) != expected_knots:
            raise ValueError(f"knots must have shape {expected_knots}; got {tuple(knots.shape)}")
        velocity = start_state.velocity if start_state.velocity is not None else self._zero_state
        acceleration = (
            start_state.acceleration if start_state.acceleration is not None else self._zero_state
        )
        jerk = start_state.jerk if start_state.jerk is not None else self._zero_state
        self._rollout(
            knots,
            start_state.position,
            velocity,
            acceleration,
            jerk,
            goal_position,
            self._candidate_position,
            self._candidate_velocity,
            self._candidate_acceleration,
            self._candidate_jerk,
            self._candidate_rollout_dt,
        )
        return self._required_dt(
            self._candidate_velocity,
            self._candidate_acceleration,
            self._candidate_jerk,
        )
