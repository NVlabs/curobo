# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU fixed-path timeopt with velocity, acceleration, jerk, and torque limits."""

from __future__ import annotations

import math

import numpy as np
import torch

from curobo._src.curobolib.backends.cuda_core_backend import trajectory_timeopt as timeopt_cu
from curobo._src.motion.motion_trajectory_timeopt_cfg import TrajectoryTimeOptCfg
from curobo._src.motion.motion_trajectory_timeopt_result import TrajectoryTimeOptResult
from curobo._src.robot.dynamics.dynamics import Dynamics
from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg
from curobo._src.state.state_joint import JointState
from curobo._src.trajectory.timeopt.timeopt_basis import CubicTimingBasis, TimeOptProblem
from curobo._src.trajectory.timeopt.timeopt_solver import CudaTimeOptSolver
from curobo._src.util.cuda_graph_util import GraphExecutor
from curobo._src.util.logging import log_and_raise


class TrajectoryTimeOpt:
    """Optimize timing along a fixed cubic path with a positive squared-speed spline.

    Uses exact rigid-body coefficients at a fixed grid and both sides of path knots.
    Success denotes sampled feasibility; continuous-time feasibility is not certified.
    Normal operation captures the complete timeopt in one CUDA graph.
    """

    def __init__(self, config: TrajectoryTimeOptCfg) -> None:
        """Allocate buffers and precompute timing basis matrices; capture lazily."""
        self.config = config
        model = config.robot.kinematics_config
        self.device = model.fixed_transforms.device
        self.batch_size, self.dof = config.batch_size, model.num_dof
        if self.device.type != "cuda" or config.robot.device_cfg.dtype != torch.float32:
            log_and_raise("Trajectory timeopt requires a CUDA float32 model")
        if not 1 <= self.dof <= 64:
            log_and_raise("Trajectory timeopt supports 1 to 64 DoF")
        for name, value, minimum in (
            ("batch_size", config.batch_size, 1),
            ("n_knots", config.n_knots, 2),
            ("timing_controls", config.timing_controls, 4),
            ("num_intervals", config.num_intervals, 1),
            ("max_iterations", config.max_iterations, 1),
        ):
            if type(value) is not int or value < minimum:
                log_and_raise(f"{name} must be an integer >= {minimum}")
        if config.timing_controls != 32:
            log_and_raise("timing_controls must be 32")
        if any(
            not math.isfinite(x) or not 0 < x <= 1
            for x in (config.limit_scale, config.jerk_limit_scale)
        ):
            log_and_raise("Limit scales must be finite and in (0,1]")
        if len(config.gravity) != 3 or not all(math.isfinite(x) for x in config.gravity):
            log_and_raise("Gravity must be three finite values")
        self._basis = CubicTimingBasis(config.timing_controls, self.device)
        self._output_grid_cpu = np.linspace(0, 1, config.num_intervals + 1)
        self._grid = self._basis.tensor(self._output_grid_cpu)
        self._dynamics = Dynamics(
            DynamicsCfg(model, config.robot.device_cfg, list(config.gravity))
        )
        limits = model.joint_limits
        chosen = []
        for name in ("velocity", "acceleration", "jerk"):
            explicit = getattr(config, name + "_limits")
            native = getattr(limits, name, None)
            if explicit is None:
                if native is None:
                    log_and_raise(f"Explicit {name} limits are required")
                explicit = torch.minimum(-native[0], native[1]).contiguous()
            self._check_tensor(name, explicit, (self.dof,))
            if not bool(torch.isfinite(explicit).all() and (explicit > 0).all()):
                log_and_raise(f"{name} limits must be positive and finite")
            chosen.append(
                explicit.detach()
                * (config.jerk_limit_scale if name == "jerk" else config.limit_scale)
            )
        lower, upper = config.torque_lower, config.torque_upper
        if (lower is None or upper is None) and limits.effort is None:
            log_and_raise("Explicit effort limits are required")
        lower = limits.effort[0].contiguous() if lower is None else lower
        upper = limits.effort[1].contiguous() if upper is None else upper
        for name, value in (("torque_lower", lower), ("torque_upper", upper)):
            self._check_tensor(name, value, (self.dof,))
        if not bool(
            torch.isfinite(lower).all() and torch.isfinite(upper).all() and (lower <= upper).all()
        ):
            log_and_raise("Torque bounds must be finite and ordered")
        self._limits = tuple(
            chosen
            + [
                torch.maximum(lower, lower * config.limit_scale).detach(),
                torch.minimum(upper, upper * config.limit_scale).detach(),
            ]
        )
        if not bool((self._limits[-2] <= self._limits[-1]).all()):
            log_and_raise("limit_scale leaves an empty torque interval")
        self._controls = self._grid.new_empty((self.batch_size, config.n_knots, self.dof))
        self._start_boundary = self._grid.new_empty((self.batch_size, 4, self.dof))
        self._goal = self._grid.new_empty((self.batch_size, self.dof))
        self._zero_state = torch.zeros_like(self._goal)
        self._zero_dt = self._grid.new_zeros((self.batch_size,))
        self._stationary = torch.empty(self.batch_size, device=self.device, dtype=torch.bool)
        self._timing = self._grid.new_empty((self.batch_size, config.timing_controls))
        out_shape = (self.batch_size, self._grid.numel(), self.dof)
        self._joint_output = self._grid.new_empty((4, *out_shape))
        self._torque_output = self._grid.new_empty(out_shape)
        self._times = self._grid.new_empty(out_shape[:2])
        self._success = torch.empty(self.batch_size, device=self.device, dtype=torch.bool)
        self._status = torch.empty(self.batch_size, device=self.device, dtype=torch.int32)
        self._optimization_ok = torch.empty_like(self._success)
        self._uniform_duration = self._grid.new_empty((self.batch_size,))
        self._input_valid = torch.empty_like(self._success)
        self._result = TrajectoryTimeOptResult(
            timing_controls=self._timing,
            path_parameter=self._grid,
            time=self._times,
            joint_state=JointState(
                position=self._joint_output[0],
                velocity=self._joint_output[1],
                acceleration=self._joint_output[2],
                jerk=self._joint_output[3],
                joint_names=list(model.joint_names),
            ),
            torque=self._torque_output,
            success=self._success,
            status=self._status,
            optimization_ok=self._optimization_ok,
            uniform_duration=self._uniform_duration,
            stationary=self._stationary,
        )
        nodes, weights = np.polynomial.legendre.leggauss(6)
        self._nodes = self._basis.tensor(0.5 * (nodes + 1))
        self._weights = self._basis.tensor(0.5 * weights)
        quad_s = self._output_grid_cpu[:-1, None] + (
            np.diff(self._output_grid_cpu)[:, None] * 0.5 * (nodes + 1)
        )
        self._time_basis = self._basis.at(quad_s.ravel())[0]
        self._executor = GraphExecutor(
            self._timeopt,
            device=self.device,
            use_cuda_graph=config.use_cuda_graph,
            clone_outputs=False,
        )
        self._sample_executor = GraphExecutor(
            self._sample,
            device=self.device,
            use_cuda_graph=config.use_cuda_graph,
            clone_outputs=False,
        )
        self._sample_count: int | None = None
        self._setup_path(reference=False)

    def _setup_path(self, reference: bool) -> None:
        """Allocate the shared sampling/solver grid, retaining the reference start span."""
        if torch.cuda.is_current_stream_capturing():
            log_and_raise("Warm up timeopt with the same knot_dt presence before CUDA capture")
        config = self.config
        self._reference = reference
        self._has_result = False
        spans = config.n_knots + 2 + reference
        internal = np.arange(1, spans) / spans
        grid = np.r_[self._output_grid_cpu, internal, internal, self._basis.breaks_cpu[1:-1]]
        segments = np.minimum(np.floor(grid * spans).astype(np.int32), spans - 1)
        local = grid * spans - segments
        offset = config.num_intervals + 1
        segments[offset : offset + spans - 1] = np.arange(spans - 1)
        local[offset : offset + spans - 1] = 1.0
        segments[offset + spans - 1 : offset + 2 * (spans - 1)] = np.arange(1, spans)
        local[offset + spans - 1 : offset + 2 * (spans - 1)] = 0.0
        self.constraint_grid = self._basis.tensor(grid)
        self._segments = torch.as_tensor(segments, device=self.device).contiguous()
        self._local = self._basis.tensor(local)
        self._path = self._grid.new_empty((4, self.batch_size, len(grid), self.dof))
        shape = (self.batch_size, 3, len(grid), self.dof)
        self._q, self._qd, self._qdd = [self._grid.new_empty(shape) for _ in range(3)]
        self._state = JointState(position=self._q, velocity=self._qd, acceleration=self._qdd)
        self._coefficients = self._grid.new_empty(shape)
        self._dynamics.setup_batch_size(batch_size=self.batch_size * 3 * len(grid))
        self._evaluation = TimeOptProblem(
            self._basis, grid, self._path, self._coefficients, self._limits, config.num_intervals
        )
        self._evaluation.stationary = self._stationary
        self._solver = CudaTimeOptSolver(self._evaluation)
        self._solver.prepare_output(self._time_basis, self._weights, config.num_intervals)
        self.reset_cuda_graph()

    def _compute_torque(self, state: JointState) -> torch.Tensor:
        """Reuse the preparation workspace for sequential dense torque validation."""
        count = state.position.numel() // self.dof
        if count > self._dynamics._tau_buffer.shape[0]:
            # Captured preparation must be rebound if validation grows the workspace.
            self._executor.reset()
            self._dynamics.setup_batch_size(count)
        return self._dynamics.compute_inverse_dynamics(state)

    def _check_tensor(self, name: str, value: torch.Tensor, shape: tuple[int, ...]) -> None:
        """Reject unsupported buffers before launching native code."""
        if (
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or value.device != self.device
            or value.dtype != torch.float32
            or not value.is_contiguous()
        ):
            log_and_raise(f"{name} must be contiguous CUDA float32 with shape {shape}")

    def _prepare(
        self,
        controls: torch.Tensor,
        start: torch.Tensor,
        goal: torch.Tensor,
        velocity: torch.Tensor | None = None,
        acceleration: torch.Tensor | None = None,
        jerk: torch.Tensor | None = None,
        knot_dt: torch.Tensor | None = None,
    ) -> None:
        """Sample geometry and prepare reusable dynamics and the reference timing seed."""
        self._solver.prepare_path(
            controls,
            start,
            goal,
            self._zero_state if velocity is None else velocity,
            self._zero_state if acceleration is None else acceleration,
            self._zero_state if jerk is None else jerk,
            self._zero_dt if knot_dt is None else knot_dt,
            self._controls,
            self._start_boundary,
            self._goal,
            self._input_valid,
            self._evaluation.stationary,
            self._segments,
            self._local,
            self._q,
            self._qd,
            self._qdd,
            int(not self._reference),
        )
        torque = self._dynamics.compute_inverse_dynamics(self._state)
        self._solver.prepare_coefficients(torque)

    def timeopt(
        self,
        controls: torch.Tensor,
        start: torch.Tensor | JointState,
        goal: torch.Tensor,
        knot_dt: torch.Tensor | None = None,
    ) -> TrajectoryTimeOptResult:
        """Optimize timing for controls [B,K,D] with a position or moving start [B,D].

        The optional knot interval [B] initializes timing from a TrajOpt solution.
        Static references are clamped to a feasible uniform speed when needed.
        Moving starts require the original knot interval [B]. Its time law is
        assumed feasible under the configured limits and seeds the same solver.
        Position, velocity, and acceleration are preserved; input jerk determines
        path geometry but is not held fixed by timeopt. Outputs borrow buffers.
        """
        derivatives = [None if knot_dt is None else self._zero_state] * 3
        if isinstance(start, JointState):
            if knot_dt is None:
                log_and_raise("A JointState start requires the original knot_dt [B]")
            derivatives = [
                self._zero_state if getattr(start, name) is None else getattr(start, name)
                for name in ("velocity", "acceleration", "jerk")
            ]
            if (
                start.joint_names is not None
                and start.joint_names != self._result.joint_state.joint_names
            ):
                log_and_raise("start joint_names must match the robot joint order")
            start = start.position
        self._check_tensor("controls", controls, tuple(self._controls.shape))
        self._check_tensor("start", start, (self.batch_size, self.dof))
        self._check_tensor("goal", goal, (self.batch_size, self.dof))
        for name, value in zip(("velocity", "acceleration", "jerk"), derivatives):
            if value is not None:
                self._check_tensor(name, value, (self.batch_size, self.dof))
        if knot_dt is not None:
            self._check_tensor("knot_dt", knot_dt, (self.batch_size,))
        if self._reference != (knot_dt is not None):
            self._setup_path(reference=knot_dt is not None)
        args = (controls, start, goal, *derivatives, knot_dt)
        if torch.cuda.is_current_stream_capturing():
            self._timeopt(*args)
        else:
            self._executor(*args)
        return self._result

    def _timeopt(
        self,
        controls: torch.Tensor,
        start: torch.Tensor,
        goal: torch.Tensor,
        velocity: torch.Tensor | None,
        acceleration: torch.Tensor | None,
        jerk: torch.Tensor | None,
        knot_dt: torch.Tensor | None,
    ) -> torch.Tensor:
        """Capture-friendly complete solve, including hard sampled acceptance checks."""
        self._prepare(controls, start, goal, velocity, acceleration, jerk, knot_dt)
        return self._finish(*self._solver.solve(self.config.max_iterations))

    def _finish(
        self,
        timing: torch.Tensor,
        initialized: torch.Tensor,
        uniform_duration: torch.Tensor,
        numerical_ok: torch.Tensor,
    ) -> torch.Tensor:
        """Check hard sampled limits, integrate time, and reconstruct joint states."""
        self._solver.finish(
            timing,
            initialized,
            uniform_duration,
            numerical_ok,
            self._input_valid,
            self._evaluation.stationary,
            self._timing,
            self._uniform_duration,
            self._optimization_ok,
            self._success,
            self._status,
            self._joint_output,
            self._torque_output,
            self._times,
        )
        self._has_result = True
        return self._times

    def sample(self, times: torch.Tensor) -> JointState:
        """Evaluate requested seconds [B,N]; first call fixes N for this sampler graph."""
        if not self._has_result:
            log_and_raise("Call timeopt before requesting sample states")
        if not isinstance(times, torch.Tensor) or times.ndim != 2 or times.shape[1] == 0:
            log_and_raise("times must have shape [B,N] with N > 0")
        self._check_tensor("times", times, (self.batch_size, times.shape[1]))
        if self._sample_count is not None and self._sample_count != times.shape[1]:
            log_and_raise("Sample count changed; reset_cuda_graph before using a new sample count")
        self._sample_count = times.shape[1]
        output = (
            self._sample(times)
            if torch.cuda.is_current_stream_capturing()
            else self._sample_executor(times)
        )
        return JointState(
            position=output[0],
            velocity=output[1],
            acceleration=output[2],
            jerk=output[3],
            joint_names=self._result.joint_state.joint_names.copy(),
        )

    def _sample(self, times: torch.Tensor) -> torch.Tensor:
        """Invert monotone quadrature with safeguarded Newton steps and apply the chain rule."""
        output = self._grid.new_empty((4, self.batch_size, times.shape[1], self.dof))
        timeopt_cu.sample(
            self._controls,
            self._start_boundary,
            self._goal,
            self._timing,
            self._basis.polynomial,
            self._nodes,
            self._weights,
            self._times,
            times,
            self._success,
            self._stationary,
            output,
            int(not self._reference),
        )
        return output

    def reset_cuda_graph(self) -> None:
        """Reset captured graphs, retaining fixed model and solver buffers."""
        self._executor.reset()
        self._sample_executor.reset()
        self._sample_count = None
