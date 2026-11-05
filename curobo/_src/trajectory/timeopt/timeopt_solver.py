# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused local-support primal-dual iterations for fixed-path timeopt."""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import torch
from cuda.core import LaunchConfig

from curobo._src.context import get_runtime
from curobo._src.curobolib.backends.cuda_core_backend.launch_helper import launch_kernel
from curobo._src.curobolib.backends.cuda_core_backend.trajectory_config import TrajectoryKernelCfg
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    from curobo._src.trajectory.timeopt.timeopt_basis import TimeOptProblem


class _StateOffset(IntEnum):
    """Float offsets matching FusedState in timeopt_common.cuh.

    Scalars occupy the first 16 entries. Each vector reserves 32 entries, and
    the lower band reserves four entries per control.
    """

    SPEED_SCALE = 0
    FIXED_CONTROLS = 1
    BEST_COST = 2
    PRIMAL_STEP = 3
    DUAL_STEP = 4
    ACCEPTED = 5
    ITERATION_VALID = 6
    NUMERICAL_OK = 7
    GAP = 9
    CENTERING = 10
    CONTROLS = 16
    BEST_CONTROLS = 48
    GRADIENT = 80
    DIAGONAL_SCALE = 112
    PREDICTOR = 144
    CORRECTOR = 176
    FACTOR = 208
    SIZE = 336


_RECORDS_PER_TILE = 64
_CONSTRAINTS_PER_RECORD = 7  # Squared velocity and signed acceleration, torque, jerk.
_PARTIALS_PER_TILE = 16
_LINE_SEARCH_STEPS = 8


class CudaTimeOptSolver:
    """Fixed-budget predictor/corrector with one banded factorization per iteration.

    Tiles group records with the same four B-spline controls. Constraint kernels
    emit local contributions; reduction kernels combine them before the one-warp
    predictor and corrector solves. Both solves reuse the same Cholesky factor.
    """

    _search_candidates = 2

    def __init__(self, evaluation: TimeOptProblem) -> None:
        """Allocate one record per sample/joint, with paired limits sharing derivatives."""
        self.evaluation = evaluation
        self.device = evaluation.path.device
        self.batch = evaluation.batch_size
        self.controls = evaluation.basis.count
        self.points, self.dof = evaluation.path.shape[-2:]
        if self.controls != 32:
            log_and_raise("Trajectory timeopt requires 32 timing controls")
        basis = evaluation.basis
        first = np.clip(
            np.searchsorted(basis.breaks_cpu, evaluation.grid_cpu, side="right") - 1,
            0,
            self.controls - 4,
        )
        columns = first[:, None] + np.arange(4)
        local_basis = np.stack(
            [
                np.take_along_axis(basis.spline(evaluation.grid_cpu, nu=k), columns, 1)
                for k in range(3)
            ]
        )
        self.basis = basis.tensor(local_basis)
        self.point_first = torch.as_tensor(first, device=self.device, dtype=torch.int32)
        # Duration uses the uniform prefix of the constraint basis and support.
        # Knot guard points only constrain. No second basis or weight table is needed.
        duration_points = evaluation.num_intervals + 1
        # Four lanes evaluate velocity, acceleration, torque and jerk for each record.
        count = self.points * self.dof
        # Record kinds match TimeOptRecord: joint limits, positive control,
        # duration samples, and padding. Each tile has one common support span.
        types = np.r_[
            np.zeros(count, dtype=np.int32), np.full(self.controls, 1), np.full(duration_points, 2)
        ]
        items = np.r_[np.arange(count), np.arange(self.controls), np.arange(duration_points)]
        support = np.r_[
            np.repeat(first, self.dof),
            np.minimum(np.arange(self.controls), self.controls - 4),
            first[:duration_points],
        ]
        metadata, offsets, starts = [], [], [0]
        for span in range(self.controls - 3):
            ids = np.flatnonzero(support == span)
            padded = ((len(ids) + _RECORDS_PER_TILE - 1) // _RECORDS_PER_TILE) * _RECORDS_PER_TILE
            tile = np.zeros((padded, 2), dtype=np.int32)
            tile[:, 0] = 3
            tile[: len(ids), 0], tile[: len(ids), 1] = types[ids], items[ids]
            metadata.append(tile)
            offsets.extend([span] * (padded // _RECORDS_PER_TILE))
            starts.append(len(offsets))
        self.tiles = len(offsets)
        self.meta = torch.as_tensor(np.concatenate(metadata), device=self.device)
        self.offsets = torch.as_tensor(offsets, device=self.device, dtype=torch.int32)
        self.starts = torch.as_tensor(starts, device=self.device, dtype=torch.int32)
        self.constraint_count = _CONSTRAINTS_PER_RECORD * count + self.controls
        shape = (self.batch, self.tiles * _RECORDS_PER_TILE)
        # Rows: [family, batch, record, local control]. Bounds hold the positive
        # and negative torque limits; the other families use unit bounds.
        self.rows = evaluation.path.new_empty((4, *shape, 4))
        self.bounds = evaluation.path.new_empty((2, *shape))
        # Only the nonlinear jerk derivative changes during optimization.
        self.jacobian = evaluation.path.new_empty((*shape, 4))
        # Slack, dual, affine complementarity product, final dual direction.
        self.row_state = evaluation.path.new_empty((4, _CONSTRAINTS_PER_RECORD, *shape))
        # Assembly and affine correction reuse this buffer with separate layouts
        # documented by AssemblyPartial and AffinePartial in the CUDA header.
        self.partials = evaluation.path.new_empty((self.batch, self.tiles, _PARTIALS_PER_TILE))
        self.corrector_terms = evaluation.path.new_empty((self.batch, 2, self.controls))
        self.step_bounds = evaluation.path.new_empty((self.batch, self.tiles, 2))
        self.candidate_masks = torch.empty(
            (self.batch, self.tiles * (_LINE_SEARCH_STEPS // self._search_candidates)),
            device=self.device,
            dtype=torch.int32,
        )
        self.state = evaluation.path.new_empty((self.batch, _StateOffset.SIZE))
        self.output = evaluation.path.new_empty((self.batch, self.controls))
        self.feasible = torch.empty(self.batch, device=self.device, dtype=torch.bool)
        self.numerical_ok = torch.empty_like(self.feasible)
        self.uniform_duration = evaluation.path.new_empty((self.batch,))
        self.seed_tiles = (count + 255) // 256
        # Prepared seed bounds survive output validation and repeated solves.
        self.seed_bounds = evaluation.path.new_empty((self.batch, self.seed_tiles, 3))
        self.output_bounds = torch.empty_like(self.seed_bounds)
        # float2 per lane: supplied squared speed and whether start derivatives fix it.
        self.reference_speed = evaluation.path.new_zeros((self.batch, 2))

    def prepare_output(
        self, time_basis: torch.Tensor, weights: torch.Tensor, intervals: int
    ) -> None:
        """Cache local quadrature support and finishing buffers for a uniform output grid."""
        nodes, _ = np.polynomial.legendre.leggauss(6)
        points = (np.arange(intervals)[:, None] + 0.5 * (nodes + 1)) / intervals
        first = np.clip(
            (points.ravel() * (self.controls - 3)).astype(np.int32), 0, self.controls - 4
        )
        indices = torch.as_tensor(
            first[:, None] + np.arange(4), device=self.device, dtype=torch.int64
        )
        self.time_basis = time_basis.gather(1, indices).contiguous()
        self.time_first = torch.as_tensor(first, device=self.device)
        self.time_weights = weights
        self.intervals = intervals
        self.dt = self.output.new_empty((self.batch, intervals))

    def prepare_path(
        self,
        controls: torch.Tensor,
        start: torch.Tensor,
        goal: torch.Tensor,
        velocity: torch.Tensor,
        acceleration: torch.Tensor,
        jerk: torch.Tensor,
        knot_dt: torch.Tensor,
        saved_controls: torch.Tensor,
        saved_start: torch.Tensor,
        saved_goal: torch.Tensor,
        valid: torch.Tensor,
        stationary: torch.Tensor,
        segments: torch.Tensor,
        local: torch.Tensor,
        q: torch.Tensor,
        qd: torch.Tensor,
        qdd: torch.Tensor,
        start_offset: int,
    ) -> None:
        """Sample the path and retain inputs while checking finite and stationary geometry."""
        self._launch(
            "inputs_and_path",
            (self.seed_tiles, self.batch),
            controls,
            start,
            goal,
            velocity,
            acceleration,
            jerk,
            knot_dt,
            self.reference_speed,
            saved_controls,
            saved_start,
            saved_goal,
            valid,
            stationary,
            segments,
            local,
            self.evaluation.path,
            q,
            qd,
            qdd,
            controls.shape[1],
            self.dof,
            self.points,
            self.batch,
            start_offset,
        )

    def prepare_coefficients(self, torque: torch.Tensor) -> None:
        """Extract A/B/C and seed bounds from inverse-dynamics states [B,3,G,D]."""
        ev = self.evaluation
        self._launch(
            "coefficients_and_seed_bounds",
            (self.seed_tiles, self.batch),
            ev.path,
            torque,
            ev.torque_coefficients,
            *ev.limits,
            self.seed_bounds,
            self.points,
            self.dof,
            self.batch,
            self.seed_tiles,
        )

    def finish(
        self,
        timing: torch.Tensor,
        initialized: torch.Tensor,
        uniform: torch.Tensor,
        numerical: torch.Tensor,
        input_valid: torch.Tensor,
        stationary: torch.Tensor,
        output_timing: torch.Tensor,
        output_uniform: torch.Tensor,
        output_numerical: torch.Tensor,
        output_success: torch.Tensor,
        output_status: torch.Tensor,
        output_joints: torch.Tensor,
        output_torque: torch.Tensor,
        output_times: torch.Tensor,
    ) -> None:
        """Reconstruct states, check sampled scale bounds, and integrate uniform-grid time."""
        ev = self.evaluation
        self._launch(
            "output",
            (self.seed_tiles, self.batch),
            timing,
            self.basis,
            self.point_first,
            ev.path,
            ev.torque_coefficients,
            *ev.limits,
            self.time_basis,
            self.time_first,
            self.time_weights,
            self.output_bounds,
            self.dt,
            output_joints,
            output_torque,
            self.points,
            self.dof,
            self.batch,
            self.intervals,
            self.seed_tiles,
        )
        self._launch(
            "finish",
            self.batch,
            self.output_bounds,
            initialized,
            uniform,
            numerical,
            input_valid,
            stationary,
            output_uniform,
            output_numerical,
            output_success,
            output_status,
            timing,
            self.dt,
            output_timing,
            output_joints,
            output_torque,
            output_times,
            self.intervals,
            self.dof,
            self.batch,
            self.seed_tiles,
        )

    def _launch(self, name: str, grid: int | tuple[int, int], *arguments: object) -> None:
        """Launch an NVRTC kernel on the current stream, including CUDA graph capture."""
        cache = get_runtime().get_cuda_core_cache()
        cfg = TrajectoryKernelCfg()
        qualified = "curobo::trajectory::timeopt::fused_" + name
        if name == "search":
            qualified += f"<{self._search_candidates}>"
        kernel = cache.get_or_compile_kernel(
            source_files=[
                cfg.kernel_dir / "timeopt" / "timeopt_common.cuh",
                cfg.kernel_dir / "timeopt" / "timeopt_path_kernel.cuh",
                cfg.kernel_dir / "timeopt" / "timeopt_io_kernel.cuh",
                cfg.kernel_dir / "timeopt" / "timeopt_solver_kernel.cuh",
            ],
            kernel_name=qualified,
            include_dirs=cfg.get_include_dirs(),
            compile_flags=cfg.get_compile_flags(),
        )
        stream = cache.get_stream_wrapper(torch.cuda.current_stream(self.device))
        launch_kernel(
            qualified,
            stream,
            LaunchConfig(grid=grid, block=32 if name in ("factor", "correct") else 256),
            kernel,
            *[x.data_ptr() if isinstance(x, torch.Tensor) else x for x in arguments],
        )

    def _initialize(self) -> None:
        """Initialize the reference or computed uniform time law and local constraint rows."""
        ev = self.evaluation
        self._launch(
            "prepare",
            (self.tiles, self.batch),
            self.meta,
            self.offsets,
            self.basis,
            ev.path,
            ev.torque_coefficients,
            *ev.limits,
            self.seed_bounds,
            self.reference_speed,
            self.state,
            self.feasible,
            self.numerical_ok,
            self.uniform_duration,
            self.rows,
            self.bounds,
            self.points,
            self.dof,
            self.batch,
            self.tiles,
            self.seed_tiles,
        )

    def _predict(self, initialize: bool) -> None:
        """Assemble the system, factor its lower band, and solve the predictor RHS."""
        self._launch(
            "assemble",
            (self.tiles, self.batch),
            self.meta,
            self.offsets,
            self.basis,
            self.rows,
            self.bounds,
            self.state,
            self.row_state,
            self.jacobian,
            self.partials,
            self.points,
            self.evaluation.num_intervals + 1,
            self.dof,
            self.batch,
            self.tiles,
            int(initialize),
        )
        self._launch(
            "factor_reduce",
            (5 * self.controls // 8 + 1, self.batch),
            self.partials,
            self.offsets,
            self.starts,
            self.state,
            self.tiles,
        )
        self._launch("factor", self.batch, self.state, self.numerical_ok)

    def _correct(self) -> None:
        """Build the centering correction and solve with the predictor's factor."""
        self._launch(
            "affine",
            (self.tiles, self.batch),
            self.meta,
            self.offsets,
            self.rows,
            self.jacobian,
            self.state,
            self.row_state,
            self.partials,
            self.batch,
            self.tiles,
        )
        self._launch(
            "correct_reduce",
            (self.controls // 8 + 1, self.batch),
            self.partials,
            self.offsets,
            self.starts,
            self.corrector_terms,
            self.state,
            self.tiles,
            self.constraint_count,
        )
        self._launch("correct", self.batch, self.corrector_terms, self.state, self.numerical_ok)

    def _search_and_accept(self) -> None:
        """Check all eight step scales and retain the best feasible timing controls."""
        self._launch(
            "bounds",
            (self.tiles, self.batch),
            self.meta,
            self.offsets,
            self.rows,
            self.jacobian,
            self.state,
            self.row_state,
            self.step_bounds,
            self.batch,
            self.tiles,
        )
        self._launch("search_bounds", self.batch, self.step_bounds, self.state, self.tiles)
        self._launch(
            "search",
            (self.candidate_masks.shape[1], self.batch),
            self.meta,
            self.offsets,
            self.basis,
            self.rows,
            self.bounds,
            self.state,
            self.candidate_masks,
            self.points,
            self.dof,
            self.batch,
            self.tiles,
        )
        self._launch(
            "accept",
            self.batch,
            self.candidate_masks,
            self.basis,
            self.point_first,
            self.state,
            self.output,
            self.candidate_masks.shape[1],
            self.evaluation.num_intervals + 1,
        )

    def solve(self, iterations: int) -> tuple[torch.Tensor, ...]:
        """Return timing controls, seed feasibility, uniform duration, and numerical status.

        Call ``prepare_coefficients`` after changing path or dynamics data.
        Execute exactly ``iterations`` updates.
        """
        self._initialize()
        for iteration in range(iterations):
            self._predict(initialize=iteration == 0)
            self._correct()
            self._search_and_accept()
        return self.output, self.feasible, self.uniform_duration, self.numerical_ok
