# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""BSPLINE_3 action seed fitting from waypoint trajectories."""

from __future__ import annotations

# Standard Library
from typing import Optional

# Third Party
import torch

# CuRobo
from curobo._src.state.state_joint import JointState
from curobo._src.util.logging import log_and_raise


class BSpline3ActionSeedGenerator:
    """Fit waypoint trajectories into BSPLINE_3 action seeds."""

    def __init__(
        self,
        n_knots: int,
        interpolation_steps: int,
        interpolated_dt: float,
    ):
        """Initialize BSPLINE_3 action seed fitting constants."""
        self.degree = 3
        self.spline_support_size = self.degree + 1
        self.n_knots = n_knots
        self.interpolation_steps = interpolation_steps
        self.interpolated_dt = interpolated_dt
        self.padded_n_knots = self.n_knots + self.spline_support_size
        self.knot_dt = interpolated_dt * interpolation_steps

    def fit_action_seed(
        self,
        waypoints: torch.Tensor,
        current_state: JointState,
        goal_state: Optional[JointState],
        use_implicit_goal: bool,
    ) -> torch.Tensor:
        """Fit position-space waypoints into BSPLINE_3 action-space knots.

        Args:
            waypoints: Seed waypoints with shape ``[batch, num_seeds, n_waypoints, dof]``.
            current_state: Start state used for start fixed knots.
            goal_state: Optional implicit goal state. When ``None``, the final waypoint
                of each seed is used as the goal position with zero velocity and acceleration.
            use_implicit_goal: Only implicit-goal fitting is currently supported.

        Returns:
            Action seed tensor with shape ``[batch, num_seeds, n_knots, dof]``.
        """
        if waypoints.ndim != 4:
            log_and_raise(
                "waypoints must have shape [batch, num_seeds, n_waypoints, dof], "
                + f"got {tuple(waypoints.shape)}"
            )

        batch_size, num_seeds, n_waypoints, dof = waypoints.shape
        fit_n_control = self.n_knots + 2 * self.spline_support_size - 1
        if n_waypoints < fit_n_control:
            log_and_raise(
                f"Need at least {fit_n_control} waypoints to fit BSPLINE_3 action seed, "
                + f"got {n_waypoints}."
            )
        if not use_implicit_goal:
            log_and_raise("BSPLINE_3 action seed fitting requires use_implicit_goal=True")
        expected_current_shape = (batch_size, dof)
        if current_state.position.shape != expected_current_shape:
            log_and_raise(
                "current_state.position must have shape "
                + f"{expected_current_shape}, got {tuple(current_state.position.shape)}"
            )

        knot_dt = torch.tensor(self.knot_dt, device=waypoints.device, dtype=waypoints.dtype)

        current_position = current_state.position.to(
                            device=waypoints.device,
                            dtype=waypoints.dtype,
                        )

        current_velocity = (
            current_state.velocity.to(device=waypoints.device, dtype=waypoints.dtype)
            if current_state.velocity is not None
            else torch.zeros_like(current_position)
        )

        current_acceleration = (
            current_state.acceleration.to(device=waypoints.device, dtype=waypoints.dtype)
            if current_state.acceleration is not None
            else torch.zeros_like(current_position)
        )

        if use_implicit_goal:
            if goal_state is None:
                goal_position = waypoints[:, :, -1, :]
                goal_velocity = torch.zeros_like(goal_position)
                goal_acceleration = torch.zeros_like(goal_position)
            else:
                goal_position = self._goal_state_component(
                    goal_state.position,
                    batch_size,
                    num_seeds,
                    dof,
                    waypoints.device,
                    waypoints.dtype,
                    "goal_state.position",
                    waypoints[:, :, -1, :],
                )
                goal_velocity = self._goal_state_component(
                    goal_state.velocity,
                    batch_size,
                    num_seeds,
                    dof,
                    waypoints.device,
                    waypoints.dtype,
                    "goal_state.velocity",
                    torch.zeros_like(goal_position),
                )
                goal_acceleration = self._goal_state_component(
                    goal_state.acceleration,
                    batch_size,
                    num_seeds,
                    dof,
                    waypoints.device,
                    waypoints.dtype,
                    "goal_state.acceleration",
                    torch.zeros_like(goal_position),
                )

        action_seed = torch.empty(
            batch_size,
            num_seeds,
            self.n_knots,
            dof,
            device=waypoints.device,
            dtype=waypoints.dtype,
        )

        n_segments = fit_n_control - self.degree
        total_duration = knot_dt * n_segments

        for b_idx in range(batch_size):
            start_fixed_knots = self._compute_fixed_knots(
                current_position[b_idx],
                current_velocity[b_idx],
                current_acceleration[b_idx],
                knot_dt,
            )

            for seed_idx in range(num_seeds):
                seed_waypoints = waypoints[b_idx, seed_idx]
                x = self._compute_time_parameter(
                    seed_waypoints,
                    current_velocity[b_idx],
                    current_acceleration[b_idx],
                    goal_velocity[b_idx, seed_idx],
                    goal_acceleration[b_idx, seed_idx],
                    total_duration,
                )
                design = self._build_design_matrix(x, fit_n_control)

                goal_fixed_knots = self._compute_fixed_knots(
                    goal_position[b_idx, seed_idx],
                    -goal_velocity[b_idx, seed_idx],
                    goal_acceleration[b_idx, seed_idx],
                    knot_dt,
                ).flip(0)

                fixed_contrib = (
                    design[:, : self.spline_support_size] @ start_fixed_knots
                    + design[:, -self.spline_support_size :] @ goal_fixed_knots
                )
                rhs = seed_waypoints - fixed_contrib

                free_design = design[
                    :,
                    self.spline_support_size : self.spline_support_size + self.n_knots - 1,
                ]
                free_knots = torch.linalg.lstsq(free_design, rhs).solution

                action_seed[b_idx, seed_idx, :self.n_knots - 1] = free_knots
                action_seed[b_idx, seed_idx, self.n_knots - 1] = goal_fixed_knots[0]

        return action_seed

    def _build_design_matrix(self, x: torch.Tensor, n_control: int) -> torch.Tensor:
        n_segments = n_control - self.degree
        design = torch.zeros(x.shape[0], n_control, device=x.device, dtype=x.dtype)

        scaled = torch.clamp(x * n_segments, 0.0, float(n_segments))
        segment = torch.floor(scaled).to(torch.long).clamp(max=n_segments - 1)
        t = scaled - segment.to(dtype=x.dtype)

        basis = self._cubic_uniform_basis(t)
        rows = torch.arange(x.shape[0], device=x.device)

        for local_idx in range(self.spline_support_size):
            design[rows, segment + local_idx] = basis[:, local_idx]

        return design

    def _cubic_uniform_basis(self, t: torch.Tensor) -> torch.Tensor:
        t2 = t * t
        t3 = t2 * t

        return torch.stack(
            (
                (1.0 - 3.0 * t + 3.0 * t2 - t3) / 6.0,
                (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0,
                (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0,
                t3 / 6.0,
            ),
            dim=-1,
        )

    def _compute_fixed_knots(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        acceleration: torch.Tensor,
        knot_dt: torch.Tensor,
    ) -> torch.Tensor:
        velocity_coeffs = torch.tensor(
            [-1.0, 0.0, 1.0, 2.0],
            device=position.device,
            dtype=position.dtype,
        )
        acceleration_coeffs = torch.tensor(
            [1.0 / 3.0, -1.0 / 6.0, 1.0 / 3.0, 11.0 / 6.0],
            device=position.device,
            dtype=position.dtype,
        )

        return (
            position.unsqueeze(0)
            + velocity_coeffs[:, None] * velocity.unsqueeze(0) * knot_dt
            + acceleration_coeffs[:, None] * acceleration.unsqueeze(0) * knot_dt * knot_dt
        )

    def _compute_time_parameter(
        self,
        waypoints: torch.Tensor,
        initial_velocity: torch.Tensor,
        initial_acceleration: torch.Tensor,
        terminal_velocity: torch.Tensor,
        terminal_acceleration: torch.Tensor,
        total_duration: torch.Tensor,
    ) -> torch.Tensor:
        chord = torch.linalg.norm(torch.diff(waypoints, dim=0), dim=1)
        cumulative = torch.cat([torch.zeros(1, device=waypoints.device, dtype=waypoints.dtype), chord.cumsum(0)])
        total_length = cumulative[-1]

        if bool((total_length <= 1e-12).item()):
            return torch.linspace(
                0.0,
                1.0,
                waypoints.shape[0],
                device=waypoints.device,
                dtype=waypoints.dtype,
            )

        start_secant = waypoints[1] - waypoints[0]
        goal_secant = waypoints[-1] - waypoints[-2]
        start_tangent = self._unit_direction(initial_velocity, start_secant)
        goal_tangent = self._unit_direction(terminal_velocity, goal_secant)

        v0 = torch.linalg.norm(initial_velocity) * total_duration
        v1 = torch.linalg.norm(terminal_velocity) * total_duration
        a0 = torch.dot(initial_acceleration, start_tangent) * total_duration * total_duration
        a1 = torch.dot(terminal_acceleration, goal_tangent) * total_duration * total_duration

        coeffs = self._solve_quintic_time_law(total_length, v0, v1, a0, a1)
        x = self._invert_time_law(coeffs, cumulative)
        x[0] = 0.0
        x[-1] = 1.0
        return x

    def _unit_direction(self, vector: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
        norm = torch.linalg.norm(vector)
        if bool((norm > 1e-9).item()):
            return vector / norm

        fallback_norm = torch.linalg.norm(fallback)
        if bool((fallback_norm > 1e-9).item()):
            return fallback / fallback_norm

        return torch.zeros_like(vector)

    def _solve_quintic_time_law(
        self,
        total_length: torch.Tensor,
        v0: torch.Tensor,
        v1: torch.Tensor,
        a0: torch.Tensor,
        a1: torch.Tensor,
    ) -> torch.Tensor:
        p1 = v0 + a0 / 4.0
        p2 = 5.0 * total_length - 2.0 * v0 - 2.0 * v1 + (a1 - a0) / 4.0
        p3 = v1 - a1 / 4.0
        if bool(((p1 < 0.0) | (p2 < 0.0) | (p3 < 0.0)).item()):
            log_and_raise(
                "Boundary state cannot guarantee a monotonic time law: "
                + f"P1={p1.item():.6g}, P2={p2.item():.6g}, P3={p3.item():.6g}."
            )

        basis_matrix = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 6.0, 12.0, 20.0],
            ],
            device=total_length.device,
            dtype=total_length.dtype,
        )
        targets = torch.stack(
            [
                torch.zeros_like(total_length),
                total_length,
                v0,
                v1,
                a0,
                a1,
            ]
        )
        return torch.linalg.solve(basis_matrix, targets)

    def _invert_time_law(
        self,
        coeffs: torch.Tensor,
        target_lengths: torch.Tensor,
        n_dense: int = 4001,
    ) -> torch.Tensor:
        x_dense = torch.linspace(
            0.0,
            1.0,
            n_dense,
            device=target_lengths.device,
            dtype=target_lengths.dtype,
        )
        powers = torch.stack([x_dense**i for i in range(coeffs.shape[0])], dim=0)
        s_dense = coeffs @ powers
        s_monotonic = torch.cummax(s_dense, dim=0).values

        clipped_targets = target_lengths.clamp(s_monotonic[0], s_monotonic[-1])
        return self._interp1d(clipped_targets, s_monotonic, x_dense)

    def _interp1d(
        self,
        values: torch.Tensor,
        xp: torch.Tensor,
        fp: torch.Tensor,
    ) -> torch.Tensor:
        idx = torch.searchsorted(xp, values, right=False).clamp(1, xp.numel() - 1)

        x0 = xp[idx - 1]
        x1 = xp[idx]
        y0 = fp[idx - 1]
        y1 = fp[idx]

        denom = torch.clamp(x1 - x0, min=torch.finfo(values.dtype).eps)
        weight = (values - x0) / denom
        return y0 + weight * (y1 - y0)

    def _goal_state_component(
        self,
        value: Optional[torch.Tensor],
        batch_size: int,
        num_seeds: int,
        dof: int,
        device: torch.device,
        dtype: torch.dtype,
        name: str,
        fallback: torch.Tensor,
    ) -> torch.Tensor:
        if value is None:
            return fallback

        value = value.to(device=device, dtype=dtype)
        if value.shape == (batch_size, num_seeds, dof):
            return value
        if value.shape == (batch_size, dof):
            return value[:, None, :].expand(batch_size, num_seeds, dof)
        if value.shape == (1, dof) and batch_size > 1:
            return value.expand(batch_size, dof)[:, None, :].expand(
                batch_size, num_seeds, dof
            )

        log_and_raise(
            f"{name} must have shape [{batch_size}, {num_seeds}, {dof}] "
            + f"or [{batch_size}, {dof}], got {tuple(value.shape)}"
        )