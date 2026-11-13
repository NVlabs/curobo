# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense FP32 equation checks, independent of CUDA storage and reductions."""

from __future__ import annotations

import numpy as np
import torch

from curobo._src.trajectory.timeopt.timeopt_basis import TimeOptProblem


def feasible_scale(
    velocity: torch.Tensor,
    acceleration: torch.Tensor,
    jerk: torch.Tensor,
    dynamic_torque: torch.Tensor,
    gravity: torch.Tensor,
    velocity_limits: torch.Tensor,
    acceleration_limits: torch.Tensor,
    jerk_limits: torch.Tensor,
    torque_lower: torch.Tensor,
    torque_upper: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Intersect scale bounds for unit-shape states [...,points,dof].

    Return alpha_upper, alpha_lower, feasible, each with the leading shape.
    Actual v/a/j scale by sqrt(alpha), alpha, alpha**1.5; torque is g+alpha*D.
    Bounds are exact at the supplied points, without an inward rounding margin.
    Infinity denotes an unbounded scale (for example, a stationary path).
    """
    valid = torch.ones_like(velocity[..., 0, 0], dtype=torch.bool)
    clean = []
    for value in (velocity, acceleration, jerk, dynamic_torque, gravity):
        valid = valid & torch.isfinite(value).all(dim=(-2, -1))
        clean.append(torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))
    v, a, j, dynamic, g = clean
    upper = torch.full_like(valid, torch.inf, dtype=velocity.dtype)
    for value, limit, exponent in (
        (v, velocity_limits, 2.0),
        (a, acceleration_limits, 1.0),
        (j, jerk_limits, 2.0 / 3.0),
    ):
        magnitude = value.abs()
        safe = torch.where(magnitude > 0, magnitude, 1.0)
        bound = torch.where(magnitude > 0, (limit / safe).pow(exponent), torch.inf)
        upper = torch.minimum(upper, bound.amin(dim=(-2, -1)))
    moving = dynamic != 0
    safe_dynamic = torch.where(moving, dynamic, 1.0)
    left = (torque_lower - g) / safe_dynamic
    right = (torque_upper - g) / safe_dynamic
    lower = torch.where(moving, torch.minimum(left, right), -torch.inf).amax(dim=(-2, -1))
    lower = lower.clamp_min(0.0)
    upper = torch.minimum(
        upper, torch.where(moving, torch.maximum(left, right), torch.inf).amin(dim=(-2, -1))
    )
    valid = valid & (moving | ((g >= torque_lower) & (g <= torque_upper))).all(dim=(-2, -1))
    valid = valid & (upper > 0) & (upper >= lower)
    return upper, lower, valid


class TimingReference:
    """Independent dense equations for tests; no iterative timeopt implementation."""

    def __init__(self, evaluation: TimeOptProblem) -> None:
        """Retain the sampled path and identity used by dense reference solves."""
        self.evaluation = evaluation
        self.samples = evaluation.basis.at(evaluation.grid_cpu)
        self.line_search = evaluation.basis.tensor(np.asarray([2.0**-k for k in range(8)]))
        self.eye = torch.eye(
            evaluation.basis.count, device=evaluation.path.device, dtype=torch.float32
        )[None]

    def duration(self, controls: torch.Tensor) -> torch.Tensor:
        """Evaluate duration at positive squared-speed controls [B,M] or [B,P,M]."""
        intervals = self.evaluation.num_intervals
        x = torch.matmul(controls, self.samples[0, : intervals + 1].T).clamp_min(1e-20)
        return x.rsqrt().mean(-1)

    def predictor_corrector_step(
        self,
        initial: torch.Tensor,
        base: torch.Tensor,
        rows: torch.Tensor,
        bounds: torch.Tensor,
        jerk: torch.Tensor,
    ) -> torch.Tensor:
        """Check one update for one path using dense autograd and Cholesky."""

        def slacks(value: torch.Tensor) -> torch.Tensor:
            """Evaluate independent inequalities in dense constraint order."""
            return self.constraints(value[None, None], base, rows, bounds, jerk)[0].flatten()

        def objective(value: torch.Tensor) -> torch.Tensor:
            """Evaluate normalized duration for a single path."""
            return self.duration(value[None])[0]

        slack = slacks(initial)
        dual = torch.full_like(slack, 1e-4)

        def lagrangian(value: torch.Tensor) -> torch.Tensor:
            """Include the nonlinear jerk constraint curvature."""
            return objective(value) - (dual * slacks(value)).sum()

        jacobian = -torch.autograd.functional.jacobian(slacks, initial, vectorize=True)
        gradient = torch.autograd.functional.jacobian(objective, initial)
        hessian = torch.autograd.functional.hessian(lagrangian, initial, vectorize=True)
        hessian += jacobian.T @ ((dual / slack)[:, None] * jacobian)
        scale = hessian.diagonal().clamp_min(1e-20).rsqrt()
        factor = torch.linalg.cholesky(
            hessian * scale[:, None] * scale[None, :] + 1e-6 * self.eye[0]
        )

        def solve(rhs: torch.Tensor) -> torch.Tensor:
            """Use two dense substitutions with diagonal equilibration."""
            return self.substitute(factor[None], scale[None], rhs[None])[0]

        predictor = solve(-gradient)
        ds = -(jacobian @ predictor)
        dl = -dual - dual / slack * ds
        ap, ad = self.boundary(slack, ds), self.boundary(dual, dl)
        gap = (slack * dual).mean()
        affine_gap = ((slack + ap * ds) * (dual + ad * dl)).mean()
        sigma = (affine_gap / gap).clamp(0, 1).pow(3)
        correction = ds * dl - sigma * gap
        step = solve(-gradient + jacobian.T @ (correction / slack))
        ap = self.boundary(slack, -(jacobian @ step), 0.995)
        candidates = initial[None] + (ap * self.line_search)[:, None] * step[None]
        candidate_slack = self.constraints(candidates[None], base, rows, bounds, jerk)[0][0]
        accepted = (candidate_slack > 0).all(-1) & torch.isfinite(candidate_slack).all(-1)
        chosen = candidates[accepted.to(torch.int32).argmax()]
        improved = accepted.any() & (objective(chosen) < objective(initial))
        return torch.where(improved, chosen, initial) * base[0]

    def linear_constraints(self, base: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Assemble static normalized inequalities in scaled timing controls y=c/base."""
        ev = self.evaluation
        b0, b1, b2 = self.samples
        q1, q2, q3 = ev.path[1], ev.path[2], ev.path[3]
        a, b, g = ev.torque_coefficients.unbind(1)
        vlim, alim, jlim, lower, upper = ev.limits
        torque_scale = torch.maximum(lower.abs(), upper.abs()).clamp_min(1e-12)
        velocity = q1[..., None].square() * b0[:, None] / vlim[None, None, :, None].square()
        acceleration = (q2[..., None] * b0[:, None] + 0.5 * q1[..., None] * b1[:, None]) / alim[
            None, None, :, None
        ]
        torque = (0.5 * a[..., None] * b1[:, None] + b[..., None] * b0[:, None]) / torque_scale[
            None, None, :, None
        ]
        jerk = (
            q3[..., None] * b0[:, None]
            + 1.5 * q2[..., None] * b1[:, None]
            + 0.5 * q1[..., None] * b2[:, None]
        ) / jlim[None, None, :, None]
        rows = torch.stack((velocity, acceleration, -acceleration, torque, -torque), 1)
        ones = torch.ones_like(g)
        bounds = torch.stack(
            (ones, ones, ones, (upper - g) / torque_scale, (g - lower) / torque_scale), 1
        )
        return (
            rows.flatten(1, 3) * base[:, None, None],
            bounds.flatten(1),
            jerk * base[:, None, None, None],
        )

    @staticmethod
    def boundary(value: torch.Tensor, delta: torch.Tensor, fraction: float = 1.0) -> torch.Tensor:
        """Largest positive step preserving positive slack or dual multipliers."""
        ratio = torch.where(delta < 0, -value / delta.clamp(max=-1e-30), torch.inf)
        return (fraction * ratio.amin(-1)).clamp(max=1.0)

    @staticmethod
    def substitute(
        factor: torch.Tensor,
        scale: torch.Tensor,
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        """Reference substitutions; the CUDA backend uses the stored lower band."""
        middle = torch.linalg.solve_triangular(factor, (rhs * scale)[..., None], upper=False)
        return (
            torch.linalg.solve_triangular(factor.transpose(1, 2), middle, upper=True)[..., 0]
            * scale
        )

    def constraints(
        self,
        state: torch.Tensor,
        base: torch.Tensor,
        rows: torch.Tensor,
        bounds: torch.Tensor,
        jerk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Actual slacks for candidates [B,P,M], plus squared speed and jerk factors."""
        x = (state @ self.samples[0].T) * base[:, None, None]
        jy = torch.einsum("bgdm,bpm->bpgd", jerk, state)
        value = x.sqrt()[..., None] * jy
        slack = torch.cat(
            (
                bounds[:, None] - (rows @ state.transpose(1, 2)).transpose(1, 2),
                (1 - value).flatten(2),
                (1 + value).flatten(2),
                state,
            ),
            -1,
        )
        return slack, x, jy

    def evaluate(self, log_shape: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return duration, scale interval, feasibility, and normalized shape [B,P,M]."""
        d = torch.exp(log_shape)
        d = d / d.mean(dim=-1, keepdim=True)
        sampled = torch.matmul(d, self.samples.reshape(-1, self.evaluation.basis.count).T)
        z, first, second = sampled.reshape(*d.shape[:-1], 3, -1).unbind(-2)
        z = z.clamp_min(torch.finfo(z.dtype).tiny)
        qs, qss, qsss = [self.evaluation.path[k, :, None] for k in (1, 2, 3)]
        a, b, g = [self.evaluation.torque_coefficients[:, k, None] for k in range(3)]
        speed = z.sqrt()[..., None]
        velocity = qs * speed
        acceleration = qss * z[..., None] + 0.5 * qs * first[..., None]
        jerk = speed * (
            qsss * z[..., None] + 1.5 * qss * first[..., None] + 0.5 * qs * second[..., None]
        )
        dynamic = 0.5 * a * first[..., None] + b * z[..., None]
        alpha, lower, feasible = feasible_scale(
            velocity, acceleration, jerk, dynamic, g, *self.evaluation.limits
        )
        unit_time = self.duration(d)
        safe_alpha = torch.where(torch.isfinite(alpha) & (alpha > 0), alpha, 1.0)
        duration = unit_time * safe_alpha.rsqrt()
        duration = torch.where(self.evaluation.stationary[:, None], 0.0, duration)
        return duration, alpha, lower, feasible, d
