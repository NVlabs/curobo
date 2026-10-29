# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare fused local-support optimization against independent dense float32 algebra."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from curobo._src.trajectory.timeopt.timeopt_basis import CubicTimingBasis, TimeOptProblem
from curobo._src.trajectory.timeopt.timeopt_solver import CudaTimeOptSolver, _StateOffset
from curobo._src.util.cuda_graph_util import GraphExecutor
from curobo.tests._src.trajectory.timeopt.reference_equations import TimingReference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def full_precision_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep dense FP32 algebra independent of IK's process-wide TF32 setting."""
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)


def prepare_coefficients(solver: CudaTimeOptSolver) -> None:
    """Synthesize the three inverse-dynamics states from the fixture's A/B/C."""
    a, b, gravity = solver.evaluation.torque_coefficients.unbind(1)
    torque = torch.stack((gravity, gravity + a, gravity + b), dim=1)
    solver.prepare_coefficients(torque)


@pytest.mark.parametrize("intervals", [1, 127, 512, 513])
def test_shared_duration_grid(intervals: int) -> None:
    """The objective averages inverse speed on uniform nodes, excluding knot guards."""
    device = torch.device("cuda:0")
    basis = CubicTimingBasis(32, device)
    uniform = np.linspace(0, 1, intervals + 1)
    grid = np.r_[uniform, basis.breaks_cpu[1:-1], 0.3, 0.3]
    path = torch.zeros((4, 1, len(grid), 1), device=device)
    torque = torch.zeros((1, 3, len(grid), 1), device=device)
    one = torch.ones(1, device=device)
    problem = TimeOptProblem(basis, grid, path, torque, (one, one, one, -one, one), intervals)
    solver = CudaTimeOptSolver(problem)
    duration_items = solver.meta[solver.meta[:, 0] == 2, 1].sort().values
    torch.testing.assert_close(
        duration_items, torch.arange(intervals + 1, device=device, dtype=torch.int32)
    )
    controls = torch.linspace(0.5, 2.0, 32, device=device)
    solver.state.zero_()
    solver.state[:, _StateOffset.CONTROLS : _StateOffset.CONTROLS + 32] = controls
    solver.state[:, _StateOffset.ITERATION_VALID] = 1
    solver.state[:, _StateOffset.BEST_COST] = torch.inf
    solver.candidate_masks.fill_(255)
    solver._launch(
        "accept",
        1,
        solver.candidate_masks,
        solver.basis,
        solver.point_first,
        solver.state,
        solver.output,
        solver.candidate_masks.shape[1],
        intervals + 1,
    )
    expected = (basis.at(uniform)[0] @ controls).rsqrt().mean()
    torch.testing.assert_close(solver.state[0, _StateOffset.BEST_COST], expected)


@pytest.mark.parametrize("points,dof", [(513, 7), (128, 2), (257, 64)])
def test_fused_shapes_and_graph_replay(points: int, dof: int) -> None:
    """Check independent feasibility across shapes, batch lanes and changed-input replay."""
    device = torch.device("cuda:0")
    basis = CubicTimingBasis(32, device)
    grid = np.linspace(0, 1, points, dtype=np.float32)
    s = torch.as_tensor(grid, device=device)[None, :, None]
    frequency = torch.linspace(8.0, 12.0, dof, device=device)[None, None]
    lanes = torch.tensor([0.8, 1.1], device=device)[:, None, None]
    path = torch.zeros((4, 2, points, dof), device=device)
    # Consistent derivatives of a smooth path with appreciable third derivatives;
    # unrelated synthetic q',q'',q''' can make the reference Gram matrix singular.
    path[1] = lanes * ((s - 1) * frequency).exp()
    path[2] = frequency * path[1]
    path[3] = frequency * path[2]
    torque = torch.zeros((2, 3, points, dof), device=device)
    torque[:, 0], torque[:, 1], torque[:, 2] = path[1] * 0.2, path[2] * 0.1, 0.3
    one = torch.ones(dof, device=device)
    ev = TimeOptProblem(
        basis, grid, path, torque, (2 * one, 3 * one, 4 * one, -one, one), len(grid) - 1
    )
    fused = CudaTimeOptSolver(ev)
    reference = TimingReference(ev)
    execute = GraphExecutor(lambda: fused.solve(1), device=device, clone_outputs=True)
    for factor in (1.0, 0.85):
        path[1:].mul_(factor)
        prepare_coefficients(fused)
        expected, good, _, ok = [x.clone() for x in fused.solve(1)]
        actual, fused_good, _, fused_ok = execute()
        assert good.all() and fused_good.all() and ok.all() and fused_ok.all()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    actual, _, uniform, fused_ok = fused.solve(10)
    assert fused_ok.all()
    assert (reference.duration(actual) < uniform * 0.999).all()
    # Independent exact scale check: current shape must need no slowdown beyond FP32 noise.
    _, upper, _, valid, _ = reference.evaluate(actual[:, None].log())
    scale = actual.mean(-1)
    assert valid.all() and (scale <= upper[:, 0] * 1.0001).all()


@pytest.mark.parametrize("dof", [7, 64])
def test_fused_search_matches_all_dense_candidates(dof: int) -> None:
    """Check all eight acceptance scales, complete rejection, and invalid batch lanes."""
    device = torch.device("cuda:0")
    controls = 32
    basis = CubicTimingBasis(controls, device)
    grid = np.linspace(0, 1, 67, dtype=np.float32)
    factors = torch.tensor([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 0], device=device)
    batch = factors.numel()
    s = torch.as_tensor(grid, device=device)[None, :, None]
    path = torch.zeros((4, batch, len(grid), dof), device=device)
    path[1], path[2], path[3] = 0.5 + s.square(), 2 * s, 2.0
    torque = torch.zeros((batch, 3, len(grid), dof), device=device)
    torque[:, 0], torque[:, 1], torque[:, 2] = 0.3 * path[1], 0.4 * path[2], 0.2
    one = torch.ones(dof, device=device)
    ev = TimeOptProblem(
        basis, grid, path, torque, (2 * one, 3 * one, 4 * one, -one, one), len(grid) - 1
    )
    solver = TimingReference(ev)
    native = CudaTimeOptSolver(ev)
    prepare_coefficients(native)
    native.solve(0)
    y = native.state[:, _StateOffset.CONTROLS : _StateOffset.CONTROLS + controls]
    y.mul_(0.25)
    step = native.state[:, _StateOffset.CORRECTOR : _StateOffset.CORRECTOR + controls]
    step.copy_(y * factors[:, None])
    native.state[-1, _StateOffset.ITERATION_VALID] = 0
    native.step_bounds.fill_(1.0)
    native._launch("search_bounds", batch, native.step_bounds, native.state, native.tiles)
    native._launch(
        "search",
        (native.candidate_masks.shape[1], batch),
        native.meta,
        native.offsets,
        native.basis,
        native.rows,
        native.bounds,
        native.state,
        native.candidate_masks,
        len(grid),
        dof,
        batch,
        native.tiles,
    )
    alpha = (0.995 * native.step_bounds[..., 0].amin(1)).clamp(max=1)
    candidates = y[:, None] + (alpha[:, None] * solver.line_search)[..., None] * step[:, None]
    rows, bounds, jerk = solver.linear_constraints(native.state[:, _StateOffset.SPEED_SCALE])
    slack = solver.constraints(
        candidates, native.state[:, _StateOffset.SPEED_SCALE], rows, bounds, jerk
    )[0]
    expected = torch.isfinite(slack).all(-1) & (slack > 0).all(-1)
    expected[-1] = False
    indices = torch.arange(8, device=device)
    rejected = ((native.candidate_masks[..., None] >> indices) & 1).any(1)
    torch.testing.assert_close(~rejected, expected)
    first_valid = torch.where(expected, indices, 8).amin(1)
    assert set(first_valid.tolist()) == set(range(9))
    expected_y = torch.where(
        (first_valid < 8)[:, None],
        candidates[torch.arange(batch, device=device), first_valid.clamp(max=7)],
        y,
    )
    native._launch(
        "accept",
        batch,
        native.candidate_masks,
        native.basis,
        native.point_first,
        native.state,
        native.output,
        native.candidate_masks.shape[1],
        ev.num_intervals + 1,
    )
    torch.testing.assert_close(y, expected_y, rtol=0, atol=0)


@pytest.mark.parametrize("fixed", [0, 2])
def test_fused_factor_and_reused_corrector(fixed: int) -> None:
    """Check both band substitutions against dense Cholesky on a conditioned SPD system."""
    device = torch.device("cuda:0")
    controls = 32
    basis = CubicTimingBasis(controls, device)
    points, batch = 65, 2
    grid = np.linspace(0, 1, points, dtype=np.float32)
    path = torch.zeros((4, batch, points, 1), device=device)
    torque = torch.zeros((batch, 3, points, 1), device=device)
    one = torch.ones(1, device=device)
    ev = TimeOptProblem(basis, grid, path, torque, (one, one, one, -one, one), len(grid) - 1)
    native = CudaTimeOptSolver(ev)
    native.state.zero_()
    native.state[:, _StateOffset.NUMERICAL_OK] = 1
    native.state[:, _StateOffset.FIXED_CONTROLS] = fixed
    native.partials.zero_()
    generator = torch.Generator(device=device).manual_seed(19)
    local = torch.randn((batch, native.tiles, 6, 4), device=device, generator=generator)
    gram = local.transpose(-2, -1) @ local + torch.eye(4, device=device)
    gradients = torch.randn((batch, native.tiles, 4), device=device, generator=generator)
    native.partials[..., :4] = gradients
    lower = torch.tril_indices(4, 4, device=device)
    native.partials[..., 4:14] = gram[..., lower[0], lower[1]]
    hessian = torch.zeros((batch, controls, controls), device=device)
    gradient = torch.zeros((batch, controls), device=device)
    for tile, first in enumerate(native.offsets.cpu().tolist()):
        hessian[:, first : first + 4, first : first + 4] += gram[:, tile]
        gradient[:, first : first + 4] += gradients[:, tile]
    native._launch(
        "factor_reduce",
        (5 * controls // 8 + 1, batch),
        native.partials,
        native.offsets,
        native.starts,
        native.state,
        native.tiles,
    )
    band = torch.zeros((batch, controls, 4), device=device)
    for diagonal in range(4):
        band[:, diagonal:, diagonal] = hessian.diagonal(offset=-diagonal, dim1=-2, dim2=-1)
    torch.testing.assert_close(
        native.state[:, _StateOffset.GRADIENT : _StateOffset.GRADIENT + controls], gradient
    )
    torch.testing.assert_close(
        native.state[:, _StateOffset.FACTOR : _StateOffset.FACTOR + 4 * controls].reshape_as(band),
        band,
    )
    hessian[:, :fixed, :] = 0
    hessian[:, :, :fixed] = 0
    hessian[:, torch.arange(fixed), torch.arange(fixed)] = 1
    scale = hessian.diagonal(dim1=-2, dim2=-1).rsqrt()
    factor = torch.linalg.cholesky(
        hessian * scale[:, :, None] * scale[:, None, :] + 1e-6 * torch.eye(controls, device=device)
    )
    rhs = -gradient.clone()
    rhs[:, :fixed] = 0
    expected = TimingReference.substitute(factor, scale, rhs)
    native._launch("factor", batch, native.state, native.numerical_ok)
    assert native.numerical_ok.all()
    torch.testing.assert_close(
        native.state[:, _StateOffset.PREDICTOR : _StateOffset.PREDICTOR + controls],
        expected,
        rtol=2e-5,
        atol=2e-6,
    )
    # Independent RHS and nonzero centering, retaining the predictor's factors.
    native.partials.zero_()
    local_rhs = torch.randn((batch, native.tiles, 8), device=device, generator=generator)
    native.partials[..., :8] = local_rhs
    native.partials[..., 8:11] = 1
    left, right = torch.zeros_like(gradient), torch.zeros_like(gradient)
    for tile, first in enumerate(native.offsets.cpu().tolist()):
        left[:, first : first + 4] += local_rhs[:, tile, :4]
        right[:, first : first + 4] += local_rhs[:, tile, 4:]
    native._launch(
        "correct_reduce",
        (controls // 8 + 1, batch),
        native.partials,
        native.offsets,
        native.starts,
        native.corrector_terms,
        native.state,
        native.tiles,
        native.constraint_count,
    )
    torch.testing.assert_close(native.corrector_terms[:, 0], left)
    torch.testing.assert_close(native.corrector_terms[:, 1], right)
    native._launch("correct", batch, native.corrector_terms, native.state, native.numerical_ok)
    rhs = -gradient + left - (native.tiles / native.constraint_count) * right
    rhs[:, :fixed] = 0
    expected = TimingReference.substitute(factor, scale, rhs)
    assert native.numerical_ok.all()
    torch.testing.assert_close(
        native.state[:, _StateOffset.CORRECTOR : _StateOffset.CORRECTOR + controls],
        expected,
        rtol=3e-5,
        atol=3e-6,
    )


@pytest.mark.parametrize("points", [37, 38])
def test_grouped_assembly_matches_autograd(points: int) -> None:
    """Paired constraints preserve the dense Lagrangian Hessian with unequal multipliers."""
    device = torch.device("cuda:0")
    controls = 32
    basis = CubicTimingBasis(controls, device)
    dof = 3
    grid = np.linspace(0, 1, points, dtype=np.float32)
    s = torch.as_tensor(grid, device=device)[None, :, None]
    path = torch.zeros((4, 1, points, dof), device=device)
    path[1] = 0.5 + s.square()
    path[2] = 2 * s
    path[3] = 2.0
    torque = torch.zeros((1, 3, points, dof), device=device)
    torque[:, 0], torque[:, 1], torque[:, 2] = 0.3 * path[1], 0.4 * path[2], 0.2
    one = torch.ones(dof, device=device)
    ev = TimeOptProblem(
        basis, grid, path, torque, (2 * one, 3 * one, 4 * one, -one, one), len(grid) - 1
    )
    solver = TimingReference(ev)
    native = CudaTimeOptSolver(ev)
    prepare_coefficients(native)
    native.solve(0)
    native.row_state.zero_()
    native.row_state[1].uniform_(1e-5, 2e-4)
    native._launch(
        "assemble",
        (native.tiles, 1),
        native.meta,
        native.offsets,
        native.basis,
        native.rows,
        native.bounds,
        native.state,
        native.row_state,
        native.jacobian,
        native.partials,
        points,
        ev.num_intervals + 1,
        dof,
        1,
        native.tiles,
        0,
    )
    dense_dual = torch.empty(native.constraint_count, device=device)
    width = points * dof
    metadata = native.meta.cpu().tolist()
    for slot, (kind, item) in enumerate(metadata):
        if kind == 0:
            for constraint in range(7):
                dense_dual[constraint * width + item] = native.row_state[1, constraint, 0, slot]
        elif kind == 1:
            dense_dual[7 * width + item] = native.row_state[1, 0, 0, slot]
    rows, bounds, jerk = solver.linear_constraints(native.state[:, _StateOffset.SPEED_SCALE])
    y = native.state[0, _StateOffset.CONTROLS : _StateOffset.CONTROLS + controls].clone()

    def slacks(value: torch.Tensor) -> torch.Tensor:
        """Independent dense constraint equations in the reference row order."""
        return solver.constraints(
            value[None, None], native.state[:, _StateOffset.SPEED_SCALE], rows, bounds, jerk
        )[0].flatten()

    def objective(value: torch.Tensor) -> torch.Tensor:
        """Normalized duration used by both primal-dual solvers."""
        return solver.duration(value[None])[0]

    def lagrangian(value: torch.Tensor) -> torch.Tensor:
        """Use the actual nonlinear constraint Hessian, including signed jerk curvature."""
        return objective(value) - (dense_dual * slacks(value)).sum()

    jacobian = torch.autograd.functional.jacobian(slacks, y, vectorize=True)
    hessian = torch.autograd.functional.hessian(lagrangian, y, vectorize=True)
    hessian += jacobian.T @ ((dense_dual / slacks(y))[:, None] * jacobian)
    gradient = torch.autograd.functional.jacobian(objective, y)
    actual_hessian = torch.zeros_like(hessian)
    actual_gradient = torch.zeros_like(gradient)
    lower = torch.tril_indices(4, 4, device=device)
    for tile, first in enumerate(native.offsets.cpu().tolist()):
        local = torch.zeros((4, 4), device=device)
        local[lower[0], lower[1]] = native.partials[0, tile, 4:14]
        local = local + local.T - torch.diag(local.diagonal())
        actual_hessian[first : first + 4, first : first + 4] += local
        actual_gradient[first : first + 4] += native.partials[0, tile, :4]
    torch.testing.assert_close(actual_gradient, gradient, rtol=3e-5, atol=1e-6)
    torch.testing.assert_close(actual_hessian, hessian, rtol=5e-4, atol=2e-4)

    # Check one predictor/corrector update from the common seed using dense
    # autograd derivatives, without retaining another iterative solver.
    native.solve(0)
    initial = native.state[0, _StateOffset.CONTROLS : _StateOffset.CONTROLS + controls].clone()
    base = native.state[:, _StateOffset.SPEED_SCALE].clone()
    rows, bounds, jerk = solver.linear_constraints(base)
    expected = solver.predictor_corrector_step(initial, base, rows, bounds, jerk)
    actual, feasible, _, numerical = native.solve(1)
    assert feasible.all() and numerical.all()
    torch.testing.assert_close(actual[0], expected, rtol=2e-3, atol=2e-6)


def test_reference_start_at_velocity_limit() -> None:
    """A fixed endpoint on its limit must not prevent optimizing the free time law."""
    device = torch.device("cuda:0")
    basis = CubicTimingBasis(32, device)
    grid = np.linspace(0, 1, 129, dtype=np.float32)
    s = torch.as_tensor(grid, device=device)[None, :, None]
    path = torch.zeros((4, 1, len(grid), 1), device=device)
    path[0], path[1], path[2], path[3] = 1 - (-s).exp(), (-s).exp(), -(-s).exp(), (-s).exp()
    torque = torch.zeros((1, 3, len(grid), 1), device=device)
    one = torch.ones(1, device=device)
    ev = TimeOptProblem(
        basis, grid, path, torque, (one, 5 * one, 100 * one, -one, one), len(grid) - 1
    )
    solver = CudaTimeOptSolver(ev)
    prepare_coefficients(solver)
    solver.reference_speed.fill_(1.0)
    timing, good, uniform, numerical = solver.solve(10)
    assert good.all() and numerical.all()
    torch.testing.assert_close(timing[:, :2], torch.ones_like(timing[:, :2]), rtol=0, atol=0)
    reference = TimingReference(ev)
    assert (reference.duration(timing) < 0.99 * uniform).all()
    _, upper, _, valid, _ = reference.evaluate(timing[:, None].log())
    assert valid.all() and (timing.mean(-1) <= upper[:, 0] * 1.0001).all()
