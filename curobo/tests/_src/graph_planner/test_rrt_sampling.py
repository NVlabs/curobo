# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for fixed-shape informed RRT ellipsoid sampling."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.graph_planner.rrt.rrt_sampling import (
    InformedEllipsoidRRTTransform,
    InformedRRTProposalGenerator,
    InformedRRTProposalTransform,
    UnitBallRRTTransform,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_module(
    weights: torch.Tensor,
    lows: torch.Tensor,
    highs: torch.Tensor,
    informed_slots: int,
    informed_start: int = 0,
) -> InformedEllipsoidRRTTransform:
    """Construct the pure transform with its fixed robot sampling profile."""
    return InformedEllipsoidRRTTransform(
        weights,
        lows,
        highs,
        informed_slots,
        informed_start,
    ).to(weights.device)


def _weighted_focus_sum(
    samples: torch.Tensor,
    starts: torch.Tensor,
    goals: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Return the prolate-hyperspheroid membership value in weighted c-space."""
    start_distance = torch.linalg.vector_norm(
        (samples - starts[:, None]) * weights,
        dim=-1,
    )
    goal_distance = torch.linalg.vector_norm(
        (samples - goals[:, None]) * weights,
        dim=-1,
    )
    return start_distance + goal_distance


def _exported_samples(module, inputs):
    """Export the fixed-shape transform and return its sample tensor."""
    exported = torch.export.export(module, inputs, strict=True)
    return exported.module()(*inputs)


@pytest.mark.parametrize("action_dim", [1, 3, 7])
def test_unit_ball_transform_uses_radial_coordinate_and_exports(action_dim: int):
    """The final uniform coordinate fills the ball instead of sampling its shell."""
    device = torch.device("cuda")
    # All rows use the same nonzero direction draw. Only the final uniform
    # coordinate changes, so output radii must be exactly u**(1 / D).
    direction_draw = torch.linspace(0.2, 0.8, action_dim, device=device)
    radial_draw = torch.tensor([0.0, 0.125, 1.0], device=device)
    unit_cube = torch.empty((1, 3, action_dim + 1), device=device)
    unit_cube[..., :action_dim] = direction_draw
    unit_cube[..., action_dim] = radial_draw
    module = UnitBallRRTTransform(action_dim).to(device)

    unit_ball = module(unit_cube)
    radii = torch.linalg.vector_norm(unit_ball, dim=-1)
    expected = radial_draw.pow(1.0 / action_dim).view(1, -1)

    assert torch.all(torch.isfinite(unit_ball))
    assert torch.all(radii <= 1.0 + 2.0e-6)
    torch.testing.assert_close(radii, expected, rtol=2.0e-5, atol=2.0e-6)
    assert radii[0, 0] == 0.0
    assert 0.0 < radii[0, 1] < 1.0
    torch.testing.assert_close(
        _exported_samples(module, (unit_cube,)),
        unit_ball,
        rtol=0.0,
        atol=0.0,
    )


def test_unit_ball_transform_has_uniform_volume_radius_and_finite_cube_edges():
    """For uniform input, r**D remains uniform and cube endpoints stay finite."""
    device = torch.device("cuda")
    action_dim = 7
    generator = torch.Generator(device=device).manual_seed(29)
    unit_cube = torch.rand(
        (4, 256, action_dim + 1),
        device=device,
        generator=generator,
    )
    unit_cube[0, 0].zero_()
    unit_cube[0, 1].fill_(1.0)
    unit_cube[0, 2, :action_dim].fill_(0.5)
    unit_cube[0, 2, -1] = 1.0
    module = UnitBallRRTTransform(action_dim).to(device)

    unit_ball = module(unit_cube)
    radii = torch.linalg.vector_norm(unit_ball, dim=-1)

    assert torch.all(torch.isfinite(unit_ball))
    assert torch.all(radii <= 1.0 + 2.0e-6)
    # The radial transform must preserve the final uniform coordinate as
    # r**D, including the deterministic fallback for a zero direction draw.
    torch.testing.assert_close(
        radii.pow(action_dim),
        unit_cube[..., -1],
        rtol=3.0e-5,
        atol=3.0e-6,
    )
    exported = _exported_samples(module, (unit_cube,))
    torch.testing.assert_close(exported, unit_ball, rtol=0.0, atol=0.0)


def test_weighted_ellipsoid_obeys_focal_sum_and_preserves_volume_radius():
    """Map the full unit ball, including its interior, into weighted c-space."""
    device = torch.device("cuda")
    dtype = torch.float32
    weights = torch.tensor([2.0, 0.5, 1.5], device=device, dtype=dtype)
    lows = torch.full((3,), -20.0, device=device, dtype=dtype)
    highs = torch.full((3,), 20.0, device=device, dtype=dtype)
    starts = torch.tensor([[-1.0, -0.5, 0.25]], device=device, dtype=dtype)
    goals = torch.tensor([[1.0, 1.5, -0.75]], device=device, dtype=dtype)
    c_min = torch.linalg.vector_norm((goals - starts) * weights, dim=-1)
    c_max = c_min * 1.5
    informed_slots = 6
    fallback = torch.full(
        (1, informed_slots + 2, 3),
        -9.0,
        device=device,
        dtype=dtype,
    )
    unit_ball = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.0, 0.0],
                [0.75, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
            ]
        ],
        device=device,
        dtype=dtype,
    )
    module = _make_module(weights, lows, highs, informed_slots)

    samples, accepted = module(fallback, unit_ball, starts, goals, c_max)
    focused = samples[:, :informed_slots]
    focus_sum = _weighted_focus_sum(focused, starts, goals, weights)
    tolerance = 2.0e-5

    assert torch.all(focus_sum <= c_max[:, None] + tolerance)
    torch.testing.assert_close(
        focused[:, 0],
        (starts + goals) * 0.5,
        rtol=0.0,
        atol=tolerance,
    )
    # Collinear latent samples at radii 0.25 and 0.75 must retain those
    # distinct interior radii. Normalizing every latent would collapse these
    # samples onto the shell and fail this ratio check.
    center = (starts + goals) * 0.5
    weighted_displacement = (focused - center[:, None]) * weights
    radial_norm = torch.linalg.vector_norm(weighted_displacement, dim=-1)
    torch.testing.assert_close(
        radial_norm[:, 2],
        3.0 * radial_norm[:, 1],
        rtol=2.0e-5,
        atol=tolerance,
    )
    assert radial_norm[0, 1] > 0.0
    # Unit-ball boundary points map to the ellipsoid boundary regardless of
    # whether they lie along the major or a minor latent axis.
    torch.testing.assert_close(
        focus_sum[:, 3:6],
        c_max[:, None].expand(-1, 3),
        rtol=2.0e-5,
        atol=tolerance,
    )
    torch.testing.assert_close(samples[:, informed_slots:], fallback[:, informed_slots:])
    assert accepted.tolist() == [[True] * informed_slots]


def test_unequal_weights_orient_major_axis_in_weighted_coordinates():
    """The first latent axis follows W(goal-start), not the unweighted q direction."""
    device = torch.device("cuda")
    weights = torch.tensor([4.0, 0.5, 2.0], device=device)
    lows = torch.full((3,), -20.0, device=device)
    highs = torch.full((3,), 20.0, device=device)
    starts = torch.tensor([[-0.5, -1.0, 0.25]], device=device)
    goals = torch.tensor([[1.0, 2.0, -0.75]], device=device)
    weighted_direction = (goals - starts) * weights
    c_min = torch.linalg.vector_norm(weighted_direction, dim=-1)
    c_max = c_min * 1.4
    unit_ball = torch.tensor(
        [[[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]],
        device=device,
    )
    fallback = torch.zeros((1, 3, 3), device=device)
    module = _make_module(weights, lows, highs, informed_slots=3)

    samples, accepted = module(fallback, unit_ball, starts, goals, c_max)
    center_z = ((starts + goals) * 0.5) * weights
    sample_z = samples * weights
    unit_direction = weighted_direction / c_min[:, None]
    major = c_max * 0.5
    expected_positive = center_z + major[:, None] * unit_direction
    expected_negative = center_z - major[:, None] * unit_direction

    torch.testing.assert_close(sample_z[:, 0], expected_positive, rtol=2.0e-5, atol=2.0e-5)
    torch.testing.assert_close(sample_z[:, 1], expected_negative, rtol=2.0e-5, atol=2.0e-5)
    minor_displacement = sample_z[:, 2] - center_z
    assert torch.all(torch.abs((minor_displacement * unit_direction).sum(dim=-1)) < 2.0e-5)
    expected_minor = 0.5 * torch.sqrt(c_max.square() - c_min.square())
    torch.testing.assert_close(
        torch.linalg.vector_norm(minor_displacement, dim=-1),
        expected_minor,
        rtol=2.0e-5,
        atol=2.0e-5,
    )
    assert accepted.tolist() == [[True, True, True]]


@pytest.mark.parametrize("c_max_value", [0.0, 2.0])
def test_equal_foci_are_finite_and_have_spherical_weighted_geometry(c_max_value: float):
    """Coincident foci avoid a zero-direction divide and use a deterministic frame."""
    device = torch.device("cuda")
    weights = torch.tensor([2.0, 0.5, 4.0], device=device)
    lows = torch.full((3,), -10.0, device=device)
    highs = torch.full((3,), 10.0, device=device)
    starts = torch.tensor([[0.5, -1.0, 0.25]], device=device)
    goals = starts.clone()
    c_max = torch.tensor([c_max_value], device=device)
    unit_ball = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, -1.0, 0.0]]],
        device=device,
    )
    fallback = torch.full((1, 3, 3), 7.0, device=device)
    module = _make_module(weights, lows, highs, informed_slots=3)

    samples, accepted = module(fallback, unit_ball, starts, goals, c_max)

    assert torch.all(torch.isfinite(samples))
    weighted_radius = torch.linalg.vector_norm((samples - starts[:, None]) * weights, dim=-1)
    expected_radius = torch.linalg.vector_norm(unit_ball, dim=-1) * (0.5 * c_max[:, None])
    torch.testing.assert_close(weighted_radius, expected_radius, rtol=2.0e-5, atol=2.0e-5)
    assert torch.all(
        _weighted_focus_sum(samples, starts, goals, weights) <= c_max[:, None] + 2.0e-5
    )
    assert accepted.tolist() == [[True, True, True]]


def test_oob_informed_slots_fall_back_per_slot_and_mixed_slots_are_preserved():
    """Reject OOB transforms without clamping and leave uniform slots bit-exact."""
    device = torch.device("cuda")
    weights = torch.ones(2, device=device)
    lows = torch.full((2,), -1.0, device=device)
    highs = torch.full((2,), 1.0, device=device)
    starts = torch.tensor([[0.0, 0.0]], device=device)
    goals = torch.tensor([[0.2, 0.0]], device=device)
    c_max = torch.tensor([4.0], device=device)
    fallback = torch.tensor(
        [[[0.11, 0.12], [0.21, 0.22], [0.31, 0.32], [0.41, 0.42]]],
        device=device,
    )
    # Center is 0.1. The first focused latent maps to the center and is valid;
    # the second maps beyond q=1 and must use its own original fallback.
    unit_ball = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], device=device)
    module = _make_module(weights, lows, highs, informed_slots=2)

    samples, accepted = module(fallback, unit_ball, starts, goals, c_max)

    torch.testing.assert_close(samples[0, 0], torch.tensor([0.1, 0.0], device=device))
    torch.testing.assert_close(samples[0, 1], fallback[0, 1], rtol=0.0, atol=0.0)
    torch.testing.assert_close(samples[0, 2:], fallback[0, 2:], rtol=0.0, atol=0.0)
    assert torch.all(samples >= lows)
    assert torch.all(samples <= highs)
    assert accepted.tolist() == [[True, False]]


def test_nonzero_informed_start_preserves_goal_bias_prefix():
    """Replace only informed slots after the exact goal-bias prefix."""
    device = torch.device("cuda")
    weights = torch.ones(2, device=device)
    lows = torch.full((2,), -10.0, device=device)
    highs = torch.full((2,), 10.0, device=device)
    starts = torch.tensor([[-1.0, 0.0]], device=device)
    goals = torch.tensor([[1.0, 0.0]], device=device)
    c_max = torch.tensor([3.0], device=device)
    goal_bias_count = 2
    informed_slots = 2
    fallback = torch.tensor(
        [
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [-7.0, -7.0],
                [-6.0, -6.0],
                [4.0, 5.0],
            ]
        ],
        device=device,
    )
    unit_ball = torch.tensor(
        [[[0.0, 0.0], [0.5, 0.0]]],
        device=device,
    )
    module = _make_module(
        weights,
        lows,
        highs,
        informed_slots,
        informed_start=goal_bias_count,
    )

    samples, accepted = module(fallback, unit_ball, starts, goals, c_max)

    torch.testing.assert_close(
        samples[:, :goal_bias_count],
        fallback[:, :goal_bias_count],
        rtol=0.0,
        atol=0.0,
    )
    assert accepted.tolist() == [[True, True]]
    assert not torch.equal(
        samples[:, goal_bias_count : goal_bias_count + informed_slots],
        fallback[:, goal_bias_count : goal_bias_count + informed_slots],
    )
    torch.testing.assert_close(
        samples[:, goal_bias_count + informed_slots :],
        fallback[:, goal_bias_count + informed_slots :],
        rtol=0.0,
        atol=0.0,
    )


def test_informed_proposal_transform_matches_composition_and_torch_export():
    """Fuse unit-cube volume sampling and mixed ellipsoid transformation exactly."""
    device = torch.device("cuda")
    weights = torch.tensor([1.0, 2.0, 0.5], device=device)
    lows = torch.full((3,), -10.0, device=device)
    highs = torch.full((3,), 10.0, device=device)
    informed_start = 1
    informed_slots = 3
    fallback = torch.tensor(
        [
            [
                [2.0, 0.0, 0.0],
                [-8.0, -8.0, -8.0],
                [-7.0, -7.0, -7.0],
                [-6.0, -6.0, -6.0],
                [5.0, 5.0, 5.0],
            ],
            [
                [-2.0, 0.0, 0.0],
                [8.0, 8.0, 8.0],
                [7.0, 7.0, 7.0],
                [6.0, 6.0, 6.0],
                [-5.0, -5.0, -5.0],
            ],
        ],
        device=device,
    )
    unit_cube = torch.tensor(
        [
            [
                [0.2, 0.5, 0.8, 0.125],
                [0.7, 0.3, 0.6, 0.5],
                [0.9, 0.4, 0.1, 1.0],
            ],
            [
                [0.8, 0.5, 0.2, 0.25],
                [0.3, 0.7, 0.4, 0.75],
                [0.1, 0.6, 0.9, 1.0],
            ],
        ],
        device=device,
    )
    starts = torch.tensor([[-1.0, 0.0, 0.5], [1.0, -0.5, 0.0]], device=device)
    goals = torch.tensor([[2.0, 1.0, -0.5], [-2.0, 0.5, 1.0]], device=device)
    c_min = torch.linalg.vector_norm((goals - starts) * weights, dim=-1)
    c_max = c_min * 1.4
    proposal_transform = InformedRRTProposalTransform(
        weights,
        lows,
        highs,
        informed_slots,
        informed_start,
    ).to(device)
    unit_ball = UnitBallRRTTransform(weights.numel()).to(device)
    ellipsoid = _make_module(
        weights,
        lows,
        highs,
        informed_slots,
        informed_start,
    )
    inputs = (fallback, unit_cube, starts, goals, c_max)

    eager_samples, eager_accepted = proposal_transform(*inputs)
    expected_samples, expected_accepted = ellipsoid(
        fallback,
        unit_ball(unit_cube),
        starts,
        goals,
        c_max,
    )
    exported_samples, exported_accepted = _exported_samples(proposal_transform, inputs)

    torch.testing.assert_close(eager_samples, expected_samples, rtol=0.0, atol=0.0)
    assert torch.equal(eager_accepted, expected_accepted)
    torch.testing.assert_close(exported_samples, eager_samples, rtol=0.0, atol=0.0)
    assert torch.equal(exported_accepted, eager_accepted)
    torch.testing.assert_close(eager_samples[:, :informed_start], fallback[:, :informed_start])
    torch.testing.assert_close(
        eager_samples[:, informed_start + informed_slots :],
        fallback[:, informed_start + informed_slots :],
    )


@pytest.mark.parametrize(
    "scale",
    (
        torch.tensor(1.2, device="cuda"),
        torch.tensor([1.1, 1.4], device="cuda"),
    ),
    ids=("scalar_scale", "per_lane_scale"),
)
def test_raw_proposal_transform_matches_manual_decode_radius_and_export(scale: torch.Tensor):
    """Fuse one raw draw exactly like bounded fallback plus informed composition."""
    device = torch.device("cuda")
    batch_size = 2
    action_dim = 3
    total_slots = 5
    informed_slots = 2
    informed_start = 1
    min_slack = 0.5
    weights = torch.tensor([1.0, 2.0, 0.5], device=device)
    lows = torch.tensor([-3.0, -4.0, -5.0], device=device)
    highs = torch.tensor([4.0, 5.0, 6.0], device=device)
    latent_values = informed_slots * (action_dim + 1)
    latent_rows = (latent_values + action_dim - 1) // action_dim
    raw_rows = total_slots + latent_rows
    raw_uniform = torch.linspace(
        0.01,
        0.99,
        batch_size * raw_rows * action_dim,
        device=device,
    ).reshape(batch_size, raw_rows, action_dim)
    starts = torch.tensor([[0.0, 0.0, 0.0], [-1.0, -0.5, 0.25]], device=device)
    goals = torch.tensor([[0.2, 0.0, 0.0], [2.0, 1.0, -0.75]], device=device)
    raw_wave = InformedRRTProposalGenerator(
        weights,
        lows,
        highs,
        total_slots,
        informed_slots,
        informed_start,
        min_slack,
    ).to(device)
    composed_wave = InformedRRTProposalTransform(
        weights,
        lows,
        highs,
        informed_slots,
        informed_start,
    ).to(device)
    inputs = (raw_uniform, starts, goals, scale)

    fallback = raw_uniform[:, :total_slots] * (highs - lows) + lows
    unit_cube = raw_uniform[:, total_slots:].reshape(batch_size, -1)
    unit_cube = unit_cube[:, :latent_values].reshape(
        batch_size,
        informed_slots,
        action_dim + 1,
    )
    c_min = torch.linalg.vector_norm((goals - starts) * weights, dim=-1)
    expected_c_max = torch.maximum(scale * c_min, c_min + min_slack)
    expected_samples, expected_accepted = composed_wave(
        fallback,
        unit_cube,
        starts,
        goals,
        expected_c_max,
    )

    eager_samples, eager_accepted = raw_wave(*inputs)
    exported_samples, exported_accepted = _exported_samples(raw_wave, inputs)

    torch.testing.assert_close(eager_samples, expected_samples, rtol=0.0, atol=0.0)
    assert torch.equal(eager_accepted, expected_accepted)
    torch.testing.assert_close(exported_samples, eager_samples, rtol=0.0, atol=0.0)
    assert torch.equal(exported_accepted, eager_accepted)
    torch.testing.assert_close(
        expected_c_max,
        torch.maximum(scale * c_min, c_min + min_slack),
        rtol=0.0,
        atol=0.0,
    )

    # The row packing can leave unused raw scalars after the I*(D+1) latent
    # prefix. They must not leak into the fixed proposal wave.
    used_values = latent_values
    available_values = latent_rows * action_dim
    if available_values > used_values:
        modified_raw = raw_uniform.clone()
        flat_latent = modified_raw[:, total_slots:].reshape(batch_size, -1)
        flat_latent[:, used_values:] = 1.0 - flat_latent[:, used_values:]
        modified_samples, modified_accepted = raw_wave(
            modified_raw,
            starts,
            goals,
            scale,
        )
        torch.testing.assert_close(modified_samples, eager_samples, rtol=0.0, atol=0.0)
        assert torch.equal(modified_accepted, eager_accepted)


def test_batched_lanes_are_independent_and_export_matches_eager():
    """Changing one focus pair cannot alter another lane with fixed latent draws."""
    device = torch.device("cuda")
    weights = torch.tensor([1.0, 2.0, 0.75], device=device)
    lows = torch.full((3,), -20.0, device=device)
    highs = torch.full((3,), 20.0, device=device)
    fallback = torch.tensor(
        [
            [[-8.0, -8.0, -8.0], [-7.0, -7.0, -7.0], [-6.0, -6.0, -6.0]],
            [[8.0, 8.0, 8.0], [7.0, 7.0, 7.0], [6.0, 6.0, 6.0]],
        ],
        device=device,
    )
    unit_ball = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.3, -0.2, 0.1]],
            [[0.2, 0.1, -0.4], [-0.5, 0.0, 0.0]],
        ],
        device=device,
    )
    starts = torch.tensor([[-1.0, 0.0, 0.5], [2.0, 1.0, -1.0]], device=device)
    goals = torch.tensor([[1.0, 0.5, -0.5], [4.0, -1.0, 1.0]], device=device)
    c_min = torch.linalg.vector_norm((goals - starts) * weights, dim=-1)
    c_max = c_min * 1.5
    module = _make_module(weights, lows, highs, informed_slots=2)
    inputs = (fallback, unit_ball, starts, goals, c_max)

    eager_samples, eager_accepted = module(*inputs)
    exported_samples, exported_accepted = _exported_samples(module, inputs)
    torch.testing.assert_close(exported_samples, eager_samples, rtol=0.0, atol=0.0)
    assert torch.equal(exported_accepted, eager_accepted)

    changed_starts = starts.clone()
    changed_goals = goals.clone()
    changed_starts[1] = torch.tensor([-5.0, 3.0, 2.0], device=device)
    changed_goals[1] = torch.tensor([-2.0, 4.0, -3.0], device=device)
    changed_c_max = (
        torch.linalg.vector_norm(
            (changed_goals - changed_starts) * weights,
            dim=-1,
        )
        * 1.5
    )
    changed, changed_accepted = module(
        fallback,
        unit_ball,
        changed_starts,
        changed_goals,
        changed_c_max,
    )

    torch.testing.assert_close(changed[0], eager_samples[0], rtol=0.0, atol=0.0)
    assert not torch.allclose(changed[1, :2], eager_samples[1, :2])
    torch.testing.assert_close(changed[:, 2:], fallback[:, 2:], rtol=0.0, atol=0.0)
    assert torch.equal(changed_accepted[0], eager_accepted[0])
