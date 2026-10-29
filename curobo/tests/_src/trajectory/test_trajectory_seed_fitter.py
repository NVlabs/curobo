# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for fixed-iteration time-optimal B-spline seed fitting."""

# Standard Library
import inspect

# Third Party
import pytest
import torch
import torch.nn.functional as F

from curobo._src.state.state_joint import JointState

# CuRobo
from curobo._src.trajectory.trajectory_seed_fitter import (
    BSplineTimeOptimalSeedFitter,
    BSplineTimeOptimalSeedFitterCfg,
)
from curobo._src.types.control_space import ControlSpace
from curobo._src.types.device_cfg import DeviceCfg

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("lanes,steps,dof", [(1, 17, 7), (4, 81, 7), (32, 83, 29)])
def test_derivative_scale_matches_reference(lanes: int, steps: int, dof: int) -> None:
    """Normalize each path separately, including partial and multi-pass blocks."""
    from curobo._src.curobolib.backends import trajectory_seed_fitter as backend

    torch.manual_seed(42)
    velocity = torch.randn((lanes, steps, dof), device="cuda")
    acceleration = torch.randn_like(velocity)
    limits = [torch.rand(dof, device="cuda") + 0.5 for _ in range(3)]
    bases = [torch.randn((steps, 15), device="cuda") for _ in range(3)]
    workspaces = [torch.empty_like(velocity) for _ in range(3)]
    output = torch.full((lanes,), torch.nan, device="cuda")
    backend.launch_derivative_scale_kernel(
        velocity,
        acceleration,
        *limits,
        *bases,
        12.0,
        10.0,
        1e-6,
        *workspaces,
        output,
    )
    weights = [
        3.0 * (velocity.abs() / limits[0]).clamp_min(1e-6).pow(4) / limits[0].square(),
        1.5 * (acceleration.abs() / limits[1]).clamp_min(1e-6) / limits[1].square(),
        torch.ones_like(velocity) / limits[2].square(),
    ]
    trace = (
        sum(
            (weight * basis.square().sum(-1)[None, :, None]).sum(dim=(1, 2))
            for weight, basis in zip(weights, bases)
        )
        / dof
    )
    torch.testing.assert_close(output, 120.0 / trace, rtol=2e-5, atol=1e-8)


def _make_fitter(
    use_cuda_graph: bool,
    trajectory_dt: float = 1.0,
    acceleration_regularization: float = 5.0e-1,
    n_knots: int = 16,
    num_lanes: int = 4,
) -> BSplineTimeOptimalSeedFitter:
    control_space = ControlSpace.BSPLINE_3
    interpolation_steps = 4
    horizon = ControlSpace.spline_total_interpolation_steps(
        control_space,
        n_knots,
        interpolation_steps,
    )
    config = BSplineTimeOptimalSeedFitterCfg(
        n_knots=n_knots,
        interpolation_steps=interpolation_steps,
        control_space=control_space,
        rollout_horizon=horizon,
        num_lanes=num_lanes,
        dof=7,
        device_cfg=DeviceCfg(device=torch.device("cuda", 0)),
        use_cuda_graph=use_cuda_graph,
        trajectory_dt=trajectory_dt,
        acceleration_regularization=acceleration_regularization,
    )
    return BSplineTimeOptimalSeedFitter(
        config,
        max_velocity=torch.full((7,), 2.175, device="cuda"),
        max_acceleration=torch.full((7,), 15.0, device="cuda"),
        max_jerk=torch.full((7,), 500.0, device="cuda"),
    )


@pytest.mark.parametrize("steps", [17, 81, 301])
def test_resampling_handles_repeated_and_stationary_paths(steps: int) -> None:
    """Preserve endpoints and zero-length segments across partial and multi-pass blocks."""
    from curobo._src.curobolib.backends import trajectory_seed_fitter as backend

    torch.manual_seed(7)
    path = torch.randn((4, steps, 7), device="cuda").cumsum(1)
    path[0].fill_(0.25)
    path[1, :4] = path[1, 0].clone()
    path[2, -4:] = path[2, -1].clone()
    progress = torch.linspace(0.0, 1.0, steps, device="cuda")
    actual = torch.empty_like(path)
    backend.launch_resample_dense_kernel(path, progress, actual)
    lengths = (path[:, 1:] - path[:, :-1]).square().sum(-1).sqrt()
    cumulative = torch.cat([torch.zeros((4, 1), device="cuda"), lengths.cumsum(1)], 1)
    query = (progress[None] * cumulative[:, -1:]).contiguous()
    segment = torch.searchsorted(cumulative[:, 1:].contiguous(), query).clamp_max(steps - 2)
    start = cumulative.gather(1, segment)
    length = lengths.gather(1, segment)
    alpha = torch.where(length > 1e-8, (query - start) / length.clamp_min(1e-8), 0).clamp(0, 1)
    index = segment[:, :, None].expand(-1, -1, 7)
    left, right = path.gather(1, index), path.gather(1, index + 1)
    expected = left + alpha[:, :, None] * (right - left)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-4)


def _make_paths(
    fitter: BSplineTimeOptimalSeedFitter,
) -> tuple[torch.Tensor, JointState, torch.Tensor]:
    cfg = fitter.config
    start = torch.linspace(-0.5, 0.5, cfg.dof, device="cuda").repeat(cfg.num_lanes, 1)
    goal = torch.flip(start, dims=[-1])
    goal = goal + torch.arange(cfg.num_lanes, device="cuda").unsqueeze(-1) * 0.05
    progress = torch.linspace(0.0, 1.0, cfg.rollout_horizon, device="cuda")
    progress = progress.square().view(1, -1, 1)
    target = start.unsqueeze(1) * (1.0 - progress) + goal.unsqueeze(1) * progress
    bend = torch.linspace(-0.3, 0.3, cfg.dof, device="cuda").view(1, 1, -1)
    target = target + torch.sin(torch.pi * progress) * bend
    target[:, 0].copy_(start)
    target[:, -1].copy_(goal)
    return target.contiguous(), JointState.from_position(start.contiguous()), goal.contiguous()


@pytest.mark.parametrize("n_knots", [16, 128])
def test_fit_tracks_route_reduces_direct_seed_dt_and_pins_goal(n_knots: int) -> None:
    """Fit the graph route while lowering the timing score of direct controls."""
    fitter = _make_fitter(use_cuda_graph=False, n_knots=n_knots)
    target, start_state, goal = _make_paths(fitter)

    direct_knots = F.interpolate(
        target.transpose(1, 2),
        size=fitter.config.n_knots,
        mode="linear",
        align_corners=True,
    ).transpose(1, 2)
    direct_knots[:, -1].copy_(goal)
    direct_dt = fitter.evaluate_required_dt(direct_knots, start_state, goal)

    fitted_knots = fitter.fit(target, start_state, goal).clone()
    fitted_dt = fitter.last_required_dt.clone()
    fitted_path = fitter.evaluate_position(fitted_knots, start_state, goal)
    path_rmse = torch.square(fitted_path - fitter._arc_target).mean().sqrt()

    torch.testing.assert_close(fitted_knots[:, -1], goal, rtol=0.0, atol=0.0)
    assert torch.all(fitted_dt < direct_dt)
    assert torch.all(torch.isfinite(fitted_dt))
    assert path_rmse < 0.03


def test_arc_length_target_matches_spline_endpoint_padding() -> None:
    """Hold endpoints only across the spline rollout's flat intervals."""
    fitter = _make_fitter(use_cuda_graph=False)
    target, start_state, goal = _make_paths(fitter)
    fitter.fit(target, start_state, goal)
    arc_target = fitter._arc_target
    steps = fitter.config.interpolation_steps

    torch.testing.assert_close(
        arc_target[:, : steps + 1],
        start_state.position.unsqueeze(1).expand(-1, steps + 1, -1),
    )
    torch.testing.assert_close(
        arc_target[:, -(steps + 1) :],
        goal.unsqueeze(1).expand(-1, steps + 1, -1),
    )
    active = arc_target[:, steps:-steps]
    segment_length = torch.linalg.vector_norm(active[:, 1:] - active[:, :-1], dim=-1)
    relative_spread = segment_length.std(dim=1) / segment_length.mean(dim=1)
    assert torch.max(relative_spread) < 0.08


@pytest.mark.parametrize("use_cuda_graph", [False, True])
@pytest.mark.parametrize("n_knots", [16, 64])
def test_batched_fits_match_independent_paths(use_cuda_graph: bool, n_knots: int) -> None:
    """Batch composition and ordering must not change a path's fitted controls."""
    batched = _make_fitter(use_cuda_graph=use_cuda_graph, n_knots=n_knots)
    single = _make_fitter(use_cuda_graph=use_cuda_graph, n_knots=n_knots, num_lanes=1)
    target, start, goal = _make_paths(batched)
    start = JointState(
        position=start.position,
        velocity=torch.arange(4, device="cuda")[:, None].expand(-1, 7).contiguous() * 0.02,
        acceleration=torch.full_like(start.position, 0.01),
        jerk=torch.zeros_like(start.position),
    )
    actual = batched.fit(target, start, goal).clone()
    actual_dt = batched.last_required_dt.clone()
    for lane in range(4):
        lane_state = JointState(
            **{
                key: getattr(start, key)[lane : lane + 1]
                for key in ("position", "velocity", "acceleration", "jerk")
            }
        )
        expected = single.fit(target[lane : lane + 1], lane_state, goal[lane : lane + 1])
        torch.testing.assert_close(actual[lane : lane + 1], expected, rtol=0, atol=0)
        torch.testing.assert_close(
            actual_dt[lane : lane + 1], single.last_required_dt, rtol=0, atol=0
        )
    order = torch.tensor([2, 0, 3, 1], device="cuda")
    reordered_state = JointState(
        **{
            key: getattr(start, key)[order]
            for key in ("position", "velocity", "acceleration", "jerk")
        }
    )
    reordered = batched.fit(target[order], reordered_state, goal[order])
    torch.testing.assert_close(reordered, actual[order], rtol=0, atol=0)
    torch.testing.assert_close(batched.last_required_dt, actual_dt[order], rtol=0, atol=0)


@pytest.mark.parametrize("n_knots", [16, 128])
def test_cuda_graph_replays_with_fixed_output_storage(n_knots: int) -> None:
    """Capture all IRLS iterations into one reusable CUDA graph."""
    fitter = _make_fitter(use_cuda_graph=True, n_knots=n_knots)
    target, start_state, goal = _make_paths(fitter)
    fitter.warmup()

    graph = fitter._executor._graph
    first = fitter.fit(target, start_state, goal).clone()
    output_pointer = fitter.fit(target, start_state, goal).data_ptr()
    second_target = target.clone()
    second_target[:, :, 2].add_(0.03 * torch.sin(torch.pi * fitter._arc_progress))
    second_target[:, 0].copy_(start_state.position)
    second_target[:, -1].copy_(goal)
    second = fitter.fit(second_target.contiguous(), start_state, goal)

    assert fitter._executor._graph is graph
    assert second.data_ptr() == output_pointer
    assert torch.max(torch.abs(first - second)) > 1.0e-4
    assert torch.all(torch.isfinite(fitter.last_required_dt))

    fitter.reset_cuda_graph()
    assert not fitter.is_cuda_graph_initialized


@pytest.mark.parametrize("use_rrt", [False, True])
def test_complete_preparation_graph_updates_inputs(use_rrt: bool) -> None:
    """Replay changing goals, moving boundaries, paths and fallback masks correctly."""
    from curobo._src.motion.motion_planner import _prepare_seed_trajectories

    captured = _make_fitter(use_cuda_graph=True)
    reference = _make_fitter(use_cuda_graph=False)
    paths, state, goal = _make_paths(captured)
    graph = None
    for iteration in range(3):
        start = JointState(
            position=state.position[:1].clone() + iteration * 0.02,
            velocity=torch.full_like(state.position[:1], iteration * 0.03),
            acceleration=torch.full_like(state.position[:1], iteration * 0.01),
            jerk=torch.zeros_like(state.position[:1]),
        )
        goals = (goal + iteration * 0.04).reshape(1, 4, 7)
        waypoints = paths.clone() + iteration * 0.02 if use_rrt else None
        if waypoints is not None:
            waypoints[:, 0] = start.position
            waypoints[:, -1] = goals.reshape(4, 7)
        mask = torch.tensor([[True, iteration == 1, False, iteration != 1]], device="cuda")
        actual = _prepare_seed_trajectories(start, goals, waypoints, mask, 16, captured).clone()
        expected = _prepare_seed_trajectories(start, goals, waypoints, mask, 16, reference)
        torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
        executor = next(iter(captured._preparation_executors.values()))
        if graph is None:
            graph = executor._graph
        assert executor._graph is graph
    assert len(captured._preparation_executors) == 1
    captured.reset_cuda_graph()
    assert not captured._preparation_executors
    replayed = _prepare_seed_trajectories(start, goals, waypoints, mask, 16, captured)
    torch.testing.assert_close(replayed, expected, atol=2e-5, rtol=2e-5)
    # The same four lanes can represent two queries with two seeds each.
    # This also exercises switching moving boundaries to omitted derivatives.
    batched_start = JointState.from_position(start.position.repeat(2, 1))
    batched_goals = goals.reshape(2, 2, 7)
    batched_mask = mask.reshape(2, 2)
    actual = _prepare_seed_trajectories(
        batched_start, batched_goals, waypoints, batched_mask, 16, captured
    ).clone()
    expected = _prepare_seed_trajectories(
        batched_start, batched_goals, waypoints, batched_mask, 16, reference
    )
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    assert len(captured._preparation_executors) == 2


def test_fit_uses_boundary_derivatives_at_configured_dt() -> None:
    """Fit moving-state boundary conditions at the execution knot interval."""
    fitter = _make_fitter(use_cuda_graph=False, trajectory_dt=0.025)
    target, start_state, goal = _make_paths(fitter)
    stationary_knots = fitter.fit(target, start_state, goal).clone()

    velocity = torch.linspace(-0.4, 0.4, fitter.config.dof, device="cuda").repeat(
        fitter.config.num_lanes, 1
    )
    acceleration = torch.linspace(2.0, -2.0, fitter.config.dof, device="cuda").repeat(
        fitter.config.num_lanes, 1
    )
    moving_state = JointState(
        position=start_state.position,
        velocity=velocity.contiguous(),
        acceleration=acceleration.contiguous(),
        jerk=torch.zeros_like(velocity),
        joint_names=start_state.joint_names,
    )
    moving_knots = fitter.fit(target, moving_state, goal).clone()

    fitter._rollout(
        moving_knots,
        moving_state.position,
        moving_state.velocity,
        moving_state.acceleration,
        moving_state.jerk,
        goal,
        fitter._candidate_position,
        fitter._candidate_velocity,
        fitter._candidate_acceleration,
        fitter._candidate_jerk,
        fitter._candidate_rollout_dt,
    )

    assert torch.max(torch.abs(moving_knots - stationary_knots)) > 1.0e-4
    torch.testing.assert_close(
        fitter._candidate_velocity[:, 0], moving_state.velocity, rtol=1.0e-5, atol=1.0e-5
    )
    torch.testing.assert_close(
        fitter._candidate_acceleration[:, 0],
        moving_state.acceleration,
        rtol=1.0e-5,
        atol=1.0e-5,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("acceleration_regularization", -1.0),
        ("irls_iterations", 0),
        ("timing_regularization", -1.0),
        ("trust_region_regularization", -1.0),
        ("irls_epsilon", 0.0),
        ("dt_epsilon", -1.0),
    ],
)
def test_configuration_validation(field, value) -> None:
    """Reject invalid optimizer settings."""
    control_space = ControlSpace.BSPLINE_3
    kwargs = {
        "n_knots": 16,
        "interpolation_steps": 4,
        "control_space": control_space,
        "rollout_horizon": ControlSpace.spline_total_interpolation_steps(control_space, 16, 4),
        "num_lanes": 4,
        "dof": 7,
        "device_cfg": DeviceCfg(device=torch.device("cuda", 0)),
        field: value,
    }
    with pytest.raises(ValueError, match=field):
        BSplineTimeOptimalSeedFitterCfg(**kwargs)


def test_acceleration_regularization_defaults_are_consistent() -> None:
    """Keep the smooth geometric initialization default at its measured value."""
    field = BSplineTimeOptimalSeedFitterCfg.__dataclass_fields__["acceleration_regularization"]
    factory_default = (
        inspect.signature(BSplineTimeOptimalSeedFitter.from_transition_model)
        .parameters["acceleration_regularization"]
        .default
    )

    assert field.default == pytest.approx(0.5)
    assert factory_default == pytest.approx(0.5)


def test_irls_iteration_defaults_are_consistent() -> None:
    """Keep the Python and exported native fitter at two IRLS updates."""
    field = BSplineTimeOptimalSeedFitterCfg.__dataclass_fields__["irls_iterations"]
    factory_default = (
        inspect.signature(BSplineTimeOptimalSeedFitter.from_transition_model)
        .parameters["irls_iterations"]
        .default
    )

    assert field.default == 2
    assert factory_default == 2
