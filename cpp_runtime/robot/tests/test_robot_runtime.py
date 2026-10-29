# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the artifact-backed robot C++ runtime."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROBOT_ROOT = Path(__file__).resolve().parents[1]


def test_initial_and_refinement_timestep_scales_are_independent() -> None:
    """Initial seed retiming must not inherit the refinement multiplier."""
    source = (ROBOT_ROOT / "src/trajectory_optimizer_runtime.cpp").read_text()
    initial = source.split("void TrajectoryOptimizerRuntime::begin_optimization(", 1)[1]
    initial = initial.split("void TrajectoryOptimizerRuntime::run_optimization_pass(", 1)[0]
    assert "config_.initial_dt_scale" in initial
    assert "finetune_dt_scale" not in initial
    refine = source.split("void TrajectoryOptimizerRuntime::prepare_finetune_pass(", 1)[1]
    assert "compute_required_dt(seed_actions_.ptr(), finetune_dt_scale" in refine


@pytest.mark.parametrize("backtracking", [False, True])
def test_search_preparation_specialization(backtracking: bool) -> None:
    """Fixed-fan preparation ignores beta; MPC applies it per problem."""
    import warp as wp

    from cpp_runtime.robot.kernels.lbfgs import create_lbfgs_runtime_kernels

    kernels = create_lbfgs_runtime_kernels(7, 3, "test_backtracking")
    action = torch.ones((2, 7), device="cuda")
    direction = torch.full_like(action, 2.0)
    scales = torch.tensor([0.1, 0.5, 1.0], device="cuda")
    beta = (
        torch.tensor([0.25, 0.5], device="cuda")
        if backtracking
        else torch.full((2,), float("nan"), device="cuda")
    )
    result = torch.empty((6, 7), device="cuda")
    kernel = kernels[
        "prepare_search_actions_backtracking" if backtracking else "prepare_search_actions"
    ]
    wp.launch(
        kernel,
        dim=(6, 7),
        inputs=[wp.from_torch(value) for value in (action, direction, scales, beta, result)],
        stream=wp.stream_from_torch(torch.cuda.current_stream()),
    )
    expected_scales = scales.unsqueeze(0).expand(2, -1)
    if backtracking:
        expected_scales = expected_scales * beta.unsqueeze(1)
    expected = 1.0 + 2.0 * expected_scales.reshape(6, 1).expand(-1, 7)
    torch.testing.assert_close(result, expected)


class RobotInfo(ctypes.Structure):
    """Shape metadata returned by the robot runtime."""

    _fields_ = [
        ("num_links", ctypes.c_int),
        ("num_dof", ctypes.c_int),
        ("num_tool_frames", ctypes.c_int),
    ]


class KinematicsRequest(ctypes.Structure):
    """Device buffers passed to the fused kinematics call."""

    _fields_ = [
        ("joint_positions", ctypes.c_uint64),
        ("link_positions", ctypes.c_uint64),
        ("link_quaternions", ctypes.c_uint64),
        ("link_jacobians", ctypes.c_uint64),
        ("stream", ctypes.c_uint64),
        ("batch_size", ctypes.c_int),
        ("horizon", ctypes.c_int),
    ]


def _load_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    library.curobo_robot_last_error.restype = ctypes.c_char_p
    library.curobo_robot_create.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.curobo_robot_create.restype = ctypes.c_int
    library.curobo_robot_destroy.argtypes = [ctypes.c_void_p]
    library.curobo_robot_get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(RobotInfo)]
    library.curobo_robot_get_info.restype = ctypes.c_int
    library.curobo_robot_joint_name.argtypes = [ctypes.c_void_p, ctypes.c_int]
    library.curobo_robot_joint_name.restype = ctypes.c_char_p
    library.curobo_robot_tool_frame_name.argtypes = [ctypes.c_void_p, ctypes.c_int]
    library.curobo_robot_tool_frame_name.restype = ctypes.c_char_p
    library.curobo_robot_compute_kinematics.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(KinematicsRequest),
    ]
    library.curobo_robot_compute_kinematics.restype = ctypes.c_int
    return library


def _check_status(library: ctypes.CDLL, status: int) -> None:
    if status != 0:
        raise RuntimeError(library.curobo_robot_last_error().decode())


def test_runtime_reuses_curobo_kernel_sources() -> None:
    """Keep compute kernels shared or exported; allow runtime-only result packing."""
    native_kernel_sources = list(ROBOT_ROOT.rglob("*.cu")) + list(ROBOT_ROOT.rglob("*.cuh"))
    assert {source.relative_to(ROBOT_ROOT) for source in native_kernel_sources} == {
        Path("src/motion_planner_cuda.cu"),
        Path("src/mpc_cuda.cu"),
        Path("src/trajectory_seed_fitter_cuda.cu"),
        Path("src/trajectory_timeopt_cuda.cu"),
        Path("tests/mpc_continuation_test.cu"),
    }
    shared_seed_fitter = (
        ROBOT_ROOT.parents[1]
        / "curobo/_src/curobolib/kernels/trajectory/seed_fitter/seed_fitter_kernel.cuh"
    )
    assert shared_seed_fitter.exists()
    assert (
        "seed_fitter_kernel.cuh" in (ROBOT_ROOT / "src/trajectory_seed_fitter_cuda.cu").read_text()
    )

    exporter = (ROBOT_ROOT / "tools" / "export_robot_aot.py").read_text()
    assert "KinematicsKernelCfg" in exporter
    assert "kinematics_forward_spheres_jacobian_kernel" in exporter
    assert "LevenbergMarquardtStep" in exporter
    assert "kernel_lbfgs_step_circular" in exporter
    assert "kinematics_link_wrench_backward_kernel" in exporter
    runtime_kernel_dir = ROBOT_ROOT / "kernels"
    shared_rrt_kernels = (
        ROBOT_ROOT.parents[1] / "curobo/_src/graph_planner/rrt/kernel_rrt_runtime.py"
    )
    ik_kernels = (runtime_kernel_dir / "ik.py").read_text()
    assert "create_lbfgs_runtime_kernels" in ik_kernels
    assert "get_tiled_collision_kernel" in exporter
    assert "create_seed_ik_kernels" in exporter
    assert "bspline_time_scaled_dt_kernel" in exporter
    assert "get_tiled_swept_collision_kernel" in exporter
    assert "TrajectoryKernelCfg" in exporter
    assert "OptimizationKernelCfg" in exporter
    assert "create_trajopt_runtime_kernels" in exporter
    assert "_make_tree_kernels" in exporter
    assert "create_rrt_runtime_kernels" in exporter
    assert shared_rrt_kernels.exists()
    assert "graph_planner.rrt.kernel_rrt_runtime" in exporter
    assert "BSplineTimeOptimalSeedFitter" in exporter
    assert "create_motion_runtime_kernels" in exporter
    assert "DynamicsKernelCfg" in exporter
    assert "rnea_forward_kernel" in exporter
    assert "rnea_backward_kernel" in exporter
    assert {path.name for path in runtime_kernel_dir.glob("*.py")} == {
        "__init__.py",
        "ik.py",
        "lbfgs.py",
        "motion.py",
        "trajopt.py",
    }
    motion_kernels = (runtime_kernel_dir / "motion.py").read_text()
    assert "seed_fitter_resample_dense" not in motion_kernels
    assert "seed_fitter_initialize" not in motion_kernels
    assert "seed_fitter_accept" not in motion_kernels
    for old_path in (
        "curobo/_src/solver/seed_ik/kernel_runtime.py",
        "curobo/_src/solver/trajopt/kernel_runtime.py",
        "curobo/_src/graph_planner/rrt/kernel_runtime.py",
        "curobo/_src/motion/kernel_runtime.py",
        "curobo/_src/optim/gradient/kernel_runtime.py",
    ):
        assert not (ROBOT_ROOT.parents[1] / old_path).exists()


def test_attachment_export_adds_fixed_collision_storage(tmp_path: Path) -> None:
    """Attachment export reserves one fixed link and its collision spheres."""
    from cpp_runtime.robot.tools.export_robot_aot import prepare_attachment_robot_config
    from curobo._src.util.config_io import load_yaml, write_yaml

    source_path = tmp_path / "source.yml"
    write_yaml(
        {
            "kinematics": {
                "collision_link_names": ["base", "gripper"],
                "tool_frames": ["gripper"],
            }
        },
        str(source_path),
    )

    output_path = prepare_attachment_robot_config(
        source_path,
        tmp_path / "artifact",
        parent_link="gripper",
        sphere_capacity=32,
    )
    kinematics = load_yaml(str(output_path))["kinematics"]

    assert kinematics["collision_link_names"] == ["base", "gripper", "attached_object"]
    assert kinematics["extra_collision_spheres"] == {"attached_object": 32}
    assert kinematics["extra_links"]["attached_object"] == {
        "fixed_transform": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "joint_name": "attached_object_fixed_joint",
        "joint_type": "FIXED",
        "link_name": "attached_object",
        "parent_link_name": "gripper",
    }


def test_rrt_edges_use_block_parallel_steering_and_reduction() -> None:
    """Keep shared steering parallel and preserve Python's sample ownership."""
    kernels = (
        ROBOT_ROOT.parents[1] / "curobo/_src/graph_planner/rrt/kernel_rrt_runtime.py"
    ).read_text()
    runtime = (ROBOT_ROOT / "src" / "rrt_runtime.cpp").read_text()

    assert kernels.count("edge_idx, lane = wp.tid()") == 2
    assert "wp.tile_max(wp.tile(local_weighted_delta))" in kernels
    assert "wp.tile_min(wp.tile(local_infeasible))" in kernels
    assert "wp.tile_sum(wp.tile(local_desired_distance_squared))" in kernels
    assert runtime.count("warp_tiled_bounds_2d(edges") == 2
    assert "desired[edge_idx, joint_idx] = samples[edge_idx, joint_idx]" in kernels
    assert "prepare_samples(next_samples(plans), plans, stream)" in runtime
    assert "sample_offset" not in runtime
    assert "run_shortcut_stage(plans, stream)" in runtime
    assert "launch_finalization_graph(plans, stream)" in runtime
    assert "refresh_scene_graphs()" in runtime
    assert "scene_->obstacle_count(ObstacleType::CUBOID) > 0" in runtime
    assert "scene_->obstacle_count(ObstacleType::CAPSULE) > 0" in runtime
    assert "scene_->obstacle_count(ObstacleType::SPHERE) > 0" in runtime
    assert "scene_->obstacle_count(ObstacleType::CYLINDER) > 0" in runtime
    assert "scene_->obstacle_count(ObstacleType::MESH) > 0" in runtime
    assert "scene_->obstacle_count(ObstacleType::VOXEL_GRID) > 0" in runtime
    assert "for (int first_edge" not in runtime
    assert "shortcut_path(" not in runtime


def test_exported_ik_seeds_use_python_seed_solver_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scale native Halton seeds with Python SeedIK's fractional joint margin."""
    monkeypatch.syspath_prepend(str(ROBOT_ROOT / "tools"))
    from export_robot_aot import build_ik_tensors

    from curobo._src.solver.seed_ik.seed_ik_solver_cfg import SeedIKSolverCfg

    position = torch.tensor(
        [[-3.0, -1.0], [3.0, 2.0]],
        dtype=torch.float32,
    )
    params = SimpleNamespace(
        num_dof=2,
        joint_limits=SimpleNamespace(
            position=position,
            effort=torch.zeros((2, 2), dtype=torch.float32),
            velocity=torch.zeros((2, 2), dtype=torch.float32),
        ),
        cspace=SimpleNamespace(
            default_joint_position=torch.zeros(2, dtype=torch.float32),
            cspace_distance_weight=torch.ones(2, dtype=torch.float32),
        ),
        link_spheres=torch.zeros((1, 1, 4), dtype=torch.float32),
        sphere_collision_mask=torch.ones((1, 1), dtype=torch.uint32),
        link_sphere_idx_map=torch.zeros(1, dtype=torch.int16),
    )
    optimizer_profile = {
        "rollout": {
            "cost_cfg": {
                "cspace_cfg": {
                    "weight": [1.0],
                    "activation_distance": 0.0,
                    "squared_l2_regularization_weight": [0.0, 0.0],
                }
            },
            "constraint_cfg": {
                "scene_collision_cfg": {"weight": 1.0, "activation_distance": 0.01},
                "self_collision_cfg": {"weight": 1.0},
            },
        }
    }
    optimizer = SimpleNamespace(step_scale=0.1, line_search_scale=[1.0])
    tensors = build_ik_tensors(
        SimpleNamespace(self_collision_config=None),
        params,
        max_seeds=4,
        optimizer_profile=optimizer_profile,
        optimizer=optimizer,
    )

    margin = (position[1] - position[0]) * SeedIKSolverCfg.__dataclass_fields__[
        "joint_limit_margin"
    ].default
    torch.testing.assert_close(tensors["joint_position_lower"], position[0] + margin)
    torch.testing.assert_close(tensors["joint_position_upper"], position[1] - margin)
    assert tensors["seed_positions"].shape == (4, 2)
    torch.testing.assert_close(
        tensors["ik_start_cspace_weight"], torch.tensor([0.001], dtype=torch.float32)
    )
    torch.testing.assert_close(
        tensors["ik_start_cspace_dof_weight"], torch.ones(2, dtype=torch.float32)
    )


def test_exporter_expands_joint_position_limits_symmetrically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply benchmark limit expansion before deriving native solver tensors."""
    monkeypatch.syspath_prepend(str(ROBOT_ROOT / "tools"))
    from export_robot_aot import expand_joint_position_limits

    original = torch.tensor(
        [[-3.0, -1.0], [3.0, 2.0]],
        dtype=torch.float32,
    )
    params = SimpleNamespace(
        joint_limits=SimpleNamespace(position=original.clone()),
    )

    expand_joint_position_limits(params, 0.2)

    torch.testing.assert_close(
        params.joint_limits.position,
        torch.tensor([[-3.2, -1.2], [3.2, 2.2]], dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="finite and non-negative"):
        expand_joint_position_limits(params, -0.1)


def test_exported_trajopt_collision_weights_match_python_rollouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Export optimizer, sparse-metrics, and dense-metrics weights separately."""
    monkeypatch.syspath_prepend(str(ROBOT_ROOT / "tools"))
    from export_robot_aot import build_trajopt_tensors

    limits = SimpleNamespace(
        position=torch.tensor([[-1.0, -2.0], [1.0, 2.0]], dtype=torch.float32),
        velocity=torch.ones((2, 2), dtype=torch.float32),
        acceleration=torch.ones((2, 2), dtype=torch.float32),
        jerk=torch.ones((2, 2), dtype=torch.float32),
        effort=torch.ones((2, 2), dtype=torch.float32),
    )
    cspace = SimpleNamespace(
        weight=torch.ones(5, dtype=torch.float32),
        activation_distance=torch.zeros(5, dtype=torch.float32),
        squared_l2_regularization_weight=torch.zeros(5, dtype=torch.float32),
        cspace_target_weight=torch.ones(1, dtype=torch.float32),
        cspace_non_terminal_weight_factor=torch.ones(1, dtype=torch.float32),
        cspace_target_dof_weight=torch.ones(2, dtype=torch.float32),
    )
    rollout = SimpleNamespace(
        cost_cfg=SimpleNamespace(
            cspace_cfg=cspace,
            tool_pose_cfg=SimpleNamespace(weight=torch.ones(2, dtype=torch.float32)),
        ),
        constraint_cfg=SimpleNamespace(
            scene_collision_cfg=SimpleNamespace(weight=torch.tensor([100.0], dtype=torch.float32)),
            self_collision_cfg=SimpleNamespace(weight=torch.tensor([8.0], dtype=torch.float32)),
        ),
        transition_model_cfg=SimpleNamespace(n_knots=6, interpolation_steps=4),
    )
    metrics_rollout = SimpleNamespace(
        constraint_cfg=SimpleNamespace(
            scene_collision_cfg=SimpleNamespace(weight=torch.tensor([20.0], dtype=torch.float32)),
            self_collision_cfg=SimpleNamespace(weight=torch.tensor([16.0], dtype=torch.float32)),
        ),
        transition_model_cfg=SimpleNamespace(interpolation_steps=4),
    )
    tensors = build_trajopt_tensors(
        SimpleNamespace(num_dof=2, joint_limits=limits),
        SimpleNamespace(metrics_rollout_config=metrics_rollout),
        rollout,
        SimpleNamespace(step_scale=0.1, line_search_scale=[1.0]),
    )

    torch.testing.assert_close(
        tensors["world_collision_weight"], torch.tensor([100.0], dtype=torch.float32)
    )
    torch.testing.assert_close(
        tensors["self_collision_weight"], torch.tensor([2.0], dtype=torch.float32)
    )
    torch.testing.assert_close(
        tensors["metrics_world_collision_weight"],
        torch.tensor([20.0], dtype=torch.float32),
    )
    torch.testing.assert_close(
        tensors["metrics_self_collision_weight"],
        torch.tensor([4.0], dtype=torch.float32),
    )
    torch.testing.assert_close(
        tensors["validation_self_collision_weight"],
        torch.tensor([0.015625], dtype=torch.float32),
    )


def test_self_collision_export_matches_python_max_pair_gradient(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Use Python's max-pair self-collision reduction and emit its gradient."""
    monkeypatch.syspath_prepend(str(ROBOT_ROOT / "tools"))
    import export_robot_aot

    requested_symbols: list[str] = []

    def compile_specialization(
        output_dir: Path,
        arch: str,
        artifact_prefix: str,
        kernel_config: object,
        kernel_names: list[str],
        symbols: list[str],
    ) -> tuple[str, dict[str, str]]:
        del output_dir, arch, artifact_prefix, kernel_config, kernel_names
        requested_symbols.extend(symbols)
        return "self_collision.cubin", {symbols[0]: "lowered_self_collision"}

    monkeypatch.setattr(
        export_robot_aot,
        "_compile_cuda_specializations",
        compile_specialization,
    )
    config = SimpleNamespace(
        collision_pairs=torch.zeros((3, 2), dtype=torch.int16),
        max_threads_per_block=256,
    )
    metadata = export_robot_aot.compile_self_collision_kernel(
        tmp_path,
        "sm_120",
        config,
        num_spheres=4,
    )

    assert metadata is not None
    assert requested_symbols == [
        "curobo::geometry::self_collision::self_collision_exact_distance_kernel"
        "<0, false, true, 4, 3, 32>"
    ]
    assert metadata["shared_memory"] == 4 * 4 * 4


def test_public_api_is_backend_neutral() -> None:
    """Keep implementation-specific types out of the public ABI."""
    header = (ROBOT_ROOT / "include" / "curobo_robot_runtime" / "api.h").read_text()
    assert "CuroboRobotKinematicsRequest" in header
    assert "curobo_robot_compute_kinematics" in header
    cpp_header = (ROBOT_ROOT / "include" / "curobo_robot_runtime" / "kinematics.h").read_text()
    scene_header = (ROBOT_ROOT / "include" / "curobo_robot_runtime" / "scene.h").read_text()
    ik_header = (
        ROBOT_ROOT / "include" / "curobo_robot_runtime" / "inverse_kinematics.h"
    ).read_text()
    trajopt_header = (
        ROBOT_ROOT / "include" / "curobo_robot_runtime" / "trajectory_optimization.h"
    ).read_text()
    rrt_header = (ROBOT_ROOT / "include" / "curobo_robot_runtime" / "rrt.h").read_text()
    motion_header = (
        ROBOT_ROOT / "include" / "curobo_robot_runtime" / "motion_planner.h"
    ).read_text()
    seed_fitter_header = (
        ROBOT_ROOT / "include" / "curobo_robot_runtime" / "trajectory_seed_fitter.h"
    ).read_text()
    attachment_header = (
        ROBOT_ROOT / "include" / "curobo_robot_runtime" / "attachment.h"
    ).read_text()
    sphere_fit_header = (
        ROBOT_ROOT / "include" / "curobo_robot_runtime" / "sphere_fit.h"
    ).read_text()
    inertial_header = (ROBOT_ROOT / "include" / "curobo_robot_runtime" / "inertial.h").read_text()
    assert "class Kinematics final" in cpp_header
    assert "void forward" in cpp_header
    assert "class Scene final" in scene_header
    assert "num_envs" not in scene_header
    assert "class InverseKinematics final" in ik_header
    assert "lm_iterations" in ik_header
    assert "lbfgs_iterations" in ik_header
    assert "L-BFGS" in ik_header
    assert "IkSolveRequest" in ik_header
    assert "max_goalset" in ik_header
    assert "goalset_index" in ik_header
    assert "current_positions" in ik_header
    assert "num_solutions" in ik_header
    assert "seed_offset" in ik_header
    assert "use_cuda_graph" in ik_header
    assert "check_effort_limits" in ik_header
    assert "num_envs" not in ik_header
    assert "class TrajectoryOptimizer final" in trajopt_header
    assert "TrajectoryOptimizationRequest" in trajopt_header
    assert "finetune_dt_scale" in trajopt_header
    assert "initial_dt_scale" in trajopt_header
    assert "seed_start_positions" in trajopt_header
    assert "start_velocity" in trajopt_header
    assert "start_acceleration" in trajopt_header
    assert "seed_goal_positions" in trajopt_header
    assert "seed_valid" in trajopt_header
    assert "check_effort_limits" in trajopt_header
    assert "use_cuda_streams" not in trajopt_header
    assert "num_envs" not in trajopt_header
    assert "class RrtPlanner final" in rrt_header
    assert "RrtRequest" in rrt_header
    assert "num_plans" in rrt_header
    assert "plan_valid" in rrt_header
    assert "reset_seed" in rrt_header
    assert "check_effort_limits" in rrt_header
    assert "num_envs" not in rrt_header
    assert "class MotionPlanner final" in motion_header
    assert "MotionPlanningRequest" in motion_header
    assert "seed_start_positions" in motion_header
    assert "start_velocity" in motion_header
    assert "start_acceleration" in motion_header
    assert "current_velocity" in motion_header
    assert "current_acceleration" in motion_header
    assert "seed_goal_positions" in motion_header
    assert "selected_seed" in motion_header
    assert "attempt_direct_trajectory_optimization" in motion_header
    assert "PoseMotionPlanningRequest" in motion_header
    assert "plan_pose" in motion_header
    assert "max_attempts" in motion_header
    assert "enable_graph_attempt" in motion_header
    assert "finetune_attempts" in motion_header
    assert "use_cuda_graph" in motion_header
    assert "use_rrt_for_constrained_phases" not in motion_header
    assert "num_envs" not in motion_header
    assert "class TrajectorySeedFitter final" in seed_fitter_header
    assert "TrajectorySeedFitterRequest" in seed_fitter_header
    assert "start_velocity" in seed_fitter_header
    assert "required_dt" in seed_fitter_header
    assert "use_cuda_graph" in seed_fitter_header
    assert "num_envs" not in seed_fitter_header
    assert "struct SphereAttachment" in attachment_header
    assert "struct LinkInertialUpdate" in inertial_header
    assert "std::optional<float> mass" in inertial_header
    assert "std::optional<std::array<float, 3>> center_of_mass" in inertial_header
    assert "std::optional<std::array<float, 6>> inertia" in inertial_header
    assert "fit_spheres_to_mesh" in sphere_fit_header
    for solver_header in (ik_header, trajopt_header, rrt_header, motion_header):
        assert "attach_object" in solver_header
        assert "detach_object" in solver_header
        assert "update_link_inertial" in solver_header
        assert "update_links_inertial" in solver_header
    for backend_name in ("torch", "warp"):
        assert backend_name not in header
        assert backend_name not in cpp_header
        assert backend_name not in scene_header
        assert backend_name not in ik_header
        assert backend_name not in trajopt_header
        assert backend_name not in rrt_header
        assert backend_name not in motion_header
        assert backend_name not in seed_fitter_header
        assert backend_name not in attachment_header
        assert backend_name not in inertial_header
        assert backend_name not in sphere_fit_header


def test_inverse_dynamics_is_shared_by_native_rollouts() -> None:
    """Keep the RNEA model and launch code out of individual solver runtimes."""
    shared_header = (ROBOT_ROOT / "src" / "rollout_runtime.h").read_text()
    shared_source = (ROBOT_ROOT / "src" / "rollout_runtime.cpp").read_text()
    trajopt_header = (ROBOT_ROOT / "src" / "trajectory_optimizer_runtime.h").read_text()
    trajopt_source = (ROBOT_ROOT / "src" / "trajectory_optimizer_runtime.cpp").read_text()
    ik_header = (ROBOT_ROOT / "src" / "ik_runtime.h").read_text()
    rrt_header = (ROBOT_ROOT / "src" / "rrt_runtime.h").read_text()

    assert "class RolloutRuntime final" in shared_header
    assert "launch_inverse_dynamics" in shared_header
    assert "launch_inverse_dynamics_backward" in shared_header
    assert "update_link_inertial" in shared_header
    assert "external_force_owners" in shared_source
    assert "external_force_transforms" in shared_source
    ik_source = (ROBOT_ROOT / "src" / "ik_runtime.cpp").read_text()
    assert "std::shared_ptr<IkModelParameters> model_" in ik_header
    assert "std::unique_ptr<RolloutRuntime> rollout_" in ik_source
    for solver_header in (trajopt_header, rrt_header):
        assert "std::unique_ptr<RolloutRuntime> rollout_" in solver_header
    assert "launch_rnea_forward" not in trajopt_header
    assert "launch_rnea_backward" not in trajopt_header
    assert "TrajectoryOptimizerRuntime::launch_rnea_forward" not in trajopt_source
    assert "TrajectoryOptimizerRuntime::launch_rnea_backward" not in trajopt_source

    ik_kernels = (ROBOT_ROOT / "kernels" / "ik.py").read_text()
    rrt_kernels = (
        ROBOT_ROOT.parents[1] / "curobo/_src/graph_planner/rrt/kernel_rrt_runtime.py"
    ).read_text()
    trajopt_kernels = (ROBOT_ROOT / "kernels" / "trajopt.py").read_text()
    assert "effort_position_gradient" in ik_kernels
    assert "effort_active" in rrt_kernels
    assert "effort_violation" in rrt_kernels
    assert "effort_active" in trajopt_kernels
    assert "validation_joint_effort_" in trajopt_source


def test_native_rollouts_always_use_parallel_cuda_streams() -> None:
    """Keep Python-style dynamics, kinematics, and cost stream fan-out internal."""
    streams_header = (ROBOT_ROOT / "src" / "rollout_streams.h").read_text()
    streams_source = (ROBOT_ROOT / "src" / "rollout_streams.cpp").read_text()
    solver_sources = [
        (ROBOT_ROOT / "src" / name).read_text()
        for name in (
            "ik_runtime.cpp",
            "rrt_runtime.cpp",
            "trajectory_optimizer_runtime.cpp",
        )
    ]

    for stream_name in ("DYNAMICS", "KINEMATICS", "CSPACE", "POSE", "WORLD", "SELF"):
        assert stream_name in streams_header
    assert "cudaStreamCreateWithFlags" in streams_source
    assert "cudaStreamNonBlocking" in streams_source
    for source in solver_sources:
        assert "std::make_unique<RolloutStreams>()" in source
        assert "RolloutEvent::STATE_READY" in source
        for stream_name in ("DYNAMICS", "KINEMATICS", "WORLD", "SELF"):
            assert f"RolloutStream::{stream_name}" in source
    for source in (solver_sources[0], solver_sources[2]):
        assert "RolloutStream::CSPACE" in source
        assert "RolloutStream::POSE" in source


def test_rollout_artifact_exports_mutable_payload_model() -> None:
    """Export original inertia plus fixed-link ownership for payload updates."""
    exporter = (ROBOT_ROOT / "tools" / "export_robot_aot.py").read_text()

    assert '"robot_rollout_manifest.json"' in exporter
    assert '"rnea_original_link_masses_com"' in exporter
    assert '"rnea_original_link_inertias"' in exporter
    assert '"rnea_external_force_owners"' in exporter
    assert '"rnea_external_force_transforms"' in exporter
    assert '"ik"' in exporter
    assert '"rrt"' in exporter
    assert '"trajopt"' in exporter
    assert "_make_forward_cspace_position_kernel(True)" in exporter


def test_exported_runtime_matches_python_kinematics() -> None:
    """Compare C++ FK and world-frame Jacobians with the Python implementation."""
    artifact_value = os.environ.get("CUROBO_ROBOT_TEST_ARTIFACT_DIR")
    if artifact_value is None:
        pytest.skip("set CUROBO_ROBOT_TEST_ARTIFACT_DIR to an exported robot artifact")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    artifact_dir = Path(artifact_value).resolve()
    manifest = json.loads((artifact_dir / "robot_kinematics_manifest.json").read_text())
    library_path = Path(
        os.environ.get(
            "CUROBO_ROBOT_TEST_LIBRARY",
            ROBOT_ROOT.parent / "build" / "robot" / "libcurobo_robot_runtime.so",
        )
    ).resolve()
    library = _load_library(library_path)

    runtime = ctypes.c_void_p()
    _check_status(
        library,
        library.curobo_robot_create(str(artifact_dir).encode(), 0, ctypes.byref(runtime)),
    )
    try:
        info = RobotInfo()
        _check_status(library, library.curobo_robot_get_info(runtime, ctypes.byref(info)))
        assert info.num_links == manifest["num_links"]
        assert info.num_dof == manifest["num_dof"]
        assert info.num_tool_frames == manifest["num_tool_frames"]
        assert [
            library.curobo_robot_joint_name(runtime, index).decode()
            for index in range(info.num_dof)
        ] == manifest["joint_names"]
        assert [
            library.curobo_robot_tool_frame_name(runtime, index).decode()
            for index in range(info.num_tool_frames)
        ] == manifest["tool_frames"]

        batch_size = 2
        horizon = 3
        joint_positions = torch.linspace(
            -0.4,
            0.4,
            batch_size * horizon * info.num_dof,
            device="cuda",
            dtype=torch.float32,
        ).reshape(batch_size, horizon, info.num_dof)
        link_positions = torch.empty(
            (batch_size, horizon, info.num_tool_frames, 3),
            device="cuda",
            dtype=torch.float32,
        )
        link_quaternions = torch.empty(
            (batch_size, horizon, info.num_tool_frames, 4),
            device="cuda",
            dtype=torch.float32,
        )
        link_jacobians = torch.empty(
            (batch_size, horizon, info.num_tool_frames, 6, info.num_dof),
            device="cuda",
            dtype=torch.float32,
        )
        stream = torch.cuda.current_stream()
        request = KinematicsRequest(
            joint_positions=joint_positions.data_ptr(),
            link_positions=link_positions.data_ptr(),
            link_quaternions=link_quaternions.data_ptr(),
            link_jacobians=link_jacobians.data_ptr(),
            stream=stream.cuda_stream,
            batch_size=batch_size,
            horizon=horizon,
        )
        _check_status(
            library,
            library.curobo_robot_compute_kinematics(runtime, ctypes.byref(request)),
        )

        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.types import JointState

        source = manifest["source"]
        if source["kind"] == "robot_config":
            python_config = KinematicsCfg.from_robot_yaml_file(
                source["robot_config"],
                urdf_path=source.get("urdf"),
                tool_frames=manifest["tool_frames"],
            )
        else:
            python_config = KinematicsCfg.from_basic_urdf(
                urdf_path=source["urdf"],
                base_link=source["base_link"],
                tool_frames=manifest["tool_frames"],
            )
        kinematics = Kinematics(
            python_config,
            compute_jacobian=True,
            compute_spheres=False,
        )
        expected = kinematics.compute_kinematics(
            JointState.from_position(joint_positions, joint_names=kinematics.joint_names)
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(link_positions, expected.tool_poses.position)
        torch.testing.assert_close(link_quaternions, expected.tool_poses.quaternion)
        torch.testing.assert_close(link_jacobians, expected.tool_jacobians)

        graph_positions = torch.empty_like(link_positions)
        graph_quaternions = torch.empty_like(link_quaternions)
        graph_jacobians = torch.empty_like(link_jacobians)
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            request.link_positions = graph_positions.data_ptr()
            request.link_quaternions = graph_quaternions.data_ptr()
            request.link_jacobians = graph_jacobians.data_ptr()
            request.stream = torch.cuda.current_stream().cuda_stream
            _check_status(
                library,
                library.curobo_robot_compute_kinematics(runtime, ctypes.byref(request)),
            )
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_positions, expected.tool_poses.position)
        torch.testing.assert_close(graph_quaternions, expected.tool_poses.quaternion)
        torch.testing.assert_close(graph_jacobians, expected.tool_jacobians)
    finally:
        library.curobo_robot_destroy(runtime)


def test_exported_inverse_kinematics_runtime() -> None:
    """Exercise pose convergence, CUDA Graph replay, and Scene rejection in C++."""
    artifact_value = os.environ.get("CUROBO_ROBOT_TEST_ARTIFACT_DIR")
    if artifact_value is None:
        pytest.skip("set CUROBO_ROBOT_TEST_ARTIFACT_DIR to an exported robot artifact")
    executable = Path(
        os.environ.get(
            "CUROBO_ROBOT_IK_TEST_EXECUTABLE",
            ROBOT_ROOT.parent / "build" / "robot" / "curobo_robot_ik_test",
        )
    ).resolve()
    if not executable.exists():
        pytest.skip(f"build the native IK test first: {executable}")
    subprocess.run(
        [str(executable), str(Path(artifact_value).resolve())],
        check=True,
    )


def test_exported_trajectory_optimization_runtime() -> None:
    """Exercise multi-seed dt-refined TrajOpt and CUDA Graph replay in C++."""
    artifact_value = os.environ.get("CUROBO_ROBOT_TEST_ARTIFACT_DIR")
    if artifact_value is None:
        pytest.skip("set CUROBO_ROBOT_TEST_ARTIFACT_DIR to an exported robot artifact")
    executable = Path(
        os.environ.get(
            "CUROBO_ROBOT_TRAJOPT_TEST_EXECUTABLE",
            ROBOT_ROOT.parent / "build" / "robot" / "curobo_robot_trajopt_test",
        )
    ).resolve()
    if not executable.exists():
        pytest.skip(f"build the native TrajOpt test first: {executable}")
    subprocess.run(
        [str(executable), str(Path(artifact_value).resolve())],
        check=True,
    )


@pytest.mark.parametrize("solver", ["IK", "TRAJOPT"])
def test_exported_solver_parallelism(solver: str) -> None:
    """Compare parallel selection across eager execution and graph replay."""
    artifact_value = os.environ.get("CUROBO_ROBOT_TEST_ARTIFACT_DIR")
    executable_value = os.environ.get(f"CUROBO_ROBOT_{solver}_PARALLELISM_TEST_EXECUTABLE")
    if artifact_value is None or executable_value is None:
        pytest.skip("set robot artifact and native solver parallelism test executable")
    subprocess.run([executable_value, artifact_value], check=True)


@pytest.mark.parametrize(
    ("executable_environment", "executable_name"),
    [
        ("CUROBO_ROBOT_IK_TEST_EXECUTABLE", "curobo_robot_ik_test"),
        ("CUROBO_ROBOT_RRT_TEST_EXECUTABLE", "curobo_robot_rrt_test"),
        ("CUROBO_ROBOT_TRAJOPT_TEST_EXECUTABLE", "curobo_robot_trajopt_test"),
    ],
)
def test_rnea_effort_limit_rejection(
    tmp_path: Path,
    executable_environment: str,
    executable_name: str,
) -> None:
    """Reject IK, RRT, and TrajOpt results with impossible torque bounds."""
    artifact_value = os.environ.get("CUROBO_ROBOT_TEST_ARTIFACT_DIR")
    if artifact_value is None:
        pytest.skip("set CUROBO_ROBOT_TEST_ARTIFACT_DIR to an exported robot artifact")
    executable = Path(
        os.environ.get(
            executable_environment,
            ROBOT_ROOT.parent / "build" / "robot" / executable_name,
        )
    ).resolve()
    if not executable.exists():
        pytest.skip(f"build the native test first: {executable}")
    artifact = tmp_path / "zero_effort_artifact"
    shutil.copytree(Path(artifact_value).resolve(), artifact)
    manifest = json.loads((artifact / "robot_rollout_manifest.json").read_text())
    effort_path = artifact / manifest["tensors"]["effort_bounds"]["file"]
    effort_path.write_bytes(np.zeros(14, dtype=np.float32).tobytes())

    result = subprocess.run(
        [str(executable), str(artifact)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


def test_exported_rrt_runtime() -> None:
    """Exercise direct and obstacle-avoiding bidirectional RRT in C++."""
    artifact_value = os.environ.get("CUROBO_ROBOT_TEST_ARTIFACT_DIR")
    if artifact_value is None:
        pytest.skip("set CUROBO_ROBOT_TEST_ARTIFACT_DIR to an exported robot artifact")
    executable = Path(
        os.environ.get(
            "CUROBO_ROBOT_RRT_TEST_EXECUTABLE",
            ROBOT_ROOT.parent / "build" / "robot" / "curobo_robot_rrt_test",
        )
    ).resolve()
    if not executable.exists():
        pytest.skip(f"build the native RRT test first: {executable}")
    artifact = Path(artifact_value).resolve()
    manifest = json.loads((artifact / "robot_rrt_manifest.json").read_text())
    assert manifest["schema_version"] == 8
    assert manifest["direct_steer_steps"] > manifest["fixed_steer_steps"]
    assert manifest["shortcut_path_capacity"] == 16
    for kernel in (
        "prepare_direct_edges",
        "interpolate_direct_edges",
        "reduce_direct_edges",
        "shortcut_setup",
        "shortcut_fill_edges",
        "shortcut_reduce_edges",
        "shortcut_select_path",
    ):
        assert kernel in manifest["kernels"]
    sample_tensor = manifest["tensors"]["samples"]
    samples = np.fromfile(artifact / sample_tensor["file"], dtype=np.float32).reshape(
        sample_tensor["shape"]
    )
    lane_rows = manifest["samples_per_iteration"]
    assert not np.array_equal(samples[:lane_rows], samples[lane_rows : 2 * lane_rows])
    subprocess.run(
        [str(executable), str(artifact)],
        check=True,
    )


def test_exported_trajectory_seed_fitter_runtime(tmp_path: Path) -> None:
    """Compare native IRLS seed fitting with the production Python fitter."""
    artifact_value = os.environ.get("CUROBO_ROBOT_TEST_ARTIFACT_DIR")
    if artifact_value is None:
        pytest.skip("set CUROBO_ROBOT_TEST_ARTIFACT_DIR to an exported robot artifact")
    executable = Path(
        os.environ.get(
            "CUROBO_ROBOT_SEED_FITTER_TEST_EXECUTABLE",
            ROBOT_ROOT.parent / "build" / "robot" / "curobo_robot_seed_fitter_test",
        )
    ).resolve()
    if not executable.exists():
        pytest.skip(f"build the native seed-fitter test first: {executable}")
    output_path = tmp_path / "seed_fitter_output.f32"
    subprocess.run(
        [str(executable), str(Path(artifact_value).resolve()), str(output_path)],
        check=True,
    )

    # The focused native executable emits its exact inputs before its outputs,
    # avoiding host-side transcendental differences in this parity check.
    seeds, dof, steps, knots = 4, 7, 81, 16
    values = np.fromfile(output_path, dtype=np.float32)
    offset = 0
    target_count = seeds * steps * dof
    target = values[offset : offset + target_count].reshape(seeds, steps, dof)
    offset += target_count
    state_count = seeds * dof
    start = values[offset : offset + state_count].reshape(seeds, dof)
    offset += state_count
    goal = values[offset : offset + state_count].reshape(seeds, dof)
    offset += state_count
    knot_count = seeds * knots * dof
    native_knots = values[offset : offset + knot_count].reshape(seeds, knots, dof)
    offset += knot_count
    native_dt = values[offset : offset + seeds]
    offset += seeds
    start_velocity = values[offset : offset + state_count].reshape(seeds, dof)
    offset += state_count
    start_acceleration = values[offset : offset + state_count].reshape(seeds, dof)
    offset += state_count
    start_jerk = values[offset : offset + state_count].reshape(seeds, dof)
    offset += state_count
    native_moving_knots = values[offset : offset + knot_count].reshape(seeds, knots, dof)
    offset += knot_count
    native_moving_dt = values[offset : offset + seeds]
    offset += seeds
    assert offset == values.size

    from cpp_runtime.robot.tools.export_robot_aot import build_trajopt_config
    from curobo._src.state.state_joint import JointState
    from curobo._src.trajectory.trajectory_seed_fitter import (
        BSplineTimeOptimalSeedFitter,
    )
    from curobo._src.transition.robot_state_transition import RobotStateTransition

    robot_config = (
        ROBOT_ROOT.parents[1] / "curobo" / "content" / "configs" / "robot" / "franka.yml"
    )
    _, rollout, _ = build_trajopt_config(robot_config, seeds)
    python_fitter = BSplineTimeOptimalSeedFitter.from_transition_model(
        RobotStateTransition(rollout.transition_model_cfg),
        num_lanes=seeds,
        use_cuda_graph=False,
    )
    target_tensor = torch.as_tensor(target, device="cuda")
    start_tensor = torch.as_tensor(start, device="cuda")
    goal_tensor = torch.as_tensor(goal, device="cuda")
    python_knots = python_fitter.fit(
        target_tensor,
        JointState.from_position(start_tensor),
        goal_tensor,
    ).clone()
    python_dt = python_fitter.last_required_dt.clone()
    torch.testing.assert_close(
        torch.as_tensor(native_knots, device="cuda"),
        python_knots,
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    torch.testing.assert_close(
        torch.as_tensor(native_dt, device="cuda"),
        python_dt,
        rtol=1.0e-5,
        atol=1.0e-7,
    )

    moving_state = JointState(
        position=start_tensor,
        velocity=torch.as_tensor(start_velocity, device="cuda"),
        acceleration=torch.as_tensor(start_acceleration, device="cuda"),
        jerk=torch.as_tensor(start_jerk, device="cuda"),
    )
    python_moving_knots = python_fitter.fit(
        target_tensor,
        moving_state,
        goal_tensor,
    ).clone()
    python_moving_dt = python_fitter.last_required_dt.clone()
    torch.testing.assert_close(
        torch.as_tensor(native_moving_knots, device="cuda"),
        python_moving_knots,
        rtol=1.0e-5,
        atol=1.0e-5,
    )
    torch.testing.assert_close(
        torch.as_tensor(native_moving_dt, device="cuda"),
        python_moving_dt,
        rtol=1.0e-5,
        atol=1.0e-7,
    )


def test_exported_motion_planner_runtime() -> None:
    """Exercise direct TrajOpt and RRT-seeded motion planning in C++."""
    artifact_value = os.environ.get("CUROBO_ROBOT_TEST_ARTIFACT_DIR")
    if artifact_value is None:
        pytest.skip("set CUROBO_ROBOT_TEST_ARTIFACT_DIR to an exported robot artifact")
    executable = Path(
        os.environ.get(
            "CUROBO_ROBOT_MOTION_TEST_EXECUTABLE",
            ROBOT_ROOT.parent / "build" / "robot" / "curobo_robot_motion_test",
        )
    ).resolve()
    if not executable.exists():
        pytest.skip(f"build the native motion-planner test first: {executable}")
    subprocess.run(
        [str(executable), str(Path(artifact_value).resolve())],
        check=True,
    )


def test_exported_collision_layers() -> None:
    """Check shared geometry and layer updates against captured native queries."""
    artifact = os.environ.get("CUROBO_ROBOT_TEST_ARTIFACT_DIR")
    executable = os.environ.get("CUROBO_ROBOT_COLLISION_LAYERS_TEST_EXECUTABLE")
    if artifact is None or executable is None:
        pytest.skip("set robot artifact and native collision-layer test executable")
    subprocess.run([executable, artifact], check=True)
