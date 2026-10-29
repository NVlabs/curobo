# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check benchmark denominators, unit conversion, and incomplete-output rejection."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pinocchio as pin
import pytest
import yaml


@pytest.fixture
def benchmark() -> ModuleType:
    """Load the standalone runner without building or running a GPU workload."""
    path = Path(__file__).resolve().parents[1] / "run_benchmarks.py"
    spec = importlib.util.spec_from_file_location("native_benchmarks", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failed_batches_remain_in_denominator_and_timing(benchmark: ModuleType) -> None:
    """A batch with zero successes contributes time and failures, but no pose error."""
    rows = [
        {"batch": 0, "time_s": 0.02, "success": 0},
        {"batch": 0, "time_s": 0.02, "success": 0},
        {
            "batch": 1,
            "time_s": 0.01,
            "success": 1,
            "position_error_m": 0.002,
            "orientation_error_rad": 0.01,
        },
        {"batch": 1, "time_s": 0.01, "success": 0},
    ]
    result = benchmark.aggregate_ik(rows)
    assert result["success_percent"] == 25.0
    assert result["batch_time_ms"] == pytest.approx(15.0)
    assert result["position_error_p90_mm"] == pytest.approx(2.0)
    assert result["orientation_error_p90_deg"] == pytest.approx(0.5729578)


def test_no_success_has_unavailable_errors(benchmark: ModuleType) -> None:
    """Unsolved inputs must not produce fabricated accuracy numbers."""
    result = benchmark.aggregate_ik([{"batch": 0, "time_s": 0.01, "success": 0}])
    assert result["success_percent"] == 0
    assert result["position_error_p90_mm"] is None
    assert result["orientation_error_p90_deg"] is None
    assert benchmark.statistics([]) is None


def test_incomplete_native_output_is_rejected(benchmark: ModuleType, tmp_path: Path) -> None:
    """Interrupted runs cannot silently become apparently complete measurements."""
    path = tmp_path / "results.jsonl"
    path.write_text('{"index":0,"success":1}\n')
    with pytest.raises(ValueError, match="expected 2 records, got 1"):
        benchmark.read_records(path, 2)


def test_nonfinite_statistics_are_rejected(benchmark: ModuleType) -> None:
    """Invalid native numbers must fail scoring and strict JSON serialization."""
    with pytest.raises(ValueError, match="nonfinite"):
        benchmark.statistics([1.0, float("nan")])


def test_tables_preserve_zero_success_and_unavailable_metrics(benchmark: ModuleType) -> None:
    """Table rendering must distinguish unmeasured accuracy from a zero error."""
    result = benchmark.aggregate_ik([{"batch": 0, "time_s": 0.02, "success": 0}])
    rendered = benchmark.render_tables({"motion": {}, "ik": {"franka_pose_only": result}})
    assert "20" in rendered
    assert "Unavailable" in rendered
    assert "franka.yml" in rendered
    assert "C-Free-IK-time(ms)" in rendered


def test_trajectory_metrics_use_absolute_power(benchmark: ModuleType) -> None:
    """A 2 kg horizontal slider gives analytical torque, absolute work, and path length."""
    model = pin.Model()
    joint = model.addJoint(0, pin.JointModelPX(), pin.SE3.Identity(), "slider")
    model.appendBodyToJoint(joint, pin.Inertia.FromSphere(2.0, 0.1), pin.SE3.Identity())
    states = np.array(
        [
            [[0.0], [1.0], [2.0]],
            [[1.0], [-1.0], [1.0]],
            [[0.5], [0.5], [0.5]],
            [[4.0], [-3.0], [2.0]],
        ],
        dtype=np.float32,
    )
    metrics = benchmark.trajectory_metrics(states, 0.1, model, model.createData())
    assert metrics["path_length_rad"] == pytest.approx(2.0)
    assert metrics["torque_nm"] == pytest.approx(1.0)
    assert metrics["energy_j"] == pytest.approx(0.3)
    assert metrics["jerk_rad_s3"] == pytest.approx(4.0)


def test_samples_map_by_joint_name_and_fill_unused_joints(benchmark: ModuleType) -> None:
    """Pose-only G1 sampling prunes joints still present in the native collision model."""
    samples = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    defaults = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    aligned = benchmark.align_joint_samples(samples, ["c", "a"], ["a", "b", "c"], defaults)
    np.testing.assert_allclose(
        aligned, np.array([[2.0, 0.2, 1.0], [4.0, 0.2, 3.0]], dtype=np.float32)
    )
    assert aligned.dtype == np.float32


def test_motion_capacity_uses_both_eligible_corpora(
    benchmark: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both motion artifacts share the eligible maximum; excluded scenes cannot inflate it."""
    import robometrics.file_structure

    def problem(count: int, eligible: bool = True) -> dict:
        """Return a small raw problem whose obstacle list is already in OBB form."""
        return {
            "collision_buffer_ik": 0 if eligible else -1,
            "start": [0.0] * 7,
            "goal_pose": {"position_xyz": [0.0] * 3, "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
            "obstacles": [{"pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], "dims": [1.0, 1.0, 1.0]}]
            * count,
        }

    (tmp_path / "mb_set.yaml").write_text(yaml.safe_dump({"mb": [problem(2), problem(9, False)]}))
    (tmp_path / "mpinets_set.yaml").write_text(yaml.safe_dump({"mpi": [problem(5)]}))
    monkeypatch.setattr(robometrics.file_structure, "get_dataset_path", lambda: tmp_path)
    monkeypatch.setattr(benchmark, "scene_boxes", lambda obstacles: obstacles)
    capacities = []
    monkeypatch.setattr(
        benchmark,
        "export_artifact",
        lambda *args, **kwargs: capacities.append(kwargs["motion_max_cuboids"]),
    )
    datasets = benchmark.prepare_motion(tmp_path, 0, "source")
    assert capacities == [5, 5]
    assert all(dataset["max_cuboids"] == 5 for dataset in datasets.values())
    assert datasets["motion_benchmaker"]["excluded"] == 1


def test_motion_export_options_invalidate_cached_artifacts(
    benchmark: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Motion kernels use only the exact cuboid capacity and 1,000 validation samples."""
    config = tmp_path / "robot.yml"
    config.write_text("robot: test\n")
    artifact = tmp_path / "artifact"
    commands = []
    monkeypatch.setattr(benchmark, "run", lambda command, log=None: commands.append(command))
    benchmark.export_artifact(config, artifact, "source", motion_max_cuboids=41)
    options = json.loads((artifact / "benchmark_export.json").read_text())["export_options"]
    assert options == {
        "max-cuboids": 41,
        "max-capsules": 0,
        "max-cylinders": 0,
        "max-meshes": 0,
        "max-voxel-grids": 0,
        "interpolation-buffer-size": 1000,
    }
    for name, value in options.items():
        assert commands[0][commands[0].index(f"--{name}") + 1] == str(value)
    benchmark.export_artifact(config, artifact, "source", motion_max_cuboids=41)
    assert len(commands) == 1
    benchmark.export_artifact(config, artifact, "source", motion_max_cuboids=42)
    assert len(commands) == 2


def test_native_ik_uses_29dof_g1_profile(
    benchmark: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 29-DoF G1 selection retains native humanoid tuning and accurate labels."""
    import sys

    import torch

    output = tmp_path / "run"
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    robot = benchmark.ROBOTS[0]
    case = {
        "robot": robot,
        "name": f"{robot}_pose_only",
        "collision_free": False,
        "artifact": str(artifact),
        "input": str(tmp_path / "input.txt"),
    }
    commands = []
    monkeypatch.setattr(sys, "argv", ["run_benchmarks.py", "--suite", "ik", "--output", str(output)])
    monkeypatch.setattr(benchmark, "run", lambda command, log=None: commands.append(command) or "")
    monkeypatch.setattr(benchmark, "source_provenance", lambda: {"runtime_source_sha256": "test"})
    monkeypatch.setattr(benchmark, "prepare_ik", lambda *args: [case])
    monkeypatch.setattr(benchmark, "sha256", lambda path: "test")
    monkeypatch.setattr(
        benchmark, "read_records", lambda *args: [{"batch": 0, "time_s": 0.01, "success": 0}]
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 0))

    benchmark.main()

    native_calls = [command for command in commands if len(command) > 1 and command[1] == "ik"]
    assert len(native_calls) == 1
    assert native_calls[0][-2:] == ["0", "1"]
    assert "unitree_g1_29dof_retarget.yml" in (output / "tables.rst").read_text()
