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


def test_target_device_requires_prepare_only(
    benchmark: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A host must not execute kernels prepared for a different device."""
    import sys

    monkeypatch.setattr(sys, "argv", [
        "run_benchmarks.py", "--output", str(tmp_path), "--target-device", "thor.json"
    ])
    monkeypatch.setattr(benchmark, "source_provenance", lambda: pytest.fail("already preparing"))
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 2


def test_export_cache_uses_device_content(
    benchmark: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing limits at the same target path invalidates the exported kernels."""
    from cpp_runtime.robot.tools.export_device import ExportDevice

    config = tmp_path / "robot.yml"
    config.write_text("robot_cfg: {}\n")
    target_path = tmp_path / "target.json"
    target = ExportDevice("Thor", "sm_110", 20, 232448).to_dict()
    target_path.write_text(json.dumps(target))
    commands = []
    monkeypatch.setattr(benchmark, "run", lambda command, log=None: commands.append(command))
    artifact = tmp_path / "artifact"
    benchmark.export_artifact(config, artifact, "source", target_device=target_path)
    benchmark.export_artifact(config, artifact, "source", target_device=target_path)
    assert len(commands) == 1
    assert commands[0][commands[0].index("--target-device") + 1] == str(target_path)
    target["sm_count"] = 16
    target_path.write_text(json.dumps(target))
    benchmark.export_artifact(config, artifact, "source", target_device=target_path)
    assert len(commands) == 2
    benchmark.export_artifact(config, artifact, "source")
    assert len(commands) == 3


def test_target_preparation_skips_host_build_and_execution(
    benchmark: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Remote preparation exports inputs only, retaining the target device provenance."""
    import sys

    from cpp_runtime.robot.tools.export_device import ExportDevice

    path = tmp_path / "device.json"
    target = ExportDevice("Thor", "sm_110", 20, 232448).to_dict()
    path.write_text(json.dumps(target))
    output = tmp_path / "prepared"
    monkeypatch.setattr(sys, "argv", [
        "run_benchmarks.py", "--suite", "motion", "--prepare-only",
        "--target-device", str(path), "--output", str(output),
    ])
    commands = []
    monkeypatch.setattr(benchmark, "run", lambda command, log=None: commands.append(command))
    monkeypatch.setattr(benchmark, "source_provenance", lambda: {"runtime_source_sha256": "test"})

    def prepare(*args: object, **kwargs: object) -> dict:
        """Verify the target reaches input preparation without generating a host binary."""
        assert kwargs["target_device"] == path
        return {}

    monkeypatch.setattr(benchmark, "prepare_motion", prepare)
    benchmark.main()
    assert commands == [["nvidia-smi"]]
    assert not (output / "build").exists()
    assert json.loads((output / "provenance.json").read_text())["target_device"] == target


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
    knots = []

    def record_export(*args: object, **kwargs: int) -> None:
        """Retain the shape options sent to both dataset artifact exports."""
        capacities.append(kwargs["motion_max_cuboids"])
        knots.append(kwargs["num_knots"])

    monkeypatch.setattr(
        benchmark,
        "export_artifact",
        record_export,
    )
    datasets = benchmark.prepare_motion(tmp_path, 0, "source", num_knots=12)
    assert capacities == [5, 5]
    assert knots == [12, 12]
    assert all(dataset["num_knots"] == 12 for dataset in datasets.values())
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
    benchmark.export_artifact(config, artifact, "source", motion_max_cuboids=42, num_knots=12)
    assert len(commands) == 3
    assert commands[-1][commands[-1].index("--num-knots") + 1] == "12"
    assert json.loads((artifact / "benchmark_export.json").read_text())["num_knots"] == 12
    benchmark.export_artifact(config, artifact, "source", motion_max_cuboids=42, num_knots=12)
    assert len(commands) == 3
    benchmark.export_artifact(config, artifact, "source", motion_max_cuboids=42, num_knots=1024)
    assert len(commands) == 4
    command = commands[-1]
    capacity = benchmark.motion_interpolation_buffer_size(1024)
    assert capacity > 4113
    assert command[command.index("--interpolation-buffer-size") + 1] == str(capacity)


@pytest.mark.parametrize("disable_inverse_dynamics", [False, True])
def test_motion_runner_uses_requested_knots_for_both_payloads(
    benchmark: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    disable_inverse_dynamics: bool,
) -> None:
    """A motion shape request reaches export preparation and both native runs."""
    import sys

    import torch

    output = tmp_path / "run"
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    commands = []
    prepared = []

    def prepare(directory: Path, limit: int, source: str, num_knots: int) -> dict:
        """Supply one frozen dataset while retaining its requested shape."""
        prepared.append(num_knots)
        return {
            "test": {
                "artifact": str(artifact),
                "input": "input.txt",
                "count": 1,
                "max_cuboids": 41,
                "num_knots": num_knots,
                "interpolation_buffer_size": 1000,
            }
        }

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_benchmarks.py", "--suite", "motion", "--num-knots", "12",
            "--output", str(output),
            *(["--disable-inverse-dynamics"] if disable_inverse_dynamics else []),
        ],
    )
    monkeypatch.setattr(benchmark, "run", lambda command, log=None: commands.append(command) or "")
    monkeypatch.setattr(benchmark, "source_provenance", lambda: {"runtime_source_sha256": "test"})
    monkeypatch.setattr(benchmark, "prepare_motion", prepare)
    monkeypatch.setattr(benchmark, "aggregate_motion", lambda *args: {})
    monkeypatch.setattr(benchmark, "render_tables", lambda *args: "test")
    monkeypatch.setattr(benchmark, "sha256", lambda path: "test")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 0))
    benchmark.main()
    assert prepared == [12]
    native = [command for command in commands if len(command) > 1 and command[1] == "motion"]
    assert len(native) == 2
    assert [command[5] for command in native] == ["0", "1"]
    assert all(command[7:9] == ["41", "12"] for command in native)
    assert all(
        ("--disable-inverse-dynamics" in command) == disable_inverse_dynamics for command in native
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["motion_num_knots"] == 12
    assert summary["motion_inverse_dynamics"] is not disable_inverse_dynamics
    for filename in ("inputs.json", "provenance.json"):
        assert json.loads((output / filename).read_text())["motion_inverse_dynamics"] is (
            not disable_inverse_dynamics
        )


def test_ik_sampling_retains_partial_valid_batches() -> None:
    """Keep every accepted configuration when a rejection draw cannot fill a batch."""
    import torch

    from benchmark.ik_benchmark import sample_ik_configs

    class Sampler:
        def __init__(self) -> None:
            self.offset = 0

        def sample_configs(self, count: int, rejection_ratio: int) -> torch.Tensor:
            result = torch.arange(self.offset, self.offset + 60, dtype=torch.float32).view(30, 2)
            self.offset += 60
            return result

    result = sample_ik_configs(Sampler(), 100)
    np.testing.assert_array_equal(result, np.arange(200, dtype=np.float32).reshape(100, 2))


def test_ik_sampling_rejects_incomplete_batches() -> None:
    """Bound preparation retries instead of measuring an incomplete or unchecked batch."""
    import torch

    from benchmark.ik_benchmark import sample_ik_configs

    class Sampler:
        def sample_configs(self, count: int, rejection_ratio: int) -> torch.Tensor:
            return torch.empty((0, 2), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="collected 0"):
        sample_ik_configs(Sampler(), 100)


def test_native_ik_uses_29dof_g1_profile(
    benchmark: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The G1 case passes the Python-derived profile and retains accurate labels."""
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
        "config": str(tmp_path / "config.txt"),
    }
    commands = []
    monkeypatch.setattr(
        sys, "argv", ["run_benchmarks.py", "--suite", "ik", "--output", str(output)]
    )
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
    assert native_calls[0][-2:] == ["0", case["config"]]
    assert "unitree_g1_29dof.yml" in (output / "tables.rst").read_text()


@pytest.mark.parametrize(
    "robot,collision_free,seeds,iterations",
    [
        (robot, collision_free, seeds, iterations)
        for robot, free_seeds, iterations in [
            ("unitree_g1_29dof", 16, 160),
            ("dual_ur10e", 16, None),
            ("franka", 8, None),
        ]
        for collision_free, seeds in [(False, 2), (True, free_seeds)]
    ],
)
def test_shared_ik_benchmark_config(
    robot: str,
    collision_free: bool,
    seeds: int,
    iterations: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both benchmark frontends use the Python robot/mode seed and iteration contract."""
    from types import SimpleNamespace

    from benchmark.ik_benchmark import IKSolverCfg, create_ik_config

    monkeypatch.setattr(IKSolverCfg, "create", lambda **kwargs: SimpleNamespace(**kwargs))
    config = create_ik_config(robot + ".yml", collision_free=collision_free)
    assert config.num_seeds == seeds
    assert config.override_iters_for_multi_link_ik == iterations
    assert config.optimizer_configs == ["ik/lbfgs_ik.yml"]
    assert config.optimizer_collision_activation_distance == 0.0025
    assert config.self_collision_check == collision_free
    assert config.use_cuda_graph


@pytest.mark.parametrize("successful_batches", [0, 1])
def test_python_ik_counts_failed_batches(
    successful_batches: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed Python batches retain their timing and denominator contribution."""
    from types import SimpleNamespace

    import torch

    import benchmark.ik_benchmark as python_benchmark

    config = SimpleNamespace(exit_early=True)

    class Solver:
        def __init__(self, config: SimpleNamespace) -> None:
            self.config = config
            self.timed_calls = 0

        def reset_seed(self) -> None:
            pass

        def sample_configs(self, count: int, rejection_ratio: int) -> torch.Tensor:
            return torch.zeros((count, 2), dtype=torch.float32)

        def compute_kinematics(self, state: object) -> SimpleNamespace:
            return SimpleNamespace(tool_poses=SimpleNamespace(as_goal=lambda: None))

        def solve_pose(self, **kwargs: object) -> SimpleNamespace:
            success = self.config.exit_early and self.timed_calls < successful_batches
            self.timed_calls += int(self.config.exit_early)
            return SimpleNamespace(
                success=torch.full((2,), success, dtype=torch.bool),
                position_error=torch.full((2,), 0.002, dtype=torch.float32),
                rotation_error=torch.full((2,), 0.01, dtype=torch.float32),
            )

    class Timer:
        calls = 0

        def start(self) -> Timer:
            return self

        def stop(self) -> float:
            Timer.calls += 1
            return Timer.calls * 0.001

    monkeypatch.setattr(python_benchmark, "create_ik_config", lambda *args: config)
    monkeypatch.setattr(python_benchmark, "IKSolver", Solver)
    monkeypatch.setattr(python_benchmark, "CudaEventTimer", Timer)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    result = python_benchmark.run_full_config_collision_free_ik("franka.yml", None, 2)
    assert result[0] == pytest.approx(0.003)
    assert result[1] == successful_batches * 20.0
    if successful_batches:
        assert result[2] == pytest.approx(0.002)
    else:
        assert np.isnan(result[2]) and np.isnan(result[3])


def test_native_profiling_keeps_the_public_driver(
    benchmark: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Profiling wraps the same executable and marks diagnostic CUDA graph traces."""
    calls = []
    monkeypatch.setattr(benchmark, "run", lambda command, log: calls.append(command))
    command = ["curobo_benchmark", "motion", "artifact", "input", "output", "1", "0", "41"]
    benchmark.run_native(command, tmp_path / "motion.log", None)
    assert calls[-1] == command
    benchmark.run_native(command, tmp_path / "motion.log", "node")
    assert calls[-1][:2] == ["nsys", "profile"]
    assert "--cuda-graph-trace=node" in calls[-1]
    assert calls[-1][-len(command) - 1 :] == [*command, "--profile"]
