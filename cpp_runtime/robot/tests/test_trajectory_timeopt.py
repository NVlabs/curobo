# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native timeopt parity, graph replay, moving boundaries, and rejected inputs."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from cpp_runtime.robot.tools.export_timeopt import export_timeopt_tables
from curobo._src.motion.motion_trajectory_timeopt import TrajectoryTimeOpt
from curobo._src.motion.motion_trajectory_timeopt_cfg import TrajectoryTimeOptCfg
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.state.state_joint import JointState


@pytest.mark.parametrize(
    "knots,reference,intervals",
    [(16, True, 511), (16, True, 512), (32, False, 511), (64, False, 511)],
)
@pytest.mark.parametrize("allow_tf32", [False, True])
def test_native_timeopt_parity(
    tmp_path: Path,
    knots: int,
    reference: bool,
    intervals: int,
    allow_tf32: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare eager, default-stream, internal and outer graphs against Python."""
    artifact = os.environ.get("CUROBO_ROBOT_TEST_ARTIFACT_DIR")
    executable = os.environ.get("CUROBO_ROBOT_TIMEOPT_TEST_EXECUTABLE")
    if artifact is None or executable is None or not torch.cuda.is_available():
        pytest.skip("set native timeopt executable and artifact directory on a CUDA host")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", allow_tf32)
    directory = Path(artifact)
    manifest = json.loads((directory / "robot_kinematics_manifest.json").read_text())
    source = manifest["source"]
    if source["kind"] != "robot_config":
        pytest.skip("parity fixture requires the exported robot configuration")
    robot = KinematicsCfg.from_robot_yaml_file(source["robot_config"])
    trajopt = json.loads((directory / "robot_trajopt_manifest.json").read_text())
    rollout = json.loads((directory / "robot_rollout_manifest.json").read_text())
    dof = manifest["num_dof"]
    if intervals != 511:
        source_directory = directory
        directory = tmp_path / "artifact"
        directory.mkdir()
        prefix = f"timeopt_{knots}_{int(reference)}"
        for path in source_directory.iterdir():
            if not path.name.startswith(prefix):
                (directory / path.name).symlink_to(path.resolve())
        export_timeopt_tables(directory, dof, knots, reference, intervals)

    def read(metadata: dict, name: str) -> torch.Tensor:
        tensor = metadata["tensors"][name]
        return torch.as_tensor(np.fromfile(directory / tensor["file"], np.float32), device="cuda")

    # Artifact exports may contain configured COM/inertia overrides. Use those
    # exact physical inputs instead of reapplying current YAML preprocessing.
    model = robot.kinematics_config
    model.link_masses_com.copy_(
        read(rollout, "original_link_masses_com").reshape_as(model.link_masses_com)
    )
    model.link_inertias.copy_(
        read(rollout, "original_link_inertias").reshape_as(model.link_inertias)
    )

    bounds = read(trajopt, "position_bounds").reshape(2, dof)
    limits = [
        read(trajopt, name + "_bounds").reshape(2, dof)
        for name in ("velocity", "acceleration", "jerk")
    ]
    limits = [torch.minimum(-x[0], x[1]).contiguous() for x in limits]
    effort = read(rollout, "effort_bounds").reshape(2, dof)
    config = TrajectoryTimeOptCfg(
        robot,
        batch_size=4,
        n_knots=knots,
        num_intervals=intervals,
        velocity_limits=limits[0],
        acceleration_limits=limits[1],
        jerk_limits=limits[2],
        torque_lower=effort[0].contiguous(),
        torque_upper=effort[1].contiguous(),
        gravity=tuple((-read(rollout, "gravity")[-3:]).cpu().tolist()),
    )
    timeopt = TrajectoryTimeOpt(config)
    start = bounds.mean(dim=0).repeat(4, 1).contiguous()
    goal = start + 0.15 * torch.cos(torch.arange(dof, device="cuda", dtype=torch.float32))[None]
    progress = torch.linspace(0, 1, knots, device="cuda")[None, :, None]
    controls = (start[:, None] + progress * (goal - start)[:, None]).contiguous()
    velocity, acceleration, jerk = [torch.zeros_like(start) for _ in range(3)]
    dt = torch.full((4,), 0.15, device="cuda", dtype=torch.float32)
    if reference:
        velocity[0, 0] = 0.01
        acceleration[0, 0] = -0.01
    controls[2] = start[2]
    goal[2] = start[2]
    controls[3, 2, 0] = torch.nan
    state = JointState(position=start, velocity=velocity, acceleration=acceleration, jerk=jerk)
    result = timeopt.timeopt(
        controls, state if reference else start, goal, knot_dt=dt if reference else None
    )
    times = (torch.arange(1000, device="cuda", dtype=torch.float32) * 0.0025)[None].repeat(4, 1)
    times[:, 0] = -0.1
    times[:, -1] = torch.nan
    samples = timeopt.sample(times)
    for name, tensor in dict(
        knots=controls,
        start=start,
        goal=goal,
        dt=dt,
        velocity=velocity,
        acceleration=acceleration,
        jerk=jerk,
        times=times,
    ).items():
        tensor.cpu().numpy().tofile(tmp_path / (name + ".bin"))
    expected = dict(
        timing=result.timing_controls,
        time=result.time,
        duration=result.duration,
        uniform=result.uniform_duration,
        success=result.success,
        status=result.status,
        states=timeopt._joint_output,
        torque=result.torque,
        samples=torch.stack(
            [getattr(samples, name) for name in ("position", "velocity", "acceleration", "jerk")]
        ),
    )
    expected = {name: value.cpu().numpy() for name, value in expected.items()}
    assert expected["success"][:3].all()
    assert expected["status"][3] == 3
    subprocess.run(
        [executable, str(directory), str(tmp_path), str(knots), str(int(reference)),
         "1000", str(intervals)],
        check=True,
    )
    for mode in ("eager", "graph"):
        for name, target in expected.items():
            actual = np.fromfile(tmp_path / f"{mode}_{name}.bin", target.dtype).reshape(
                target.shape
            )
            if target.dtype.kind in "biu":
                np.testing.assert_array_equal(actual, target)
                continue
            np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(target))
            if name in ("states", "samples"):
                # Normalize derivatives by physical limits; near-zero jerk has
                # cancellation error while its absolute error remains small.
                scales = np.stack([np.ones(dof, np.float32), *[x.cpu().numpy() for x in limits]])
                np.testing.assert_allclose(
                    actual / scales[:, None, None],
                    target / scales[:, None, None],
                    rtol=2e-4,
                    atol=2e-4,
                    equal_nan=True,
                )
            else:
                np.testing.assert_allclose(actual, target, rtol=2e-4, atol=2e-4, equal_nan=True)


@pytest.mark.parametrize("schema", [1, 2])
def test_native_timeopt_rejects_legacy_tables(tmp_path: Path, schema: int) -> None:
    """Old quadrature artifacts must be re-exported before using the uniform objective."""
    artifact = os.environ.get("CUROBO_ROBOT_TEST_ARTIFACT_DIR")
    executable = os.environ.get("CUROBO_ROBOT_TIMEOPT_TEST_EXECUTABLE")
    if artifact is None or executable is None or not torch.cuda.is_available():
        pytest.skip("set native timeopt executable and artifact directory on a CUDA host")
    filename = "timeopt_16_1.json"
    source = Path(artifact)
    for path in source.iterdir():
        if path.name != filename:
            (tmp_path / path.name).symlink_to(path.resolve())
    manifest = json.loads((source / filename).read_text())
    manifest["schema_version"] = schema
    (tmp_path / filename).write_text(json.dumps(manifest))
    result = subprocess.run(
        [executable, str(tmp_path), str(tmp_path), "16", "1", "1000", "511"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "unsupported timeopt table schema; re-export timeopt tables" in result.stderr
