#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare inputs, run native IK/motion benchmarks, and aggregate their measured results."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ROBOTS = ("unitree_g1_29dof_retarget", "dual_ur10e", "franka")


def run(command: list[str], log: Path | None = None) -> str:
    """Run a checked subprocess from the checkout, optionally retaining its output."""
    if log is not None:
        with log.open("w") as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        return ""
    return subprocess.check_output(command, cwd=ROOT, text=True)


def sha256(path: Path) -> str:
    """Hash a file without loading large trajectory buffers into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path: Path, value: Any) -> None:
    """Write strict, human-readable JSON."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def source_provenance() -> dict[str, Any]:
    """Record the revision and actual runtime/exporter source content, including edits."""
    paths = run(["git", "ls-files", "curobo", "cpp_runtime/robot"]).splitlines()
    hashes = {name: sha256(ROOT / name) for name in paths if (ROOT / name).is_file()}
    return {
        "commit": run(["git", "rev-parse", "HEAD"]).strip(),
        "status": run(["git", "status", "--short"]),
        "runtime_source_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
        "source_files": hashes,
    }


def write_values(stream: TextIO, values: Any) -> None:
    """Write one row of float32-compatible native input values."""
    stream.write(
        " ".join(format(float(value), ".9g") for value in np.asarray(values).ravel()) + "\n"
    )


def write_scene(stream: TextIO, boxes: list[dict[str, Any]]) -> None:
    """Write converted OBB geometry consumed by the native Scene API."""
    stream.write(f"{len(boxes)}\n")
    for box in boxes:
        write_values(stream, box["pose"] + box["dims"])


def scene_boxes(obstacles: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply the canonical benchmark's OBB conversion to all obstacle types."""
    from copy import deepcopy

    from curobo._src.geom.types import SceneCfg

    world = SceneCfg.create(deepcopy(obstacles)).get_obb_world()
    boxes = [{"pose": list(box.pose), "dims": list(box.dims)} for box in world.cuboid]
    return boxes


def export_artifact(
    config: Path,
    artifact: Path,
    source_hash: str,
    expansion: float = 0.0,
    motion_max_cuboids: int | None = None,
) -> None:
    """Export fresh kernels or reuse an artifact with matching source and configuration hashes."""
    signature = {
        "source": source_hash,
        "config": sha256(config),
        "joint_limit_expansion": expansion,
    }
    export_options = {}
    if motion_max_cuboids is not None:
        export_options = {
            "max-cuboids": motion_max_cuboids,
            "max-capsules": 0,
            "max-cylinders": 0,
            "max-meshes": 0,
            "max-voxel-grids": 0,
            "interpolation-buffer-size": 1000,
        }
        signature["export_options"] = export_options
    stamp = artifact / "benchmark_export.json"
    if stamp.exists() and json.loads(stamp.read_text()) == signature:
        return
    artifact.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {config.stem}", flush=True)
    run(
        [
            sys.executable,
            "cpp_runtime/robot/tools/export_robot.py",
            "--robot-config",
            str(config),
            "--out",
            str(artifact),
            "--max-ik-seeds",
            "32",
            "--max-trajopt-seeds",
            "4",
            "--joint-limit-expansion",
            str(expansion),
            *[
                part
                for name, value in export_options.items()
                for part in (f"--{name}", str(value))
            ],
        ],
        artifact / "export.log",
    )
    save_json(stamp, signature)


def prepare_motion(output: Path, limit: int, source_hash: str) -> dict[str, Any]:
    """Freeze the complete corpus, exclusions, OBB inputs, and dataset-specific robot models."""
    from robometrics.file_structure import get_dataset_path

    datasets = {}
    max_cuboids = 0
    for name, filename in (("motion_benchmaker", "mb_set.yaml"), ("mpinets", "mpinets_set.yaml")):
        path = get_dataset_path() / filename
        with path.open() as stream:
            corpus = yaml.load(stream, Loader=yaml.CSafeLoader)
        records, excluded = [], []
        for scene, problems in corpus.items():
            for index, problem in enumerate(problems):
                identity = {"dataset": name, "scene": scene, "problem": index}
                if problem["collision_buffer_ik"] < 0:
                    excluded.append({**identity, "reason": "negative collision_buffer_ik"})
                    continue
                records.append(
                    {
                        **identity,
                        "start": problem["start"],
                        "goal_pose": problem["goal_pose"],
                        "boxes": scene_boxes(problem["obstacles"]),
                    }
                )
        if limit:
            records = records[:limit]
        max_cuboids = max(max_cuboids, max((len(row["boxes"]) for row in records), default=0))
        save_json(output / f"{name}_problems.json", records)
        save_json(output / f"{name}_excluded.json", excluded)
        input_path = output / f"{name}.txt"
        with input_path.open("w") as stream:
            stream.write(f"{len(records)}\n")
            for record in records:
                write_values(stream, record["start"])
                write_values(stream, record["goal_pose"]["position_xyz"])
                write_values(stream, record["goal_pose"]["quaternion_wxyz"])
                write_scene(stream, record["boxes"])
        config = yaml.safe_load((ROOT / "curobo/content/configs/robot/franka.yml").read_text())
        robot = config["robot_cfg"]
        robot["load_dynamics"] = True
        kin = robot["kinematics"]
        kin["collision_sphere_buffer"] = 0.0
        kin["collision_link_names"].remove("attached_object")
        kin["tool_frames"] = ["panda_hand"]
        if name == "mpinets":
            kin["lock_joints"] = {"panda_finger_joint1": 0.025, "panda_finger_joint2": 0.025}
        config_path = output / f"franka_{name}.yml"
        config_path.write_text(yaml.safe_dump(config))
        artifact = output / "artifacts" / name
        datasets[name] = {
            "input": str(input_path),
            "artifact": str(artifact),
            "config": str(config_path),
            "count": len(records),
            "excluded": len(excluded),
            "dataset_sha256": sha256(path),
            "input_sha256": sha256(input_path),
        }
    for dataset in datasets.values():
        dataset["max_cuboids"] = max_cuboids
        export_artifact(
            Path(dataset["config"]),
            Path(dataset["artifact"]),
            source_hash,
            expansion=0.2,
            motion_max_cuboids=max_cuboids,
        )
    return datasets


def align_joint_samples(
    samples: np.ndarray,
    sample_names: list[str],
    native_names: list[str],
    default_position: np.ndarray,
) -> np.ndarray:
    """Map sampled [B,D] joints into native order, filling omitted joints with model defaults."""
    if samples.shape[1] != len(sample_names) or default_position.shape != (len(native_names),):
        raise ValueError("joint sample/model dimensions do not match their joint names")
    indices = [native_names.index(name) for name in sample_names]
    result = np.broadcast_to(default_position, (samples.shape[0], len(native_names))).copy()
    result[:, indices] = samples
    return result


def prepare_ik(output: Path, source_hash: str) -> list[dict[str, Any]]:
    """Generate five reachable 100-query batches per robot/mode with rejection sampling."""
    import torch

    from curobo._src.solver.solver_ik import IKSolver
    from curobo._src.solver.solver_ik_cfg import IKSolverCfg

    torch.manual_seed(2)
    np.random.seed(2)
    from curobo._src.util_file import get_scene_configs_path

    scene_path = Path(get_scene_configs_path()) / "collision_table.yml"
    boxes = scene_boxes(yaml.safe_load(scene_path.read_text()))
    cases = []
    for robot in ROBOTS:
        config_path = ROOT / f"curobo/content/configs/robot/{robot}.yml"
        artifact = output / "artifacts" / robot
        export_artifact(config_path, artifact, source_hash)
        manifest = json.loads((artifact / "robot_kinematics_manifest.json").read_text())
        native_names = manifest["joint_names"]
        default_position = np.fromfile(artifact / "default_joint_position.f32", dtype=np.float32)
        for collision_free in (False, True):
            name = f"{robot}_{'collision_free' if collision_free else 'pose_only'}"
            print(f"Preparing {name}", flush=True)
            input_path = output / f"{name}.txt"
            config = IKSolverCfg.create(
                robot=str(config_path),
                scene_model="collision_table.yml" if collision_free else None,
                num_seeds=2,
                max_batch_size=100,
                use_cuda_graph=True,
                self_collision_check=collision_free,
                load_collision_spheres=collision_free,
            )
            sampler = IKSolver(config)
            with input_path.open("w") as stream:
                write_scene(stream, boxes)
                stream.write("100 5\n")
                for _ in range(5):
                    for ratio in range(10, 101, 10):
                        samples = sampler.sample_configs(100, rejection_ratio=ratio)
                        if samples.shape[0] >= 100:
                            break
                    else:
                        raise RuntimeError(f"could not sample 100 valid configurations for {name}")
                    aligned = align_joint_samples(
                        samples[:100].cpu().numpy(),
                        sampler.joint_names,
                        native_names,
                        default_position,
                    )
                    write_values(stream, aligned)
            del sampler
            torch.cuda.empty_cache()
            cases.append(
                {
                    "name": name,
                    "robot": robot,
                    "collision_free": collision_free,
                    "input": str(input_path),
                    "artifact": str(artifact),
                    "num_dof": len(native_names),
                    "num_tool_frames": manifest["num_tool_frames"],
                    "input_sha256": sha256(input_path),
                }
            )
    return cases


def read_records(path: Path, expected: int) -> list[dict[str, Any]]:
    """Reject partial native runs instead of silently reporting a reduced denominator."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != expected:
        raise ValueError(f"{path}: expected {expected} records, got {len(rows)}")
    return rows


def statistics(values: list[float]) -> dict[str, float] | None:
    """Return population statistics, preserving unavailable metrics as null."""
    if not values:
        return None
    data = np.asarray(values)
    if not np.isfinite(data).all():
        raise ValueError("nonfinite benchmark metric")
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        **{
            name: float(np.percentile(data, percentile))
            for name, percentile in (("median", 50), ("p75", 75), ("p98", 98))
        },
    }


def aggregate_ik(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Include every query in success and average the five successful-query error percentiles."""
    batches = sorted({row["batch"] for row in rows})
    position, orientation, times = [], [], []
    for batch in batches:
        group = [row for row in rows if row["batch"] == batch]
        times.append(group[0]["time_s"] * 1000.0)
        valid = [row for row in group if row["success"]]
        if valid:
            position.append(
                float(np.percentile([r["position_error_m"] * 1000 for r in valid], 90))
            )
            orientation.append(
                float(np.percentile([np.degrees(r["orientation_error_rad"]) for r in valid], 90))
            )
    return {
        "queries": len(rows),
        "successes": sum(bool(row["success"]) for row in rows),
        "success_percent": 100 * sum(bool(row["success"]) for row in rows) / len(rows),
        "batch_size": len(rows) // len(batches),
        "batch_time_ms": float(np.mean(times)),
        "position_error_p90_mm": float(np.mean(position)) if position else None,
        "orientation_error_p90_deg": float(np.mean(orientation)) if orientation else None,
    }


def trajectory_metrics(states: np.ndarray, dt: float, model: Any, data: Any) -> dict[str, float]:
    """Measure executable [q,v,a,jerk] states with shape [4,T,D] outside native timing."""
    import pinocchio as pin

    q, v, a, jerk = states
    torques = np.asarray([pin.rnea(model, data, qi, vi, ai).copy() for qi, vi, ai in zip(q, v, a)])
    return {
        "path_length_rad": float(np.linalg.norm(np.diff(q, axis=0), axis=-1).sum()),
        "jerk_rad_s3": float(np.max(np.abs(jerk))),
        "energy_j": float(np.sum(np.abs(torques * v)) * dt),
        "torque_nm": float(np.max(np.abs(torques))),
    }


def aggregate_motion(output: Path, datasets: dict[str, Any], dynamics: bool) -> dict[str, Any]:
    """Score all native attempts using the matching robot inertia."""
    import pinocchio as pin

    from curobo._src.types.robot import RobotCfg

    rows = []
    for name, dataset in datasets.items():
        path = output / f"{name}_{'3kg' if dynamics else '0kg'}.jsonl"
        records = read_records(path, dataset["count"])
        identities = json.loads((output / f"{name}_problems.json").read_text())
        config = yaml.safe_load(Path(dataset["config"]).read_text())["robot_cfg"]
        kin = RobotCfg.create(config).kinematics.kinematics_config
        kin.link_masses_com[kin.link_name_to_idx_map["attached_object"], 3] = (
            3.0 if dynamics else 0.0
        )
        urdf = output / f"dynamics_{name}_{int(dynamics)}.urdf"
        kin.export_to_urdf(robot_name="benchmark", output_path=str(urdf), include_spheres=False)
        model = pin.buildModelFromUrdf(str(urdf))
        data = model.createData()
        states = (
            np.memmap(str(path) + ".states.f32", dtype=np.float32, mode="r")
            if any(r["success"] for r in records)
            else None
        )
        for index, row in enumerate(records):
            if row["index"] != index:
                raise ValueError("native motion output is out of order")
            row.update({key: identities[index][key] for key in ("dataset", "scene", "problem")})
            if row["success"]:
                offset, steps = row["state_offset"], row["steps"]
                trajectory = states[offset : offset + 4 * steps * model.nq].reshape(
                    4, steps, model.nq
                )
                row.update(trajectory_metrics(trajectory, row["dt"], model, data))
                row["position_error_mm"] = row["position_error_m"] * 1000
            rows.append(row)
    save_json(output / f"motion_{'3kg' if dynamics else '0kg'}_metrics.json", rows)
    success = [row for row in rows if row["success"]]
    metrics = (
        "plan_time_s",
        "position_error_mm",
        "path_length_rad",
        "motion_time_s",
        "jerk_rad_s3",
        "energy_j",
        "torque_nm",
    )
    return {
        "problems": len(rows),
        "successes": len(success),
        "success_percent": 100 * len(success) / len(rows),
        "excluded": sum(d["excluded"] for d in datasets.values()),
        "statistics": {key: statistics([row[key] for row in success]) for key in metrics},
        "all_attempt_plan_time_s": statistics([row["plan_time_s"] for row in rows]),
        "failures": [
            {key: row[key] for key in ("dataset", "scene", "problem", "attempts")}
            for row in rows
            if not row["success"]
        ],
    }


def render_tables(summary: dict[str, Any]) -> str:
    """Render all benchmark metrics in the reference page's original grid-table format."""
    from tabulate import tabulate

    sections = []
    labels = (
        ("plan_time_s", "Plan Time (s)"),
        ("position_error_mm", "Position Error (mm)"),
        ("path_length_rad", "Path Length (rad.)"),
        ("motion_time_s", "Motion Time(s)"),
        ("jerk_rad_s3", "Jerk (rad/s³)"),
        ("energy_j", "Energy (J)"),
        ("torque_nm", "Torque (N·m)"),
    )
    for name, result in summary["motion"].items():
        caption = (
            "Unloaded (0 kg):"
            if name == "0kg"
            else f"With torque limits at full payload of {name.removesuffix('kg')} kg:"
        )
        rows = [
            ["Successful problems", f"{result['successes']} / {result['problems']}"],
            ["Success %", f"{result['success_percent']:.2f}"],
        ]
        for key, label in labels:
            value = result["statistics"].get(key)
            precision = 5 if key == "plan_time_s" else 3
            description = (
                "Unavailable"
                if value is None
                else (
                    f"mean: {value['mean']:.{precision}f} ± {value['std']:.{precision}f}  "
                    f"median: {value['median']:.{precision}f}  "
                    f"75%: {value['p75']:.{precision}f}  98%: {value['p98']:.{precision}f}"
                )
            )
            rows.append([label, description])
        sections.append(
            caption
            + "\n\n"
            + tabulate(rows, headers=["Metric", "Value"], tablefmt="grid", disable_numparse=True)
        )
    if summary["ik"]:
        headers = [
            "robot",
            "IK-time(ms)",
            "Batch-Size",
            "Success-IK",
            "Position-Error(mm)",
            "Orientation-Error(deg)",
            "C-Free-IK-time(ms)",
            "Success-C-Free-IK",
            "Position-Error-C-Free-IK(mm)",
            "Orientation-Error-C-Free-IK(deg)",
        ]

        def metric(result: dict[str, Any] | None, key: str) -> str:
            """Format one IK metric while preserving missing measurements."""
            value = result.get(key) if result is not None else None
            return f"{value:.6g}" if value is not None else "Unavailable"

        rows = []
        for robot in ROBOTS:
            pose = summary["ik"].get(f"{robot}_pose_only")
            collision_free = summary["ik"].get(f"{robot}_collision_free")
            if pose is None and collision_free is None:
                continue
            rows.append(
                [
                    f"{robot}.yml",
                    metric(pose, "batch_time_ms"),
                    metric(pose or collision_free, "batch_size"),
                    metric(pose, "success_percent"),
                    metric(pose, "position_error_p90_mm"),
                    metric(pose, "orientation_error_p90_deg"),
                    metric(collision_free, "batch_time_ms"),
                    metric(collision_free, "success_percent"),
                    metric(collision_free, "position_error_p90_mm"),
                    metric(collision_free, "orientation_error_p90_deg"),
                ]
            )
        sections.append(
            tabulate(rows, headers=headers, tablefmt="grid", showindex=True, disable_numparse=True)
        )
    return "\n\n".join(sections) + "\n\n" if sections else ""


def main() -> None:
    """Build, prepare, run and summarize the selected native benchmark suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("all", "motion", "ik"), default="all")
    parser.add_argument(
        "--max-problems", type=int, default=0, help="Per-corpus motion limit; zero runs all"
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.max_problems < 0:
        parser.error("--max-problems must be nonnegative")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    provenance = source_provenance()
    provenance.update(
        {
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "suite": args.suite,
            "max_problems": args.max_problems,
            "gpu": run(["nvidia-smi"]),
            "command": sys.argv,
            "benchmark_sources": {
                p.name: sha256(p) for p in Path(__file__).parent.iterdir() if p.is_file()
            },
        }
    )
    save_json(output / "provenance.json", provenance)
    build = output / "build"
    import torch

    capability = torch.cuda.get_device_capability()
    run(
        [
            "cmake",
            "-S",
            "cpp_runtime/benchmarks",
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_CUDA_ARCHITECTURES={capability[0]}{capability[1]}",
            "-DBUILD_TESTING=OFF",
        ],
        output / "configure.log",
    )
    run(["cmake", "--build", str(build), "-j", "4"], output / "build.log")
    executable = build / "curobo_benchmark"
    source_hash = provenance["runtime_source_sha256"]
    datasets = prepare_motion(output, args.max_problems, source_hash) if args.suite != "ik" else {}
    cases = prepare_ik(output, source_hash) if args.suite != "motion" else []
    save_json(output / "inputs.json", {"motion": datasets, "ik": cases})
    artifact_paths = {Path(case["artifact"]) for case in cases}
    artifact_paths.update(Path(dataset["artifact"]) for dataset in datasets.values())
    save_json(
        output / "artifact_hashes.json",
        {
            str(path.relative_to(output)): sha256(path)
            for artifact in sorted(artifact_paths)
            for path in sorted(artifact.iterdir())
            if path.is_file() and path.suffix != ".log"
        },
    )
    if args.prepare_only:
        return
    summary: dict[str, Any] = {
        "backend": "C++ runtime",
        "cuda_graph": True,
        "timeopt": True,
        "partial": args.max_problems > 0,
        "motion": {},
        "ik": {},
    }
    if datasets:
        summary["motion_scene_capacities"] = {
            "cuboid": next(iter(datasets.values()))["max_cuboids"],
            "capsule": 0,
            "cylinder": 0,
            "mesh": 0,
            "voxel": 0,
        }
        summary["motion_interpolation_buffer_size"] = 1000
    for dynamics in (False, True):
        if not datasets:
            break
        for name, dataset in datasets.items():
            stem = f"{name}_{'3kg' if dynamics else '0kg'}"
            print(f"Running {stem}: {dataset['count']} problems", flush=True)
            run(
                [
                    str(executable),
                    "motion",
                    dataset["artifact"],
                    dataset["input"],
                    str(output / f"{stem}.jsonl"),
                    str(int(dynamics)),
                    "0",
                    str(dataset["max_cuboids"]),
                ],
                output / f"{stem}.log",
            )
        summary["motion"]["3kg" if dynamics else "0kg"] = aggregate_motion(
            output, datasets, dynamics
        )
        save_json(output / "summary.json", summary)
    for case in cases:
        print(f"Running {case['name']}: 500 IK queries", flush=True)
        path = output / f"{case['name']}.jsonl"
        run(
            [
                str(executable),
                "ik",
                case["artifact"],
                case["input"],
                str(path),
                str(int(case["collision_free"])),
                str(int(case["robot"] == "unitree_g1_29dof_retarget")),
            ],
            output / f"{case['name']}.log",
        )
        summary["ik"][case["name"]] = aggregate_ik(read_records(path, 500))
        save_json(output / "summary.json", summary)
    provenance["completed_utc"] = datetime.now(timezone.utc).isoformat()
    provenance["executable_sha256"] = sha256(executable)
    provenance["results"] = {
        p.name: sha256(p) for p in output.iterdir() if p.suffix in (".jsonl", ".f32")
    }
    save_json(output / "provenance.json", provenance)
    (output / "tables.rst").write_text(render_tables(summary))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
