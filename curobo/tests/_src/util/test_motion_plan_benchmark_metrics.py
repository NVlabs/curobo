# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the motion-planning benchmark interface."""

from __future__ import annotations

# Standard Library
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

# Third Party
import yaml


def _load_benchmark_module() -> ModuleType:
    """Load the benchmark script as a module."""
    module_path = Path(__file__).resolve().parents[4] / "benchmark" / "motion_plan_benchmark.py"
    spec = importlib.util.spec_from_file_location(
        "motion_plan_benchmark_test",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load motion_plan_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_exposes_only_motion_benchmark_options() -> None:
    """Keep graph, optimizer-ablation, and NVTX controls out of this benchmark."""
    benchmark = _load_benchmark_module()
    parser = benchmark.create_argument_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option != "--help"
    }

    assert options == {
        "-h",
        "--enable_profile",
        "--save_path",
        "--file_name",
        "--mesh",
        "--kpi",
        "--demo",
        "--disable_cuda_graph",
        "--use-dynamics",
        "--enable-timeopt",
        "--write_benchmark",
        "--save_usd",
        "--save_plot",
        "--mass",
    }


def test_kpi_contains_motion_and_rrt_metrics(tmp_path: Path) -> None:
    """Serialize canonical motion results with RRT timing telemetry."""
    benchmark = _load_benchmark_module()
    statistic = SimpleNamespace(
        mean=1.0,
        std=0.1,
        median=0.9,
        percent_75=1.1,
        percent_95=1.2,
        percent_98=1.3,
        max=1.4,
    )
    metrics = SimpleNamespace(
        group_size=10,
        success=90.0,
        time=statistic,
        solve_time=statistic,
        position_error=statistic,
        orientation_error=statistic,
        cspace_path_length=statistic,
        motion_time=statistic,
        jerk=statistic,
        energy=statistic,
        torque=statistic,
        graph_invoked_problems=3,
        graph_successful_problems=2,
        graph_calls=4,
        graph_time=statistic,
        graph_time_per_call=statistic,
    )
    args = SimpleNamespace(
        demo=False,
        save_path=str(tmp_path),
        file_name="motion",
        use_dynamics=False,
        enable_timeopt=False,
        mass=3.0,
    )

    benchmark.write_kpi(metrics, args)

    output = yaml.safe_load((tmp_path / "motion.yml").read_text())
    assert output["Benchmark"] == "motion"
    assert benchmark.benchmark_dataset_name(SimpleNamespace(demo=True)) == "demo"
    assert output["Dataset"] == "all"
    assert output["Successful Problems"] == 9
    assert output["Total Problems"] == 10
    assert output["Success"] == 90.0
    assert output["Attached Mass (kg)"] == 0.0
    assert output["TimeOpt Enabled"] is False
    assert output["Planning Time"] == 1.0
    assert output["Solve Time"] == 1.0
    assert output["RRT Invoked Problems"] == 3
    assert output["RRT Successful Problems"] == 2
    assert output["RRT Calls"] == 4
    assert output["RRT Time (s)"] == 1.0
    assert output["RRT Time Per Call (s)"] == 1.0
    assert not any("Graph" in key for key in output)
