#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export a robot model and current cuRobo kernels for the C++ runtime."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
import warp as wp
from cuda import pathfinder
from cuda.core import Device, ObjectCode, Program, ProgramOptions

CUROBO_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
MAX_RRT_PLANS = 128
RRT_SAMPLE_BUFFER_SIZE = 2000

TENSOR_SPECS = {
    "fixed_transforms": (torch.float32, "float32", "fixed_transforms.f32"),
    "joint_map_type": (torch.int8, "int8", "joint_map_type.i8"),
    "joint_map": (torch.int16, "int16", "joint_map.i16"),
    "link_map": (torch.int16, "int16", "link_map.i16"),
    "tool_frame_map": (torch.int16, "int16", "tool_frame_map.i16"),
    "joint_offset_map": (torch.float32, "float32", "joint_offset_map.f32"),
    "tool_joint_links_data": (torch.int16, "int16", "tool_joint_links_data.i16"),
    "tool_joint_links_offsets": (torch.int16, "int16", "tool_joint_links_offsets.i16"),
    "joint_link_influence_words": (
        torch.int32,
        "int32",
        "joint_link_influence_words.i32",
    ),
    "joint_links_data": (torch.int16, "int16", "joint_links_data.i16"),
    "joint_links_offsets": (torch.int16, "int16", "joint_links_offsets.i16"),
    "link_masses_com": (torch.float32, "float32", "link_masses_com.f32"),
}

TRAJOPT_TENSOR_SPECS = {
    "position_bounds": (torch.float32, "float32", "trajopt_position_bounds.f32"),
    "velocity_bounds": (torch.float32, "float32", "trajopt_velocity_bounds.f32"),
    "acceleration_bounds": (
        torch.float32,
        "float32",
        "trajopt_acceleration_bounds.f32",
    ),
    "jerk_bounds": (torch.float32, "float32", "trajopt_jerk_bounds.f32"),
    "state_bound_weight": (
        torch.float32,
        "float32",
        "trajopt_state_bound_weight.f32",
    ),
    "state_activation_distance": (
        torch.float32,
        "float32",
        "trajopt_state_activation_distance.f32",
    ),
    "state_regularization_weight": (
        torch.float32,
        "float32",
        "trajopt_state_regularization_weight.f32",
    ),
    "cspace_target_weight": (
        torch.float32,
        "float32",
        "trajopt_cspace_target_weight.f32",
    ),
    "cspace_non_terminal_weight": (
        torch.float32,
        "float32",
        "trajopt_cspace_non_terminal_weight.f32",
    ),
    "cspace_target_dof_weight": (
        torch.float32,
        "float32",
        "trajopt_cspace_target_dof_weight.f32",
    ),
    "tool_pose_weight": (
        torch.float32,
        "float32",
        "trajopt_tool_pose_weight.f32",
    ),
    "world_collision_weight": (
        torch.float32,
        "float32",
        "trajopt_world_collision_weight.f32",
    ),
    "self_collision_weight": (
        torch.float32,
        "float32",
        "trajopt_self_collision_weight.f32",
    ),
    "metrics_world_collision_weight": (
        torch.float32,
        "float32",
        "trajopt_metrics_world_collision_weight.f32",
    ),
    "metrics_self_collision_weight": (
        torch.float32,
        "float32",
        "trajopt_metrics_self_collision_weight.f32",
    ),
    "validation_self_collision_weight": (
        torch.float32,
        "float32",
        "trajopt_validation_self_collision_weight.f32",
    ),
    "line_search_scale": (
        torch.float32,
        "float32",
        "trajopt_line_search_scale.f32",
    ),
    "optimization_step_max": (
        torch.float32,
        "float32",
        "trajopt_optimization_step_max.f32",
    ),
}

ROLLOUT_TENSOR_SPECS = {
    "rnea_fixed_transforms": (torch.float32, "float32", "rnea_fixed_transforms.f32"),
    "rnea_link_masses_com": (torch.float32, "float32", "rnea_link_masses_com.f32"),
    "rnea_link_inertias": (torch.float32, "float32", "rnea_link_inertias.f32"),
    "rnea_original_link_masses_com": (
        torch.float32,
        "float32",
        "rnea_original_link_masses_com.f32",
    ),
    "rnea_original_link_inertias": (
        torch.float32,
        "float32",
        "rnea_original_link_inertias.f32",
    ),
    "rnea_external_force_owners": (
        torch.int16,
        "int16",
        "rnea_external_force_owners.i16",
    ),
    "rnea_external_force_transforms": (
        torch.float32,
        "float32",
        "rnea_external_force_transforms.f32",
    ),
    "rnea_effort_bounds": (torch.float32, "float32", "rnea_effort_bounds.f32"),
    "rnea_joint_map_type": (torch.int8, "int8", "rnea_joint_map_type.i8"),
    "rnea_joint_map": (torch.int16, "int16", "rnea_joint_map.i16"),
    "rnea_link_map": (torch.int16, "int16", "rnea_link_map.i16"),
    "rnea_chain_map": (torch.int16, "int16", "rnea_chain_map.i16"),
    "rnea_joint_offset_map": (torch.float32, "float32", "rnea_joint_offset_map.f32"),
    "rnea_gravity": (torch.float32, "float32", "rnea_gravity.f32"),
    "rnea_level_starts": (torch.int16, "int16", "rnea_level_starts.i16"),
    "rnea_level_links": (torch.int16, "int16", "rnea_level_links.i16"),
}

RRT_TENSOR_SPECS = {
    "rrt_position_lower": (torch.float32, "float32", "rrt_position_lower.f32"),
    "rrt_position_upper": (torch.float32, "float32", "rrt_position_upper.f32"),
    "rrt_distance_weight": (torch.float32, "float32", "rrt_distance_weight.f32"),
    "rrt_samples": (torch.float32, "float32", "rrt_samples.f32"),
    "rrt_shortcut_pair_sources": (
        torch.int32,
        "int32",
        "rrt_shortcut_pair_sources.i32",
    ),
    "rrt_shortcut_pair_destinations": (
        torch.int32,
        "int32",
        "rrt_shortcut_pair_destinations.i32",
    ),
    "rrt_shortcut_edge_lookup": (
        torch.int32,
        "int32",
        "rrt_shortcut_edge_lookup.i32",
    ),
}

MOTION_TENSOR_SPECS = {
    "motion_arc_progress": (
        torch.float32,
        "float32",
        "motion_arc_progress.f32",
    ),
    "motion_fit_matrix": (
        torch.float32,
        "float32",
        "motion_fit_matrix.f32",
    ),
    "motion_start_coefficient": (
        torch.float32,
        "float32",
        "motion_start_coefficient.f32",
    ),
    "motion_start_velocity_coefficient": (
        torch.float32,
        "float32",
        "motion_start_velocity_coefficient.f32",
    ),
    "motion_start_acceleration_coefficient": (
        torch.float32,
        "float32",
        "motion_start_acceleration_coefficient.f32",
    ),
    "motion_start_jerk_coefficient": (
        torch.float32,
        "float32",
        "motion_start_jerk_coefficient.f32",
    ),
    "motion_goal_coefficient": (
        torch.float32,
        "float32",
        "motion_goal_coefficient.f32",
    ),
    "motion_basis_position": (
        torch.float32,
        "float32",
        "motion_basis_position.f32",
    ),
    "motion_basis_velocity": (
        torch.float32,
        "float32",
        "motion_basis_velocity.f32",
    ),
    "motion_basis_acceleration": (
        torch.float32,
        "float32",
        "motion_basis_acceleration.f32",
    ),
    "motion_basis_jerk": (
        torch.float32,
        "float32",
        "motion_basis_jerk.f32",
    ),
    "motion_path_normal": (
        torch.float32,
        "float32",
        "motion_path_normal.f32",
    ),
    "motion_max_velocity": (
        torch.float32,
        "float32",
        "motion_max_velocity.f32",
    ),
    "motion_max_acceleration": (
        torch.float32,
        "float32",
        "motion_max_acceleration.f32",
    ),
    "motion_max_jerk": (
        torch.float32,
        "float32",
        "motion_max_jerk.f32",
    ),
}

IK_TENSOR_SPECS = {
    "ik_position_bounds": (torch.float32, "float32", "ik_position_bounds.f32"),
    "joint_position_lower": (torch.float32, "float32", "joint_position_lower.f32"),
    "joint_position_upper": (torch.float32, "float32", "joint_position_upper.f32"),
    "default_joint_position": (torch.float32, "float32", "default_joint_position.f32"),
    "seed_positions": (torch.float32, "float32", "ik_seed_positions.f32"),
    "link_spheres": (torch.float32, "float32", "link_spheres.f32"),
    "sphere_collision_mask": (torch.uint32, "uint32", "sphere_collision_mask.u32"),
    "link_sphere_map": (torch.int16, "int16", "link_sphere_map.i16"),
    "self_collision_padding": (
        torch.float32,
        "float32",
        "self_collision_padding.f32",
    ),
    "self_collision_pairs": (torch.int16, "int16", "self_collision_pairs.i16"),
    "ik_velocity_bounds": (torch.float32, "float32", "ik_velocity_bounds.f32"),
    "ik_cspace_weight": (torch.float32, "float32", "ik_cspace_weight.f32"),
    "ik_cspace_activation_distance": (
        torch.float32,
        "float32",
        "ik_cspace_activation_distance.f32",
    ),
    "ik_cspace_target_weight": (
        torch.float32,
        "float32",
        "ik_cspace_target_weight.f32",
    ),
    "ik_cspace_target_dof_weight": (
        torch.float32,
        "float32",
        "ik_cspace_target_dof_weight.f32",
    ),
    "ik_cspace_regularization_weight": (
        torch.float32,
        "float32",
        "ik_cspace_regularization_weight.f32",
    ),
    "ik_start_cspace_weight": (
        torch.float32,
        "float32",
        "ik_start_cspace_weight.f32",
    ),
    "ik_start_cspace_dof_weight": (
        torch.float32,
        "float32",
        "ik_start_cspace_dof_weight.f32",
    ),
    "ik_world_collision_weight": (
        torch.float32,
        "float32",
        "ik_world_collision_weight.f32",
    ),
    "ik_self_collision_weight": (
        torch.float32,
        "float32",
        "ik_self_collision_weight.f32",
    ),
    "ik_optimizer_collision_activation_distance": (
        torch.float32,
        "float32",
        "ik_optimizer_collision_activation_distance.f32",
    ),
    "ik_line_search_scale": (
        torch.float32,
        "float32",
        "ik_line_search_scale.f32",
    ),
    "ik_optimization_step_max": (
        torch.float32,
        "float32",
        "ik_optimization_step_max.f32",
    ),
}


def ensure_curobo_on_path() -> Path:
    """Put the selected cuRobo checkout first on the Python import path."""
    source_dir = Path(os.environ.get("CUROBO_SOURCE_DIR", CUROBO_REPOSITORY_ROOT)).resolve()
    runtime_root = str(CUROBO_REPOSITORY_ROOT)
    if runtime_root not in sys.path:
        sys.path.insert(0, runtime_root)
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    return source_dir


def prepare_attachment_robot_config(
    robot_config: Path,
    output_dir: Path,
    parent_link: str,
    sphere_capacity: int,
) -> Path:
    """Write the resolved robot YAML with fixed payload collision storage."""
    ensure_curobo_on_path()
    from curobo._src.util.config_io import load_yaml, write_yaml

    if not parent_link:
        raise ValueError("attachment parent link must be non-empty")
    if sphere_capacity <= 0:
        raise ValueError("attachment sphere capacity must be positive")

    robot_data = load_yaml(str(robot_config.resolve()))
    config_root = robot_data.get("robot_cfg", robot_data)
    if "kinematics" not in config_root:
        raise ValueError("robot config has no kinematics section")
    kinematics = config_root["kinematics"]

    collision_links = kinematics.setdefault("collision_link_names", [])
    if "attached_object" not in collision_links:
        collision_links.append("attached_object")
    kinematics.setdefault("extra_collision_spheres", {})["attached_object"] = sphere_capacity
    kinematics.setdefault("extra_links", {})["attached_object"] = {
        "fixed_transform": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "joint_name": "attached_object_fixed_joint",
        "joint_type": "FIXED",
        "link_name": "attached_object",
        "parent_link_name": parent_link,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = output_dir / "robot_config.runtime.yml"
    write_yaml(robot_data, str(resolved_path))
    return resolved_path


def build_kinematics_config(
    args: argparse.Namespace, robot_config_override: Path | None = None
) -> tuple[Any, dict[str, Any]]:
    """Load the authoring model and return its resolved source description."""
    ensure_curobo_on_path()
    from curobo.kinematics import KinematicsCfg

    tool_frames = args.tool_frame or None
    urdf_path = str(args.urdf.resolve()) if args.urdf is not None else None
    selected_robot_config = robot_config_override or args.robot_config
    if selected_robot_config is not None:
        robot_config = str(selected_robot_config.resolve())
        config = KinematicsCfg.from_robot_yaml_file(
            robot_config,
            tool_frames=tool_frames,
            urdf_path=urdf_path,
        )
        source = {
            "kind": "robot_config",
            "robot_config": robot_config,
        }
        if urdf_path is not None:
            source["urdf"] = urdf_path
        return config, source

    if args.urdf is None or args.base_link is None or not args.tool_frame:
        raise ValueError(
            "direct URDF export requires --urdf, --base-link, and at least one --tool-frame"
        )
    config = KinematicsCfg.from_basic_urdf(
        urdf_path=urdf_path,
        base_link=args.base_link,
        tool_frames=args.tool_frame,
    )
    return config, {
        "kind": "urdf",
        "urdf": urdf_path,
        "base_link": args.base_link,
    }


def expand_joint_position_limits(params: Any, expansion: float) -> None:
    """Expand lower and upper active-joint position limits symmetrically in radians."""
    if not math.isfinite(expansion) or expansion < 0.0:
        raise ValueError("joint-limit expansion must be finite and non-negative")
    if expansion == 0.0:
        return
    params.joint_limits.position[0].sub_(expansion)
    params.joint_limits.position[1].add_(expansion)


def kinematics_source() -> tuple[str, list[str]]:
    """Read the current cuRobo kinematics source and return its include paths."""
    ensure_curobo_on_path()
    from curobo._src.context import get_runtime
    from curobo._src.curobolib.backends.cuda_core_backend.kinematics_config import (
        KinematicsKernelCfg,
    )

    kernel_config = KinematicsKernelCfg()
    include_paths = kernel_config.get_include_dirs()
    source_files = [
        kernel_config.kernel_dir / filename
        for filename in kernel_config.get_kernel_files("forward")
    ]
    source = get_runtime().get_cuda_core_cache()._read_sources(source_files, include_paths)
    include_dirs = [str(path) for path in include_paths]
    for library_name in ("cudart", "nvrtc"):
        path = pathfinder.find_nvidia_header_directory(library_name)
        if path is not None and os.path.isdir(path):
            include_dirs.append(path)
    return source, list(dict.fromkeys(include_dirs))


def export_tensor(name: str, tensor: torch.Tensor, output_dir: Path) -> dict[str, Any]:
    """Validate and write one contiguous model tensor."""
    if name in TENSOR_SPECS:
        specs = TENSOR_SPECS
    elif name in IK_TENSOR_SPECS:
        specs = IK_TENSOR_SPECS
    elif name in RRT_TENSOR_SPECS:
        specs = RRT_TENSOR_SPECS
    elif name in MOTION_TENSOR_SPECS:
        specs = MOTION_TENSOR_SPECS
    elif name in ROLLOUT_TENSOR_SPECS:
        specs = ROLLOUT_TENSOR_SPECS
    else:
        specs = TRAJOPT_TENSOR_SPECS
    expected_torch_dtype, dtype_name, filename = specs[name]
    value = tensor.detach().contiguous().cpu()
    if value.dtype != expected_torch_dtype:
        raise TypeError(f"{name} must be {expected_torch_dtype}, got {value.dtype}")
    payload = value.numpy().tobytes(order="C")
    (output_dir / filename).write_bytes(payload)
    return {
        "file": filename,
        "dtype": dtype_name,
        "shape": list(value.shape),
        "bytes": len(payload),
    }


def compile_kinematics_kernel(
    output_dir: Path,
    num_links: int,
    num_dof: int,
    num_tool_frames: int,
    num_spheres: int,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Compile the existing fused cuRobo FK/Jacobian specialization."""
    device = Device()
    device.set_current()
    arch = f"sm_{int(device.arch)}"
    requested_symbol = (
        "curobo::kinematics::kinematics_forward_spheres_jacobian_kernel"
        f"<{num_links}, 32, false, false, {num_dof}, 0, {num_tool_frames}, 1>"
    )
    ik_requested_symbol = (
        "curobo::kinematics::kinematics_forward_spheres_jacobian_kernel"
        f"<{num_links}, 32, true, false, {num_dof}, {num_spheres}, "
        f"{num_tool_frames}, 1>"
    )
    trajopt_requested_symbol = (
        "curobo::kinematics::kinematics_forward_spheres_kernel"
        f"<{num_links}, 32, true, false, {num_dof}, {num_spheres}, "
        f"{num_tool_frames}, 1>"
    )
    source, include_dirs = kinematics_source()
    options = ProgramOptions(
        std="c++17",
        arch=arch,
        include_path=include_dirs,
        ftz=True,
        fma=True,
        prec_div=False,
        prec_sqrt=False,
        lineinfo=True,
        device_code_optimize=True,
    )
    obj = Program(
        source,
        code_type="c++",
        options=options,
    ).compile(
        "cubin",
        name_expressions=tuple(
            dict.fromkeys((requested_symbol, ik_requested_symbol, trajopt_requested_symbol))
        ),
    )
    symbol_mapping = {
        key: value.decode("utf-8") if isinstance(value, bytes) else value
        for key, value in obj.symbol_mapping.items()
    }
    load_symbol = symbol_mapping.get(requested_symbol, requested_symbol)
    cubin_name = f"robot_kinematics.{arch}.cubin"
    cubin_path = output_dir / cubin_name
    cubin_path.write_bytes(obj.code)

    reloaded = ObjectCode.from_cubin(
        str(cubin_path),
        symbol_mapping={key: value.encode("utf-8") for key, value in symbol_mapping.items()},
    )
    reloaded.get_kernel(requested_symbol)
    reloaded.get_kernel(ik_requested_symbol)
    reloaded.get_kernel(trajopt_requested_symbol)
    base_kernel = {
        "cubin": cubin_name,
        "symbol": load_symbol,
        "requested_symbol": requested_symbol,
        "source": "curobo_kinematics_kernel",
        "threads_per_batch": 32,
        "max_batches_per_block": 8,
        "max_threads_per_block": 256,
        "max_dynamic_shared_memory": 49152,
    }
    ik_kernel = {
        **base_kernel,
        "symbol": symbol_mapping.get(ik_requested_symbol, ik_requested_symbol),
        "requested_symbol": ik_requested_symbol,
    }
    trajopt_kernel = {
        **base_kernel,
        "symbol": symbol_mapping.get(trajopt_requested_symbol, trajopt_requested_symbol),
        "requested_symbol": trajopt_requested_symbol,
    }
    return arch, base_kernel, ik_kernel, trajopt_kernel


def _export_warp_kernel(
    kernel: Any,
    output_dir: Path,
    artifact_name: str,
    arch: str,
    *,
    strip_hash: bool = True,
    block_dim: int = 128,
) -> dict[str, Any]:
    """Compile one existing Warp kernel module and return its load metadata."""
    kernel.module.options["block_dim"] = block_dim
    with tempfile.TemporaryDirectory(prefix="curobo_robot_warp_") as build_dir:
        wp.compile_aot_module(
            kernel.module,
            arch=int(arch.removeprefix("sm_")),
            module_dir=build_dir,
            use_ptx=False,
            strip_hash=strip_hash,
        )
        symbol = f"{kernel.get_mangled_name()}_cuda_kernel_forward"
        build_path = Path(build_dir)
        cubin_path = next(build_path.glob("*.cubin"))
        meta_path = next(build_path.glob("*.meta"))
        cubin_name = f"{artifact_name}.{arch}.cubin"
        meta_name = f"{artifact_name}.{arch}.warp_meta.json"
        shutil.copy2(cubin_path, output_dir / cubin_name)
        shutil.copy2(meta_path, output_dir / meta_name)
        metadata = json.loads(meta_path.read_text())
    return {
        "cubin": cubin_name,
        "symbol": symbol,
        "source": "curobo_python_warp_kernel",
        "shared_memory": int(metadata.get(f"{symbol}_smem_bytes", 0)),
        "block_dim": block_dim,
    }


def _compile_cuda_specializations(
    output_dir: Path,
    arch: str,
    artifact_name: str,
    kernel_config: Any,
    kernel_types: list[str],
    requested_symbols: list[str],
) -> tuple[str, dict[str, str]]:
    """Compile named specializations from current in-tree CUDA sources."""
    ensure_curobo_on_path()
    from curobo._src.context import get_runtime

    include_paths = kernel_config.get_include_dirs()
    source_files = []
    for kernel_type in kernel_types:
        source_files.extend(
            kernel_config.kernel_dir / filename
            for filename in kernel_config.get_kernel_files(kernel_type)
        )
    source_files = list(dict.fromkeys(source_files))
    source = get_runtime().get_cuda_core_cache()._read_sources(source_files, include_paths)
    include_dirs = [str(path) for path in include_paths]
    for library_name in ("cudart", "nvrtc"):
        path = pathfinder.find_nvidia_header_directory(library_name)
        if path is not None and os.path.isdir(path):
            include_dirs.append(path)
    options = ProgramOptions(
        std="c++17",
        arch=arch,
        include_path=list(dict.fromkeys(include_dirs)),
        ftz=True,
        fma=True,
        prec_div=False,
        prec_sqrt=False,
        lineinfo=True,
        device_code_optimize=True,
    )
    obj = Program(source, code_type="c++", options=options).compile(
        "cubin", name_expressions=tuple(requested_symbols)
    )
    symbol_mapping = {
        key: value.decode("utf-8") if isinstance(value, bytes) else value
        for key, value in obj.symbol_mapping.items()
    }
    cubin_name = f"{artifact_name}.{arch}.cubin"
    cubin_path = output_dir / cubin_name
    cubin_path.write_bytes(obj.code)
    reloaded = ObjectCode.from_cubin(
        str(cubin_path),
        symbol_mapping={key: value.encode("utf-8") for key, value in symbol_mapping.items()},
    )
    for requested_symbol in requested_symbols:
        reloaded.get_kernel(requested_symbol)
    return cubin_name, symbol_mapping


def compile_kinematics_backward_kernel(
    output_dir: Path,
    arch: str,
    params: Any,
) -> dict[str, Any]:
    """Compile the shared sphere/link-gradient to joint-gradient specialization."""
    from curobo._src.curobolib.backends.cuda_core_backend.kinematics_config import (
        KinematicsKernelCfg,
    )

    backward_threads = 128
    requested_symbol = (
        "curobo::kinematics::kinematics_link_wrench_backward_kernel"
        f"<float, {params.num_links}, {params.num_dof}, {params.total_spheres}, "
        f"{params.num_pose_links}, false, {backward_threads}, "
        f"{str(params.joint_links_data.shape[0] == params.num_dof).lower()}>"
    )
    cubin_name, symbols = _compile_cuda_specializations(
        output_dir,
        arch,
        "robot_kinematics_backward",
        KinematicsKernelCfg(),
        ["backward"],
        [requested_symbol],
    )
    return {
        "cubin": cubin_name,
        "symbol": symbols.get(requested_symbol, requested_symbol),
        "source": "curobo_kinematics_kernel",
        "shared_memory": params.num_links * (8 + 12) * 4,
        "block_dim": backward_threads,
    }


def compile_lbfgs_kernels(
    output_dir: Path,
    arch: str,
    artifact_prefix: str,
    optimization_dimension: int,
    max_problems: int,
    optimizer: Any,
) -> dict[str, dict[str, Any]]:
    """Compile cuRobo's current L-BFGS direction and line-search kernels."""
    from curobo._src.curobolib.backends.cuda_core_backend.optimization_config import (
        LBFGSLaunchCfg,
        LineSearchLaunchCfg,
        OptimizationKernelCfg,
    )

    history = optimizer.history
    num_line_search = len(optimizer.line_search_scale)
    items_per_thread, lbfgs_threads = LBFGSLaunchCfg.calculate_items_and_threads(
        optimization_dimension
    )
    if items_per_thread == 0:
        raise ValueError("L-BFGS optimization dimension exceeds the native kernel limit")
    properties = torch.cuda.get_device_properties(optimizer.device_cfg.device)
    device_shared_memory = int(
        getattr(properties, "shared_memory_per_block_optin", 0)
        or properties.shared_memory_per_block
    )
    _, use_shared_history, lbfgs_shared = LBFGSLaunchCfg.calculate_config(
        max_problems,
        optimization_dimension,
        history,
        True,
        device_shared_memory,
    )
    lbfgs_symbol = (
        "curobo::optimization::kernel_lbfgs_step_circular"
        f"<float, false, {history if history <= 31 else -1}, "
        f"{str(use_shared_history).lower()}, {items_per_thread}, "
        f"{str(optimizer.use_geometric_scaling).lower()}>"
    )
    line_search_symbols = {
        name: (
            "curobo::optimization::line_search::kernel_line_search"
            f"<float, {num_line_search}, 2, "
            f"{str(optimization_dimension <= LineSearchLaunchCfg.MAX_THREADS_PER_BLOCK).lower()}, "
            f"{str(backtracking).lower()}>"
        )
        for name, backtracking in (("line_search", False), ("line_search_backtracking", True))
    }
    optimization_config = OptimizationKernelCfg()
    lbfgs_cubin, lbfgs_symbols = _compile_cuda_specializations(
        output_dir,
        arch,
        f"{artifact_prefix}_lbfgs",
        optimization_config,
        ["lbfgs"],
        [lbfgs_symbol],
    )
    line_cubin, line_symbols = _compile_cuda_specializations(
        output_dir,
        arch,
        f"{artifact_prefix}_line_search",
        optimization_config,
        ["line_search"],
        list(line_search_symbols.values()),
    )
    line_threads = min(((optimization_dimension + 31) // 32) * 32, 1024)
    return {
        "lbfgs": {
            "cubin": lbfgs_cubin,
            "symbol": lbfgs_symbols.get(lbfgs_symbol, lbfgs_symbol),
            "source": "curobo_lbfgs_kernel",
            "shared_memory": lbfgs_shared,
            "block_dim": lbfgs_threads,
        },
        **{
            name: {
                "cubin": line_cubin,
                "symbol": line_symbols.get(symbol, symbol),
                "source": "curobo_line_search_kernel",
                "shared_memory": 0,
                "block_dim": line_threads,
            }
            for name, symbol in line_search_symbols.items()
        },
    }


def _collision_block_dim(num_spheres: int) -> int:
    """Choose the same supported tiled width used for small Python collision batches."""
    if num_spheres <= 64:
        return 64
    if num_spheres <= 128:
        return 128
    if num_spheres <= 256:
        return 256
    if num_spheres <= 512:
        return 512
    return 1024


def export_ik_kernels(
    output_dir: Path,
    arch: str,
    params: Any,
    optimizer: Any,
    max_seeds: int,
    scene_capacities: dict[str, int],
) -> tuple[dict[str, Any], int]:
    """Export current cuRobo LM, L-BFGS, cost, and collision kernels for IK."""
    ensure_curobo_on_path()
    from cpp_runtime.robot.kernels.ik import create_seed_ik_kernels
    from curobo._src.cost.wp_cspace_position import _make_forward_cspace_position_kernel
    from curobo._src.geom.collision.wp_collision_kernel import get_tiled_collision_kernel
    from curobo._src.geom.data.data_capsule import CapsuleDataWarp
    from curobo._src.geom.data.data_cuboid import CuboidDataWarp
    from curobo._src.geom.data.data_cylinder import CylinderDataWarp
    from curobo._src.geom.data.data_mesh import MeshDataWarp
    from curobo._src.geom.data.data_voxel import VoxelDataWarp
    from curobo._src.optim.util.levenberg_marquardt_step import LevenbergMarquardtStep
    from curobo._src.util.warp import init_warp

    init_warp()
    num_dof = params.num_dof
    num_tool_frames = params.num_pose_links
    num_spheres = params.total_spheres
    num_line_search = len(optimizer.line_search_scale)
    kernels = create_seed_ik_kernels(num_dof, num_tool_frames, num_line_search)
    num_residuals = int(kernels.pop("num_residuals"))
    exported = {
        name: _export_warp_kernel(kernel, output_dir, f"ik_{name}", arch)
        for name, kernel in kernels.items()
    }
    lm_kernel = LevenbergMarquardtStep.create_lm_warp_kernel(num_dof, num_residuals)
    exported["levenberg_marquardt"] = _export_warp_kernel(
        lm_kernel, output_dir, "ik_levenberg_marquardt", arch, block_dim=32
    )
    exported["cspace"] = _export_warp_kernel(
        _make_forward_cspace_position_kernel(True),
        output_dir,
        "ik_cspace",
        arch,
        block_dim=128,
    )
    exported.update(
        compile_lbfgs_kernels(
            output_dir,
            arch,
            "ik",
            num_dof,
            max_seeds,
            optimizer,
        )
    )

    if num_spheres <= 0:
        return exported, num_residuals

    obstacle_types = {
        "cuboid": CuboidDataWarp,
        "capsule": CapsuleDataWarp,
        "cylinder": CylinderDataWarp,
        "mesh": MeshDataWarp,
        "voxel": VoxelDataWarp,
    }
    block_dim = _collision_block_dim(num_spheres)
    for name, obstacle_type in obstacle_types.items():
        capacity = scene_capacities[name]
        if capacity <= 0:
            continue
        collision_kernel = get_tiled_collision_kernel(capacity, 1, num_spheres, block_dim)
        overload = wp.overload(
            collision_kernel,
            [
                obstacle_type,
                wp.array(dtype=wp.vec4),
                wp.array(dtype=wp.float32),
                wp.array(dtype=wp.float32),
                wp.array(dtype=wp.int32),
                wp.array(dtype=wp.float32),
                wp.array(dtype=wp.float32),
                wp.uint8,
                wp.uint8,
                wp.uint8,
                wp.array(dtype=wp.uint32),
                wp.array2d(dtype=wp.uint32),
            ],
        )
        collision_metadata = _export_warp_kernel(
            overload,
            output_dir,
            f"ik_collision_{name}",
            arch,
            strip_hash=False,
            block_dim=block_dim,
        )
        collision_metadata["block_dim"] = block_dim
        collision_metadata["max_obstacles"] = capacity
        exported[f"collision_{name}"] = collision_metadata
    return exported, num_residuals


def compile_self_collision_kernel(
    output_dir: Path,
    arch: str,
    self_collision_config: Any,
    num_spheres: int,
) -> dict[str, Any] | None:
    """Compile cuRobo's current max-pair self-collision cost and gradient."""
    if (
        self_collision_config is None
        or self_collision_config.collision_pairs is None
        or self_collision_config.collision_pairs.shape[0] == 0
        or num_spheres == 0
    ):
        return None
    from curobo._src.curobolib.backends.cuda_core_backend.geometry_config import (
        GeometryKernelCfg,
    )

    num_pairs = int(self_collision_config.collision_pairs.shape[0])
    max_threads = int(self_collision_config.max_threads_per_block)
    block_dim = min(max_threads, num_pairs)
    block_dim = min(((block_dim + 31) // 32) * 32, max_threads)
    requested_symbol = (
        "curobo::geometry::self_collision::self_collision_exact_distance_kernel"
        f"<0, false, true, {num_spheres}, {num_pairs}, {block_dim}>"
    )
    cubin_name, symbols = _compile_cuda_specializations(
        output_dir,
        arch,
        "robot_self_collision",
        GeometryKernelCfg(),
        ["self_collision"],
        [requested_symbol],
    )
    return {
        "cubin": cubin_name,
        "symbol": symbols.get(requested_symbol, requested_symbol),
        "requested_symbol": requested_symbol,
        "source": "curobo_self_collision_kernel",
        "shared_memory": num_spheres * 4 * 4,
        "block_dim": block_dim,
        "num_pairs": num_pairs,
    }


def build_ik_optimizer_config(params: Any) -> tuple[dict[str, Any], Any]:
    """Load and validate the current Python L-BFGS IK deployment profile."""
    ensure_curobo_on_path()
    from curobo._src.optim.gradient.lbfgs import LBFGSOptCfg
    from curobo._src.solver.solver_ik_cfg import IKSolverCfg
    from curobo._src.util.config_io import load_yaml
    from curobo.content import get_task_configs_path

    profile = load_yaml(str(get_task_configs_path() / "ik" / "lbfgs_ik.yml"))
    profile["rollout"]["constraint_cfg"]["scene_collision_cfg"]["activation_distance"] = (
        IKSolverCfg.__dataclass_fields__["optimizer_collision_activation_distance"].default
    )
    optimizer = LBFGSOptCfg(
        **LBFGSOptCfg.create_data_dict(
            profile["optimizer"],
            device_cfg=params.device_cfg,
        )
    )
    if optimizer.line_search_type.value != "approx_wolfe":
        raise ValueError("C++ IK currently requires approximate-Wolfe line search")
    if len(optimizer.line_search_scale) < 1 or len(optimizer.line_search_scale) > 32:
        raise ValueError("C++ IK requires one through 32 line-search probes")
    return profile, optimizer


def build_ik_tensors(
    config: Any,
    params: Any,
    max_seeds: int,
    optimizer_profile: dict[str, Any],
    optimizer: Any,
) -> dict[str, torch.Tensor]:
    """Build model, seed, cost, and optimizer tensors for deployment IK."""
    from curobo._src.solver.seed_ik.seed_ik_solver_cfg import SeedIKSolverCfg
    from curobo._src.util.config_io import load_yaml
    from curobo._src.util.sampling.sequencer_halton import HaltonSequencer
    from curobo.content import get_task_configs_path

    position_bounds = params.joint_limits.position.detach().to(dtype=torch.float32)
    joint_limit_margin = (
        position_bounds[1] - position_bounds[0]
    ) * SeedIKSolverCfg.__dataclass_fields__["joint_limit_margin"].default
    lower = position_bounds[0] + joint_limit_margin
    upper = position_bounds[1] - joint_limit_margin
    if torch.any(lower >= upper):
        raise ValueError("joint limits are too narrow for the IK safety margin")
    default = params.cspace.default_joint_position
    if default is None:
        default = 0.5 * (lower + upper)
    default = default.detach().to(dtype=torch.float32)
    unit_seeds = torch.as_tensor(
        HaltonSequencer(params.num_dof, seed=451).random(max_seeds),
        device=lower.device,
        dtype=torch.float32,
    )
    seeds = lower.unsqueeze(0) + unit_seeds * (upper - lower).unsqueeze(0)
    rollout = optimizer_profile["rollout"]
    cspace = rollout["cost_cfg"]["cspace_cfg"]
    metrics_profile = load_yaml(str(get_task_configs_path() / "metrics_base.yml"))
    start_cspace = metrics_profile["rollout"]["convergence_cfg"]["start_cspace_dist_cfg"]
    scene_collision = rollout["constraint_cfg"]["scene_collision_cfg"]
    self_collision = rollout["constraint_cfg"]["self_collision_cfg"]
    device = lower.device
    cspace_target_dof_weight = cspace.get("cspace_target_dof_weight")
    if cspace_target_dof_weight is None:
        cspace_target_dof_weight = torch.ones(
            params.num_dof,
            device=device,
            dtype=torch.float32,
        )
    else:
        cspace_target_dof_weight = torch.as_tensor(
            cspace_target_dof_weight,
            device=device,
            dtype=torch.float32,
        )
    optimization_step_max = optimizer.step_scale * torch.abs(
        position_bounds[1] - position_bounds[0]
    )
    tensors = {
        "joint_position_lower": lower,
        "joint_position_upper": upper,
        "default_joint_position": default,
        "seed_positions": seeds,
        "link_spheres": params.link_spheres,
        "sphere_collision_mask": params.sphere_collision_mask,
        "link_sphere_map": params.link_sphere_idx_map,
        "ik_position_bounds": position_bounds,
        "ik_velocity_bounds": params.joint_limits.velocity,
        "ik_cspace_weight": torch.as_tensor(cspace["weight"], device=device, dtype=torch.float32),
        "ik_cspace_activation_distance": torch.as_tensor(
            cspace["activation_distance"], device=device, dtype=torch.float32
        ),
        "ik_cspace_target_weight": torch.as_tensor(
            [cspace.get("cspace_target_weight", 0.0)],
            device=device,
            dtype=torch.float32,
        ),
        "ik_cspace_target_dof_weight": cspace_target_dof_weight,
        "ik_cspace_regularization_weight": torch.as_tensor(
            cspace.get("squared_l2_regularization_weight", [0.0, 0.0]),
            device=device,
            dtype=torch.float32,
        ),
        "ik_start_cspace_weight": torch.as_tensor(
            [start_cspace["weight"]], device=device, dtype=torch.float32
        ),
        "ik_start_cspace_dof_weight": params.cspace.cspace_distance_weight.detach().to(
            dtype=torch.float32
        ),
        "ik_world_collision_weight": torch.as_tensor(
            [scene_collision["weight"]], device=device, dtype=torch.float32
        ),
        "ik_self_collision_weight": torch.as_tensor(
            [self_collision["weight"]], device=device, dtype=torch.float32
        ),
        "ik_optimizer_collision_activation_distance": torch.as_tensor(
            [scene_collision["activation_distance"]],
            device=device,
            dtype=torch.float32,
        ),
        "ik_line_search_scale": torch.as_tensor(
            optimizer.line_search_scale, device=device, dtype=torch.float32
        ),
        "ik_optimization_step_max": optimization_step_max,
    }
    self_collision = config.self_collision_config
    if self_collision is not None and self_collision.collision_pairs is not None:
        tensors["self_collision_padding"] = self_collision.sphere_padding
        tensors["self_collision_pairs"] = self_collision.collision_pairs
    return tensors


def build_trajopt_config(
    robot_config: Path, max_seeds: int, interpolation_buffer_size: int = 2000
) -> tuple[Any, Any, Any]:
    """Load the current default B-spline TrajOpt configuration."""
    ensure_curobo_on_path()
    from curobo._src.solver.solver_trajopt_cfg import TrajOptSolverCfg
    from curobo._src.types.control_space import ControlSpace

    config = TrajOptSolverCfg.create(
        str(robot_config.resolve()),
        num_seeds=max_seeds,
        use_cuda_graph=True,
        # Native MPC shares these kernels and validates slower timing lanes at <=4 ms.
        interpolation_buffer_size=interpolation_buffer_size,
    )
    rollout = config.optimizer_rollout_configs[0]
    optimizer = config.optimizer_configs[0]
    transition = rollout.transition_model_cfg
    if transition.control_space != ControlSpace.BSPLINE_3:
        raise ValueError("C++ TrajOpt currently requires the default cubic B-spline")
    if optimizer.line_search_type.value != "approx_wolfe":
        raise ValueError("C++ TrajOpt currently requires approximate-Wolfe line search")
    if len(optimizer.line_search_scale) < 1 or len(optimizer.line_search_scale) > 32:
        raise ValueError("C++ TrajOpt requires one through 32 line-search probes")
    return config, rollout, optimizer


def build_trajopt_tensors(
    params: Any, config: Any, rollout: Any, optimizer: Any
) -> dict[str, torch.Tensor]:
    """Collect current state-cost and optimizer constants for native TrajOpt."""
    limits = params.joint_limits
    cspace = rollout.cost_cfg.cspace_cfg
    tool_pose = rollout.cost_cfg.tool_pose_cfg
    scene_collision = rollout.constraint_cfg.scene_collision_cfg
    self_collision = rollout.constraint_cfg.self_collision_cfg
    metrics_scene_collision = config.metrics_rollout_config.constraint_cfg.scene_collision_cfg
    metrics_self_collision = config.metrics_rollout_config.constraint_cfg.self_collision_cfg
    optimizer_interpolation_steps = rollout.transition_model_cfg.interpolation_steps
    metrics_interpolation_steps = (
        config.metrics_rollout_config.transition_model_cfg.interpolation_steps
    )
    optimizer_self_collision_weight = self_collision.weight / optimizer_interpolation_steps
    metrics_self_collision_weight = metrics_self_collision.weight / metrics_interpolation_steps
    # SolverCore constructs two metrics RobotRollouts before TrajOptSolver adds
    # the dense-validation rollout. Each RobotRollout constructs two cost
    # managers, and the self-collision manager scales its shared configuration
    # once per construction. The dense rollout therefore observes the fifth
    # interpolation scaling of the metrics weight.
    validation_self_collision_weight = metrics_self_collision.weight / (
        metrics_interpolation_steps**5
    )
    target_dof_weight = cspace.cspace_target_dof_weight
    if target_dof_weight.numel() == 0:
        target_dof_weight = torch.ones(
            params.num_dof,
            device=limits.position.device,
            dtype=torch.float32,
        )
    n_knots = rollout.transition_model_cfg.n_knots
    step_max = (optimizer.step_scale * torch.abs(limits.position[1] - limits.position[0])).repeat(
        n_knots
    )
    return {
        "position_bounds": limits.position,
        "velocity_bounds": limits.velocity,
        "acceleration_bounds": limits.acceleration,
        "jerk_bounds": limits.jerk,
        "state_bound_weight": cspace.weight,
        "state_activation_distance": cspace.activation_distance,
        "state_regularization_weight": cspace.squared_l2_regularization_weight,
        "cspace_target_weight": cspace.cspace_target_weight,
        "cspace_non_terminal_weight": cspace.cspace_non_terminal_weight_factor,
        "cspace_target_dof_weight": target_dof_weight,
        "tool_pose_weight": tool_pose.weight,
        "world_collision_weight": scene_collision.weight,
        "self_collision_weight": optimizer_self_collision_weight,
        "metrics_world_collision_weight": metrics_scene_collision.weight,
        "metrics_self_collision_weight": metrics_self_collision_weight,
        "validation_self_collision_weight": validation_self_collision_weight,
        "line_search_scale": torch.as_tensor(
            optimizer.line_search_scale,
            device=limits.position.device,
            dtype=torch.float32,
        ),
        "optimization_step_max": step_max,
    }


def build_rollout_artifact(
    output_dir: Path,
    arch: str,
    params: Any,
    profile_max_states: dict[str, int],
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]], dict[str, int]]:
    """Export one mutable RNEA model and solver-specific launch profiles."""
    import curobo._src.runtime as curobo_runtime
    from curobo._src.curobolib.backends.cuda_core_backend.dynamics_config import (
        DynamicsKernelCfg,
        DynamicsLaunchCfg,
    )
    from curobo._src.robot.dynamics.dynamics import (
        _collapse_fixed_bodies,
        _compute_runtime_threads_per_batch,
    )
    from curobo._src.robot.dynamics.dynamics_cfg import DynamicsCfg

    model = _collapse_fixed_bodies(params)
    properties = torch.cuda.get_device_properties(params.fixed_transforms.device)
    policy_sm_count = curobo_runtime.resolve_kernel_launch_sm_count(
        properties.multi_processor_count
    )

    def bool_literal(value: bool) -> str:
        return str(value).lower()

    kernel_config = DynamicsKernelCfg()
    tensors = {
        "rnea_fixed_transforms": model.fixed_transforms,
        "rnea_link_masses_com": model.link_masses_com,
        "rnea_link_inertias": model.link_inertias,
        "rnea_original_link_masses_com": params.link_masses_com,
        "rnea_original_link_inertias": params.link_inertias,
        "rnea_external_force_owners": model.external_force_owners,
        "rnea_external_force_transforms": model.external_force_transforms,
        "rnea_effort_bounds": params.joint_limits.effort,
        "rnea_joint_map_type": model.joint_map_type,
        "rnea_joint_map": model.joint_map,
        "rnea_link_map": model.link_map,
        "rnea_chain_map": model.chain_map,
        "rnea_joint_offset_map": model.joint_offset_map,
        "rnea_gravity": DynamicsCfg(
            kinematics_config=params,
            device_cfg=params.device_cfg,
        ).get_gravity_spatial(),
        "rnea_level_starts": model.level_starts,
        "rnea_level_links": model.level_links,
    }
    profiles: dict[str, dict[str, Any]] = {}
    for profile_name, max_states in profile_max_states.items():
        threads_per_batch = _compute_runtime_threads_per_batch(
            model.max_level_width,
            model.num_links,
            model.num_levels,
            max_states,
            policy_sm_count,
        )
        template_suffix = (
            f"<{model.num_links}, {params.num_dof}, {threads_per_batch}, false, "
            f"{bool_literal(model.all_movable)}, {bool_literal(model.direct_joints)}, false>"
        )
        forward_symbol = f"curobo::dynamics::rnea_forward_kernel{template_suffix}"
        backward_symbol = f"curobo::dynamics::rnea_backward_kernel{template_suffix}"
        forward_cubin, forward_symbols = _compile_cuda_specializations(
            output_dir,
            arch,
            f"{profile_name}_rnea_forward",
            kernel_config,
            ["forward"],
            [forward_symbol],
        )
        backward_cubin, backward_symbols = _compile_cuda_specializations(
            output_dir,
            arch,
            f"{profile_name}_rnea_backward",
            kernel_config,
            ["backward"],
            [backward_symbol],
        )
        forward_config = DynamicsLaunchCfg.calculate_forward_config(
            max_states,
            model.num_links,
            threads_per_batch,
            policy_sm_count=policy_sm_count,
        )
        backward_config = DynamicsLaunchCfg.calculate_backward_config(
            max_states,
            model.num_links,
            threads_per_batch,
            policy_sm_count=policy_sm_count,
        )
        profiles[profile_name] = {
            "threads_per_batch": threads_per_batch,
            "cache_floats_per_link": 32 if threads_per_batch > 1 else 20,
            "kernels": {
                "forward": {
                    "cubin": forward_cubin,
                    "symbol": forward_symbols.get(forward_symbol, forward_symbol),
                    "requested_symbol": forward_symbol,
                    "source": "curobo_rnea_kernel",
                    "shared_memory": int(forward_config.shmem_size),
                    "block_dim": int(forward_config.block[0]),
                },
                "backward": {
                    "cubin": backward_cubin,
                    "symbol": backward_symbols.get(backward_symbol, backward_symbol),
                    "requested_symbol": backward_symbol,
                    "source": "curobo_rnea_kernel",
                    "shared_memory": int(backward_config.shmem_size),
                    "block_dim": int(backward_config.block[0]),
                },
            },
        }
    metadata = {
        "num_links": model.num_links,
        "num_original_links": model.num_original_links,
        "num_levels": model.num_levels,
        "level_link_indices": int(model.level_links.numel()),
    }
    return tensors, profiles, metadata


def build_rrt_config(
    robot_config: Path,
    samples_per_iteration: int,
    max_iterations: int,
    max_nodes_per_tree: int,
) -> tuple[Any, Any]:
    """Resolve the current uniform bidirectional-RRT deployment profile."""
    ensure_curobo_on_path()
    from curobo._src.graph_planner.config_graph_planner_resources import (
        GraphPlannerResourcesCfg,
    )
    from curobo._src.graph_planner.rrt.planner_bi_rrt import BiRRTConfig

    resources = GraphPlannerResourcesCfg.create(str(robot_config.resolve()))
    planner = BiRRTConfig(
        samples_per_iteration=samples_per_iteration,
        max_iterations=max_iterations,
        failure_extension_iterations=0,
        max_nodes_per_tree=max_nodes_per_tree,
        sampling_policy="uniform",
    )
    return resources, planner


def resolve_rrt_direct_steer_steps(params: Any, resources: Any, planner: Any) -> int:
    """Return fixed storage sufficient for any in-bounds complete direct edge."""
    joint_range = params.joint_limits.position[1] - params.joint_limits.position[0]
    steps = int(torch.ceil(torch.max(torch.abs(joint_range)) / planner.cspace_resolution)) + 1
    if steps > resources.steer_buffer_size:
        raise ValueError(
            "graph-planner steer buffer is too small for a complete native direct edge"
        )
    return max(2, steps)


def build_rrt_tensors(params: Any, resources: Any, planner: Any) -> dict[str, torch.Tensor]:
    """Create Python-compatible buffered Halton samples and shortcut tables."""
    from curobo._src.graph_planner.rrt.rrt_path_shortcut import (
        _build_shortcut_pair_tables,
    )
    from curobo._src.util.sampling.sequencer_halton import HaltonSequencer

    lower = params.joint_limits.position[0].detach().to(dtype=torch.float32)
    upper = params.joint_limits.position[1].detach().to(dtype=torch.float32)
    sample_count = max(
        RRT_SAMPLE_BUFFER_SIZE,
        2 * MAX_RRT_PLANS * planner.samples_per_iteration,
    )
    unit_samples = torch.as_tensor(
        HaltonSequencer(params.num_dof, seed=resources.sampler_seed).random(sample_count),
        device=lower.device,
        dtype=torch.float32,
    )
    samples = lower.unsqueeze(0) + unit_samples * (upper - lower).unsqueeze(0)
    pairs, lookup = _build_shortcut_pair_tables(planner.shortcut_path_capacity)
    return {
        "rrt_position_lower": lower,
        "rrt_position_upper": upper,
        "rrt_distance_weight": torch.ones_like(lower),
        "rrt_samples": samples,
        "rrt_shortcut_pair_sources": torch.tensor(
            [source for source, _ in pairs], device=lower.device, dtype=torch.int32
        ),
        "rrt_shortcut_pair_destinations": torch.tensor(
            [destination for _, destination in pairs], device=lower.device, dtype=torch.int32
        ),
        "rrt_shortcut_edge_lookup": torch.tensor(lookup, device=lower.device, dtype=torch.int32),
    }


def export_rrt_kernels(
    output_dir: Path,
    arch: str,
    num_dof: int,
    planner: Any,
    direct_steer_steps: int,
    shortcut_state_capacity: int,
) -> tuple[dict[str, dict[str, Any]], int]:
    """Export current cuRobo tree kernels plus native-runtime RRT glue."""
    from curobo._src.graph_planner.rrt.kernel_rrt_runtime import (
        create_rrt_runtime_kernels,
    )
    from curobo._src.graph_planner.rrt.rrt_path_shortcut import (
        RRTPathShortcutterConfig,
        _make_shortcut_kernels,
    )
    from curobo._src.graph_planner.rrt.rrt_tree_kernel import (
        RRTTreeKernelConfig,
        _make_tree_kernels,
    )

    max_path_nodes = 2 * (planner.max_iterations + 2)
    profile = RRTTreeKernelConfig(
        action_dim=num_dof,
        max_candidates=planner.samples_per_iteration,
        max_path_nodes=max_path_nodes,
    )
    tree_bundle = _make_tree_kernels(
        profile.action_dim,
        profile.block_dim,
        profile.max_path_nodes,
    )
    kernels = {
        name: _export_warp_kernel(
            kernel,
            output_dir,
            f"rrt_{name}",
            arch,
            block_dim=profile.block_dim,
        )
        for name, kernel in {
            "nearest_target": tree_bundle.nearest_target,
            "packed_append": tree_bundle.packed_append,
            "reconstruct_path": tree_bundle.reconstruct_path,
        }.items()
    }
    glue = create_rrt_runtime_kernels(
        num_dof,
        planner.samples_per_iteration,
        planner.fixed_steer_steps,
        direct_steer_steps,
    )
    edge_block_dim = 32
    while edge_block_dim < max(num_dof, planner.fixed_steer_steps):
        edge_block_dim *= 2
    if edge_block_dim > 256:
        raise ValueError("native RRT steering supports at most 256 fixed edge steps")
    native_glue_names = (
        "prepare_samples",
        "steer_edges",
        "evaluate_edges",
        "prepare_direct_edges",
        "interpolate_direct_edges",
        "reduce_direct_edges",
        "classify_states",
        "select_connection",
    )
    kernels.update(
        {
            name: _export_warp_kernel(
                glue[name],
                output_dir,
                f"rrt_{name}",
                arch,
                block_dim=(edge_block_dim if name in {"steer_edges", "evaluate_edges"} else 128),
            )
            for name in native_glue_names
        }
    )
    shortcut_profile = RRTPathShortcutterConfig(
        action_dim=num_dof,
        path_capacity=planner.shortcut_path_capacity,
        raw_path_capacity=max_path_nodes,
        state_capacity=shortcut_state_capacity,
        cspace_resolution=planner.cspace_resolution,
    )
    shortcut_bundle = _make_shortcut_kernels(
        shortcut_profile.action_dim,
        shortcut_profile.path_capacity,
        int(shortcut_profile.raw_path_capacity),
        shortcut_profile.state_capacity,
        shortcut_profile.cspace_resolution,
        shortcut_profile.edge_capacity,
        shortcut_profile.block_dim,
    )
    for name, kernel in {
        "shortcut_setup": shortcut_bundle.setup,
        "shortcut_fill_edges": shortcut_bundle.fill_edges,
        "shortcut_reduce_edges": shortcut_bundle.reduce_edges,
        "shortcut_select_path": shortcut_bundle.select_path,
    }.items():
        kernels[name] = _export_warp_kernel(
            kernel,
            output_dir,
            f"rrt_{name}",
            arch,
            block_dim=shortcut_profile.block_dim,
        )
    return kernels, max_path_nodes


def build_motion_fit_tensors(
    trajopt_rollout: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, float | int]]:
    """Derive the production fitter's exact fixed-shape constants."""
    from curobo._src.trajectory.trajectory_seed_fitter import (
        BSplineTimeOptimalSeedFitter,
    )
    from curobo._src.transition.robot_state_transition import RobotStateTransition

    transition = RobotStateTransition(trajopt_rollout.transition_model_cfg)
    fitter = BSplineTimeOptimalSeedFitter.from_transition_model(
        transition,
        num_lanes=1,
        use_cuda_graph=False,
    )
    fitter._initialize_normal_operator()
    if (
        fitter._normal_inverse is None
        or fitter._basis_position is None
        or fitter._basis_velocity is None
        or fitter._basis_acceleration is None
        or fitter._basis_jerk is None
        or fitter._path_normal is None
        or fitter._path_normal_trace is None
        or fitter._path_diagonal_scale is None
        or fitter._fit_matrix is None
        or fitter._start_position_coefficient is None
        or fitter._start_velocity_coefficient is None
        or fitter._start_acceleration_coefficient is None
        or fitter._start_jerk_coefficient is None
        or fitter._goal_position_coefficient is None
    ):
        raise RuntimeError("failed to initialize the B-spline fit operator")

    tensors = {
        "motion_arc_progress": fitter._arc_progress,
        "motion_fit_matrix": fitter._fit_matrix,
        "motion_start_coefficient": fitter._start_position_coefficient,
        "motion_start_velocity_coefficient": fitter._start_velocity_coefficient,
        "motion_start_acceleration_coefficient": fitter._start_acceleration_coefficient,
        "motion_start_jerk_coefficient": fitter._start_jerk_coefficient,
        "motion_goal_coefficient": fitter._goal_position_coefficient,
        "motion_basis_position": fitter._basis_position,
        "motion_basis_velocity": fitter._basis_velocity,
        "motion_basis_acceleration": fitter._basis_acceleration,
        "motion_basis_jerk": fitter._basis_jerk,
        "motion_path_normal": fitter._path_normal,
        "motion_max_velocity": fitter._max_velocity,
        "motion_max_acceleration": fitter._max_acceleration,
        "motion_max_jerk": fitter._max_jerk,
    }
    config = fitter.config
    metadata: dict[str, float | int] = {
        "irls_iterations": config.irls_iterations,
        "regularization": config.regularization,
        "timing_regularization": config.timing_regularization,
        "trust_region_regularization": config.trust_region_regularization,
        "irls_epsilon": config.irls_epsilon,
        "dt_epsilon": config.dt_epsilon,
        "trajectory_dt": config.trajectory_dt,
        "path_normal_trace": float(fitter._path_normal_trace.item()),
        "path_diagonal_scale": float(fitter._path_diagonal_scale.item()),
    }
    return tensors, metadata


def export_motion_kernels(
    output_dir: Path,
    arch: str,
    num_dof: int,
    num_knots: int,
    trajectory_steps: int,
    max_path_nodes: int,
    num_seeds: int,
) -> dict[str, dict[str, Any]]:
    """Export graph-path fitting glue derived from the Python motion planner."""
    from cpp_runtime.robot.kernels.motion import create_motion_runtime_kernels

    kernels = create_motion_runtime_kernels(
        num_dof,
        num_knots,
        trajectory_steps,
        max_path_nodes,
        num_seeds,
    )
    return {
        name: _export_warp_kernel(
            kernel,
            output_dir,
            f"motion_{name}",
            arch,
            block_dim=128,
        )
        for name, kernel in kernels.items()
    }


def compile_trajopt_self_collision_kernel(
    output_dir: Path,
    arch: str,
    self_collision_config: Any,
    num_spheres: int,
) -> dict[str, Any] | None:
    """Compile the shared self-collision specialization for TrajOpt."""
    return compile_self_collision_kernel(output_dir, arch, self_collision_config, num_spheres)


def export_trajopt_kernels(
    output_dir: Path,
    arch: str,
    params: Any,
    kinematics_kernel: dict[str, Any],
    kinematics_backward_kernel: dict[str, Any],
    trajopt_config: Any,
    rollout: Any,
    optimizer: Any,
    scene_capacities: dict[str, int],
    max_seeds: int,
) -> dict[str, dict[str, Any]]:
    """Export current cuRobo kernels used by fixed-dt joint-space TrajOpt."""
    from cpp_runtime.robot.kernels.trajopt import (
        create_trajopt_runtime_kernels,
    )
    from curobo._src.cost.wp_cspace_state import forward_cspace_state_warp
    from curobo._src.cost.wp_tool_pose import (
        create_goalset_pose_distance_kernel_with_constants,
    )
    from curobo._src.curobolib.backends.cuda_core_backend.trajectory_config import (
        TrajectoryKernelCfg,
    )
    from curobo._src.geom.collision.wp_collision_kernel import (
        get_tiled_collision_kernel,
    )
    from curobo._src.geom.collision.wp_sweep_collision_kernel import (
        get_tiled_swept_collision_kernel,
    )
    from curobo._src.geom.data.data_capsule import CapsuleDataWarp
    from curobo._src.geom.data.data_cuboid import CuboidDataWarp
    from curobo._src.geom.data.data_cylinder import CylinderDataWarp
    from curobo._src.geom.data.data_mesh import MeshDataWarp
    from curobo._src.geom.data.data_voxel import VoxelDataWarp
    from curobo._src.util.warp import init_warp

    init_warp()
    transition = rollout.transition_model_cfg
    num_dof = params.num_dof
    num_spheres = params.total_spheres
    n_knots = transition.n_knots
    horizon = transition.horizon
    degree = 3
    opt_dim = n_knots * num_dof
    num_line_search = len(optimizer.line_search_scale)

    kernels: dict[str, dict[str, Any]] = {
        "kinematics": kinematics_kernel,
        "kinematics_backward": kinematics_backward_kernel,
    }

    forward_symbol = (
        "curobo::trajectory::bspline::interpolate_bspline_kernel"
        f"<float, {degree}, (curobo::trajectory::bspline::BasisBackend)2>"
    )
    backward_symbol = (
        "curobo::trajectory::bspline::bspline_backward_kernel"
        f"<{degree}, float, (curobo::trajectory::bspline::BasisBackend)2>"
    )
    time_scaled_dt_symbol = (
        "curobo::trajectory::bspline::bspline_time_scaled_dt_kernel"
        f"<float, {degree}, (curobo::trajectory::bspline::BasisBackend)2>"
    )
    single_dt_symbol = (
        "curobo::trajectory::bspline::interpolate_bspline_single_dt_kernel"
        f"<float, {degree}, (curobo::trajectory::bspline::BasisBackend)2>"
    )
    cubin_name, symbols = _compile_cuda_specializations(
        output_dir,
        arch,
        "trajopt_bspline",
        TrajectoryKernelCfg(),
        [
            "bspline_forward",
            "bspline_backward",
            "bspline_time_scaled_dt",
            "bspline_single_dt",
        ],
        [forward_symbol, backward_symbol, time_scaled_dt_symbol, single_dt_symbol],
    )
    kernels["bspline_forward"] = {
        "cubin": cubin_name,
        "symbol": symbols.get(forward_symbol, forward_symbol),
        "source": "curobo_bspline_kernel",
        "shared_memory": 0,
        "block_dim": 128,
    }
    kernels["bspline_backward"] = {
        "cubin": cubin_name,
        "symbol": symbols.get(backward_symbol, backward_symbol),
        "source": "curobo_bspline_kernel",
        "shared_memory": 0,
        "block_dim": 128,
    }
    kernels["time_scaled_dt"] = {
        "cubin": cubin_name,
        "symbol": symbols.get(time_scaled_dt_symbol, time_scaled_dt_symbol),
        "source": "curobo_bspline_kernel",
        "shared_memory": 0,
        "block_dim": 128,
    }
    kernels["bspline_single_dt"] = {
        "cubin": cubin_name,
        "symbol": symbols.get(single_dt_symbol, single_dt_symbol),
        "source": "curobo_bspline_kernel",
        "shared_memory": 0,
        "block_dim": 256,
    }

    kernels.update(
        compile_lbfgs_kernels(
            output_dir,
            arch,
            "trajopt",
            opt_dim,
            max_seeds,
            optimizer,
        )
    )

    kernels["cspace"] = _export_warp_kernel(
        forward_cspace_state_warp,
        output_dir,
        "trajopt_cspace",
        arch,
        block_dim=128,
    )
    kernels["pose"] = _export_warp_kernel(
        create_goalset_pose_distance_kernel_with_constants(1),
        output_dir,
        "trajopt_pose",
        arch,
        block_dim=128,
    )
    glue_kernels = create_trajopt_runtime_kernels(
        num_dof,
        n_knots,
        horizon,
        trajopt_config.interpolation_buffer_size,
        max_seeds,
        num_line_search,
    )
    kernels.update(
        {
            name: _export_warp_kernel(
                glue_kernel,
                output_dir,
                f"trajopt_{name}",
                arch,
                block_dim=128,
            )
            for name, glue_kernel in glue_kernels.items()
        }
    )

    if num_spheres > 0:
        obstacle_types = {
            "cuboid": CuboidDataWarp,
            "capsule": CapsuleDataWarp,
            "cylinder": CylinderDataWarp,
            "mesh": MeshDataWarp,
            "voxel": VoxelDataWarp,
        }
        block_dim = _collision_block_dim(num_spheres)
        for name, obstacle_type in obstacle_types.items():
            capacity = scene_capacities[name]
            if capacity <= 0:
                continue
            collision_kernel = get_tiled_swept_collision_kernel(
                capacity, True, horizon, num_spheres, block_dim
            )
            overload = wp.overload(
                collision_kernel,
                [
                    obstacle_type,
                    wp.array(dtype=wp.vec4),
                    wp.array(dtype=wp.float32),
                    wp.array(dtype=wp.float32),
                    wp.array(dtype=wp.float32),
                    wp.array(dtype=wp.int32),
                    wp.array(dtype=wp.float32),
                    wp.array(dtype=wp.float32),
                    wp.uint8,
                    wp.uint8,
                    wp.uint8,
                    wp.array(dtype=wp.uint32),
                    wp.array2d(dtype=wp.uint32),
                ],
            )
            metadata = _export_warp_kernel(
                overload,
                output_dir,
                f"trajopt_collision_{name}",
                arch,
                strip_hash=False,
                block_dim=block_dim,
            )
            metadata["max_obstacles"] = capacity
            kernels[f"collision_{name}"] = metadata

            metrics_collision_kernel = get_tiled_collision_kernel(
                capacity, horizon, num_spheres, block_dim
            )
            metrics_overload = wp.overload(
                metrics_collision_kernel,
                [
                    obstacle_type,
                    wp.array(dtype=wp.vec4),
                    wp.array(dtype=wp.float32),
                    wp.array(dtype=wp.float32),
                    wp.array(dtype=wp.int32),
                    wp.array(dtype=wp.float32),
                    wp.array(dtype=wp.float32),
                    wp.uint8,
                    wp.uint8,
                    wp.uint8,
                    wp.array(dtype=wp.uint32),
                    wp.array2d(dtype=wp.uint32),
                ],
            )
            metrics_metadata = _export_warp_kernel(
                metrics_overload,
                output_dir,
                f"trajopt_metrics_collision_{name}",
                arch,
                strip_hash=False,
                block_dim=block_dim,
            )
            metrics_metadata["max_obstacles"] = capacity
            kernels[f"metrics_collision_{name}"] = metrics_metadata

            validation_collision_kernel = get_tiled_collision_kernel(
                capacity,
                trajopt_config.interpolation_buffer_size,
                num_spheres,
                block_dim,
            )
            validation_overload = wp.overload(
                validation_collision_kernel,
                [
                    obstacle_type,
                    wp.array(dtype=wp.vec4),
                    wp.array(dtype=wp.float32),
                    wp.array(dtype=wp.float32),
                    wp.array(dtype=wp.int32),
                    wp.array(dtype=wp.float32),
                    wp.array(dtype=wp.float32),
                    wp.uint8,
                    wp.uint8,
                    wp.uint8,
                    wp.array(dtype=wp.uint32),
                    wp.array2d(dtype=wp.uint32),
                ],
            )
            validation_metadata = _export_warp_kernel(
                validation_overload,
                output_dir,
                f"trajopt_validation_collision_{name}",
                arch,
                strip_hash=False,
                block_dim=block_dim,
            )
            validation_metadata["max_obstacles"] = capacity
            kernels[f"validation_collision_{name}"] = validation_metadata

    self_collision_kernel = compile_trajopt_self_collision_kernel(
        output_dir,
        arch,
        trajopt_config.robot_config.kinematics.self_collision_config,
        num_spheres,
    )
    if self_collision_kernel is not None:
        kernels["self_collision"] = self_collision_kernel
    return kernels


def export_robot(args: argparse.Namespace) -> Path:
    """Export the selected robot into an artifact directory."""
    output_dir = args.out.resolve()
    selected_robot_config = args.robot_config
    if args.attachment_parent_link is not None:
        if selected_robot_config is None:
            raise ValueError("attachment storage requires --robot-config")
        selected_robot_config = prepare_attachment_robot_config(
            selected_robot_config,
            output_dir,
            args.attachment_parent_link,
            args.attachment_spheres,
        )
    config, source = build_kinematics_config(args, selected_robot_config)
    if args.attachment_parent_link is not None:
        source["input_robot_config"] = str(args.robot_config.resolve())
        source["attachment"] = {
            "link_name": "attached_object",
            "parent_link_name": args.attachment_parent_link,
            "sphere_capacity": args.attachment_spheres,
        }
    params = config.kinematics_config
    expand_joint_position_limits(params, args.joint_limit_expansion)
    params.validate_shapes()
    if params.num_dof <= 0:
        raise ValueError("robot must have at least one active degree of freedom")
    if params.num_pose_links <= 0:
        raise ValueError("robot must have at least one tool frame")
    link_names = [""] * params.num_links
    for link_name, link_index in params.link_name_to_idx_map.items():
        link_names[link_index] = link_name
    if any(not link_name for link_name in link_names):
        raise ValueError("robot link-name mapping does not cover every link index")

    output_dir.mkdir(parents=True, exist_ok=True)
    tensors = {
        name: export_tensor(name, getattr(params, name), output_dir) for name in TENSOR_SPECS
    }
    arch, kernel, ik_kinematics_kernel, trajopt_kinematics_kernel = compile_kinematics_kernel(
        output_dir,
        num_links=params.num_links,
        num_dof=params.num_dof,
        num_tool_frames=params.num_pose_links,
        num_spheres=params.total_spheres,
    )
    manifest = {
        "schema_version": 1,
        "arch": arch,
        "curobo_source_dir": str(ensure_curobo_on_path()),
        "source": source,
        "base_link": params.base_link,
        "num_links": params.num_links,
        "num_dof": params.num_dof,
        "num_tool_frames": params.num_pose_links,
        "joint_position_limit_expansion_radians": args.joint_limit_expansion,
        "link_names": link_names,
        "joint_names": params.joint_names,
        "tool_frames": params.tool_frames,
        "tensors": tensors,
        "kernel": kernel,
    }
    manifest_path = output_dir / "robot_kinematics_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if not args.kinematics_only:
        scene_capacities = {
            "cuboid": args.max_cuboids,
            "capsule": args.max_capsules,
            "cylinder": args.max_cylinders,
            "mesh": args.max_meshes,
            "voxel": args.max_voxel_grids,
        }
        ik_optimizer_profile, ik_optimizer = build_ik_optimizer_config(params)
        ik_tensor_values = build_ik_tensors(
            config,
            params,
            args.ik_seed_samples,
            ik_optimizer_profile,
            ik_optimizer,
        )
        ik_tensors = {
            name: export_tensor(name, value, output_dir)
            for name, value in ik_tensor_values.items()
        }
        ik_kernels, num_residuals = export_ik_kernels(
            output_dir,
            arch,
            params,
            ik_optimizer,
            args.max_ik_seeds,
            scene_capacities,
        )
        kinematics_backward_kernel = compile_kinematics_backward_kernel(
            output_dir,
            arch,
            params,
        )
        ik_kernels["kinematics"] = ik_kinematics_kernel
        ik_kernels["kinematics_backward"] = kinematics_backward_kernel
        self_collision_kernel = compile_self_collision_kernel(
            output_dir, arch, config.self_collision_config, params.total_spheres
        )
        if self_collision_kernel is not None:
            ik_kernels["self_collision"] = self_collision_kernel
        ik_manifest = {
            "schema_version": 13,
            "arch": arch,
            "num_links": params.num_links,
            "num_dof": params.num_dof,
            "num_tool_frames": params.num_pose_links,
            "num_spheres": params.total_spheres,
            "num_residuals": num_residuals,
            "max_seeds": args.max_ik_seeds,
            "num_seed_samples": args.ik_seed_samples,
            "max_goalset": args.max_ik_goalset,
            "num_self_collision_pairs": (
                0 if self_collision_kernel is None else self_collision_kernel["num_pairs"]
            ),
            "history": ik_optimizer.history,
            "num_line_search": len(ik_optimizer.line_search_scale),
            "optimizer": {
                "num_iterations": ik_optimizer.num_iters,
                "epsilon": ik_optimizer.epsilon,
                "initial_step_scale": ik_optimizer.initial_step_scale,
                "armijo_threshold": ik_optimizer.line_search_wolfe_c_1,
                "curvature_threshold": ik_optimizer.line_search_wolfe_c_2,
                "convergence_iteration": ik_optimizer.convergence_iteration,
                "cost_delta_threshold": ik_optimizer.cost_delta_threshold,
                "cost_relative_threshold": ik_optimizer.cost_relative_threshold,
                "position_weight": ik_optimizer_profile["rollout"]["cost_cfg"]["tool_pose_cfg"][
                    "weight"
                ][0],
                "orientation_weight": ik_optimizer_profile["rollout"]["cost_cfg"]["tool_pose_cfg"][
                    "weight"
                ][1],
            },
            "scene_capacities": scene_capacities,
            "tensors": ik_tensors,
            "kernels": ik_kernels,
        }
        (output_dir / "robot_ik_manifest.json").write_text(
            json.dumps(ik_manifest, indent=2, sort_keys=True) + "\n"
        )
        if selected_robot_config is not None:
            trajopt_config, trajopt_rollout, trajopt_optimizer = build_trajopt_config(
                selected_robot_config, args.max_trajopt_seeds, args.interpolation_buffer_size
            )
            transition = trajopt_rollout.transition_model_cfg
            if transition.robot_config.kinematics.kinematics_config.num_dof != params.num_dof:
                raise ValueError("TrajOpt and kinematics robot dimensions do not match")
            trajopt_tensor_values = build_trajopt_tensors(
                params, trajopt_config, trajopt_rollout, trajopt_optimizer
            )
            from cpp_runtime.robot.tools.export_mpc_weights import export_mpc_weights

            export_mpc_weights(output_dir)
            rrt_resources, rrt_planner = build_rrt_config(
                selected_robot_config,
                args.rrt_samples_per_iteration,
                args.rrt_max_iterations,
                args.rrt_max_nodes_per_tree,
            )
            rrt_direct_steer_steps = resolve_rrt_direct_steer_steps(
                params, rrt_resources, rrt_planner
            )
            rrt_shortcut_state_capacity = (
                rrt_resources.feasibility_buffer_size
                if rrt_planner.shortcut_state_capacity is None
                else rrt_planner.shortcut_state_capacity
            )
            rrt_rollout_states = max(
                MAX_RRT_PLANS * rrt_planner.samples_per_iteration * rrt_planner.fixed_steer_steps,
                2 * MAX_RRT_PLANS * rrt_direct_steer_steps,
                rrt_shortcut_state_capacity,
            )
            rnea_tensor_values, rnea_profiles, rnea_metadata = build_rollout_artifact(
                output_dir,
                arch,
                params,
                {
                    "ik": args.max_ik_seeds * len(ik_optimizer.line_search_scale),
                    "rrt": rrt_rollout_states,
                    "trajopt": args.max_trajopt_seeds
                    * len(trajopt_optimizer.line_search_scale)
                    * transition.horizon,
                },
            )
            rollout_tensors = {
                name.removeprefix("rnea_"): export_tensor(name, value, output_dir)
                for name, value in rnea_tensor_values.items()
            }
            rollout_manifest = {
                "schema_version": 1,
                "arch": arch,
                "num_dof": params.num_dof,
                **rnea_metadata,
                "tensors": rollout_tensors,
                "profiles": rnea_profiles,
            }
            (output_dir / "robot_rollout_manifest.json").write_text(
                json.dumps(rollout_manifest, indent=2, sort_keys=True) + "\n"
            )
            trajopt_tensors = {
                name: export_tensor(name, value, output_dir)
                for name, value in trajopt_tensor_values.items()
            }
            for name in (
                "link_spheres",
                "sphere_collision_mask",
                "link_sphere_map",
                "self_collision_padding",
                "self_collision_pairs",
            ):
                if name in ik_tensors:
                    trajopt_tensors[name] = ik_tensors[name]
            trajopt_kernels = export_trajopt_kernels(
                output_dir,
                arch,
                params,
                trajopt_kinematics_kernel,
                kinematics_backward_kernel,
                trajopt_config,
                trajopt_rollout,
                trajopt_optimizer,
                scene_capacities,
                args.max_trajopt_seeds,
            )
            trajopt_manifest = {
                "schema_version": 18,
                "arch": arch,
                "num_dof": params.num_dof,
                "num_links": params.num_links,
                "num_tool_frames": params.num_pose_links,
                "num_spheres": params.total_spheres,
                "num_self_collision_pairs": (
                    0
                    if "self_collision" not in trajopt_kernels
                    else trajopt_kernels["self_collision"]["num_pairs"]
                ),
                "num_knots": transition.n_knots,
                "trajectory_steps": transition.horizon,
                "spline_degree": 3,
                "max_seeds": args.max_trajopt_seeds,
                "history": trajopt_optimizer.history,
                "num_line_search": len(trajopt_optimizer.line_search_scale),
                "base_dt": transition.dt_traj_params.base_dt,
                "minimum_dt": trajopt_config.minimum_trajectory_dt,
                "maximum_dt": trajopt_config.maximum_trajectory_dt,
                "interpolation_dt": trajopt_config.interpolation_dt,
                "interpolation_buffer_size": trajopt_config.interpolation_buffer_size,
                "optimizer": {
                    "epsilon": trajopt_optimizer.epsilon,
                    "initial_step_scale": trajopt_optimizer.initial_step_scale,
                    "armijo_threshold": trajopt_optimizer.line_search_wolfe_c_1,
                    "curvature_threshold": trajopt_optimizer.line_search_wolfe_c_2,
                    "convergence_iteration": trajopt_optimizer.convergence_iteration,
                    "cost_delta_threshold": trajopt_optimizer.cost_delta_threshold,
                    "cost_relative_threshold": trajopt_optimizer.cost_relative_threshold,
                },
                "scene_capacities": scene_capacities,
                "tensors": trajopt_tensors,
                "kernels": trajopt_kernels,
            }
            (output_dir / "robot_trajopt_manifest.json").write_text(
                json.dumps(trajopt_manifest, indent=2, sort_keys=True) + "\n"
            )
            rrt_tensor_values = build_rrt_tensors(params, rrt_resources, rrt_planner)
            rrt_tensors = {
                name.removeprefix("rrt_"): export_tensor(name, value, output_dir)
                for name, value in rrt_tensor_values.items()
            }
            for name in (
                "link_spheres",
                "sphere_collision_mask",
                "link_sphere_map",
                "self_collision_padding",
                "self_collision_pairs",
            ):
                if name in ik_tensors:
                    rrt_tensors[name] = ik_tensors[name]
            rrt_kernels, max_path_nodes = export_rrt_kernels(
                output_dir,
                arch,
                params.num_dof,
                rrt_planner,
                rrt_direct_steer_steps,
                rrt_shortcut_state_capacity,
            )
            rrt_kernels["kinematics"] = trajopt_kinematics_kernel
            for name in (
                "collision_cuboid",
                "collision_capsule",
                "collision_cylinder",
                "collision_mesh",
                "collision_voxel",
                "self_collision",
            ):
                if name in ik_kernels:
                    rrt_kernels[name] = ik_kernels[name]
            rrt_manifest = {
                "schema_version": 8,
                "arch": arch,
                "num_dof": params.num_dof,
                "num_links": params.num_links,
                "num_tool_frames": params.num_pose_links,
                "num_spheres": params.total_spheres,
                "num_self_collision_pairs": (
                    0
                    if "self_collision" not in rrt_kernels
                    else rrt_kernels["self_collision"]["num_pairs"]
                ),
                "samples_per_iteration": rrt_planner.samples_per_iteration,
                "max_iterations": rrt_planner.max_iterations,
                "max_nodes_per_tree": rrt_planner.max_nodes_per_tree,
                "fixed_steer_steps": rrt_planner.fixed_steer_steps,
                "direct_steer_steps": rrt_direct_steer_steps,
                "max_path_nodes": max_path_nodes,
                "sample_buffer_size": RRT_SAMPLE_BUFFER_SIZE,
                "max_batched_lanes": MAX_RRT_PLANS,
                "shortcut_path_capacity": rrt_planner.shortcut_path_capacity,
                "shortcut_state_capacity": rrt_shortcut_state_capacity,
                "shortcut_edge_capacity": (
                    (rrt_planner.shortcut_path_capacity - 1)
                    * (rrt_planner.shortcut_path_capacity - 2)
                    // 2
                ),
                "cspace_resolution": rrt_planner.cspace_resolution,
                "max_extension_distance": rrt_planner.max_extension_distance,
                "similarity_threshold": rrt_resources.cspace_similarity_threshold,
                "sampler_seed": rrt_resources.sampler_seed,
                "scene_capacities": scene_capacities,
                "tensors": rrt_tensors,
                "kernels": rrt_kernels,
            }
            (output_dir / "robot_rrt_manifest.json").write_text(
                json.dumps(rrt_manifest, indent=2, sort_keys=True) + "\n"
            )
            motion_tensor_values, motion_fit_metadata = build_motion_fit_tensors(trajopt_rollout)
            motion_tensors = {
                name.removeprefix("motion_"): export_tensor(name, value, output_dir)
                for name, value in motion_tensor_values.items()
            }
            motion_kernels = export_motion_kernels(
                output_dir,
                arch,
                params.num_dof,
                transition.n_knots,
                transition.horizon,
                max_path_nodes,
                args.max_trajopt_seeds,
            )
            motion_manifest = {
                "schema_version": 10,
                "arch": arch,
                "num_dof": params.num_dof,
                "num_knots": transition.n_knots,
                "trajectory_steps": transition.horizon,
                "interpolation_steps": transition.interpolation_steps,
                "max_rrt_path_nodes": max_path_nodes,
                "max_trajopt_seeds": args.max_trajopt_seeds,
                "grasp_contact_link_names": params.grasp_contact_link_names or [],
                "fit": "bspline_time_optimal_irls",
                **motion_fit_metadata,
                "tensors": motion_tensors,
                "kernels": motion_kernels,
            }
            (output_dir / "robot_motion_manifest.json").write_text(
                json.dumps(motion_manifest, indent=2, sort_keys=True) + "\n"
            )
    if not args.kinematics_only and selected_robot_config is not None:
        from cpp_runtime.robot.tools.export_timeopt import export_timeopt_tables

        for knots in sorted({transition.n_knots, 32, 64}):
            for reference in (False, True):
                export_timeopt_tables(output_dir, params.num_dof, knots, reference)
    return manifest_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-config", type=Path)
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--base-link")
    parser.add_argument("--tool-frame", action="append")
    parser.add_argument(
        "--attachment-parent-link",
        help="reserve an attached_object collision link fixed to this robot link",
    )
    parser.add_argument(
        "--attachment-spheres",
        type=int,
        default=32,
        help="fixed collision-sphere capacity for attached_object (default: 32)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--kinematics-only", action="store_true")
    parser.add_argument("--max-ik-seeds", type=int, default=64)
    parser.add_argument("--ik-seed-samples", type=int, default=2000)
    parser.add_argument("--max-ik-goalset", type=int, default=64)
    parser.add_argument("--max-trajopt-seeds", type=int, default=8)
    parser.add_argument("--interpolation-buffer-size", type=int, default=2000)
    parser.add_argument(
        "--joint-limit-expansion",
        type=float,
        default=0.0,
        help="symmetrically expand active-joint position limits by this many radians",
    )
    parser.add_argument("--rrt-samples-per-iteration", type=int, default=24)
    parser.add_argument("--rrt-max-iterations", type=int, default=32)
    parser.add_argument("--rrt-max-nodes-per-tree", type=int, default=1024)
    parser.add_argument("--max-cuboids", type=int, default=64)
    parser.add_argument("--max-capsules", type=int, default=64)
    parser.add_argument("--max-cylinders", type=int, default=32)
    parser.add_argument("--max-meshes", type=int, default=16)
    parser.add_argument("--max-voxel-grids", type=int, default=0)
    args = parser.parse_args()
    if args.robot_config is None and args.urdf is None:
        parser.error("provide --robot-config or --urdf")
    if args.attachment_parent_link is not None and args.robot_config is None:
        parser.error("--attachment-parent-link requires --robot-config")
    if args.attachment_spheres <= 0:
        parser.error("--attachment-spheres must be positive")
    if args.max_ik_seeds <= 0:
        parser.error("--max-ik-seeds must be positive")
    if args.ik_seed_samples < args.max_ik_seeds:
        parser.error("--ik-seed-samples must be at least --max-ik-seeds")
    if args.max_ik_goalset <= 0:
        parser.error("--max-ik-goalset must be positive")
    if args.max_trajopt_seeds <= 0:
        parser.error("--max-trajopt-seeds must be positive")
    if args.interpolation_buffer_size < 2:
        parser.error("--interpolation-buffer-size must be at least two")
    if not math.isfinite(args.joint_limit_expansion) or args.joint_limit_expansion < 0.0:
        parser.error("--joint-limit-expansion must be finite and non-negative")
    if args.rrt_samples_per_iteration <= 0 or args.rrt_samples_per_iteration > 256:
        parser.error("--rrt-samples-per-iteration must be in [1, 256]")
    if args.rrt_max_iterations <= 0:
        parser.error("--rrt-max-iterations must be positive")
    if args.rrt_max_nodes_per_tree <= 0:
        parser.error("--rrt-max-nodes-per-tree must be positive")
    for name in (
        "max_cuboids",
        "max_capsules",
        "max_cylinders",
        "max_meshes",
        "max_voxel_grids",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    return args


def main() -> None:
    """Export one robot artifact."""
    manifest_path = export_robot(parse_args())
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
