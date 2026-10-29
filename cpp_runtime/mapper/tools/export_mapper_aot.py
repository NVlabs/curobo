#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Export cuRobo mapper Warp AOT kernels and PBA CUBIN for the C++ runtime."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import torch
import warp as wp
from cuda import pathfinder
from cuda.core import Device, Program, ProgramOptions

CUROBO_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def default_curobo_source_dir() -> Path:
    env_path = os.environ.get("CUROBO_SOURCE_DIR")
    if env_path:
        return Path(env_path)
    if (CUROBO_REPOSITORY_ROOT / "curobo").is_dir():
        return CUROBO_REPOSITORY_ROOT
    try:
        import curobo

        return Path(curobo.__file__).resolve().parents[1]
    except ImportError:
        return CUROBO_REPOSITORY_ROOT


DEFAULT_CUROBO_SOURCE_DIR = default_curobo_source_dir()

BASE_KERNEL_NAMES = [
    "mark_blocks_in_frustum_kernel",
    "apply_decay_and_sum_kernel",
    "recycle_empty_blocks_kernel",
    "compute_block_keys_only_kernel",
    "allocate_visible_blocks_from_keys_kernel",
    "build_support_pixels_from_keys_kernel",
    "clear_new_blocks_kernel",
    "clear_new_block_grid_rgb_kernel",
    "clear_new_block_features_kernel",
    "clear_new_block_instances_kernel",
    "clear_instance_slot_kernel",
    "mark_instance_removal_blocks_kernel",
    "remove_instance_geometry_kernel",
    "integrate_voxels_kernel",
    "integrate_block_grid_rgb_kernel",
    "integrate_instance_grid_kernel",
    "rescale_block_accumulators_kernel",
    "rescale_block_grid_rgb_kernel",
    "count_occupied_voxels_kernel",
    "extract_occupied_voxels_kernel",
    "set_point_query_kernel",
    "set_point_target_class_kernel",
    "count_point_edges_kernel",
    "sample_point_edges_kernel",
    "compact_surface_points_kernel",
    "seed_esdf_sites_gather_kernel",
    "compute_esdf_from_min_tsdf_kernel",
    "raycast_block_sparse_kernel",
    "raycast_block_sparse_color_kernel",
    "append_active_blocks_kernel",
    "count_surface_cubes_from_blocks_kernel",
    "append_surface_cubes_from_blocks_kernel",
    "count_total_triangles_kernel",
    "generate_mesh_kernel",
    "project_mesh_uvs_kernel",
]

FEATURE_KERNEL_NAMES = {
    False: "integrate_features_from_support_grouped_kernel",
    True: "integrate_features_from_support_tiled_kernel",
}

FEATURE_OUTPUT_KERNEL_NAMES = [
    "raycast_block_sparse_feature_kernel",
    "gather_rendered_features_fp32_kernel",
    "gather_rendered_features_fp16_kernel",
]

SEMANTIC_KERNEL_NAMES = [
    "prepare_semantic_node_matches_kernel",
    "prepare_semantic_argmax_node_matches_kernel",
    "prepare_semantic_block_summary_kernel",
    "count_occupied_voxels_semantic_kernel",
    "extract_occupied_voxels_semantic_kernel",
    "seed_esdf_sites_gather_semantic_kernel",
    "compute_esdf_from_min_tsdf_semantic_kernel",
    "raycast_block_sparse_semantic_kernel",
    "raycast_block_sparse_color_semantic_kernel",
    "raycast_block_sparse_feature_semantic_kernel",
    "append_active_blocks_semantic_kernel",
    "append_surface_cubes_from_blocks_semantic_kernel",
]


def uses_tiled_feature_kernel(integrator: object) -> bool:
    """Return the resolved camera feature-kernel policy."""
    tsdf_integrator = getattr(integrator, "_tsdf_integrator", integrator)
    return bool(tsdf_integrator._camera_integrator.use_tiled_feature_kernel)


POSE_REFINER_KERNEL_NAME = "ray_sdf_alignment_block_sparse_tiled_kernel"
POSE_REFINER_LM_SYMBOLS = {
    "pose_refiner_count_valid_depth_kernel": (
        "curobo::optimization::pose_refiner::count_valid_depth_kernel"
    ),
    "pose_refiner_initialize_state_kernel": (
        "curobo::optimization::pose_refiner::initialize_state_kernel"
    ),
    "pose_refiner_solve_candidate_kernel": (
        "curobo::optimization::pose_refiner::solve_candidate_kernel"
    ),
    "pose_refiner_accept_candidate_kernel": (
        "curobo::optimization::pose_refiner::accept_candidate_kernel"
    ),
    "pose_refiner_copy_best_pose_kernel": (
        "curobo::optimization::pose_refiner::copy_best_pose_kernel"
    ),
}

PBA_NS = "curobo::parallel_banding::"
PBA_SYMBOLS = {
    "flood_z": PBA_NS + "kernel_flood_z",
    "maurer_axis": PBA_NS + "kernel_maurer_axis",
    "color_axis": PBA_NS + "kernel_color_axis",
}


def ensure_curobo_on_path() -> Path:
    source_dir = DEFAULT_CUROBO_SOURCE_DIR.resolve()
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    return source_dir


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def cuda_arch() -> tuple[int, str]:
    device = Device()
    device.set_current()
    arch = int(device.arch)
    return arch, f"sm_{arch}"


def make_integrator(config: dict[str, Any]):
    ensure_curobo_on_path()
    from curobo._src.perception.mapper.integrator_esdf import (
        BlockSparseESDFIntegrator,
        BlockSparseESDFIntegratorCfg,
    )
    from curobo._src.util.warp import init_warp

    init_warp()
    cfg = BlockSparseESDFIntegratorCfg(
        voxel_size=float(config["voxel_size"]),
        origin=torch.tensor(config.get("origin", [0.0, 0.0, 0.0]), dtype=torch.float32),
        esdf_voxel_size=float(config["esdf_voxel_size"]),
        esdf_grid_shape=tuple(int(v) for v in config["esdf_grid_shape"]),
        truncation_distance=float(config["truncation_distance"]),
        max_blocks=int(config["max_blocks"]),
        hash_capacity=int(config["hash_capacity"]),
        depth_minimum_distance=float(config["depth_minimum_distance"]),
        depth_maximum_distance=float(config["depth_maximum_distance"]),
        minimum_tsdf_weight=float(config["minimum_tsdf_weight"]),
        adjacent_skip_steps=float(config.get("adjacent_skip_steps", 1.0)),
        grid_shape=tuple(int(v) for v in config["grid_shape"]),
        image_height=int(config["image_height"]),
        image_width=int(config["image_width"]),
        enable_static=False,
        seeding_method="gather",
        edt_solver="pba",
        num_cameras=int(config["num_cameras"]),
        block_size=int(config["block_size"]),
        feature_dim=int(config.get("feature_dim", 0)),
        feature_block_grid_size=int(config.get("feature_block_grid_size", 1)),
        instance_grid_size=int(config.get("instance_grid_size", 0)),
        feature_grid_height=config.get("feature_grid_height"),
        feature_grid_width=config.get("feature_grid_width"),
        feature_channels_per_thread=int(config.get("feature_channels_per_thread", 4)),
        max_feature_tile_channels=int(config.get("max_feature_tile_channels", 4096)),
        feature_integration_kernel=str(config.get("feature_integration_kernel", "auto")),
        max_visible_blocks_per_integration=int(config["max_visible_blocks_per_integration"]),
        max_support_pixels_per_block_camera=int(config["max_support_pixels_per_block_camera"]),
        color_grid_size=int(config.get("color_grid_size", 1)),
        texture_num_cameras=int(config.get("texture_num_cameras", config["num_cameras"])),
        texture_camera_image_height=int(
            config.get("texture_camera_image_height", config["image_height"])
        ),
        texture_camera_image_width=int(
            config.get("texture_camera_image_width", config["image_width"])
        ),
        accumulator_w_max=float(config["accumulator_w_max"]),
        frustum_decay=float(config.get("frustum_decay_factor", 1.0)),
        time_decay=float(config.get("decay_factor", 1.0)),
        use_cuda_graph=False,
    )
    return cfg, BlockSparseESDFIntegrator(cfg)


def parse_forward_symbol(cu_path: Path) -> str:
    text = cu_path.read_text(errors="ignore")
    matches = re.findall(
        r'extern "C" __global__ void\s+([A-Za-z0-9_]+_cuda_kernel_forward)\s*\(', text
    )
    if not matches:
        raise RuntimeError(f"could not find forward symbol in {cu_path}")
    if len(matches) != 1:
        # The mapper builder currently emits one forward kernel per AOT module.
        # Keep this explicit so a future Warp codegen change is caught.
        raise RuntimeError(f"expected one forward symbol in {cu_path}, got {matches}")
    return matches[0]


def export_warp_kernels(integrator, out_dir: Path, arch: int, arch_name: str) -> dict[str, Any]:
    build_dir = out_dir / "_warp_build"
    shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True)

    kernels = integrator.tsdf.kernels
    manifest: dict[str, Any] = {}
    compiled_modules: set[str] = set()
    kernel_names = list(BASE_KERNEL_NAMES)
    if integrator.tsdf.data.has_features:
        kernel_names.append(FEATURE_KERNEL_NAMES[uses_tiled_feature_kernel(integrator)])
        kernel_names.extend(FEATURE_OUTPUT_KERNEL_NAMES)
        kernel_names.extend(SEMANTIC_KERNEL_NAMES)
    for name in kernel_names:
        kernel = getattr(kernels, name)
        module = kernel.module
        if module.name not in compiled_modules:
            wp.compile_aot_module(
                module,
                device=wp.get_device("cuda:0"),
                module_dir=str(build_dir),
                strip_hash=True,
                use_ptx=False,
            )
            compiled_modules.add(module.name)

        source_cubin = build_dir / f"wp_{module.name}.{arch_name.replace('_', '')}.cubin"
        source_cu = build_dir / f"wp_{module.name}.cu"
        source_meta = build_dir / f"wp_{module.name}.meta"
        if not source_cubin.exists():
            candidates = sorted(build_dir.glob(f"wp_{module.name}.sm*.cubin"))
            if not candidates:
                raise RuntimeError(f"missing CUBIN for module {module.name}")
            source_cubin = candidates[0]
        symbol = parse_forward_symbol(source_cu)
        meta = json.loads(source_meta.read_text())
        shared_mem = int(meta.get(f"{symbol}_smem_bytes", 0))

        cubin_name = f"warp_{name}.{arch_name}.cubin"
        shutil.copy2(source_cubin, out_dir / cubin_name)
        manifest[name] = {
            "cubin": cubin_name,
            "symbol": symbol,
            "shared_mem": shared_mem,
            "source": "curobo_mapper_kernel_bundle",
            "module": module.name,
            "kernel_key": getattr(kernel, "key", name),
        }
    return manifest


def export_pose_refiner_warp_kernel(
    integrator,
    out_dir: Path,
    arch: int,
    arch_name: str,
    n_samples_per_ray: int,
) -> dict[str, Any]:
    ensure_curobo_on_path()
    from curobo._src.perception.mapper.kernel.wp_raycast_pose_refine import (
        create_ray_sdf_alignment_block_sparse_tiled_kernel,
    )

    kernel = create_ray_sdf_alignment_block_sparse_tiled_kernel(
        n_samples_per_ray=n_samples_per_ray,
        kernels=integrator.tsdf.kernels,
    )
    build_dir = out_dir / "_pose_refiner_warp_build"
    shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True)
    wp.compile_aot_module(
        kernel.module,
        device=wp.get_device("cuda:0"),
        module_dir=str(build_dir),
        strip_hash=True,
        use_ptx=False,
    )

    module_name = kernel.module.name
    source_cubin = build_dir / f"wp_{module_name}.{arch_name.replace('_', '')}.cubin"
    source_cu = build_dir / f"wp_{module_name}.cu"
    source_meta = build_dir / f"wp_{module_name}.meta"
    if not source_cubin.exists():
        candidates = sorted(build_dir.glob(f"wp_{module_name}.sm*.cubin"))
        if not candidates:
            raise RuntimeError(f"missing pose refiner CUBIN for module {module_name}")
        source_cubin = candidates[0]
    symbol = parse_forward_symbol(source_cu)
    meta = json.loads(source_meta.read_text())
    shared_mem = int(meta.get(f"{symbol}_smem_bytes", 0))
    cubin_name = f"warp_{POSE_REFINER_KERNEL_NAME}.{arch_name}.cubin"
    shutil.copy2(source_cubin, out_dir / cubin_name)
    shutil.rmtree(build_dir, ignore_errors=True)
    return {
        POSE_REFINER_KERNEL_NAME: {
            "cubin": cubin_name,
            "symbol": symbol,
            "shared_mem": shared_mem,
            "source": "curobo_mapper_kernel_bundle",
            "module": module_name,
            "kernel_key": getattr(kernel, "key", POSE_REFINER_KERNEL_NAME),
            "n_samples_per_ray": n_samples_per_ray,
        }
    }


def compile_pba(out_dir: Path, arch_name: str) -> dict[str, Any]:
    ensure_curobo_on_path()
    from curobo._src.curobolib.backends.cuda_core_backend.pba_config import (
        PBAKernelCfg,
    )

    cfg = PBAKernelCfg()
    include_dirs = [str(path) for path in cfg.get_include_dirs()]
    for libname in ("cudart", "nvrtc"):
        cuda_include = pathfinder.find_nvidia_header_directory(libname)
        if cuda_include is not None and os.path.isdir(cuda_include):
            include_dirs.append(cuda_include)

    options = ProgramOptions(
        std="c++17",
        arch=arch_name,
        include_path=include_dirs,
        ftz=True,
        fma=True,
        prec_div=False,
        prec_sqrt=False,
        lineinfo=True,
        device_code_optimize=True,
    )
    source = '#include "pba3d_kernel.cuh"\n'
    obj = Program(source, code_type="c++", options=options).compile(
        "cubin", name_expressions=tuple(PBA_SYMBOLS.values())
    )
    cubin_name = f"pba3d.{arch_name}.cubin"
    (out_dir / cubin_name).write_bytes(obj.code)

    symbol_mapping = {
        key: (value.decode("utf-8") if isinstance(value, bytes) else value)
        for key, value in obj.symbol_mapping.items()
    }
    out: dict[str, Any] = {}
    for short_name, requested in PBA_SYMBOLS.items():
        out[short_name] = {
            "cubin": cubin_name,
            "symbol": symbol_mapping.get(requested, requested),
            "requested_symbol": requested,
            "shared_mem": 0,
        }
    return out


def compile_pose_refiner_lm(out_dir: Path, arch_name: str) -> dict[str, Any]:
    """Compile the shared cuda.core pose-refiner kernels for native launch."""
    ensure_curobo_on_path()
    from curobo._src.curobolib.backends.cuda_core_backend.optimization_config import (
        OptimizationKernelCfg,
    )

    cfg = OptimizationKernelCfg()
    include_dirs = [str(path) for path in cfg.get_include_dirs()]
    for library_name in ("cudart", "nvrtc"):
        cuda_include = pathfinder.find_nvidia_header_directory(library_name)
        if cuda_include is not None and os.path.isdir(cuda_include):
            include_dirs.append(cuda_include)

    options = ProgramOptions(
        std="c++17",
        arch=arch_name,
        include_path=list(dict.fromkeys(include_dirs)),
        ftz=True,
        fma=True,
        prec_div=False,
        prec_sqrt=False,
        lineinfo=True,
        device_code_optimize=True,
    )
    source = '#include "pose_refiner_lm_kernel.cuh"\n'
    requested_symbols = tuple(POSE_REFINER_LM_SYMBOLS.values())
    obj = Program(source, code_type="c++", options=options).compile(
        "cubin", name_expressions=requested_symbols
    )
    cubin_name = f"pose_refiner_lm.{arch_name}.cubin"
    (out_dir / cubin_name).write_bytes(obj.code)
    symbol_mapping = {
        key: (value.decode("utf-8") if isinstance(value, bytes) else value)
        for key, value in obj.symbol_mapping.items()
    }
    return {
        name: {
            "cubin": cubin_name,
            "symbol": symbol_mapping.get(requested, requested),
            "requested_symbol": requested,
            "shared_mem": 0,
            "source": "curobo_cuda_core_pose_refiner",
        }
        for name, requested in POSE_REFINER_LM_SYMBOLS.items()
    }


def normalized_config(config: dict[str, Any], kernels, integrator: object) -> dict[str, Any]:
    cfg = dict(config)
    cfg["grid_shape"] = [int(v) for v in config["grid_shape"]]
    cfg["esdf_grid_shape"] = [int(v) for v in config["esdf_grid_shape"]]
    cfg["origin"] = [float(v) for v in config.get("origin", [0.0, 0.0, 0.0])]
    cfg["num_samples"] = int(kernels.num_samples)
    cfg["num_block_key_candidates"] = (
        int(config["num_cameras"])
        * int(config["image_height"])
        * int(config["image_width"])
        * int(kernels.num_samples)
    )
    cfg["feature_dim"] = int(kernels.feature_dim)
    cfg["feature_block_grid_size"] = int(kernels.feature_block_grid_size)
    cfg["feature_grid_voxels"] = int(kernels.feature_grid_voxels)
    cfg["instance_grid_size"] = int(kernels.instance_grid_size)
    cfg["instance_grid_voxels"] = int(kernels.instance_grid_voxels)
    cfg["feature_grid_height"] = (
        int(kernels.feature_grid_shape[0]) if kernels.feature_grid_shape is not None else 0
    )
    cfg["feature_grid_width"] = (
        int(kernels.feature_grid_shape[1]) if kernels.feature_grid_shape is not None else 0
    )
    cfg["feature_channels_per_thread"] = int(kernels.feature_channels_per_thread)
    cfg["max_feature_tile_channels"] = int(kernels.max_feature_tile_channels)
    cfg["feature_integration_kernel"] = FEATURE_KERNEL_NAMES[uses_tiled_feature_kernel(integrator)]
    cfg["color_grid_size"] = int(config.get("color_grid_size", 1))
    cfg["color_grid_voxels"] = cfg["color_grid_size"] ** 3
    cfg["texture_num_cameras"] = int(config.get("texture_num_cameras", config["num_cameras"]))
    cfg["texture_camera_image_height"] = int(
        config.get("texture_camera_image_height", config["image_height"])
    )
    cfg["texture_camera_image_width"] = int(
        config.get("texture_camera_image_width", config["image_width"])
    )
    cfg["enable_static"] = False
    cfg["seeding_method"] = "gather"
    cfg["edt_solver"] = "pba"
    cfg["decay_factor"] = float(config.get("decay_factor", 1.0))
    cfg["frustum_decay_factor"] = float(config.get("frustum_decay_factor", 1.0))
    cfg["pose_refiner_n_samples_per_ray"] = int(config.get("pose_refiner_n_samples_per_ray", 1))
    return cfg


def export_mc_tables() -> dict[str, Any]:
    ensure_curobo_on_path()
    from curobo._src.perception.mapper.marching_cubes.kernel.wp_mc_common import (
        NUM_TRIANGLES_TABLE,
        TRIANGLE_TABLE,
    )

    return {
        "num_triangles": [int(v) for v in NUM_TRIANGLES_TABLE],
        "triangle_table": [int(v) for v in TRIANGLE_TABLE.reshape(-1).tolist()],
        "triangle_table_shape": [int(v) for v in TRIANGLE_TABLE.shape],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    config = load_json(args.config)
    arch, arch_name = cuda_arch()
    cfg, integrator = make_integrator(config)
    warp_manifest = export_warp_kernels(integrator, args.out, arch, arch_name)
    pose_refiner_n_samples_per_ray = int(config.get("pose_refiner_n_samples_per_ray", 1))
    warp_manifest.update(
        export_pose_refiner_warp_kernel(
            integrator,
            args.out,
            arch,
            arch_name,
            pose_refiner_n_samples_per_ray,
        )
    )
    warp_manifest.update(compile_pose_refiner_lm(args.out, arch_name))
    pba_manifest = compile_pba(args.out, arch_name)

    manifest = {
        "schema_version": 1,
        "arch": arch_name,
        "warp_version": wp.__version__,
        "curobo_source_dir": str(DEFAULT_CUROBO_SOURCE_DIR.resolve()),
        "config": normalized_config(config, integrator.tsdf.kernels, integrator),
        "kernels": warp_manifest,
        "pba": pba_manifest,
        "mc_tables": export_mc_tables(),
    }
    (args.out / "mapper_aot_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    shutil.rmtree(args.out / "_warp_build", ignore_errors=True)
    shutil.rmtree(args.out / "_pose_refiner_warp_build", ignore_errors=True)
    print(f"wrote {args.out / 'mapper_aot_manifest.json'}")


if __name__ == "__main__":
    main()
