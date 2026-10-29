# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in CUDA smoke tests for exported mapper runtime artifacts."""

from __future__ import annotations

import ctypes
import json
import math
import os
from pathlib import Path

import pytest
import torch


class _Frame(ctypes.Structure):
    _fields_ = [
        ("depth_images", ctypes.c_uint64),
        ("rgb_images", ctypes.c_uint64),
        ("intrinsics", ctypes.c_uint64),
        ("cam_positions", ctypes.c_uint64),
        ("cam_quaternions", ctypes.c_uint64),
        ("instance_image", ctypes.c_uint64),
        ("stream", ctypes.c_uint64),
    ]


class _FeatureFrame(ctypes.Structure):
    _fields_ = [("frame", _Frame), ("feature_grid", ctypes.c_uint64)]


class _SemanticFilter(ctypes.Structure):
    _fields_ = [
        ("target", ctypes.c_uint64),
        ("target_class", ctypes.c_int),
        ("min_similarity", ctypes.c_float),
        ("use_min_similarity", ctypes.c_int),
        ("require_argmax", ctypes.c_int),
        ("exclude", ctypes.c_int),
        ("exclude_instance_slot", ctypes.c_int),
    ]


class _DeviceView(ctypes.Structure):
    _fields_ = [
        ("block_grid_rgb", ctypes.c_uint64),
        ("block_features", ctypes.c_uint64),
        ("block_feature_weight", ctypes.c_uint64),
        ("block_instance_state", ctypes.c_uint64),
        ("block_coords", ctypes.c_uint64),
        ("block_to_hash_slot", ctypes.c_uint64),
        ("num_allocated", ctypes.c_uint64),
        ("esdf_dist_field", ctypes.c_uint64),
        ("max_blocks", ctypes.c_int),
        ("block_size", ctypes.c_int),
        ("block_voxels", ctypes.c_int),
        ("color_grid_size", ctypes.c_int),
        ("color_grid_voxels", ctypes.c_int),
        ("feature_dim", ctypes.c_int),
        ("feature_block_grid_size", ctypes.c_int),
        ("feature_grid_voxels", ctypes.c_int),
        ("instance_grid_size", ctypes.c_int),
        ("instance_grid_voxels", ctypes.c_int),
        ("grid_shape", ctypes.c_int * 3),
        ("esdf_shape", ctypes.c_int * 3),
        ("origin", ctypes.c_float * 3),
        ("voxel_size", ctypes.c_float),
        ("esdf_voxel_size", ctypes.c_float),
    ]


class _PoseRefinerConfig(ctypes.Structure):
    _fields_ = [
        ("n_points", ctypes.c_int),
        ("minimum_valid_depth_pixels", ctypes.c_int),
        ("max_iterations", ctypes.c_int),
        ("inner_iterations", ctypes.c_int),
        ("n_samples_per_ray", ctypes.c_int),
        ("tile_block_dim", ctypes.c_int),
        ("distance_threshold", ctypes.c_float),
        ("min_valid_ratio", ctypes.c_float),
        ("depth_minimum_distance", ctypes.c_float),
        ("depth_maximum_distance", ctypes.c_float),
        ("minimum_tsdf_weight", ctypes.c_float),
        ("lambda_initial", ctypes.c_float),
        ("lambda_factor", ctypes.c_float),
        ("lambda_min", ctypes.c_float),
        ("lambda_max", ctypes.c_float),
        ("rho_min", ctypes.c_float),
    ]


class _PoseRefinerResult(ctypes.Structure):
    _fields_ = [
        ("final_error", ctypes.c_float),
        ("final_lambda", ctypes.c_float),
        ("iterations", ctypes.c_int),
        ("valid_count", ctypes.c_int),
        ("accepted_steps", ctypes.c_int),
        ("status", ctypes.c_int),
    ]


def _bind_library(path: Path) -> ctypes.CDLL:
    library = ctypes.CDLL(str(path))
    void_pointer = ctypes.c_void_p
    uint64 = ctypes.c_uint64
    integer = ctypes.c_int
    float32 = ctypes.c_float
    semantic_pointer = ctypes.POINTER(_SemanticFilter)

    library.curobo_mapper_last_error.restype = ctypes.c_char_p
    library.curobo_mapper_create.argtypes = [
        ctypes.c_char_p,
        integer,
        ctypes.POINTER(void_pointer),
    ]
    library.curobo_mapper_create.restype = integer
    library.curobo_mapper_destroy.argtypes = [void_pointer]
    library.curobo_mapper_integrate.argtypes = [void_pointer, ctypes.POINTER(_Frame)]
    library.curobo_mapper_integrate.restype = integer
    library.curobo_mapper_get_device_view.argtypes = [
        void_pointer,
        ctypes.POINTER(_DeviceView),
    ]
    library.curobo_mapper_get_device_view.restype = integer
    library.curobo_mapper_integrate_features.argtypes = [
        void_pointer,
        ctypes.POINTER(_FeatureFrame),
    ]
    library.curobo_mapper_integrate_features.restype = integer
    library.curobo_mapper_render_features.argtypes = [
        void_pointer,
        *([uint64] * 9),
        *([integer] * 7),
        uint64,
    ]
    library.curobo_mapper_render_features.restype = integer
    library.curobo_mapper_gather_features.argtypes = [
        void_pointer,
        *([uint64] * 4),
        *([integer] * 6),
        uint64,
    ]
    library.curobo_mapper_gather_features.restype = integer
    library.curobo_mapper_render_depth_semantic.argtypes = [
        void_pointer,
        *([uint64] * 7),
        *([integer] * 3),
        semantic_pointer,
        uint64,
    ]
    library.curobo_mapper_render_depth_semantic.restype = integer
    library.curobo_mapper_extract_occupied_semantic.argtypes = [
        void_pointer,
        *([uint64] * 3),
        integer,
        integer,
        float32,
        semantic_pointer,
        uint64,
        ctypes.POINTER(integer),
        ctypes.POINTER(integer),
    ]
    library.curobo_mapper_extract_occupied_semantic.restype = integer
    library.curobo_mapper_extract_occupied_points.argtypes = [
        void_pointer, uint64, uint64, uint64, integer, integer, uint64,
        semantic_pointer, integer, uint64,
    ]
    library.curobo_mapper_extract_occupied_points.restype = integer
    library.curobo_mapper_extract_mesh_semantic.argtypes = [
        void_pointer,
        *([uint64] * 3),
        integer,
        float32,
        integer,
        integer,
        semantic_pointer,
        uint64,
        *([ctypes.POINTER(integer)] * 4),
    ]
    library.curobo_mapper_extract_mesh_semantic.restype = integer
    library.curobo_mapper_compute_esdf_semantic.argtypes = [
        void_pointer,
        uint64,
        semantic_pointer,
        uint64,
    ]
    library.curobo_mapper_compute_esdf_semantic.restype = integer
    library.curobo_mapper_get_esdf_graph_capture_count.argtypes = [void_pointer, ctypes.POINTER(uint64)]
    library.curobo_mapper_get_esdf_graph_capture_count.restype = integer
    library.curobo_mapper_config_float.argtypes = [
        void_pointer,
        ctypes.c_char_p,
        ctypes.POINTER(float32),
    ]
    library.curobo_mapper_config_float.restype = integer
    library.curobo_mapper_set_tsdf_runtime_parameters.argtypes = [
        void_pointer,
        float32,
        float32,
    ]
    library.curobo_mapper_set_tsdf_runtime_parameters.restype = integer
    library.curobo_mapper_reset.argtypes = [void_pointer, uint64]
    library.curobo_mapper_reset.restype = integer
    library.curobo_mapper_clear_instance_slot.argtypes = [
        void_pointer,
        integer,
        uint64,
    ]
    library.curobo_mapper_clear_instance_slot.restype = integer
    library.curobo_mapper_remove_instance_geometry.argtypes = [
        void_pointer,
        integer,
        uint64,
        uint64,
    ]
    library.curobo_mapper_remove_instance_geometry.restype = integer
    library.curobo_mapper_default_pose_refiner_config.argtypes = [
        ctypes.POINTER(_PoseRefinerConfig)
    ]
    library.curobo_mapper_default_pose_refiner_config.restype = integer
    library.curobo_mapper_refine_pose.argtypes = [
        void_pointer,
        *([uint64] * 6),
        ctypes.POINTER(_PoseRefinerConfig),
        uint64,
        ctypes.POINTER(_PoseRefinerResult),
    ]
    library.curobo_mapper_refine_pose.restype = integer
    return library


def _check(library: ctypes.CDLL, result: int, operation: str) -> None:
    if result != 0:
        error = library.curobo_mapper_last_error().decode()
        raise RuntimeError(f"{operation}: {error}")


def test_exporter_has_no_runtime_private_warp_kernels() -> None:
    """Keep mapper Warp implementations in the Python mapper kernel bundle."""
    mapper_root = Path(__file__).resolve().parents[1]
    tool_sources = sorted((mapper_root / "tools").glob("*.py"))
    assert [path.name for path in tool_sources] == ["export_mapper_aot.py"]

    exporter = tool_sources[0].read_text()
    assert '"apply_decay_and_sum_kernel"' in exporter
    assert '"remove_instance_geometry_kernel"' in exporter
    assert '"pose_refiner_solve_candidate_kernel"' in exporter
    assert '"pose_refiner_lm_kernel.cuh"' in exporter
    assert "export_runtime_warp_kernels" not in exporter
    assert not (mapper_root / "src" / "instance_removal.cu").exists()
    assert not (mapper_root / "src" / "pose_refiner_lm.cu").exists()
    assert not (mapper_root / "src" / "pose_refiner_lm.h").exists()


def test_surface_points_runtime() -> None:
    """Check observed surfaces, filtering, and graph replay for the artifact batch size."""
    artifact = os.environ.get("CUROBO_MAPPER_TEST_ARTIFACT_DIR")
    if not artifact or not torch.cuda.is_available():
        pytest.skip("CUDA and an exported mapper artifact are required")
    config = json.loads((Path(artifact) / "mapper_aot_manifest.json").read_text())["config"]
    library = _bind_library(Path(os.environ["CUROBO_MAPPER_TEST_LIBRARY"]))
    runtime = ctypes.c_void_p()
    _check(library, library.curobo_mapper_create(artifact.encode(), 0, ctypes.byref(runtime)), "create")
    try:
        with torch.cuda.stream(torch.cuda.Stream()):
            batch, height, width = config["num_cameras"], config["image_height"], config["image_width"]
            depth = torch.full((batch, height, width), .4, device="cuda")
            rgb = torch.zeros((batch, height, width, 3), dtype=torch.uint8, device="cuda")
            instances = torch.full((batch, height, width), 3, dtype=torch.uint8, device="cuda")
            intrinsics = torch.tensor([[[600., 0., width / 2], [0., 600., height / 2],
                                        [0., 0., 1.]]], device="cuda").repeat(batch, 1, 1)
            positions = torch.zeros((batch, 3), device="cuda")
            positions[:, :2] = torch.tensor(config["origin"][:2], device="cuda")
            positions[:, 2] = config["origin"][2] - .2
            if batch > 1:
                positions[1:, 0] += .03
            quaternions = torch.tensor([[1., 0., 0., 0.]], device="cuda").repeat(batch, 1)
            features = torch.zeros((batch, config["feature_grid_height"],
                                    config["feature_grid_width"], config["feature_dim"]),
                                   dtype=torch.float16, device="cuda")
            features[..., 0] = 1
            stream = torch.cuda.current_stream().cuda_stream
            frame = _FeatureFrame(_Frame(depth.data_ptr(), rgb.data_ptr(), intrinsics.data_ptr(),
                                  positions.data_ptr(), quaternions.data_ptr(), instances.data_ptr(), stream),
                                  features.data_ptr())
            points = torch.zeros((2000, 3), device="cuda")
            sources = torch.zeros((2000, 2), dtype=torch.int32, device="cuda")
            count = torch.zeros(1, dtype=torch.int32, device="cuda")
            semantic = _SemanticFilter(0, 0, 0., 0, 1, 0, 0)

            def extract(slot: int = 3, selection: int = 0) -> None:
                _check(library, library.curobo_mapper_extract_occupied_points(
                    runtime, points.data_ptr(), sources.data_ptr(), count.data_ptr(),
                    2000, slot, selection, ctypes.byref(semantic), 7,
                    torch.cuda.current_stream().cuda_stream), "surface points")

            extract()
            assert count.item() == 0
            for _ in range(3):
                _check(library, library.curobo_mapper_integrate_features(runtime, ctypes.byref(frame)), "integrate")
            extract()
            n = count.item()
            assert n > 1800
            assert torch.isfinite(points[:n]).all()
            torch.testing.assert_close(points[:n, 2], torch.full((n,), config["origin"][2] + .2,
                                        device="cuda"), atol=config["voxel_size"] * .6, rtol=0)
            assert sources[:n, 0].min() >= 0
            assert sources[:n, 1].max() < config["block_size"] ** 3
            expected = points.clone()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                extract()
            graph.replay()
            assert count.item() == n
            torch.testing.assert_close(points[:n], expected[:n], rtol=0, atol=0)
            extract(99)
            assert count.item() == 0
            semantic.target_class = 1
            extract()
            assert count.item() == 0
            semantic.target_class = 0
            selection = torch.zeros((config["max_blocks"], config["block_size"] ** 3),
                                    dtype=torch.uint8, device="cuda")
            extract(selection=selection.data_ptr())
            assert count.item() == 0
            # Graph nodes retain query values; later host queries must not corrupt replay.
            graph.replay()
            assert count.item() == n
            target = torch.zeros(config["feature_dim"], device="cuda")
            target[0] = 1
            semantic.target = target.data_ptr()
            semantic.target_class = -1
            extract()
            target_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(target_graph):
                extract()
            target_graph.replay()
            assert count.item() == n
            target[0], target[1] = 0, 1
            target_graph.replay()
            assert count.item() == 0
            torch.cuda.synchronize()
    finally:
        library.curobo_mapper_destroy(runtime)


def test_pose_refiner_runtime_uses_exported_cuda_core_kernels() -> None:
    """Refine an integrated plane through the standalone C ABI."""
    artifact_value = os.environ.get("CUROBO_MAPPER_TEST_ARTIFACT_DIR")
    if artifact_value is None:
        pytest.skip("set CUROBO_MAPPER_TEST_ARTIFACT_DIR to an exported mapper artifact")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    artifact_dir = Path(artifact_value)
    manifest = json.loads((artifact_dir / "mapper_aot_manifest.json").read_text())
    config = manifest["config"]
    pose_kernel_names = {
        "pose_refiner_count_valid_depth_kernel",
        "pose_refiner_initialize_state_kernel",
        "pose_refiner_solve_candidate_kernel",
        "pose_refiner_accept_candidate_kernel",
        "pose_refiner_copy_best_pose_kernel",
    }
    assert all(
        manifest["kernels"][name]["source"] == "curobo_cuda_core_pose_refiner"
        for name in pose_kernel_names
    )

    runtime_root = Path(__file__).resolve().parents[2]
    library_path = Path(
        os.environ.get(
            "CUROBO_MAPPER_TEST_LIBRARY",
            runtime_root / "build" / "mapper" / "libcurobo_mapper_runtime.so",
        )
    )
    library = _bind_library(library_path)
    runtime = ctypes.c_void_p()
    _check(
        library,
        library.curobo_mapper_create(str(artifact_dir).encode(), 0, ctypes.byref(runtime)),
        "create",
    )
    try:
        image_height = int(config["image_height"])
        image_width = int(config["image_width"])
        voxel_size = float(config["voxel_size"])
        grid_depth, grid_height, grid_width = (int(value) for value in config["grid_shape"])
        origin_x, origin_y, origin_z = (float(value) for value in config["origin"])
        plane_depth = min(0.3, 0.4 * grid_depth * voxel_size)
        device = torch.device("cuda:0")
        depth = torch.full(
            (1, image_height, image_width),
            plane_depth,
            dtype=torch.float32,
            device=device,
        )
        rgb = torch.zeros((1, image_height, image_width, 3), dtype=torch.uint8, device=device)
        focal = float(max(image_height, image_width))
        intrinsics = torch.tensor(
            [
                [
                    [focal, 0.0, (image_width - 1) / 2],
                    [0.0, focal, (image_height - 1) / 2],
                    [0.0, 0.0, 1.0],
                ]
            ],
            dtype=torch.float32,
            device=device,
        )
        position = torch.tensor(
            [
                [
                    origin_x + 0.5 * grid_width * voxel_size,
                    origin_y + 0.5 * grid_height * voxel_size,
                    origin_z + 2.0 * voxel_size,
                ]
            ],
            dtype=torch.float32,
            device=device,
        )
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
        stream = torch.cuda.current_stream().cuda_stream
        frame = _Frame(
            depth.data_ptr(),
            rgb.data_ptr(),
            intrinsics.data_ptr(),
            position.data_ptr(),
            quaternion.data_ptr(),
            0,
            stream,
        )
        for _ in range(3):
            _check(
                library,
                library.curobo_mapper_integrate(runtime, ctypes.byref(frame)),
                "integrate",
            )

        initial_position = position.clone()
        initial_position[:, 2].add_(min(0.015, voxel_size * 0.5))
        output_position = torch.empty_like(position)
        output_quaternion = torch.empty_like(quaternion)
        refiner_config = _PoseRefinerConfig()
        _check(
            library,
            library.curobo_mapper_default_pose_refiner_config(ctypes.byref(refiner_config)),
            "default pose-refiner config",
        )
        refiner_config.max_iterations = 4
        refiner_config.minimum_tsdf_weight = float(config["minimum_tsdf_weight"])
        result = _PoseRefinerResult()
        _check(
            library,
            library.curobo_mapper_refine_pose(
                runtime,
                depth[0].data_ptr(),
                intrinsics[0].data_ptr(),
                initial_position[0].data_ptr(),
                quaternion[0].data_ptr(),
                output_position[0].data_ptr(),
                output_quaternion[0].data_ptr(),
                ctypes.byref(refiner_config),
                stream,
                ctypes.byref(result),
            ),
            "refine pose",
        )
        torch.cuda.synchronize()
        assert result.status == 0
        assert result.iterations == 4
        assert result.valid_count > refiner_config.minimum_valid_depth_pixels
        assert math.isfinite(result.final_error)
        assert torch.isfinite(output_position).all()
        assert output_quaternion.norm().item() == pytest.approx(1.0, abs=1.0e-5)
    finally:
        library.curobo_mapper_destroy(runtime)


def test_feature_and_semantic_runtime_parity() -> None:
    if not os.environ.get("CUROBO_MAPPER_TEST_ARTIFACT_DIR"):
        pytest.skip("set CUROBO_MAPPER_TEST_ARTIFACT_DIR to an exported feature artifact")
    # Deployment uses a nondefault stream; stream zero bypasses native graphs.
    with torch.cuda.stream(torch.cuda.Stream()):
        _feature_and_semantic_runtime_parity()


def _feature_and_semantic_runtime_parity() -> None:
    """Exercise fused features and semantic outputs through the native C ABI."""
    artifact_value = os.environ.get("CUROBO_MAPPER_TEST_ARTIFACT_DIR")
    if artifact_value is None:
        pytest.skip("set CUROBO_MAPPER_TEST_ARTIFACT_DIR to an exported feature artifact")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    artifact_dir = Path(artifact_value)
    manifest = json.loads((artifact_dir / "mapper_aot_manifest.json").read_text())
    config = manifest["config"]
    assert config["feature_dim"] > 0
    assert all(
        kernel["source"] in {"curobo_mapper_kernel_bundle", "curobo_cuda_core_pose_refiner"}
        for kernel in manifest["kernels"].values()
    )

    runtime_root = Path(__file__).resolve().parents[2]
    library_path = Path(
        os.environ.get(
            "CUROBO_MAPPER_TEST_LIBRARY",
            runtime_root / "build" / "mapper" / "libcurobo_mapper_runtime.so",
        )
    )
    library = _bind_library(library_path)
    runtime = ctypes.c_void_p()
    _check(
        library,
        library.curobo_mapper_create(str(artifact_dir).encode(), 0, ctypes.byref(runtime)),
        "create",
    )
    try:
        _check(
            library,
            library.curobo_mapper_set_tsdf_runtime_parameters(
                runtime, ctypes.c_float(0.25), ctypes.c_float(1.0)
            ),
            "set TSDF runtime parameters",
        )
        for name, expected in ((b"decay_factor", 0.25), (b"minimum_tsdf_weight", 1.0)):
            actual = ctypes.c_float()
            _check(
                library,
                library.curobo_mapper_config_float(runtime, name, ctypes.byref(actual)),
                f"read {name.decode()}",
            )
            assert actual.value == pytest.approx(expected)
        image_height = int(config["image_height"])
        image_width = int(config["image_width"])
        feature_height = int(config["feature_grid_height"])
        feature_width = int(config["feature_grid_width"])
        feature_dim = int(config["feature_dim"])
        device = torch.device("cuda:0")
        depth = torch.full(
            (1, image_height, image_width),
            0.3,
            dtype=torch.float32,
            device=device,
        )
        rgb = torch.zeros((1, image_height, image_width, 3), dtype=torch.uint8, device=device)
        intrinsics = torch.tensor(
            [
                [
                    [100.0, 0.0, (image_width - 1) / 2],
                    [0.0, 100.0, (image_height - 1) / 2],
                    [0.0, 0.0, 1.0],
                ]
            ],
            dtype=torch.float32,
            device=device,
        )
        positions = torch.zeros((1, 3), dtype=torch.float32, device=device)
        quaternions = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
        features = torch.zeros(
            (1, feature_height, feature_width, feature_dim),
            dtype=torch.float16,
            device=device,
        )
        features[..., 0] = 1.0
        instance_image = torch.full(
            (1, image_height, image_width),
            3,
            dtype=torch.uint8,
            device=device,
        )
        stream = torch.cuda.current_stream().cuda_stream
        frame = _FeatureFrame(
            _Frame(
                depth.data_ptr(),
                rgb.data_ptr(),
                intrinsics.data_ptr(),
                positions.data_ptr(),
                quaternions.data_ptr(),
                instance_image.data_ptr(),
                stream,
            ),
            features.data_ptr(),
        )
        for _ in range(2):
            _check(
                library,
                library.curobo_mapper_integrate_features(runtime, ctypes.byref(frame)),
                "integrate features",
            )

        pixel_count = image_height * image_width
        normals = torch.empty((pixel_count, 3), dtype=torch.float32, device=device)
        depths = torch.empty(pixel_count, dtype=torch.float32, device=device)
        mask = torch.empty(pixel_count, dtype=torch.uint8, device=device)
        feature_valid = torch.empty(pixel_count, dtype=torch.uint8, device=device)
        block_indices = torch.empty(pixel_count, dtype=torch.int32, device=device)
        node_indices = torch.empty(pixel_count, dtype=torch.int16, device=device)
        _check(
            library,
            library.curobo_mapper_render_features(
                runtime,
                intrinsics.data_ptr(),
                positions.data_ptr(),
                quaternions.data_ptr(),
                normals.data_ptr(),
                depths.data_ptr(),
                mask.data_ptr(),
                feature_valid.data_ptr(),
                block_indices.data_ptr(),
                node_indices.data_ptr(),
                1,
                image_height,
                image_width,
                0,
                0,
                image_height,
                image_width,
                stream,
            ),
            "render features",
        )
        rendered_features = torch.empty(
            (pixel_count, feature_dim), dtype=torch.float32, device=device
        )
        _check(
            library,
            library.curobo_mapper_gather_features(
                runtime,
                feature_valid.data_ptr(),
                block_indices.data_ptr(),
                node_indices.data_ptr(),
                rendered_features.data_ptr(),
                1,
                image_height,
                image_width,
                0,
                feature_dim,
                0,
                stream,
            ),
            "gather features",
        )
        torch.cuda.synchronize()
        valid = feature_valid.bool()
        assert valid.any()
        expected = torch.zeros(feature_dim, dtype=torch.float32, device=device)
        expected[0] = 1.0
        torch.testing.assert_close(
            rendered_features[valid], expected.expand(int(valid.sum().item()), -1)
        )

        target = expected.clone()
        only = _SemanticFilter(target.data_ptr(), -1, 0.99, 1, 1, 0)
        exclude = _SemanticFilter(target.data_ptr(), -1, 0.99, 1, 1, 1)
        robot_class = _SemanticFilter(0, 1, 0.0, 0, 1, 1)
        view = _DeviceView()
        _check(
            library,
            library.curobo_mapper_get_device_view(runtime, ctypes.byref(view)),
            "device view",
        )
        assert view.esdf_dist_field != 0
        assert view.block_grid_rgb != 0
        assert view.block_features != 0
        assert view.block_instance_state != 0
        assert view.feature_dim == feature_dim
        assert view.instance_grid_size == int(config["instance_grid_size"])
        assert tuple(view.esdf_shape) == tuple(config["esdf_grid_shape"])
        hit_points = torch.empty((pixel_count, 3), dtype=torch.float32, device=device)
        for semantic_filter, expect_geometry in ((only, True), (exclude, False)):
            _check(
                library,
                library.curobo_mapper_render_depth_semantic(
                    runtime,
                    intrinsics.data_ptr(),
                    positions.data_ptr(),
                    quaternions.data_ptr(),
                    hit_points.data_ptr(),
                    normals.data_ptr(),
                    depths.data_ptr(),
                    mask.data_ptr(),
                    1,
                    image_height,
                    image_width,
                    ctypes.byref(semantic_filter),
                    stream,
                ),
                "semantic render",
            )
            torch.cuda.synchronize()
            assert bool(mask.any()) is expect_geometry

            required = ctypes.c_int()
            written = ctypes.c_int()
            _check(
                library,
                library.curobo_mapper_extract_occupied_semantic(
                    runtime,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0.0,
                    ctypes.byref(semantic_filter),
                    stream,
                    ctypes.byref(required),
                    ctypes.byref(written),
                ),
                "semantic occupied extraction",
            )
            assert (required.value > 0) is expect_geometry

            mesh_counts = [ctypes.c_int() for _ in range(4)]
            _check(
                library,
                library.curobo_mapper_extract_mesh_semantic(
                    runtime,
                    0,
                    0,
                    0,
                    0,
                    0.0,
                    0,
                    0,
                    ctypes.byref(semantic_filter),
                    stream,
                    *(ctypes.byref(value) for value in mesh_counts),
                ),
                "semantic mesh extraction",
            )
            assert (mesh_counts[1].value > 0) is expect_geometry

        esdf = torch.empty(tuple(config["esdf_grid_shape"]), dtype=torch.float16, device=device)
        _check(
            library,
            library.curobo_mapper_compute_esdf_semantic(
                runtime, esdf.data_ptr(), ctypes.byref(only), stream
            ),
            "semantic ESDF",
        )
        torch.cuda.synchronize()
        assert torch.isfinite(esdf).all()
        _check(
            library,
            library.curobo_mapper_compute_esdf_semantic(
                runtime, esdf.data_ptr(), ctypes.byref(robot_class), stream
            ),
            "categorical semantic ESDF",
        )
        torch.cuda.synchronize()
        first_categorical_esdf = esdf.clone()
        _check(
            library,
            library.curobo_mapper_compute_esdf_semantic(
                runtime, esdf.data_ptr(), ctypes.byref(robot_class), stream
            ),
            "replayed categorical semantic ESDF",
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(esdf, first_categorical_esdf)
        # Subtract only the selected instance's TSDF sites from the planning
        # ESDF. The original geometry must remain available on the next call.
        if int(config["instance_grid_size"]) == int(config["block_size"]):
            without_instance = _SemanticFilter(0, 1, 0.0, 0, 1, 1, 3)
            captures_before = ctypes.c_uint64()
            _check(library, library.curobo_mapper_get_esdf_graph_capture_count(
                runtime, ctypes.byref(captures_before)), "read warmed capture count")
            assert captures_before.value > 0
            # Both outputs belong to the same immutable TSDF snapshot. Alternate
            # destinations and selected IDs without overwriting the full field.
            excluded_esdf = torch.empty_like(esdf)
            for selected_slot in (3, 4, 3, 0):
                without_instance.exclude_instance_slot = selected_slot
                _check(library, library.curobo_mapper_compute_esdf_semantic(
                    runtime, excluded_esdf.data_ptr(), ctypes.byref(without_instance), stream),
                    "paired planning ESDF")
                torch.cuda.synchronize()
                torch.testing.assert_close(esdf, first_categorical_esdf)
                if selected_slot == 3:
                    assert torch.any(excluded_esdf > esdf + .02)
                else:
                    torch.testing.assert_close(excluded_esdf, esdf)
            without_instance.exclude_instance_slot = 3
            for _ in range(2):
                _check(library, library.curobo_mapper_compute_esdf_semantic(
                    runtime, esdf.data_ptr(), ctypes.byref(without_instance), stream),
                    "instance-subtracted ESDF")
                torch.cuda.synchronize()
                assert torch.all(esdf >= first_categorical_esdf)
                assert torch.any(esdf > first_categorical_esdf + .02)
            _check(library, library.curobo_mapper_compute_esdf_semantic(
                runtime, esdf.data_ptr(), ctypes.byref(robot_class), stream),
                "restore full ESDF")
            torch.cuda.synchronize()
            torch.testing.assert_close(esdf, first_categorical_esdf)
            unrelated_instance = _SemanticFilter(0, 1, 0.0, 0, 1, 1, 4)
            _check(library, library.curobo_mapper_compute_esdf_semantic(
                runtime, esdf.data_ptr(), ctypes.byref(unrelated_instance), stream),
                "preserve unrelated instance in ESDF")
            torch.cuda.synchronize()
            torch.testing.assert_close(esdf, first_categorical_esdf)
            captures_after = ctypes.c_uint64()
            _check(library, library.curobo_mapper_get_esdf_graph_capture_count(
                runtime, ctypes.byref(captures_after)), "read replay capture count")
            assert captures_after.value == captures_before.value
            # Inclusion plus instance exclusion: retain the selected semantic
            # class, but remove only the selected owner. Check graph replay
            # switches owners without recapture and does not mutate the TSDF.
            included = _SemanticFilter(0, 0, 0.0, 0, 1, 0, 0)
            included_esdf = torch.empty_like(esdf)
            _check(library, library.curobo_mapper_compute_esdf_semantic(
                runtime, included_esdf.data_ptr(), ctypes.byref(included), stream),
                "included class ESDF")
            torch.cuda.synchronize()
            torch.testing.assert_close(included_esdf, first_categorical_esdf)
            _check(library, library.curobo_mapper_get_esdf_graph_capture_count(
                runtime, ctypes.byref(captures_before)), "inclusion capture count")
            for selected_slot in (3, 4, 3, 0):
                included.exclude_instance_slot = selected_slot
                _check(library, library.curobo_mapper_compute_esdf_semantic(
                    runtime, esdf.data_ptr(), ctypes.byref(included), stream),
                    "included class without selected instance")
                torch.cuda.synchronize()
                if selected_slot == 3:
                    assert torch.any(esdf > included_esdf + .02)
                else:
                    torch.testing.assert_close(esdf, included_esdf)
            _check(library, library.curobo_mapper_get_esdf_graph_capture_count(
                runtime, ctypes.byref(captures_after)), "inclusion replay count")
            assert captures_after.value == captures_before.value
        _check(
            library,
            library.curobo_mapper_remove_instance_geometry(
                runtime, 3, (1 << 2) | (1 << 3), stream
            ),
            "preserve unmatched instance geometry",
        )
        preserved_count = ctypes.c_int()
        preserved_written = ctypes.c_int()
        _check(
            library,
            library.curobo_mapper_extract_occupied_semantic(
                runtime,
                0,
                0,
                0,
                0,
                0,
                0.0,
                ctypes.byref(only),
                stream,
                ctypes.byref(preserved_count),
                ctypes.byref(preserved_written),
            ),
            "extract preserved instance geometry",
        )
        assert preserved_count.value > 0

        _check(library, library.curobo_mapper_reset(runtime, stream), "reset mapper")
        for _ in range(2):
            _check(
                library,
                library.curobo_mapper_integrate_features(runtime, ctypes.byref(frame)),
                "restore instance ownership",
            )
        _check(
            library,
            library.curobo_mapper_remove_instance_geometry(runtime, 3, 1 << 0, stream),
            "remove matching instance geometry",
        )
        removed_count = ctypes.c_int()
        removed_written = ctypes.c_int()
        _check(
            library,
            library.curobo_mapper_extract_occupied_semantic(
                runtime,
                0,
                0,
                0,
                0,
                0,
                0.0,
                ctypes.byref(only),
                stream,
                ctypes.byref(removed_count),
                ctypes.byref(removed_written),
            ),
            "extract removed instance geometry",
        )
        assert removed_count.value == 0
        _check(
            library,
            library.curobo_mapper_clear_instance_slot(runtime, 3, stream),
            "clear instance slot",
        )
    finally:
        library.curobo_mapper_destroy(runtime)
