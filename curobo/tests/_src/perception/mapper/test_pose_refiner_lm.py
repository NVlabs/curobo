# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parity tests for the shared cuda.core pose-refiner LM kernels."""

from __future__ import annotations

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.curobolib.backends.cuda_core_backend.pose_refiner import (
    launch_pose_refiner_accept_candidate,
    launch_pose_refiner_initialize_state,
    launch_pose_refiner_solve_candidate,
)
from curobo._src.perception.mapper.integrator_esdf import (
    BlockSparseESDFIntegrator,
    BlockSparseESDFIntegratorCfg,
)
from curobo._src.perception.mapper.pose_refiner import (
    BlockSparseRaycastPoseRefiner,
    BlockSparseRaycastRefinerCfg,
)
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


def _reference_candidate(
    position: torch.Tensor,
    quaternion: torch.Tensor,
    delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose a left-multiplied XYZ-extrinsic pose update in wxyz format."""
    rx, ry, rz = 0.5 * delta[3], 0.5 * delta[4], 0.5 * delta[5]
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    update = torch.stack(
        (
            cx * cy * cz + sx * sy * sz,
            sx * cy * cz - cx * sy * sz,
            cx * sy * cz + sx * cy * sz,
            cx * cy * sz - sx * sy * cz,
        )
    )
    update_vector = update[1:]
    source_position = position.view(3)
    uv = torch.linalg.cross(update_vector, source_position)
    uuv = torch.linalg.cross(update_vector, uv)
    output_position = delta[:3] + source_position + 2.0 * (update[0] * uv + uuv)

    aw, ax, ay, az = update
    bw, bx, by, bz = quaternion.view(4)
    output_quaternion = torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + bw * ax + ay * bz - az * by,
            aw * by + bw * ay + az * bx - ax * bz,
            aw * bz + bw * az + ax * by - ay * bx,
        )
    )
    return output_position.view(1, 3), (output_quaternion / output_quaternion.norm()).view(1, 4)


def test_pose_refiner_solve_candidate_matches_torch() -> None:
    """Match Torch Cholesky, predicted reduction, and pose composition."""
    device = torch.device("cuda:0")
    torch.manual_seed(7)
    matrix = torch.randn((6, 6), dtype=torch.float32, device=device)
    jtj = matrix.T @ matrix + 0.2 * torch.eye(6, dtype=torch.float32, device=device)
    jtr = torch.randn(6, dtype=torch.float32, device=device)
    lambda_damping = torch.tensor([1.0e-3], dtype=torch.float32, device=device)
    position = torch.tensor([[0.2, -0.1, 0.5]], dtype=torch.float32, device=device)
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    candidate_position = torch.empty_like(position)
    candidate_quaternion = torch.empty_like(quaternion)
    predicted_reduction = torch.empty(1, dtype=torch.float32, device=device)
    result_int = torch.zeros(4, dtype=torch.int32, device=device)

    launch_pose_refiner_solve_candidate(
        position,
        quaternion,
        jtj,
        jtr,
        lambda_damping,
        candidate_position,
        candidate_quaternion,
        predicted_reduction,
        result_int,
    )

    delta = torch.linalg.solve(jtj + lambda_damping * torch.eye(6, device=device), -jtr)
    expected_position, expected_quaternion = _reference_candidate(position, quaternion, delta)
    expected_reduction = -torch.dot(delta, jtr) - 0.5 * torch.dot(delta, jtj @ delta)
    torch.testing.assert_close(candidate_position, expected_position, atol=2.0e-6, rtol=2.0e-6)
    torch.testing.assert_close(candidate_quaternion, expected_quaternion, atol=2.0e-6, rtol=2.0e-6)
    torch.testing.assert_close(predicted_reduction[0], expected_reduction)
    assert result_int[3].item() == 0


def test_pose_refiner_initialize_and_accept_updates_state() -> None:
    """Initialize state and preserve the native accept/reject bookkeeping contract."""
    device = torch.device("cuda:0")
    position = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32, device=device)
    quaternion = torch.tensor([[2.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    jtj = torch.eye(6, dtype=torch.float32, device=device)
    jtr = torch.arange(6, dtype=torch.float32, device=device)
    sum_sq = torch.tensor([4.0], dtype=torch.float32, device=device)
    valid_count = torch.tensor([20], dtype=torch.int32, device=device)
    valid_depth_count = torch.tensor([100], dtype=torch.int32, device=device)
    best_position = torch.empty_like(position)
    best_quaternion = torch.empty_like(quaternion)
    best_jtj = torch.empty_like(jtj)
    best_jtr = torch.empty_like(jtr)
    best_sum_sq = torch.empty_like(sum_sq)
    best_error = torch.empty_like(sum_sq)
    best_valid_count = torch.empty_like(valid_count)
    lambda_damping = torch.empty_like(sum_sq)
    result_float = torch.empty(2, dtype=torch.float32, device=device)
    result_int = torch.empty(4, dtype=torch.int32, device=device)

    launch_pose_refiner_initialize_state(
        position,
        quaternion,
        jtj,
        jtr,
        sum_sq,
        valid_count,
        valid_depth_count,
        10,
        10,
        1.0e-3,
        best_position,
        best_quaternion,
        best_jtj,
        best_jtr,
        best_sum_sq,
        best_error,
        best_valid_count,
        lambda_damping,
        result_float,
        result_int,
    )
    torch.testing.assert_close(best_quaternion, quaternion / quaternion.norm())
    torch.testing.assert_close(best_error, torch.tensor([4.0**0.5 / 20.0**0.5], device=device))
    assert result_int.tolist() == [0, 20, 0, 0]

    candidate_position = torch.tensor([[0.3, 0.2, 0.1]], device=device)
    candidate_quaternion = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device)
    candidate_jtj = 2.0 * jtj
    candidate_jtr = -jtr
    candidate_sum_sq = torch.tensor([1.0], device=device)
    candidate_valid_count = torch.tensor([25], dtype=torch.int32, device=device)
    predicted_reduction = torch.tensor([1.0], device=device)
    launch_pose_refiner_accept_candidate(
        candidate_position,
        candidate_quaternion,
        candidate_jtj,
        candidate_jtr,
        candidate_sum_sq,
        candidate_valid_count,
        best_position,
        best_quaternion,
        best_jtj,
        best_jtr,
        best_sum_sq,
        best_error,
        best_valid_count,
        lambda_damping,
        predicted_reduction,
        10,
        10.0,
        1.0e-7,
        1.0e7,
        result_float,
        result_int,
    )
    torch.testing.assert_close(best_position, candidate_position)
    torch.testing.assert_close(best_jtj, candidate_jtj)
    torch.testing.assert_close(lambda_damping, torch.tensor([1.0e-4], device=device))
    assert result_int.tolist() == [1, 25, 1, 0]

    rejected_position = torch.full_like(candidate_position, 9.0)
    rejected_sum_sq = torch.tensor([2.0], device=device)
    launch_pose_refiner_accept_candidate(
        rejected_position,
        candidate_quaternion,
        candidate_jtj,
        candidate_jtr,
        rejected_sum_sq,
        candidate_valid_count,
        best_position,
        best_quaternion,
        best_jtj,
        best_jtr,
        best_sum_sq,
        best_error,
        best_valid_count,
        lambda_damping,
        predicted_reduction,
        10,
        10.0,
        1.0e-7,
        1.0e7,
        result_float,
        result_int,
    )
    torch.testing.assert_close(best_position, candidate_position)
    torch.testing.assert_close(lambda_damping, torch.tensor([1.0e-3], device=device))
    assert result_int.tolist() == [2, 25, 1, 0]


def test_pose_refiner_reports_cholesky_failure() -> None:
    """Return the shared linear-solve failure status without changing the pose."""
    device = torch.device("cuda:0")
    position = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32, device=device)
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    jtj = -torch.eye(6, dtype=torch.float32, device=device)
    jtr = torch.ones(6, dtype=torch.float32, device=device)
    lambda_damping = torch.tensor([1.0e-3], dtype=torch.float32, device=device)
    candidate_position = torch.full_like(position, 9.0)
    candidate_quaternion = torch.full_like(quaternion, 9.0)
    predicted_reduction = torch.full((), 9.0, dtype=torch.float32, device=device).view(1)
    result_int = torch.zeros(4, dtype=torch.int32, device=device)

    launch_pose_refiner_solve_candidate(
        position,
        quaternion,
        jtj,
        jtr,
        lambda_damping,
        candidate_position,
        candidate_quaternion,
        predicted_reduction,
        result_int,
    )

    assert result_int[3].item() == 3
    torch.testing.assert_close(candidate_position, torch.full_like(position, 9.0))
    torch.testing.assert_close(candidate_quaternion, torch.full_like(quaternion, 9.0))


def test_block_sparse_pose_refiner_replays_shared_lm_graph() -> None:
    """Refine the same integrated plane twice through the captured Python path."""
    device = torch.device("cuda:0")
    image_height = 96
    image_width = 128
    integrator = BlockSparseESDFIntegrator(
        BlockSparseESDFIntegratorCfg(
            voxel_size=0.03,
            origin=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
            esdf_voxel_size=0.03,
            esdf_grid_shape=(32, 32, 32),
            truncation_distance=0.09,
            max_blocks=4096,
            hash_capacity=8192,
            depth_minimum_distance=0.1,
            depth_maximum_distance=3.0,
            minimum_tsdf_weight=0.1,
            grid_shape=(32, 32, 32),
            image_height=image_height,
            image_width=image_width,
            enable_static=False,
            seeding_method="gather",
            edt_solver="pba",
            num_cameras=1,
            block_size=4,
            max_visible_blocks_per_integration=4096,
            max_support_pixels_per_block_camera=8,
            use_cuda_graph=False,
        )
    )
    depth = torch.full((1, image_height, image_width), 0.3, dtype=torch.float32, device=device)
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
    position = torch.tensor([[0.45, 0.45, 0.0]], dtype=torch.float32, device=device)
    quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    observation = CameraObservation(
        depth_image=depth,
        rgb_image=rgb,
        intrinsics=intrinsics,
        pose=Pose(position, quaternion),
    )
    for _ in range(3):
        integrator.integrate(observation)

    refiner = BlockSparseRaycastPoseRefiner(
        integrator,
        BlockSparseRaycastRefinerCfg(
            n_points=image_height * image_width,
            minimum_valid_depth_pixels=100,
            max_iterations=8,
            inner_iterations=4,
            n_samples_per_ray=1,
            minimum_tsdf_weight=0.1,
        ),
    )
    initial_position = position.clone()
    initial_position[:, 2].add_(0.015)
    outputs = []
    for _ in range(2):
        output_pose, error, iterations = refiner.refine_pose(
            depth[0],
            intrinsics[0],
            Pose(initial_position.clone(), quaternion.clone()),
        )
        outputs.append((output_pose.position.clone(), output_pose.quaternion.clone(), error))
        assert iterations == 2
        assert torch.isfinite(output_pose.position).all()
        assert torch.isfinite(output_pose.quaternion).all()
        assert error < float("inf")

    torch.testing.assert_close(outputs[0][0], outputs[1][0], atol=1.0e-6, rtol=1.0e-6)
    torch.testing.assert_close(outputs[0][1], outputs[1][1], atol=1.0e-6, rtol=1.0e-6)
    assert outputs[0][2] == pytest.approx(outputs[1][2], abs=1.0e-6)
