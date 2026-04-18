# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SDFPoseDetector CUDA graph support.

Tests that the GraphExecutor properly updates state when using CUDA graphs.
"""

import pytest
import trimesh

from curobo._src.perception.pose_estimation.mesh_robot import RobotMesh
from curobo._src.perception.pose_estimation.sdf_pose_detector import SDFPoseDetector
from curobo._src.perception.pose_estimation.sdf_pose_detector_cfg import SDFDetectorCfg
from curobo._src.types.pose import Pose
from curobo._src.util.warp import init_warp


@pytest.fixture(scope="module")
def setup_warp():
    """Initialize Warp once for all tests."""
    init_warp()


@pytest.fixture
def box_mesh(setup_warp):
    """Create a simple box mesh for testing."""
    box = trimesh.creation.box(extents=[0.2, 0.2, 0.2])
    return RobotMesh.from_trimesh(box, device="cuda:0")


@pytest.fixture
def test_points(box_mesh):
    """Sample points from mesh and apply a known offset."""
    points, _ = box_mesh.sample_surface_points(1000)
    points = points.to("cuda:0")

    # Ground truth pose offset
    gt_pose = Pose.from_list([0.02, -0.01, 0.005, 1, 0, 0, 0])
    gt_pose = Pose(
        position=gt_pose.position.to("cuda:0"),
        quaternion=gt_pose.quaternion.to("cuda:0"),
    )

    transformed_points = gt_pose.transform_points(points)
    return transformed_points, gt_pose


class TestSDFCudaGraph:
    """Tests for CUDA graph support in SDFPoseDetector."""

    def test_refine_inner_iterations_updates_state_without_graph(
        self, box_mesh, test_points
    ):
        """Test that _refine_inner_iterations updates state correctly without CUDA graph."""
        transformed_points, gt_pose = test_points

        config = SDFDetectorCfg(
            max_iterations=20,
            inner_iterations=5,
            use_cuda_graph=False,
        )
        detector = SDFPoseDetector(box_mesh, config)

        identity = Pose.from_list([0, 0, 0, 1, 0, 0, 0])
        identity = Pose(
            position=identity.position.to("cuda:0"),
            quaternion=identity.quaternion.to("cuda:0"),
        )

        state = detector._setup_refinement(transformed_points, identity)
        initial_error = state.best_error.item()

        # Call _refine_inner_iterations directly
        state_out = detector._refine_inner_iterations(state)
        final_error = state_out.best_error.item()

        # Error should decrease
        assert final_error < initial_error * 0.9, (
            f"Error should decrease: {initial_error:.4f} -> {final_error:.4f}"
        )

        # Position should be close to ground truth
        pos_error = (state_out.best_position - gt_pose.position.squeeze()).norm().item()
        assert pos_error < 0.001, f"Position error {pos_error*1000:.2f}mm should be < 1mm"

    def test_graph_executor_updates_state_without_graph(
        self, box_mesh, test_points
    ):
        """Test that GraphExecutor updates state correctly when CUDA graph is disabled."""
        transformed_points, gt_pose = test_points

        config = SDFDetectorCfg(
            max_iterations=100,
            inner_iterations=20,
            use_cuda_graph=False,
        )
        detector = SDFPoseDetector(box_mesh, config)

        identity = Pose.from_list([0, 0, 0, 1, 0, 0, 0])
        identity = Pose(
            position=identity.position.to("cuda:0"),
            quaternion=identity.quaternion.to("cuda:0"),
        )

        state = detector._setup_refinement(transformed_points, identity)
        initial_error = state.best_error.item()

        # Call via GraphExecutor (graph disabled)
        state_out = detector._refine_inner_executor(state.clone())
        final_error = state_out.best_error.item()

        # Error should decrease significantly
        assert final_error < initial_error * 0.9, (
            f"Error should decrease: {initial_error:.4f} -> {final_error:.4f}"
        )


    def test_multiple_graph_replays_converge(
        self, box_mesh, test_points
    ):
        """Test that multiple CUDA graph replays continue to improve the result.

        THIS TEST CURRENTLY FAILS - state doesn't update between replays.
        """
        transformed_points, gt_pose = test_points

        config = SDFDetectorCfg(
            max_iterations=20,
            inner_iterations=5,
            use_cuda_graph=True,
        )
        detector = SDFPoseDetector(box_mesh, config)

        identity = Pose.from_list([0, 0, 0, 1, 0, 0, 0])
        identity = Pose(
            position=identity.position.to("cuda:0"),
            quaternion=identity.quaternion.to("cuda:0"),
        )

        state = detector._setup_refinement(transformed_points, identity)

        errors = [state.best_error.item()]
        positions = [state.best_position.clone()]

        # Multiple calls
        for i in range(3):
            state = detector._refine_inner_executor(state.clone())
            errors.append(state.best_error.item())
            positions.append(state.best_position.clone())

        # Errors should monotonically decrease (or stay same if converged)
        for i in range(1, len(errors)):
            assert errors[i] <= errors[i-1] + 1e-6, (
                f"Error should not increase: call {i-1}={errors[i-1]:.4f} -> call {i}={errors[i]:.4f}"
            )

        # Final error should be much smaller than initial
        assert errors[-1] < errors[0] * 0.9, (
            f"Final error {errors[-1]:.4f} should be < 90% of initial {errors[0]:.4f}"
        )

    def test_cuda_graph_matches_no_graph_result(
        self, box_mesh, test_points
    ):
        """Test that CUDA graph produces same result as no-graph version.

        THIS TEST CURRENTLY FAILS - graph version doesn't converge properly.
        """
        transformed_points, gt_pose = test_points

        identity = Pose.from_list([0, 0, 0, 1, 0, 0, 0])
        identity = Pose(
            position=identity.position.to("cuda:0"),
            quaternion=identity.quaternion.to("cuda:0"),
        )

        # Run without CUDA graph
        config_no_graph = SDFDetectorCfg(
            max_iterations=100,
            inner_iterations=25,
            use_cuda_graph=False,
        )
        detector_no_graph = SDFPoseDetector(box_mesh, config_no_graph)
        result_no_graph = detector_no_graph.detect_from_points(
            transformed_points, initial_pose=identity
        )

        # Run with CUDA graph
        config_graph = SDFDetectorCfg(
            max_iterations=100,
            inner_iterations=25,
            use_cuda_graph=True,
        )
        detector_graph = SDFPoseDetector(box_mesh, config_graph)
        result_graph = detector_graph.detect_from_points(
            transformed_points, initial_pose=identity
        )

        # Errors should be similar (within 10x)
        assert result_graph.alignment_error <= result_no_graph.alignment_error * 2, (
            f"Graph error {result_graph.alignment_error*1000:.2f}mm should be similar to "
            f"no-graph error {result_no_graph.alignment_error*1000:.2f}mm"
        )


