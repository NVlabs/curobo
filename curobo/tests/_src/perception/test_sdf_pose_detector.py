# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for SDF-based pose detector.

Tests the SDFPoseDetector class for pose estimation using mesh SDF queries.
"""

# Third Party
import numpy as np
import pytest
import torch
import trimesh

from curobo._src.perception.pose_estimation.mesh_robot import RobotMesh
from curobo._src.perception.pose_estimation.sdf_pose_detector import SDFPoseDetector

# CuRobo
from curobo._src.perception.pose_estimation.sdf_pose_detector_cfg import SDFDetectorCfg
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.types.pose import Pose
from curobo._src.util.warp import init_warp
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml

# Initialize Warp at module load
init_warp()


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(scope="module")
def simple_box_mesh():
    """Create a simple box mesh for testing."""
    return trimesh.creation.box(extents=[0.2, 0.2, 0.2])


@pytest.fixture(scope="module")
def sphere_mesh():
    """Create a sphere mesh for testing."""
    return trimesh.creation.icosphere(subdivisions=3, radius=0.1)


@pytest.fixture(scope="module")
def franka_kinematics():
    """Create Franka kinematics for testing."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    cfg = KinematicsCfg.from_robot_yaml_file(
        robot_data,
        tool_frames=robot_data["robot_cfg"]["kinematics"]["mesh_link_names"],
    )
    return Kinematics(cfg)


# ============================================================================
# Helper Functions
# ============================================================================


def create_perturbed_pose(
    translation: tuple = (0.0, 0.0, 0.0),
    rotation_axis: tuple = (0.0, 0.0, 1.0),
    rotation_angle: float = 0.0,
) -> Pose:
    """Create a pose with given translation and axis-angle rotation."""
    position = torch.tensor([translation], dtype=torch.float32)

    # Convert axis-angle to quaternion
    axis = np.array(rotation_axis)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    half_angle = rotation_angle / 2
    w = np.cos(half_angle)
    xyz = axis * np.sin(half_angle)
    quaternion = torch.tensor([[w, xyz[0], xyz[1], xyz[2]]], dtype=torch.float32)

    return Pose(position=position, quaternion=quaternion)


def sample_points_on_mesh(
    robot_mesh: RobotMesh,
    n_points: int,
    transform: Pose,
) -> torch.Tensor:
    """Sample points from mesh surface and transform to world frame."""
    points, _ = robot_mesh.sample_surface_points(n_points)
    # Transform points to world frame using the given pose
    return transform.transform_points(points.unsqueeze(0)).squeeze(0)


# ============================================================================
# Test Basic Creation
# ============================================================================


class TestBasicCreation:
    """Test basic detector creation."""

    def test_create_with_defaults(self, simple_box_mesh):
        """Test creating detector with default config."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")
        detector = SDFPoseDetector(robot_mesh)

        assert detector is not None
        assert detector.config.max_iterations == 100

    def test_create_with_custom_config(self, simple_box_mesh):
        """Test creating detector with custom config."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")
        config = SDFDetectorCfg(max_iterations=50, n_points=2000)
        detector = SDFPoseDetector(robot_mesh, config)

        assert detector.config.max_iterations == 50
        assert detector.config.n_points == 2000


# ============================================================================
# Test Identity Pose (Perfect Alignment)
# ============================================================================


class TestIdentityPose:
    """Test detection with identity pose (points already on surface)."""

    def test_converges_quickly(self, simple_box_mesh):
        """Test that detector converges quickly when already aligned."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")
        detector = SDFPoseDetector(robot_mesh)

        # Sample points at identity pose
        identity_pose = Pose.from_list([0, 0, 0, 1, 0, 0, 0])
        observed_points, _ = robot_mesh.sample_surface_points(1000)
        observed_points = observed_points.to("cuda:0")

        result = detector.detect_from_points(observed_points, initial_pose=identity_pose)

        # Should converge quickly with low error
        assert result.alignment_error < 0.001

    def test_pose_unchanged(self, simple_box_mesh):
        """Test that pose stays unchanged when already aligned."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")
        detector = SDFPoseDetector(robot_mesh)

        identity_pose = Pose.from_list([0, 0, 0, 1, 0, 0, 0])
        observed_points, _ = robot_mesh.sample_surface_points(1000)
        observed_points = observed_points.to("cuda:0")

        result = detector.detect_from_points(observed_points, initial_pose=identity_pose)

        # Position should be near zero
        assert torch.allclose(
            result.pose.position, identity_pose.position.to("cuda:0"), atol=0.001
        )


# ============================================================================
# Test Small Perturbations
# ============================================================================


class TestSmallPerturbations:
    """Test detection with small pose perturbations."""

    def test_recovers_from_small_translation(self, simple_box_mesh):
        """Test recovery from small translation perturbation."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")
        detector = SDFPoseDetector(robot_mesh)

        # True pose (points are sampled here)
        true_pose = Pose.from_list([0, 0, 0, 1, 0, 0, 0])

        # Sample points at true pose
        observed_points, _ = robot_mesh.sample_surface_points(2000)
        observed_points = observed_points.to("cuda:0")

        # Start from perturbed pose
        initial_pose = Pose.from_list([0.01, 0.005, -0.01, 1, 0, 0, 0])

        result = detector.detect_from_points(observed_points, initial_pose=initial_pose)

        # Should recover close to true pose
        assert result.alignment_error < 0.01
        assert result.confidence > 0.5

    def test_recovers_from_small_rotation(self, sphere_mesh):
        """Test recovery from small rotation perturbation."""
        robot_mesh = RobotMesh.from_trimesh(sphere_mesh, device="cuda:0")
        detector = SDFPoseDetector(robot_mesh)

        # Sample points at identity
        observed_points, _ = robot_mesh.sample_surface_points(2000)
        observed_points = observed_points.to("cuda:0")

        # Start from rotated pose (5 degrees around z)
        angle = 0.087  # ~5 degrees
        initial_pose = create_perturbed_pose(rotation_axis=(0, 0, 1), rotation_angle=angle)
        initial_pose = Pose(
            position=initial_pose.position.to("cuda:0"),
            quaternion=initial_pose.quaternion.to("cuda:0"),
        )

        result = detector.detect_from_points(observed_points, initial_pose=initial_pose)

        # For sphere, rotation doesn't matter much, just check it converges
        assert result.alignment_error < 0.01


# ============================================================================
# Test Larger Perturbations
# ============================================================================


class TestLargerPerturbations:
    """Test detection with larger pose perturbations."""

    def test_recovers_from_medium_translation(self, simple_box_mesh):
        """Test recovery from medium translation (3cm)."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")
        config = SDFDetectorCfg(max_iterations=200, distance_threshold=0.1)
        detector = SDFPoseDetector(robot_mesh, config)

        # Sample points at identity
        observed_points, _ = robot_mesh.sample_surface_points(3000)
        observed_points = observed_points.to("cuda:0")

        # Start from perturbed pose
        initial_pose = Pose.from_list([0.03, 0.0, 0.0, 1, 0, 0, 0])
        initial_pose = Pose(
            position=initial_pose.position.to("cuda:0"),
            quaternion=initial_pose.quaternion.to("cuda:0"),
        )

        result = detector.detect_from_points(observed_points, initial_pose=initial_pose)

        # Should recover with reasonable error
        assert result.alignment_error < 0.02


# ============================================================================
# Test with Articulated Robot
# ============================================================================


class TestArticulatedRobot:
    """Test detection with articulated robot mesh."""

    def test_basic_detection(self, franka_kinematics):
        """Test basic detection on articulated robot."""
        robot_mesh = RobotMesh.from_kinematics(franka_kinematics, device="cuda:0")
        detector = SDFPoseDetector(robot_mesh)

        # Sample points at zero configuration
        observed_points, _ = robot_mesh.sample_surface_points(2000)
        observed_points = observed_points.to("cuda:0")

        # Start from identity pose
        initial_pose = Pose.from_list([0, 0, 0, 1, 0, 0, 0])
        initial_pose = Pose(
            position=initial_pose.position.to("cuda:0"),
            quaternion=initial_pose.quaternion.to("cuda:0"),
        )

        result = detector.detect_from_points(observed_points, initial_pose=initial_pose)

        assert result.alignment_error < 0.01
        assert result.confidence > 0.5

    def test_with_joint_angles(self, franka_kinematics):
        """Test detection with updated joint angles."""
        robot_mesh = RobotMesh.from_kinematics(franka_kinematics, device="cuda:0")
        detector = SDFPoseDetector(robot_mesh)

        # Update to new joint configuration
        joint_angles = torch.tensor(
            [0.3, -0.3, 0.2, -1.2, 0.1, 0.8, 0.4],
            device="cuda:0",
        )
        robot_mesh.update(joint_angles)

        # Sample points at this configuration
        observed_points, _ = robot_mesh.sample_surface_points(2000)
        observed_points = observed_points.to("cuda:0")

        # Detect with same joint angles
        initial_pose = Pose.from_list([0, 0, 0, 1, 0, 0, 0])
        initial_pose = Pose(
            position=initial_pose.position.to("cuda:0"),
            quaternion=initial_pose.quaternion.to("cuda:0"),
        )

        result = detector.detect_from_points(observed_points, config=joint_angles, initial_pose=initial_pose)

        assert result.alignment_error < 0.01


# ============================================================================
# Test Result Properties
# ============================================================================


class TestResultProperties:
    """Test properties of detection result."""

    def test_result_has_pose(self, simple_box_mesh):
        """Test that result has valid pose."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")
        detector = SDFPoseDetector(robot_mesh)

        observed_points, _ = robot_mesh.sample_surface_points(1000)
        observed_points = observed_points.to("cuda:0")
        initial_pose = Pose.from_list([0, 0, 0, 1, 0, 0, 0])
        initial_pose = Pose(
            position=initial_pose.position.to("cuda:0"),
            quaternion=initial_pose.quaternion.to("cuda:0"),
        )

        result = detector.detect_from_points(observed_points, initial_pose=initial_pose)

        assert result.pose is not None
        assert result.pose.position.shape == (1, 3)
        assert result.pose.quaternion.shape == (1, 4)

    def test_result_has_confidence(self, simple_box_mesh):
        """Test that result has valid confidence."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")
        detector = SDFPoseDetector(robot_mesh)

        observed_points, _ = robot_mesh.sample_surface_points(1000)
        observed_points = observed_points.to("cuda:0")
        initial_pose = Pose.from_list([0, 0, 0, 1, 0, 0, 0])
        initial_pose = Pose(
            position=initial_pose.position.to("cuda:0"),
            quaternion=initial_pose.quaternion.to("cuda:0"),
        )

        result = detector.detect_from_points(observed_points, initial_pose=initial_pose)

        assert 0 <= result.confidence <= 1.0

    def test_result_has_alignment_error(self, simple_box_mesh):
        """Test that result has valid alignment error."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")
        detector = SDFPoseDetector(robot_mesh)

        observed_points, _ = robot_mesh.sample_surface_points(1000)
        observed_points = observed_points.to("cuda:0")
        initial_pose = Pose.from_list([0, 0, 0, 1, 0, 0, 0])
        initial_pose = Pose(
            position=initial_pose.position.to("cuda:0"),
            quaternion=initial_pose.quaternion.to("cuda:0"),
        )

        result = detector.detect_from_points(observed_points, initial_pose=initial_pose)

        assert result.alignment_error >= 0


# ============================================================================
# Test Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases."""

    def test_few_points(self, simple_box_mesh):
        """Test with very few observed points."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")
        config = SDFDetectorCfg(n_points=50, min_valid_ratio=0.1)
        detector = SDFPoseDetector(robot_mesh, config)

        observed_points, _ = robot_mesh.sample_surface_points(50)
        observed_points = observed_points.to("cuda:0")
        initial_pose = Pose.from_list([0, 0, 0, 1, 0, 0, 0])
        initial_pose = Pose(
            position=initial_pose.position.to("cuda:0"),
            quaternion=initial_pose.quaternion.to("cuda:0"),
        )

        result = detector.detect_from_points(observed_points, initial_pose=initial_pose)

        # Should still produce a result
        assert result is not None

    def test_many_points(self, simple_box_mesh):
        """Test with many observed points (subsampling)."""
        robot_mesh = RobotMesh.from_trimesh(simple_box_mesh, device="cuda:0")
        config = SDFDetectorCfg(n_points=500)
        detector = SDFPoseDetector(robot_mesh, config)

        # More points than config allows
        observed_points, _ = robot_mesh.sample_surface_points(5000)
        observed_points = observed_points.to("cuda:0")
        initial_pose = Pose.from_list([0, 0, 0, 1, 0, 0, 0])
        initial_pose = Pose(
            position=initial_pose.position.to("cuda:0"),
            quaternion=initial_pose.quaternion.to("cuda:0"),
        )

        result = detector.detect_from_points(observed_points, initial_pose=initial_pose)

        assert result.alignment_error < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

