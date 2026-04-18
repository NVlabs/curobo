# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for DetectionResult dataclass."""

# Standard Library
from dataclasses import fields

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.perception.pose_estimation.detection_result import DetectionResult
from curobo._src.types.pose import Pose


class TestDetectionResultInitialization:
    """Test DetectionResult initialization and data structure."""

    def test_basic_initialization(self):
        """Test basic DetectionResult creation."""
        pose = Pose(
            position=torch.tensor([[1.0, 2.0, 3.0]]),
            quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )

        result = DetectionResult(
            pose=pose, config=None, confidence=0.95, alignment_error=0.01, n_iterations=25
        )

        assert result.pose == pose
        assert result.config is None
        assert result.confidence == 0.95
        assert result.alignment_error == 0.01
        assert result.n_iterations == 25

    def test_with_joint_position(self):
        """Test DetectionResult with joint configuration (for robots)."""
        pose = Pose(
            position=torch.tensor([[0.5, 0.0, 0.8]]),
            quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )

        joint_position = torch.tensor([0.0, -1.57, 1.57, 0.0, 1.57, 0.0])

        result = DetectionResult(
            pose=pose, config=joint_position, confidence=0.88, alignment_error=0.02, n_iterations=40
        )

        assert result.pose == pose
        assert torch.allclose(result.config, joint_position)
        assert result.confidence == 0.88
        assert result.alignment_error == 0.02
        assert result.n_iterations == 40

    def test_rigid_object_detection(self):
        """Test DetectionResult for rigid object (config=None)."""
        pose = Pose(
            position=torch.tensor([[0.0, 0.0, 1.0]]),
            quaternion=torch.tensor([[0.707, 0.0, 0.0, 0.707]]),
        )

        result = DetectionResult(
            pose=pose, config=None, confidence=0.92, alignment_error=0.015, n_iterations=30
        )

        # For rigid objects, config should be None
        assert result.config is None
        assert result.pose == pose


class TestDetectionResultProperties:
    """Test DetectionResult properties and validation."""

    def test_confidence_range(self):
        """Test confidence is typically in [0, 1] range."""
        pose = Pose(
            position=torch.tensor([[0.0, 0.0, 0.0]]),
            quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )

        # High confidence
        result_high = DetectionResult(
            pose=pose, config=None, confidence=0.99, alignment_error=0.001, n_iterations=10
        )
        assert 0.0 <= result_high.confidence <= 1.0

        # Medium confidence
        result_med = DetectionResult(
            pose=pose, config=None, confidence=0.50, alignment_error=0.05, n_iterations=30
        )
        assert 0.0 <= result_med.confidence <= 1.0

        # Low confidence
        result_low = DetectionResult(
            pose=pose, config=None, confidence=0.10, alignment_error=0.20, n_iterations=50
        )
        assert 0.0 <= result_low.confidence <= 1.0

    def test_alignment_error_positive(self):
        """Test alignment error is non-negative."""
        pose = Pose(
            position=torch.tensor([[0.0, 0.0, 0.0]]),
            quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )

        result = DetectionResult(
            pose=pose, config=None, confidence=0.80, alignment_error=0.03, n_iterations=20
        )

        assert result.alignment_error >= 0.0

    def test_n_iterations_positive(self):
        """Test iteration count is positive."""
        pose = Pose(
            position=torch.tensor([[0.0, 0.0, 0.0]]),
            quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )

        result = DetectionResult(
            pose=pose, config=None, confidence=0.85, alignment_error=0.02, n_iterations=15
        )

        assert result.n_iterations > 0

    def test_confidence_vs_alignment_error_correlation(self):
        """Test that high confidence typically correlates with low alignment error."""
        pose = Pose(
            position=torch.tensor([[0.0, 0.0, 0.0]]),
            quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )

        # Good result: high confidence, low error
        good_result = DetectionResult(
            pose=pose, config=None, confidence=0.95, alignment_error=0.005, n_iterations=20
        )

        # Poor result: low confidence, high error
        poor_result = DetectionResult(
            pose=pose, config=None, confidence=0.30, alignment_error=0.15, n_iterations=50
        )

        # Verify correlation (not enforced, just typical behavior)
        assert good_result.confidence > poor_result.confidence
        assert good_result.alignment_error < poor_result.alignment_error


class TestDetectionResultWithPoseOperations:
    """Test DetectionResult with various Pose operations."""

    def test_identity_pose(self):
        """Test detection with identity pose."""
        pose = Pose(
            position=torch.tensor([[0.0, 0.0, 0.0]]),
            quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )

        result = DetectionResult(
            pose=pose, config=None, confidence=1.0, alignment_error=0.0, n_iterations=1
        )

        assert torch.allclose(result.pose.position, torch.tensor([[0.0, 0.0, 0.0]]))
        assert torch.allclose(result.pose.quaternion, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))

    def test_translated_pose(self):
        """Test detection with translated pose."""
        pose = Pose(
            position=torch.tensor([[1.5, -0.5, 2.0]]),
            quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )

        result = DetectionResult(
            pose=pose, config=None, confidence=0.90, alignment_error=0.01, n_iterations=25
        )

        assert torch.allclose(result.pose.position, torch.tensor([[1.5, -0.5, 2.0]]))

    def test_rotated_pose(self):
        """Test detection with rotated pose."""
        # 90-degree rotation around Z-axis (properly normalized)
        sqrt2_inv = 1.0 / torch.sqrt(torch.tensor(2.0))
        pose = Pose(
            position=torch.tensor([[0.0, 0.0, 0.0]]),
            quaternion=torch.tensor([[sqrt2_inv, 0.0, 0.0, sqrt2_inv]]),
            normalize_rotation=True,
        )

        result = DetectionResult(
            pose=pose, config=None, confidence=0.85, alignment_error=0.02, n_iterations=30
        )

        # Quaternion should be normalized
        quat_norm = torch.norm(result.pose.quaternion)
        assert torch.isclose(quat_norm, torch.tensor(1.0), atol=1e-4)


class TestDataclassFieldsDetectionResult:
    """Test dataclass field definitions and types for DetectionResult."""

    def test_detection_result_field_count(self):
        """Test DetectionResult has expected number of fields."""
        result_fields = fields(DetectionResult)
        assert len(result_fields) == 6  # pose, config, confidence, alignment_error, n_iterations, compute_time

    def test_detection_result_field_names(self):
        """Test DetectionResult has all expected field names."""
        result_fields = {f.name for f in fields(DetectionResult)}

        expected_fields = {"pose", "config", "confidence", "alignment_error", "n_iterations", "compute_time"}

        assert result_fields == expected_fields


class TestEdgeCasesDetectionResult:
    """Test edge cases and boundary conditions for DetectionResult."""

    def test_zero_confidence(self):
        """Test detection result with zero confidence."""
        pose = Pose(
            position=torch.tensor([[0.0, 0.0, 0.0]]),
            quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )

        result = DetectionResult(
            pose=pose, config=None, confidence=0.0, alignment_error=1.0, n_iterations=50
        )

        assert result.confidence == 0.0

    def test_perfect_confidence(self):
        """Test detection result with perfect confidence."""
        pose = Pose(
            position=torch.tensor([[0.0, 0.0, 0.0]]),
            quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        )

        result = DetectionResult(
            pose=pose, config=None, confidence=1.0, alignment_error=0.0, n_iterations=1
        )

        assert result.confidence == 1.0
        assert result.alignment_error == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
