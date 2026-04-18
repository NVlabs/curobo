# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for perception utility functions."""

import pytest
import torch

from curobo._src.perception.pose_estimation.util import (
    compute_pose_point_to_plane_cholesky,
    compute_pose_point_to_plane_svd,
    extract_observed_points,
    find_nearest_neighbors,
    huber_loss,
    resample_points,
)
from curobo._src.types.camera import CameraObservation
from curobo._src.types.pose import Pose


def _svd_to_matrix(source, target, normals, **kwargs):
    """Helper: call SVD solver and return a 4x4 transform matrix."""
    position, quaternion = compute_pose_point_to_plane_svd(
        source, target, normals, **kwargs
    )
    return Pose(position=position, quaternion=quaternion).get_matrix().squeeze(0)


class TestHuberLoss:
    """Test Huber loss function."""

    @pytest.fixture
    def device(self):
        """Device fixture."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_huber_loss_small_residuals(self, device):
        """Test quadratic behavior for small residuals."""
        # Small residuals should use quadratic loss: 0.5 * r^2
        residuals = torch.tensor([0.01, 0.005, -0.01], device=device, dtype=torch.float32)
        delta = 0.02

        losses = huber_loss(residuals, delta)

        expected = 0.5 * residuals**2
        assert torch.allclose(losses, expected), "Small residuals should use quadratic loss"

    def test_huber_loss_large_residuals(self, device):
        """Test linear behavior for large residuals."""
        # Large residuals should use linear loss: delta * (|r| - 0.5 * delta)
        residuals = torch.tensor([0.1, -0.05, 0.03], device=device, dtype=torch.float32)
        delta = 0.02

        losses = huber_loss(residuals, delta)

        abs_res = torch.abs(residuals)
        expected_linear = delta * (abs_res - 0.5 * delta)

        # All residuals are larger than delta
        assert torch.allclose(losses, expected_linear), "Large residuals should use linear loss"

    def test_huber_loss_mixed_residuals(self, device):
        """Test mixed small and large residuals."""
        residuals = torch.tensor([0.005, 0.05, -0.01, 0.1], device=device, dtype=torch.float32)
        delta = 0.02

        losses = huber_loss(residuals, delta)

        # Check each manually
        abs_res = torch.abs(residuals)
        expected = torch.zeros_like(residuals)
        for i in range(len(residuals)):
            if abs_res[i] < delta:
                expected[i] = 0.5 * residuals[i]**2
            else:
                expected[i] = delta * (abs_res[i] - 0.5 * delta)

        assert torch.allclose(losses, expected), "Mixed residuals should use appropriate loss"

    def test_huber_loss_zero_residuals(self, device):
        """Test zero residuals."""
        residuals = torch.zeros(5, device=device, dtype=torch.float32)
        delta = 0.02

        losses = huber_loss(residuals, delta)

        assert torch.all(losses == 0), "Zero residuals should have zero loss"

    def test_huber_loss_multidimensional(self, device):
        """Test with multidimensional residuals."""
        residuals = torch.randn(10, 3, device=device, dtype=torch.float32) * 0.05
        delta = 0.02

        losses = huber_loss(residuals, delta)

        assert losses.shape == residuals.shape, "Output shape should match input"


class TestFindNearestNeighbors:
    """Test nearest neighbor finding."""

    @pytest.fixture
    def device(self):
        """Device fixture."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_find_exact_matches(self, device):
        """Test finding exact matches."""
        source = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], device=device, dtype=torch.float32)
        target = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], device=device, dtype=torch.float32)

        indices = find_nearest_neighbors(source, target)

        expected = torch.tensor([0, 1, 2], device=device)
        assert torch.all(indices == expected), "Should find exact matches"

    def test_find_nearest_with_distance(self, device):
        """Test finding nearest with some distance."""
        source = torch.tensor([[0, 0, 0], [1.1, 0, 0]], device=device, dtype=torch.float32)
        target = torch.tensor([[0, 0, 0], [1, 0, 0]], device=device, dtype=torch.float32)

        indices = find_nearest_neighbors(source, target)

        expected = torch.tensor([0, 1], device=device)
        assert torch.all(indices == expected), "Should find nearest neighbors"

    def test_find_with_threshold(self, device):
        """Test distance threshold filtering."""
        source = torch.tensor([[0, 0, 0], [10, 0, 0]], device=device, dtype=torch.float32)
        target = torch.tensor([[0, 0, 0], [1, 0, 0]], device=device, dtype=torch.float32)
        threshold = 2.0

        indices = find_nearest_neighbors(source, target, distance_threshold=threshold)

        assert indices[0] == 0, "First point should match"
        assert indices[1] == -1, "Second point should be rejected by threshold"

    def test_find_empty_target(self, device):
        """Test with empty target."""
        source = torch.tensor([[0, 0, 0], [1, 0, 0]], device=device, dtype=torch.float32)
        target = torch.empty(0, 3, device=device, dtype=torch.float32)

        with pytest.raises((RuntimeError, IndexError)):
            # Should fail because target is empty
            indices = find_nearest_neighbors(source, target)

    def test_find_single_target(self, device):
        """Test with single target point."""
        source = torch.tensor([[0, 0, 0], [1, 0, 0], [2, 0, 0]], device=device, dtype=torch.float32)
        target = torch.tensor([[0.5, 0, 0]], device=device, dtype=torch.float32)

        indices = find_nearest_neighbors(source, target)

        assert torch.all(indices == 0), "All should match single target"


class TestResamplePoints:
    """Test point resampling."""

    @pytest.fixture
    def device(self):
        """Device fixture."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_downsample(self, device):
        """Test downsampling to fewer points."""
        points = torch.randn(100, 3, device=device, dtype=torch.float32)
        target_count = 50

        resampled = resample_points(points, target_count, device)

        assert resampled.shape == (50, 3), "Should downsample to target count"
        assert resampled.device.type == device.type, "Should be on correct device type"

    def test_upsample(self, device):
        """Test upsampling to more points."""
        points = torch.randn(10, 3, device=device, dtype=torch.float32)
        target_count = 50

        resampled = resample_points(points, target_count, device)

        assert resampled.shape == (50, 3), "Should upsample to target count"
        assert resampled.device.type == device.type, "Should be on correct device type"

    def test_exact_count(self, device):
        """Test when point count already matches."""
        points = torch.randn(50, 3, device=device, dtype=torch.float32)
        target_count = 50

        resampled = resample_points(points, target_count, device)

        assert resampled.shape == (50, 3), "Should maintain same count"

    def test_device_inference(self, device):
        """Test device inference from input."""
        points = torch.randn(10, 3, device=device, dtype=torch.float32)
        target_count = 5

        resampled = resample_points(points, target_count)

        assert resampled.device == points.device, "Should infer device from input"

    def test_single_point_upsample(self, device):
        """Test upsampling from single point."""
        points = torch.randn(1, 3, device=device, dtype=torch.float32)
        target_count = 10

        resampled = resample_points(points, target_count, device)

        assert resampled.shape == (10, 3), "Should upsample from single point"


class TestComputeTransformPointToPlane:
    """Test point-to-plane ICP transform computation."""

    @pytest.fixture
    def device(self):
        """Device fixture."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_identity_transform(self, device):
        """Test when source and target are already aligned."""
        # Create aligned points
        source = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], device=device, dtype=torch.float32
        )
        target = source.clone()
        normals = torch.tensor(
            [[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], device=device, dtype=torch.float32
        )

        transform = _svd_to_matrix(source, target, normals)

        # Should be close to identity
        identity = torch.eye(4, device=device, dtype=torch.float32)
        assert torch.allclose(transform, identity, atol=1e-3), "Should be identity transform"

    def test_pure_translation_on_plane(self, device):
        """Test translation on a planar surface with consistent normals."""
        # Create points on a plane (z=0) with consistent upward normals
        torch.manual_seed(42)
        xy = torch.randn(100, 2, device=device, dtype=torch.float32) * 2
        source = torch.cat([xy, torch.zeros(100, 1, device=device)], dim=1)

        # Pure translation
        translation = torch.tensor([0.5, 0.3, 0.2], device=device, dtype=torch.float32)
        target = source + translation

        # All normals point up (perpendicular to plane)
        normals = torch.zeros(100, 3, device=device, dtype=torch.float32)
        normals[:, 2] = 1.0

        transform = _svd_to_matrix(source, target, normals)

        # Apply transform
        source_hom = torch.cat([source, torch.ones(len(source), 1, device=device)], dim=1)
        source_transformed = (transform @ source_hom.T).T[:, :3]

        # For planar surfaces with consistent normals, point-to-plane should work well
        # for the normal direction
        error_z = torch.abs(source_transformed[:, 2] - target[:, 2]).mean()
        assert error_z < 0.05, f"Z translation should be accurate (error={error_z:.4f})"

    def test_small_rotation_on_sphere(self, device):
        """Test small rotation on spherical surface."""
        import math
        # Create points on a sphere (normals point radially outward)
        torch.manual_seed(123)
        source = torch.nn.functional.normalize(
            torch.randn(100, 3, device=device, dtype=torch.float32), dim=1
        )
        normals = source.clone()  # For sphere, normals = positions

        # Small rotation (5 degrees) - point-to-plane works better with small transforms
        angle = math.pi / 36  # 5 degrees
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        R_true = torch.tensor(
            [
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1],
            ],
            device=device,
            dtype=torch.float32,
        )

        target = (R_true @ source.T).T
        normals_rotated = (R_true @ normals.T).T

        transform = _svd_to_matrix(source, target, normals_rotated)

        # Apply transform
        source_hom = torch.cat([source, torch.ones(len(source), 1, device=device)], dim=1)
        source_transformed = (transform @ source_hom.T).T[:, :3]

        # Check alignment improves
        error_before = torch.norm(source - target, dim=1).mean()
        error_after = torch.norm(source_transformed - target, dim=1).mean()

        # Point-to-plane ICP is iterative - single step may not fully converge
        # But should at least move in the right direction or stay stable
        assert error_after <= error_before * 1.1, \
            f"Should not make alignment worse (before={error_before:.4f}, after={error_after:.4f})"

        # Verify transform is valid
        R = transform[:3, :3]
        det = torch.det(R)
        assert 0.9 < det < 1.1, f"Rotation determinant should be ~1, got {det:.4f}"

    def test_with_huber_weights(self, device):
        """Test with Huber loss weighting."""
        source = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 10, 10]], device=device, dtype=torch.float32
        )
        target = torch.tensor(
            [[0, 0, 0.1], [1, 0, 0.1], [0, 1, 0.1], [10, 10, 100]], device=device, dtype=torch.float32
        )
        normals = torch.tensor(
            [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]], device=device, dtype=torch.float32
        )

        # Without Huber (outlier affects result)
        transform_no_huber = _svd_to_matrix(
            source, target, normals, use_huber=False
        )

        # With Huber (outlier downweighted)
        transform_huber = _svd_to_matrix(
            source, target, normals, use_huber=True, huber_delta=1.0
        )

        # Transforms should be different (Huber should be less affected by outlier)
        assert not torch.allclose(transform_no_huber, transform_huber, atol=1e-3), \
            "Huber should produce different result"

    def test_with_custom_weights(self, device):
        """Test with custom weights."""
        source = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]], device=device, dtype=torch.float32
        )
        target = torch.tensor(
            [[0, 0, 0.1], [1, 0, 0.1], [0, 1, 0.1]], device=device, dtype=torch.float32
        )
        normals = torch.tensor(
            [[0, 0, 1], [0, 0, 1], [0, 0, 1]], device=device, dtype=torch.float32
        )
        weights = torch.tensor([1.0, 1.0, 0.1], device=device, dtype=torch.float32)

        transform = _svd_to_matrix(source, target, normals, weights=weights)

        # Should compute successfully
        assert transform.shape == (4, 4), "Should return 4x4 transform"

    def test_reflection_handling(self, device):
        """Test that reflections are properly handled."""
        # Create a case that might produce reflection
        source = torch.randn(10, 3, device=device, dtype=torch.float32)
        target = source + torch.randn(10, 3, device=device, dtype=torch.float32) * 0.01
        normals = torch.nn.functional.normalize(
            torch.randn(10, 3, device=device, dtype=torch.float32), dim=1
        )

        transform = _svd_to_matrix(source, target, normals)

        # Check determinant is positive (not reflection)
        R = transform[:3, :3]
        det = torch.det(R)
        assert det > 0, "Should not produce reflection (det should be positive)"
        assert torch.abs(det - 1.0) < 0.1, "Determinant should be close to 1"

    def test_iterative_convergence(self, device):
        """Test that multiple ICP iterations improve alignment."""
        import math
        torch.manual_seed(456)

        # Create a surface with points and normals
        source = torch.randn(200, 3, device=device, dtype=torch.float32) * 2
        normals = torch.nn.functional.normalize(torch.randn(200, 3, device=device), dim=1)

        # Apply known transform
        angle = math.pi / 8  # 22.5 degrees
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        R_true = torch.tensor(
            [[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]],
            device=device,
            dtype=torch.float32,
        )
        t_true = torch.tensor([0.3, 0.4, 0.2], device=device, dtype=torch.float32)

        target = (R_true @ source.T).T + t_true
        normals_transformed = (R_true @ normals.T).T

        # Run iterative ICP
        T_current = torch.eye(4, device=device, dtype=torch.float32)
        errors = []

        for iteration in range(10):
            # Transform source with current estimate
            source_hom = torch.cat([source, torch.ones(len(source), 1, device=device)], dim=1)
            source_current = (T_current @ source_hom.T).T[:, :3]
            normals_current = (T_current[:3, :3] @ normals.T).T

            # Compute error before update
            error = torch.norm(source_current - target, dim=1).mean().item()
            errors.append(error)

            # Compute transform update
            T_update = _svd_to_matrix(
                source_current, target, normals_transformed, use_huber=True, huber_delta=0.1
            )

            # Apply update
            T_current = T_update @ T_current

            # Early stopping
            if len(errors) > 1 and abs(errors[-1] - errors[-2]) < 1e-5:
                break

        # Verify convergence
        assert len(errors) >= 3, "Should run multiple iterations"

        # Print error history for debugging
        print(f"\nICP Convergence: {len(errors)} iterations")
        print(f"Errors: {[f'{e:.4f}' for e in errors]}")

        # Should show significant improvement
        assert errors[-1] < errors[0] * 0.5, \
            f"Should converge significantly (initial={errors[0]:.4f}, final={errors[-1]:.4f})"

        # Verify generally decreasing trend (allow some noise)
        # Check that later half has lower average than first half
        mid = len(errors) // 2
        first_half_avg = sum(errors[:mid]) / mid
        second_half_avg = sum(errors[mid:]) / (len(errors) - mid)
        assert second_half_avg < first_half_avg, \
            f"Should show decreasing trend (first_half={first_half_avg:.4f}, second_half={second_half_avg:.4f})"

        # Final error should be reasonable for ICP (not perfect, but improved)
        assert errors[-1] < 0.3, f"Final error should be reasonable (got {errors[-1]:.4f})"


class TestComputePosePointToPlaneCholesky:
    """Test Cholesky-based point-to-plane ICP transform computation."""

    @pytest.fixture
    def device(self):
        """Device fixture."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_returns_position_and_quaternion(self, device):
        """Test that function returns position and quaternion tuple."""
        source = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]], device=device, dtype=torch.float32
        )
        target = source.clone()
        normals = torch.tensor(
            [[0, 0, 1], [0, 0, 1], [0, 0, 1]], device=device, dtype=torch.float32
        )

        position, quaternion = compute_pose_point_to_plane_cholesky(source, target, normals)

        assert position.shape == (3,), "Position should be [3]"
        assert quaternion.shape == (4,), "Quaternion should be [4] (wxyz)"

    def test_identity_transform(self, device):
        """Test when source and target are already aligned."""
        source = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], device=device, dtype=torch.float32
        )
        target = source.clone()
        normals = torch.tensor(
            [[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]], device=device, dtype=torch.float32
        )

        position, quaternion = compute_pose_point_to_plane_cholesky(source, target, normals)

        # Position should be near zero
        assert torch.allclose(position, torch.zeros(3, device=device), atol=1e-3), \
            f"Position should be zero, got {position}"

        # Quaternion should be identity [1, 0, 0, 0]
        identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
        assert torch.allclose(quaternion.abs(), identity_quat.abs(), atol=1e-2), \
            f"Quaternion should be identity, got {quaternion}"

    def test_small_translation(self, device):
        """Test small translation detection."""
        torch.manual_seed(42)
        source = torch.randn(50, 3, device=device, dtype=torch.float32)

        # Small translation along z
        translation = torch.tensor([0.0, 0.0, 0.1], device=device, dtype=torch.float32)
        target = source + translation

        # Normals pointing up
        normals = torch.zeros(50, 3, device=device, dtype=torch.float32)
        normals[:, 2] = 1.0

        position, quaternion = compute_pose_point_to_plane_cholesky(source, target, normals)

        # Should detect z translation
        assert abs(position[2] - 0.1) < 0.05, f"Should detect z translation, got {position[2]}"

    def test_with_huber_weights_none(self, device):
        """Test Huber loss with weights=None (bug regression test)."""
        source = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 10, 10]], device=device, dtype=torch.float32
        )
        target = torch.tensor(
            [[0, 0, 0.1], [1, 0, 0.1], [0, 1, 0.1], [10, 10, 100]], device=device, dtype=torch.float32
        )
        normals = torch.tensor(
            [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]], device=device, dtype=torch.float32
        )

        # This should not raise TypeError (regression test for weights=None bug)
        position, quaternion = compute_pose_point_to_plane_cholesky(
            source, target, normals, use_huber=True, huber_delta=1.0
        )

        assert position.shape == (3,), "Should return valid position"
        assert quaternion.shape == (4,), "Should return valid quaternion"

    def test_with_custom_weights(self, device):
        """Test with custom weights."""
        source = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0]], device=device, dtype=torch.float32
        )
        target = torch.tensor(
            [[0, 0, 0.1], [1, 0, 0.1], [0, 1, 0.1]], device=device, dtype=torch.float32
        )
        normals = torch.tensor(
            [[0, 0, 1], [0, 0, 1], [0, 0, 1]], device=device, dtype=torch.float32
        )
        weights = torch.tensor([1.0, 1.0, 0.1], device=device, dtype=torch.float32)

        position, quaternion = compute_pose_point_to_plane_cholesky(
            source, target, normals, weights=weights
        )

        assert position.shape == (3,), "Should return valid position"
        assert quaternion.shape == (4,), "Should return valid quaternion"

    def test_quaternion_is_normalized(self, device):
        """Test that returned quaternion is normalized."""
        torch.manual_seed(123)
        source = torch.randn(20, 3, device=device, dtype=torch.float32)
        target = source + torch.randn(20, 3, device=device) * 0.1
        normals = torch.nn.functional.normalize(
            torch.randn(20, 3, device=device, dtype=torch.float32), dim=1
        )

        position, quaternion = compute_pose_point_to_plane_cholesky(source, target, normals)

        quat_norm = quaternion.norm()
        assert abs(quat_norm - 1.0) < 0.1, f"Quaternion should be normalized, got norm={quat_norm}"


class TestExtractObservedPoints:
    """Test observed points extraction from camera observations."""

    @pytest.fixture
    def device(self):
        """Device fixture."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _create_camera_obs(
        self,
        points: torch.Tensor,
        mask: torch.Tensor = None,
        resolution: tuple = (480, 640),
    ) -> CameraObservation:
        """Create synthetic camera observation for testing."""
        H, W = resolution
        device = points.device
        dtype = points.dtype

        depth = torch.zeros((H, W), device=device, dtype=dtype)
        seg_mask = None

        n_points = min(len(points), H * W)
        indices = torch.arange(n_points, device=device)
        h_idx = indices // W
        w_idx = indices % W

        depth[h_idx, w_idx] = points[:n_points, 2].abs()

        if mask is not None:
            seg_mask = torch.zeros((H, W), device=device, dtype=torch.bool)
            seg_mask[h_idx, w_idx] = mask[:n_points]

        y_coords = torch.linspace(-0.5, 0.5, H, device=device, dtype=dtype)
        x_coords = torch.linspace(-0.5, 0.5, W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        projection_rays = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
        projection_rays = projection_rays / torch.norm(projection_rays, dim=-1, keepdim=True)
        projection_rays = projection_rays.view(-1, 3)

        return CameraObservation(
            depth_image=depth,
            image_segmentation=seg_mask,
            projection_rays=projection_rays,
            resolution=[H, W],
        )

    def test_extract_with_segmentation(self, device):
        """Test extracting points with segmentation mask."""
        n_points = 500
        points = torch.randn(n_points, 3, device=device, dtype=torch.float32)
        points[:, 2] = torch.abs(points[:, 2]) + 0.5  # Positive depth > 0.1

        mask = torch.ones(n_points, device=device, dtype=torch.bool)
        mask[250:] = False  # Only first half

        camera_obs = self._create_camera_obs(points, mask)
        extracted = extract_observed_points(camera_obs)

        assert extracted.shape[1] == 3
        assert len(extracted) > 0
        assert len(extracted) <= 250  # At most first half

    def test_extract_filters_nan(self, device):
        """Test that NaN values are filtered."""
        n_points = 200
        points = torch.randn(n_points, 3, device=device, dtype=torch.float32)
        points[:, 2] = torch.abs(points[:, 2]) + 0.5
        points[:50, :] = float('nan')  # Some invalid

        camera_obs = self._create_camera_obs(points)
        extracted = extract_observed_points(camera_obs)

        assert torch.isfinite(extracted).all()

    def test_extract_filters_small_depth(self, device):
        """Test that small depth values are filtered."""
        n_points = 200
        points = torch.randn(n_points, 3, device=device, dtype=torch.float32)
        points[:100, 2] = 0.05  # Too close, should be filtered
        points[100:, 2] = 0.5  # Valid

        camera_obs = self._create_camera_obs(points)
        extracted = extract_observed_points(camera_obs, min_depth=0.1)

        assert (extracted[:, 2].abs() > 0.1).all()

    def test_custom_min_depth(self, device):
        """Test custom minimum depth threshold."""
        n_points = 100
        points = torch.randn(n_points, 3, device=device, dtype=torch.float32)
        points[:, 2] = 0.15  # Just above default 0.1

        camera_obs = self._create_camera_obs(points)

        # Should pass with default min_depth=0.1
        extracted_default = extract_observed_points(camera_obs)
        assert len(extracted_default) > 0

        # Should be filtered with higher threshold
        extracted_high = extract_observed_points(camera_obs, min_depth=0.2)
        assert len(extracted_high) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

