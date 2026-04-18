# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for PoseDetector using point-to-plane ICP.

Tests pose detection using synthetic point clouds without rendering.
"""

# Third Party
import pytest
import torch
import trimesh

from curobo._src.perception.pose_estimation.detection_result import DetectionResult
from curobo._src.perception.pose_estimation.geometry import (
    ArticulatedRobotGeometry,
    RigidObjectGeometry,
)
from curobo._src.perception.pose_estimation.pose_detector import PoseDetector
from curobo._src.perception.pose_estimation.pose_detector_cfg import DetectorCfg

# CuRobo
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.types.camera import CameraObservation
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


@pytest.fixture(scope="module")
def simple_box_mesh():
    """Create a simple box mesh for testing."""
    return trimesh.creation.box(extents=[0.2, 0.2, 0.2])


@pytest.fixture(scope="module")
def rigid_geometry(simple_box_mesh, cuda_device_cfg):
    """Create rigid object geometry."""
    return RigidObjectGeometry(mesh=simple_box_mesh, device_cfg=cuda_device_cfg)


@pytest.fixture(scope="module")
def franka_robot_cfg(cuda_device_cfg):
    """Load Franka robot configuration."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    mesh_link_names = robot_data["robot_cfg"]["kinematics"]["mesh_link_names"]
    cfg = KinematicsCfg.from_robot_yaml_file(robot_data, tool_frames=mesh_link_names)
    return cfg


@pytest.fixture(scope="module")
def franka_robot_model(franka_robot_cfg):
    """Create Franka robot model."""
    return Kinematics(franka_robot_cfg)


@pytest.fixture(scope="module")
def robot_geometry(franka_robot_model, cuda_device_cfg):
    """Create robot geometry."""
    return ArticulatedRobotGeometry(
        robot_model=franka_robot_model,
        device_cfg=cuda_device_cfg,
        points_per_cubic_meter=50000.0,
        min_points_per_link=20,
        max_points_per_link=100,
    )


@pytest.fixture
def detector_config(cuda_device_cfg):
    """Create detector configuration."""
    return DetectorCfg(
        n_mesh_points_coarse=200,
        n_observed_points_coarse=500,
        n_rotation_samples=8,  # Small for fast tests
        n_iterations_coarse=10,
        distance_threshold_coarse=0.1,
        n_mesh_points_fine=500,
        n_observed_points_fine=1000,
        n_iterations_fine=20,
        distance_threshold_fine=0.02,
        use_huber_loss=True,
        huber_delta=0.02,
        save_iterations=False,
        device_cfg=cuda_device_cfg,
    )


def create_synthetic_camera_observation(
    points: torch.Tensor,
    mask: torch.Tensor,
    resolution: tuple = (480, 640),
    cuda_device_cfg: DeviceCfg = None,
) -> CameraObservation:
    """Create synthetic camera observation from point cloud.

    Projects 3D points onto a virtual pinhole camera so that
    ``depth * ray`` recovers the original positions.

    Args:
        points: Point cloud in camera frame [N, 3]
        mask: Segmentation mask [N] or [H, W]
        resolution: Image resolution (H, W)
        cuda_device_cfg: Device configuration

    Returns:
        CameraObservation with synthetic depth and projection rays
    """
    H, W = resolution
    device = points.device
    dtype = points.dtype

    # Synthetic pinhole intrinsics (fx=fy=focal, cx=W/2, cy=H/2)
    focal = float(W)
    cx, cy = W / 2.0, H / 2.0

    depth = torch.zeros((H, W), device=device, dtype=dtype)
    seg_mask = torch.zeros((H, W), device=device, dtype=torch.bool)

    # Build per-pixel projection rays: ray[h,w] = normalize([x_ndc, y_ndc, 1])
    h_idx_grid = torch.arange(H, device=device, dtype=dtype)
    w_idx_grid = torch.arange(W, device=device, dtype=dtype)
    hh, ww = torch.meshgrid(h_idx_grid, w_idx_grid, indexing="ij")
    ray_dirs = torch.stack([(ww - cx) / focal, (hh - cy) / focal, torch.ones_like(ww)], dim=-1)
    ray_norms = torch.norm(ray_dirs, dim=-1, keepdim=True)
    projection_rays = (ray_dirs / ray_norms).view(-1, 3)

    # Project each 3D point to a pixel and store depth = ||point|| along ray
    n_points = len(points)
    px = (points[:, 0] / points[:, 2] * focal + cx).long().clamp(0, W - 1)
    py = (points[:, 1] / points[:, 2] * focal + cy).long().clamp(0, H - 1)

    flat_idx = py * W + px
    ray_at_point = projection_rays[flat_idx]
    point_depth = (points * ray_at_point).sum(dim=-1)

    depth[py, px] = point_depth
    if mask.ndim == 1:
        seg_mask[py, px] = mask[:n_points]
    else:
        seg_mask = mask

    return CameraObservation(
        depth_image=depth,
        image_segmentation=seg_mask,
        projection_rays=projection_rays,
        resolution=[H, W],
    )


class TestPoseDetectorInitialization:
    """Test PoseDetector initialization."""

    def test_init_with_rigid_geometry(self, rigid_geometry, detector_config, capsys):
        """Test initialization with rigid object geometry."""
        detector = PoseDetector(geometry=rigid_geometry, config=detector_config)

        assert detector.geometry is rigid_geometry
        assert detector.config is detector_config
        assert detector.device_cfg == detector_config.device_cfg

        # Verify printed output
        captured = capsys.readouterr()
        assert "PoseDetector initialized" in captured.out
        assert "0 DoF" in captured.out  # Rigid object

    def test_init_with_robot_geometry(self, robot_geometry, detector_config, capsys):
        """Test initialization with robot geometry."""
        detector = PoseDetector(geometry=robot_geometry, config=detector_config)

        assert detector.geometry is robot_geometry
        assert detector.config is detector_config

        captured = capsys.readouterr()
        assert "7 DoF" in captured.out  # Franka robot


class TestSampleRotations:
    """Test random rotation sampling."""

    def test_sample_single_rotation(self, rigid_geometry, detector_config):
        """Test sampling a single rotation."""
        detector = PoseDetector(geometry=rigid_geometry, config=detector_config)

        rotations = detector._sample_rotations(1)

        assert rotations.shape == (1, 3, 3)
        assert rotations.device == detector_config.device_cfg.device
        assert rotations.dtype == detector_config.device_cfg.dtype

    def test_sample_multiple_rotations(self, rigid_geometry, detector_config):
        """Test sampling multiple rotations."""
        detector = PoseDetector(geometry=rigid_geometry, config=detector_config)

        n_samples = 10
        rotations = detector._sample_rotations(n_samples)

        assert rotations.shape == (n_samples, 3, 3)

    def test_rotations_are_valid(self, rigid_geometry, detector_config):
        """Test that sampled rotations are valid rotation matrices."""
        detector = PoseDetector(geometry=rigid_geometry, config=detector_config)

        rotations = detector._sample_rotations(20)

        for R in rotations:
            # Check orthogonality: R @ R.T = I
            I = R @ R.T
            assert torch.allclose(I, torch.eye(3, device=R.device, dtype=R.dtype), atol=1e-5)

            # Check determinant = 1 (proper rotation, not reflection)
            det = torch.det(R)
            assert torch.isclose(det, torch.tensor(1.0, device=R.device, dtype=R.dtype), atol=1e-5)

    def test_rotations_are_diverse(self, rigid_geometry, detector_config):
        """Test that sampled rotations are diverse."""
        detector = PoseDetector(geometry=rigid_geometry, config=detector_config)

        rotations = detector._sample_rotations(50)

        # Check that not all rotations are the same
        for i in range(5):
            for j in range(i + 1, 5):
                assert not torch.allclose(rotations[i], rotations[j], atol=0.1)


class TestExtractObservedPoints:
    """Test point extraction from camera observations."""

    def test_extract_with_segmentation(self, rigid_geometry, detector_config, cuda_device_cfg):
        """Test extracting points with segmentation mask."""
        detector = PoseDetector(geometry=rigid_geometry, config=detector_config)

        # Create synthetic points
        n_points = 1000
        points = torch.randn(n_points, 3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        points[:, 2] += 1.0  # Ensure positive depth
        mask = torch.ones(n_points, device=cuda_device_cfg.device, dtype=torch.bool)
        mask[500:] = False  # Only first 500 points

        camera_obs = create_synthetic_camera_observation(points, mask, cuda_device_cfg=cuda_device_cfg)

        extracted = detector._extract_observed_points(camera_obs)

        # Should extract masked points
        assert extracted.shape[1] == 3
        assert len(extracted) > 0

    def test_extract_filters_invalid_depth(self, rigid_geometry, detector_config, cuda_device_cfg):
        """Test that invalid depth values are filtered."""
        detector = PoseDetector(geometry=rigid_geometry, config=detector_config)

        # Create points with some invalid depths
        n_points = 500
        points = torch.randn(n_points, 3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        points[:100, 2] = float('nan')  # Invalid
        points[100:200, 2] = float('inf')  # Invalid
        points[200:300, 2] = 0.05  # Too close, should be filtered
        points[300:, 2] += 1.0  # Valid

        mask = torch.ones(n_points, device=cuda_device_cfg.device, dtype=torch.bool)
        camera_obs = create_synthetic_camera_observation(points, mask, cuda_device_cfg=cuda_device_cfg)

        extracted = detector._extract_observed_points(camera_obs)

        # Should filter out invalid points
        assert torch.isfinite(extracted).all()
        assert (extracted[:, 2].abs() > 0.1).all()


class TestICPWithSyntheticData:
    """Test ICP with synthetic point clouds."""

    def test_icp_coarse_convergence(self, rigid_geometry, detector_config, cuda_device_cfg):
        """Test coarse ICP convergence with known transform."""
        detector = PoseDetector(geometry=rigid_geometry, config=detector_config)

        # Create ground truth transform
        R_gt = torch.eye(3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        t_gt = torch.tensor([0.1, 0.05, 0.5], device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        # Sample mesh points
        config = torch.zeros(1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        mesh_points, _ = rigid_geometry.sample_surface_points(500)

        # Create observed points by transforming mesh
        observed_points = (R_gt @ mesh_points.T).T + t_gt
        observed_points += torch.randn_like(observed_points) * 0.002  # 2mm noise

        # Run coarse ICP
        T_result, error, best_hyp, history = detector._icp_coarse(observed_points, config)

        # Should converge to reasonable error (coarse stage)
        assert error < 0.02  # Less than 20mm error (coarse is approximate)
        assert T_result.shape == (4, 4)
        assert best_hyp >= 0

    def test_icp_fine_refinement(self, rigid_geometry, detector_config, cuda_device_cfg):
        """Test fine ICP refinement."""
        detector = PoseDetector(geometry=rigid_geometry, config=detector_config)

        # Create ground truth
        R_gt = torch.eye(3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        t_gt = torch.tensor([0.05, -0.03, 0.4], device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        config = torch.zeros(1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        mesh_points, _ = rigid_geometry.sample_surface_points(1000)

        observed_points = (R_gt @ mesh_points.T).T + t_gt
        observed_points += torch.randn_like(observed_points) * 0.001  # 1mm noise

        # Start from approximate initial guess
        T_init = torch.eye(4, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        T_init[:3, 3] = t_gt + torch.randn(3, device=cuda_device_cfg.device) * 0.02  # 20mm offset

        # Run fine ICP
        T_result, error, num_iters, history = detector._icp_fine(T_init, observed_points, config)

        # Should refine to reasonable error
        assert error < 0.03  # Less than 30mm (fine refinement with initial offset)
        assert num_iters > 0
        assert T_result.shape == (4, 4)


class TestPoseDetectorWithRigidObject:
    """Test full pose detection pipeline with rigid objects."""

    def test_detect_identity_transform(self, rigid_geometry, detector_config, cuda_device_cfg):
        """Test detection with identity transform."""
        detector = PoseDetector(geometry=rigid_geometry, config=detector_config)

        config = torch.zeros(1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        mesh_points, _ = rigid_geometry.sample_surface_points(800)

        # Place object in front of the camera so the pinhole projection is valid
        t_offset = torch.tensor(
            [0.0, 0.0, 0.5], device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
        )
        observed_points = mesh_points.clone() + t_offset
        observed_points += torch.randn_like(observed_points) * 0.005  # 5mm noise

        mask = torch.ones(len(observed_points), device=cuda_device_cfg.device, dtype=torch.bool)
        camera_obs = create_synthetic_camera_observation(
            observed_points, mask, cuda_device_cfg=cuda_device_cfg
        )

        result = detector.detect(camera_obs, config)

        assert isinstance(result, DetectionResult)
        assert result.alignment_error < 0.20  # Less than 200mm (full pipeline with synthetic data)
        assert 0.0 <= result.confidence <= 1.0

    def test_detect_with_translation(self, rigid_geometry, detector_config, cuda_device_cfg):
        """Test detection with translation."""
        torch.manual_seed(42)  # For reproducibility
        detector = PoseDetector(geometry=rigid_geometry, config=detector_config)

        # Ground truth translation
        t_gt = torch.tensor([0.15, -0.10, 0.60], device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        config = torch.zeros(1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        mesh_points, _ = rigid_geometry.sample_surface_points(800)

        observed_points = mesh_points + t_gt
        observed_points += torch.randn_like(observed_points) * 0.003

        mask = torch.ones(len(observed_points), device=cuda_device_cfg.device, dtype=torch.bool)
        camera_obs = create_synthetic_camera_observation(
            observed_points, mask, cuda_device_cfg=cuda_device_cfg
        )

        result = detector.detect(camera_obs, config)

        assert result.alignment_error < 0.15
        # Check translation is approximately correct (with seed for reproducibility)
        detected_translation = result.pose.position[0]
        assert torch.allclose(detected_translation, t_gt, atol=0.20)  # Within 200mm

    def test_detect_result_structure(self, rigid_geometry, detector_config, cuda_device_cfg):
        """Test that detection result has expected structure."""
        detector = PoseDetector(geometry=rigid_geometry, config=detector_config)

        config = torch.zeros(1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        mesh_points, _ = rigid_geometry.sample_surface_points(600)

        observed_points = mesh_points + torch.randn_like(mesh_points) * 0.005
        mask = torch.ones(len(observed_points), device=cuda_device_cfg.device, dtype=torch.bool)
        camera_obs = create_synthetic_camera_observation(
            observed_points, mask, cuda_device_cfg=cuda_device_cfg
        )

        result = detector.detect(camera_obs, config)

        assert hasattr(result, 'pose')
        assert hasattr(result, 'config')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'alignment_error')
        assert hasattr(result, 'n_iterations')
        assert isinstance(result.pose, Pose)


class TestPoseDetectorWithRobot:
    """Test pose detection with articulated robot."""

    def test_detect_robot_at_zero_config(self, robot_geometry, detector_config, cuda_device_cfg):
        """Test robot detection at zero configuration."""
        detector = PoseDetector(geometry=robot_geometry, config=detector_config)

        config = torch.zeros(1, 7, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        robot_geometry.update(config)
        mesh_points, _ = robot_geometry.sample_surface_points(1000)

        # Add small transform
        t_gt = torch.tensor([0.05, 0.03, 0.4], device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        observed_points = mesh_points + t_gt
        observed_points += torch.randn_like(observed_points) * 0.005

        mask = torch.ones(len(observed_points), device=cuda_device_cfg.device, dtype=torch.bool)
        camera_obs = create_synthetic_camera_observation(
            observed_points, mask, cuda_device_cfg=cuda_device_cfg
        )

        result = detector.detect(camera_obs, config)

        assert result.alignment_error < 0.50  # Within 500mm for robot (more complex)
        assert result.config.shape == (1, 7)

    def test_detect_robot_at_different_configs(self, robot_geometry, detector_config, cuda_device_cfg):
        """Test robot detection at different configurations."""
        detector = PoseDetector(geometry=robot_geometry, config=detector_config)

        # Test multiple configurations
        configs = [
            torch.zeros(1, 7, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype),
            torch.ones(1, 7, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype) * 0.3,
            torch.randn(1, 7, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype) * 0.2,
        ]

        for config in configs:
            robot_geometry.update(config)
            mesh_points, _ = robot_geometry.sample_surface_points(800)

            observed_points = mesh_points + torch.randn_like(mesh_points) * 0.008
            mask = torch.ones(len(observed_points), device=cuda_device_cfg.device, dtype=torch.bool)
            camera_obs = create_synthetic_camera_observation(
                observed_points, mask, cuda_device_cfg=cuda_device_cfg
            )

            result = detector.detect(camera_obs, config)

            assert result.alignment_error < 0.25  # Within 250mm for robot (more complex)
            assert result.confidence >= 0.0  # May be zero if alignment fails


class TestDebugMode:
    """Test debug/iteration tracking mode."""

    def test_save_iterations_disabled(self, rigid_geometry, cuda_device_cfg):
        """Test that iterations are not saved when disabled."""
        config = DetectorCfg(
            save_iterations=False,
            n_mesh_points_coarse=200,
            n_observed_points_coarse=500,
            n_rotation_samples=4,
            device_cfg=cuda_device_cfg,
        )
        detector = PoseDetector(geometry=rigid_geometry, config=config)

        mesh_config = torch.zeros(1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        mesh_points, _ = rigid_geometry.sample_surface_points(600)

        observed_points = mesh_points + torch.randn_like(mesh_points) * 0.005
        mask = torch.ones(len(observed_points), device=cuda_device_cfg.device, dtype=torch.bool)
        camera_obs = create_synthetic_camera_observation(
            observed_points, mask, cuda_device_cfg=cuda_device_cfg
        )

        result = detector.detect(camera_obs, mesh_config)

        assert result.coarse_iterations is None
        assert result.fine_iterations is None

    def test_save_iterations_enabled(self, rigid_geometry, cuda_device_cfg):
        """Test that iterations are saved when enabled."""
        config = DetectorCfg(
            save_iterations=True,
            n_mesh_points_coarse=200,
            n_observed_points_coarse=500,
            n_rotation_samples=4,
            n_iterations_coarse=5,
            n_iterations_fine=5,
            device_cfg=cuda_device_cfg,
        )
        detector = PoseDetector(geometry=rigid_geometry, config=config)

        mesh_config = torch.zeros(1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        mesh_points, _ = rigid_geometry.sample_surface_points(600)

        observed_points = mesh_points + torch.randn_like(mesh_points) * 0.005
        mask = torch.ones(len(observed_points), device=cuda_device_cfg.device, dtype=torch.bool)
        camera_obs = create_synthetic_camera_observation(
            observed_points, mask, cuda_device_cfg=cuda_device_cfg
        )

        result = detector.detect(camera_obs, mesh_config)

        # Should have iteration history
        assert result.coarse_iterations is not None
        assert result.fine_iterations is not None
        assert len(result.coarse_iterations) > 0
        assert len(result.fine_iterations) > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_with_very_few_points(self, rigid_geometry, detector_config, cuda_device_cfg):
        """Test with very few observed points."""
        detector = PoseDetector(geometry=rigid_geometry, config=detector_config)

        config = torch.zeros(1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)

        # Only 20 points, with fixed seed for determinism; abs+1 ensures z>0 so all
        # points project to valid, distinct pixels through the depth image round-trip.
        torch.manual_seed(0)
        observed_points = torch.randn(20, 3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        observed_points[:, 2] = observed_points[:, 2].abs() + 1.0

        mask = torch.ones(20, device=cuda_device_cfg.device, dtype=torch.bool)
        camera_obs = create_synthetic_camera_observation(
            observed_points, mask, cuda_device_cfg=cuda_device_cfg
        )

        # Should not crash
        result = detector.detect(camera_obs, config)

        assert isinstance(result, DetectionResult)

    def test_with_noisy_observations(self, rigid_geometry, detector_config, cuda_device_cfg):
        """Test with high noise in observations."""
        detector = PoseDetector(geometry=rigid_geometry, config=detector_config)

        config = torch.zeros(1, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        mesh_points, _ = rigid_geometry.sample_surface_points(500)

        # Place object in front of camera so pinhole projection is valid
        t_offset = torch.tensor(
            [0.0, 0.0, 0.5], device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype
        )
        observed_points = mesh_points + t_offset + torch.randn_like(mesh_points) * 0.01

        mask = torch.ones(len(observed_points), device=cuda_device_cfg.device, dtype=torch.bool)
        camera_obs = create_synthetic_camera_observation(
            observed_points, mask, cuda_device_cfg=cuda_device_cfg
        )

        result = detector.detect(camera_obs, config)

        # Should still work but with higher error due to high noise
        assert result.alignment_error < 0.2  # Within 200mm with high noise
        assert result.confidence >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

