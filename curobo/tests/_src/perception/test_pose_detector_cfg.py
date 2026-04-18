# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for DetectorCfg configuration dataclass."""

# Standard Library
from dataclasses import fields

# Third Party
import pytest

# CuRobo
from curobo._src.perception.pose_estimation.pose_detector_cfg import DetectorCfg
from curobo._src.types.device_cfg import DeviceCfg


class TestDetectorCfgInitialization:
    """Test DetectorCfg initialization and default values."""

    def test_default_initialization(self):
        """Test default configuration values."""
        cfg = DetectorCfg()

        # Coarse stage defaults
        assert cfg.n_mesh_points_coarse == 500
        assert cfg.n_observed_points_coarse == 2000
        assert cfg.n_rotation_samples == 64
        assert cfg.n_iterations_coarse == 50
        assert cfg.distance_threshold_coarse == 0.5

        # Fine stage defaults
        assert cfg.n_mesh_points_fine == 2000
        assert cfg.n_observed_points_fine == 10000
        assert cfg.n_iterations_fine == 50
        assert cfg.distance_threshold_fine == 0.01

        # Robust estimation defaults
        assert cfg.use_huber_loss is True
        assert cfg.huber_delta == 0.02

        # Debug defaults
        assert cfg.save_iterations is False

        # Tensor config
        assert isinstance(cfg.device_cfg, DeviceCfg)

    def test_custom_coarse_parameters(self):
        """Test custom coarse stage parameters."""
        cfg = DetectorCfg(
            n_mesh_points_coarse=1000,
            n_observed_points_coarse=5000,
            n_rotation_samples=128,
            n_iterations_coarse=100,
            distance_threshold_coarse=0.8,
        )

        assert cfg.n_mesh_points_coarse == 1000
        assert cfg.n_observed_points_coarse == 5000
        assert cfg.n_rotation_samples == 128
        assert cfg.n_iterations_coarse == 100
        assert cfg.distance_threshold_coarse == 0.8

    def test_custom_fine_parameters(self):
        """Test custom fine stage parameters."""
        cfg = DetectorCfg(
            n_mesh_points_fine=5000,
            n_observed_points_fine=20000,
            n_iterations_fine=100,
            distance_threshold_fine=0.005,
        )

        assert cfg.n_mesh_points_fine == 5000
        assert cfg.n_observed_points_fine == 20000
        assert cfg.n_iterations_fine == 100
        assert cfg.distance_threshold_fine == 0.005

    def test_robust_estimation_parameters(self):
        """Test robust estimation configuration."""
        cfg = DetectorCfg(use_huber_loss=False, huber_delta=0.05)

        assert cfg.use_huber_loss is False
        assert cfg.huber_delta == 0.05

    def test_debug_parameters(self):
        """Test debug options."""
        cfg = DetectorCfg(save_iterations=True)

        assert cfg.save_iterations is True

    def test_tensor_config_override(self, cpu_device_cfg):
        """Test custom tensor configuration."""
        cfg = DetectorCfg(device_cfg=cpu_device_cfg)

        assert cfg.device_cfg.device == cpu_device_cfg.device
        assert cfg.device_cfg.dtype == cpu_device_cfg.dtype


class TestDetectorCfgValidation:
    """Test DetectorCfg parameter validation and constraints."""

    def test_point_count_progression(self):
        """Test that fine stage uses more points than coarse stage (typical)."""
        cfg = DetectorCfg()

        # Typical configuration has more points in fine stage
        assert cfg.n_mesh_points_fine > cfg.n_mesh_points_coarse
        assert cfg.n_observed_points_fine > cfg.n_observed_points_coarse

    def test_distance_threshold_progression(self):
        """Test that fine stage uses smaller threshold than coarse stage."""
        cfg = DetectorCfg()

        # Fine stage should have tighter distance threshold
        assert cfg.distance_threshold_fine < cfg.distance_threshold_coarse

    def test_all_fields_accessible(self):
        """Test that all dataclass fields are accessible."""
        cfg = DetectorCfg()

        # Get all field names
        field_names = [f.name for f in fields(DetectorCfg)]

        # Verify all fields can be accessed
        for field_name in field_names:
            assert hasattr(cfg, field_name)
            getattr(cfg, field_name)  # Should not raise

    def test_positive_values(self):
        """Test that numeric parameters are positive."""
        cfg = DetectorCfg()

        # All these should be positive
        assert cfg.n_mesh_points_coarse > 0
        assert cfg.n_observed_points_coarse > 0
        assert cfg.n_rotation_samples > 0
        assert cfg.n_iterations_coarse > 0
        assert cfg.distance_threshold_coarse > 0

        assert cfg.n_mesh_points_fine > 0
        assert cfg.n_observed_points_fine > 0
        assert cfg.n_iterations_fine > 0
        assert cfg.distance_threshold_fine > 0

        assert cfg.huber_delta > 0


class TestDataclassFieldsDetectorCfg:
    """Test dataclass field definitions and types for DetectorCfg."""

    def test_detector_cfg_field_count(self):
        """Test DetectorCfg has expected number of fields."""
        cfg_fields = fields(DetectorCfg)
        assert len(cfg_fields) == 14  # All configuration parameters

    def test_detector_cfg_field_names(self):
        """Test DetectorCfg has all expected field names."""
        cfg_fields = {f.name for f in fields(DetectorCfg)}

        expected_fields = {
            "n_mesh_points_coarse",
            "n_observed_points_coarse",
            "n_rotation_samples",
            "n_iterations_coarse",
            "distance_threshold_coarse",
            "n_mesh_points_fine",
            "n_observed_points_fine",
            "n_iterations_fine",
            "distance_threshold_fine",
            "use_huber_loss",
            "huber_delta",
            "save_iterations",
            "device_cfg",
            "use_svd",
        }

        assert cfg_fields == expected_fields


class TestConfigurationScenarios:
    """Test realistic configuration scenarios."""

    def test_fast_coarse_detection(self):
        """Test configuration for fast coarse detection."""
        cfg = DetectorCfg(
            n_mesh_points_coarse=200,
            n_observed_points_coarse=500,
            n_rotation_samples=32,
            n_iterations_coarse=20,
        )

        # Fast configuration uses fewer points and iterations
        assert cfg.n_mesh_points_coarse < 500
        assert cfg.n_observed_points_coarse < 2000
        assert cfg.n_rotation_samples < 64
        assert cfg.n_iterations_coarse < 50

    def test_accurate_fine_detection(self):
        """Test configuration for accurate fine detection."""
        cfg = DetectorCfg(
            n_mesh_points_fine=10000,
            n_observed_points_fine=50000,
            n_iterations_fine=100,
            distance_threshold_fine=0.005,
        )

        # Accurate configuration uses more points and iterations
        assert cfg.n_mesh_points_fine > 2000
        assert cfg.n_observed_points_fine > 10000
        assert cfg.n_iterations_fine > 50
        assert cfg.distance_threshold_fine < 0.01

    def test_outlier_robust_configuration(self):
        """Test configuration with outlier robustness enabled."""
        cfg = DetectorCfg(use_huber_loss=True, huber_delta=0.05)

        assert cfg.use_huber_loss is True
        assert cfg.huber_delta == 0.05

    def test_non_robust_configuration(self):
        """Test configuration without outlier robustness."""
        cfg = DetectorCfg(use_huber_loss=False)

        assert cfg.use_huber_loss is False

    def test_debug_configuration(self):
        """Test configuration with debug options enabled."""
        cfg = DetectorCfg(save_iterations=True)

        assert cfg.save_iterations is True


class TestEdgeCasesDetectorCfg:
    """Test edge cases and boundary conditions for DetectorCfg."""

    def test_minimal_point_counts(self):
        """Test with minimal point counts."""
        cfg = DetectorCfg(
            n_mesh_points_coarse=10,
            n_observed_points_coarse=10,
            n_mesh_points_fine=10,
            n_observed_points_fine=10,
        )

        assert cfg.n_mesh_points_coarse == 10
        assert cfg.n_observed_points_coarse == 10
        assert cfg.n_mesh_points_fine == 10
        assert cfg.n_observed_points_fine == 10

    def test_single_iteration(self):
        """Test with single iteration."""
        cfg = DetectorCfg(n_iterations_coarse=1, n_iterations_fine=1)

        assert cfg.n_iterations_coarse == 1
        assert cfg.n_iterations_fine == 1

    def test_single_rotation_sample(self):
        """Test with single rotation sample."""
        cfg = DetectorCfg(n_rotation_samples=1)

        assert cfg.n_rotation_samples == 1

    def test_very_large_threshold(self):
        """Test with very large distance threshold."""
        cfg = DetectorCfg(distance_threshold_coarse=100.0)

        assert cfg.distance_threshold_coarse == 100.0

    def test_very_small_threshold(self):
        """Test with very small distance threshold."""
        cfg = DetectorCfg(distance_threshold_fine=0.0001)

        assert cfg.distance_threshold_fine == 0.0001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
