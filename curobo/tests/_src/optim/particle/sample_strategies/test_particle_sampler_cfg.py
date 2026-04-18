# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for ParticleSamplerCfg."""

import pytest
import torch

# CuRobo
from curobo._src.optim.particle.sample_strategies.particle_sampler_cfg import ParticleSamplerCfg
from curobo._src.types.device_cfg import DeviceCfg


class TestParticleSamplerCfg:
    """Test cases for ParticleSamplerCfg dataclass."""

    @pytest.fixture
    def device_cfg(self):
        """Fixture for tensor arguments."""
        return DeviceCfg(device=torch.device("cpu"), dtype=torch.float32)

    def test_init_basic(self, device_cfg):
        """Test basic initialization with required arguments."""
        cfg = ParticleSamplerCfg(device_cfg=device_cfg)

        assert cfg.device_cfg == device_cfg
        assert cfg.fixed_samples is True
        assert cfg.seed == 0
        assert cfg.n_knots == 3
        assert cfg.degree == 3
        assert cfg.sample_method == "halton"
        assert cfg.stencil_type == "3point"

    def test_init_with_custom_values(self, device_cfg):
        """Test initialization with custom values."""
        custom_sample_ratio = {"halton": 0.5, "random": 0.3, "stomp": 0.2}
        custom_filter_coeffs = [0.2, 0.4, 0.4]

        cfg = ParticleSamplerCfg(
            device_cfg=device_cfg,
            fixed_samples=False,
            sample_ratio=custom_sample_ratio,
            seed=42,
            filter_coeffs=custom_filter_coeffs,
            n_knots=5,
            scale_tril=0.1,
            sample_method="random",
            stencil_type="5point",
            degree=4,
        )

        assert cfg.device_cfg == device_cfg
        assert cfg.fixed_samples is False
        assert cfg.sample_ratio == custom_sample_ratio
        assert cfg.seed == 42
        assert cfg.filter_coeffs == custom_filter_coeffs
        assert cfg.n_knots == 5
        assert cfg.scale_tril == 0.1
        assert cfg.sample_method == "random"
        assert cfg.stencil_type == "5point"
        assert cfg.degree == 4

    def test_default_sample_ratio(self, device_cfg):
        """Test default sample ratio values."""
        cfg = ParticleSamplerCfg(device_cfg=device_cfg)

        expected_ratio = {
            "halton": 1.0,
            "halton-knot": 0.0,
            "random": 0.0,
            "random-knot": 0.0,
            "stomp": 0.0,
        }

        assert cfg.sample_ratio == expected_ratio

    def test_default_filter_coeffs(self, device_cfg):
        """Test default filter coefficients."""
        cfg = ParticleSamplerCfg(device_cfg=device_cfg)

        assert cfg.filter_coeffs == [0.3, 0.3, 0.4]

    def test_optional_fields_none(self, device_cfg):
        """Test that optional fields can be None."""
        cfg = ParticleSamplerCfg(
            device_cfg=device_cfg,
            scale_tril=None,
            covariance_matrix=None,
        )

        assert cfg.scale_tril is None
        assert cfg.covariance_matrix is None

    def test_covariance_matrix_assignment(self, device_cfg):
        """Test covariance matrix assignment."""
        cov_matrix = torch.eye(3, dtype=device_cfg.dtype, device=device_cfg.device)

        cfg = ParticleSamplerCfg(
            device_cfg=device_cfg,
            covariance_matrix=cov_matrix,
        )

        torch.testing.assert_close(cfg.covariance_matrix, cov_matrix)

    def test_sample_ratio_validation_types(self, device_cfg):
        """Test that sample_ratio accepts different valid configurations."""
        # All zeros (edge case)
        zero_ratio = {"halton": 0.0, "random": 0.0, "stomp": 0.0}
        cfg1 = ParticleSamplerCfg(device_cfg=device_cfg, sample_ratio=zero_ratio)
        assert cfg1.sample_ratio == zero_ratio

        # Only halton
        halton_only = {"halton": 1.0}
        cfg2 = ParticleSamplerCfg(device_cfg=device_cfg, sample_ratio=halton_only)
        assert cfg2.sample_ratio == halton_only

        # Mixed strategies
        mixed_ratio = {"halton": 0.4, "random": 0.3, "stomp": 0.3}
        cfg3 = ParticleSamplerCfg(device_cfg=device_cfg, sample_ratio=mixed_ratio)
        assert cfg3.sample_ratio == mixed_ratio

    def test_sample_method_values(self, device_cfg):
        """Test different sample method values."""
        valid_methods = ["halton", "random", "roberts", "sobol"]

        for method in valid_methods:
            cfg = ParticleSamplerCfg(device_cfg=device_cfg, sample_method=method)
            assert cfg.sample_method == method

    def test_stencil_type_values(self, device_cfg):
        """Test different stencil type values."""
        valid_stencils = ["3point", "5point", "7point"]

        for stencil in valid_stencils:
            cfg = ParticleSamplerCfg(device_cfg=device_cfg, stencil_type=stencil)
            assert cfg.stencil_type == stencil

    def test_knots_and_degree_relationship(self, device_cfg):
        """Test different knot and degree combinations."""
        # Test various combinations
        combinations = [
            (2, 1),  # Linear with 2 knots
            (3, 2),  # Quadratic with 3 knots
            (4, 3),  # Cubic with 4 knots
            (5, 3),  # Cubic with 5 knots
            (6, 4),  # Quartic with 6 knots
        ]

        for n_knots, degree in combinations:
            cfg = ParticleSamplerCfg(
                device_cfg=device_cfg,
                n_knots=n_knots,
                degree=degree,
            )
            assert cfg.n_knots == n_knots
            assert cfg.degree == degree

    def test_seed_values(self, device_cfg):
        """Test different seed values."""
        seeds = [0, 1, 42, 123, 999, 12345]

        for seed in seeds:
            cfg = ParticleSamplerCfg(device_cfg=device_cfg, seed=seed)
            assert cfg.seed == seed

    def test_fixed_samples_boolean(self, device_cfg):
        """Test fixed_samples boolean flag."""
        for fixed in [True, False]:
            cfg = ParticleSamplerCfg(device_cfg=device_cfg, fixed_samples=fixed)
            assert cfg.fixed_samples == fixed

    def test_scale_tril_values(self, device_cfg):
        """Test different scale_tril values."""
        scale_values = [None, 0.1, 0.5, 1.0, 2.0]

        for scale in scale_values:
            cfg = ParticleSamplerCfg(device_cfg=device_cfg, scale_tril=scale)
            assert cfg.scale_tril == scale

    def test_filter_coeffs_validation(self, device_cfg):
        """Test filter coefficients with different configurations."""
        # Standard 3-coefficient filter
        coeffs_3 = [0.2, 0.3, 0.5]
        cfg1 = ParticleSamplerCfg(device_cfg=device_cfg, filter_coeffs=coeffs_3)
        assert cfg1.filter_coeffs == coeffs_3

        # 2-coefficient filter
        coeffs_2 = [0.6, 0.4]
        cfg2 = ParticleSamplerCfg(device_cfg=device_cfg, filter_coeffs=coeffs_2)
        assert cfg2.filter_coeffs == coeffs_2

        # Single coefficient
        coeffs_1 = [1.0]
        cfg3 = ParticleSamplerCfg(device_cfg=device_cfg, filter_coeffs=coeffs_1)
        assert cfg3.filter_coeffs == coeffs_1

        # No filtering
        cfg4 = ParticleSamplerCfg(device_cfg=device_cfg, filter_coeffs=None)
        assert cfg4.filter_coeffs is None

    def test_tensor_args_device_dtype_consistency(self):
        """Test device_cfg with different devices and dtypes."""
        devices = [torch.device("cpu")]
        dtypes = [torch.float32, torch.float64]

        # Add CUDA device if available
        if torch.cuda.is_available():
            devices.append(torch.device("cuda:0"))

        for device in devices:
            for dtype in dtypes:
                device_cfg = DeviceCfg(device=device, dtype=dtype)
                cfg = ParticleSamplerCfg(device_cfg=device_cfg)

                assert cfg.device_cfg.device == device
                assert cfg.device_cfg.dtype == dtype

    def test_covariance_matrix_device_consistency(self, device_cfg):
        """Test that covariance matrix matches device_cfg device."""
        cov_matrix = torch.eye(4, dtype=device_cfg.dtype, device=device_cfg.device)

        cfg = ParticleSamplerCfg(
            device_cfg=device_cfg,
            covariance_matrix=cov_matrix,
        )

        assert cfg.covariance_matrix.device == device_cfg.device
        assert cfg.covariance_matrix.dtype == device_cfg.dtype

    def test_dataclass_immutability_modification(self, device_cfg):
        """Test that we can modify fields after creation (dataclass behavior)."""
        cfg = ParticleSamplerCfg(device_cfg=device_cfg)

        # Modify some fields
        cfg.seed = 999
        cfg.fixed_samples = False
        cfg.sample_method = "random"

        assert cfg.seed == 999
        assert cfg.fixed_samples is False
        assert cfg.sample_method == "random"

    def test_complex_sample_ratio_configuration(self, device_cfg):
        """Test complex sample ratio configurations."""
        # Configuration with all strategies
        complex_ratio = {
            "halton": 0.3,
            "halton-knot": 0.2,
            "random": 0.2,
            "random-knot": 0.2,
            "stomp": 0.1,
        }

        cfg = ParticleSamplerCfg(device_cfg=device_cfg, sample_ratio=complex_ratio)
        assert cfg.sample_ratio == complex_ratio

        # Verify sum is 1.0 (good practice check)
        total_ratio = sum(complex_ratio.values())
        assert abs(total_ratio - 1.0) < 1e-6

    def test_edge_case_parameters(self, device_cfg):
        """Test edge case parameter values."""
        # Minimum valid values
        cfg_min = ParticleSamplerCfg(
            device_cfg=device_cfg,
            n_knots=1,
            degree=0,
            seed=0,
        )
        assert cfg_min.n_knots == 1
        assert cfg_min.degree == 0
        assert cfg_min.seed == 0

        # Large values
        cfg_large = ParticleSamplerCfg(
            device_cfg=device_cfg,
            n_knots=20,
            degree=10,
            seed=2**31 - 1,
        )
        assert cfg_large.n_knots == 20
        assert cfg_large.degree == 10
        assert cfg_large.seed == 2**31 - 1
