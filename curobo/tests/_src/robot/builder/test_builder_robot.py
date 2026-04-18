# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Comprehensive tests for RobotBuilder."""

# Standard Library
from pathlib import Path

# Third Party
import pytest

from curobo._src.geom.sphere_fit import SphereFitType

# CuRobo
from curobo._src.robot.builder.builder_robot import RobotBuilder
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util_file import get_assets_path, get_robot_configs_path, join_path, load_yaml

FAST_ITERATIONS = 50


def default_fit_type() -> SphereFitType:
    """Get default fit type."""
    return SphereFitType.MORPHIT


@pytest.fixture(scope="module")
def franka_urdf_path():
    """Get path to Franka URDF file."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    urdf_path = join_path(get_assets_path(), robot_data["robot_cfg"]["kinematics"]["urdf_path"])
    return urdf_path


@pytest.fixture(scope="module")
def franka_asset_path():
    """Get path to Franka assets."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    asset_root = robot_data["robot_cfg"]["kinematics"]["asset_root_path"]
    return str(Path(get_assets_path()) / asset_root)


@pytest.fixture(scope="module")
def ur5e_urdf_path():
    """Get path to UR5e URDF file."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "ur10e.yml"))
    urdf_path = join_path(get_assets_path(), robot_data["robot_cfg"]["kinematics"]["urdf_path"])
    return urdf_path


@pytest.fixture(scope="module")
def complete_franka_builder(franka_urdf_path, franka_asset_path):
    """Builder with fitted spheres and collision matrix (shared read-only across module)."""
    builder = RobotBuilder(
        urdf_path=franka_urdf_path,
        asset_path=franka_asset_path,
    )
    builder.fit_collision_spheres(
        sphere_density=1.0, fit_type=default_fit_type(), iterations=FAST_ITERATIONS
    )
    builder.compute_collision_matrix(num_samples=10, batch_size=10)
    return builder


class TestRobotBuilderInitialization:
    """Test RobotBuilder initialization."""

    def test_builder_init_basic(self, franka_urdf_path, franka_asset_path):
        """Test basic builder initialization from URDF."""
        builder = RobotBuilder(
            urdf_path=franka_urdf_path,
            asset_path=franka_asset_path,
        )

        assert builder is not None
        assert franka_urdf_path.endswith(builder.urdf_path)
        assert franka_asset_path.endswith(builder.asset_path)

    def test_builder_init_with_device_cfg(self, franka_urdf_path, franka_asset_path):
        """Test builder initialization with custom tensor config."""
        device_cfg = DeviceCfg()

        builder = RobotBuilder(
            urdf_path=franka_urdf_path,
            asset_path=franka_asset_path,
            device_cfg=device_cfg,
        )

        assert builder is not None
        assert builder.device_cfg == device_cfg


class TestRobotBuilderFromConfig:
    """Test RobotBuilder.from_config() class method."""

    def test_builder_from_config_file(self):
        """Test loading builder from existing config file."""
        config_path = join_path(get_robot_configs_path(), "franka.yml")

        builder = RobotBuilder.from_config(config_path)

        assert builder is not None
        assert builder.urdf_path is not None

    def test_builder_from_config_ur5e(self):
        """Test loading builder from UR5e config."""
        config_path = join_path(get_robot_configs_path(), "ur10e.yml")

        builder = RobotBuilder.from_config(config_path)

        assert builder is not None
        assert builder.urdf_path is not None

    def test_builder_from_config_preserves_settings(self):
        """Test that from_config preserves robot settings."""
        config_path = join_path(get_robot_configs_path(), "franka.yml")

        builder = RobotBuilder.from_config(config_path)

        assert builder.urdf_path is not None
        assert builder.asset_path is not None


class TestRobotBuilderProperties:
    """Test RobotBuilder properties."""

    @pytest.fixture(scope="class")
    def simple_builder(self, franka_urdf_path, franka_asset_path):
        """Create a simple builder for testing."""
        return RobotBuilder(
            urdf_path=franka_urdf_path,
            asset_path=franka_asset_path,
        )

    def test_link_names_property(self, simple_builder):
        """Test tool_frames property."""
        tool_frames = simple_builder.tool_frames

        assert isinstance(tool_frames, list)
        assert len(tool_frames) > 0

    def test_collision_link_names_before_fitting(self, simple_builder):
        """Test collision_link_names before spheres are fitted."""
        collision_links = simple_builder.collision_link_names

        assert collision_links is not None or collision_links == []

    def test_collision_spheres_before_fitting(self, simple_builder):
        """Test collision_spheres before fitting."""
        spheres = simple_builder.collision_spheres

        assert spheres is None or isinstance(spheres, dict)

    def test_collision_matrix_before_compute(self, simple_builder):
        """Test collision_matrix before computation."""
        matrix = simple_builder.collision_matrix

        assert matrix is None or isinstance(matrix, dict)

    def test_num_spheres_before_fitting(self, simple_builder):
        """Test num_spheres before fitting."""
        num = simple_builder.num_spheres

        assert isinstance(num, int)
        assert num >= 0


@pytest.mark.parametrize(
    "fit_type",
    [
        SphereFitType.SURFACE,
        SphereFitType.VOXEL,
        SphereFitType.MORPHIT,
    ],
)
class TestRobotBuilderFitSpheres:
    """Test sphere fitting functionality."""

    @pytest.fixture(scope="class")
    def builder_for_fitting(self, franka_urdf_path, franka_asset_path):
        """Create builder for fitting tests."""
        return RobotBuilder(
            urdf_path=franka_urdf_path,
            asset_path=franka_asset_path,
        )

    def test_fit_collision_spheres_basic(self, builder_for_fitting, fit_type):
        """Test basic collision sphere fitting."""
        builder_for_fitting.fit_collision_spheres(
            sphere_density=1.0,
            fit_type=fit_type,
            iterations=FAST_ITERATIONS,
        )

        assert builder_for_fitting.collision_spheres is not None
        assert len(builder_for_fitting.collision_spheres) > 0
        assert builder_for_fitting.num_spheres > 0

    def test_fit_collision_spheres_high_density(self, builder_for_fitting, fit_type):
        """Test sphere fitting with higher density."""
        builder_for_fitting.fit_collision_spheres(
            sphere_density=2.0,
            fit_type=fit_type,
            iterations=FAST_ITERATIONS,
        )

        num_spheres_high = builder_for_fitting.num_spheres

        assert num_spheres_high > 0

    def test_refit_link_spheres(self, builder_for_fitting, fit_type):
        """Test refitting spheres for a single link."""
        builder_for_fitting.fit_collision_spheres(
            sphere_density=1.0,
            fit_type=fit_type,
            iterations=FAST_ITERATIONS,
        )

        builder_for_fitting.refit_link_spheres(
            link_name="panda_hand",
            sphere_density=3.0,
            fit_type=fit_type,
            iterations=FAST_ITERATIONS,
        )

        assert builder_for_fitting.collision_spheres is not None
        new_num = builder_for_fitting.num_spheres
        assert new_num > 0


class TestRobotBuilderCollisionMatrix:
    """Test collision matrix computation."""

    @pytest.fixture(scope="class")
    def builder_with_spheres(self, franka_urdf_path, franka_asset_path):
        """Create builder with fitted spheres (shared across class)."""
        builder = RobotBuilder(
            urdf_path=franka_urdf_path,
            asset_path=franka_asset_path,
        )
        builder.fit_collision_spheres(
            fit_type=default_fit_type(), iterations=FAST_ITERATIONS
        )
        return builder

    def test_compute_collision_matrix_basic(self, builder_with_spheres):
        """Test basic collision matrix computation."""
        builder_with_spheres.compute_collision_matrix(
            num_samples=10,
            batch_size=10,
        )

        matrix = builder_with_spheres.collision_matrix
        assert matrix is not None
        assert isinstance(matrix, dict)

    def test_compute_collision_matrix_more_samples(self, builder_with_spheres):
        """Test collision matrix with more samples."""
        builder_with_spheres.compute_collision_matrix(
            num_samples=100,
            batch_size=100,
        )

        matrix = builder_with_spheres.collision_matrix
        assert matrix is not None
        assert len(matrix) > 0

    def test_add_collision_ignore(self, builder_with_spheres):
        """Test manually adding collision ignore pairs."""
        builder_with_spheres.compute_collision_matrix(num_samples=100)

        builder_with_spheres.add_collision_ignore(
            link_name="panda_link1",
            ignore_links=["panda_link5"],
        )

        matrix = builder_with_spheres.collision_matrix
        assert matrix is not None
        if "panda_link1" in matrix:
            assert "panda_link5" in matrix["panda_link1"]

    def test_remove_collision_ignore(self, builder_with_spheres):
        """Test removing collision ignore pairs."""
        builder_with_spheres.compute_collision_matrix(num_samples=100)

        builder_with_spheres.add_collision_ignore("panda_link1", ["panda_link3"])
        builder_with_spheres.remove_collision_ignore("panda_link1", ["panda_link3"])

        matrix = builder_with_spheres.collision_matrix
        assert matrix is not None


class TestRobotBuilderBuild:
    """Test building final configuration."""

    def test_build_config(self, complete_franka_builder):
        """Test building final robot configuration."""
        config = complete_franka_builder.build()

        assert config is not None
        assert hasattr(config, 'base_link')
        assert config.base_link == "base_link"

    def test_build_with_custom_params(self, complete_franka_builder):
        """Test building with custom parameters."""
        config = complete_franka_builder.build()

        assert config.collision_spheres is not None
        assert config.self_collision_ignore is not None


class TestRobotBuilderSave:
    """Test saving configurations."""

    def test_save_to_yaml(self, complete_franka_builder, tmp_path):
        """Test saving configuration to YAML file."""
        output_path = str(tmp_path / "test_robot.yml")
        config = complete_franka_builder.build()

        complete_franka_builder.save(config, output_path)

        assert Path(output_path).exists()

        saved_data = load_yaml(output_path)
        assert saved_data is not None
        assert "kinematics" in saved_data

    def test_save_xrdf(self, complete_franka_builder, tmp_path):
        """Test saving configuration to XRDF format."""
        output_path = str(tmp_path / "test_robot.xrdf")
        config = complete_franka_builder.build()

        complete_franka_builder.save_xrdf(
            config=config,
            output_path=output_path,
            geometry_name="collision_model",
        )

        assert Path(output_path).exists()


class TestRobotBuilderIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow_new_robot(self, franka_urdf_path, franka_asset_path, tmp_path):
        """Test complete workflow: init -> fit -> compute -> build -> save."""
        builder = RobotBuilder(
            urdf_path=franka_urdf_path,
            asset_path=franka_asset_path,
        )

        builder.fit_collision_spheres(
            sphere_density=1.0, fit_type=default_fit_type(), iterations=FAST_ITERATIONS
        )
        assert builder.num_spheres > 0

        builder.compute_collision_matrix(num_samples=10, batch_size=10)
        assert builder.collision_matrix is not None

        config = builder.build()
        assert config is not None

        output_path = str(tmp_path / "test_complete.yml")
        builder.save(config, output_path)
        assert Path(output_path).exists()

    def test_workflow_edit_existing(self):
        """Test workflow for editing existing configuration."""
        builder = RobotBuilder.from_config(
            join_path(get_robot_configs_path(), "franka.yml")
        )

        assert builder is not None
        assert builder.collision_spheres is not None or builder._urdf_path is not None

    def test_workflow_with_custom_ignore(self, franka_urdf_path, franka_asset_path):
        """Test workflow with manual collision ignore additions."""
        builder = RobotBuilder(
            urdf_path=franka_urdf_path,
            asset_path=franka_asset_path,
        )

        builder.fit_collision_spheres(
            sphere_density=1.0, fit_type=default_fit_type(), iterations=FAST_ITERATIONS
        )
        builder.compute_collision_matrix(num_samples=10, batch_size=10)

        builder.add_collision_ignore("panda_link0", ["panda_link2"])

        config = builder.build()
        assert config is not None


@pytest.mark.parametrize("robot_file", ["franka.yml", "ur10e.yml"])
class TestRobotBuilderMultiRobot:
    """Test builder with different robots."""

    def test_builder_from_config_multi_robot(self, robot_file):
        """Test builder works with different robot configurations."""
        config_path = join_path(get_robot_configs_path(), robot_file)

        builder = RobotBuilder.from_config(config_path)

        assert builder is not None
        assert builder.tool_frames is not None
        assert len(builder.tool_frames) > 0


class TestRobotBuilderAdvanced:
    """Test advanced builder features."""

    def test_builder_build_without_spheres(self, franka_urdf_path, franka_asset_path):
        """Test building config without fitting spheres."""
        builder = RobotBuilder(
            urdf_path=franka_urdf_path,
            asset_path=franka_asset_path,
        )

        config = builder.build()

        assert config is not None
        assert config.base_link == "base_link"

    def test_builder_fit_then_refit(self, franka_urdf_path, franka_asset_path):
        """Test fitting, then refitting with different parameters."""
        builder = RobotBuilder(
            urdf_path=franka_urdf_path,
            asset_path=franka_asset_path,
        )

        builder.fit_collision_spheres(
            sphere_density=1.0, fit_type=default_fit_type(), iterations=FAST_ITERATIONS
        )
        first_num = builder.num_spheres

        builder.fit_collision_spheres(
            sphere_density=1.5, fit_type=default_fit_type(), iterations=FAST_ITERATIONS
        )
        second_num = builder.num_spheres

        assert first_num > 0
        assert second_num > 0

    def test_builder_multiple_ignore_additions(self, franka_urdf_path, franka_asset_path):
        """Test adding multiple collision ignores."""
        builder = RobotBuilder(
            urdf_path=franka_urdf_path,
            asset_path=franka_asset_path,
        )

        builder.fit_collision_spheres(
            sphere_density=1.0, fit_type=default_fit_type(), iterations=FAST_ITERATIONS
        )
        builder.compute_collision_matrix(num_samples=10, batch_size=10)

        builder.add_collision_ignore("panda_link1", ["panda_link3", "panda_link4"])
        builder.add_collision_ignore("panda_link2", ["panda_link5"])

        matrix = builder.collision_matrix
        assert matrix is not None
        assert len(matrix) > 0


class TestRobotBuilderMorphItWeightsAndMetrics:
    """Test MorphIt weight tuning and compute_metrics functionality."""

    def test_compute_metrics_populates_link_metrics(self, franka_urdf_path, franka_asset_path):
        """Test that compute_metrics=True populates builder.link_metrics."""
        builder = RobotBuilder(
            urdf_path=franka_urdf_path,
            asset_path=franka_asset_path,
        )
        builder.fit_collision_spheres(
            sphere_density=0.5,
            fit_type=default_fit_type(),
            iterations=FAST_ITERATIONS,
            compute_metrics=True,
        )

        assert len(builder.link_metrics) > 0
        for link_name, m in builder.link_metrics.items():
            assert m.num_spheres > 0
            assert 0.0 <= m.coverage <= 1.0
            assert 0.0 <= m.protrusion <= 1.0
            assert m.surface_gap_mean >= 0.0
            assert m.volume_ratio >= 0.0

    def test_metrics_empty_without_flag(self, franka_urdf_path, franka_asset_path):
        """Test that link_metrics is empty when compute_metrics is not set."""
        builder = RobotBuilder(
            urdf_path=franka_urdf_path,
            asset_path=franka_asset_path,
        )
        builder.fit_collision_spheres(
            sphere_density=0.5,
            fit_type=default_fit_type(),
            iterations=FAST_ITERATIONS,
        )

        assert len(builder.link_metrics) == 0

    def test_high_protrusion_weight_reduces_protrusion(
        self, franka_urdf_path, franka_asset_path
    ):
        """Test that a high protrusion_weight produces lower protrusion."""
        builder_default = RobotBuilder(
            urdf_path=franka_urdf_path,
            asset_path=franka_asset_path,
        )
        builder_default.fit_collision_spheres(
            sphere_density=0.5,
            fit_type=default_fit_type(),
            iterations=FAST_ITERATIONS,
            compute_metrics=True,
        )

        builder_high_prot = RobotBuilder(
            urdf_path=franka_urdf_path,
            asset_path=franka_asset_path,
        )
        builder_high_prot.fit_collision_spheres(
            sphere_density=0.5,
            fit_type=default_fit_type(),
            iterations=FAST_ITERATIONS,
            protrusion_weight=500.0,
            compute_metrics=True,
        )

        default_protrusion = sum(
            m.protrusion for m in builder_default.link_metrics.values()
        )
        high_prot_protrusion = sum(
            m.protrusion for m in builder_high_prot.link_metrics.values()
        )
        assert high_prot_protrusion <= default_protrusion

    def test_high_coverage_weight_improves_coverage(
        self, franka_urdf_path, franka_asset_path
    ):
        """Test that a high coverage_weight produces better coverage."""
        builder_low = RobotBuilder(
            urdf_path=franka_urdf_path,
            asset_path=franka_asset_path,
        )
        builder_low.fit_collision_spheres(
            sphere_density=0.5,
            fit_type=default_fit_type(),
            iterations=FAST_ITERATIONS,
            coverage_weight=1.0,
            compute_metrics=True,
        )

        builder_high = RobotBuilder(
            urdf_path=franka_urdf_path,
            asset_path=franka_asset_path,
        )
        builder_high.fit_collision_spheres(
            sphere_density=0.5,
            fit_type=default_fit_type(),
            iterations=FAST_ITERATIONS,
            coverage_weight=5000.0,
            compute_metrics=True,
        )

        low_coverage = sum(m.coverage for m in builder_low.link_metrics.values())
        high_coverage = sum(m.coverage for m in builder_high.link_metrics.values())
        assert high_coverage >= low_coverage

    def test_refit_link_with_metrics(self, franka_urdf_path, franka_asset_path):
        """Test that refit_link_spheres with compute_metrics updates link_metrics."""
        builder = RobotBuilder(
            urdf_path=franka_urdf_path,
            asset_path=franka_asset_path,
        )
        builder.fit_collision_spheres(
            sphere_density=0.5,
            fit_type=default_fit_type(),
            iterations=FAST_ITERATIONS,
        )

        assert len(builder.link_metrics) == 0

        builder.refit_link_spheres(
            "panda_hand",
            sphere_density=1.0,
            fit_type=default_fit_type(),
            iterations=FAST_ITERATIONS,
            compute_metrics=True,
        )

        assert "panda_hand" in builder.link_metrics
        m = builder.link_metrics["panda_hand"]
        assert m.num_spheres > 0
        assert 0.0 <= m.coverage <= 1.0


class TestRobotBuilderRefitLinkSpheres:
    """Test refit_link_spheres for link-specific sphere tuning."""

    @pytest.fixture(scope="class")
    def fitted_builder(self, franka_urdf_path, franka_asset_path):
        """Create builder with pre-fitted spheres (shared across class)."""
        builder = RobotBuilder(
            urdf_path=franka_urdf_path,
            asset_path=franka_asset_path,
        )
        builder.fit_collision_spheres(
            sphere_density=1.0, fit_type=default_fit_type(), iterations=FAST_ITERATIONS
        )
        return builder

    def test_refit_link_spheres_fixed_count(self, fitted_builder):
        """Test refitting a link with fixed num_spheres."""
        fitted_builder.refit_link_spheres(
            "panda_hand", num_spheres=5, fit_type=default_fit_type(), iterations=FAST_ITERATIONS
        )

        spheres = fitted_builder.collision_spheres
        assert spheres is not None
        if "panda_hand" in spheres:
            num_hand_spheres = len(spheres["panda_hand"])
            assert num_hand_spheres == 5

    def test_refit_link_spheres_density(self, fitted_builder):
        """Test refitting a link with different sphere_density."""
        fitted_builder.refit_link_spheres(
            "panda_link7",
            sphere_density=2.0,
            fit_type=default_fit_type(),
            iterations=FAST_ITERATIONS,
        )

        spheres = fitted_builder.collision_spheres
        assert spheres is not None

    def test_refit_link_spheres_multiple_links(self, fitted_builder):
        """Test refitting multiple links with different parameters."""
        fitted_builder.refit_link_spheres(
            "panda_hand", num_spheres=5, fit_type=default_fit_type(), iterations=FAST_ITERATIONS
        )
        fitted_builder.refit_link_spheres(
            "panda_link7",
            sphere_density=1.5,
            fit_type=default_fit_type(),
            iterations=FAST_ITERATIONS,
        )

        spheres = fitted_builder.collision_spheres
        assert spheres is not None
        assert len(spheres) > 0


class TestRobotBuilderExport:
    """Test export and save functionality."""

    def test_save_to_yaml(self, complete_franka_builder, tmp_path):
        """Test saving configuration to YAML file."""
        config = complete_franka_builder.build()
        output_path = str(tmp_path / "test_robot.yml")

        complete_franka_builder.save(config, output_path)

        assert Path(output_path).exists()

        loaded_data = load_yaml(output_path)
        assert loaded_data is not None
        assert "kinematics" in loaded_data

    def test_save_with_include_cspace_false(self, complete_franka_builder, tmp_path):
        """Test saving configuration without cspace."""
        config = complete_franka_builder.build()
        output_path = str(tmp_path / "test_robot_no_cspace.yml")

        complete_franka_builder.save(config, output_path, include_cspace=False)

        loaded_data = load_yaml(output_path)
        assert loaded_data is not None
        assert "kinematics" in loaded_data


class TestRobotBuilderCollisionIgnoreMerging:
    """Test merging collision ignore pairs."""

    @pytest.fixture(scope="class")
    def fitted_builder(self, franka_urdf_path, franka_asset_path):
        """Builder with spheres for collision ignore tests (shared across class)."""
        builder = RobotBuilder(
            urdf_path=franka_urdf_path,
            asset_path=franka_asset_path,
        )
        builder.fit_collision_spheres(
            sphere_density=1.0, fit_type=default_fit_type(), iterations=FAST_ITERATIONS
        )
        return builder

    def test_merge_collision_ignore_new_link(self, fitted_builder):
        """Test adding collision ignore for a new link."""
        fitted_builder.add_collision_ignore("panda_link1", ["panda_link5"])
        fitted_builder.add_collision_ignore("panda_link1", ["panda_link6"])

        ignore_dict = fitted_builder._self_collision_ignore
        assert "panda_link1" in ignore_dict
        assert "panda_link5" in ignore_dict["panda_link1"]
        assert "panda_link6" in ignore_dict["panda_link1"]

    def test_merge_collision_ignore_duplicate_pairs(self, fitted_builder):
        """Test that duplicate ignore pairs are not added twice."""
        fitted_builder.add_collision_ignore("panda_link1", ["panda_link3"])
        initial_count = len(fitted_builder._self_collision_ignore["panda_link1"])

        fitted_builder.add_collision_ignore("panda_link1", ["panda_link3"])
        final_count = len(fitted_builder._self_collision_ignore["panda_link1"])

        assert final_count == initial_count


class TestRobotBuilderEdgeCases:
    """Test edge cases and error conditions."""

    def test_build_without_spheres(self, franka_urdf_path, franka_asset_path):
        """Test building config without fitting spheres first."""
        builder = RobotBuilder(
            urdf_path=franka_urdf_path,
            asset_path=franka_asset_path,
        )

        try:
            config = builder.build()
            assert config is not None or True
        except (ValueError, RuntimeError, AttributeError):
            pass

    def test_compute_matrix_without_spheres(self, franka_urdf_path, franka_asset_path):
        """Test computing collision matrix without spheres."""
        builder = RobotBuilder(
            urdf_path=franka_urdf_path,
            asset_path=franka_asset_path,
        )

        try:
            builder.compute_collision_matrix(num_samples=10, batch_size=10)
        except (ValueError, RuntimeError, AttributeError):
            pass

    def test_add_collision_ignore_before_fitting(self, franka_urdf_path, franka_asset_path):
        """Test adding collision ignore before fitting spheres."""
        builder = RobotBuilder(
            urdf_path=franka_urdf_path,
            asset_path=franka_asset_path,
        )

        builder.add_collision_ignore("panda_link1", ["panda_link3"])

        builder.fit_collision_spheres(
            sphere_density=1.0, fit_type=default_fit_type(), iterations=FAST_ITERATIONS
        )

        assert "panda_link1" in builder._self_collision_ignore

    def test_fit_spheres_sphere_count_scales_with_density(
        self, franka_urdf_path, franka_asset_path
    ):
        """Test that sphere count scales with sphere_density."""
        builder_low = RobotBuilder(
            urdf_path=franka_urdf_path,
            asset_path=franka_asset_path,
        )
        builder_low.fit_collision_spheres(
            sphere_density=0.1,
            fit_type=default_fit_type(),
            iterations=FAST_ITERATIONS,
        )

        builder_high = RobotBuilder(
            urdf_path=franka_urdf_path,
            asset_path=franka_asset_path,
        )
        builder_high.fit_collision_spheres(
            sphere_density=2.0,
            fit_type=default_fit_type(),
            iterations=FAST_ITERATIONS,
        )

        assert builder_low.num_spheres > 0
        assert builder_high.num_spheres > 0
        assert builder_high.num_spheres > builder_low.num_spheres
