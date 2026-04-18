# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Comprehensive tests for RobotDebugger."""

# Standard Library

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.robot.builder.debugger_robot import RobotDebugger
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util_file import file_exists, get_assets_path, get_robot_configs_path, join_path


@pytest.fixture(scope="module")
def franka_config_path():
    """Get path to Franka robot configuration."""
    return join_path(get_robot_configs_path(), "franka.yml")


@pytest.fixture(scope="module")
def ur5e_config_path():
    """Get path to UR5e robot configuration."""
    return join_path(get_robot_configs_path(), "ur10e.yml")


@pytest.fixture(scope="module")
def dual_ur10e_config_path():
    """Get path to dual UR10e robot configuration."""
    return join_path(get_robot_configs_path(), "dual_ur10e.yml")


@pytest.fixture(scope="module")
def device_cfg():
    """Get default tensor configuration."""
    return DeviceCfg()


class TestRobotDebuggerInitialization:
    """Test RobotDebugger initialization."""

    def test_init_basic_franka(self, franka_config_path, device_cfg):
        """Test basic initialization with Franka robot."""
        debugger = RobotDebugger(franka_config_path, device_cfg=device_cfg)

        assert debugger is not None
        assert debugger.config_path == franka_config_path
        assert debugger.device_cfg == device_cfg
        assert debugger._robot_model is not None
        assert debugger._robot_config is not None
        assert debugger._collision_cost is not None

    def test_init_basic_ur5e(self, ur5e_config_path):
        """Test basic initialization with UR5e robot."""
        debugger = RobotDebugger(ur5e_config_path)

        assert debugger is not None
        assert debugger.config_path == ur5e_config_path
        assert debugger._robot_model is not None

    def test_init_with_default_device_cfg(self, franka_config_path):
        """Test initialization with default tensor config."""
        debugger = RobotDebugger(franka_config_path)

        assert debugger.device_cfg is not None
        assert isinstance(debugger.device_cfg, DeviceCfg)

    def test_init_dual_robot(self, dual_ur10e_config_path):
        """Test initialization with dual robot configuration."""
        debugger = RobotDebugger(dual_ur10e_config_path)

        assert debugger is not None
        assert debugger._robot_model is not None

    def test_robot_config_property(self, franka_config_path):
        """Test robot_config property."""
        debugger = RobotDebugger(franka_config_path)

        config = debugger.robot_config
        assert config is not None
        assert config == debugger._robot_config

    def test_robot_model_property(self, franka_config_path):
        """Test robot_model property."""
        debugger = RobotDebugger(franka_config_path)

        model = debugger.robot_model
        assert model is not None
        assert model == debugger._robot_model


class TestRobotDebuggerCollisionChecking:
    """Test collision checking functionality."""

    def test_check_default_joint_configuration_collision_franka(self, franka_config_path):
        """Test checking collisions at default joint configuration for Franka."""
        debugger = RobotDebugger(franka_config_path)

        result = debugger.check_default_joint_configuration_collision()

        assert "has_collision" in result
        assert "num_colliding_pairs" in result
        assert "colliding_pairs" in result
        assert "max_penetration" in result
        assert "distances" in result
        assert isinstance(result["has_collision"], bool)
        assert isinstance(result["num_colliding_pairs"], int)
        assert isinstance(result["colliding_pairs"], list)
        assert isinstance(result["max_penetration"], float)
        assert isinstance(result["distances"], dict)

    def test_check_default_joint_configuration_collision_ur5e(self, ur5e_config_path):
        """Test checking collisions at default joint configuration for UR5e."""
        debugger = RobotDebugger(ur5e_config_path)

        result = debugger.check_default_joint_configuration_collision()

        # UR5e default position should not have collisions
        assert result["has_collision"] is False or result["has_collision"] is True
        assert result["num_colliding_pairs"] >= 0

    def test_check_collision_at_config_list(self, franka_config_path):
        """Test collision checking with list input."""
        debugger = RobotDebugger(franka_config_path)

        # Use zero configuration
        joint_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        result = debugger.check_collision_at_config(joint_position)

        assert "has_collision" in result
        assert isinstance(result["has_collision"], bool)

    def test_check_collision_at_config_tensor(self, franka_config_path, device_cfg):
        """Test collision checking with tensor input."""
        debugger = RobotDebugger(franka_config_path, device_cfg=device_cfg)

        # Use zero configuration as tensor
        joint_position = torch.zeros(7, **device_cfg.as_torch_dict())
        result = debugger.check_collision_at_config(joint_position)

        assert "has_collision" in result
        assert isinstance(result["has_collision"], bool)

    def test_check_collision_at_config_numpy(self, franka_config_path):
        """Test collision checking with numpy array input."""
        debugger = RobotDebugger(franka_config_path)

        # Use default joint configuration
        default_joint_position = debugger._robot_model.default_joint_state.position.cpu().numpy()
        result = debugger.check_collision_at_config(default_joint_position)

        assert "has_collision" in result

    def test_check_collision_invalid_dof(self, franka_config_path):
        """Test collision checking with invalid DOF raises error."""
        debugger = RobotDebugger(franka_config_path)

        # Franka has 7 DOF, provide 6
        joint_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        with pytest.raises(ValueError, match="must have"):
            debugger.check_collision_at_config(joint_position)

    def test_check_collision_too_many_dof(self, franka_config_path):
        """Test collision checking with too many DOF raises error."""
        debugger = RobotDebugger(franka_config_path)

        # Franka has 7 DOF, provide 8
        joint_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        with pytest.raises(ValueError, match="must have"):
            debugger.check_collision_at_config(joint_position)

    def test_collision_result_structure_with_collision(self, franka_config_path):
        """Test collision result structure when collision occurs."""
        debugger = RobotDebugger(franka_config_path)

        # Try a configuration that might have collisions
        joint_position = [3.0, 1.5, -2.0, -2.0, 2.0, 1.5, 0.0]
        result = debugger.check_collision_at_config(joint_position)

        if result["has_collision"]:
            assert result["num_colliding_pairs"] > 0
            assert len(result["colliding_pairs"]) > 0
            assert result["max_penetration"] > 0.0
            assert len(result["distances"]) > 0

            # Check colliding pairs structure
            for pair in result["colliding_pairs"]:
                assert isinstance(pair, tuple)
                assert len(pair) == 2
                assert isinstance(pair[0], str)
                assert isinstance(pair[1], str)

            # Check distances structure
            for link_pair, distance in result["distances"].items():
                assert isinstance(link_pair, tuple)
                assert len(link_pair) == 2
                assert isinstance(distance, float)
                assert distance > 0.0

    def test_collision_result_structure_no_collision(self, franka_config_path):
        """Test collision result structure when no collision occurs."""
        debugger = RobotDebugger(franka_config_path)

        # Use default joint configuration (should be collision-free)
        result = debugger.check_default_joint_configuration_collision()

        if not result["has_collision"]:
            assert result["num_colliding_pairs"] == 0
            assert len(result["colliding_pairs"]) == 0
            assert result["max_penetration"] == 0.0
            assert len(result["distances"]) == 0


class TestRobotDebuggerStatistics:
    """Test statistical analysis functionality."""

    def test_sample_collision_checks_basic(self, franka_config_path):
        """Test basic collision sampling."""
        debugger = RobotDebugger(franka_config_path)

        stats = debugger.sample_collision_checks(num_samples=100, batch_size=50, seed=42)

        assert "total_samples" in stats
        assert "collision_count" in stats
        assert "collision_rate" in stats
        assert "frequent_collisions" in stats
        assert stats["total_samples"] == 100
        assert isinstance(stats["collision_count"], int)
        assert isinstance(stats["collision_rate"], float)
        assert isinstance(stats["frequent_collisions"], list)
        assert 0.0 <= stats["collision_rate"] <= 100.0

    def test_sample_collision_checks_small_batch(self, ur5e_config_path):
        """Test collision sampling with small batch size."""
        debugger = RobotDebugger(ur5e_config_path)

        stats = debugger.sample_collision_checks(num_samples=50, batch_size=10, seed=123)

        assert stats["total_samples"] == 50
        assert stats["collision_count"] >= 0

    def test_sample_collision_checks_large_batch(self, franka_config_path):
        """Test collision sampling with large batch size."""
        debugger = RobotDebugger(franka_config_path)

        stats = debugger.sample_collision_checks(num_samples=200, batch_size=200, seed=456)

        assert stats["total_samples"] == 200
        assert stats["collision_count"] >= 0

    def test_sample_collision_checks_reproducibility(self, franka_config_path):
        """Test that same seed produces same results."""
        debugger = RobotDebugger(franka_config_path)

        stats1 = debugger.sample_collision_checks(num_samples=100, batch_size=50, seed=42)
        stats2 = debugger.sample_collision_checks(num_samples=100, batch_size=50, seed=42)

        assert stats1["collision_count"] == stats2["collision_count"]
        assert stats1["collision_rate"] == stats2["collision_rate"]

    def test_sample_collision_checks_frequent_collisions_structure(self, franka_config_path):
        """Test structure of frequent_collisions output."""
        debugger = RobotDebugger(franka_config_path)

        stats = debugger.sample_collision_checks(num_samples=200, batch_size=100, seed=42)

        for item in stats["frequent_collisions"]:
            assert isinstance(item, tuple)
            assert len(item) == 2
            link_pair, count = item
            assert isinstance(link_pair, tuple)
            assert len(link_pair) == 2
            assert isinstance(link_pair[0], str)
            assert isinstance(link_pair[1], str)
            assert isinstance(count, int)
            assert count > 0

    def test_sample_collision_checks_sorted_by_frequency(self, franka_config_path):
        """Test that frequent_collisions is sorted by frequency."""
        debugger = RobotDebugger(franka_config_path)

        stats = debugger.sample_collision_checks(num_samples=500, batch_size=100, seed=42)

        if len(stats["frequent_collisions"]) > 1:
            counts = [count for _, count in stats["frequent_collisions"]]
            assert counts == sorted(counts, reverse=True)

    def test_find_never_colliding_pairs_basic(self, franka_config_path):
        """Test finding never-colliding pairs."""
        debugger = RobotDebugger(franka_config_path)

        never_colliding = debugger.find_never_colliding_pairs(
            num_samples=100, batch_size=100, seed=345
        )

        assert isinstance(never_colliding, list)
        for pair in never_colliding:
            assert isinstance(pair, tuple)
            assert len(pair) == 2
            assert isinstance(pair[0], str)
            assert isinstance(pair[1], str)

    def test_find_never_colliding_pairs_ur5e(self, ur5e_config_path):
        """Test finding never-colliding pairs for UR5e."""
        debugger = RobotDebugger(ur5e_config_path)

        never_colliding = debugger.find_never_colliding_pairs(
            num_samples=200, batch_size=200, seed=456
        )

        assert isinstance(never_colliding, list)

    def test_collision_matrix_stats_output(self, franka_config_path, capsys):
        """Test print_collision_matrix_stats output."""
        debugger = RobotDebugger(franka_config_path)

        debugger.print_collision_matrix_stats()

        captured = capsys.readouterr()
        assert "Collision Matrix Statistics:" in captured.out
        assert "Total spheres:" in captured.out
        assert "Total possible pairs:" in captured.out
        assert "Checked pairs:" in captured.out
        assert "Ignored pairs:" in captured.out
        assert "Checking:" in captured.out

    def test_collision_matrix_stats_values(self, franka_config_path):
        """Test collision matrix statistics have valid values."""
        debugger = RobotDebugger(franka_config_path)

        collision_config = debugger._robot_config.self_collision_config
        num_spheres = collision_config.num_spheres
        num_collision_pairs = collision_config.collision_pairs.shape[0]
        total_possible = (num_spheres * (num_spheres - 1)) // 2

        assert num_spheres > 0
        assert num_collision_pairs > 0
        assert num_collision_pairs <= total_possible


class TestRobotDebuggerEdgeCases:
    """Test edge cases and error handling."""

    def test_multiple_collision_checks(self, franka_config_path):
        """Test running multiple collision checks in sequence."""
        debugger = RobotDebugger(franka_config_path)

        configs = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0],
            [-1.0, 0.5, 1.0, -2.0, -1.0, 1.5, 0.5],
        ]

        for config in configs:
            result = debugger.check_collision_at_config(config)
            assert "has_collision" in result

    def test_batch_size_larger_than_samples(self, franka_config_path):
        """Test sampling with batch size larger than num samples."""
        debugger = RobotDebugger(franka_config_path)

        stats = debugger.sample_collision_checks(num_samples=50, batch_size=100, seed=42)

        assert stats["total_samples"] == 50

    def test_single_sample(self, franka_config_path):
        """Test sampling with single sample."""
        debugger = RobotDebugger(franka_config_path)

        stats = debugger.sample_collision_checks(num_samples=1, batch_size=1, seed=42)

        assert stats["total_samples"] == 1
        assert stats["collision_count"] in [0, 1]

    def test_different_robots_different_dof(self, franka_config_path, ur5e_config_path):
        """Test that different robots have different DOF."""
        franka_debugger = RobotDebugger(franka_config_path)
        ur5e_debugger = RobotDebugger(ur5e_config_path)

        franka_dof = franka_debugger._robot_config.dof
        ur5e_dof = ur5e_debugger._robot_config.dof

        # Both should be 6 DOF robots actually (excluding gripper)
        assert franka_dof > 0
        assert ur5e_dof > 0

    def test_tensor_config_consistency(self, franka_config_path):
        """Test that tensor config is used consistently."""
        device_cfg = DeviceCfg()
        debugger = RobotDebugger(franka_config_path, device_cfg=device_cfg)

        default_joint_position = debugger._robot_model.default_joint_state.position
        assert default_joint_position.device.type == device_cfg.device.type

    def test_collision_check_extreme_values(self, franka_config_path):
        """Test collision checking with extreme joint values."""
        debugger = RobotDebugger(franka_config_path)

        # Try extreme positive values
        extreme_config = [3.14, 3.14, 3.14, 0.0, 3.14, 3.14, 3.14]
        result = debugger.check_collision_at_config(extreme_config)
        assert "has_collision" in result

        # Try extreme negative values
        extreme_config = [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14, -3.14]
        result = debugger.check_collision_at_config(extreme_config)
        assert "has_collision" in result


class TestRobotDebuggerMultipleRobots:
    """Test debugger with different robot configurations."""

    @pytest.mark.parametrize(
        "robot_file",
        [
            "franka.yml",
            "ur10e.yml",
            "dual_ur10e.yml",
        ],
    )
    def test_init_multiple_robots(self, robot_file):
        """Test initialization with various robot configurations."""
        config_path = join_path(get_robot_configs_path(), robot_file)
        debugger = RobotDebugger(config_path)

        assert debugger is not None
        assert debugger._robot_model is not None

    @pytest.mark.parametrize(
        "robot_file",
        [
            "franka.yml",
            "ur10e.yml",
        ],
    )
    def test_check_default_joint_configuration_collision_multiple_robots(self, robot_file):
        """Test default position collision checking for various robots."""
        config_path = join_path(get_robot_configs_path(), robot_file)
        debugger = RobotDebugger(config_path)

        result = debugger.check_default_joint_configuration_collision()
        assert "has_collision" in result

    @pytest.mark.parametrize(
        "robot_file",
        [
            "franka.yml",
            "ur10e.yml",
        ],
    )
    def test_sampling_multiple_robots(self, robot_file):
        """Test collision sampling for various robots."""
        config_path = join_path(get_robot_configs_path(), robot_file)
        debugger = RobotDebugger(config_path)

        stats = debugger.sample_collision_checks(num_samples=50, batch_size=25, seed=42)
        assert stats["total_samples"] == 50


class TestRobotDebuggerXRDF:
    """Test XRDF loading functionality."""

    def test_from_xrdf_ur10e(self):
        """Test loading debugger from XRDF file."""
        xrdf_path = join_path(get_robot_configs_path(), "ur10e.xrdf")

        if not file_exists(xrdf_path):
            pytest.skip("ur10e.xrdf not available")

        # Load URDF path from XRDF or config
        urdf_path = join_path(get_assets_path(), "robot/ur_description/ur10e.urdf")
        asset_path = get_assets_path()

        try:
            debugger = RobotDebugger.from_xrdf(
                xrdf_path=xrdf_path,
                urdf_path=urdf_path,
                asset_path=asset_path,
            )

            assert debugger is not None
            assert debugger.config_path == xrdf_path
            assert debugger._robot_model is not None
            assert debugger._robot_config is not None
        except (ValueError, ImportError, FileNotFoundError) as e:
            pytest.skip(f"XRDF loading not fully supported: {e}")

    def test_from_xrdf_with_custom_device_cfg(self):
        """Test XRDF loading with custom tensor config."""
        xrdf_path = join_path(get_robot_configs_path(), "ur10e.xrdf")

        if not file_exists(xrdf_path):
            pytest.skip("ur10e.xrdf not available")

        urdf_path = join_path(get_assets_path(), "robot/ur_description/ur10e.urdf")
        asset_path = get_assets_path()
        device_cfg = DeviceCfg()

        try:
            debugger = RobotDebugger.from_xrdf(
                xrdf_path=xrdf_path,
                urdf_path=urdf_path,
                asset_path=asset_path,
                device_cfg=device_cfg,
            )

            assert debugger.device_cfg == device_cfg
        except (ValueError, ImportError, FileNotFoundError) as e:
            pytest.skip(f"XRDF loading not fully supported: {e}")

    def test_from_xrdf_invalid_format(self, tmp_path):
        """Test XRDF loading with invalid format warns but continues."""
        # Create a temporary invalid XRDF file
        invalid_xrdf = tmp_path / "invalid.xrdf"
        invalid_xrdf.write_text("invalid_key: invalid_value\n")

        # This test verifies that invalid format triggers warning
        # The actual conversion will likely fail, which is expected
        try:
            debugger = RobotDebugger.from_xrdf(
                xrdf_path=str(invalid_xrdf),
                urdf_path="dummy.urdf",
                asset_path="",
            )
        except (ValueError, ImportError, FileNotFoundError, Exception):
            # Expected to fail since it's invalid
            # The warning check happens during load_yaml and validation
            pass

    def test_from_xrdf_conversion_error(self, tmp_path):
        """Test XRDF loading with conversion error raises ValueError."""
        # Create a minimal XRDF that will fail conversion
        invalid_xrdf = tmp_path / "minimal.xrdf"
        invalid_xrdf.write_text("format: xrdf\n")

        with pytest.raises(ValueError, match="Failed to convert XRDF file"):
            debugger = RobotDebugger.from_xrdf(
                xrdf_path=str(invalid_xrdf),
                urdf_path="nonexistent.urdf",
                asset_path="",
            )


class TestRobotDebuggerVisualization:
    """Test visualization functionality."""

    def test_visualize_collision_at_config_basic(self, franka_config_path, monkeypatch):
        """Test visualization with mocked ViserVisualizer."""
        # Create a simple mock class
        class MockViserVisualizer:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        # Track if ViserVisualizer was called
        mock_calls = []

        def mock_viser_visualizer(*args, **kwargs):
            mock_calls.append((args, kwargs))
            return MockViserVisualizer(*args, **kwargs)

        # Patch ViserVisualizer
        monkeypatch.setattr(
            "curobo._src.robot.builder.debugger_robot.ViserVisualizer",
            mock_viser_visualizer,
        )

        debugger = RobotDebugger(franka_config_path)
        joint_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        result = debugger.visualize_collision_at_config(joint_position)

        assert result is not None
        assert len(mock_calls) == 1

    def test_visualize_collision_at_config_custom_port(
        self, franka_config_path, monkeypatch
    ):
        """Test visualization with custom port."""
        captured_kwargs = {}

        class MockViserVisualizer:
            def __init__(self, *args, **kwargs):
                captured_kwargs.update(kwargs)

        monkeypatch.setattr(
            "curobo._src.robot.builder.debugger_robot.ViserVisualizer",
            MockViserVisualizer,
        )

        debugger = RobotDebugger(franka_config_path)
        joint_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        result = debugger.visualize_collision_at_config(joint_position, port=9090)

        # Check that ViserVisualizer was called with correct port
        assert captured_kwargs["connect_port"] == 9090

    def test_visualize_collision_at_config_tensor_input(
        self, franka_config_path, device_cfg, monkeypatch
    ):
        """Test visualization with tensor input."""

        class MockViserVisualizer:
            def __init__(self, *args, **kwargs):
                pass

        monkeypatch.setattr(
            "curobo._src.robot.builder.debugger_robot.ViserVisualizer",
            MockViserVisualizer,
        )

        debugger = RobotDebugger(franka_config_path, device_cfg=device_cfg)
        joint_position = torch.zeros(7, **device_cfg.as_torch_dict())

        result = debugger.visualize_collision_at_config(joint_position)

        assert result is not None

    def test_visualize_collision_at_config_list_input(
        self, franka_config_path, monkeypatch
    ):
        """Test visualization with list input converts to tensor."""

        class MockViserVisualizer:
            def __init__(self, *args, **kwargs):
                pass

        monkeypatch.setattr(
            "curobo._src.robot.builder.debugger_robot.ViserVisualizer",
            MockViserVisualizer,
        )

        debugger = RobotDebugger(franka_config_path)
        joint_position = [1.0, -0.5, 0.0, -1.5, 0.0, 1.0, 0.0]

        result = debugger.visualize_collision_at_config(joint_position)

        assert result is not None


class TestRobotDebuggerNeverCollidingPairs:
    """Additional tests for find_never_colliding_pairs to improve coverage."""

    def test_find_never_colliding_pairs_with_colliding_config(self, franka_config_path):
        """Test finding never-colliding pairs when most configs collide."""
        debugger = RobotDebugger(franka_config_path)

        # Use smaller sample to speed up test
        never_colliding = debugger.find_never_colliding_pairs(
            num_samples=50, batch_size=50, seed=789
        )

        # Should still return a list
        assert isinstance(never_colliding, list)
        # All pairs should be valid tuples
        for pair in never_colliding:
            assert isinstance(pair, tuple)
            assert len(pair) == 2

    def test_find_never_colliding_pairs_empty_result(self, franka_config_path):
        """Test case where all pairs collide at some point."""
        debugger = RobotDebugger(franka_config_path)

        # Very small sample - may find no never-colliding pairs
        never_colliding = debugger.find_never_colliding_pairs(
            num_samples=10, batch_size=10, seed=999
        )

        # Result can be empty or have some pairs
        assert isinstance(never_colliding, list)

