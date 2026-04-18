# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Standard Library
from unittest.mock import MagicMock, patch

# Third Party
import pytest

# CuRobo
from curobo._src.util.xrdf_util import (
    convert_curobo_to_xrdf,
    convert_xrdf_to_curobo,
    return_value_if_exists,
)


class TestReturnValueIfExists:
    def test_key_exists(self):
        """Test returning value when key exists."""
        input_dict = {"key1": "value1", "key2": 123}
        result = return_value_if_exists(input_dict, "key1")
        assert result == "value1"

    def test_key_missing_raise_error(self):
        """Test raising error when key is missing."""
        input_dict = {"key1": "value1"}
        with pytest.raises(ValueError):
            return_value_if_exists(input_dict, "missing_key")

    def test_key_missing_no_error(self):
        """Test returning None when key is missing and raise_error=False."""
        input_dict = {"key1": "value1"}
        result = return_value_if_exists(input_dict, "missing_key", raise_error=False)
        assert result is None

    def test_custom_suffix(self):
        """Test custom suffix in error message."""
        input_dict = {"key1": "value1"}
        with pytest.raises(ValueError, match="custom_suffix"):
            return_value_if_exists(input_dict, "missing_key", suffix="custom_suffix")


class TestConvertXrdfToCurobo:
    @pytest.fixture
    def sample_xrdf_dict(self):
        """Create a sample XRDF dictionary for testing."""
        return {
            "format": "xrdf",
            "format_version": 1.0,
            "tool_frames": ["tool0", "tool1"],
            "geometry": {
                "collision_model": {
                    "spheres": {
                        "link1": [[0.0, 0.0, 0.0, 0.1]],
                        "link2": [[0.0, 0.0, 0.1, 0.1]],
                    }
                }
            },
            "collision": {"geometry": "collision_model", "buffer_distance": 0.01},
            "self_collision": {
                "geometry": "collision_model",
                "ignore": [["link1", "link2"]],
                "buffer_distance": {},
            },
            "cspace": {
                "joint_names": ["joint1", "joint2"],
                "acceleration_limits": [1.0, 1.5],
                "jerk_limits": [10.0, 15.0],
            },
            "default_joint_positions": {"joint1": 0.0, "joint2": 0.5},
        }

    @pytest.fixture
    def mock_content_path(self):
        """Create a mock ContentPath object."""
        # CuRobo
        from curobo._src.types.content_path import ContentPath

        content_path = ContentPath(
            robot_urdf_file="robot.urdf",
            robot_xrdf_file="robot.xrdf",
            robot_asset_subroot_path="assets",
        )
        return content_path

    @pytest.fixture
    def mock_kinematics_parser(self):
        """Create a mock UrdfRobotParser."""
        parser = MagicMock()
        parser.get_controlled_joint_names.return_value = ["joint1", "joint2", "joint3"]
        parser.root_link = "base_link"
        return parser

    def test_convert_xrdf_to_curobo_basic(
        self, sample_xrdf_dict, mock_content_path, mock_kinematics_parser
    ):
        """Test basic XRDF to CuRobo conversion."""
        with patch(
            "curobo._src.util.xrdf_util.UrdfRobotParser", return_value=mock_kinematics_parser
        ):
            result = convert_xrdf_to_curobo(mock_content_path, sample_xrdf_dict)

        assert "robot_cfg" in result
        assert "kinematics" in result["robot_cfg"]
        kin = result["robot_cfg"]["kinematics"]

        assert "collision_spheres" in kin
        assert "collision_sphere_buffer" in kin
        assert kin["collision_sphere_buffer"] == 0.01
        assert "self_collision_ignore" in kin
        assert "cspace" in kin
        assert "base_link" in kin
        assert kin["base_link"] == "base_link"

    def test_convert_xrdf_invalid_format(self, mock_content_path, mock_kinematics_parser):
        """Test error when format is not xrdf."""
        invalid_dict = {"format": "invalid", "format_version": 1.0}
        with pytest.raises(ValueError, match="format is not xrdf"):
            convert_xrdf_to_curobo(mock_content_path, invalid_dict)

    def test_convert_xrdf_no_urdf_path(self, sample_xrdf_dict):
        """Test error when URDF path is not provided."""
        # CuRobo
        from curobo._src.types.content_path import ContentPath

        content_path = ContentPath()
        with pytest.raises(ValueError):
            convert_xrdf_to_curobo(content_path, sample_xrdf_dict)

    def test_convert_xrdf_no_collision(self, mock_content_path, mock_kinematics_parser):
        """Test conversion when collision section is missing."""
        xrdf_dict = {
            "format": "xrdf",
            "format_version": 1.0,
            "tool_frames": ["tool0"],
            "cspace": {
                "joint_names": ["joint1"],
                "acceleration_limits": [1.0],
                "jerk_limits": [10.0],
            },
            "default_joint_positions": {"joint1": 0.0},
        }
        with patch(
            "curobo._src.util.xrdf_util.UrdfRobotParser", return_value=mock_kinematics_parser
        ):
            # When collision is missing, code logs warning, continues but doesn't check self_collision
            # The function completes successfully with warnings
            result = convert_xrdf_to_curobo(mock_content_path, xrdf_dict)
            # Result should still be valid but without collision data
            assert "robot_cfg" in result

    def test_convert_xrdf_with_modifiers(
        self, sample_xrdf_dict, mock_content_path, mock_kinematics_parser
    ):
        """Test conversion with modifiers."""
        sample_xrdf_dict["modifiers"] = [
            {"set_base_frame": "custom_base"},
            {
                "add_frame": {
                    "frame_name": "custom_frame",
                    "parent_frame_name": "link1",
                    "joint_name": "custom_joint",
                    "joint_type": "fixed",
                    "fixed_transform": {
                        "position": [0.1, 0.2, 0.3],
                        "orientation": {"w": 1.0, "xyz": [0.0, 0.0, 0.0]},
                    },
                }
            },
        ]

        with patch(
            "curobo._src.util.xrdf_util.UrdfRobotParser", return_value=mock_kinematics_parser
        ):
            result = convert_xrdf_to_curobo(mock_content_path, sample_xrdf_dict)

        kin = result["robot_cfg"]["kinematics"]
        assert kin["base_link"] == "custom_base"
        assert "extra_links" in kin
        assert "custom_frame" in kin["extra_links"]

    def test_convert_xrdf_locked_joints(
        self, sample_xrdf_dict, mock_content_path, mock_kinematics_parser
    ):
        """Test conversion with locked joints."""
        with patch(
            "curobo._src.util.xrdf_util.UrdfRobotParser", return_value=mock_kinematics_parser
        ):
            result = convert_xrdf_to_curobo(mock_content_path, sample_xrdf_dict)

        kin = result["robot_cfg"]["kinematics"]
        # joint3 should be locked since it's in URDF but not in active joints
        assert "lock_joints" in kin
        assert "joint3" in kin["lock_joints"]

    def test_convert_xrdf_format_version_warning(self, sample_xrdf_dict, mock_content_path, mock_kinematics_parser):
        """Test warning when format_version > 1.0."""
        sample_xrdf_dict["format_version"] = 2.0
        with patch(
            "curobo._src.util.xrdf_util.UrdfRobotParser", return_value=mock_kinematics_parser
        ):
            result = convert_xrdf_to_curobo(mock_content_path, sample_xrdf_dict)
        assert result is not None

    def test_convert_xrdf_string_content_path(self, sample_xrdf_dict):
        """Test error when content_path is a string instead of ContentPath."""
        # The check happens after accessing robot_urdf_absolute_path, so we get AttributeError first
        with pytest.raises(AttributeError):
            convert_xrdf_to_curobo("/path/to/file", sample_xrdf_dict)

    def test_convert_xrdf_no_spheres_in_geometry(self, sample_xrdf_dict, mock_content_path, mock_kinematics_parser):
        """Test error when spheres key is missing from geometry."""
        sample_xrdf_dict["geometry"]["collision_model"] = {}
        with patch(
            "curobo._src.util.xrdf_util.UrdfRobotParser", return_value=mock_kinematics_parser
        ):
            with pytest.raises(ValueError, match="spheres key not found"):
                convert_xrdf_to_curobo(mock_content_path, sample_xrdf_dict)

    def test_convert_xrdf_mismatched_self_collision_geometry(
        self, sample_xrdf_dict, mock_content_path, mock_kinematics_parser
    ):
        """Test error when self_collision geometry doesn't match collision geometry."""
        sample_xrdf_dict["self_collision"]["geometry"] = "different_model"
        with patch(
            "curobo._src.util.xrdf_util.UrdfRobotParser", return_value=mock_kinematics_parser
        ):
            with pytest.raises(ValueError, match="self_collision geometry does not match"):
                convert_xrdf_to_curobo(mock_content_path, sample_xrdf_dict)

    def test_convert_xrdf_no_buffer_distance(self, sample_xrdf_dict, mock_content_path, mock_kinematics_parser):
        """Test conversion when buffer_distance is not specified."""
        del sample_xrdf_dict["collision"]["buffer_distance"]
        with patch(
            "curobo._src.util.xrdf_util.UrdfRobotParser", return_value=mock_kinematics_parser
        ):
            result = convert_xrdf_to_curobo(mock_content_path, sample_xrdf_dict)
        kin = result["robot_cfg"]["kinematics"]
        # Should default to 0.0
        assert kin["collision_sphere_buffer"] == 0.0

    def test_convert_xrdf_no_self_collision_buffer(self, sample_xrdf_dict, mock_content_path, mock_kinematics_parser):
        """Test conversion when self_collision buffer_distance is not specified."""
        del sample_xrdf_dict["self_collision"]["buffer_distance"]
        with patch(
            "curobo._src.util.xrdf_util.UrdfRobotParser", return_value=mock_kinematics_parser
        ):
            result = convert_xrdf_to_curobo(mock_content_path, sample_xrdf_dict)
        kin = result["robot_cfg"]["kinematics"]
        # Should default to empty dict
        assert kin["self_collision_buffer"] == {}

    def test_convert_xrdf_modifier_multiple_keys_error(self, sample_xrdf_dict, mock_content_path, mock_kinematics_parser):
        """Test error when modifier has multiple keys."""
        sample_xrdf_dict["modifiers"] = [
            {"set_base_frame": "base", "another_key": "value"}
        ]
        with patch(
            "curobo._src.util.xrdf_util.UrdfRobotParser", return_value=mock_kinematics_parser
        ):
            with pytest.raises(ValueError, match="Each modifier should have only one key"):
                convert_xrdf_to_curobo(mock_content_path, sample_xrdf_dict)

    def test_convert_xrdf_unknown_modifier(self, sample_xrdf_dict, mock_content_path, mock_kinematics_parser):
        """Test warning for unknown modifier type."""
        sample_xrdf_dict["modifiers"] = [
            {"unknown_modifier": "some_value"}
        ]
        with patch(
            "curobo._src.util.xrdf_util.UrdfRobotParser", return_value=mock_kinematics_parser
        ):
            result = convert_xrdf_to_curobo(mock_content_path, sample_xrdf_dict)
        # Should complete with warning
        assert result is not None

    def test_convert_xrdf_with_dynamics(self, sample_xrdf_dict, mock_content_path, mock_kinematics_parser):
        """Test conversion with dynamics section."""
        sample_xrdf_dict["dynamics"] = {"some_key": "some_value"}
        with patch(
            "curobo._src.util.xrdf_util.UrdfRobotParser", return_value=mock_kinematics_parser
        ):
            result = convert_xrdf_to_curobo(mock_content_path, sample_xrdf_dict)
        # Dynamics should be preserved in output
        assert "dynamics" in result["robot_cfg"]
        assert result["robot_cfg"]["dynamics"] == {"some_key": "some_value"}


class TestConvertCuroboToXrdf:
    @pytest.fixture
    def sample_curobo_dict(self):
        """Create a sample CuRobo dictionary for testing."""
        return {
            "robot_cfg": {
                "kinematics": {
                    "tool_frames": ["tool0", "tool1"],
                    "collision_spheres": {
                        "link1": [[0.0, 0.0, 0.0, 0.1]],
                        "link2": [[0.0, 0.0, 0.1, 0.1]],
                    },
                    "collision_sphere_buffer": 0.01,
                    "self_collision_ignore": [["link1", "link2"]],
                    "self_collision_buffer": {},
                    "cspace": {
                        "joint_names": ["joint1", "joint2"],
                        "default_joint_position": [0.0, 0.5],
                        "max_acceleration": [1.0, 1.5],
                        "max_jerk": [10.0, 15.0],
                    },
                    "lock_joints": {},
                    "base_link": "base_link",
                    "extra_links": {},
                }
            }
        }

    def test_convert_curobo_to_xrdf_basic(self, sample_curobo_dict):
        """Test basic CuRobo to XRDF conversion."""
        result = convert_curobo_to_xrdf(sample_curobo_dict)

        assert result["format"] == "xrdf"
        assert result["format_version"] == 1.0
        assert "tool_frames" in result
        assert "geometry" in result
        assert "collision" in result
        assert "self_collision" in result
        assert "cspace" in result
        assert "default_joint_positions" in result

    def test_convert_curobo_to_xrdf_without_envelope(self):
        """Test conversion with just kinematics key (no robot_cfg wrapper)."""
        inner_dict = {
            "kinematics": {
                "tool_frames": ["tool0"],
                "collision_spheres": {},
                "self_collision_ignore": [],
                "self_collision_buffer": {},
                "collision_sphere_buffer": 0.0,
                "cspace": {
                    "joint_names": ["joint1"],
                    "default_joint_position": [0.0],
                    "max_acceleration": [1.0],
                    "max_jerk": [10.0],
                },
                "lock_joints": {},
            }
        }
        result = convert_curobo_to_xrdf(inner_dict)
        assert result["format"] == "xrdf"

    def test_convert_curobo_to_xrdf_with_modifiers(self, sample_curobo_dict):
        """Test conversion with base_link and extra_links."""
        sample_curobo_dict["robot_cfg"]["kinematics"]["base_link"] = "custom_base"
        sample_curobo_dict["robot_cfg"]["kinematics"]["extra_links"] = {
            "custom_frame": {
                "link_name": "custom_frame",
                "parent_link_name": "link1",
                "joint_name": "custom_joint",
                "joint_type": "fixed",
                "fixed_transform": [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
            }
        }

        result = convert_curobo_to_xrdf(sample_curobo_dict)

        assert "modifiers" in result
        assert len(result["modifiers"]) == 2
        assert result["modifiers"][0]["set_base_frame"] == "custom_base"
        assert "add_frame" in result["modifiers"][1]

    def test_convert_curobo_to_xrdf_no_collision_spheres(self, sample_curobo_dict):
        """Test conversion without collision spheres."""
        del sample_curobo_dict["robot_cfg"]["kinematics"]["collision_spheres"]

        result = convert_curobo_to_xrdf(sample_curobo_dict)

        assert "geometry" in result
        assert "collision" in result
        # Should have empty spheres dict
        assert result["geometry"]["collision_model"]["spheres"] == {}

    def test_convert_curobo_to_xrdf_scalar_limits(self, sample_curobo_dict):
        """Test conversion with scalar acceleration/jerk limits."""
        sample_curobo_dict["robot_cfg"]["kinematics"]["cspace"]["max_acceleration"] = 2.0
        sample_curobo_dict["robot_cfg"]["kinematics"]["cspace"]["max_jerk"] = 20.0

        result = convert_curobo_to_xrdf(sample_curobo_dict)

        assert "cspace" in result
        assert isinstance(result["cspace"]["acceleration_limits"], list)
        assert isinstance(result["cspace"]["jerk_limits"], list)

    def test_convert_curobo_to_xrdf_invalid_input(self):
        """Test error handling for invalid input."""
        invalid_dict = {"invalid": "data"}
        with pytest.raises(ValueError):
            convert_curobo_to_xrdf(invalid_dict)

    def test_convert_curobo_to_xrdf_with_locked_joints(self, sample_curobo_dict):
        """Test conversion with locked joints."""
        sample_curobo_dict["robot_cfg"]["kinematics"]["lock_joints"] = {"joint3": 0.7}
        sample_curobo_dict["robot_cfg"]["kinematics"]["cspace"]["joint_names"] = [
            "joint1",
            "joint2",
            "joint3",
        ]
        sample_curobo_dict["robot_cfg"]["kinematics"]["cspace"]["default_joint_position"] = [0.0, 0.5, 0.7]
        sample_curobo_dict["robot_cfg"]["kinematics"]["cspace"]["max_acceleration"] = [
            1.0,
            1.5,
            2.0,
        ]
        sample_curobo_dict["robot_cfg"]["kinematics"]["cspace"]["max_jerk"] = [10.0, 15.0, 20.0]

        result = convert_curobo_to_xrdf(sample_curobo_dict)

        # Only joint1 and joint2 should be in cspace (joint3 is locked)
        assert result["cspace"]["joint_names"] == ["joint1", "joint2"]
        # Default positions should include all joints
        assert "joint3" in result["default_joint_positions"]
        assert result["default_joint_positions"]["joint3"] == 0.7

    def test_convert_curobo_to_xrdf_minimal_kinematics(self):
        """Test conversion with minimal kinematics dict."""
        minimal_dict = {
            "robot_cfg": {
                "kinematics": {
                    "tool_frames": ["tool0"],
                    "self_collision_ignore": [],
                    "self_collision_buffer": {},
                    "cspace": {
                        "joint_names": [],
                        "default_joint_position": [],
                        "max_acceleration": [],
                        "max_jerk": [],
                    },
                    "lock_joints": None,
                }
            }
        }
        result = convert_curobo_to_xrdf(minimal_dict)
        # Should still return a valid XRDF structure
        assert result["format"] == "xrdf"
        assert "cspace" in result
        assert result["tool_frames"] == ["tool0"]

    def test_convert_curobo_to_xrdf_custom_geometry_name(self, sample_curobo_dict):
        """Test conversion with custom geometry name."""
        result = convert_curobo_to_xrdf(sample_curobo_dict, geometry_name="custom_geometry")

        assert "custom_geometry" in result["geometry"]
        assert result["collision"]["geometry"] == "custom_geometry"
        assert result["self_collision"]["geometry"] == "custom_geometry"


class TestXrdfRoundTripWithRealConfig:
    """Integration tests using real robot configuration files."""

    def test_load_franka_config(self):
        """Test loading Franka robot configuration."""
        # CuRobo
        from curobo._src.util.config_io import load_yaml
        from curobo.content import get_robot_configs_path

        franka_path = get_robot_configs_path() / "franka.yml"
        if not franka_path.exists():
            pytest.skip("Franka config not found")

        config = load_yaml(str(franka_path))
        assert "robot_cfg" in config
        assert "kinematics" in config["robot_cfg"]

    def test_convert_franka_to_xrdf(self):
        """Test converting Franka CuRobo config to XRDF."""
        # CuRobo
        from curobo._src.util.config_io import load_yaml
        from curobo.content import get_robot_configs_path

        franka_path = get_robot_configs_path() / "franka.yml"
        if not franka_path.exists():
            pytest.skip("Franka config not found")

        curobo_config = load_yaml(str(franka_path))

        # Mock the load_yaml call for collision_spheres file
        mock_spheres = {"link1": [[0.0, 0.0, 0.0, 0.1]]}
        with patch("curobo._src.util.xrdf_util.load_yaml", return_value=mock_spheres):
            xrdf_config = convert_curobo_to_xrdf(curobo_config)

        # Verify XRDF structure
        assert xrdf_config["format"] == "xrdf"
        assert xrdf_config["format_version"] == 1.0
        assert "tool_frames" in xrdf_config
        assert "geometry" in xrdf_config
        assert "collision" in xrdf_config
        assert "self_collision" in xrdf_config
        assert "cspace" in xrdf_config
        assert "default_joint_positions" in xrdf_config

    def test_xrdf_round_trip_consistency(self):
        """Test round-trip conversion: CuRobo -> XRDF -> CuRobo."""
        # CuRobo
        from curobo._src.util.config_io import load_yaml
        from curobo.content import get_robot_configs_path

        franka_path = get_robot_configs_path() / "franka.yml"
        if not franka_path.exists():
            pytest.skip("Franka config not found")

        # Load original CuRobo config
        original_curobo = load_yaml(str(franka_path))

        # Mock the load_yaml call for collision_spheres file
        mock_spheres = {"panda_link0": [[0.0, 0.0, 0.0, 0.1]]}
        with patch("curobo._src.util.xrdf_util.load_yaml", return_value=mock_spheres):
            # Convert to XRDF
            xrdf_config = convert_curobo_to_xrdf(original_curobo)

        # Verify key fields are preserved
        original_kin = original_curobo["robot_cfg"]["kinematics"]
        assert xrdf_config["tool_frames"] == original_kin["tool_frames"]

        # Verify cspace
        cspace_joint_names = original_kin["cspace"]["joint_names"]
        locked_joints = original_kin.get("lock_joints", {})
        # Active joints are those not locked
        active_joints = [j for j in cspace_joint_names if j not in locked_joints]
        assert xrdf_config["cspace"]["joint_names"] == active_joints

    def test_xrdf_geometry_conversion(self):
        """Test that collision geometry is properly converted."""
        # CuRobo
        from curobo._src.util.config_io import load_yaml
        from curobo.content import get_robot_configs_path

        franka_path = get_robot_configs_path() / "franka.yml"
        if not franka_path.exists():
            pytest.skip("Franka config not found")

        curobo_config = load_yaml(str(franka_path))

        # Mock the load_yaml call for collision_spheres file
        mock_spheres = {"panda_link0": [[0.0, 0.0, 0.0, 0.1]]}
        with patch("curobo._src.util.xrdf_util.load_yaml", return_value=mock_spheres):
            xrdf_config = convert_curobo_to_xrdf(curobo_config)

        # Should have collision geometry
        assert "geometry" in xrdf_config
        assert "collision_model" in xrdf_config["geometry"]
        assert "spheres" in xrdf_config["geometry"]["collision_model"]

    def test_xrdf_modifiers_conversion(self):
        """Test that base_link and extra_links are converted to modifiers."""
        # CuRobo
        from curobo._src.util.config_io import load_yaml
        from curobo.content import get_robot_configs_path

        franka_path = get_robot_configs_path() / "franka.yml"
        if not franka_path.exists():
            pytest.skip("Franka config not found")

        curobo_config = load_yaml(str(franka_path))
        original_kin = curobo_config["robot_cfg"]["kinematics"]

        # Mock the load_yaml call for collision_spheres file
        mock_spheres = {"panda_link0": [[0.0, 0.0, 0.0, 0.1]]}
        with patch("curobo._src.util.xrdf_util.load_yaml", return_value=mock_spheres):
            xrdf_config = convert_curobo_to_xrdf(curobo_config)

        # Should have modifiers if base_link or extra_links exist
        if "base_link" in original_kin or "extra_links" in original_kin:
            assert "modifiers" in xrdf_config

        # If extra_links exist, check they're converted
        if "extra_links" in original_kin and original_kin["extra_links"]:
            add_frame_modifiers = [m for m in xrdf_config["modifiers"] if "add_frame" in m]
            assert len(add_frame_modifiers) > 0

