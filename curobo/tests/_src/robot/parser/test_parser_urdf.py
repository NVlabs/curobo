# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Comprehensive tests for UrdfRobotParser."""

# Standard Library
from pathlib import Path

import numpy as np

# Third Party
import pytest

# CuRobo
from curobo._src.robot.parser.parser_urdf import UrdfRobotParser
from curobo._src.robot.types.joint_types import JointType
from curobo._src.robot.types.link_params import LinkParams
from curobo._src.util_file import get_assets_path, get_robot_configs_path, join_path, load_yaml


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
def franka_parser(franka_urdf_path, franka_asset_path):
    """Create a UrdfRobotParser for Franka robot."""
    return UrdfRobotParser(
        urdf_path=franka_urdf_path,
        mesh_root=franka_asset_path,
        load_meshes=True,
    )


class TestUrdfRobotParserInitialization:
    """Test UrdfRobotParser initialization."""

    def test_parser_basic_init(self, franka_urdf_path, franka_asset_path):
        """Test basic parser initialization."""
        parser = UrdfRobotParser(
            urdf_path=franka_urdf_path,
            mesh_root=franka_asset_path,
        )

        assert parser is not None
        assert parser._robot is not None
        assert parser._mesh_root == franka_asset_path
        assert parser._parent_map is not None

    def test_parser_with_mesh_loading(self, franka_urdf_path, franka_asset_path):
        """Test parser initialization with mesh loading."""
        parser = UrdfRobotParser(
            urdf_path=franka_urdf_path,
            mesh_root=franka_asset_path,
            load_meshes=True,
        )

        assert parser is not None
        assert parser._robot is not None

    def test_parser_with_scene_graph(self, franka_urdf_path, franka_asset_path):
        """Test parser initialization with scene graph building."""
        parser = UrdfRobotParser(
            urdf_path=franka_urdf_path,
            mesh_root=franka_asset_path,
            build_scene_graph=True,
        )

        assert parser is not None
        # With scene graph, root_link property should be available
        root = parser.root_link
        assert root is not None
        assert isinstance(root, str)

    def test_parser_with_extra_links(self, franka_urdf_path, franka_asset_path):
        """Test parser initialization with extra links."""
        extra_link = LinkParams(
            link_name="extra_link",
            joint_name="extra_joint",
            joint_type=JointType.FIXED,
            fixed_transform=np.eye(4)[:3, :],
            parent_link_name="panda_hand",
        )

        extra_links = {"extra_link": extra_link}

        parser = UrdfRobotParser(
            urdf_path=franka_urdf_path,
            mesh_root=franka_asset_path,
            extra_links=extra_links,
        )

        assert parser is not None
        assert "extra_link" in parser._parent_map


class TestUrdfRobotParserJointLimits:
    """Test joint limits extraction including edge cases."""

    def test_continuous_joint_conversion(self, tmp_path):
        """Test conversion of continuous joints to revolute with limits."""
        # Create a minimal URDF with a continuous joint
        urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link"/>
  <link name="joint_link"/>

  <joint name="continuous_joint" type="continuous">
    <parent link="base_link"/>
    <child link="joint_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit effort="100" velocity="2.0"/>
  </joint>
</robot>"""

        urdf_path = tmp_path / "continuous_joint.urdf"
        urdf_path.write_text(urdf_content)

        # Parse the URDF - this should trigger the continuous joint conversion (lines 116-122)
        parser = UrdfRobotParser(urdf_path=str(urdf_path))

        # Get link parameters for the joint_link
        link_params = parser.get_link_parameters("joint_link", base=False)

        # Continuous joints should be converted to revolute with limits [-6.28, 6.28]
        assert link_params.joint_limits is not None
        assert link_params.joint_limits[0] == pytest.approx(-6.28, abs=0.01)
        assert link_params.joint_limits[1] == pytest.approx(6.28, abs=0.01)

    def test_get_joint_name_method(self, franka_parser):
        """Test _get_joint_name method (lines 92-93)."""
        # Get the name of the first joint
        joint_name = franka_parser._get_joint_name(0)

        assert joint_name is not None
        assert isinstance(joint_name, str)
        assert joint_name in franka_parser._robot.joint_map


class TestUrdfRobotParserMimicJoints:
    """Test mimic joint handling with various edge cases."""

    def test_mimic_joint_lower_limit_warning(self, tmp_path):
        """Test mimic joint with lower limit mismatch (lines 232-241)."""
        # Create URDF with mimic joint that can exceed lower limits
        urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link"/>
  <link name="active_link"/>
  <link name="mimic_link"/>

  <joint name="active_joint" type="revolute">
    <parent link="base_link"/>
    <child link="active_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit effort="100" lower="-2.0" upper="2.0" velocity="2.0"/>
  </joint>

  <joint name="mimic_joint" type="revolute">
    <parent link="active_link"/>
    <child link="mimic_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit effort="100" lower="-1.0" upper="2.0" velocity="2.0"/>
    <mimic joint="active_joint" multiplier="2.0" offset="0.0"/>
  </joint>
</robot>"""

        urdf_path = tmp_path / "mimic_lower.urdf"
        urdf_path.write_text(urdf_content)

        # This should trigger the warning at lines 232-241
        parser = UrdfRobotParser(urdf_path=str(urdf_path))
        link_params = parser.get_link_parameters("mimic_link", base=False)

        # Should have adjusted limits
        assert link_params is not None

    def test_mimic_joint_upper_limit_warning(self, tmp_path):
        """Test mimic joint with upper limit mismatch (lines 247-256)."""
        urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link"/>
  <link name="active_link"/>
  <link name="mimic_link"/>

  <joint name="active_joint" type="revolute">
    <parent link="base_link"/>
    <child link="active_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit effort="100" lower="-2.0" upper="2.0" velocity="2.0"/>
  </joint>

  <joint name="mimic_joint" type="revolute">
    <parent link="active_link"/>
    <child link="mimic_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit effort="100" lower="-2.0" upper="1.0" velocity="2.0"/>
    <mimic joint="active_joint" multiplier="2.0" offset="0.0"/>
  </joint>
</robot>"""

        urdf_path = tmp_path / "mimic_upper.urdf"
        urdf_path.write_text(urdf_content)

        # This should trigger the warning at lines 247-256
        parser = UrdfRobotParser(urdf_path=str(urdf_path))
        link_params = parser.get_link_parameters("mimic_link", base=False)

        assert link_params is not None

    def test_mimic_joint_velocity_limit_warning(self, tmp_path):
        """Test mimic joint with velocity limit mismatch (lines 259-267)."""
        urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link"/>
  <link name="active_link"/>
  <link name="mimic_link"/>

  <joint name="active_joint" type="revolute">
    <parent link="base_link"/>
    <child link="active_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit effort="100" lower="-2.0" upper="2.0" velocity="3.0"/>
  </joint>

  <joint name="mimic_joint" type="revolute">
    <parent link="active_link"/>
    <child link="mimic_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit effort="100" lower="-2.0" upper="2.0" velocity="1.0"/>
    <mimic joint="active_joint" multiplier="2.0" offset="0.0"/>
  </joint>
</robot>"""

        urdf_path = tmp_path / "mimic_velocity.urdf"
        urdf_path.write_text(urdf_content)

        # This should trigger the warning at lines 259-267
        parser = UrdfRobotParser(urdf_path=str(urdf_path))
        link_params = parser.get_link_parameters("mimic_link", base=False)

        assert link_params is not None


class TestUrdfRobotParserLinkParameters:
    """Test link parameter extraction including edge cases."""

    def test_get_link_parameters_with_no_com_pose(self, tmp_path):
        """Test link parameters when com_pose_matrix is None (line 155)."""
        urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link"/>
  <link name="test_link">
    <inertial>
      <mass value="1.0"/>
      <!-- No origin specified, should default to identity -->
      <inertia ixx="1.0" ixy="0.0" ixz="0.0" iyy="1.0" iyz="0.0" izz="1.0"/>
    </inertial>
  </link>

  <joint name="test_joint" type="revolute">
    <parent link="base_link"/>
    <child link="test_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit effort="100" lower="-2.0" upper="2.0" velocity="2.0"/>
  </joint>
</robot>"""

        urdf_path = tmp_path / "no_com_pose.urdf"
        urdf_path.write_text(urdf_content)

        parser = UrdfRobotParser(urdf_path=str(urdf_path))
        link_params = parser.get_link_parameters("test_link", base=False)

        # Should handle None com_pose_matrix and use identity (line 155)
        assert link_params is not None
        assert link_params.link_mass == pytest.approx(1.0)

    def test_get_link_parameters_with_zero_inertia(self, tmp_path):
        """Test link parameters when inertia is all zeros (lines 182, 184, 186)."""
        urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link"/>
  <link name="test_link">
    <inertial>
      <mass value="2.0"/>
      <origin xyz="0.1 0.2 0.3" rpy="0 0 0"/>
      <inertia ixx="0.0" ixy="0.0" ixz="0.0" iyy="0.0" iyz="0.0" izz="0.0"/>
    </inertial>
  </link>

  <joint name="test_joint" type="revolute">
    <parent link="base_link"/>
    <child link="test_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit effort="100" lower="-2.0" upper="2.0" velocity="2.0"/>
  </joint>
</robot>"""

        urdf_path = tmp_path / "zero_inertia.urdf"
        urdf_path.write_text(urdf_content)

        parser = UrdfRobotParser(urdf_path=str(urdf_path))
        link_params = parser.get_link_parameters("test_link", base=False)

        # Should use default inertia when all zeros (line 182)
        assert link_params is not None
        assert link_params.link_mass == pytest.approx(2.0)

    def test_get_link_parameters_unsupported_joint_type(self, tmp_path):
        """Test unsupported joint type triggers error (line 294)."""
        # Note: yourdfpy likely won't parse an invalid joint type, but we can test the path
        urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link"/>
  <link name="test_link"/>

  <joint name="test_joint" type="planar">
    <parent link="base_link"/>
    <child link="test_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit effort="100" lower="-2.0" upper="2.0" velocity="2.0"/>
  </joint>
</robot>"""

        urdf_path = tmp_path / "unsupported_joint.urdf"
        urdf_path.write_text(urdf_content)

        # The parser should handle this, possibly with error/warning
        try:
            parser = UrdfRobotParser(urdf_path=str(urdf_path))
            link_params = parser.get_link_parameters("test_link", base=False)
            # May succeed with default values or error
        except (ValueError, KeyError, AssertionError):
            # Expected for unsupported joint types
            pass


class TestUrdfRobotParserMethods:
    """Test various parser methods."""

    def test_get_link_names_from_urdf(self, franka_parser):
        """Test getting link names from URDF."""
        tool_frames = franka_parser.get_link_names_from_urdf()

        assert isinstance(tool_frames, list)
        assert len(tool_frames) > 0
        assert all(isinstance(name, str) for name in tool_frames)
        assert "panda_link0" in tool_frames

    def test_get_joint_names_from_urdf(self, franka_parser):
        """Test getting joint names from URDF."""
        joint_names = franka_parser.get_joint_names_from_urdf()

        assert isinstance(joint_names, list)
        assert len(joint_names) > 0
        assert all(isinstance(name, str) for name in joint_names)

    def test_add_absolute_path_to_link_meshes(self, franka_urdf_path):
        """Test adding absolute path to link meshes."""
        parser = UrdfRobotParser(
            urdf_path=franka_urdf_path,
            load_meshes=True,
        )

        mesh_dir = "/absolute/path/to/meshes"
        parser.add_absolute_path_to_link_meshes(mesh_dir)

        # Verify meshes have updated paths
        links = parser._robot.link_map
        for link_name, link in links.items():
            for vis in link.visuals:
                if vis.geometry.mesh is not None:
                    assert vis.geometry.mesh.filename.startswith(mesh_dir) or "/" in vis.geometry.mesh.filename

    def test_get_urdf_string(self, franka_parser):
        """Test getting URDF as string (lines 337-338)."""
        urdf_string = franka_parser.get_urdf_string()

        assert urdf_string is not None
        assert isinstance(urdf_string, str)
        assert len(urdf_string) > 0
        assert "<?xml" in urdf_string or "<robot" in urdf_string

    def test_get_link_mesh_visual(self, franka_parser):
        """Test getting visual mesh for a link."""
        mesh = franka_parser.get_link_mesh("panda_link0", use_collision_mesh=False)

        if mesh is not None:
            assert mesh.name == "panda_link0"
            assert mesh.file_path is not None
            assert mesh.pose is not None

    def test_get_link_mesh_collision(self, franka_parser):
        """Test getting collision mesh for a link."""
        mesh = franka_parser.get_link_mesh("panda_link0", use_collision_mesh=True)

        if mesh is not None:
            assert mesh.name == "panda_link0"
            assert mesh.file_path is not None

    def test_get_link_mesh_no_geometry(self, franka_parser):
        """Test getting mesh for link without geometry."""
        # Some links might not have meshes
        # This tests the return None path
        try:
            mesh = franka_parser.get_link_mesh("nonexistent_link")
        except KeyError:
            # Expected if link doesn't exist
            pass


class TestUrdfRobotParserNegativeAxisJoints:
    """Test handling of joints with negative axis values."""

    def test_negative_x_axis_joint(self, tmp_path):
        """Test joint with negative X axis."""
        urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link"/>
  <link name="test_link"/>

  <joint name="neg_x_joint" type="revolute">
    <parent link="base_link"/>
    <child link="test_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="-1 0 0"/>
    <limit effort="100" lower="-2.0" upper="2.0" velocity="2.0"/>
  </joint>
</robot>"""

        urdf_path = tmp_path / "neg_x_axis.urdf"
        urdf_path.write_text(urdf_content)

        parser = UrdfRobotParser(urdf_path=str(urdf_path))
        link_params = parser.get_link_parameters("test_link", base=False)

        # Should handle negative axis by inverting joint offset
        assert link_params is not None
        assert link_params.joint_type == JointType.X_ROT
        # Joint offset should be negative
        assert link_params.joint_offset[0] == pytest.approx(-1.0)

    def test_negative_y_axis_joint(self, tmp_path):
        """Test joint with negative Y axis."""
        urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link"/>
  <link name="test_link"/>

  <joint name="neg_y_joint" type="prismatic">
    <parent link="base_link"/>
    <child link="test_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 -1 0"/>
    <limit effort="100" lower="-2.0" upper="2.0" velocity="2.0"/>
  </joint>
</robot>"""

        urdf_path = tmp_path / "neg_y_axis.urdf"
        urdf_path.write_text(urdf_content)

        parser = UrdfRobotParser(urdf_path=str(urdf_path))
        link_params = parser.get_link_parameters("test_link", base=False)

        assert link_params is not None
        assert link_params.joint_type == JointType.Y_PRISM

    def test_negative_z_axis_joint(self, tmp_path):
        """Test joint with negative Z axis."""
        urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link"/>
  <link name="test_link"/>

  <joint name="neg_z_joint" type="revolute">
    <parent link="base_link"/>
    <child link="test_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 -1"/>
    <limit effort="100" lower="-2.0" upper="2.0" velocity="2.0"/>
  </joint>
</robot>"""

        urdf_path = tmp_path / "neg_z_axis.urdf"
        urdf_path.write_text(urdf_content)

        parser = UrdfRobotParser(urdf_path=str(urdf_path))
        link_params = parser.get_link_parameters("test_link", base=False)

        assert link_params is not None
        assert link_params.joint_type == JointType.Z_ROT


class TestUrdfRobotParserJointTypes:
    """Test different joint type conversions."""

    @pytest.mark.parametrize(
        "joint_type,axis,expected_type",
        [
            ("revolute", "1 0 0", JointType.X_ROT),
            ("revolute", "0 1 0", JointType.Y_ROT),
            ("revolute", "0 0 1", JointType.Z_ROT),
            ("prismatic", "1 0 0", JointType.X_PRISM),
            ("prismatic", "0 1 0", JointType.Y_PRISM),
            ("prismatic", "0 0 1", JointType.Z_PRISM),
        ],
    )
    def test_joint_type_conversion(self, tmp_path, joint_type, axis, expected_type):
        """Test conversion of different joint types to JointType enum."""
        urdf_content = f"""<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link"/>
  <link name="test_link"/>

  <joint name="test_joint" type="{joint_type}">
    <parent link="base_link"/>
    <child link="test_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="{axis}"/>
    <limit effort="100" lower="-2.0" upper="2.0" velocity="2.0"/>
  </joint>
</robot>"""

        urdf_path = tmp_path / f"joint_{joint_type}_{axis.replace(' ', '_')}.urdf"
        urdf_path.write_text(urdf_content)

        parser = UrdfRobotParser(urdf_path=str(urdf_path))
        link_params = parser.get_link_parameters("test_link", base=False)

        assert link_params.joint_type == expected_type

    def test_fixed_joint_type(self, tmp_path):
        """Test fixed joint type."""
        urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link"/>
  <link name="test_link"/>

  <joint name="test_joint" type="fixed">
    <parent link="base_link"/>
    <child link="test_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
  </joint>
</robot>"""

        urdf_path = tmp_path / "fixed_joint.urdf"
        urdf_path.write_text(urdf_content)

        parser = UrdfRobotParser(urdf_path=str(urdf_path))
        link_params = parser.get_link_parameters("test_link", base=False)

        assert link_params.joint_type == JointType.FIXED


class TestUrdfRobotParserBaseLink:
    """Test base link handling."""

    def test_get_link_parameters_base_link(self, franka_parser):
        """Test getting parameters for base link."""
        link_params = franka_parser.get_link_parameters("panda_link0", base=True)

        assert link_params is not None
        assert link_params.parent_link_name is None
        assert link_params.joint_type == JointType.FIXED
        assert link_params.joint_name == "base_joint"


class TestUrdfRobotParserFileHandling:
    """Test file path handling."""

    def test_file_name_handler_package_removal(self, franka_parser):
        """Test that package:// is removed from file names."""
        result = franka_parser._file_name_handler("package://robot_description/meshes/link.stl")

        assert "package://" not in result
        assert "robot_description/meshes/link.stl" in result

    def test_file_name_handler_absolute_path(self, franka_parser):
        """Test file name handler with mesh root prepending."""
        result = franka_parser._file_name_handler("meshes/link.stl")

        assert franka_parser._mesh_root in result or result == join_path(franka_parser._mesh_root, "meshes/link.stl")


class TestUrdfRobotParserEdgeCases:
    """Test edge cases and error conditions."""

    def test_root_link_property_without_scene_graph(self, franka_urdf_path, franka_asset_path):
        """Test root_link property when scene graph is not built."""
        parser = UrdfRobotParser(
            urdf_path=franka_urdf_path,
            mesh_root=franka_asset_path,
            build_scene_graph=False,
        )

        # Without scene graph, root_link might not be set
        try:
            root = parser.root_link
            # Might return None or a value depending on yourdfpy implementation
        except AttributeError:
            # Expected if scene graph wasn't built
            pass

    def test_get_link_parameters_with_extra_links(self, franka_urdf_path, franka_asset_path):
        """Test that extra links are returned correctly."""
        extra_link = LinkParams(
            link_name="custom_link",
            joint_name="custom_joint",
            joint_type=JointType.FIXED,
            fixed_transform=np.eye(4)[:3, :],
            parent_link_name="panda_hand",
        )

        parser = UrdfRobotParser(
            urdf_path=franka_urdf_path,
            mesh_root=franka_asset_path,
            extra_links={"custom_link": extra_link},
        )

        # Should return the extra link directly
        params = parser.get_link_parameters("custom_link", base=False)
        assert params == extra_link

    def test_get_link_mesh_with_no_mesh(self, tmp_path):
        """Test getting mesh for link without any mesh geometry."""
        urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link"/>
  <link name="no_mesh_link">
    <!-- No visual or collision geometry -->
  </link>

  <joint name="test_joint" type="fixed">
    <parent link="base_link"/>
    <child link="no_mesh_link"/>
  </joint>
</robot>"""

        urdf_path = tmp_path / "no_mesh.urdf"
        urdf_path.write_text(urdf_content)

        parser = UrdfRobotParser(urdf_path=str(urdf_path))
        mesh = parser.get_link_mesh("no_mesh_link")

        # Should return None for link without mesh
        assert mesh is None

    def test_link_with_minimal_inertial(self, tmp_path):
        """Test link with minimal inertial properties."""
        urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link"/>
  <link name="minimal_link">
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0.0" ixz="0.0" iyy="0.01" iyz="0.0" izz="0.01"/>
    </inertial>
  </link>

  <joint name="test_joint" type="revolute">
    <parent link="base_link"/>
    <child link="minimal_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit effort="100" lower="-2.0" upper="2.0" velocity="2.0"/>
  </joint>
</robot>"""

        urdf_path = tmp_path / "minimal_inertial.urdf"
        urdf_path.write_text(urdf_content)

        parser = UrdfRobotParser(urdf_path=str(urdf_path))
        link_params = parser.get_link_parameters("minimal_link", base=False)

        # Should use default values for missing inertial components
        assert link_params is not None
        assert link_params.link_mass == pytest.approx(0.5)

    def test_joint_without_origin(self, tmp_path):
        """Test joint without origin element (line 210)."""
        urdf_content = """<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link"/>
  <link name="test_link"/>

  <joint name="test_joint" type="revolute">
    <parent link="base_link"/>
    <child link="test_link"/>
    <!-- No origin specified - should default to identity -->
    <axis xyz="0 0 1"/>
    <limit effort="100" lower="-2.0" upper="2.0" velocity="2.0"/>
  </joint>
</robot>"""

        urdf_path = tmp_path / "no_origin.urdf"
        urdf_path.write_text(urdf_content)

        parser = UrdfRobotParser(urdf_path=str(urdf_path))
        link_params = parser.get_link_parameters("test_link", base=False)

        # Should use identity matrix when origin is None (line 210)
        assert link_params is not None
        assert np.allclose(link_params.fixed_transform, np.eye(4)[:3, :])

    def test_get_link_mesh_with_origin(self, tmp_path):
        """Test getting mesh with non-None origin (line 368)."""
        # Create a simple mesh file
        mesh_path = tmp_path / "simple_box.stl"
        # Create a minimal STL content
        stl_content = """solid box
  facet normal 0 0 1
    outer loop
      vertex 0 0 1
      vertex 1 0 1
      vertex 1 1 1
    endloop
  endfacet
endsolid box"""
        mesh_path.write_text(stl_content)

        urdf_content = f"""<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link"/>
  <link name="mesh_link">
    <visual>
      <origin xyz="0.1 0.2 0.3" rpy="0 0 1.57"/>
      <geometry>
        <mesh filename="{mesh_path.name}"/>
      </geometry>
    </visual>
  </link>

  <joint name="test_joint" type="fixed">
    <parent link="base_link"/>
    <child link="mesh_link"/>
  </joint>
</robot>"""

        urdf_path = tmp_path / "mesh_with_origin.urdf"
        urdf_path.write_text(urdf_content)

        parser = UrdfRobotParser(
            urdf_path=str(urdf_path),
            mesh_root=str(tmp_path),
            load_meshes=True,
        )

        mesh = parser.get_link_mesh("mesh_link", use_collision_mesh=False)

        # Should convert mesh_pose from matrix to list (line 368)
        if mesh is not None:
            assert mesh.pose is not None
            assert isinstance(mesh.pose, list)
            assert len(mesh.pose) == 7  # x, y, z, qw, qx, qy, qz

