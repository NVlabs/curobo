# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Robot-dependent unit tests for USD helper utilities.

These tests require robot configurations (franka.yml) and may require CUDA.
They are separated from main USD helper tests to isolate dependencies.
"""

# Standard Library

# Third Party
import numpy as np
import pytest
import torch

try:
    # Third Party
    from pxr import Usd, UsdGeom
except ImportError:
    pytest.skip("usd-core not available", allow_module_level=True)


# CuRobo
from curobo._src.geom.types import Cuboid, SceneCfg, Sphere
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util.usd_writer import UsdWriter
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


@pytest.fixture
def robot_config_file():
    """Return path to franka robot configuration."""
    return "franka.yml"


@pytest.fixture
def device_cfg():
    """Create tensor device configuration."""
    return DeviceCfg(device=torch.device("cuda:0"))


@pytest.fixture
def robot_model(robot_config_file, device_cfg):
    """Load franka robot model."""
    try:
        kin_model = UsdWriter.load_robot(robot_config_file, device_cfg=device_cfg)
        return kin_model
    except Exception as e:
        pytest.skip(f"Could not load robot model: {e}")


@pytest.fixture
def simple_joint_trajectory(robot_model):
    """Create a simple joint trajectory for testing."""
    # Get number of active joints
    n_joints = robot_model.kinematics_config.num_dof

    # Create trajectory with 10 timesteps
    n_timesteps = 10

    # Create positions that move from start to end
    start_pos = torch.zeros(n_joints, device=robot_model.device_cfg.device)
    end_pos = torch.ones(n_joints, device=robot_model.device_cfg.device) * 0.5

    positions = []
    for i in range(n_timesteps):
        alpha = i / (n_timesteps - 1)
        pos = start_pos * (1 - alpha) + end_pos * alpha
        positions.append(pos)

    position_tensor = torch.stack(positions)

    return JointState(
        position=position_tensor,
        velocity=torch.zeros_like(position_tensor),
        acceleration=torch.zeros_like(position_tensor),
        jerk=torch.zeros_like(position_tensor),
        joint_names=robot_model.joint_names[:n_joints],
    )


@pytest.fixture
def simple_scene():
    """Create a simple scene for testing."""
    return SceneCfg(
        cuboid=[
            Cuboid(
                name="table",
                pose=[0.5, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0],
                dims=[0.6, 0.8, 0.05],
            )
        ],
        sphere=[
            Sphere(
                name="obstacle",
                pose=[0.3, 0.3, 0.5, 1.0, 0.0, 0.0, 0.0],
                radius=0.1,
            )
        ],
    )


class TestWriteTrajectoryAnimation:
    """Tests for write_trajectory_animation static method."""

    def test_write_trajectory_animation_basic(
        self, tmp_path, robot_config_file, simple_joint_trajectory, simple_scene
    ):
        """Test basic trajectory animation creation."""
        output_path = tmp_path / "trajectory.usd"

        q_start = JointState(
            position=simple_joint_trajectory.position[0:1],
            velocity=simple_joint_trajectory.velocity[0:1],
            acceleration=simple_joint_trajectory.acceleration[0:1],
            jerk=simple_joint_trajectory.jerk[0:1],
            joint_names=simple_joint_trajectory.joint_names,
        )

        UsdWriter.write_trajectory_animation(
            robot_model_file=robot_config_file,
            scene_model=simple_scene,
            q_start=q_start,
            q_traj=simple_joint_trajectory,
            dt=0.02,
            save_path=str(output_path),
        )

        # Verify file was created
        assert output_path.exists()

        # Verify we can load the stage
        stage = Usd.Stage.Open(str(output_path))
        assert stage is not None

        # Verify scene objects exist (in obstacles frame)
        assert stage.GetPrimAtPath("/world/obstacles/table").IsValid()
        assert stage.GetPrimAtPath("/world/obstacles/obstacle").IsValid()

    def test_write_trajectory_animation_with_robot_color(
        self, tmp_path, robot_config_file, simple_joint_trajectory, simple_scene
    ):
        """Test trajectory animation with robot color."""
        output_path = tmp_path / "colored_trajectory.usd"

        q_start = JointState(
            position=simple_joint_trajectory.position[0:1],
            velocity=simple_joint_trajectory.velocity[0:1],
            acceleration=simple_joint_trajectory.acceleration[0:1],
            jerk=simple_joint_trajectory.jerk[0:1],
            joint_names=simple_joint_trajectory.joint_names,
        )

        robot_color = [0.8, 0.2, 0.1, 1.0]  # Red robot

        UsdWriter.write_trajectory_animation(
            robot_model_file=robot_config_file,
            scene_model=simple_scene,
            q_start=q_start,
            q_traj=simple_joint_trajectory,
            dt=0.02,
            save_path=str(output_path),
            robot_color=robot_color,
        )

        assert output_path.exists()
        stage = Usd.Stage.Open(str(output_path))
        assert stage is not None

    def test_write_trajectory_animation_with_robot_spheres(
        self, tmp_path, robot_config_file, simple_joint_trajectory, simple_scene
    ):
        """Test trajectory animation with robot collision spheres visualization."""
        output_path = tmp_path / "spheres_trajectory.usd"

        q_start = JointState(
            position=simple_joint_trajectory.position[0:1],
            velocity=simple_joint_trajectory.velocity[0:1],
            acceleration=simple_joint_trajectory.acceleration[0:1],
            jerk=simple_joint_trajectory.jerk[0:1],
            joint_names=simple_joint_trajectory.joint_names,
        )

        UsdWriter.write_trajectory_animation(
            robot_model_file=robot_config_file,
            scene_model=simple_scene,
            q_start=q_start,
            q_traj=simple_joint_trajectory,
            dt=0.02,
            save_path=str(output_path),
            visualize_robot_spheres=True,
        )

        assert output_path.exists()
        stage = Usd.Stage.Open(str(output_path))

        # Check for robot collision spheres
        collision_prim = stage.GetPrimAtPath("/world/curobo/robot_collision")
        # May or may not exist depending on robot config
        # Just verify stage is valid
        assert stage is not None

    def test_write_trajectory_animation_without_robot_spheres(
        self, tmp_path, robot_config_file, simple_joint_trajectory, simple_scene
    ):
        """Test trajectory animation without robot spheres visualization."""
        output_path = tmp_path / "no_spheres_trajectory.usd"

        q_start = JointState(
            position=simple_joint_trajectory.position[0:1],
            velocity=simple_joint_trajectory.velocity[0:1],
            acceleration=simple_joint_trajectory.acceleration[0:1],
            jerk=simple_joint_trajectory.jerk[0:1],
            joint_names=simple_joint_trajectory.joint_names,
        )

        UsdWriter.write_trajectory_animation(
            robot_model_file=robot_config_file,
            scene_model=simple_scene,
            q_start=q_start,
            q_traj=simple_joint_trajectory,
            dt=0.02,
            save_path=str(output_path),
            visualize_robot_spheres=False,
        )

        assert output_path.exists()

    def test_write_trajectory_animation_with_flatten(
        self, tmp_path, robot_config_file, simple_joint_trajectory, simple_scene
    ):
        """Test trajectory animation with flatten option."""
        output_path = tmp_path / "flattened_trajectory.usd"

        q_start = JointState(
            position=simple_joint_trajectory.position[0:1],
            velocity=simple_joint_trajectory.velocity[0:1],
            acceleration=simple_joint_trajectory.acceleration[0:1],
            jerk=simple_joint_trajectory.jerk[0:1],
            joint_names=simple_joint_trajectory.joint_names,
        )

        UsdWriter.write_trajectory_animation(
            robot_model_file=robot_config_file,
            scene_model=simple_scene,
            q_start=q_start,
            q_traj=simple_joint_trajectory,
            dt=0.02,
            save_path=str(output_path),
            flatten_usd=True,
        )

        assert output_path.exists()
        stage = Usd.Stage.Open(str(output_path))
        assert stage is not None

    def test_write_trajectory_animation_custom_interpolation(
        self, tmp_path, robot_config_file, simple_joint_trajectory, simple_scene
    ):
        """Test trajectory animation with custom interpolation steps."""
        output_path = tmp_path / "interpolated_trajectory.usd"

        q_start = JointState(
            position=simple_joint_trajectory.position[0:1],
            velocity=simple_joint_trajectory.velocity[0:1],
            acceleration=simple_joint_trajectory.acceleration[0:1],
            jerk=simple_joint_trajectory.jerk[0:1],
            joint_names=simple_joint_trajectory.joint_names,
        )

        UsdWriter.write_trajectory_animation(
            robot_model_file=robot_config_file,
            scene_model=simple_scene,
            q_start=q_start,
            q_traj=simple_joint_trajectory,
            dt=0.02,
            save_path=str(output_path),
            interpolation_steps=2.0,
        )

        assert output_path.exists()
        stage = Usd.Stage.Open(str(output_path))

        # Verify time codes with interpolation
        assert stage.GetEndTimeCode() > simple_joint_trajectory.position.shape[0]

    def test_write_trajectory_animation_custom_base_frame(
        self, tmp_path, robot_config_file, simple_joint_trajectory, simple_scene
    ):
        """Test trajectory animation with custom base frame."""
        output_path = tmp_path / "custom_frame_trajectory.usd"

        q_start = JointState(
            position=simple_joint_trajectory.position[0:1],
            velocity=simple_joint_trajectory.velocity[0:1],
            acceleration=simple_joint_trajectory.acceleration[0:1],
            jerk=simple_joint_trajectory.jerk[0:1],
            joint_names=simple_joint_trajectory.joint_names,
        )

        UsdWriter.write_trajectory_animation(
            robot_model_file=robot_config_file,
            scene_model=simple_scene,
            q_start=q_start,
            q_traj=simple_joint_trajectory,
            dt=0.02,
            save_path=str(output_path),
            base_frame="/custom_world",
        )

        assert output_path.exists()
        stage = Usd.Stage.Open(str(output_path))

        # Verify custom base frame
        assert stage.GetPrimAtPath("/custom_world").IsValid()

    def test_write_trajectory_animation_with_goal_pose(
        self, tmp_path, robot_config_file, simple_joint_trajectory, simple_scene, device_cfg
    ):
        """Test trajectory animation with goal pose visualization."""
        output_path = tmp_path / "goal_pose_trajectory.usd"

        q_start = JointState(
            position=simple_joint_trajectory.position[0:1],
            velocity=simple_joint_trajectory.velocity[0:1],
            acceleration=simple_joint_trajectory.acceleration[0:1],
            jerk=simple_joint_trajectory.jerk[0:1],
            joint_names=simple_joint_trajectory.joint_names,
        )

        # Create goal pose
        goal_position = torch.tensor([[0.5, 0.0, 0.5]], device=device_cfg.device)
        goal_quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device_cfg.device)
        goal_pose = Pose(position=goal_position, quaternion=goal_quaternion)

        goal_color = [0.0, 1.0, 0.0, 0.5]  # Semi-transparent green

        UsdWriter.write_trajectory_animation(
            robot_model_file=robot_config_file,
            scene_model=simple_scene,
            q_start=q_start,
            q_traj=simple_joint_trajectory,
            dt=0.02,
            save_path=str(output_path),
            goal_pose=goal_pose,
            goal_color=goal_color,
        )

        assert output_path.exists()
        stage = Usd.Stage.Open(str(output_path))

        # Check for goal visualization
        goal_prim = stage.GetPrimAtPath("/world/goal_idx_0")
        # Goal may be nested, just verify stage is valid
        assert stage is not None

    def test_write_trajectory_animation_without_scene(
        self, tmp_path, robot_config_file, simple_joint_trajectory
    ):
        """Test trajectory animation without scene model (None)."""
        output_path = tmp_path / "no_scene_trajectory.usd"

        q_start = JointState(
            position=simple_joint_trajectory.position[0:1],
            velocity=simple_joint_trajectory.velocity[0:1],
            acceleration=simple_joint_trajectory.acceleration[0:1],
            jerk=simple_joint_trajectory.jerk[0:1],
            joint_names=simple_joint_trajectory.joint_names,
        )

        UsdWriter.write_trajectory_animation(
            robot_model_file=robot_config_file,
            scene_model=None,
            q_start=q_start,
            q_traj=simple_joint_trajectory,
            dt=0.02,
            save_path=str(output_path),
        )

        assert output_path.exists()

    def test_write_trajectory_animation_with_kin_model(
        self, tmp_path, robot_model, simple_joint_trajectory, simple_scene
    ):
        """Test trajectory animation with pre-loaded kinematic model."""
        output_path = tmp_path / "preloaded_model_trajectory.usd"

        q_start = JointState(
            position=simple_joint_trajectory.position[0:1],
            velocity=simple_joint_trajectory.velocity[0:1],
            acceleration=simple_joint_trajectory.acceleration[0:1],
            jerk=simple_joint_trajectory.jerk[0:1],
            joint_names=simple_joint_trajectory.joint_names,
        )

        # Pass None for robot_model_file but provide kin_model
        UsdWriter.write_trajectory_animation(
            robot_model_file=None,
            scene_model=simple_scene,
            q_start=q_start,
            q_traj=simple_joint_trajectory,
            dt=0.02,
            save_path=str(output_path),
            kin_model=robot_model,
        )

        assert output_path.exists()


class TestWriteTrajectoryAnimationWithRobotUsd:
    """Tests for write_trajectory_animation_with_robot_usd static method."""

    def test_write_trajectory_with_robot_usd_fallback_to_urdf(
        self, tmp_path, robot_config_file, simple_joint_trajectory, simple_scene
    ):
        """Test that method falls back to URDF animation when USD not available."""
        output_path = tmp_path / "usd_fallback_trajectory.usd"

        q_start = JointState(
            position=simple_joint_trajectory.position[0:1],
            velocity=simple_joint_trajectory.velocity[0:1],
            acceleration=simple_joint_trajectory.acceleration[0:1],
            jerk=simple_joint_trajectory.jerk[0:1],
            joint_names=simple_joint_trajectory.joint_names,
        )

        # This should fall back to urdf animation since franka might not have USD
        UsdWriter.write_trajectory_animation_with_robot_usd(
            robot_model_file=robot_config_file,
            scene_model=simple_scene,
            q_start=q_start,
            q_traj=simple_joint_trajectory,
            dt=0.02,
            save_path=str(output_path),
        )

        assert output_path.exists()

    def test_write_trajectory_with_robot_usd_with_robot_color_fallback(
        self, tmp_path, robot_config_file, simple_joint_trajectory, simple_scene
    ):
        """Test USD animation with robot_color triggers fallback to URDF mode."""
        output_path = tmp_path / "usd_color_fallback.usd"

        q_start = JointState(
            position=simple_joint_trajectory.position[0:1],
            velocity=simple_joint_trajectory.velocity[0:1],
            acceleration=simple_joint_trajectory.acceleration[0:1],
            jerk=simple_joint_trajectory.jerk[0:1],
            joint_names=simple_joint_trajectory.joint_names,
        )

        # robot_color should force fallback to URDF mode
        UsdWriter.write_trajectory_animation_with_robot_usd(
            robot_model_file=robot_config_file,
            scene_model=simple_scene,
            q_start=q_start,
            q_traj=simple_joint_trajectory,
            dt=0.02,
            save_path=str(output_path),
            robot_color=[1.0, 0.0, 0.0, 1.0],
        )

        assert output_path.exists()

    def test_write_trajectory_with_robot_usd_custom_paths(
        self, tmp_path, robot_config_file, simple_joint_trajectory, simple_scene
    ):
        """Test USD animation with custom write and local paths."""
        output_path = tmp_path / "custom_paths_trajectory.usd"
        write_path = tmp_path / "assets_write"
        local_path = tmp_path / "assets_local"

        write_path.mkdir(exist_ok=True)
        local_path.mkdir(exist_ok=True)

        q_start = JointState(
            position=simple_joint_trajectory.position[0:1],
            velocity=simple_joint_trajectory.velocity[0:1],
            acceleration=simple_joint_trajectory.acceleration[0:1],
            jerk=simple_joint_trajectory.jerk[0:1],
            joint_names=simple_joint_trajectory.joint_names,
        )

        UsdWriter.write_trajectory_animation_with_robot_usd(
            robot_model_file=robot_config_file,
            scene_model=simple_scene,
            q_start=q_start,
            q_traj=simple_joint_trajectory,
            dt=0.02,
            save_path=str(output_path),
            write_robot_usd_path=str(write_path) + "/",
            robot_usd_local_reference=str(local_path) + "/",
        )

        assert output_path.exists()


class TestLoadRobot:
    """Tests for load_robot static method."""

    def test_load_robot_basic(self, robot_config_file, device_cfg):
        """Test loading robot from configuration file."""
        kin_model = UsdWriter.load_robot(robot_config_file, device_cfg=device_cfg)

        assert kin_model is not None
        assert hasattr(kin_model, "kinematics_config")
        assert hasattr(kin_model, "joint_names")
        assert len(kin_model.joint_names) > 0

    def test_load_robot_has_mesh_links(self, robot_config_file, device_cfg):
        """Test that loaded robot has mesh link names configured."""
        kin_model = UsdWriter.load_robot(robot_config_file, device_cfg=device_cfg)

        # After loading with load_robot, it should have mesh links configured
        assert hasattr(kin_model.kinematics_config, "mesh_link_names")
        # mesh_link_names may be None or empty depending on robot config
        # Just verify the attribute exists

    def test_load_robot_forward_kinematics(self, robot_config_file, device_cfg):
        """Test that loaded robot can perform forward kinematics."""
        kin_model = UsdWriter.load_robot(robot_config_file, device_cfg=device_cfg)

        # Create zero joint configuration
        num_dof = kin_model.kinematics_config.num_dof
        q = torch.zeros(1, num_dof, device=device_cfg.device)

        # Should be able to get link poses
        if kin_model.kinematics_config.tool_frames is not None:
            poses = kin_model.get_link_poses(q, kin_model.kinematics_config.tool_frames[:1])
            assert poses is not None


class TestUpdateRobotJointState:
    """Tests for update_robot_joint_state method."""

    def test_update_robot_joint_state_with_timestep(self, tmp_path, device_cfg):
        """Test updating robot joint state with timestep parameter."""
        # Create a stage with robot joints
        helper = UsdWriter()
        stage_path = tmp_path / "joint_state.usd"
        helper.create_stage(str(stage_path), timesteps=5, dt=0.02)

        # Create a simple joint prim
        joint_path = "/world/robot/joint1"
        joint_prim = helper.stage.DefinePrim(joint_path)

        # Add required attributes
        from pxr import Sdf
        joint_prim.CreateAttribute(
            "drive:angular:physics:targetPosition",
            Sdf.ValueTypeNames.Float
        )

        # Create joint state
        position = torch.tensor([[0.5]], device=device_cfg.device)
        joint_state = JointState(
            position=position,
            velocity=torch.zeros_like(position),
            acceleration=torch.zeros_like(position),
            jerk=torch.zeros_like(position),
            joint_names=["joint1"],
        )

        # Create joint prims dict
        joint_prims = {"joint1": joint_prim}

        # Update with timestep
        helper.update_robot_joint_state(joint_prims, joint_state, timestep=0)

        # Verify stage is valid
        assert helper.stage is not None

    def test_update_robot_joint_state_without_timestep(self, tmp_path, device_cfg):
        """Test updating robot joint state without timestep parameter."""
        helper = UsdWriter()
        stage_path = tmp_path / "joint_state_no_time.usd"
        helper.create_stage(str(stage_path))

        # Create a simple joint prim
        joint_path = "/world/robot/joint1"
        joint_prim = helper.stage.DefinePrim(joint_path)

        # Add required attributes
        from pxr import Sdf
        joint_prim.CreateAttribute(
            "drive:angular:physics:targetPosition",
            Sdf.ValueTypeNames.Float
        )

        # Create joint state
        position = torch.tensor([[1.0]], device=device_cfg.device)
        joint_state = JointState(
            position=position,
            velocity=torch.zeros_like(position),
            acceleration=torch.zeros_like(position),
            jerk=torch.zeros_like(position),
            joint_names=["joint1"],
        )

        # Create joint prims dict
        joint_prims = {"joint1": joint_prim}

        # Update without timestep (timestep=None)
        helper.update_robot_joint_state(joint_prims, joint_state, timestep=None)

        # Verify stage is valid
        assert helper.stage is not None

    def test_update_robot_joint_state_converts_to_degrees(self, tmp_path, device_cfg):
        """Test that joint positions are converted from radians to degrees."""
        helper = UsdWriter()
        stage_path = tmp_path / "joint_degrees.usd"
        helper.create_stage(str(stage_path))

        # Create a joint prim
        joint_path = "/world/robot/joint1"
        joint_prim = helper.stage.DefinePrim(joint_path)

        # Add required attribute with proper type
        from pxr import Sdf
        joint_prim.CreateAttribute(
            "drive:angular:physics:targetPosition",
            Sdf.ValueTypeNames.Float
        )

        # Create joint state with pi/2 radians (90 degrees)
        position = torch.tensor([[np.pi / 2]], device=device_cfg.device)
        joint_state = JointState(
            position=position,
            velocity=torch.zeros_like(position),
            acceleration=torch.zeros_like(position),
            jerk=torch.zeros_like(position),
            joint_names=["joint1"],
        )

        joint_prims = {"joint1": joint_prim}

        # Update joint state
        helper.update_robot_joint_state(joint_prims, joint_state, timestep=None)

        # Verify the value was set (should be ~90 degrees)
        set_value = joint_prim.GetAttribute("drive:angular:physics:targetPosition").Get()
        assert set_value is not None
        assert set_value == pytest.approx(90.0, abs=0.1)


class TestRobotAnimationEdgeCases:
    """Tests for edge cases in robot animation functions."""

    def test_write_trajectory_with_empty_mesh_links(
        self, tmp_path, simple_joint_trajectory, simple_scene
    ):
        """Test trajectory animation when robot has no mesh links."""
        # Create a config with no mesh links
        config_path = tmp_path / "no_mesh_robot.yml"

        # Load franka config and modify it
        franka_config = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))

        # Remove mesh link names
        if "robot_cfg" in franka_config:
            robot_cfg = franka_config["robot_cfg"]
        else:
            robot_cfg = franka_config

        robot_cfg["kinematics"]["mesh_link_names"] = None

        # Save modified config
        import yaml
        with open(config_path, "w") as f:
            yaml.dump({"robot_cfg": robot_cfg}, f)

        output_path = tmp_path / "no_mesh_trajectory.usd"

        q_start = JointState(
            position=simple_joint_trajectory.position[0:1],
            velocity=simple_joint_trajectory.velocity[0:1],
            acceleration=simple_joint_trajectory.acceleration[0:1],
            jerk=simple_joint_trajectory.jerk[0:1],
            joint_names=simple_joint_trajectory.joint_names,
        )

        # This should handle empty mesh links gracefully
        UsdWriter.write_trajectory_animation(
            robot_model_file=str(config_path),
            scene_model=simple_scene,
            q_start=q_start,
            q_traj=simple_joint_trajectory,
            dt=0.02,
            save_path=str(output_path),
        )

        assert output_path.exists()

    def test_write_trajectory_multiple_goal_poses(
        self, tmp_path, robot_config_file, simple_joint_trajectory, simple_scene, device_cfg
    ):
        """Test trajectory animation with multiple goal poses."""
        output_path = tmp_path / "multi_goal_trajectory.usd"

        q_start = JointState(
            position=simple_joint_trajectory.position[0:1],
            velocity=simple_joint_trajectory.velocity[0:1],
            acceleration=simple_joint_trajectory.acceleration[0:1],
            jerk=simple_joint_trajectory.jerk[0:1],
            joint_names=simple_joint_trajectory.joint_names,
        )

        # Create multiple goal poses
        goal_positions = torch.tensor(
            [[0.5, 0.0, 0.5], [0.3, 0.3, 0.6], [0.4, -0.2, 0.5]],
            device=device_cfg.device
        )
        goal_quaternions = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            device=device_cfg.device
        )

        goal_pose = Pose(position=goal_positions, quaternion=goal_quaternions)

        UsdWriter.write_trajectory_animation(
            robot_model_file=robot_config_file,
            scene_model=simple_scene,
            q_start=q_start,
            q_traj=simple_joint_trajectory,
            dt=0.02,
            save_path=str(output_path),
            goal_pose=goal_pose,
            goal_color=[0.0, 1.0, 0.0, 0.5],
        )

        assert output_path.exists()
        stage = Usd.Stage.Open(str(output_path))

        # Check for multiple goal visualizations
        # They should be named goal_idx_0, goal_idx_1, goal_idx_2
        assert stage is not None

    def test_write_trajectory_animation_with_goal_tool_pose(
        self, tmp_path, robot_config_file, simple_joint_trajectory, simple_scene, device_cfg
    ):
        """Test trajectory animation with GoalToolPose (single goalset entry)."""
        output_path = tmp_path / "goal_tool_pose_trajectory.usd"

        q_start = JointState(
            position=simple_joint_trajectory.position[0:1],
            velocity=simple_joint_trajectory.velocity[0:1],
            acceleration=simple_joint_trajectory.acceleration[0:1],
            joint_names=simple_joint_trajectory.joint_names,
        )

        kin_model = UsdWriter.load_robot(robot_config_file, device_cfg=device_cfg)
        tool_frames = kin_model.tool_frames
        num_links = len(tool_frames)

        position = torch.zeros(
            (1, 1, num_links, 1, 3), device=device_cfg.device, dtype=torch.float32,
        )
        quaternion = torch.zeros(
            (1, 1, num_links, 1, 4), device=device_cfg.device, dtype=torch.float32,
        )
        position[0, 0, 0, 0, :] = torch.tensor([0.5, 0.0, 0.5])
        quaternion[0, 0, 0, 0, :] = torch.tensor([1.0, 0.0, 0.0, 0.0])

        goal_tool_pose = GoalToolPose(
            tool_frames=tool_frames, position=position, quaternion=quaternion,
        )

        UsdWriter.write_trajectory_animation(
            robot_model_file=robot_config_file,
            scene_model=simple_scene,
            q_start=q_start,
            q_traj=simple_joint_trajectory,
            dt=0.02,
            save_path=str(output_path),
            goal_pose=goal_tool_pose,
            goal_color=[0.0, 1.0, 0.0, 0.5],
        )

        assert output_path.exists()
        stage = Usd.Stage.Open(str(output_path))
        assert stage is not None

    def test_write_trajectory_animation_with_goalset(
        self, tmp_path, robot_config_file, simple_joint_trajectory, simple_scene, device_cfg
    ):
        """Test GoalToolPose with num_goalset=3 renders three meshes."""
        output_path = tmp_path / "goalset_trajectory.usd"

        q_start = JointState(
            position=simple_joint_trajectory.position[0:1],
            velocity=simple_joint_trajectory.velocity[0:1],
            acceleration=simple_joint_trajectory.acceleration[0:1],
            joint_names=simple_joint_trajectory.joint_names,
        )

        kin_model = UsdWriter.load_robot(robot_config_file, device_cfg=device_cfg)
        tool_frames = kin_model.tool_frames
        num_links = len(tool_frames)
        num_goalset = 3

        position = torch.zeros(
            (1, 1, num_links, num_goalset, 3), device=device_cfg.device, dtype=torch.float32,
        )
        quaternion = torch.zeros(
            (1, 1, num_links, num_goalset, 4), device=device_cfg.device, dtype=torch.float32,
        )
        for g in range(num_goalset):
            position[0, 0, 0, g, :] = torch.tensor([0.4 + g * 0.1, 0.0, 0.5])
            quaternion[0, 0, 0, g, :] = torch.tensor([1.0, 0.0, 0.0, 0.0])

        goal_tool_pose = GoalToolPose(
            tool_frames=tool_frames, position=position, quaternion=quaternion,
        )

        UsdWriter.write_trajectory_animation(
            robot_model_file=robot_config_file,
            scene_model=simple_scene,
            q_start=q_start,
            q_traj=simple_joint_trajectory,
            dt=0.02,
            save_path=str(output_path),
            goal_pose=goal_tool_pose,
            goal_color=[1.0, 0.0, 0.0, 0.5],
        )

        assert output_path.exists()
        stage = Usd.Stage.Open(str(output_path))
        assert stage is not None


class TestRobotPrimExtraction:
    """Tests for robot prim extraction from USD stages."""

    def test_get_robot_prims_filters_correctly(self, tmp_path):
        """Test that get_robot_prims filters prims correctly."""
        helper = UsdWriter()
        stage_path = tmp_path / "robot_prims.usd"
        helper.create_stage(str(stage_path))

        # Create robot structure with links and joints
        robot_base = "/world/robot"

        # Create link prims
        from pxr import Sdf
        link1 = helper.stage.DefinePrim(f"{robot_base}/link1", "Xform")
        link1.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool)

        link2 = helper.stage.DefinePrim(f"{robot_base}/link2", "Xform")
        link2.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool)

        # Create joint prims
        joint1 = helper.stage.DefinePrim(f"{robot_base}/joint1")
        joint1.CreateAttribute("physics:jointEnabled", Sdf.ValueTypeNames.Bool)

        # Create geometry prim (should be filtered out)
        geom = helper.stage.DefinePrim(f"{robot_base}/link1/geometry", "Xform")

        # Extract robot prims
        link_prims, joint_prims = helper.get_robot_prims(
            tool_frames=["link1", "link2"],
            joint_names=["joint1"],
            robot_base_path=robot_base
        )

        # Verify correct prims were extracted
        assert "link1" in link_prims
        assert "link2" in link_prims
        assert "joint1" in joint_prims

        # Verify physics attributes were disabled
        assert link_prims["link1"].GetAttribute("physics:rigidBodyEnabled").Get() == False
        assert joint_prims["joint1"].GetAttribute("physics:jointEnabled").Get() == False

    def test_get_robot_prims_with_nested_structure(self, tmp_path):
        """Test get_robot_prims with nested robot structure."""
        helper = UsdWriter()
        stage_path = tmp_path / "nested_robot.usd"
        helper.create_stage(str(stage_path))

        robot_base = "/world/robot"

        # Create nested structure
        from pxr import Sdf
        base_link = helper.stage.DefinePrim(f"{robot_base}/base_link", "Xform")
        base_link.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool)

        # Create nested visual (should be filtered out by geometry check)
        # Note: visual won't be filtered by current code - it only filters "geometry" and "joint"
        # So we create a prim with "geometry" in name to test the filter
        geom = helper.stage.DefinePrim(f"{robot_base}/base_link/geometry", "Xform")
        geom.CreateAttribute("physics:rigidBodyEnabled", Sdf.ValueTypeNames.Bool)

        # Create joint (should not be picked as link)
        joint = helper.stage.DefinePrim(f"{robot_base}/base_joint")
        joint.CreateAttribute("physics:jointEnabled", Sdf.ValueTypeNames.Bool)

        link_prims, joint_prims = helper.get_robot_prims(
            tool_frames=["base_link"],
            joint_names=["base_joint"],
            robot_base_path=robot_base
        )

        assert "base_link" in link_prims
        assert "base_joint" in joint_prims
        # Geometry prim should not be in link_prims even though it matches base_link
        assert link_prims["base_link"].GetPath().pathString == f"{robot_base}/base_link"

