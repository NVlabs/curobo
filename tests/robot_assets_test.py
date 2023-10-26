#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

# Third Party
import pytest
import torch

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig


@pytest.mark.parametrize(
    "robot_file",
    [
        "kinova_gen3.yml",
        "iiwa.yml",
        "iiwa_allegro.yml",
        "franka.yml",
        "ur10e.yml",
    ],
)
class TestRobots:
    def test_robot_config(self, robot_file):
        tensor_args = TensorDeviceType()
        robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
        robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
        pass

    def test_kinematics(self, robot_file):
        tensor_args = TensorDeviceType()

        robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
        robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
        robot_model = CudaRobotModel(robot_cfg.kinematics)
        robot_cfg.cspace.inplace_reindex(robot_model.joint_names)
        robot_model.get_state(robot_cfg.cspace.retract_config)
        pass

    def test_ik(self, robot_file):
        world_file = "collision_table.yml"
        tensor_args = TensorDeviceType()

        robot_cfg = RobotConfig.from_dict(
            load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
        )
        world_cfg = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), world_file))
        )
        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=50,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=tensor_args,
        )
        ik_solver = IKSolver(ik_config)
        b_size = 10
        q_sample = ik_solver.sample_configs(b_size)
        kin_state = ik_solver.fk(q_sample)
        goal = Pose(kin_state.ee_position, kin_state.ee_quaternion)
        result = ik_solver.solve(goal)
        result = ik_solver.solve(goal)

        success = result.success
        assert torch.count_nonzero(success).item() >= 9.0  # we check if atleast 90% are successful

    def notest_motion_gen(self, robot_file):
        """This test causes pytest to crash when running on many robot configurations

        Args:
            robot_file: _description_
        """
        tensor_args = TensorDeviceType()

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_file,
            "collision_table.yml",
            tensor_args,
            trajopt_tsteps=40,
            use_cuda_graph=True,
        )
        motion_gen = MotionGen(motion_gen_config)
        robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
        robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)

        retract_cfg = motion_gen.get_retract_config()
        state = motion_gen.rollout_fn.compute_kinematics(
            JointState.from_position(retract_cfg.view(1, -1))
        )

        retract_pose = Pose(state.ee_pos_seq.squeeze(), quaternion=state.ee_quat_seq.squeeze())
        start_state = JointState.from_position(retract_cfg.view(1, -1))
        start_state.position[0, -1] += 0.2
        result = motion_gen.plan(start_state, retract_pose, enable_graph=False)
        assert result.success.item()
