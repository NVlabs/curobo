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
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.cuda_robot_model.urdf_kinematics_parser import UrdfKinematicsParser
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.util_file import (
    file_exists,
    get_assets_path,
    get_robot_configs_path,
    join_path,
    load_yaml,
)

try:
    # CuRobo
    from curobo.cuda_robot_model.usd_kinematics_parser import UsdKinematicsParser
except ImportError:
    pytest.skip("usd-core is not available, skipping USD tests", allow_module_level=True)


def check_usd_file_exists():
    robot_file = "franka.yml"
    robot_params = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    robot_usd = join_path(get_assets_path(), robot_params["kinematics"]["usd_path"])
    return file_exists(robot_usd)


if not check_usd_file_exists():
    pytest.skip("Franka Panda USD is not available, skipping USD tests", allow_module_level=True)


@pytest.fixture(scope="module")
def robot_params_all():
    robot_file = "franka.yml"
    robot_params = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    robot_params["kinematics"]["ee_link"] = "panda_link7"
    robot_params["kinematics"]["base_link"] = "panda_link0"
    robot_params["kinematics"]["collision_link_names"] = []
    robot_params["kinematics"]["lock_joints"] = None
    robot_params["kinematics"]["extra_links"] = None

    return robot_params


@pytest.fixture(scope="module")
def usd_parser(robot_params_all):
    robot_params = robot_params_all["kinematics"]
    robot_usd = join_path(get_assets_path(), robot_params["usd_path"])
    kinematics_parser = UsdKinematicsParser(
        usd_path=robot_usd,
        flip_joints=robot_params["usd_flip_joints"],
        usd_robot_root=robot_params["usd_robot_root"],
    )
    return kinematics_parser


@pytest.fixture(scope="module")
def urdf_parser(robot_params_all):
    robot_params = robot_params_all["kinematics"]
    robot_urdf = join_path(get_assets_path(), robot_params["urdf_path"])
    kinematics_parser = UrdfKinematicsParser(robot_urdf, build_scene_graph=True)
    return kinematics_parser


@pytest.fixture(scope="module")
def retract_state(robot_params_all):
    tensor_args = TensorDeviceType()
    q = tensor_args.to_device(robot_params_all["kinematics"]["cspace"]["retract_config"]).view(
        1, -1
    )
    return q


@pytest.fixture(scope="module")
def usd_cuda_robot(robot_params_all):
    robot_params = robot_params_all["kinematics"]
    robot_params["use_usd_kinematics"] = True

    usd_cuda_config = CudaRobotModelConfig.from_data_dict(robot_params)

    usd_cuda_robot = CudaRobotModel(usd_cuda_config)
    return usd_cuda_robot


@pytest.fixture(scope="module")
def urdf_cuda_robot(robot_params_all):
    robot_params = robot_params_all["kinematics"]
    robot_params["use_usd_kinematics"] = False
    urdf_cuda_config = CudaRobotModelConfig.from_data_dict(robot_params)

    urdf_cuda_robot = CudaRobotModel(urdf_cuda_config)
    return urdf_cuda_robot


def test_chain_parse(urdf_parser, usd_parser, robot_params_all):
    robot_params = robot_params_all

    urdf_chain = urdf_parser.get_chain(
        robot_params["kinematics"]["base_link"], robot_params["kinematics"]["ee_link"]
    )

    usd_chain = usd_parser.get_chain(
        robot_params["kinematics"]["base_link"], robot_params["kinematics"]["ee_link"]
    )
    assert usd_chain == urdf_chain


def test_joint_transform_parse(usd_cuda_robot, urdf_cuda_robot):
    usd_pose = usd_cuda_robot.get_all_link_transforms()
    urdf_pose = urdf_cuda_robot.get_all_link_transforms()
    p, q = usd_pose.distance(urdf_pose)
    p = torch.linalg.norm(q)
    q = torch.linalg.norm(q)
    assert p < 1e-5 and q < 1e-5


def test_basic_ee_pose(usd_cuda_robot, urdf_cuda_robot, retract_state):
    q = retract_state[:, : usd_cuda_robot.get_dof()]

    usd_state = usd_cuda_robot.get_state(q)
    usd_pose = Pose(usd_state.ee_position, usd_state.ee_quaternion)

    urdf_state = urdf_cuda_robot.get_state(q)
    urdf_pose = Pose(urdf_state.ee_position, urdf_state.ee_quaternion)
    p_d, q_d = usd_pose.distance(urdf_pose)
    assert p_d < 1e-5
    assert q_d < 1e-5


def test_urdf_parser(urdf_parser):
    assert urdf_parser.root_link == "base_link"
    assert len(urdf_parser.get_controlled_joint_names()) == 9
