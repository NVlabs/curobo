# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

# Third Party
import pytest
import torch

from curobo._src.geom.quaternion import quat_multiply
from curobo._src.geom.transform import matrix_to_quaternion, quaternion_to_matrix
from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsCfg

# CuRobo
from curobo._src.robot.loader.kinematics_loader import KinematicsLoaderCfg
from curobo._src.robot.types.cspace_params import CSpaceParams
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


@pytest.fixture(scope="module")
def cfg():
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    robot_data["robot_cfg"]["kinematics"]["extra_collision_spheres"] = {"attached_object": 100}
    cfg = KinematicsCfg.from_robot_yaml_file(robot_data, ["panda_hand"])
    return cfg


def test_quaternion():
    device_cfg = DeviceCfg()

    def test_q(in_quat):
        out_rot = quaternion_to_matrix(in_quat)
        out_quat = matrix_to_quaternion(out_rot.clone())
        out_quat[..., 1:] *= -1.0
        q_res = quat_multiply(in_quat, out_quat, in_quat.clone())
        q_res[..., 0] = 0.0
        assert torch.sum(q_res).item() <= 1e-5

    in_quat = device_cfg.to_device([1.0, 0.0, 0.0, 0.0]).view(1, 4)
    test_q(in_quat)
    test_q(-1.0 * in_quat)
    in_quat = device_cfg.to_device([0.0, 1.0, 0.0, 0.0]).view(1, 4)
    test_q(in_quat)
    test_q(-1.0 * in_quat)
    in_quat = device_cfg.to_device([0.0, 0.0, 1.0, 0.0]).view(1, 4)
    test_q(in_quat)
    test_q(-1.0 * in_quat)
    in_quat = device_cfg.to_device([0.0, 0.0, 0.0, 1.0]).view(1, 4)
    test_q(in_quat)
    test_q(-1.0 * in_quat)
    in_quat = device_cfg.to_device([0.7071068, 0.0, 0.0, 0.7071068]).view(1, 4)
    test_q(in_quat)
    test_q(-1.0 * in_quat)


def test_franka_kinematics(cfg):
    device_cfg = DeviceCfg()

    robot_model = Kinematics(cfg)
    tool_frames = robot_model.tool_frames
    q_test = torch.as_tensor(
        [0.0, -1.2, 0.0, -2.0, 0.0, 1.0, 0.0], **(device_cfg.as_torch_dict())
    ).view(1, -1)
    ee_position = torch.as_tensor(
        [6.0860e-02, -4.7547e-12, 7.6373e-01], **(device_cfg.as_torch_dict())
    ).view(1, -1)
    ee_quat = torch.as_tensor(
        [0.0382, 0.9193, 0.3808, 0.0922], **(device_cfg.as_torch_dict())
    ).view(1, -1)
    b_list = [1, 10, 100, 5000][:1]
    for b in b_list:
        state = robot_model.compute_kinematics(
            JointState.from_position(
                q_test.repeat(b, 1).clone(), joint_names=robot_model.joint_names
            )
        )
        pos_err = torch.linalg.norm(state.tool_poses[tool_frames[0]].position - ee_position)
        q_err = torch.linalg.norm(state.tool_poses[tool_frames[0]].quaternion - ee_quat)
        # check if all values are equal to position and quaternion
        assert pos_err < 1e-3
        assert q_err < 1e-1


def test_franka_attached_object_kinematics(cfg):
    device_cfg = DeviceCfg()

    robot_model = Kinematics(cfg)
    q_test = torch.as_tensor(
        [0.0, -1.2, 0.0, -2.0, 0.0, 1.0, 0.0], **(device_cfg.as_torch_dict())
    ).view(1, -1)
    sph_idx = robot_model.kinematics_config.get_sphere_index_from_link_name("attached_object")
    state = robot_model.compute_kinematics(
        JointState.from_position(q_test.clone(), joint_names=robot_model.joint_names)
    )

    attached_spheres = state.robot_spheres[:, 0, sph_idx, :]
    # check if all values are equal to position and quaternion

    assert torch.norm(attached_spheres[:, :, 0] - 0.061) < 0.01
    assert torch.norm(attached_spheres[:, :, 1]) < 0.0001

    # attach an object:
    new_object = torch.zeros((2, 4), **(device_cfg.as_torch_dict()))
    new_object[:, 3] = 0.01
    new_object[:, 1] = 0.1
    robot_model.kinematics_config.update_link_spheres("attached_object", new_object)
    state = robot_model.compute_kinematics(
        JointState.from_position(q_test.clone(), joint_names=robot_model.joint_names)
    )
    attached_spheres = state.robot_spheres[:, 0, sph_idx, :]
    robot_model.kinematics_config.reset_link_spheres("attached_object")
    # assert torch.norm(attached_spheres[:, :2, 0] - 0.0829) < 0.01
    # assert torch.norm(attached_spheres[:, 2:4, 0] - 0.0829) < 0.01


def test_franka_attach_object_kinematics(cfg):
    device_cfg = DeviceCfg()

    robot_model = Kinematics(cfg)
    q_test = torch.as_tensor(
        [0.0, -1.2, 0.0, -2.0, 0.0, 1.0, 0.0], **(device_cfg.as_torch_dict())
    ).view(1, -1)
    sph_idx = robot_model.kinematics_config.get_sphere_index_from_link_name("attached_object")
    state = robot_model.compute_kinematics(
        JointState.from_position(q_test.clone(), joint_names=robot_model.joint_names)
    )

    attached_spheres = state.robot_spheres[:, 0, sph_idx, :]

    assert torch.norm(attached_spheres[:, :, 0] - attached_spheres[0, 0, 0]) < 0.1
    assert torch.norm(attached_spheres[:, :, 1]) < 0.0001

    # attach an object:
    new_object = torch.zeros((100, 4), **(device_cfg.as_torch_dict()))
    new_object[:2, 3] = 0.01
    new_object[:2, 1] = 0.1

    robot_model.kinematics_config.update_link_spheres("attached_object", new_object)
    state = robot_model.compute_kinematics(
        JointState.from_position(q_test.clone(), joint_names=robot_model.joint_names)
    )
    attached_spheres = state.robot_spheres[:, 0, sph_idx, :]
    assert (torch.norm(attached_spheres[:, :2, 0] - 0.13)) < 0.01
    assert torch.norm(attached_spheres[:, 2:, 0] - 0.061) < 0.01

    robot_model.kinematics_config.reset_link_spheres("attached_object")
    state = robot_model.compute_kinematics(
        JointState.from_position(q_test.clone(), joint_names=robot_model.joint_names)
    )
    attached_spheres = state.robot_spheres[:, 0, sph_idx, :]

    assert torch.norm(attached_spheres[:, :, 0] - 0.061) < 0.01
    assert torch.norm(attached_spheres[:, :, 1]) < 0.0001


def test_locked_joints_kinematics():
    device_cfg = DeviceCfg()

    config_file = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))

    cfg = KinematicsCfg.from_config(
        KinematicsLoaderCfg(**config_file["robot_cfg"]["kinematics"], device_cfg=device_cfg)
    )
    robot_model = Kinematics(cfg)
    tool_frames = robot_model.tool_frames
    cspace = CSpaceParams(**config_file["robot_cfg"]["kinematics"]["cspace"])
    cspace.inplace_reindex(robot_model.joint_names)
    q = cspace.default_joint_position.unsqueeze(0)

    out = robot_model.compute_kinematics(
        JointState.from_position(q, joint_names=robot_model.joint_names)
    )
    j_idx = 2
    lock_joint_name = cspace.joint_names[j_idx]
    # lock a joint:
    cfg.lock_joints = {lock_joint_name: cspace.default_joint_position[j_idx].item()}
    cfg = KinematicsCfg.from_config(
        KinematicsLoaderCfg(**config_file["robot_cfg"]["kinematics"], device_cfg=device_cfg)
    )

    robot_model = Kinematics(cfg)
    cspace = CSpaceParams(**config_file["robot_cfg"]["kinematics"]["cspace"])

    cspace.inplace_reindex(robot_model.joint_names)
    q = cspace.default_joint_position.unsqueeze(0).clone()
    out_locked = robot_model.compute_kinematics(
        JointState.from_position(q, joint_names=robot_model.joint_names)
    )
    assert (
        torch.linalg.norm(
            out.tool_poses[tool_frames[0]].position - out_locked.tool_poses[tool_frames[0]].position
        )
        < 1e-5
    )
    assert (
        torch.linalg.norm(
            out.tool_poses[tool_frames[0]].quaternion - out_locked.tool_poses[tool_frames[0]].quaternion
        )
        < 1e-5
    )
    assert torch.linalg.norm(out.robot_spheres - out_locked.robot_spheres) < 1e-5


def test_franka_toggle_link_collision(cfg):
    device_cfg = DeviceCfg()

    robot_model = Kinematics(cfg)
    q_test = torch.as_tensor(
        [0.0, -1.2, 0.0, -2.0, 0.0, 1.0, 0.0], **(device_cfg.as_torch_dict())
    ).view(1, -1)
    sph_idx = robot_model.kinematics_config.get_sphere_index_from_link_name("panda_link5")
    state = robot_model.compute_kinematics(
        JointState.from_position(q_test.clone(), joint_names=robot_model.joint_names)
    )

    attached_spheres = state.robot_spheres[:, 0, sph_idx, :]
    link_radius = attached_spheres[..., 3].clone()

    assert torch.count_nonzero(link_radius <= 0.0) == 0

    robot_model.kinematics_config.disable_link_spheres("panda_link5")

    state = robot_model.compute_kinematics(
        JointState.from_position(q_test.clone(), joint_names=robot_model.joint_names)
    )

    attached_spheres = state.robot_spheres[:, 0, sph_idx, :]

    sph_radius = attached_spheres[..., 3].clone()
    assert torch.count_nonzero(sph_radius < 0.0)

    robot_model.kinematics_config.enable_link_spheres("panda_link5")

    state = robot_model.compute_kinematics(
        JointState.from_position(q_test.clone(), joint_names=robot_model.joint_names)
    )

    radius = link_radius - state.robot_spheres[:, 0, sph_idx, 3]

    assert torch.count_nonzero(radius == 0.0)
