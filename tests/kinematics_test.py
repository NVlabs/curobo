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
from curobo.cuda_robot_model.cuda_robot_generator import CudaRobotGeneratorConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.cuda_robot_model.types import CSpaceConfig
from curobo.geom.transform import matrix_to_quaternion, quaternion_to_matrix
from curobo.geom.types import Cuboid
from curobo.types.base import TensorDeviceType
from curobo.types.math import quat_multiply
from curobo.types.state import JointState
from curobo.util_file import get_robot_configs_path, join_path, load_yaml


@pytest.fixture(scope="module")
def cfg():
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    robot_data["robot_cfg"]["kinematics"]["extra_collision_spheres"] = {"attached_object": 100}
    cfg = CudaRobotModelConfig.from_robot_yaml_file(robot_data, "panda_hand")
    return cfg


def test_quaternion():
    tensor_args = TensorDeviceType()

    def test_q(in_quat):
        out_rot = quaternion_to_matrix(in_quat)
        out_quat = matrix_to_quaternion(out_rot.clone())
        out_quat[..., 1:] *= -1.0
        q_res = quat_multiply(in_quat, out_quat, in_quat.clone())
        q_res[..., 0] = 0.0
        assert torch.sum(q_res).item() <= 1e-5

    in_quat = tensor_args.to_device([1.0, 0.0, 0.0, 0.0]).view(1, 4)
    test_q(in_quat)
    test_q(-1.0 * in_quat)
    in_quat = tensor_args.to_device([0.0, 1.0, 0.0, 0.0]).view(1, 4)
    test_q(in_quat)
    test_q(-1.0 * in_quat)
    in_quat = tensor_args.to_device([0.0, 0.0, 1.0, 0.0]).view(1, 4)
    test_q(in_quat)
    test_q(-1.0 * in_quat)
    in_quat = tensor_args.to_device([0.0, 0.0, 0.0, 1.0]).view(1, 4)
    test_q(in_quat)
    test_q(-1.0 * in_quat)
    in_quat = tensor_args.to_device([0.7071068, 0.0, 0.0, 0.7071068]).view(1, 4)
    test_q(in_quat)
    test_q(-1.0 * in_quat)


def test_franka_kinematics(cfg):
    tensor_args = TensorDeviceType()

    robot_model = CudaRobotModel(cfg)
    q_test = torch.as_tensor(
        [0.0, -1.2, 0.0, -2.0, 0.0, 1.0, 0.0], **(tensor_args.as_torch_dict())
    ).view(1, -1)
    ee_position = torch.as_tensor(
        [6.0860e-02, -4.7547e-12, 7.6373e-01], **(tensor_args.as_torch_dict())
    ).view(1, -1)
    ee_quat = torch.as_tensor(
        [0.0382, 0.9193, 0.3808, 0.0922], **(tensor_args.as_torch_dict())
    ).view(1, -1)
    b_list = [1, 10, 100, 5000][:1]
    for b in b_list:
        state = robot_model.get_state(q_test.repeat(b, 1).clone())
        pos_err = torch.linalg.norm(state.ee_position - ee_position)
        q_err = torch.linalg.norm(state.ee_quaternion - ee_quat)
        # check if all values are equal to position and quaternion
        assert pos_err < 1e-3
        assert q_err < 1e-1


def test_franka_attached_object_kinematics(cfg):
    tensor_args = TensorDeviceType()

    robot_model = CudaRobotModel(cfg)
    q_test = torch.as_tensor(
        [0.0, -1.2, 0.0, -2.0, 0.0, 1.0, 0.0], **(tensor_args.as_torch_dict())
    ).view(1, -1)
    sph_idx = robot_model.kinematics_config.get_sphere_index_from_link_name("attached_object")
    state = robot_model.get_state(q_test.clone())

    attached_spheres = state.link_spheres_tensor[:, sph_idx, :]
    # check if all values are equal to position and quaternion

    assert torch.norm(attached_spheres[:, :, 0] - 0.061) < 0.01
    assert torch.norm(attached_spheres[:, :, 1]) < 0.0001

    # attach an object:
    new_object = torch.zeros((2, 4), **(tensor_args.as_torch_dict()))
    new_object[:, 3] = 0.01
    new_object[:, 1] = 0.1
    robot_model.kinematics_config.update_link_spheres("attached_object", new_object)
    state = robot_model.get_state(q_test.clone())
    attached_spheres = state.link_spheres_tensor[:, sph_idx, :]
    robot_model.kinematics_config.detach_object()
    # assert torch.norm(attached_spheres[:, :2, 0] - 0.0829) < 0.01
    # assert torch.norm(attached_spheres[:, 2:4, 0] - 0.0829) < 0.01


def test_franka_attach_object_kinematics(cfg):
    tensor_args = TensorDeviceType()

    robot_model = CudaRobotModel(cfg)
    q_test = torch.as_tensor(
        [0.0, -1.2, 0.0, -2.0, 0.0, 1.0, 0.0], **(tensor_args.as_torch_dict())
    ).view(1, -1)
    sph_idx = robot_model.kinematics_config.get_sphere_index_from_link_name("attached_object")
    state = robot_model.get_state(q_test.clone())

    attached_spheres = state.link_spheres_tensor[:, sph_idx, :]
    # check if all values are equal to position and quaternion

    assert torch.norm(attached_spheres[:, :, 0] - attached_spheres[0, 0, 0]) < 0.1
    assert torch.norm(attached_spheres[:, :, 1]) < 0.0001

    # attach an object:
    new_object = torch.zeros((100, 4), **(tensor_args.as_torch_dict()))
    new_object[:2, 3] = 0.01
    new_object[:2, 1] = 0.1
    # print(attached_spheres[:, sph_idx,:].shape)

    robot_model.kinematics_config.attach_object(sphere_tensor=new_object)
    state = robot_model.get_state(q_test.clone())
    attached_spheres = state.link_spheres_tensor[:, sph_idx, :]
    assert (torch.norm(attached_spheres[:, :2, 0] - 0.13)) < 0.01
    assert torch.norm(attached_spheres[:, 2:, 0] - 0.061) < 0.01

    robot_model.kinematics_config.detach_object()
    state = robot_model.get_state(q_test.clone())
    attached_spheres = state.link_spheres_tensor[:, sph_idx, :]

    assert torch.norm(attached_spheres[:, :, 0] - 0.061) < 0.01
    assert torch.norm(attached_spheres[:, :, 1]) < 0.0001
    robot_model.kinematics_config.detach_object()


def test_franka_attach_external_object_kinematics(cfg):
    tensor_args = TensorDeviceType()

    robot_model = CudaRobotModel(cfg)
    q_test = torch.as_tensor(
        [0.0, -1.2, 0.0, -2.0, 0.0, 1.0, 0.0], **(tensor_args.as_torch_dict())
    ).view(1, -1)
    sph_idx = robot_model.kinematics_config.get_sphere_index_from_link_name("attached_object")
    state = robot_model.get_state(q_test.clone())

    attached_spheres = state.link_spheres_tensor[:, sph_idx, :]
    # check if all values are equal to position and quaternion
    sphere = attached_spheres[0, 0, 0].item()
    assert torch.norm(attached_spheres[:, :, 0] - sphere) < 0.01

    external_object = Cuboid(
        name="external_object", dims=[0.1, 0.1, 0.1], pose=[0, 0, 0, 1, 0, 0, 0]
    )
    robot_model.attach_external_objects_to_robot(
        JointState.from_position(q_test.clone()), [external_object]
    )

    state = robot_model.get_state(q_test.clone())
    attached_spheres = state.link_spheres_tensor[:, sph_idx, :]

    assert torch.norm(attached_spheres[:, :, 0] - sphere) > 0.1

    robot_model.kinematics_config.detach_object()
    state = robot_model.get_state(q_test.clone())
    attached_spheres = state.link_spheres_tensor[:, sph_idx, :]

    assert torch.norm(attached_spheres[:, :, 0] - sphere) < 0.01
    assert torch.norm(attached_spheres[:, :, 1]) < 0.0001
    robot_model.kinematics_config.detach_object()


def test_locked_joints_kinematics():
    tensor_args = TensorDeviceType()

    config_file = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))

    cfg = CudaRobotModelConfig.from_config(
        CudaRobotGeneratorConfig(**config_file["robot_cfg"]["kinematics"], tensor_args=tensor_args)
    )
    robot_model = CudaRobotModel(cfg)

    cspace = CSpaceConfig(**config_file["robot_cfg"]["kinematics"]["cspace"])
    cspace.inplace_reindex(robot_model.joint_names)
    q = cspace.retract_config.unsqueeze(0)

    out = robot_model.get_state(q)
    j_idx = 2
    lock_joint_name = cspace.joint_names[j_idx]
    # lock a joint:
    cfg.lock_joints = {lock_joint_name: cspace.retract_config[j_idx].item()}
    cfg = CudaRobotModelConfig.from_config(
        CudaRobotGeneratorConfig(**config_file["robot_cfg"]["kinematics"], tensor_args=tensor_args)
    )

    robot_model = CudaRobotModel(cfg)
    cspace = CSpaceConfig(**config_file["robot_cfg"]["kinematics"]["cspace"])

    cspace.inplace_reindex(robot_model.joint_names)
    q = cspace.retract_config.unsqueeze(0).clone()
    out_locked = robot_model.get_state(q)
    assert torch.linalg.norm(out.ee_position - out_locked.ee_position) < 1e-5
    assert torch.linalg.norm(out.ee_quaternion - out_locked.ee_quaternion) < 1e-5
    assert torch.linalg.norm(out.link_spheres_tensor - out_locked.link_spheres_tensor) < 1e-5


def test_franka_toggle_link_collision(cfg):
    tensor_args = TensorDeviceType()

    robot_model = CudaRobotModel(cfg)
    q_test = torch.as_tensor(
        [0.0, -1.2, 0.0, -2.0, 0.0, 1.0, 0.0], **(tensor_args.as_torch_dict())
    ).view(1, -1)
    sph_idx = robot_model.kinematics_config.get_sphere_index_from_link_name("panda_link5")
    state = robot_model.get_state(q_test.clone())

    attached_spheres = state.link_spheres_tensor[:, sph_idx, :]
    # check if all values are equal to position and quaternion
    link_radius = attached_spheres[..., 3].clone()

    assert torch.count_nonzero(link_radius <= 0.0) == 0

    robot_model.kinematics_config.disable_link_spheres("panda_link5")

    state = robot_model.get_state(q_test.clone())

    attached_spheres = state.link_spheres_tensor[:, sph_idx, :]

    sph_radius = attached_spheres[..., 3].clone()
    assert torch.count_nonzero(sph_radius < 0.0)

    robot_model.kinematics_config.enable_link_spheres("panda_link5")

    state = robot_model.get_state(q_test.clone())

    radius = link_radius - state.link_spheres_tensor[:, sph_idx, 3]

    assert torch.count_nonzero(radius == 0.0)
