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

# Standard Library
import copy

# Third Party
import pytest
import torch

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.rollout.cost.self_collision_cost import SelfCollisionCost, SelfCollisionCostConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml


@pytest.mark.parametrize(
    "batch_size, horizon",
    [
        pytest.param(1, 1, id="1"),
        pytest.param(10, 1, id="10"),
        pytest.param(100000, 1, id="100k"),
        pytest.param(100, 70, id="horizon"),
    ],
)
def test_self_collision_experimental(batch_size, horizon):
    robot_file = "franka.yml"
    tensor_args = TensorDeviceType()

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
    robot_cfg["kinematics"]["debug"] = {"self_collision_experimental": False}
    robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
    kinematics = CudaRobotModel(robot_cfg.kinematics)
    self_collision_data = kinematics.get_self_collision_config()
    self_collision_config = SelfCollisionCostConfig(
        **{"weight": 1.0, "classify": True, "self_collision_kin_config": self_collision_data},
        tensor_args=tensor_args
    )
    cost_fn = SelfCollisionCost(self_collision_config)
    cost_fn.self_collision_kin_config.experimental_kernel = True

    b = batch_size
    h = horizon

    q = (
        torch.rand(
            (b * h, kinematics.get_dof()), device=tensor_args.device, dtype=tensor_args.dtype
        )
        * 10
    )
    kin_state = kinematics.get_state(q)

    in_spheres = kin_state.link_spheres_tensor
    in_spheres = in_spheres.view(b, h, -1, 4).contiguous()

    for _ in range(1):
        out = cost_fn.forward(in_spheres)
    k = out.clone()
    cost_fn.self_collision_kin_config.experimental_kernel = False
    cost_fn._out_distance[:] = 0.0
    for _ in range(1):
        out = cost_fn.forward(in_spheres)

    assert torch.norm(k - out).item() < 1e-8


def test_self_collision_franka():
    tensor_args = TensorDeviceType()

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
    robot_cfg["kinematics"]["debug"] = {"self_collision_experimental": False}

    robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
    robot_cfg.kinematics.self_collision_config.experimental_kernel = True
    kinematics = CudaRobotModel(robot_cfg.kinematics)
    self_collision_data = kinematics.get_self_collision_config()
    self_collision_config = SelfCollisionCostConfig(
        **{"weight": 5000.0, "classify": False, "self_collision_kin_config": self_collision_data},
        tensor_args=tensor_args
    )
    cost_fn = SelfCollisionCost(self_collision_config)
    cost_fn.self_collision_kin_config.experimental_kernel = True

    b = 10
    h = 1

    q = torch.rand(
        (b * h, kinematics.get_dof()), device=tensor_args.device, dtype=tensor_args.dtype
    )

    test_q = tensor_args.to_device([2.7735, -1.6737, 0.4998, -2.9865, 0.3386, 0.8413, 0.4371])
    q[:] = test_q
    kin_state = kinematics.get_state(q)

    in_spheres = kin_state.link_spheres_tensor
    in_spheres = in_spheres.view(b, h, -1, 4).contiguous()

    out = cost_fn.forward(in_spheres)
    assert out.sum().item() > 0.0
    cost_fn.self_collision_kin_config.experimental_kernel = False
    cost_fn._out_distance[:] = 0.0
    out = cost_fn.forward(in_spheres)
    assert out.sum().item() > 0.0


def test_self_collision_10k_spheres_franka():
    tensor_args = TensorDeviceType()

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
    robot_cfg["kinematics"]["debug"] = {"self_collision_experimental": False}

    robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
    robot_cfg.kinematics.self_collision_config.experimental_kernel = True
    kinematics = CudaRobotModel(robot_cfg.kinematics)
    self_collision_data = kinematics.get_self_collision_config()
    self_collision_config = SelfCollisionCostConfig(
        **{"weight": 1.0, "classify": False, "self_collision_kin_config": self_collision_data},
        tensor_args=tensor_args
    )
    cost_fn = SelfCollisionCost(self_collision_config)
    cost_fn.self_collision_kin_config.experimental_kernel = True

    b = 10
    h = 1

    q = torch.rand(
        (b * h, kinematics.get_dof()), device=tensor_args.device, dtype=tensor_args.dtype
    )

    test_q = tensor_args.to_device([2.7735, -1.6737, 0.4998, -2.9865, 0.3386, 0.8413, 0.4371])
    q[0, :] = test_q
    kin_state = kinematics.get_state(q)

    in_spheres = kin_state.link_spheres_tensor
    in_spheres = in_spheres.view(b, h, -1, 4).contiguous()

    out = cost_fn.forward(in_spheres)
    assert out.sum().item() > 0.0

    # create a franka robot with 10k spheres:
    tensor_args = TensorDeviceType()

    robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
    robot_cfg["kinematics"]["debug"] = {"self_collision_experimental": False}

    sphere_cfg = load_yaml(
        join_path(get_robot_configs_path(), robot_cfg["kinematics"]["collision_spheres"])
    )["collision_spheres"]
    n_times = 10
    for k in sphere_cfg.keys():
        sphere_cfg[k] = [copy.deepcopy(x) for x in sphere_cfg[k] for _ in range(n_times)]

    robot_cfg["kinematics"]["collision_spheres"] = sphere_cfg
    robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)
    robot_cfg.kinematics.self_collision_config.experimental_kernel = False

    kinematics = CudaRobotModel(robot_cfg.kinematics)
    self_collision_data = kinematics.get_self_collision_config()
    self_collision_config = SelfCollisionCostConfig(
        **{"weight": 1.0, "classify": False, "self_collision_kin_config": self_collision_data},
        tensor_args=tensor_args
    )
    cost_fn = SelfCollisionCost(self_collision_config)
    cost_fn.self_collision_kin_config.experimental_kernel = False

    kin_state = kinematics.get_state(q)

    in_spheres = kin_state.link_spheres_tensor
    in_spheres = in_spheres.view(b, h, -1, 4).contiguous()

    out_10k = cost_fn.forward(in_spheres)
    assert out_10k.sum().item() > 0.0
    assert torch.linalg.norm(out - out_10k) < 1e-3
