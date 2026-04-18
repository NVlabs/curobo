# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Test per-batch sphere configuration via idxs_env in FK.

Verifies that when link_spheres has multiple configs [K, N, 4] and
idxs_env maps batch rows to different configs, the FK output
robot_spheres differ per batch row for the configured link.
"""

import pytest
import torch

from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


@pytest.fixture(scope="module")
def franka_cfg():
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    robot_data["robot_cfg"]["kinematics"]["extra_collision_spheres"] = {"attached_object": 10}
    return KinematicsCfg.from_robot_yaml_file(robot_data, ["panda_hand"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestMultiEnvSpheres:

    def test_single_config_default(self, franka_cfg):
        """With K=1 and idxs_env=None, FK works as before."""
        robot_model = Kinematics(franka_cfg)
        device_cfg = DeviceCfg()

        assert robot_model.kinematics_config.num_envs == 1
        assert robot_model.kinematics_config.link_spheres.ndim == 3

        q = torch.zeros((2, robot_model.get_dof()), **device_cfg.as_torch_dict())
        state = robot_model.compute_kinematics(
            JointState.from_position(q, joint_names=robot_model.joint_names)
        )

        assert state.robot_spheres is not None
        assert state.robot_spheres.shape == (2, 1, robot_model.kinematics_config.num_spheres, 4)

    def test_multi_config_different_attached_objects(self, franka_cfg):
        """Two configs with different attached objects produce different FK spheres."""
        robot_model = Kinematics(franka_cfg)
        device_cfg = DeviceCfg()
        kin_cfg = robot_model.kinematics_config

        sph_idx = kin_cfg.get_sphere_index_from_link_name("attached_object")
        n_attach = len(sph_idx)

        # Expand link_spheres from [1, N, 4] to [2, N, 4]
        config0 = kin_cfg.link_spheres[0].clone()
        config1 = config0.clone()

        # Config 0: attached_object spheres at y=+0.1 with radius 0.02
        config0[sph_idx, :3] = 0.0
        config0[sph_idx, 1] = 0.1
        config0[sph_idx, 3] = 0.02

        # Config 1: attached_object spheres at y=-0.2 with radius 0.05
        config1[sph_idx, :3] = 0.0
        config1[sph_idx, 1] = -0.2
        config1[sph_idx, 3] = 0.05

        kin_cfg.link_spheres = torch.stack([config0, config1], dim=0)
        kin_cfg.reference_link_spheres = kin_cfg.link_spheres.clone()

        assert kin_cfg.num_envs == 2
        assert kin_cfg.link_spheres.shape == (2, config0.shape[0], 4)

        q = torch.zeros((2, robot_model.get_dof()), **device_cfg.as_torch_dict())

        # Batch 0 → config 0, Batch 1 → config 1
        idxs_env = torch.tensor([0, 1], dtype=torch.int32, device=device_cfg.device)
        state = robot_model.compute_kinematics(
            JointState.from_position(q, joint_names=robot_model.joint_names),
            idxs_env=idxs_env,
        )

        spheres_b0 = state.robot_spheres[0, 0, sph_idx, :]
        spheres_b1 = state.robot_spheres[1, 0, sph_idx, :]

        # Radii should differ
        assert not torch.allclose(spheres_b0[:, 3], spheres_b1[:, 3]), (
            "Attached object radii should differ between configs"
        )

        # World-frame positions should differ (different local y offset transformed by same EE pose)
        assert not torch.allclose(spheres_b0[:, :3], spheres_b1[:, :3], atol=1e-3), (
            "Attached object positions should differ between configs"
        )

    def test_same_idxs_same_result(self, franka_cfg):
        """When all batch rows use the same config, spheres should match."""
        robot_model = Kinematics(franka_cfg)
        device_cfg = DeviceCfg()
        kin_cfg = robot_model.kinematics_config

        sph_idx = kin_cfg.get_sphere_index_from_link_name("attached_object")

        # Two configs but both batch rows point to config 0
        config0 = kin_cfg.link_spheres[0].clone()
        config1 = config0.clone()
        config1[sph_idx, 1] = 0.5  # different, but unused

        kin_cfg.link_spheres = torch.stack([config0, config1], dim=0)
        kin_cfg.reference_link_spheres = kin_cfg.link_spheres.clone()

        q = torch.zeros((2, robot_model.get_dof()), **device_cfg.as_torch_dict())
        idxs_env = torch.tensor([0, 0], dtype=torch.int32, device=device_cfg.device)
        state = robot_model.compute_kinematics(
            JointState.from_position(q, joint_names=robot_model.joint_names),
            idxs_env=idxs_env,
        )

        assert torch.allclose(
            state.robot_spheres[0, 0], state.robot_spheres[1, 0], atol=1e-6
        ), "Same config index should produce identical spheres"

    def test_non_attached_spheres_unaffected(self, franka_cfg):
        """Non-attached links should produce the same spheres regardless of config."""
        robot_model = Kinematics(franka_cfg)
        device_cfg = DeviceCfg()
        kin_cfg = robot_model.kinematics_config

        attach_idx = kin_cfg.get_sphere_index_from_link_name("attached_object")
        all_idx = torch.arange(kin_cfg.num_spheres, device=device_cfg.device)
        non_attach_mask = ~torch.isin(all_idx, attach_idx)
        non_attach_idx = all_idx[non_attach_mask]

        config0 = kin_cfg.link_spheres[0].clone()
        config1 = config0.clone()
        config1[attach_idx, 1] = 0.5

        kin_cfg.link_spheres = torch.stack([config0, config1], dim=0)
        kin_cfg.reference_link_spheres = kin_cfg.link_spheres.clone()

        q = torch.zeros((2, robot_model.get_dof()), **device_cfg.as_torch_dict())
        idxs_env = torch.tensor([0, 1], dtype=torch.int32, device=device_cfg.device)
        state = robot_model.compute_kinematics(
            JointState.from_position(q, joint_names=robot_model.joint_names),
            idxs_env=idxs_env,
        )

        spheres_b0_non_attach = state.robot_spheres[0, 0, non_attach_idx, :]
        spheres_b1_non_attach = state.robot_spheres[1, 0, non_attach_idx, :]

        assert torch.allclose(spheres_b0_non_attach, spheres_b1_non_attach, atol=1e-6), (
            "Non-attached link spheres should be identical across configs"
        )

    def test_gradient_flows_through_multi_config(self, franka_cfg):
        """Gradients should flow back through joint positions with multi-config spheres."""
        robot_model = Kinematics(franka_cfg)
        device_cfg = DeviceCfg()
        kin_cfg = robot_model.kinematics_config

        sph_idx = kin_cfg.get_sphere_index_from_link_name("attached_object")

        config0 = kin_cfg.link_spheres[0].clone()
        config1 = config0.clone()
        config1[sph_idx, 1] = 0.1
        config1[sph_idx, 3] = 0.02

        kin_cfg.link_spheres = torch.stack([config0, config1], dim=0)
        kin_cfg.reference_link_spheres = kin_cfg.link_spheres.clone()

        q = torch.zeros(
            (2, robot_model.get_dof()), **device_cfg.as_torch_dict()
        ).requires_grad_(True)
        idxs_env = torch.tensor([0, 1], dtype=torch.int32, device=device_cfg.device)

        state = robot_model.compute_kinematics(
            JointState.from_position(q, joint_names=robot_model.joint_names),
            idxs_env=idxs_env,
        )
        loss = state.robot_spheres[:, :, sph_idx, :3].sum()
        loss.backward()

        assert q.grad is not None, "Gradients should flow through multi-config FK"
        assert not torch.all(q.grad == 0), "Gradients should be nonzero"
