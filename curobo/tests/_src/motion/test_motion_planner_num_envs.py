# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Verify that num_envs flows through config creation to link_spheres shape."""

import pytest
import torch

from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.motion.motion_planner_batch import BatchMotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.types.device_cfg import DeviceCfg


@pytest.fixture(scope="module")
def device_cfg():
    if torch.cuda.is_available():
        return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    pytest.skip("CUDA not available")


class TestNumEnvsSingleEnv:
    """Default (single env) should produce link_spheres [1, N, 4]."""

    def test_motion_planner_default_num_envs(self, device_cfg):
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=device_cfg,
            num_ik_seeds=4,
            num_trajopt_seeds=2,
            use_cuda_graph=False,
        )
        planner = MotionPlanner(config)
        kparams = planner.kinematics.config.kinematics_config
        assert kparams.num_envs == 1
        assert kparams.link_spheres.shape[0] == 1

    def test_ik_solver_default_num_envs(self, device_cfg):
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=device_cfg,
            num_seeds=4,
            use_cuda_graph=False,
        )
        solver = IKSolver(config)
        kparams = solver.kinematics.config.kinematics_config
        assert kparams.num_envs == 1
        assert kparams.link_spheres.shape[0] == 1


class TestNumEnvsMultiEnv:
    """multi_env=True with max_batch_size=N should produce link_spheres [N, _, 4]."""

    def test_motion_planner_multi_env(self, device_cfg):
        num_envs = 4
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=device_cfg,
            num_ik_seeds=4,
            num_trajopt_seeds=2,
            use_cuda_graph=False,
            multi_env=True,
            max_batch_size=num_envs,
        )
        planner = BatchMotionPlanner(config)
        kparams = planner.kinematics.config.kinematics_config
        assert kparams.num_envs == num_envs
        assert kparams.link_spheres.shape[0] == num_envs

    def test_ik_solver_multi_env(self, device_cfg):
        num_envs = 3
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=device_cfg,
            num_seeds=4,
            use_cuda_graph=False,
            multi_env=True,
            max_batch_size=num_envs,
        )
        solver = IKSolver(config)
        kparams = solver.kinematics.config.kinematics_config
        assert kparams.num_envs == num_envs
        assert kparams.link_spheres.shape[0] == num_envs

    def test_multi_env_reference_spheres_match(self, device_cfg):
        """reference_link_spheres should have the same K dimension."""
        num_envs = 2
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=device_cfg,
            num_seeds=4,
            use_cuda_graph=False,
            multi_env=True,
            max_batch_size=num_envs,
        )
        solver = IKSolver(config)
        kparams = solver.kinematics.config.kinematics_config
        assert kparams.reference_link_spheres.shape[0] == num_envs

    def test_multi_env_all_configs_identical(self, device_cfg):
        """All env slots should start with identical sphere data."""
        num_envs = 3
        config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=device_cfg,
            num_seeds=4,
            use_cuda_graph=False,
            multi_env=True,
            max_batch_size=num_envs,
        )
        solver = IKSolver(config)
        kparams = solver.kinematics.config.kinematics_config
        for i in range(1, num_envs):
            assert torch.allclose(
                kparams.link_spheres[0], kparams.link_spheres[i]
            )
