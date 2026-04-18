# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Verify that all solvers within a MotionPlanner share the same KinematicsParams.

This invariant ensures that mutating link_spheres (e.g., attaching an object) on one
solver's kinematics automatically propagates to all other solvers without fan-out.
"""

import pytest
import torch

from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.types.device_cfg import DeviceCfg


@pytest.fixture(scope="module")
def cuda_device_cfg():
    if torch.cuda.is_available():
        return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    pytest.skip("CUDA not available")


@pytest.fixture(scope="module")
def planner(cuda_device_cfg):
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        device_cfg=cuda_device_cfg,
        num_ik_seeds=4,
        num_trajopt_seeds=2,
        use_cuda_graph=False,
    )
    return MotionPlanner(config)


class TestSharedKinematicsParams:
    """All solvers within a planner must share the same KinematicsParams object."""

    def test_ik_and_trajopt_share_kinematics_params(self, planner):
        ik_kparams = planner.ik_solver.kinematics.config.kinematics_config
        trajopt_kparams = planner.trajopt_solver.kinematics.config.kinematics_config
        assert ik_kparams is trajopt_kparams

    def test_ik_and_trajopt_share_link_spheres_tensor(self, planner):
        ik_spheres = planner.ik_solver.kinematics.config.kinematics_config.link_spheres
        trajopt_spheres = planner.trajopt_solver.kinematics.config.kinematics_config.link_spheres
        assert ik_spheres.data_ptr() == trajopt_spheres.data_ptr()

    def test_mutation_propagates_across_solvers(self, planner):
        """Mutating link_spheres via one solver is visible from the other."""
        kparams = planner.ik_solver.kinematics.config.kinematics_config
        sph_idx = kparams.get_sphere_index_from_link_name("attached_object")
        original = kparams.link_spheres[0, sph_idx[0], 3].item()

        kparams.link_spheres[0, sph_idx[0], 3] = 0.123

        trajopt_val = (
            planner.trajopt_solver.kinematics.config.kinematics_config
            .link_spheres[0, sph_idx[0], 3].item()
        )
        assert trajopt_val == pytest.approx(0.123, abs=1e-6)

        kparams.link_spheres[0, sph_idx[0], 3] = original

    def test_all_rollouts_within_solver_share_kinematics_params(self, planner):
        """All rollouts within a single solver share the same KinematicsParams."""
        solver = planner.trajopt_solver
        base_kparams = solver.kinematics.config.kinematics_config
        for rollout in solver.get_all_rollout_instances():
            rollout_kparams = rollout.transition_model.robot_model.config.kinematics_config
            assert rollout_kparams is base_kparams

    def test_graph_planner_shares_kinematics_params_if_present(self, planner):
        if planner.graph_planner is None:
            pytest.skip("No graph planner configured")
        graph_kparams = planner.graph_planner.kinematics.config.kinematics_config
        trajopt_kparams = planner.trajopt_solver.kinematics.config.kinematics_config
        assert graph_kparams is trajopt_kparams
