# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.motion.motion_retargeter_cfg import MotionRetargeterCfg


def _sample_criteria():
    return {
        "link_a": ToolPoseCriteria.track_position_and_orientation(
            xyz=[1.0, 1.0, 1.0], rpy=[0.5, 0.5, 0.5],
        ),
        "link_b": ToolPoseCriteria.track_position(xyz=[1.0, 1.0, 1.0]),
    }


class TestMotionRetargeterCfg:

    def test_create_defaults(self):
        cfg = MotionRetargeterCfg.create(
            robot="unitree_g1_29dof_retarget.yml",
            tool_pose_criteria=_sample_criteria(),
        )
        assert cfg.num_envs == 1
        assert cfg.use_mpc is False
        assert cfg.self_collision_check is True
        assert cfg.num_seeds_global == 64
        assert cfg.num_seeds_local == 1
        assert cfg.steps_per_target == 4
        assert cfg.num_control_points == 12
        assert cfg.optimization_dt == 0.05

    def test_tool_frames_derived_from_criteria(self):
        criteria = _sample_criteria()
        cfg = MotionRetargeterCfg.create(
            robot="unitree_g1_29dof_retarget.yml",
            tool_pose_criteria=criteria,
        )
        assert cfg.tool_frames == ["link_a", "link_b"]

    def test_create_mpc_mode(self):
        cfg = MotionRetargeterCfg.create(
            robot="unitree_g1_29dof_retarget.yml",
            tool_pose_criteria=_sample_criteria(),
            use_mpc=True,
            steps_per_target=8,
        )
        assert cfg.use_mpc is True
        assert cfg.steps_per_target == 8

    def test_create_custom_num_envs(self):
        cfg = MotionRetargeterCfg.create(
            robot="unitree_g1_29dof_retarget.yml",
            tool_pose_criteria=_sample_criteria(),
            num_envs=4,
        )
        assert cfg.num_envs == 4

    def test_collision_spheres_auto_disabled(self):
        """load_collision_spheres should auto-disable when no collision checking."""
        cfg = MotionRetargeterCfg.create(
            robot="unitree_g1_29dof_retarget.yml",
            tool_pose_criteria=_sample_criteria(),
            self_collision_check=False,
        )
        assert cfg.load_collision_spheres is False

    def test_self_collision_requires_spheres(self):
        """Cannot have self_collision_check=True with load_collision_spheres=False."""
        with pytest.raises(Exception):
            MotionRetargeterCfg(
                robot="unitree_g1_29dof_retarget.yml",
                tool_pose_criteria=_sample_criteria(),
                self_collision_check=True,
                load_collision_spheres=False,
            )

    def test_scene_model_requires_spheres(self):
        """Cannot have scene_model set with load_collision_spheres=False."""
        with pytest.raises(Exception):
            MotionRetargeterCfg(
                robot="unitree_g1_29dof_retarget.yml",
                tool_pose_criteria=_sample_criteria(),
                self_collision_check=False,
                scene_model="some_scene.yml",
                load_collision_spheres=False,
            )

    def test_custom_optimizer_configs(self):
        cfg = MotionRetargeterCfg.create(
            robot="unitree_g1_29dof_retarget.yml",
            tool_pose_criteria=_sample_criteria(),
            ik_optimizer_configs=["ik/custom.yml"],
            mpc_optimizer_configs=["mpc/custom.yml"],
        )
        assert cfg.ik_optimizer_configs == ["ik/custom.yml"]
        assert cfg.mpc_optimizer_configs == ["mpc/custom.yml"]

    def test_default_optimizer_configs(self):
        cfg = MotionRetargeterCfg.create(
            robot="unitree_g1_29dof_retarget.yml",
            tool_pose_criteria=_sample_criteria(),
        )
        assert cfg.ik_optimizer_configs == ["ik/lbfgs_retarget_ik.yml"]
        assert cfg.mpc_optimizer_configs == ["mpc/lbfgs_retarget_mpc.yml"]
