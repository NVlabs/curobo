# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Test that GoalRegistry.idxs_env is correctly forwarded to scene collision cost.

Regression test for a bug where the cost manager read ``idxs_env`` from
``RobotState`` (where it doesn't exist) instead of from
``GoalRegistry.idxs_env``. That caused multi-env collision to silently use
env 0 for all batch rows.
"""

import pytest
import torch

from curobo._src.cost.cost_scene_collision_cfg import SceneCollisionCostCfg
from curobo._src.geom.collision import SceneCollisionCfg, create_collision_checker
from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.robot.kinematics.kinematics_state import KinematicsState
from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
from curobo._src.rollout.goal_registry import GoalRegistry
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_robot import RobotState


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestBatchEnvWorldIdx:
    """Verify that idxs_env from GoalRegistry reaches the scene collision checker."""

    @pytest.fixture
    def franka_transition_model(self, cuda_device_cfg):
        from curobo._src.transition.robot_state_transition import RobotStateTransition
        from curobo._src.transition.robot_state_transition_cfg import RobotStateTransitionCfg
        from curobo._src.util_file import (
            get_robot_configs_path,
            get_task_configs_path,
            join_path,
            load_yaml,
        )

        robot_file = join_path(get_robot_configs_path(), "franka.yml")
        robot_cfg_dict = load_yaml(robot_file)["robot_cfg"]

        task_file = join_path(get_task_configs_path(), "ik/transition_ik.yml")
        transition_dict = load_yaml(task_file).get("transition_model_cfg", {})
        transition_dict["horizon"] = 1
        transition_dict["batch_size"] = 2

        transition_cfg = RobotStateTransitionCfg.create(
            transition_dict, robot_cfg_dict, cuda_device_cfg,
        )
        return RobotStateTransition(transition_cfg)

    @pytest.fixture
    def two_env_collision_checker(self, cuda_device_cfg):
        """Create a collision checker with 2 environments.

        Env 0: cuboid obstacle at origin (will collide with spheres placed at origin).
        Env 1: no obstacles (empty scene).
        """
        env0_scene = SceneCfg(
            cuboid=[
                Cuboid(
                    name="obstacle",
                    pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    dims=[0.5, 0.5, 0.5],
                )
            ],
        )
        env1_scene = SceneCfg()

        world_cfg = SceneCollisionCfg(
            device_cfg=cuda_device_cfg,
            max_distance=0.1,
            scene_model=[env0_scene, env1_scene],
            cache={"primitive": 10},
        )
        return create_collision_checker(world_cfg)

    def test_idxs_world_reaches_collision_checker(
        self, cuda_device_cfg, franka_transition_model, two_env_collision_checker,
    ):
        """Collision costs must differ when batch rows map to different environments.

        Batch row 0 → env 0 (has obstacle at origin)
        Batch row 1 → env 1 (empty)

        Robot spheres are placed at origin so row 0 should have nonzero collision
        cost while row 1 should have zero.
        """
        batch_size = 2
        horizon = 1
        num_spheres = franka_transition_model.robot_model.total_spheres

        scene_collision_cfg = SceneCollisionCostCfg(
            weight=1.0,
            device_cfg=cuda_device_cfg,
        )
        config = RobotCostManagerCfg(scene_collision_cfg=scene_collision_cfg)

        manager = RobotCostManager(device_cfg=cuda_device_cfg)
        manager.initialize_from_config(
            config,
            franka_transition_model,
            scene_collision_checker=two_env_collision_checker,
        )

        if not manager.has_cost("scene_collision"):
            pytest.skip("Scene collision cost not registered (no collision spheres)")

        dof = franka_transition_model.num_dof
        joint_state = JointState(
            position=torch.zeros(batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict()),
            velocity=torch.zeros(batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict()),
            acceleration=torch.zeros(
                batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict()
            ),
            jerk=torch.zeros(batch_size, horizon, dof, **cuda_device_cfg.as_torch_dict()),
            device_cfg=cuda_device_cfg,
        )

        # Place all spheres at origin with radius 0.05; overlaps with env 0's cuboid
        robot_spheres = torch.zeros(
            batch_size, horizon, num_spheres, 4, **cuda_device_cfg.as_torch_dict()
        )
        robot_spheres[..., 3] = 0.05

        cuda_robot_model_state = KinematicsState(robot_spheres=robot_spheres)
        robot_state = RobotState(
            joint_state=joint_state, cuda_robot_model_state=cuda_robot_model_state,
        )

        # GoalRegistry with idxs_env: batch 0 → env 0, batch 1 → env 1
        goal = GoalRegistry(
            idxs_env=torch.tensor([[0], [1]], dtype=torch.int32, device=cuda_device_cfg.device),
        )

        manager.setup_batch_tensors(batch_size, horizon)
        result = manager.compute_costs(robot_state, goal=goal)

        assert "scene_collision" in result.names

        idx = result.names.index("scene_collision")
        costs = result.values[idx]  # shape: [batch, horizon]
        cost_env0 = costs[0].sum().item()
        cost_env1 = costs[1].sum().item()

        # Env 0 has an obstacle at origin overlapping our spheres → nonzero cost
        assert cost_env0 > 0.0, f"Expected nonzero collision cost for env 0, got {cost_env0}"

        assert cost_env1 == 0.0, (
            f"Expected zero collision cost for env 1 (empty scene), got {cost_env1}. "
            "This indicates idxs_env is not being forwarded from GoalRegistry "
            "to the scene collision checker."
        )
