# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

import pytest
import torch

from curobo._src.collision.attachment_manager import AttachmentManager
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.sphere_fit.types import SphereFitType
from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


@pytest.fixture(scope="module")
def device_cfg():
    return DeviceCfg()


@pytest.fixture(scope="module")
def kinematics_cfg():
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    robot_data["robot_cfg"]["kinematics"]["extra_collision_spheres"] = {
        "attached_object": 100,
    }
    return KinematicsCfg.from_robot_yaml_file(robot_data, ["panda_hand"])


@pytest.fixture(scope="module")
def robot_model(kinematics_cfg):
    return Kinematics(kinematics_cfg)


@pytest.fixture
def manager(robot_model, device_cfg):
    return AttachmentManager(
        kinematics=robot_model,
        scene_collision=None,
        device_cfg=device_cfg,
    )


@pytest.fixture
def grasp_joint_state(device_cfg):
    q = torch.as_tensor(
        [0.0, -1.2, 0.0, -2.0, 0.0, 1.0, 0.0], **(device_cfg.as_torch_dict())
    ).view(1, -1)
    return JointState.from_position(q)


@pytest.fixture
def cube_obstacle():
    return Cuboid(
        name="test_cube",
        pose=[0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],
        dims=[0.05, 0.05, 0.05],
    )


class TestFitSpheres:
    """Test fit_spheres method."""

    def test_fit_returns_tensor_with_4_columns(self, manager, cube_obstacle):
        result = manager.fit_spheres(
            [cube_obstacle], num_spheres=10, sphere_fit_type=SphereFitType.VOXEL,
        )
        assert result.dim() == 2
        assert result.shape[1] == 4
        assert result.shape[0] > 0

    def test_fit_radii_positive(self, manager, cube_obstacle):
        result = manager.fit_spheres(
            [cube_obstacle], num_spheres=10, sphere_fit_type=SphereFitType.VOXEL,
        )
        assert (result[:, 3] > 0).all()

    def test_fit_stores_last_result(self, manager, cube_obstacle):
        result = manager.fit_spheres(
            [cube_obstacle], num_spheres=5, sphere_fit_type=SphereFitType.VOXEL,
        )
        assert manager._last_fit_result is not None
        assert manager._last_fit_result.num_spheres == result.shape[0]


class TestUpdate:
    """Test update method."""

    def test_update_writes_to_link_spheres(
        self, manager, grasp_joint_state, device_cfg
    ):
        kparams = manager.kinematics_params
        sph_idx = kparams.get_sphere_index_from_link_name("attached_object")

        sphere_tensor = torch.zeros((2, 4), **(device_cfg.as_torch_dict()))
        sphere_tensor[:, 3] = 0.01
        sphere_tensor[0, 0] = 0.1
        sphere_tensor[1, 0] = 0.2

        manager.update(sphere_tensor, grasp_joint_state)

        updated = kparams.link_spheres[0, sph_idx, :]
        assert updated[0, 3].item() == pytest.approx(0.01, abs=1e-5)
        assert updated[1, 3].item() == pytest.approx(0.01, abs=1e-5)
        assert updated[2, 3].item() == pytest.approx(-100.0)

        kparams.reset_link_spheres("attached_object")

    def test_update_identity_offset(self, manager, grasp_joint_state, device_cfg):
        """When world_objects_pose_offset is None, obstacle-frame spheres are written as-is."""
        sphere_tensor = torch.zeros((3, 4), **(device_cfg.as_torch_dict()))
        sphere_tensor[0, :3] = torch.tensor([0.1, 0.2, 0.3])
        sphere_tensor[:, 3] = 0.02

        manager.update(sphere_tensor, grasp_joint_state)

        kparams = manager.kinematics_params
        sph_idx = kparams.get_sphere_index_from_link_name("attached_object")
        written = kparams.link_spheres[0, sph_idx[:3], :]
        assert written[0, 0].item() == pytest.approx(0.1, abs=1e-5)
        assert written[0, 1].item() == pytest.approx(0.2, abs=1e-5)
        assert written[0, 2].item() == pytest.approx(0.3, abs=1e-5)

        kparams.reset_link_spheres("attached_object")

    def test_update_too_many_spheres_raises(
        self, manager, grasp_joint_state, device_cfg
    ):
        kparams = manager.kinematics_params
        n_slots = kparams.get_number_of_spheres("attached_object")
        sphere_tensor = torch.zeros(
            (n_slots + 1, 4), **(device_cfg.as_torch_dict())
        )
        with pytest.raises(Exception):
            manager.update(sphere_tensor, grasp_joint_state)


@pytest.fixture(scope="module")
def multi_env_robot_model():
    """Robot model with link_spheres initialized to 2 envs via its own config."""
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    robot_data["robot_cfg"]["kinematics"]["extra_collision_spheres"] = {
        "attached_object": 100,
    }
    cfg = KinematicsCfg.from_robot_yaml_file(robot_data, ["panda_hand"])
    model = Kinematics(cfg)
    kparams = model.config.kinematics_config
    kparams.link_spheres = kparams.link_spheres.repeat(2, 1, 1)
    kparams.reference_link_spheres = kparams.reference_link_spheres.repeat(2, 1, 1)
    return model


@pytest.fixture
def multi_env_manager(multi_env_robot_model, device_cfg):
    return AttachmentManager(
        kinematics=multi_env_robot_model,
        scene_collision=None,
        device_cfg=device_cfg,
    )


class TestUpdateMultiEnv:
    """Test update with multiple envs (K=2)."""

    def test_multi_env_writes_per_env(self, multi_env_manager, device_cfg):
        manager = multi_env_manager
        kparams = manager.kinematics_params
        assert kparams.num_envs == 2

        q = torch.zeros((2, kparams.num_dof), **(device_cfg.as_torch_dict()))
        q[0] = torch.tensor(
            [0.0, -1.2, 0.0, -2.0, 0.0, 1.0, 0.0], **(device_cfg.as_torch_dict())
        )
        q[1] = torch.tensor(
            [0.5, -0.8, 0.3, -1.5, 0.2, 0.8, 0.1], **(device_cfg.as_torch_dict())
        )
        joint_state = JointState.from_position(q)

        obj_pos = torch.tensor(
            [[0.5, 0.0, 0.5], [0.3, 0.1, 0.6]], **(device_cfg.as_torch_dict())
        )
        obj_quat = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            **(device_cfg.as_torch_dict()),
        )
        offset = Pose(position=obj_pos, quaternion=obj_quat)

        sphere_tensor = torch.zeros((5, 4), **(device_cfg.as_torch_dict()))
        sphere_tensor[:, 3] = 0.01

        manager.update(sphere_tensor, joint_state, world_objects_pose_offset=offset)

        sph_idx = kparams.get_sphere_index_from_link_name("attached_object")
        env0_spheres = kparams.link_spheres[0, sph_idx[:5], :]
        env1_spheres = kparams.link_spheres[1, sph_idx[:5], :]

        # Different robot + object poses → different link-local spheres
        assert not torch.allclose(env0_spheres[:, :3], env1_spheres[:, :3])
        # Radii should be identical
        assert torch.allclose(env0_spheres[:, 3], env1_spheres[:, 3])

        kparams.reset_link_spheres("attached_object")

    def test_multi_env_attach_detach_roundtrip(
        self, multi_env_manager, device_cfg
    ):
        manager = multi_env_manager
        kparams = manager.kinematics_params
        sph_idx = kparams.get_sphere_index_from_link_name("attached_object")
        original_env0 = kparams.link_spheres[0, sph_idx, :].clone()
        original_env1 = kparams.link_spheres[1, sph_idx, :].clone()

        cube = Cuboid(
            name="test_cube",
            pose=[0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],
            dims=[0.05, 0.05, 0.05],
        )
        q = torch.zeros((2, kparams.num_dof), **(device_cfg.as_torch_dict()))
        q[0] = torch.tensor(
            [0.0, -1.2, 0.0, -2.0, 0.0, 1.0, 0.0], **(device_cfg.as_torch_dict())
        )
        q[1] = torch.tensor(
            [0.5, -0.8, 0.3, -1.5, 0.2, 0.8, 0.1], **(device_cfg.as_torch_dict())
        )

        manager.attach(
            JointState.from_position(q), [cube],
            num_spheres=10, sphere_fit_type=SphereFitType.VOXEL,
        )
        manager.detach()

        assert torch.allclose(kparams.link_spheres[0, sph_idx, :], original_env0)
        assert torch.allclose(kparams.link_spheres[1, sph_idx, :], original_env1)


class TestAttachDetach:
    """Test attach and detach convenience methods."""

    def test_attach_writes_spheres(
        self, manager, grasp_joint_state, cube_obstacle, device_cfg
    ):
        kparams = manager.kinematics_params
        sph_idx = kparams.get_sphere_index_from_link_name("attached_object")

        manager.attach(
            grasp_joint_state, [cube_obstacle],
            num_spheres=10, sphere_fit_type=SphereFitType.VOXEL,
        )

        n_fitted = manager._last_fit_result.num_spheres
        updated = kparams.link_spheres[0, sph_idx, :]
        assert (updated[:n_fitted, 3] > 0).all()
        assert manager._attached_link_name == "attached_object"

        kparams.reset_link_spheres("attached_object")
        manager._attached_link_name = None

    def test_detach_resets_spheres(
        self, manager, grasp_joint_state, cube_obstacle, device_cfg
    ):
        kparams = manager.kinematics_params
        sph_idx = kparams.get_sphere_index_from_link_name("attached_object")
        original = kparams.link_spheres[0, sph_idx, :].clone()

        manager.attach(
            grasp_joint_state, [cube_obstacle],
            num_spheres=10, sphere_fit_type=SphereFitType.VOXEL,
        )
        manager.detach()

        restored = kparams.link_spheres[0, sph_idx, :]
        assert torch.allclose(original, restored)
        assert manager._attached_link_name is None

    def test_detach_without_attach_is_noop(self, manager):
        manager.detach()

    def test_attach_with_world_objects_pose(
        self, manager, grasp_joint_state, cube_obstacle, device_cfg
    ):
        offset = Pose(
            position=torch.tensor(
                [[0.5, 0.0, 0.5]], **(device_cfg.as_torch_dict())
            ),
            quaternion=torch.tensor(
                [[1.0, 0.0, 0.0, 0.0]], **(device_cfg.as_torch_dict())
            ),
        )
        manager.attach(
            grasp_joint_state, [cube_obstacle],
            num_spheres=10, sphere_fit_type=SphereFitType.VOXEL,
            world_objects_pose_offset=offset,
        )

        n_fitted = manager._last_fit_result.num_spheres
        kparams = manager.kinematics_params
        sph_idx = kparams.get_sphere_index_from_link_name("attached_object")
        updated = kparams.link_spheres[0, sph_idx, :]
        assert (updated[:n_fitted, 3] > 0).all()

        manager.detach()


class TestObstacleDisableEnable:
    """Test that attach disables and detach re-enables world obstacles."""

    @pytest.fixture
    def scene_collision(self, device_cfg):
        scene_cfg = SceneCfg(
            cuboid=[
                Cuboid(
                    name="world_cube",
                    pose=[0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],
                    dims=[0.1, 0.1, 0.1],
                ),
            ],
        )
        cfg = SceneCollisionCfg(
            device_cfg=device_cfg,
            scene_model=scene_cfg,
            cache={"cuboid": 10},
        )
        return SceneCollision.from_config(cfg)

    @pytest.fixture
    def manager_with_scene(self, robot_model, scene_collision, device_cfg):
        return AttachmentManager(
            kinematics=robot_model,
            scene_collision=scene_collision,
            device_cfg=device_cfg,
        )

    def test_attach_disables_obstacle(
        self, manager_with_scene, scene_collision,
        grasp_joint_state, cube_obstacle, device_cfg,
    ):
        cuboid_idx = scene_collision.data.cuboids.get_idx("world_cube", env_idx=0)
        assert scene_collision.data.cuboids.enable[0, cuboid_idx].item() == 1

        manager_with_scene.attach(
            grasp_joint_state, [cube_obstacle],
            num_spheres=10, sphere_fit_type=SphereFitType.VOXEL,
            disable_obstacle_names=["world_cube"],
        )

        assert scene_collision.data.cuboids.enable[0, cuboid_idx].item() == 0

        manager_with_scene.detach()

    def test_detach_re_enables_obstacle(
        self, manager_with_scene, scene_collision,
        grasp_joint_state, cube_obstacle, device_cfg,
    ):
        cuboid_idx = scene_collision.data.cuboids.get_idx("world_cube", env_idx=0)

        manager_with_scene.attach(
            grasp_joint_state, [cube_obstacle],
            num_spheres=10, sphere_fit_type=SphereFitType.VOXEL,
            disable_obstacle_names=["world_cube"],
        )
        assert scene_collision.data.cuboids.enable[0, cuboid_idx].item() == 0

        manager_with_scene.detach()
        assert scene_collision.data.cuboids.enable[0, cuboid_idx].item() == 1


class TestAttachFromScene:
    """Test attach_from_scene that looks up obstacles by name from SceneCollision."""

    @pytest.fixture
    def scene_collision(self, device_cfg):
        scene_cfg = SceneCfg(
            cuboid=[
                Cuboid(
                    name="scene_cube",
                    pose=[0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0],
                    dims=[0.05, 0.05, 0.05],
                ),
            ],
        )
        cfg = SceneCollisionCfg(
            device_cfg=device_cfg,
            scene_model=scene_cfg,
            cache={"cuboid": 10},
        )
        return SceneCollision.from_config(cfg)

    @pytest.fixture
    def manager_with_scene(self, robot_model, scene_collision, device_cfg):
        return AttachmentManager(
            kinematics=robot_model,
            scene_collision=scene_collision,
            device_cfg=device_cfg,
        )

    def test_attach_from_scene_writes_spheres(
        self, manager_with_scene, grasp_joint_state,
    ):
        manager_with_scene.attach_from_scene(
            grasp_joint_state, ["scene_cube"],
            num_spheres=10, sphere_fit_type=SphereFitType.VOXEL,
        )

        n_fitted = manager_with_scene._last_fit_result.num_spheres
        kparams = manager_with_scene.kinematics_params
        sph_idx = kparams.get_sphere_index_from_link_name("attached_object")
        updated = kparams.link_spheres[0, sph_idx, :]
        assert (updated[:n_fitted, 3] > 0).all()

        manager_with_scene.detach()

    def test_attach_from_scene_disables_obstacle(
        self, manager_with_scene, scene_collision, grasp_joint_state,
    ):
        cuboid_idx = scene_collision.data.cuboids.get_idx("scene_cube", env_idx=0)
        assert scene_collision.data.cuboids.enable[0, cuboid_idx].item() == 1

        manager_with_scene.attach_from_scene(
            grasp_joint_state, ["scene_cube"],
            num_spheres=10, sphere_fit_type=SphereFitType.VOXEL,
        )
        assert scene_collision.data.cuboids.enable[0, cuboid_idx].item() == 0

        manager_with_scene.detach()
        assert scene_collision.data.cuboids.enable[0, cuboid_idx].item() == 1

    def test_attach_from_scene_missing_name_raises(
        self, manager_with_scene, grasp_joint_state,
    ):
        with pytest.raises(Exception):
            manager_with_scene.attach_from_scene(
                grasp_joint_state, ["nonexistent_obstacle"],
            )

    def test_attach_from_scene_no_scene_collision_raises(
        self, robot_model, device_cfg, grasp_joint_state,
    ):
        manager_no_scene = AttachmentManager(
            kinematics=robot_model,
            scene_collision=None,
            device_cfg=device_cfg,
        )
        with pytest.raises(Exception):
            manager_no_scene.attach_from_scene(
                grasp_joint_state, ["scene_cube"],
            )
