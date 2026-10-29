# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for MotionPlannerCfg configuration class."""

# Third Party
import pytest
import torch

from curobo._src.graph_planner.graph_planner_prm import PRMGraphPlannerCfg

# CuRobo
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.solver.solver_trajopt_cfg import TrajOptSolverCfg
from curobo._src.types.device_cfg import DeviceCfg


@pytest.fixture(scope="module")
def cuda_device_cfg():
    """Create a CUDA device configuration for testing."""
    if torch.cuda.is_available():
        return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    pytest.skip("CUDA not available")


@pytest.fixture(scope="module")
def motion_planner_config(cuda_device_cfg):
    """Create MotionPlannerCfg for Franka robot."""
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        device_cfg=cuda_device_cfg,
        num_ik_seeds=16,
        num_trajopt_seeds=2,
        use_cuda_graph=False,
    )
    return config


class TestMotionPlannerCfgDataclassDefinition:
    """Test MotionPlannerCfg dataclass definition."""

    def test_has_ik_solver_config_field(self):
        """Test dataclass has ik_solver_config field."""
        assert hasattr(MotionPlannerCfg, "__dataclass_fields__")
        assert "ik_solver_config" in MotionPlannerCfg.__dataclass_fields__

    def test_has_trajopt_solver_config_field(self):
        """Test dataclass has trajopt_solver_config field."""
        assert "trajopt_solver_config" in MotionPlannerCfg.__dataclass_fields__

    def test_has_algorithm_specific_graph_planner_config_fields(self):
        """Keep PRM and RRT construction configurations mutually explicit."""
        assert "prm_graph_planner_config" in MotionPlannerCfg.__dataclass_fields__
        assert "rrt_graph_planner_config" in MotionPlannerCfg.__dataclass_fields__
        assert "constrained_rrt_planner_config" in MotionPlannerCfg.__dataclass_fields__

    def test_has_scene_collision_cfg_field(self):
        """Test dataclass has scene_collision_cfg field."""
        assert "scene_collision_cfg" in MotionPlannerCfg.__dataclass_fields__

    def test_has_device_cfg_field(self):
        """Test dataclass has device_cfg field."""
        assert "device_cfg" in MotionPlannerCfg.__dataclass_fields__


class TestMotionPlannerCfgCreateMethod:
    """Test MotionPlannerCfg.create factory method."""

    def test_create_method_exists(self):
        """Test create method exists."""
        assert hasattr(MotionPlannerCfg, "create")
        assert callable(getattr(MotionPlannerCfg, "create"))

    def test_create_is_static_method(self):
        """Test create is a static method."""
        assert isinstance(MotionPlannerCfg.__dict__["create"], staticmethod)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_creates_config(self, cuda_device_cfg):
        """Test create creates a valid config."""
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            use_cuda_graph=False,
        )
        assert config is not None
        assert isinstance(config, MotionPlannerCfg)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_creates_ik_solver_config(self, cuda_device_cfg):
        """Test create creates IKSolverCfg."""
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            use_cuda_graph=False,
        )
        assert config.ik_solver_config is not None
        assert isinstance(config.ik_solver_config, IKSolverCfg)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_creates_trajopt_solver_config(self, cuda_device_cfg):
        """Test create creates TrajOptSolverCfg."""
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            use_cuda_graph=False,
        )
        assert config.trajopt_solver_config is not None
        assert isinstance(config.trajopt_solver_config, TrajOptSolverCfg)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_creates_graph_planner_config(self, cuda_device_cfg):
        """Test create creates PRMGraphPlannerCfg."""
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            use_cuda_graph=False,
        )
        assert config.prm_graph_planner_config is not None
        assert isinstance(config.prm_graph_planner_config, PRMGraphPlannerCfg)
        assert config.rrt_graph_planner_config is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_with_scene_model(self, cuda_device_cfg):
        """Test create with scene model."""
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            scene_model="collision_test.yml",
            device_cfg=cuda_device_cfg,
            use_cuda_graph=False,
        )
        assert config.scene_collision_cfg is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_sets_num_ik_seeds(self, cuda_device_cfg):
        """Test create sets num_ik_seeds correctly."""
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            num_ik_seeds=64,
            use_cuda_graph=False,
        )
        assert config.ik_solver_config.num_seeds == 64

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_sets_num_trajopt_seeds(self, cuda_device_cfg):
        """Test create sets num_trajopt_seeds correctly."""
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            num_trajopt_seeds=8,
            use_cuda_graph=False,
        )
        assert config.trajopt_solver_config.num_seeds == 8

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_sets_interpolation_config(self, cuda_device_cfg) -> None:
        """Test interpolation settings are forwarded to TrajOpt."""
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            interpolation_dt=0.05,
            interpolation_buffer_size=2048,
            use_cuda_graph=False,
        )

        assert config.trajopt_solver_config.interpolation_dt == 0.05
        assert config.trajopt_solver_config.interpolation_buffer_size == 2048

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_uses_single_problem_seed_defaults(self, cuda_device_cfg):
        """Test create uses the standard seed defaults for one problem."""
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            use_cuda_graph=False,
        )

        assert config.ik_solver_config.num_seeds == 32
        assert config.trajopt_solver_config.num_seeds == 4

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_uses_batch_seed_defaults(self, cuda_device_cfg):
        """Test create reduces default seed counts for batched problems."""
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            max_batch_size=4,
            use_cuda_graph=False,
        )

        assert config.ik_solver_config.num_seeds == 16
        assert config.trajopt_solver_config.num_seeds == 2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_sets_position_tolerance(self, cuda_device_cfg):
        """Test create sets position_tolerance correctly."""
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            position_tolerance=0.01,
            use_cuda_graph=False,
        )
        assert config.ik_solver_config.position_tolerance == 0.01
        assert config.trajopt_solver_config.position_tolerance == 0.01

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_sets_orientation_tolerance(self, cuda_device_cfg):
        """Test create sets orientation_tolerance correctly."""
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            orientation_tolerance=0.1,
            use_cuda_graph=False,
        )
        assert config.ik_solver_config.orientation_tolerance == 0.1
        assert config.trajopt_solver_config.orientation_tolerance == 0.1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_forwards_ik_rank_weights(self, cuda_device_cfg) -> None:
        """Test IK ranking weights do not alter the TrajOpt configuration."""
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            ik_position_rank_weight=3.0,
            ik_orientation_rank_weight=4.0,
            ik_cspace_rank_weight=0.5,
            use_cuda_graph=False,
        )

        assert config.ik_solver_config.position_rank_weight == 3.0
        assert config.ik_solver_config.orientation_rank_weight == 4.0
        assert config.ik_solver_config.cspace_rank_weight == 0.5

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_create_supports_trajopt_only_metrics_rollout(self, cuda_device_cfg):
        trajopt_metrics = {
            "rollout": {
                "constraint_cfg": {
                    "tool_velocity_limit_cfg": {"weight": [1.0, 1.0]},
                }
            }
        }

        config = MotionPlannerCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            use_cuda_graph=False,
            trajopt_metrics_rollout=trajopt_metrics,
        )

        ik_metrics = config.ik_solver_config.metrics_rollout_config
        trajopt_metrics_cfg = config.trajopt_solver_config.metrics_rollout_config
        assert ik_metrics.constraint_cfg.tool_velocity_limit_cfg is None
        assert ik_metrics.transition_model_cfg.compute_jacobian is False
        assert trajopt_metrics_cfg.constraint_cfg.tool_velocity_limit_cfg is not None
        assert trajopt_metrics_cfg.transition_model_cfg.compute_jacobian is True
        assert "transition_model_cfg" not in trajopt_metrics["rollout"]


class TestMotionPlannerCfgDirectConstruction:
    """Test MotionPlannerCfg direct construction."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_direct_construction_with_configs(self, cuda_device_cfg):
        """Test direct construction with IK and TrajOpt configs."""
        # Create minimal IK config
        ik_config = IKSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            use_cuda_graph=False,
        )

        # Create minimal TrajOpt config
        trajopt_config = TrajOptSolverCfg.create(
            robot="franka.yml",
            device_cfg=cuda_device_cfg,
            use_cuda_graph=False,
        )

        # Create MotionPlannerCfg directly
        config = MotionPlannerCfg(
            ik_solver_config=ik_config,
            trajopt_solver_config=trajopt_config,
            device_cfg=cuda_device_cfg,
        )

        assert config.ik_solver_config is ik_config
        assert config.trajopt_solver_config is trajopt_config
        assert config.device_cfg == cuda_device_cfg

    def test_direct_construction_defaults(self):
        """Test direct construction uses defaults for optional fields."""
        fields = MotionPlannerCfg.__dataclass_fields__

        assert fields["prm_graph_planner_config"].default is None
        assert fields["rrt_graph_planner_config"].default is None
        assert fields["constrained_rrt_planner_config"].default is None

        # scene_collision_cfg should have a default
        assert fields["scene_collision_cfg"].default is None

    @pytest.mark.parametrize(
        "field", ("always_run_trajopt", "run_trajopt_on_graph_seed", "enable_trajopt")
    )
    def test_trajopt_policy_flags_require_booleans(self, field):
        """Reject ambiguous truthy values for TrajOpt policy controls."""
        with pytest.raises(ValueError, match=f"{field} must be a boolean"):
            MotionPlannerCfg(
                ik_solver_config=None,
                trajopt_solver_config=None,
                **{field: "false"},
            )

    def test_direct_construction_preserves_legacy_positional_order(self):
        """Keep new planner selectors after the original positional fields."""
        ik_config = object()
        trajopt_config = object()
        graph_config = object.__new__(PRMGraphPlannerCfg)
        scene_config = object()
        device_config = object()

        config = MotionPlannerCfg(
            ik_config,
            trajopt_config,
            graph_config,
            scene_config,
            device_config,
        )

        assert config.ik_solver_config is ik_config
        assert config.trajopt_solver_config is trajopt_config
        assert config.prm_graph_planner_config is graph_config
        assert config.scene_collision_cfg is scene_config
        assert config.device_cfg is device_config
        assert config.graph_planner_algorithm == "prm"
        assert config.rrt_graph_planner_config is None
        assert config.constrained_rrt_planner_config is None
        assert config.fit_graph_seed_to_bspline is True
        assert config.always_run_trajopt is True
        assert config.run_trajopt_on_graph_seed is True
        assert config.enable_trajopt is True


class TestMotionPlannerCfgDefaults:
    """Test MotionPlannerCfg default parameter values in create signature."""

    def test_default_num_ik_seeds_parameter(self):
        """Test default num_ik_seeds parameter in create."""
        import inspect

        sig = inspect.signature(MotionPlannerCfg.create)
        assert sig.parameters["num_ik_seeds"].default is None

    def test_default_num_trajopt_seeds_parameter(self):
        """Test default num_trajopt_seeds parameter in create."""
        import inspect

        sig = inspect.signature(MotionPlannerCfg.create)
        assert sig.parameters["num_trajopt_seeds"].default is None

    def test_default_position_tolerance_parameter(self):
        """Test default position_tolerance parameter in create."""
        import inspect

        sig = inspect.signature(MotionPlannerCfg.create)
        assert sig.parameters["position_tolerance"].default == 0.005

    def test_default_orientation_tolerance_parameter(self):
        """Test default orientation_tolerance parameter in create."""
        import inspect

        sig = inspect.signature(MotionPlannerCfg.create)
        assert sig.parameters["orientation_tolerance"].default == 0.05

    def test_default_use_cuda_graph_parameter(self):
        """Test default use_cuda_graph parameter in create."""
        import inspect

        sig = inspect.signature(MotionPlannerCfg.create)
        assert sig.parameters["use_cuda_graph"].default is True

    def test_default_random_seed_parameter(self):
        """Test default random_seed parameter in create."""
        import inspect

        sig = inspect.signature(MotionPlannerCfg.create)
        assert sig.parameters["random_seed"].default == 123

    def test_graph_planner_parameters_are_appended(self):
        """Do not shift existing positional parameters in the create factory."""
        import inspect

        parameter_names = list(inspect.signature(MotionPlannerCfg.create).parameters)
        assert parameter_names[-10:-4] == [
            "graph_planner_algorithm",
            "rrt_planner_config",
            "fit_graph_seed_to_bspline",
            "always_run_trajopt",
            "run_trajopt_on_graph_seed",
            "enable_trajopt",
        ]
        assert parameter_names[-4:] == [
            "ik_metrics_rollout",
            "trajopt_metrics_rollout",
            "create_constrained_rrt",
            "constrained_rrt_planner_config",
        ]

    def test_default_create_constrained_rrt_parameter(self):
        """Create the dedicated constrained RRT by default."""
        import inspect

        sig = inspect.signature(MotionPlannerCfg.create)
        assert sig.parameters["create_constrained_rrt"].default is True

    def test_default_graph_seed_bspline_fit_parameter(self):
        """Fit fixed-horizon graph samples to B-spline knots by default."""
        import inspect

        sig = inspect.signature(MotionPlannerCfg.create)
        assert sig.parameters["fit_graph_seed_to_bspline"].default is True

    def test_default_always_run_trajopt_parameter(self):
        """Preserve TrajOpt refinement unless the validated fast path is enabled."""
        import inspect

        sig = inspect.signature(MotionPlannerCfg.create)
        assert sig.parameters["always_run_trajopt"].default is True

    def test_default_run_trajopt_on_graph_seed_parameter(self):
        """Refine feasible graph paths by default because they can be long."""
        import inspect

        sig = inspect.signature(MotionPlannerCfg.create)
        assert sig.parameters["run_trajopt_on_graph_seed"].default is True

    def test_default_enable_trajopt_parameter(self):
        """Keep trajectory optimization enabled by default."""
        import inspect

        sig = inspect.signature(MotionPlannerCfg.create)
        assert sig.parameters["enable_trajopt"].default is True


class TestMotionPlannerCfgTypeAnnotations:
    """Test MotionPlannerCfg type annotations."""

    def test_ik_solver_config_type(self):
        """Test ik_solver_config type annotation."""
        fields = MotionPlannerCfg.__dataclass_fields__
        # Type stored as string due to from __future__ import annotations
        assert fields["ik_solver_config"].type == "IKSolverCfg"

    def test_trajopt_solver_config_type(self):
        """Test trajopt_solver_config type annotation."""
        fields = MotionPlannerCfg.__dataclass_fields__
        # Type stored as string due to from __future__ import annotations
        assert fields["trajopt_solver_config"].type == "TrajOptSolverCfg"

    def test_device_cfg_type(self):
        """Test device_cfg type annotation."""
        fields = MotionPlannerCfg.__dataclass_fields__
        # Type stored as string due to from __future__ import annotations
        assert fields["device_cfg"].type == "DeviceCfg"


class TestMotionPlannerCfgConfigContents:
    """Test MotionPlannerCfg config contents match expected values."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_config_device_cfg_stored(self, motion_planner_config, cuda_device_cfg):
        """Test device_cfg is stored in config."""
        assert motion_planner_config.device_cfg == cuda_device_cfg

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_ik_config_has_correct_num_seeds(self, motion_planner_config):
        """Test IK config has correct num_seeds from constructor."""
        assert motion_planner_config.ik_solver_config.num_seeds == 16

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_trajopt_config_has_correct_num_seeds(self, motion_planner_config):
        """Test TrajOpt config has correct num_seeds from constructor."""
        assert motion_planner_config.trajopt_solver_config.num_seeds == 2

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_config_without_scene_has_none_scene_collision_cfg(self, motion_planner_config):
        """Test config without scene model has None scene_collision_cfg."""
        assert motion_planner_config.scene_collision_cfg is None
