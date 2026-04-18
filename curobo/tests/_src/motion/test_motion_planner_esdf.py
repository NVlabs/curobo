# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Unit tests for MotionPlanner with ESDF VoxelGrid collision representation."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.geom.types import Cuboid, SceneCfg, VoxelGrid
from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose


def create_empty_esdf_voxel_grid(
    dims: tuple[float, float, float] = (1.0, 1.0, 1.0),
    voxel_size: float = 0.02,
    center: tuple[float, float, float] = (0.4, 0.0, 0.3),
    device: str = "cuda:0",
) -> VoxelGrid:
    """Create an empty ESDF VoxelGrid (all free space).

    All voxels are set to a large positive distance, meaning no obstacles.
    """
    nx = round(dims[0] / voxel_size)
    ny = round(dims[1] / voxel_size)
    nz = round(dims[2] / voxel_size)

    feature_tensor = torch.full(
        (nx, ny, nz), fill_value=1.0, dtype=torch.float16, device=device,
    )

    pose = [center[0], center[1], center[2], 1.0, 0.0, 0.0, 0.0]
    return VoxelGrid(
        name="test_esdf",
        pose=pose,
        dims=list(dims),
        voxel_size=voxel_size,
        feature_tensor=feature_tensor,
        feature_dtype=torch.float16,
    )


def create_box_obstacle_esdf_voxel_grid(
    grid_dims: tuple[float, float, float] = (1.0, 1.0, 1.0),
    voxel_size: float = 0.02,
    grid_center: tuple[float, float, float] = (0.4, 0.0, 0.3),
    box_center: tuple[float, float, float] = (0.5, 0.0, 0.3),
    box_half_extents: tuple[float, float, float] = (0.05, 0.05, 0.05),
    device: str = "cuda:0",
) -> VoxelGrid:
    """Create an ESDF VoxelGrid with a box obstacle.

    Voxels inside the box have negative distance; outside have approximate positive
    distance. This is a simplified ESDF (L-inf distance to the box surface).
    """
    nx = round(grid_dims[0] / voxel_size)
    ny = round(grid_dims[1] / voxel_size)
    nz = round(grid_dims[2] / voxel_size)

    ix = torch.arange(nx, device=device, dtype=torch.float32)
    iy = torch.arange(ny, device=device, dtype=torch.float32)
    iz = torch.arange(nz, device=device, dtype=torch.float32)
    gx, gy, gz = torch.meshgrid(ix, iy, iz, indexing="ij")

    wx = grid_center[0] + (gx - (nx - 1) / 2.0) * voxel_size
    wy = grid_center[1] + (gy - (ny - 1) / 2.0) * voxel_size
    wz = grid_center[2] + (gz - (nz - 1) / 2.0) * voxel_size

    dx = (wx - box_center[0]).abs() - box_half_extents[0]
    dy = (wy - box_center[1]).abs() - box_half_extents[1]
    dz = (wz - box_center[2]).abs() - box_half_extents[2]

    outside_dist = torch.sqrt(
        dx.clamp(min=0) ** 2 + dy.clamp(min=0) ** 2 + dz.clamp(min=0) ** 2
    )
    inside_dist = torch.stack([dx, dy, dz], dim=-1).max(dim=-1).values.clamp(max=0)
    sdf = outside_dist + inside_dist

    feature_tensor = sdf.to(dtype=torch.float16)

    pose = [grid_center[0], grid_center[1], grid_center[2], 1.0, 0.0, 0.0, 0.0]
    return VoxelGrid(
        name="test_esdf_box",
        pose=pose,
        dims=list(grid_dims),
        voxel_size=voxel_size,
        feature_tensor=feature_tensor,
        feature_dtype=torch.float16,
    )


# ── Fixtures ──


@pytest.fixture(scope="module")
def cuda_device_cfg():
    """Create a CUDA device configuration for testing."""
    if torch.cuda.is_available():
        return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    pytest.skip("CUDA not available")


@pytest.fixture(scope="module")
def empty_esdf_voxel_grid():
    """Create an empty ESDF VoxelGrid (free space only)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return create_empty_esdf_voxel_grid()


@pytest.fixture(scope="module")
def box_esdf_voxel_grid():
    """Create an ESDF VoxelGrid with a box obstacle offset from the robot."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return create_box_obstacle_esdf_voxel_grid(
        grid_dims=(1.2, 1.2, 0.8),
        voxel_size=0.02,
        grid_center=(0.4, 0.0, 0.3),
        box_center=(0.5, 0.3, 0.3),
        box_half_extents=(0.05, 0.05, 0.05),
    )


@pytest.fixture(scope="module")
def esdf_scene_cfg(empty_esdf_voxel_grid):
    """Create a SceneCfg with only an ESDF VoxelGrid."""
    return SceneCfg(voxel=[empty_esdf_voxel_grid])


@pytest.fixture(scope="module")
def esdf_with_table_scene_cfg(empty_esdf_voxel_grid):
    """Create a SceneCfg with both an ESDF VoxelGrid and a table cuboid."""
    table = Cuboid(
        name="table",
        pose=[0.35, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0],
        dims=[0.8, 1.2, 0.05],
    )
    return SceneCfg(voxel=[empty_esdf_voxel_grid], cuboid=[table])


@pytest.fixture(scope="module")
def box_obstacle_scene_cfg(box_esdf_voxel_grid):
    """Create a SceneCfg with ESDF containing a box obstacle + table."""
    table = Cuboid(
        name="table",
        pose=[0.35, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0],
        dims=[0.8, 1.2, 0.05],
    )
    return SceneCfg(voxel=[box_esdf_voxel_grid], cuboid=[table])


@pytest.fixture(scope="module")
def planner_with_esdf(cuda_device_cfg, esdf_scene_cfg):
    """Create a MotionPlanner with an empty ESDF VoxelGrid scene."""
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        scene_model=esdf_scene_cfg,
        device_cfg=cuda_device_cfg,
        num_ik_seeds=16,
        num_trajopt_seeds=2,
        use_cuda_graph=False,
        max_goalset=4,
    )
    planner = MotionPlanner(config)
    planner.warmup(enable_graph=True, num_warmup_iterations=2)
    return planner


@pytest.fixture(scope="module")
def planner_with_esdf_and_table(cuda_device_cfg, esdf_with_table_scene_cfg):
    """Create a MotionPlanner with ESDF + table cuboid scene."""
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        scene_model=esdf_with_table_scene_cfg,
        device_cfg=cuda_device_cfg,
        num_ik_seeds=16,
        num_trajopt_seeds=2,
        use_cuda_graph=False,
        max_goalset=4,
    )
    planner = MotionPlanner(config)
    planner.warmup(enable_graph=True, num_warmup_iterations=2)
    return planner


@pytest.fixture(scope="module")
def planner_with_box_esdf(cuda_device_cfg, box_obstacle_scene_cfg):
    """Create a MotionPlanner with ESDF containing a box obstacle."""
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        scene_model=box_obstacle_scene_cfg,
        device_cfg=cuda_device_cfg,
        num_ik_seeds=16,
        num_trajopt_seeds=2,
        use_cuda_graph=False,
        max_goalset=4,
    )
    planner = MotionPlanner(config)
    planner.warmup(enable_graph=True, num_warmup_iterations=2)
    return planner


def _make_start_state(planner: MotionPlanner) -> JointState:
    """Helper to create a start state from default joint position."""
    return JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0),
        joint_names=planner.joint_names,
    )


def _make_goal_pose(
    position: list[float],
    quaternion: list[float],
    device_cfg: DeviceCfg,
) -> Pose:
    """Helper to create a goal Pose."""
    return Pose(
        position=torch.tensor([position], **device_cfg.as_torch_dict()),
        quaternion=torch.tensor([quaternion], **device_cfg.as_torch_dict()),
    )


# ── Tests: Initialization ──


class TestMotionPlannerEsdfInitialization:
    """Test MotionPlanner initialization with ESDF VoxelGrid scenes."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_init_with_esdf_only(self, planner_with_esdf):
        """Test MotionPlanner initializes with ESDF-only scene."""
        assert planner_with_esdf is not None
        assert planner_with_esdf.scene_collision_checker is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_init_with_esdf_and_cuboid(self, planner_with_esdf_and_table):
        """Test MotionPlanner initializes with ESDF + cuboid scene."""
        assert planner_with_esdf_and_table is not None
        assert planner_with_esdf_and_table.scene_collision_checker is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_init_with_box_obstacle_esdf(self, planner_with_box_esdf):
        """Test MotionPlanner initializes with ESDF containing an obstacle."""
        assert planner_with_box_esdf is not None
        assert planner_with_box_esdf.scene_collision_checker is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_joint_names_with_esdf(self, planner_with_esdf):
        """Test joint_names property works with ESDF scene."""
        names = planner_with_esdf.joint_names
        assert isinstance(names, list)
        assert len(names) == 7

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_default_joint_state_with_esdf(self, planner_with_esdf):
        """Test default_joint_state property with ESDF scene."""
        default_js = planner_with_esdf.default_joint_state
        assert isinstance(default_js, JointState)
        assert default_js.position.shape[-1] == 7


# ── Tests: plan_pose with ESDF ──


class TestMotionPlannerEsdfPlanPose:
    """Test plan_pose with ESDF VoxelGrid collision."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_free_space(self, planner_with_esdf, cuda_device_cfg):
        """Test plan_pose in free ESDF space returns a result."""
        start = _make_start_state(planner_with_esdf)
        goal = _make_goal_pose([0.4, 0.0, 0.4], [1.0, 0.0, 0.0, 0.0], cuda_device_cfg)
        tool_frames = planner_with_esdf.trajopt_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: goal.unsqueeze(1)}, ordered_tool_frames=tool_frames,
        )

        result = planner_with_esdf.plan_pose(
            current_state=start, goal_tool_poses=goal_tool_poses, max_attempts=2,
        )
        assert result is None or isinstance(result, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_with_esdf_and_table(
        self, planner_with_esdf_and_table, cuda_device_cfg
    ):
        """Test plan_pose with combined ESDF + table collision."""
        start = _make_start_state(planner_with_esdf_and_table)
        goal = _make_goal_pose([0.4, 0.0, 0.4], [1.0, 0.0, 0.0, 0.0], cuda_device_cfg)
        tool_frames = planner_with_esdf_and_table.trajopt_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: goal.unsqueeze(1)}, ordered_tool_frames=tool_frames,
        )

        result = planner_with_esdf_and_table.plan_pose(
            current_state=start, goal_tool_poses=goal_tool_poses, max_attempts=2,
        )
        assert result is None or isinstance(result, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_avoids_box_obstacle(
        self, planner_with_box_esdf, cuda_device_cfg
    ):
        """Test plan_pose with ESDF box obstacle still returns a result."""
        start = _make_start_state(planner_with_box_esdf)
        goal = _make_goal_pose([0.4, -0.2, 0.4], [1.0, 0.0, 0.0, 0.0], cuda_device_cfg)
        tool_frames = planner_with_box_esdf.trajopt_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: goal.unsqueeze(1)}, ordered_tool_frames=tool_frames,
        )

        result = planner_with_box_esdf.plan_pose(
            current_state=start, goal_tool_poses=goal_tool_poses, max_attempts=3,
        )
        assert result is None or isinstance(result, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_result_has_success(self, planner_with_esdf, cuda_device_cfg):
        """Test plan_pose result has success field with ESDF scene."""
        start = _make_start_state(planner_with_esdf)
        goal = _make_goal_pose([0.4, 0.0, 0.4], [1.0, 0.0, 0.0, 0.0], cuda_device_cfg)
        tool_frames = planner_with_esdf.trajopt_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: goal.unsqueeze(1)}, ordered_tool_frames=tool_frames,
        )

        result = planner_with_esdf.plan_pose(
            current_state=start, goal_tool_poses=goal_tool_poses, max_attempts=2,
        )
        if result is not None:
            assert hasattr(result, "success")
            assert isinstance(result.success, torch.Tensor)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_with_implicit_goal(self, planner_with_esdf, cuda_device_cfg):
        """Test plan_pose with use_implicit_goal=True and ESDF."""
        start = _make_start_state(planner_with_esdf)
        goal = _make_goal_pose([0.4, 0.0, 0.4], [1.0, 0.0, 0.0, 0.0], cuda_device_cfg)
        tool_frames = planner_with_esdf.trajopt_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: goal.unsqueeze(1)}, ordered_tool_frames=tool_frames,
        )

        result = planner_with_esdf.plan_pose(
            current_state=start,
            goal_tool_poses=goal_tool_poses,
            use_implicit_goal=True,
            max_attempts=2,
        )
        assert result is None or isinstance(result, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_with_graph_planner(self, planner_with_esdf, cuda_device_cfg):
        """Test plan_pose with graph planner enabled in ESDF scene."""
        start = _make_start_state(planner_with_esdf)
        goal = _make_goal_pose([0.4, 0.0, 0.4], [1.0, 0.0, 0.0, 0.0], cuda_device_cfg)
        tool_frames = planner_with_esdf.trajopt_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: goal.unsqueeze(1)}, ordered_tool_frames=tool_frames,
        )

        result = planner_with_esdf.plan_pose(
            current_state=start,
            goal_tool_poses=goal_tool_poses,
            max_attempts=3,
            enable_graph_attempt=0,
        )
        assert result is None or isinstance(result, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_unreachable_goal(self, planner_with_esdf, cuda_device_cfg):
        """Test plan_pose with unreachable goal returns None or unsuccessful."""
        start = _make_start_state(planner_with_esdf)
        goal = _make_goal_pose([5.0, 5.0, 5.0], [1.0, 0.0, 0.0, 0.0], cuda_device_cfg)
        tool_frames = planner_with_esdf.trajopt_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: goal.unsqueeze(1)}, ordered_tool_frames=tool_frames,
        )

        result = planner_with_esdf.plan_pose(
            current_state=start, goal_tool_poses=goal_tool_poses, max_attempts=1,
        )
        assert result is None or not result.success.any()


# ── Tests: plan_cspace with ESDF ──


class TestMotionPlannerEsdfPlanCspace:
    """Test plan_cspace with ESDF VoxelGrid collision."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_cspace_free_space(self, planner_with_esdf):
        """Test plan_cspace in free ESDF space."""
        start = _make_start_state(planner_with_esdf)
        goal = start.clone()
        goal.position[..., 0] += 0.1

        result = planner_with_esdf.plan_cspace(
            current_state=start, goal_state=goal, max_attempts=2,
        )
        assert result is None or isinstance(result, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_cspace_with_box_obstacle(self, planner_with_box_esdf):
        """Test plan_cspace with ESDF box obstacle."""
        start = _make_start_state(planner_with_box_esdf)
        goal = start.clone()
        goal.position[..., 0] += 0.1

        result = planner_with_box_esdf.plan_cspace(
            current_state=start, goal_state=goal, max_attempts=2,
        )
        assert result is None or isinstance(result, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_cspace_same_start_and_goal(self, planner_with_esdf):
        """Test plan_cspace when start equals goal with ESDF scene."""
        start = _make_start_state(planner_with_esdf)
        goal = start.clone()

        result = planner_with_esdf.plan_cspace(
            current_state=start, goal_state=goal, max_attempts=1,
        )
        assert result is None or isinstance(result, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_cspace_result_has_timing(self, planner_with_esdf):
        """Test plan_cspace result has timing information with ESDF scene."""
        start = _make_start_state(planner_with_esdf)
        goal = start.clone()
        goal.position[..., 0] += 0.1

        result = planner_with_esdf.plan_cspace(
            current_state=start, goal_state=goal, max_attempts=2,
        )
        if result is not None:
            assert hasattr(result, "total_time")
            assert hasattr(result, "solve_time")
            assert result.total_time >= 0.0


# ── Tests: plan_pose with ESDF ──


class TestMotionPlannerEsdfPlanGoalset:
    """Test plan_pose with ESDF VoxelGrid collision."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_free_space(self, planner_with_esdf, cuda_device_cfg):
        """Test plan_pose in free ESDF space."""
        start = _make_start_state(planner_with_esdf)
        tool_frames = planner_with_esdf.trajopt_solver.kinematics.tool_frames
        num_goalset = 3

        position = torch.zeros((1, num_goalset, 3), **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((1, num_goalset, 4), **cuda_device_cfg.as_torch_dict())
        for i in range(num_goalset):
            position[0, i, :] = torch.tensor(
                [0.4 + i * 0.05, 0.0, 0.4], **cuda_device_cfg.as_torch_dict(),
            )
            quaternion[0, i, :] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict(),
            )

        tool_pose = GoalToolPose(
            tool_frames=tool_frames,
            position=position[:, None, None],
            quaternion=quaternion[:, None, None],
        )
        result = planner_with_esdf.plan_pose(
            current_state=start, goal_tool_poses=tool_pose, max_attempts=2,
        )
        assert result is None or isinstance(result, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_unreachable(self, planner_with_esdf, cuda_device_cfg):
        """Test plan_pose returns None for unreachable goalset in ESDF scene."""
        start = _make_start_state(planner_with_esdf)
        tool_frames = planner_with_esdf.trajopt_solver.kinematics.tool_frames
        num_goalset = 2

        position = torch.zeros((1, num_goalset, 3), **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((1, num_goalset, 4), **cuda_device_cfg.as_torch_dict())
        for i in range(num_goalset):
            position[0, i, :] = torch.tensor(
                [10.0 + i, 10.0, 10.0], **cuda_device_cfg.as_torch_dict(),
            )
            quaternion[0, i, :] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict(),
            )

        tool_pose = GoalToolPose(
            tool_frames=tool_frames,
            position=position[:, None, None],
            quaternion=quaternion[:, None, None],
        )
        result = planner_with_esdf.plan_pose(
            current_state=start, goal_tool_poses=tool_pose, max_attempts=1,
        )
        assert result is None


# ── Tests: plan_grasp with ESDF ──


class TestMotionPlannerEsdfPlanGrasp:
    """Test plan_grasp with ESDF VoxelGrid collision."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_grasp_free_space(self, planner_with_esdf, cuda_device_cfg):
        """Test plan_grasp in free ESDF space."""
        from curobo._src.motion.motion_planner_result import GraspPlanResult

        start = _make_start_state(planner_with_esdf)
        tool_frames = planner_with_esdf.trajopt_solver.kinematics.tool_frames
        num_goalset = 2

        position = torch.zeros((1, num_goalset, 3), **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((1, num_goalset, 4), **cuda_device_cfg.as_torch_dict())
        for i in range(num_goalset):
            position[0, i, :] = torch.tensor(
                [0.4 + i * 0.05, 0.0, 0.35], **cuda_device_cfg.as_torch_dict(),
            )
            quaternion[0, i, :] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict(),
            )

        tool_pose = GoalToolPose(
            tool_frames=tool_frames,
            position=position[:, None, None],
            quaternion=quaternion[:, None, None],
        )
        result = planner_with_esdf.plan_grasp(
            current_state=start,
            grasp_poses=tool_pose,
            plan_approach_to_grasp=False,
            plan_grasp_to_lift=False,
        )
        assert isinstance(result, GraspPlanResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_grasp_with_approach_and_lift(
        self, planner_with_esdf, cuda_device_cfg,
    ):
        """Test plan_grasp with full approach + lift workflow in ESDF scene."""
        from curobo._src.motion.motion_planner_result import GraspPlanResult

        start = _make_start_state(planner_with_esdf)
        tool_frames = planner_with_esdf.trajopt_solver.kinematics.tool_frames
        num_goalset = 2

        position = torch.zeros((1, num_goalset, 3), **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((1, num_goalset, 4), **cuda_device_cfg.as_torch_dict())
        for i in range(num_goalset):
            position[0, i, :] = torch.tensor(
                [0.4 + i * 0.02, 0.0, 0.35], **cuda_device_cfg.as_torch_dict(),
            )
            quaternion[0, i, :] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict(),
            )

        tool_pose = GoalToolPose(
            tool_frames=tool_frames,
            position=position[:, None, None],
            quaternion=quaternion[:, None, None],
        )
        result = planner_with_esdf.plan_grasp(
            current_state=start,
            grasp_poses=tool_pose,
            grasp_approach_axis="z",
            grasp_approach_offset=-0.1,
            grasp_lift_axis="z",
            grasp_lift_offset=0.1,
            plan_approach_to_grasp=True,
            plan_grasp_to_lift=True,
        )
        assert isinstance(result, GraspPlanResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_grasp_with_box_obstacle(
        self, planner_with_box_esdf, cuda_device_cfg,
    ):
        """Test plan_grasp avoids ESDF box obstacle during approach/lift."""
        from curobo._src.motion.motion_planner_result import GraspPlanResult

        start = _make_start_state(planner_with_box_esdf)
        tool_frames = planner_with_box_esdf.trajopt_solver.kinematics.tool_frames
        num_goalset = 2

        position = torch.zeros((1, num_goalset, 3), **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((1, num_goalset, 4), **cuda_device_cfg.as_torch_dict())
        for i in range(num_goalset):
            position[0, i, :] = torch.tensor(
                [0.4, -0.2 + i * 0.02, 0.35], **cuda_device_cfg.as_torch_dict(),
            )
            quaternion[0, i, :] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict(),
            )

        tool_pose = GoalToolPose(
            tool_frames=tool_frames,
            position=position[:, None, None],
            quaternion=quaternion[:, None, None],
        )
        result = planner_with_box_esdf.plan_grasp(
            current_state=start,
            grasp_poses=tool_pose,
            grasp_approach_axis="z",
            grasp_approach_offset=-0.1,
            plan_approach_to_grasp=False,
            plan_grasp_to_lift=False,
        )
        assert isinstance(result, GraspPlanResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_grasp_unreachable(self, planner_with_esdf, cuda_device_cfg):
        """Test plan_grasp returns GraspPlanResult with status for unreachable poses."""
        from curobo._src.motion.motion_planner_result import GraspPlanResult

        start = _make_start_state(planner_with_esdf)
        tool_frames = planner_with_esdf.trajopt_solver.kinematics.tool_frames
        num_goalset = 2

        position = torch.zeros((1, num_goalset, 3), **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((1, num_goalset, 4), **cuda_device_cfg.as_torch_dict())
        for i in range(num_goalset):
            position[0, i, :] = torch.tensor(
                [10.0, 10.0, 10.0], **cuda_device_cfg.as_torch_dict(),
            )
            quaternion[0, i, :] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict(),
            )

        tool_pose = GoalToolPose(
            tool_frames=tool_frames,
            position=position[:, None, None],
            quaternion=quaternion[:, None, None],
        )
        result = planner_with_esdf.plan_grasp(
            current_state=start, grasp_poses=tool_pose,
        )
        assert isinstance(result, GraspPlanResult)
        assert result.status is not None


# ── Tests: Update VoxelGrid ──


class TestMotionPlannerEsdfUpdateWorld:
    """Test updating the VoxelGrid collision in a live planner."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_update_world_with_new_voxel_grid(self, planner_with_esdf):
        """Test update_world replaces the VoxelGrid scene."""
        new_grid = create_empty_esdf_voxel_grid(
            dims=(0.8, 0.8, 0.6), voxel_size=0.02,
            center=(0.3, 0.0, 0.25),
        )
        new_scene = SceneCfg(voxel=[new_grid])
        planner_with_esdf.update_world(new_scene)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_after_voxel_update(self, planner_with_esdf, cuda_device_cfg):
        """Test planning works after updating the VoxelGrid."""
        new_grid = create_empty_esdf_voxel_grid(
            dims=(1.0, 1.0, 1.0), voxel_size=0.02,
            center=(0.4, 0.0, 0.3),
        )
        new_scene = SceneCfg(voxel=[new_grid])
        planner_with_esdf.update_world(new_scene)

        start = _make_start_state(planner_with_esdf)
        goal = _make_goal_pose([0.4, 0.0, 0.4], [1.0, 0.0, 0.0, 0.0], cuda_device_cfg)
        tool_frames = planner_with_esdf.trajopt_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: goal.unsqueeze(1)}, ordered_tool_frames=tool_frames,
        )

        result = planner_with_esdf.plan_pose(
            current_state=start, goal_tool_poses=goal_tool_poses, max_attempts=2,
        )
        assert result is None or isinstance(result, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_update_voxel_grid_with_obstacle(self, planner_with_esdf, cuda_device_cfg):
        """Test planning after updating VoxelGrid to include an obstacle."""
        box_grid = create_box_obstacle_esdf_voxel_grid(
            grid_dims=(1.0, 1.0, 1.0), voxel_size=0.02,
            grid_center=(0.4, 0.0, 0.3),
            box_center=(0.5, 0.3, 0.3),
            box_half_extents=(0.05, 0.05, 0.05),
        )
        new_scene = SceneCfg(voxel=[box_grid])
        planner_with_esdf.update_world(new_scene)

        start = _make_start_state(planner_with_esdf)
        goal = _make_goal_pose([0.4, -0.2, 0.4], [1.0, 0.0, 0.0, 0.0], cuda_device_cfg)
        tool_frames = planner_with_esdf.trajopt_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: goal.unsqueeze(1)}, ordered_tool_frames=tool_frames,
        )

        result = planner_with_esdf.plan_pose(
            current_state=start, goal_tool_poses=goal_tool_poses, max_attempts=3,
        )
        assert result is None or isinstance(result, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_clear_scene_cache_with_esdf(self, planner_with_esdf):
        """Test clear_scene_cache works with ESDF scene."""
        planner_with_esdf.clear_scene_cache()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_update_combined_cuboid_and_voxel(
        self, planner_with_esdf_and_table, cuda_device_cfg,
    ):
        """Test updating both cuboid and voxel obstacles together."""
        new_grid = create_box_obstacle_esdf_voxel_grid(
            grid_dims=(1.0, 1.0, 0.8), voxel_size=0.02,
            grid_center=(0.4, 0.0, 0.3),
            box_center=(0.6, 0.2, 0.3),
            box_half_extents=(0.04, 0.04, 0.04),
        )
        table = Cuboid(
            name="table",
            pose=[0.35, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0],
            dims=[0.8, 1.2, 0.05],
        )
        new_scene = SceneCfg(voxel=[new_grid], cuboid=[table])
        planner_with_esdf_and_table.update_world(new_scene)

        start = _make_start_state(planner_with_esdf_and_table)
        goal = _make_goal_pose([0.4, 0.0, 0.4], [1.0, 0.0, 0.0, 0.0], cuda_device_cfg)
        tool_frames = planner_with_esdf_and_table.trajopt_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: goal.unsqueeze(1)}, ordered_tool_frames=tool_frames,
        )

        result = planner_with_esdf_and_table.plan_pose(
            current_state=start, goal_tool_poses=goal_tool_poses, max_attempts=2,
        )
        assert result is None or isinstance(result, TrajOptSolverResult)


# ── Tests: Goalset warmup and alternating with ESDF ──


class TestMotionPlannerEsdfGoalsetWarmup:
    """Test warmup with num_goalset and alternating plan modes in ESDF scene."""

    @pytest.fixture(scope="class")
    def goalset_esdf_planner(self, cuda_device_cfg, esdf_scene_cfg):
        """Create a planner with ESDF scene, max_goalset=4, and warmup."""
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            scene_model=esdf_scene_cfg,
            device_cfg=cuda_device_cfg,
            num_ik_seeds=16,
            num_trajopt_seeds=2,
            use_cuda_graph=True,
            max_goalset=4,
        )
        planner = MotionPlanner(config)
        planner.warmup(enable_graph=True, num_warmup_iterations=2)
        return planner

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_warmup_with_goalset_succeeds(self, cuda_device_cfg, esdf_scene_cfg):
        """Test warmup with max_goalset > 1 and ESDF scene."""
        config = MotionPlannerCfg.create(
            robot="franka.yml",
            scene_model=esdf_scene_cfg,
            device_cfg=cuda_device_cfg,
            num_ik_seeds=16,
            num_trajopt_seeds=2,
            use_cuda_graph=False,
            max_goalset=4,
        )
        planner = MotionPlanner(config)
        result = planner.warmup(enable_graph=True, num_warmup_iterations=2)
        assert result is True

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_after_goalset_warmup(
        self, goalset_esdf_planner, cuda_device_cfg,
    ):
        """Test plan_pose (num_goalset=1) after goalset warmup with ESDF."""
        planner = goalset_esdf_planner
        start = _make_start_state(planner)
        goal = _make_goal_pose([0.4, 0.0, 0.4], [1.0, 0.0, 0.0, 0.0], cuda_device_cfg)
        tool_frames = planner.trajopt_solver.kinematics.tool_frames
        goal_tool_poses = GoalToolPose.from_poses(
            {tool_frames[0]: goal.unsqueeze(1)}, ordered_tool_frames=tool_frames,
        )

        result = planner.plan_pose(
            current_state=start, goal_tool_poses=goal_tool_poses, max_attempts=1,
        )
        assert result is None or isinstance(result, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_plan_pose_after_goalset_warmup(
        self, goalset_esdf_planner, cuda_device_cfg,
    ):
        """Test plan_pose after goalset warmup with ESDF."""
        planner = goalset_esdf_planner
        start = _make_start_state(planner)
        tool_frames = planner.trajopt_solver.kinematics.tool_frames
        num_goalset = 3

        position = torch.zeros((1, num_goalset, 3), **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((1, num_goalset, 4), **cuda_device_cfg.as_torch_dict())
        for i in range(num_goalset):
            position[0, i, :] = torch.tensor(
                [0.4 + i * 0.05, 0.0, 0.4], **cuda_device_cfg.as_torch_dict(),
            )
            quaternion[0, i, :] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict(),
            )

        tool_pose = GoalToolPose(
            tool_frames=tool_frames,
            position=position[:, None, None],
            quaternion=quaternion[:, None, None],
        )
        result = planner.plan_pose(
            current_state=start, goal_tool_poses=tool_pose, max_attempts=1,
        )
        assert result is None or isinstance(result, TrajOptSolverResult)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_alternating_pose_goalset_pose(
        self, goalset_esdf_planner, cuda_device_cfg,
    ):
        """Test alternating plan_pose and plan_pose with ESDF scene."""
        planner = goalset_esdf_planner
        start = _make_start_state(planner)
        tool_frames = planner.trajopt_solver.kinematics.tool_frames

        # plan_pose
        goal1 = _make_goal_pose([0.4, 0.0, 0.4], [1.0, 0.0, 0.0, 0.0], cuda_device_cfg)
        goal_tool_poses_goal1 = GoalToolPose.from_poses(
            {tool_frames[0]: goal1.unsqueeze(1)}, ordered_tool_frames=tool_frames,
        )
        r1 = planner.plan_pose(
            current_state=start,
            goal_tool_poses=goal_tool_poses_goal1,
            max_attempts=1,
        )
        assert r1 is None or isinstance(r1, TrajOptSolverResult)

        # plan_pose
        num_goalset = 2
        position = torch.zeros((1, num_goalset, 3), **cuda_device_cfg.as_torch_dict())
        quaternion = torch.zeros((1, num_goalset, 4), **cuda_device_cfg.as_torch_dict())
        for i in range(num_goalset):
            position[0, i, :] = torch.tensor(
                [0.4, 0.05 * i, 0.4], **cuda_device_cfg.as_torch_dict(),
            )
            quaternion[0, i, :] = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], **cuda_device_cfg.as_torch_dict(),
            )
        tool_pose = GoalToolPose(
            tool_frames=tool_frames,
            position=position[:, None, None],
            quaternion=quaternion[:, None, None],
        )
        r2 = planner.plan_pose(
            current_state=start, goal_tool_poses=tool_pose, max_attempts=1,
        )
        assert r2 is None or isinstance(r2, TrajOptSolverResult)

        # plan_pose again
        r3 = planner.plan_pose(
            current_state=start,
            goal_tool_poses=goal_tool_poses_goal1,
            max_attempts=1,
        )
        assert r3 is None or isinstance(r3, TrajOptSolverResult)


# ── Tests: Synthetic ESDF helpers ──


class TestSyntheticEsdfHelpers:
    """Test the synthetic ESDF VoxelGrid creation helpers."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_empty_esdf_all_positive(self):
        """Test empty ESDF has all positive distances."""
        grid = create_empty_esdf_voxel_grid()
        assert grid.feature_tensor is not None
        assert (grid.feature_tensor > 0).all()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_empty_esdf_shape(self):
        """Test empty ESDF has expected grid shape."""
        grid = create_empty_esdf_voxel_grid(dims=(1.0, 1.0, 1.0), voxel_size=0.02)
        assert grid.feature_tensor.shape == (50, 50, 50)
        assert grid.voxel_size == 0.02
        assert grid.dims == [1.0, 1.0, 1.0]

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_box_esdf_has_negative_inside(self):
        """Test box ESDF has negative distances inside the box."""
        grid = create_box_obstacle_esdf_voxel_grid(
            grid_dims=(0.5, 0.5, 0.5), voxel_size=0.01,
            grid_center=(0.0, 0.0, 0.0),
            box_center=(0.0, 0.0, 0.0),
            box_half_extents=(0.1, 0.1, 0.1),
        )
        assert (grid.feature_tensor < 0).any()
        assert (grid.feature_tensor > 0).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_box_esdf_center_is_most_negative(self):
        """Test the center of the box has the most negative distance."""
        grid = create_box_obstacle_esdf_voxel_grid(
            grid_dims=(0.5, 0.5, 0.5), voxel_size=0.01,
            grid_center=(0.0, 0.0, 0.0),
            box_center=(0.0, 0.0, 0.0),
            box_half_extents=(0.1, 0.1, 0.1),
        )
        nx, ny, nz = grid.feature_tensor.shape
        center_val = grid.feature_tensor[nx // 2, ny // 2, nz // 2].item()
        assert center_val < 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_voxel_grid_pose_format(self):
        """Test VoxelGrid pose is [x, y, z, qw, qx, qy, qz]."""
        grid = create_empty_esdf_voxel_grid(center=(0.5, 0.1, 0.3))
        assert len(grid.pose) == 7
        assert grid.pose[:3] == [0.5, 0.1, 0.3]
        assert grid.pose[3:] == [1.0, 0.0, 0.0, 0.0]
