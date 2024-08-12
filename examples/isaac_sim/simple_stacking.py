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


try:
    # Third Party
    import isaacsim
except ImportError:
    pass


# Third Party
import torch

a = torch.zeros(
    4, device="cuda:0"
)  # this is necessary to allow isaac sim to use this torch instance
# Third Party
import numpy as np

np.set_printoptions(suppress=True)
# Standard Library

# Standard Library
import argparse

## import curobo:

parser = argparse.ArgumentParser()

parser.add_argument(
    "--headless_mode",
    type=str,
    default=None,
    help="To run headless, use one of [native, websocket], webrtc might not work.",
)

parser.add_argument(
    "--constrain_grasp_approach",
    action="store_true",
    help="When True, approaches grasp with fixed orientation and motion only along z axis.",
    default=False,
)
args = parser.parse_args()

# Third Party
from omni.isaac.kit import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": args.headless_mode is not None,
        "width": "1920",
        "height": "1080",
    }
)
# Standard Library
from typing import Optional

# Third Party
import carb
from helper import add_extensions
from omni.isaac.core import World
from omni.isaac.core.controllers import BaseController
from omni.isaac.core.tasks import Stacking as BaseStacking
from omni.isaac.core.utils.prims import is_prim_path_valid
from omni.isaac.core.utils.stage import get_stage_units
from omni.isaac.core.utils.string import find_unique_string_name
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.franka import Franka

# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import WorldConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.types.state import JointState
from curobo.util.usd_helper import UsdHelper
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    MotionGenResult,
    PoseCostMetric,
)


class CuroboController(BaseController):
    def __init__(
        self,
        my_world: World,
        my_task: BaseStacking,
        name: str = "curobo_controller",
        constrain_grasp_approach: bool = False,
    ) -> None:
        BaseController.__init__(self, name=name)
        self._save_log = False
        self.my_world = my_world
        self.my_task = my_task
        self._step_idx = 0
        n_obstacle_cuboids = 20
        n_obstacle_mesh = 2
        # warmup curobo instance
        self.usd_help = UsdHelper()
        self.init_curobo = False
        self.world_file = "collision_table.yml"
        self.cmd_js_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]
        self.tensor_args = TensorDeviceType()
        self.robot_cfg = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))["robot_cfg"]
        self.robot_cfg["kinematics"][
            "base_link"
        ] = "panda_link0"  # controls which frame the controller is controlling

        self.robot_cfg["kinematics"][
            "ee_link"
        ] = "panda_hand"  # controls which frame the controller is controlling
        # self.robot_cfg["kinematics"]["cspace"]["max_acceleration"] = 10.0 # controls how fast robot moves
        self.robot_cfg["kinematics"]["extra_collision_spheres"] = {"attached_object": 100}
        # @self.robot_cfg["kinematics"]["collision_sphere_buffer"] = 0.0
        self.robot_cfg["kinematics"]["collision_spheres"] = "spheres/franka_collision_mesh.yml"

        world_cfg_table = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
        )
        self._world_cfg_table = world_cfg_table

        world_cfg1 = WorldConfig.from_dict(
            load_yaml(join_path(get_world_configs_path(), "collision_table.yml"))
        ).get_mesh_world()
        world_cfg1.mesh[0].pose[2] = -10.5

        self._world_cfg = WorldConfig(cuboid=world_cfg_table.cuboid, mesh=world_cfg1.mesh)

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            self._world_cfg,
            self.tensor_args,
            trajopt_tsteps=32,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=True,
            interpolation_dt=0.01,
            collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
            store_ik_debug=self._save_log,
            store_trajopt_debug=self._save_log,
        )
        self.motion_gen = MotionGen(motion_gen_config)
        print("warming up...")
        self.motion_gen.warmup(parallel_finetune=True)
        pose_metric = None
        if constrain_grasp_approach:
            pose_metric = PoseCostMetric.create_grasp_approach_metric(
                offset_position=0.1, tstep_fraction=0.8
            )

        self.plan_config = MotionGenPlanConfig(
            enable_graph=False,
            max_attempts=10,
            enable_graph_attempt=None,
            enable_finetune_trajopt=True,
            partial_ik_opt=False,
            parallel_finetune=True,
            pose_cost_metric=pose_metric,
            time_dilation_factor=0.75,
        )
        self.usd_help.load_stage(self.my_world.stage)
        self.cmd_plan = None
        self.cmd_idx = 0
        self._step_idx = 0
        self.idx_list = None

    def attach_obj(
        self,
        sim_js: JointState,
        js_names: list,
    ) -> None:
        cube_name = self.my_task.get_cube_prim(self.my_task.target_cube)

        cu_js = JointState(
            position=self.tensor_args.to_device(sim_js.positions),
            velocity=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=js_names,
        )

        self.motion_gen.attach_objects_to_robot(
            cu_js,
            [cube_name],
            sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
            world_objects_pose_offset=Pose.from_list([0, 0, 0.01, 1, 0, 0, 0], self.tensor_args),
        )

    def detach_obj(self) -> None:
        self.motion_gen.detach_object_from_robot()

    def plan(
        self,
        ee_translation_goal: np.array,
        ee_orientation_goal: np.array,
        sim_js: JointState,
        js_names: list,
    ) -> MotionGenResult:
        ik_goal = Pose(
            position=self.tensor_args.to_device(ee_translation_goal),
            quaternion=self.tensor_args.to_device(ee_orientation_goal),
        )
        cu_js = JointState(
            position=self.tensor_args.to_device(sim_js.positions),
            velocity=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            acceleration=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            jerk=self.tensor_args.to_device(sim_js.velocities) * 0.0,
            joint_names=js_names,
        )
        cu_js = cu_js.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)
        result = self.motion_gen.plan_single(cu_js.unsqueeze(0), ik_goal, self.plan_config.clone())
        if self._save_log:  # and not result.success.item(): # logging for debugging
            UsdHelper.write_motion_gen_log(
                result,
                {"robot_cfg": self.robot_cfg},
                self._world_cfg,
                cu_js,
                ik_goal,
                join_path("log/usd/", "cube") + "_debug",
                write_ik=False,
                write_trajopt=True,
                visualize_robot_spheres=True,
                link_spheres=self.motion_gen.kinematics.kinematics_config.link_spheres,
                grid_space=2,
                write_robot_usd_path="log/usd/assets",
            )
        return result

    def forward(
        self,
        sim_js: JointState,
        js_names: list,
    ) -> ArticulationAction:
        assert self.my_task.target_position is not None
        assert self.my_task.target_cube is not None

        if self.cmd_plan is None:
            self.cmd_idx = 0
            self._step_idx = 0
            # Set EE goals
            ee_translation_goal = self.my_task.target_position
            ee_orientation_goal = np.array([0, 0, -1, 0])
            # compute curobo solution:
            result = self.plan(ee_translation_goal, ee_orientation_goal, sim_js, js_names)
            succ = result.success.item()
            if succ:
                cmd_plan = result.get_interpolated_plan()
                self.idx_list = [i for i in range(len(self.cmd_js_names))]
                self.cmd_plan = cmd_plan.get_ordered_joint_state(self.cmd_js_names)
            else:
                carb.log_warn("Plan did not converge to a solution.")
                return None
        if self._step_idx % 3 == 0:
            cmd_state = self.cmd_plan[self.cmd_idx]
            self.cmd_idx += 1

            # get full dof state
            art_action = ArticulationAction(
                cmd_state.position.cpu().numpy(),
                cmd_state.velocity.cpu().numpy() * 0.0,
                joint_indices=self.idx_list,
            )
            if self.cmd_idx >= len(self.cmd_plan.position):
                self.cmd_idx = 0
                self.cmd_plan = None
        else:
            art_action = None
        self._step_idx += 1
        return art_action

    def reached_target(self, observations: dict) -> bool:
        curr_ee_position = observations["my_franka"]["end_effector_position"]
        if np.linalg.norm(
            self.my_task.target_position - curr_ee_position
        ) < 0.04 and (  # This is half gripper width, curobo succ threshold is 0.5 cm
            self.cmd_plan is None
        ):
            if self.my_task.cube_in_hand is None:
                print("reached picking target: ", self.my_task.target_cube)
            else:
                print("reached placing target: ", self.my_task.target_cube)
            return True
        else:
            return False

    def reset(
        self,
        ignore_substring: str,
        robot_prim_path: str,
    ) -> None:
        # init
        self.update(ignore_substring, robot_prim_path)
        self.init_curobo = True
        self.cmd_plan = None
        self.cmd_idx = 0

    def update(
        self,
        ignore_substring: str,
        robot_prim_path: str,
    ) -> None:
        # print("updating world...")
        obstacles = self.usd_help.get_obstacles_from_stage(
            ignore_substring=ignore_substring, reference_prim_path=robot_prim_path
        ).get_collision_check_world()
        # add ground plane as it's not readable:
        obstacles.add_obstacle(self._world_cfg_table.cuboid[0])
        self.motion_gen.update_world(obstacles)
        self._world_cfg = obstacles


class MultiModalStacking(BaseStacking):
    def __init__(
        self,
        name: str = "multi_modal_stacking",
        offset: Optional[np.ndarray] = None,
    ) -> None:
        BaseStacking.__init__(
            self,
            name=name,
            cube_initial_positions=np.array(
                [
                    [0.50, 0.0, 0.1],
                    [0.50, -0.20, 0.1],
                    [0.50, 0.20, 0.1],
                    [0.30, -0.20, 0.1],
                    [0.30, 0.0, 0.1],
                    [0.30, 0.20, 0.1],
                    [0.70, -0.20, 0.1],
                    [0.70, 0.0, 0.1],
                    [0.70, 0.20, 0.1],
                ]
            )
            / get_stage_units(),
            cube_initial_orientations=None,
            stack_target_position=None,
            cube_size=np.array([0.045, 0.045, 0.07]),
            offset=offset,
        )
        self.cube_list = None
        self.target_position = None
        self.target_cube = None
        self.cube_in_hand = None

    def reset(self) -> None:
        self.cube_list = self.get_cube_names()
        self.target_position = None
        self.target_cube = None
        self.cube_in_hand = None

    def update_task(self) -> bool:
        # after detaching the cube in hand
        assert self.target_cube is not None
        assert self.cube_in_hand is not None
        self.cube_list.insert(0, self.cube_in_hand)
        self.target_cube = None
        self.target_position = None
        self.cube_in_hand = None
        if len(self.cube_list) <= 1:
            task_finished = True
        else:
            task_finished = False
        return task_finished

    def get_cube_prim(self, cube_name: str):
        for i in range(self._num_of_cubes):
            if cube_name == self._cubes[i].name:
                return self._cubes[i].prim_path

    def get_place_position(self, observations: dict) -> None:
        assert self.target_cube is not None
        self.cube_in_hand = self.target_cube
        self.target_cube = self.cube_list[0]
        ee_to_grasped_cube = (
            observations["my_franka"]["end_effector_position"][2]
            - observations[self.cube_in_hand]["position"][2]
        )
        self.target_position = observations[self.target_cube]["position"] + [
            0,
            0,
            self._cube_size[2] + ee_to_grasped_cube + 0.02,
        ]
        self.cube_list.remove(self.target_cube)

    def get_pick_position(self, observations: dict) -> None:
        assert self.cube_in_hand is None
        self.target_cube = self.cube_list[1]
        self.target_position = observations[self.target_cube]["position"] + [
            0,
            0,
            self._cube_size[2] / 2 + 0.092,
        ]
        self.cube_list.remove(self.target_cube)

    def set_robot(self) -> Franka:
        franka_prim_path = find_unique_string_name(
            initial_name="/World/Franka", is_unique_fn=lambda x: not is_prim_path_valid(x)
        )
        franka_robot_name = find_unique_string_name(
            initial_name="my_franka", is_unique_fn=lambda x: not self.scene.object_exists(x)
        )
        return Franka(
            prim_path=franka_prim_path, name=franka_robot_name, end_effector_prim_name="panda_hand"
        )


robot_prim_path = "/World/Franka/panda_link0"
ignore_substring = ["Franka", "TargetCube", "material", "Plane"]
my_world = World(stage_units_in_meters=1.0)
stage = my_world.stage
stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

my_task = MultiModalStacking()
my_world.add_task(my_task)
my_world.reset()
robot_name = my_task.get_params()["robot_name"]["value"]
my_franka = my_world.scene.get_object(robot_name)
my_controller = CuroboController(
    my_world=my_world, my_task=my_task, constrain_grasp_approach=args.constrain_grasp_approach
)
articulation_controller = my_franka.get_articulation_controller()
set_camera_view(eye=[2, 0, 1], target=[0.00, 0.00, 0.00], camera_prim_path="/OmniverseKit_Persp")
wait_steps = 8

my_franka.set_solver_velocity_iteration_count(4)
my_franka.set_solver_position_iteration_count(124)
my_world._physics_context.set_solver_type("TGS")
initial_steps = 100
################################################################
print("Start simulation...")
robot = my_franka
print(
    my_world._physics_context.get_solver_type(),
    robot.get_solver_position_iteration_count(),
    robot.get_solver_velocity_iteration_count(),
)
print(my_world._physics_context.use_gpu_pipeline)
print(articulation_controller.get_gains())
print(articulation_controller.get_max_efforts())
robot = my_franka
print("**********************")
if False:
    robot.enable_gravity()
    articulation_controller.set_gains(
        kps=np.array(
            [100000000, 6000000.0, 10000000, 600000.0, 25000.0, 15000.0, 50000.0, 6000.0, 6000.0]
        )
    )

    articulation_controller.set_max_efforts(
        values=np.array([100000, 52.199997, 100000, 52.199997, 7.2, 7.2, 7.2, 50.0, 50])
    )

print("Updated gains:")
print(articulation_controller.get_gains())
print(articulation_controller.get_max_efforts())
# exit()
my_franka.gripper.open()
for _ in range(wait_steps):
    my_world.step(render=True)
my_task.reset()
task_finished = False
observations = my_world.get_observations()
my_task.get_pick_position(observations)

i = 0

add_extensions(simulation_app, args.headless_mode)

while simulation_app.is_running():
    my_world.step(render=True)  # necessary to visualize changes
    i += 1

    if task_finished or i < initial_steps:
        continue

    if not my_controller.init_curobo:
        my_controller.reset(ignore_substring, robot_prim_path)

    step_index = my_world.current_time_step_index
    observations = my_world.get_observations()
    sim_js = my_franka.get_joints_state()

    if my_controller.reached_target(observations):
        if my_franka.gripper.get_joint_positions()[0] < 0.035:  # reached placing target
            my_franka.gripper.open()
            for _ in range(wait_steps):
                my_world.step(render=True)
            my_controller.detach_obj()
            my_controller.update(
                ignore_substring, robot_prim_path
            )  # update world collision configuration
            task_finished = my_task.update_task()
            if task_finished:
                print("\nTASK DONE\n")
                for _ in range(wait_steps):
                    my_world.step(render=True)
                continue
            else:
                my_task.get_pick_position(observations)

        else:  # reached picking target
            my_franka.gripper.close()
            for _ in range(wait_steps):
                my_world.step(render=True)
            sim_js = my_franka.get_joints_state()
            my_controller.update(ignore_substring, robot_prim_path)
            my_controller.attach_obj(sim_js, my_franka.dof_names)
            my_task.get_place_position(observations)

    else:  # target position has been set
        sim_js = my_franka.get_joints_state()
        art_action = my_controller.forward(sim_js, my_franka.dof_names)
        if art_action is not None:
            articulation_controller.apply_action(art_action)
            # for _ in range(2):
            #    my_world.step(render=False)

simulation_app.close()
