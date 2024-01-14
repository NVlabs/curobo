from flexiv_controller import FlexivController

from curobo.geom.types import WorldConfig, Cuboid
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.types.base import TensorDeviceType
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.types.robot import JointState, RobotConfig
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.math import Pose
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig

import time
import numpy as np

def get_custom_world_model(table_height=0.02, y_offset=0.54+0.313, up_height=1.35):
    table = Cuboid(
        name="table",
        dims=[4.0,4.0,4.0],
        pose=[0.0, 0.0, -2.0+table_height, 1.0, 0, 0, 0],
        color=[0, 1.0, 0, 1.0],
    )
    wall_l = Cuboid(
        name="wall_l",
        dims=[4,0.04,2*up_height],
        pose=[0.0, y_offset, 0.0, 1, 0, 0, 0],
        color=[0, 1.0, 0, 1.0],
    )
    wall_r = Cuboid(
        name="wall_r",
        dims=[4,0.04,2*up_height],
        pose=[0.0, -y_offset, 0.0, 1, 0, 0, 0],
        color=[0, 1.0, 0, 1.0],
    )
    wall_up = Cuboid(
        name="wall_up",
        dims=[4,4,4],
        pose=[0.0, 0.0, 2.0+up_height, 1, 0, 0, 0],
        color=[0, 1.0, 0, 1.0],
    )
    return WorldConfig(
        cuboid=[table, wall_l,wall_r,wall_up],
    )


class BiFlexivController():
    def __init__(self) -> None:
        self.left_robot = FlexivController(world_model=get_custom_world_model(),
                                      local_ip="192.168.2.223",
                                      robot_ip="192.168.2.100",
                                      origin_offset=[0,0.313,0])
        self.left_robot.init_mpc()
        self.right_robot = FlexivController(world_model=get_custom_world_model(),
                                      local_ip="192.168.2.223",
                                      robot_ip="192.168.2.101",
                                      origin_offset=[0,-0.313,0])
        self.right_robot.init_mpc()
        self.joint_names = [f"joint{i}" for i in range(1,8)]+[f"joint{i}_1" for i in range(1,8)]
        print("current q: ", self.left_robot.get_current_q()+self.right_robot.get_current_q())

        self.tensor_args = TensorDeviceType()
        robot_cfg = load_yaml(join_path(get_robot_configs_path(), 'dual_flexiv.yml'))["robot_cfg"]
        bigger_robot_cfg = load_yaml(join_path(get_robot_configs_path(), 'dual_flexiv_bigger.yml'))["robot_cfg"]
        self.robot_cfg = RobotConfig.from_dict(robot_cfg, self.tensor_args)
        self.bigger_robot_cfg = RobotConfig.from_dict(bigger_robot_cfg, self.tensor_args)

        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.robot_cfg,
            get_custom_world_model(),
            self.tensor_args,
            interpolation_dt=0.02,
        )
        self.motion_gen = MotionGen(self.motion_gen_config)
        self.motion_gen.warmup(enable_graph=False)

        self.mpc_config = MpcSolverConfig.load_from_robot_config(
            self.bigger_robot_cfg,
            get_custom_world_model(),
            use_cuda_graph=True,
            use_cuda_graph_metrics=True,
            use_cuda_graph_full_step=False,
            use_lbfgs=False,
            use_es=False,
            use_mppi=True,
            store_rollouts=True,
            step_dt=0.02,
        )
        self.mpc_init()
        #print(self.mpc.rollout_fn.dynamics_model.retract_config.shape)
        self.left_robot.init_retract(self.mpc.rollout_fn.dynamics_model.retract_config[:7].unsqueeze(0))
        self.right_robot.init_retract(self.mpc.rollout_fn.dynamics_model.retract_config[7:].unsqueeze(0))

        self.birobot_go_home()

        self.ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            get_custom_world_model(),
            rotation_threshold=0.05,
            position_threshold=0.005,
            num_seeds=20,
            self_collision_check=True,
            self_collision_opt=True,
            tensor_args=self.tensor_args,
            use_cuda_graph=True,
            # use_fixed_samples=True,
        )
        self.ik_solver = IKSolver(self.ik_config)
    
    def mpc_init(self):
        self.mpc = MpcSolver(self.mpc_config)
        self.retract_cfg = self.mpc.rollout_fn.dynamics_model.retract_config.unsqueeze(0)
        retract_state = JointState.from_position(self.retract_cfg, joint_names=self.mpc.rollout_fn.joint_names)
        state = self.mpc.rollout_fn.compute_kinematics(retract_state)
        retract_pose = Pose(state.ee_pos_seq, quaternion=state.ee_quat_seq)
        goal = Goal(
            current_state=self.get_current_jointstate(),
            goal_state=self.get_current_jointstate(),
            goal_pose=self.left_robot.get_current_Pose(),
            links_goal_pose={"ee_target_1": self.right_robot.get_current_Pose()},
        )
        self.goal_buffer = self.mpc.setup_solve_single(goal, 1)
        self.mpc.update_goal(self.goal_buffer)
        self.past_pose_l = None
        self.past_rot_l = None
        self.past_pose_r = None
        self.past_rot_r = None
        
    def birobot_go_home(self):
        self.left_robot.tracking_state=False
        self.right_robot.tracking_state=False
        self.left_robot.homing_state=True
        self.right_robot.homing_state=True

        cu_js = self.get_current_jointstate()
        #print(cu_js)
        result = self.motion_gen.plan_single(cu_js.unsqueeze(0),
                                             goal_pose=self.left_robot.retract_pose,
                                            #  Pose(
                                            #      position=self.tensor_args.to_device([0.6,-0.1+0.313,0.4]),
                                            #      quaternion=self.tensor_args.to_device([1,0,0,0]),
                                            #  ),
                                             link_poses={"ee_target_1":self.right_robot.retract_pose},
                                             plan_config=MotionGenPlanConfig(
                                                                                enable_graph=False, 
                                                                                enable_graph_attempt=4, 
                                                                                max_attempts=10, 
                                                                                enable_finetune_trajopt=True
                                                                            ))
        
        if result.success:
            traj = result.get_interpolated_plan()
            cmd_plan = self.motion_gen.get_full_js(traj)
            for i in range(len(cmd_plan.position)):
                cmd_state = cmd_plan[i]
                self.left_robot.move(cmd_state.position.cpu().numpy().flatten()[:7])
                self.right_robot.move(cmd_state.position.cpu().numpy().flatten()[7:])
                time.sleep(0.05)
            
        else:
            print("go home plan failed")
        self.left_robot.homing_state=False
        self.right_robot.homing_state=False
        self.mpc_init()
        
    def mpc_excute(self, left_target:np.ndarray, right_target:np.ndarray):
        target_position_l, target_orientation_l = left_target[:3],left_target[3:]
        target_position_r, target_orientation_r = right_target[:3],right_target[3:]
        if self.past_pose_l is None: self.past_pose_l = target_position_l + 1.0
        if self.past_rot_l is None: self.past_rot_l = target_orientation_l +1.0
        if self.past_pose_r is None: self.past_pose_r = target_position_r + 1.0
        if self.past_rot_r is None: self.past_rot_r = target_orientation_r +1.0

        if (
            np.linalg.norm(target_position_l - self.past_pose_l) > 1e-2 
            or np.linalg.norm(target_orientation_l - self.past_rot_l) > 1e-3
            or np.linalg.norm(target_position_r - self.past_pose_r) > 1e-2 
            or np.linalg.norm(target_orientation_r - self.past_rot_r) > 1e-3
        ):
            link_pose = Pose(
                position=self.tensor_args.to_device(target_position_r),
                quaternion=self.tensor_args.to_device(target_orientation_r),
            )
            self.goal_buffer.links_goal_pose["ee_target_1"].copy_(link_pose)
            ik_goal = Pose(
                position=self.tensor_args.to_device(target_position_l),
                quaternion=self.tensor_args.to_device(target_orientation_l),
            )

            self.goal_buffer.goal_pose.copy_(ik_goal)
            self.mpc.update_goal(self.goal_buffer)
            self.past_pose = target_position_l
            self.past_rot = target_orientation_l
            self.past_pose_r = target_position_r
            self.past_rot_r = target_orientation_r

        mpc_result = self.mpc.step(self.get_current_jointstate(), max_attempts=2)
        state = mpc_result.js_action.position.cpu().numpy().reshape(14)
        self.left_robot.move(state[:7])
        self.right_robot.move(state[7:])

    def get_current_jointstate(self):
        q = self.left_robot.get_current_q() + self.right_robot.get_current_q()
        #print(q)
        return JointState(
            position=self.tensor_args.to_device(q),
            velocity=self.tensor_args.to_device(q) * 0.0,
            acceleration=self.tensor_args.to_device(q) * 0.0,
            jerk=self.tensor_args.to_device(q) * 0.0,
            joint_names=self.joint_names,
        )
    
if __name__ == "__main__":
    bi_con = BiFlexivController()