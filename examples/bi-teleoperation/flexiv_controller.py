import os
current_path = os.path.dirname(os.path.abspath(__file__))
flexivrdk_root_path = os.path.join(current_path, "flexiv_rdk-main")

import sys
sys.path.insert(0, flexivrdk_root_path+"/lib_py")
import flexivrdk
sys.path.insert(0, flexivrdk_root_path+"/example_py")

import numpy as np
import transforms3d as t3d
import time
from typing import List

from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.geom.types import WorldConfig
from curobo.types.math import Pose

class VirtualFlexivController():
    def __init__(self,world_model:WorldConfig,origin_offset=[0,0.313,0]) -> None:
        self.homing_state=False
        self.tracking_state=False
        self.locked=False
        
        self.origin_offset = np.array(origin_offset).flatten()

        self.tensor_args = TensorDeviceType()
        single_robot_cfg = load_yaml(join_path(get_robot_configs_path(), "flexiv.yml"))["robot_cfg"]
        self.single_robot_cfg = RobotConfig.from_dict(single_robot_cfg, self.tensor_args)
        self.single_mpc_config = MpcSolverConfig.load_from_robot_config(
            self.single_robot_cfg,
            world_model,
            use_cuda_graph=True,
            use_cuda_graph_metrics=True,
            use_cuda_graph_full_step=False,
            use_lbfgs=False,
            use_es=False,
            use_mppi=True,
            store_rollouts=True,
            step_dt=0.02,
        )
        self.single_mpc = MpcSolver(self.single_mpc_config)
        self.single_joint_names = [f"joint{i}" for i in range(1,8)]
        self.retract_cfg = self.single_mpc.rollout_fn.dynamics_model.retract_config.unsqueeze(0)
        self.retract_state = JointState.from_position(self.retract_cfg, 
                                                      joint_names=self.single_mpc.rollout_fn.joint_names)
        state = self.single_mpc.rollout_fn.compute_kinematics(self.retract_state)
        self.retract_pose = Pose(position=state.ee_pos_seq
                                 + self.tensor_args.to_device(self.origin_offset), 
                                 quaternion=state.ee_quat_seq)

class FlexivController():
    def __init__(self,world_model:WorldConfig,local_ip="192.168.2.223",robot_ip="192.168.2.100",origin_offset=[0,0.313,0]) -> None:
        self.homing_state=False
        self.tracking_state=False
        self.locked=False
        self.origin_offset = np.array(origin_offset).flatten()

        self.tensor_args = TensorDeviceType()
        self.world_model = world_model
        

        try:
            self.robot_states = flexivrdk.RobotStates()
            self.log = flexivrdk.Log()
            self.mode = flexivrdk.Mode
            self.robot = flexivrdk.Robot(robot_ip, local_ip)
            self.gripper = flexivrdk.Gripper(self.robot)

            if self.robot.isFault():
                self.log.warn("Fault occurred on robot server, trying to clear ...")
                self.robot.clearFault()
                time.sleep(2)
                # Check again
                if self.robot.isFault():
                    self.log.error("Fault cannot be cleared, exiting ...")
                    return
                self.log.info("Fault on robot server is cleared")
            self.log.info("Enabling left robot ...")
            self.robot.enable()
            seconds_waited = 0
            while not self.robot.isOperational():
                time.sleep(1)
                seconds_waited += 1
                if seconds_waited == 10:
                    self.log.warn(
                        "Still waiting for robot to become operational, please check that the robot 1) "
                        "has no fault, 2) is in [Auto (remote)] mode")
            self.log.info("Left robot is now operational")
            self.robot.setMode(self.mode.NRT_JOINT_POSITION)
        except Exception as e:
            self.log.error("Error occurred while connecting to robot server: %s" % str(e))
            return None
        
        self.start_real_tcp = self.get_current_tcp()
        self.start_unity_tcp = np.zeros(7)
        self.start_unity_tcp[4] = 1
    
    def init_mpc(self,config_name="flexiv.yml"):
        single_robot_cfg = load_yaml(join_path(get_robot_configs_path(), config_name))["robot_cfg"]
        self.single_robot_cfg = RobotConfig.from_dict(single_robot_cfg, self.tensor_args)
        self.single_mpc_config = MpcSolverConfig.load_from_robot_config(
            self.single_robot_cfg,
            self.world_model,
            use_cuda_graph=True,
            use_cuda_graph_metrics=True,
            use_cuda_graph_full_step=False,
            use_lbfgs=False,
            use_es=False,
            use_mppi=True,
            store_rollouts=True,
            step_dt=0.02,
        )
        self.single_mpc = MpcSolver(self.single_mpc_config)
        self.single_joint_names = [f"joint{i}" for i in range(1,8)]

        self.init_retract(self.single_mpc.rollout_fn.dynamics_model.retract_config.unsqueeze(0))

    def init_motion_gen(self,config_name="flexiv.yml"):
        robot_cfg = load_yaml(join_path(get_robot_configs_path(), config_name))["robot_cfg"]
        self.motion_gen_robot_cfg = RobotConfig.from_dict(robot_cfg, self.tensor_args)

        trajopt_tsteps = 36
        trajopt_dt = 0.05
        optimize_dt = False
        max_attempts = 1
        trim_steps = [1, None]

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.motion_gen_robot_cfg,
            self.world_model,
            self.tensor_args,
            #collision_checker_type=CollisionCheckerType.MESH,
            num_trajopt_seeds=12,
            num_graph_seeds=12,
            interpolation_dt=0.05,
            #collision_cache={"obb": n_obstacle_cuboids, "mesh": n_obstacle_mesh},
            optimize_dt=optimize_dt,
            trajopt_dt=trajopt_dt,
            trajopt_tsteps=trajopt_tsteps,
            trim_steps=trim_steps,
        )
        self.motion_gen = MotionGen(motion_gen_config)
        print("motion_gen warming up...")
        self.motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False, parallel_finetune=True)
        print("motion_gen is Ready")

        self.plan_config = MotionGenPlanConfig(
            enable_graph=False,
            enable_graph_attempt=2,
            max_attempts=max_attempts,
            enable_finetune_trajopt=True,
            parallel_finetune=True,
        )
        self.init_retract(self.motion_gen.get_retract_config().view(1, -1))

    def init_retract(self,tensor):
        self.retract_cfg = tensor #self.single_mpc.rollout_fn.dynamics_model.retract_config.unsqueeze(0)#tensor
        #print(self.retract_cfg)
        self.retract_state = JointState.from_position(self.retract_cfg, 
                                                      joint_names=self.single_mpc.rollout_fn.joint_names)
        state = self.single_mpc.rollout_fn.compute_kinematics(self.retract_state)
        new_pos = state.ee_pos_seq.cpu().numpy()+self.origin_offset#+np.array([0,0,-0.04])
        #print(new_pos)
        self.retract_pose = Pose(position=self.tensor_args.to_device(new_pos), 
                                 quaternion=state.ee_quat_seq)

    def get_current_q(self) -> List[float]:
        self.robot.getRobotStates(self.robot_states)
        return self.robot_states.q
    
    def get_current_jointstate(self):
        q = self.get_current_q()
        return JointState.from_position(self.tensor_args.to_device(q), joint_names=self.single_joint_names)

    def get_current_tcp(self) -> np.ndarray:
        state = self.single_mpc.rollout_fn.compute_kinematics(
            self.get_current_jointstate()
        )
        pos = state.ee_pos_seq.cpu().numpy().flatten()
        pos += self.origin_offset
        tcp = np.array(pos.tolist() + state.ee_quat_seq.cpu().numpy().flatten().tolist())
        return tcp
    
    def get_current_Pose(self):
        temp = self.get_current_tcp()
        return Pose(
            position=self.tensor_args.to_device(temp[:3]),
            quaternion=self.tensor_args.to_device(temp[3:]),
        )
        
    def can_move(self):
        return (not self.locked) and (self.homing_state or self.tracking_state)

    def move(self, target_q):
        v = [1.5]*7
        a = [0.8]*7
        if self.can_move():
            self.robot.sendJointPosition(
                    target_q, 
                    [0.0]*7, 
                    [0.0]*7, 
                    v, 
                    a)
    
    def set_start_tcp(self, pos_quat:np.ndarray):
        self.start_real_tcp = self.get_current_tcp()
        self.start_unity_tcp = pos_quat
        self.tracking_state=True

    def get_relative_target(self, pos_from_unity):

        target=np.zeros(7)
        target[:3]=pos_from_unity[:3] - self.start_unity_tcp[:3] + self.start_real_tcp[:3]
        target_rot_mat = t3d.quaternions.quat2mat(pos_from_unity[3:]) \
                        @ np.linalg.inv(t3d.quaternions.quat2mat(self.start_unity_tcp[3:])) \
                        @ t3d.quaternions.quat2mat(self.start_real_tcp[3:])
        target[3:]=t3d.quaternions.mat2quat(target_rot_mat).tolist()

        return target
    
    def robot_go_home(self):
        self.tracking_state=False
        self.homing_state=True
        cu_js = self.get_current_jointstate()
        result = self.motion_gen.plan_single(cu_js.unsqueeze(0),
                                             goal_pose=self.retract_pose,
                                             plan_config=self.plan_config)
        if result.success:
            traj = result.get_interpolated_plan()
            cmd_plan = self.motion_gen.get_full_js(traj)
            for i in range(len(cmd_plan.position)):
                cmd_state = cmd_plan[i]
                self.move(cmd_state.position.cpu().numpy().flatten()[:7])
                time.sleep(0.05)
        else:
            print("go home plan failed")
        self.homing_state=False
