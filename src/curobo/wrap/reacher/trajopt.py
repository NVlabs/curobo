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


# Standard Library
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

# Third Party
import torch
import torch.autograd.profiler as profiler

# CuRobo
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelState
from curobo.geom.sdf.utils import create_collision_checker
from curobo.geom.sdf.world import CollisionCheckerType, WorldCollision, WorldCollisionConfig
from curobo.geom.types import WorldConfig
from curobo.opt.newton.lbfgs import LBFGSOpt, LBFGSOptConfig
from curobo.opt.newton.newton_base import NewtonOptBase, NewtonOptConfig
from curobo.opt.particle.parallel_es import ParallelES, ParallelESConfig
from curobo.opt.particle.parallel_mppi import ParallelMPPI, ParallelMPPIConfig
from curobo.rollout.arm_reacher import ArmReacher, ArmReacherConfig
from curobo.rollout.cost.pose_cost import PoseCostMetric
from curobo.rollout.dynamics_model.integration_utils import (
    action_interpolate_kernel,
    interpolate_kernel,
)
from curobo.rollout.rollout_base import Goal, RolloutBase, RolloutMetrics
from curobo.types.base import TensorDeviceType
from curobo.types.robot import JointState, RobotConfig
from curobo.types.tensor import T_BDOF, T_DOF, T_BValue_bool, T_BValue_float
from curobo.util.helpers import list_idx_if_not_none
from curobo.util.logger import log_error, log_info, log_warn
from curobo.util.trajectory import (
    InterpolateType,
    calculate_dt_no_clamp,
    get_batch_interpolated_trajectory,
)
from curobo.util_file import get_robot_configs_path, get_task_configs_path, join_path, load_yaml
from curobo.wrap.reacher.evaluator import TrajEvaluator, TrajEvaluatorConfig
from curobo.wrap.reacher.types import ReacherSolveState, ReacherSolveType
from curobo.wrap.wrap_base import WrapBase, WrapConfig, WrapResult


@dataclass
class TrajOptSolverConfig:
    robot_config: RobotConfig
    solver: WrapBase
    rollout_fn: ArmReacher
    position_threshold: float
    rotation_threshold: float
    traj_tsteps: int
    use_cspace_seed: bool = True
    interpolation_type: InterpolateType = InterpolateType.LINEAR_CUDA
    interpolation_steps: int = 1000
    world_coll_checker: Optional[WorldCollision] = None
    seed_ratio: Optional[Dict[str, int]] = None
    num_seeds: int = 1
    bias_node: Optional[T_DOF] = None
    interpolation_dt: float = 0.01
    traj_evaluator_config: TrajEvaluatorConfig = TrajEvaluatorConfig()
    traj_evaluator: Optional[TrajEvaluator] = None
    evaluate_interpolated_trajectory: bool = True
    cspace_threshold: float = 0.1
    tensor_args: TensorDeviceType = TensorDeviceType()
    sync_cuda_time: bool = True
    interpolate_rollout: Optional[ArmReacher] = None
    use_cuda_graph_metrics: bool = False
    trim_steps: Optional[List[int]] = None
    store_debug_in_result: bool = False
    optimize_dt: bool = True
    use_cuda_graph: bool = True

    @staticmethod
    @profiler.record_function("trajopt_config/load_from_robot_config")
    def load_from_robot_config(
        robot_cfg: Union[str, Dict, RobotConfig],
        world_model: Optional[
            Union[Union[List[Dict], List[WorldConfig]], Union[Dict, WorldConfig]]
        ] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        position_threshold: float = 0.005,
        rotation_threshold: float = 0.05,
        cspace_threshold: float = 0.05,
        world_coll_checker=None,
        base_cfg_file: str = "base_cfg.yml",
        particle_file: str = "particle_trajopt.yml",
        gradient_file: str = "gradient_trajopt.yml",
        traj_tsteps: Optional[int] = None,
        interpolation_type: InterpolateType = InterpolateType.LINEAR_CUDA,
        interpolation_steps: int = 10000,
        interpolation_dt: float = 0.01,
        use_cuda_graph: bool = True,
        self_collision_check: bool = False,
        self_collision_opt: bool = True,
        grad_trajopt_iters: Optional[int] = None,
        num_seeds: int = 2,
        seed_ratio: Dict[str, int] = {"linear": 1.0, "bias": 0.0, "start": 0.0, "end": 0.0},
        use_particle_opt: bool = True,
        collision_checker_type: Optional[CollisionCheckerType] = CollisionCheckerType.MESH,
        traj_evaluator_config: TrajEvaluatorConfig = TrajEvaluatorConfig(),
        traj_evaluator: Optional[TrajEvaluator] = None,
        minimize_jerk: bool = True,
        use_gradient_descent: bool = False,
        collision_cache: Optional[Dict[str, int]] = None,
        n_collision_envs: Optional[int] = None,
        use_es: Optional[bool] = None,
        es_learning_rate: Optional[float] = 0.1,
        use_fixed_samples: Optional[bool] = None,
        aux_rollout: Optional[ArmReacher] = None,
        evaluate_interpolated_trajectory: bool = True,
        fixed_iters: Optional[bool] = None,
        store_debug: bool = False,
        sync_cuda_time: bool = True,
        collision_activation_distance: Optional[float] = None,
        trajopt_dt: Optional[float] = None,
        trim_steps: Optional[List[int]] = None,
        store_debug_in_result: bool = False,
        smooth_weight: Optional[List[float]] = None,
        state_finite_difference_mode: Optional[str] = None,
        filter_robot_command: bool = False,
        optimize_dt: bool = True,
        project_pose_to_goal_frame: bool = True,
    ):
        # NOTE: Don't have default optimize_dt, instead read from a configuration file.
        # use default values, disable environment collision checking
        if isinstance(robot_cfg, str):
            robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_cfg))["robot_cfg"]

        if isinstance(robot_cfg, dict):
            robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)

        base_config_data = load_yaml(join_path(get_task_configs_path(), base_cfg_file))
        if collision_cache is not None:
            base_config_data["world_collision_checker_cfg"]["cache"] = collision_cache
        if n_collision_envs is not None:
            base_config_data["world_collision_checker_cfg"]["n_envs"] = n_collision_envs
        if not self_collision_check:
            base_config_data["constraint"]["self_collision_cfg"]["weight"] = 0.0
            self_collision_opt = False
        if not minimize_jerk:
            filter_robot_command = False

        if collision_checker_type is not None:
            base_config_data["world_collision_checker_cfg"]["checker_type"] = collision_checker_type

        if world_coll_checker is None and world_model is not None:
            world_cfg = WorldCollisionConfig.load_from_dict(
                base_config_data["world_collision_checker_cfg"], world_model, tensor_args
            )
            world_coll_checker = create_collision_checker(world_cfg)

        config_data = load_yaml(join_path(get_task_configs_path(), particle_file))
        grad_config_data = load_yaml(join_path(get_task_configs_path(), gradient_file))

        if traj_tsteps is None:
            traj_tsteps = grad_config_data["model"]["horizon"]

        base_config_data["cost"]["pose_cfg"]["project_distance"] = project_pose_to_goal_frame
        base_config_data["convergence"]["pose_cfg"]["project_distance"] = project_pose_to_goal_frame
        config_data["cost"]["pose_cfg"]["project_distance"] = project_pose_to_goal_frame
        grad_config_data["cost"]["pose_cfg"]["project_distance"] = project_pose_to_goal_frame

        base_config_data["cost"]["pose_cfg"]["project_distance"] = project_pose_to_goal_frame
        base_config_data["convergence"]["pose_cfg"]["project_distance"] = project_pose_to_goal_frame
        config_data["cost"]["pose_cfg"]["project_distance"] = project_pose_to_goal_frame
        grad_config_data["cost"]["pose_cfg"]["project_distance"] = project_pose_to_goal_frame

        config_data["model"]["horizon"] = traj_tsteps
        grad_config_data["model"]["horizon"] = traj_tsteps
        if minimize_jerk is not None:
            if not minimize_jerk:
                grad_config_data["cost"]["bound_cfg"]["smooth_weight"][2] = 0.0
                grad_config_data["cost"]["bound_cfg"]["smooth_weight"][1] *= 2.0
                grad_config_data["lbfgs"]["cost_delta_threshold"] = 0.1
            if minimize_jerk and grad_config_data["cost"]["bound_cfg"]["smooth_weight"][2] == 0.0:
                log_warn("minimize_jerk flag is enabled but weight term is zero")

        if state_finite_difference_mode is not None:
            config_data["model"]["state_finite_difference_mode"] = state_finite_difference_mode
            grad_config_data["model"]["state_finite_difference_mode"] = state_finite_difference_mode
        config_data["model"]["filter_robot_command"] = filter_robot_command
        grad_config_data["model"]["filter_robot_command"] = filter_robot_command

        if collision_activation_distance is not None:
            config_data["cost"]["primitive_collision_cfg"][
                "activation_distance"
            ] = collision_activation_distance
            grad_config_data["cost"]["primitive_collision_cfg"][
                "activation_distance"
            ] = collision_activation_distance

        if grad_trajopt_iters is not None:
            grad_config_data["lbfgs"]["n_iters"] = grad_trajopt_iters
            grad_config_data["lbfgs"]["cold_start_n_iters"] = grad_trajopt_iters
        if use_fixed_samples is not None:
            config_data["mppi"]["sample_params"]["fixed_samples"] = use_fixed_samples
        if smooth_weight is not None:
            grad_config_data["cost"]["bound_cfg"]["smooth_weight"][
                : len(smooth_weight)
            ] = smooth_weight  # velocity

        if store_debug:
            use_cuda_graph = False
            fixed_iters = True
            grad_config_data["lbfgs"]["store_debug"] = store_debug
            config_data["mppi"]["store_debug"] = store_debug
            store_debug_in_result = True

        if use_cuda_graph is not None:
            config_data["mppi"]["use_cuda_graph"] = use_cuda_graph
            grad_config_data["lbfgs"]["use_cuda_graph"] = use_cuda_graph
        else:
            use_cuda_graph = grad_config_data["lbfgs"]["use_cuda_graph"]
        if not self_collision_opt:
            config_data["cost"]["self_collision_cfg"]["weight"] = 0.0
            grad_config_data["cost"]["self_collision_cfg"]["weight"] = 0.0
        config_data["mppi"]["n_problems"] = 1
        grad_config_data["lbfgs"]["n_problems"] = 1

        if fixed_iters is not None:
            grad_config_data["lbfgs"]["fixed_iters"] = fixed_iters

        grad_cfg = ArmReacherConfig.from_dict(
            robot_cfg,
            grad_config_data["model"],
            grad_config_data["cost"],
            base_config_data["constraint"],
            base_config_data["convergence"],
            base_config_data["world_collision_checker_cfg"],
            world_model,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
        )

        cfg = ArmReacherConfig.from_dict(
            robot_cfg,
            config_data["model"],
            config_data["cost"],
            base_config_data["constraint"],
            base_config_data["convergence"],
            base_config_data["world_collision_checker_cfg"],
            world_model,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
        )

        safety_robot_model = robot_cfg.kinematics
        safety_robot_cfg = RobotConfig(**vars(robot_cfg))
        safety_robot_cfg.kinematics = safety_robot_model
        safety_cfg = ArmReacherConfig.from_dict(
            safety_robot_cfg,
            config_data["model"],
            config_data["cost"],
            base_config_data["constraint"],
            base_config_data["convergence"],
            base_config_data["world_collision_checker_cfg"],
            world_model,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
        )
        arm_rollout_mppi = None
        with profiler.record_function("trajopt_config/create_rollouts"):
            if use_particle_opt:
                arm_rollout_mppi = ArmReacher(cfg)
            arm_rollout_grad = ArmReacher(grad_cfg)

            arm_rollout_safety = ArmReacher(safety_cfg)
            if aux_rollout is None:
                aux_rollout = ArmReacher(safety_cfg)
            interpolate_rollout = ArmReacher(safety_cfg)
        if trajopt_dt is not None:
            if arm_rollout_mppi is not None:
                arm_rollout_mppi.update_traj_dt(dt=trajopt_dt)
            aux_rollout.update_traj_dt(dt=trajopt_dt)
            arm_rollout_grad.update_traj_dt(dt=trajopt_dt)
            arm_rollout_safety.update_traj_dt(dt=trajopt_dt)
        if arm_rollout_mppi is not None:
            config_dict = ParallelMPPIConfig.create_data_dict(
                config_data["mppi"], arm_rollout_mppi, tensor_args
            )
        parallel_mppi = None
        if use_es is not None and use_es:
            mppi_cfg = ParallelESConfig(**config_dict)
            if es_learning_rate is not None:
                mppi_cfg.learning_rate = es_learning_rate
            parallel_mppi = ParallelES(mppi_cfg)
        elif use_particle_opt:
            mppi_cfg = ParallelMPPIConfig(**config_dict)
            parallel_mppi = ParallelMPPI(mppi_cfg)

        config_dict = LBFGSOptConfig.create_data_dict(
            grad_config_data["lbfgs"], arm_rollout_grad, tensor_args
        )
        lbfgs_cfg = LBFGSOptConfig(**config_dict)

        if use_gradient_descent:
            newton_keys = NewtonOptConfig.__dataclass_fields__.keys()
            newton_d = {}
            lbfgs_k = vars(lbfgs_cfg)
            for k in newton_keys:
                newton_d[k] = lbfgs_k[k]
            newton_d["step_scale"] = 0.9
            newton_cfg = NewtonOptConfig(**newton_d)
            lbfgs = NewtonOptBase(newton_cfg)
        else:
            lbfgs = LBFGSOpt(lbfgs_cfg)
        if use_particle_opt:
            opt_list = [parallel_mppi]
        else:
            opt_list = []
        opt_list.append(lbfgs)
        cfg = WrapConfig(
            safety_rollout=arm_rollout_safety,
            optimizers=opt_list,
            compute_metrics=True,  # not evaluate_interpolated_trajectory,
            use_cuda_graph_metrics=grad_config_data["lbfgs"]["use_cuda_graph"],
            sync_cuda_time=sync_cuda_time,
        )
        trajopt = WrapBase(cfg)
        trajopt_cfg = TrajOptSolverConfig(
            robot_config=robot_cfg,
            rollout_fn=aux_rollout,
            solver=trajopt,
            position_threshold=position_threshold,
            rotation_threshold=rotation_threshold,
            cspace_threshold=cspace_threshold,
            traj_tsteps=traj_tsteps,
            interpolation_steps=interpolation_steps,
            interpolation_dt=interpolation_dt,
            interpolation_type=interpolation_type,
            world_coll_checker=world_coll_checker,
            bias_node=robot_cfg.kinematics.cspace.retract_config,
            seed_ratio=seed_ratio,
            num_seeds=num_seeds,
            traj_evaluator_config=traj_evaluator_config,
            traj_evaluator=traj_evaluator,
            evaluate_interpolated_trajectory=evaluate_interpolated_trajectory,
            tensor_args=tensor_args,
            sync_cuda_time=sync_cuda_time,
            interpolate_rollout=interpolate_rollout,
            use_cuda_graph_metrics=use_cuda_graph,
            trim_steps=trim_steps,
            store_debug_in_result=store_debug_in_result,
            optimize_dt=optimize_dt,
            use_cuda_graph=use_cuda_graph,
        )
        return trajopt_cfg


@dataclass
class TrajResult(Sequence):
    success: T_BValue_bool
    goal: Goal
    solution: JointState
    seed: T_BDOF
    solve_time: float
    debug_info: Optional[Any] = None
    metrics: Optional[RolloutMetrics] = None
    interpolated_solution: Optional[JointState] = None
    path_buffer_last_tstep: Optional[List[int]] = None
    position_error: Optional[T_BValue_float] = None
    rotation_error: Optional[T_BValue_float] = None
    cspace_error: Optional[T_BValue_float] = None
    smooth_error: Optional[T_BValue_float] = None
    smooth_label: Optional[T_BValue_bool] = None
    optimized_dt: Optional[torch.Tensor] = None
    raw_solution: Optional[JointState] = None
    raw_action: Optional[torch.Tensor] = None
    goalset_index: Optional[torch.Tensor] = None

    def __getitem__(self, idx):
        # position_error = rotation_error = cspace_error = path_buffer_last_tstep = None
        # metrics = interpolated_solution = None

        d_list = [
            self.interpolated_solution,
            self.metrics,
            self.path_buffer_last_tstep,
            self.position_error,
            self.rotation_error,
            self.cspace_error,
            self.goalset_index,
        ]
        idx_vals = list_idx_if_not_none(d_list, idx)

        return TrajResult(
            goal=self.goal[idx],
            solution=self.solution[idx],
            success=self.success[idx],
            seed=self.seed[idx],
            debug_info=self.debug_info,
            solve_time=self.solve_time,
            interpolated_solution=idx_vals[0],
            metrics=idx_vals[1],
            path_buffer_last_tstep=idx_vals[2],
            position_error=idx_vals[3],
            rotation_error=idx_vals[4],
            cspace_error=idx_vals[5],
            goalset_index=idx_vals[6],
        )

    def __len__(self):
        return self.success.shape[0]


class TrajOptSolver(TrajOptSolverConfig):
    def __init__(self, config: TrajOptSolverConfig) -> None:
        super().__init__(**vars(config))
        self.dof = self.rollout_fn.d_action
        self.action_horizon = self.rollout_fn.action_horizon
        self.delta_vec = interpolate_kernel(2, self.action_horizon, self.tensor_args).unsqueeze(0)

        self.waypoint_delta_vec = interpolate_kernel(
            3, int(self.action_horizon / 2), self.tensor_args
        ).unsqueeze(0)
        assert self.action_horizon / 2 != 0.0
        self.solver.update_nproblems(self.num_seeds)
        self._max_joint_vel = (
            self.solver.safety_rollout.state_bounds.velocity.view(2, self.dof)[1, :].reshape(
                1, 1, self.dof
            )
        ) - 0.02
        self._max_joint_acc = self.rollout_fn.state_bounds.acceleration[1, :] - 0.02
        self._max_joint_jerk = self.rollout_fn.state_bounds.jerk[1, :] - 0.02
        self._num_seeds = -1
        self._col = None
        if self.traj_evaluator is None:
            self.traj_evaluator = TrajEvaluator(self.traj_evaluator_config)
        self._interpolation_dt_tensor = self.tensor_args.to_device([self.interpolation_dt])
        self._n_seeds = self._get_seed_numbers(self.num_seeds)
        self._goal_buffer = None
        self._solve_state = None
        self._velocity_bounds = self.solver.rollout_fn.state_bounds.velocity[1]
        self._og_newton_iters = self.solver.optimizers[-1].outer_iters
        self._og_newton_fixed_iters = self.solver.optimizers[-1].fixed_iters
        self._interpolated_traj_buffer = None
        self._kin_list = None
        self._rollout_list = None

    def get_all_rollout_instances(self) -> List[RolloutBase]:
        if self._rollout_list is None:
            self._rollout_list = [
                self.rollout_fn,
                self.interpolate_rollout,
            ] + self.solver.get_all_rollout_instances()
        return self._rollout_list

    def get_all_kinematics_instances(self) -> List[CudaRobotModel]:
        if self._kin_list is None:
            self._kin_list = [
                i.dynamics_model.robot_model for i in self.get_all_rollout_instances()
            ]
        return self._kin_list

    def attach_object_to_robot(
        self,
        sphere_radius: float,
        sphere_tensor: Optional[torch.Tensor] = None,
        link_name: str = "attached_object",
    ) -> None:
        for k in self.get_all_kinematics_instances():
            k.attach_object(
                sphere_radius=sphere_radius, sphere_tensor=sphere_tensor, link_name=link_name
            )

    def detach_object_from_robot(self, link_name: str = "attached_object") -> None:
        for k in self.get_all_kinematics_instances():
            k.detach_object(link_name)

    def update_goal_buffer(
        self,
        solve_state: ReacherSolveState,
        goal: Goal,
    ):
        self._solve_state, self._goal_buffer, update_reference = solve_state.update_goal(
            goal,
            self._solve_state,
            self._goal_buffer,
            self.tensor_args,
        )

        if update_reference:
            if self.use_cuda_graph and self._col is not None:
                log_error("changing goal type, breaking previous cuda graph.")
                self.reset_cuda_graph()

            self.solver.update_nproblems(self._solve_state.get_batch_size())
            self._col = torch.arange(
                0, goal.batch, device=self.tensor_args.device, dtype=torch.long
            )
            self.reset_shape()

        return self._goal_buffer

    def solve_any(
        self,
        solve_type: ReacherSolveType,
        goal: Goal,
        seed_traj: Optional[JointState] = None,
        use_nn_seed: bool = False,
        return_all_solutions: bool = False,
        num_seeds: Optional[int] = None,
        seed_success: Optional[torch.Tensor] = None,
        newton_iters: Optional[int] = None,
    ) -> TrajResult:
        if solve_type == ReacherSolveType.SINGLE:
            return self.solve_single(
                goal,
                seed_traj,
                use_nn_seed,
                return_all_solutions,
                num_seeds,
                newton_iters=newton_iters,
            )
        elif solve_type == ReacherSolveType.GOALSET:
            return self.solve_goalset(
                goal,
                seed_traj,
                use_nn_seed,
                return_all_solutions,
                num_seeds,
                newton_iters=newton_iters,
            )
        elif solve_type == ReacherSolveType.BATCH:
            return self.solve_batch(
                goal,
                seed_traj,
                use_nn_seed,
                return_all_solutions,
                num_seeds,
                seed_success,
                newton_iters=newton_iters,
            )
        elif solve_type == ReacherSolveType.BATCH_GOALSET:
            return self.solve_batch_goalset(
                goal,
                seed_traj,
                use_nn_seed,
                return_all_solutions,
                num_seeds,
                seed_success,
                newton_iters=newton_iters,
            )
        elif solve_type == ReacherSolveType.BATCH_ENV:
            return self.solve_batch_env(
                goal,
                seed_traj,
                use_nn_seed,
                return_all_solutions,
                num_seeds,
                seed_success,
                newton_iters=newton_iters,
            )
        elif solve_type == ReacherSolveType.BATCH_ENV_GOALSET:
            return self.solve_batch_env_goalset(
                goal,
                seed_traj,
                use_nn_seed,
                return_all_solutions,
                num_seeds,
                seed_success,
                newton_iters=newton_iters,
            )

    def solve_from_solve_state(
        self,
        solve_state: ReacherSolveState,
        goal: Goal,
        seed_traj: Optional[JointState] = None,
        use_nn_seed: bool = False,
        return_all_solutions: bool = False,
        num_seeds: Optional[int] = None,
        seed_success: Optional[torch.Tensor] = None,
        newton_iters: Optional[int] = None,
    ):
        if solve_state.batch_env:
            if solve_state.batch_size > self.world_coll_checker.n_envs:
                raise ValueError("Batch Env is less that goal batch")
        if newton_iters is not None:
            self.solver.newton_optimizer.outer_iters = newton_iters
            self.solver.newton_optimizer.fixed_iters = True
        goal_buffer = self.update_goal_buffer(solve_state, goal)
        # if self.evaluate_interpolated_trajectory:
        self.interpolate_rollout.update_params(goal_buffer)
        # get seeds:
        seed_traj = self.get_seed_set(
            goal_buffer, seed_traj, seed_success, num_seeds, solve_state.batch_mode
        )

        # remove goal state if goal pose is not None:
        if goal_buffer.goal_pose.position is not None:
            goal_buffer.goal_state = None
        self.solver.reset()
        result = self.solver.solve(goal_buffer, seed_traj)
        log_info("Ran TO")
        traj_result = self._get_result(
            result,
            return_all_solutions,
            goal_buffer,
            seed_traj,
            num_seeds,
            solve_state.batch_mode,
        )
        if traj_result.goalset_index is not None:
            traj_result.goalset_index[traj_result.goalset_index >= goal.goal_pose.n_goalset] = 0
        if newton_iters is not None:
            self.solver.newton_optimizer.outer_iters = self._og_newton_iters
            self.solver.newton_optimizer.fixed_iters = self._og_newton_fixed_iters
        return traj_result

    def solve_single(
        self,
        goal: Goal,
        seed_traj: Optional[JointState] = None,
        use_nn_seed: bool = False,
        return_all_solutions: bool = False,
        num_seeds: Optional[int] = None,
        newton_iters: Optional[int] = None,
    ) -> TrajResult:
        if num_seeds is None:
            num_seeds = self.num_seeds
        solve_state = ReacherSolveState(
            ReacherSolveType.SINGLE,
            num_trajopt_seeds=num_seeds,
            batch_size=1,
            n_envs=1,
            n_goalset=1,
        )

        return self.solve_from_solve_state(
            solve_state,
            goal,
            seed_traj,
            use_nn_seed,
            return_all_solutions,
            num_seeds,
            newton_iters=newton_iters,
        )

    def solve_goalset(
        self,
        goal: Goal,
        seed_traj: Optional[JointState] = None,
        use_nn_seed: bool = False,
        return_all_solutions: bool = False,
        num_seeds: Optional[int] = None,
        newton_iters: Optional[int] = None,
    ) -> TrajResult:
        if num_seeds is None:
            num_seeds = self.num_seeds
        solve_state = ReacherSolveState(
            ReacherSolveType.GOALSET,
            num_trajopt_seeds=num_seeds,
            batch_size=1,
            n_envs=1,
            n_goalset=goal.n_goalset,
        )
        return self.solve_from_solve_state(
            solve_state,
            goal,
            seed_traj,
            use_nn_seed,
            return_all_solutions,
            num_seeds,
            newton_iters=newton_iters,
        )

    def solve_batch(
        self,
        goal: Goal,
        seed_traj: Optional[JointState] = None,
        use_nn_seed: bool = False,
        return_all_solutions: bool = False,
        num_seeds: Optional[int] = None,
        seed_success: Optional[torch.Tensor] = None,
        newton_iters: Optional[int] = None,
    ) -> TrajResult:
        if num_seeds is None:
            num_seeds = self.num_seeds
        solve_state = ReacherSolveState(
            ReacherSolveType.BATCH,
            num_trajopt_seeds=num_seeds,
            batch_size=goal.batch,
            n_envs=1,
            n_goalset=1,
        )
        return self.solve_from_solve_state(
            solve_state,
            goal,
            seed_traj,
            use_nn_seed,
            return_all_solutions,
            num_seeds,
            seed_success,
            newton_iters=newton_iters,
        )

    def solve_batch_goalset(
        self,
        goal: Goal,
        seed_traj: Optional[JointState] = None,
        use_nn_seed: bool = False,
        return_all_solutions: bool = False,
        num_seeds: Optional[int] = None,
        seed_success: Optional[torch.Tensor] = None,
        newton_iters: Optional[int] = None,
    ) -> TrajResult:
        if num_seeds is None:
            num_seeds = self.num_seeds
        solve_state = ReacherSolveState(
            ReacherSolveType.BATCH_GOALSET,
            num_trajopt_seeds=num_seeds,
            batch_size=goal.batch,
            n_envs=1,
            n_goalset=goal.n_goalset,
        )
        return self.solve_from_solve_state(
            solve_state,
            goal,
            seed_traj,
            use_nn_seed,
            return_all_solutions,
            num_seeds,
            newton_iters=newton_iters,
        )

    def solve_batch_env(
        self,
        goal: Goal,
        seed_traj: Optional[JointState] = None,
        use_nn_seed: bool = False,
        return_all_solutions: bool = False,
        num_seeds: Optional[int] = None,
        seed_success: Optional[torch.Tensor] = None,
        newton_iters: Optional[int] = None,
    ) -> TrajResult:
        if num_seeds is None:
            num_seeds = self.num_seeds
        solve_state = ReacherSolveState(
            ReacherSolveType.BATCH_ENV,
            num_trajopt_seeds=num_seeds,
            batch_size=goal.batch,
            n_envs=goal.batch,
            n_goalset=1,
        )
        return self.solve_from_solve_state(
            solve_state,
            goal,
            seed_traj,
            use_nn_seed,
            return_all_solutions,
            num_seeds,
            seed_success,
            newton_iters=newton_iters,
        )

    def solve_batch_env_goalset(
        self,
        goal: Goal,
        seed_traj: Optional[JointState] = None,
        use_nn_seed: bool = False,
        return_all_solutions: bool = False,
        num_seeds: Optional[int] = None,
        seed_success: Optional[torch.Tensor] = None,
        newton_iters: Optional[int] = None,
    ) -> TrajResult:
        if num_seeds is None:
            num_seeds = self.num_seeds
        solve_state = ReacherSolveState(
            ReacherSolveType.BATCH_ENV_GOALSET,
            num_trajopt_seeds=num_seeds,
            batch_size=goal.batch,
            n_envs=goal.batch,
            n_goalset=goal.n_goalset,
        )
        return self.solve_from_solve_state(
            solve_state,
            goal,
            seed_traj,
            use_nn_seed,
            return_all_solutions,
            num_seeds,
            newton_iters=newton_iters,
        )

    def solve(
        self,
        goal: Goal,
        seed_traj: Optional[JointState] = None,
        use_nn_seed: bool = False,
        return_all_solutions: bool = False,
        num_seeds: Optional[int] = None,
        newton_iters: Optional[int] = None,
    ) -> TrajResult:
        """Only for single goal

        Args:
            goal (Goal): _description_
            seed_traj (Optional[JointState], optional): _description_. Defaults to None.
            use_nn_seed (bool, optional): _description_. Defaults to False.

        Raises:
            NotImplementedError: _description_

        Returns:
            TrajResult: _description_
        """
        log_warn("TrajOpt.solve() is deprecated, use TrajOpt.solve_single or others instead")
        if goal.goal_pose.batch == 1 and goal.goal_pose.n_goalset == 1:
            return self.solve_single(
                goal,
                seed_traj,
                use_nn_seed,
                return_all_solutions,
                num_seeds,
                newton_iters=newton_iters,
            )
        if goal.goal_pose.batch == 1 and goal.goal_pose.n_goalset > 1:
            return self.solve_goalset(
                goal,
                seed_traj,
                use_nn_seed,
                return_all_solutions,
                num_seeds,
                newton_iters=newton_iters,
            )
        if goal.goal_pose.batch > 1 and goal.goal_pose.n_goalset == 1:
            return self.solve_batch(
                goal,
                seed_traj,
                use_nn_seed,
                return_all_solutions,
                num_seeds,
                newton_iters=newton_iters,
            )

        raise NotImplementedError()

    @profiler.record_function("trajopt/get_result")
    def _get_result(
        self,
        result: WrapResult,
        return_all_solutions: bool,
        goal: Goal,
        seed_traj: JointState,
        num_seeds: int,
        batch_mode: bool = False,
    ):
        st_time = time.time()
        if self.trim_steps is not None:
            result.action = result.action.trim_trajectory(self.trim_steps[0], self.trim_steps[1])
        interpolated_trajs, last_tstep, opt_dt = self.get_interpolated_trajectory(result.action)
        if self.sync_cuda_time:
            torch.cuda.synchronize()
        interpolation_time = time.time() - st_time
        if self.evaluate_interpolated_trajectory:
            with profiler.record_function("trajopt/evaluate_interpolated"):
                if self.use_cuda_graph_metrics:
                    metrics = self.interpolate_rollout.get_metrics_cuda_graph(interpolated_trajs)
                else:
                    metrics = self.interpolate_rollout.get_metrics(interpolated_trajs)
                result.metrics.feasible = metrics.feasible
                result.metrics.position_error = metrics.position_error
                result.metrics.rotation_error = metrics.rotation_error
                result.metrics.cspace_error = metrics.cspace_error
                result.metrics.goalset_index = metrics.goalset_index

        st_time = time.time()
        feasible = torch.all(result.metrics.feasible, dim=-1)

        if result.metrics.position_error is not None:
            converge = torch.logical_and(
                result.metrics.position_error[..., -1] <= self.position_threshold,
                result.metrics.rotation_error[..., -1] <= self.rotation_threshold,
            )
        elif result.metrics.cspace_error is not None:
            converge = result.metrics.cspace_error[..., -1] <= self.cspace_threshold
        else:
            raise ValueError("convergence check requires either goal_pose or goal_state")

        success = torch.logical_and(feasible, converge)
        if return_all_solutions:
            traj_result = TrajResult(
                success=success,
                goal=goal,
                solution=result.action.scale(self.solver_dt / opt_dt.view(-1, 1, 1)),
                seed=seed_traj,
                position_error=result.metrics.position_error,
                rotation_error=result.metrics.rotation_error,
                solve_time=result.solve_time,
                metrics=result.metrics,  # TODO: index this also
                interpolated_solution=interpolated_trajs,
                debug_info={"solver": result.debug, "interpolation_time": interpolation_time},
                path_buffer_last_tstep=last_tstep,
                cspace_error=result.metrics.cspace_error,
                optimized_dt=opt_dt,
                raw_solution=result.action,
                raw_action=result.raw_action,
                goalset_index=result.metrics.goalset_index,
            )
        else:
            # get path length:
            if self.evaluate_interpolated_trajectory:
                smooth_label, smooth_cost = self.traj_evaluator.evaluate_interpolated_smootheness(
                    interpolated_trajs,
                    opt_dt,
                    self.solver.rollout_fn.dynamics_model.cspace_distance_weight,
                    self._velocity_bounds,
                )

            else:
                smooth_label, smooth_cost = self.traj_evaluator.evaluate(
                    result.action,
                    self.solver.rollout_fn.traj_dt,
                    self.solver.rollout_fn.dynamics_model.cspace_distance_weight,
                    self._velocity_bounds,
                )

            with profiler.record_function("trajopt/best_select"):
                success[~smooth_label] = False
                # get the best solution:
                if result.metrics.pose_error is not None:
                    convergence_error = result.metrics.pose_error[..., -1]
                elif result.metrics.cspace_error is not None:
                    convergence_error = result.metrics.cspace_error[..., -1]
                else:
                    raise ValueError("convergence check requires either goal_pose or goal_state")
                running_cost = torch.mean(result.metrics.cost, dim=-1) * 0.0001
                error = convergence_error + smooth_cost + running_cost
                error[~success] += 10000.0
                if batch_mode:
                    idx = torch.argmin(error.view(goal.batch, num_seeds), dim=-1)
                    idx = idx + num_seeds * self._col
                    last_tstep = [last_tstep[i] for i in idx]
                    success = success[idx]
                else:
                    idx = torch.argmin(error, dim=0)

                    last_tstep = [last_tstep[idx.item()]]
                    success = success[idx : idx + 1]

                best_act_seq = result.action[idx]
                best_raw_action = result.raw_action[idx]
                interpolated_traj = interpolated_trajs[idx]
                goalset_index = position_error = rotation_error = cspace_error = None
                if result.metrics.position_error is not None:
                    position_error = result.metrics.position_error[idx, -1]
                if result.metrics.rotation_error is not None:
                    rotation_error = result.metrics.rotation_error[idx, -1]
                if result.metrics.cspace_error is not None:
                    cspace_error = result.metrics.cspace_error[idx, -1]
                if result.metrics.goalset_index is not None:
                    goalset_index = result.metrics.goalset_index[idx, -1]

                opt_dt = opt_dt[idx]
            if self.sync_cuda_time:
                torch.cuda.synchronize()
            if len(best_act_seq.shape) == 3:
                opt_dt_v = opt_dt.view(-1, 1, 1)
            else:
                opt_dt_v = opt_dt.view(1, 1)
            opt_solution = best_act_seq.scale(self.solver_dt / opt_dt_v)
            select_time = time.time() - st_time
            debug_info = None
            if self.store_debug_in_result:
                debug_info = {
                    "solver": result.debug,
                    "interpolation_time": interpolation_time,
                    "select_time": select_time,
                }

            traj_result = TrajResult(
                success=success,
                goal=goal,
                solution=opt_solution,
                seed=seed_traj,
                position_error=position_error,
                rotation_error=rotation_error,
                cspace_error=cspace_error,
                solve_time=result.solve_time,
                metrics=result.metrics,  # TODO: index this also
                interpolated_solution=interpolated_traj,
                debug_info=debug_info,
                path_buffer_last_tstep=last_tstep,
                smooth_error=smooth_cost,
                smooth_label=smooth_label,
                optimized_dt=opt_dt,
                raw_solution=best_act_seq,
                raw_action=best_raw_action,
                goalset_index=goalset_index,
            )
        return traj_result

    def batch_solve(
        self,
        goal: Goal,
        seed_traj: Optional[JointState] = None,
        seed_success: Optional[torch.Tensor] = None,
        use_nn_seed: bool = False,
        return_all_solutions: bool = False,
        num_seeds: Optional[int] = None,
    ) -> TrajResult:
        """Only for single goal

        Args:
            goal (Goal): _description_
            seed_traj (Optional[JointState], optional): _description_. Defaults to None.
            use_nn_seed (bool, optional): _description_. Defaults to False.

        Raises:
            NotImplementedError: _description_

        Returns:
            TrajResult: _description_
        """
        log_warn("TrajOpt.solve() is deprecated, use TrajOpt.solve_single or others instead")
        if goal.n_goalset == 1:
            return self.solve_batch(
                goal, seed_traj, use_nn_seed, return_all_solutions, num_seeds, seed_success
            )
        if goal.n_goalset > 1:
            return self.solve_batch_goalset(
                goal, seed_traj, use_nn_seed, return_all_solutions, num_seeds, seed_success
            )

    def get_linear_seed(self, start_state, goal_state):
        start_q = start_state.position.view(-1, 1, self.dof)
        end_q = goal_state.position.view(-1, 1, self.dof)
        edges = torch.cat((start_q, end_q), dim=1)

        seed = self.delta_vec @ edges
        return seed

    def get_start_seed(self, start_state):
        start_q = start_state.position.view(-1, 1, self.dof)
        edges = torch.cat((start_q, start_q), dim=1)
        seed = self.delta_vec @ edges
        return seed

    def _get_seed_numbers(self, num_seeds):
        n_seeds = {"linear": 0, "bias": 0, "start": 0, "goal": 0}
        k = n_seeds.keys
        t_seeds = 0
        for k in n_seeds:
            if k not in self.seed_ratio:
                continue
            if self.seed_ratio[k] > 0.0:
                n_seeds[k] = math.floor(self.seed_ratio[k] * num_seeds)
                t_seeds += n_seeds[k]
        if t_seeds < num_seeds:
            n_seeds["linear"] += num_seeds - t_seeds
        return n_seeds

    def get_seed_set(
        self,
        goal: Goal,
        seed_traj: Union[JointState, torch.Tensor, None] = None,  # n_seeds,batch, h, dof
        seed_success: Optional[torch.Tensor] = None,  # batch, n_seeds
        num_seeds: Optional[int] = None,
        batch_mode: bool = False,
    ):
        # if batch_mode:
        total_seeds = goal.batch * num_seeds
        # else:
        #    total_seeds = num_seeds

        if isinstance(seed_traj, JointState):
            seed_traj = seed_traj.position
        if seed_traj is None:
            if goal.goal_state is not None and self.use_cspace_seed:
                # get linear seed
                seed_traj = self.get_seeds(goal.current_state, goal.goal_state, num_seeds=num_seeds)
                # .view(batch_size, self.num_seeds, self.action_horizon, self.dof)
            else:
                # get start repeat seed:
                log_info("No goal state found, using current config to seed")
                seed_traj = self.get_seeds(
                    goal.current_state, goal.current_state, num_seeds=num_seeds
                )
        elif seed_success is not None:
            lin_seed_traj = self.get_seeds(goal.current_state, goal.goal_state, num_seeds=num_seeds)
            lin_seed_traj[seed_success] = seed_traj  # [seed_success]
            seed_traj = lin_seed_traj
            total_seeds = goal.batch * num_seeds
        elif num_seeds > seed_traj.shape[0]:
            new_seeds = self.get_seeds(
                goal.current_state, goal.goal_state, num_seeds - seed_traj.shape[0]
            )
            seed_traj = torch.cat((seed_traj, new_seeds), dim=0)  # n_seed, batch, h, dof

        seed_traj = seed_traj.view(
            total_seeds, self.action_horizon, self.dof
        )  #  n_seeds,batch, h, dof
        return seed_traj

    def get_seeds(self, start_state, goal_state, num_seeds=None):
        # repeat seeds:
        if num_seeds is None:
            num_seeds = self.num_seeds
            n_seeds = self._n_seeds
        else:
            n_seeds = self._get_seed_numbers(num_seeds)
        # linear seed: batch x dof -> batch x n_seeds x dof
        seed_set = []
        if n_seeds["linear"] > 0:
            linear_seed = self.get_linear_seed(start_state, goal_state)

            linear_seeds = linear_seed.view(1, -1, self.action_horizon, self.dof).repeat(
                1, n_seeds["linear"], 1, 1
            )
            seed_set.append(linear_seeds)
        if n_seeds["bias"] > 0:
            bias_seed = self.get_bias_seed(start_state, goal_state)
            bias_seeds = bias_seed.view(1, -1, self.action_horizon, self.dof).repeat(
                1, n_seeds["bias"], 1, 1
            )
            seed_set.append(bias_seeds)
        if n_seeds["start"] > 0:
            bias_seed = self.get_start_seed(start_state)

            bias_seeds = bias_seed.view(1, -1, self.action_horizon, self.dof).repeat(
                1, n_seeds["start"], 1, 1
            )
            seed_set.append(bias_seeds)
        if n_seeds["goal"] > 0:
            bias_seed = self.get_start_seed(goal_state)

            bias_seeds = bias_seed.view(1, -1, self.action_horizon, self.dof).repeat(
                1, n_seeds["goal"], 1, 1
            )
            seed_set.append(bias_seeds)
        all_seeds = torch.cat(
            seed_set, dim=1
        )  # .#transpose(0,1).contiguous()  # n_seed, batch, h, dof

        return all_seeds

    def get_bias_seed(self, start_state, goal_state):
        start_q = start_state.position.view(-1, 1, self.dof)
        end_q = goal_state.position.view(-1, 1, self.dof)

        bias_q = self.bias_node.view(-1, 1, self.dof).repeat(start_q.shape[0], 1, 1)
        edges = torch.cat((start_q, bias_q, end_q), dim=1)
        seed = self.waypoint_delta_vec @ edges
        # print(seed)
        return seed

    @profiler.record_function("trajopt/interpolation")
    def get_interpolated_trajectory(self, traj_state: JointState):
        # do interpolation:
        if (
            self._interpolated_traj_buffer is None
            or traj_state.position.shape[0] != self._interpolated_traj_buffer.position.shape[0]
        ):
            b, _, dof = traj_state.position.shape
            self._interpolated_traj_buffer = JointState.zeros(
                (b, self.interpolation_steps, dof), self.tensor_args
            )
            self._interpolated_traj_buffer.joint_names = self.rollout_fn.joint_names
        state, last_tstep, opt_dt = get_batch_interpolated_trajectory(
            traj_state,
            self.interpolation_dt,
            self._max_joint_vel,
            self._max_joint_acc,
            self._max_joint_jerk,
            self.solver_dt,
            kind=self.interpolation_type,
            tensor_args=self.tensor_args,
            out_traj_state=self._interpolated_traj_buffer,
            min_dt=self.traj_evaluator_config.min_dt,
            optimize_dt=self.optimize_dt,
        )

        return state, last_tstep, opt_dt

    def calculate_trajectory_dt(
        self,
        trajectory: JointState,
    ) -> torch.Tensor:
        opt_dt = calculate_dt_no_clamp(
            trajectory.velocity,
            trajectory.acceleration,
            trajectory.jerk,
            self._max_joint_vel,
            self._max_joint_acc,
            self._max_joint_jerk,
        )
        return opt_dt

    def reset_seed(self):
        self.solver.reset_seed()

    def reset_cuda_graph(self):
        self.solver.reset_cuda_graph()
        self.interpolate_rollout.reset_cuda_graph()
        self.rollout_fn.reset_cuda_graph()

    def reset_shape(self):
        self.solver.reset_shape()
        self.interpolate_rollout.reset_shape()
        self.rollout_fn.reset_shape()

    @property
    def kinematics(self) -> CudaRobotModel:
        return self.rollout_fn.dynamics_model.robot_model

    @property
    def retract_config(self):
        return self.rollout_fn.dynamics_model.retract_config.view(1, -1)

    def fk(self, q: torch.Tensor) -> CudaRobotModelState:
        return self.kinematics.get_state(q)

    @property
    def solver_dt(self):
        return self.solver.safety_rollout.dynamics_model.dt_traj_params.base_dt

    def update_solver_dt(
        self,
        dt: Union[float, torch.Tensor],
        base_dt: Optional[float] = None,
        max_dt: Optional[float] = None,
        base_ratio: Optional[float] = None,
    ):
        all_rollouts = self.get_all_rollout_instances()
        for rollout in all_rollouts:
            rollout.update_traj_dt(dt, base_dt, max_dt, base_ratio)

    def compute_metrics(self, opt_trajectory: bool, interpolated_trajectory: bool):
        self.solver.compute_metrics = opt_trajectory
        self.evaluate_interpolated_trajectory = interpolated_trajectory

    def get_full_js(self, active_js: JointState) -> JointState:
        return self.rollout_fn.get_full_dof_from_solution(active_js)

    def update_pose_cost_metric(
        self,
        metric: PoseCostMetric,
    ):
        rollouts = self.get_all_rollout_instances()
        [
            rollout.update_pose_cost_metric(metric)
            for rollout in rollouts
            if isinstance(rollout, ArmReacher)
        ]
