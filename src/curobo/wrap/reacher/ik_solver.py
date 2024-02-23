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
from curobo.rollout.rollout_base import Goal, RolloutBase, RolloutMetrics
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.types.tensor import T_BDOF, T_BValue_bool, T_BValue_float
from curobo.util.logger import log_error, log_warn
from curobo.util.sample_lib import HaltonGenerator
from curobo.util_file import (
    get_robot_configs_path,
    get_task_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.types import ReacherSolveState, ReacherSolveType
from curobo.wrap.wrap_base import WrapBase, WrapConfig, WrapResult


@dataclass
class IKSolverConfig:
    robot_config: RobotConfig
    solver: WrapBase
    num_seeds: int
    position_threshold: float
    rotation_threshold: float
    rollout_fn: ArmReacher
    ik_nn_seeder: Optional[str] = None
    world_coll_checker: Optional[WorldCollision] = None
    sample_rejection_ratio: int = 50
    tensor_args: TensorDeviceType = TensorDeviceType()
    use_cuda_graph: bool = True

    @staticmethod
    @profiler.record_function("ik_solver/load_from_robot_config")
    def load_from_robot_config(
        robot_cfg: Union[str, Dict, RobotConfig],
        world_model: Optional[
            Union[Union[List[Dict], List[WorldConfig]], Union[Dict, WorldConfig]]
        ] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        num_seeds: int = 100,
        position_threshold: float = 0.005,
        rotation_threshold: float = 0.05,
        world_coll_checker=None,
        base_cfg_file: str = "base_cfg.yml",
        particle_file: str = "particle_ik.yml",
        gradient_file: str = "gradient_ik.yml",
        use_cuda_graph: bool = True,
        self_collision_check: bool = True,
        self_collision_opt: bool = True,
        grad_iters: Optional[int] = None,
        use_particle_opt: bool = True,
        collision_checker_type: Optional[CollisionCheckerType] = CollisionCheckerType.MESH,
        sync_cuda_time: Optional[bool] = None,
        use_gradient_descent: bool = False,
        collision_cache: Optional[Dict[str, int]] = None,
        n_collision_envs: Optional[int] = None,
        ee_link_name: Optional[str] = None,
        use_es: Optional[bool] = None,
        es_learning_rate: Optional[float] = 0.1,
        use_fixed_samples: Optional[bool] = None,
        store_debug: bool = False,
        regularization: bool = True,
        collision_activation_distance: Optional[float] = None,
        high_precision: bool = False,
        project_pose_to_goal_frame: bool = True,
    ):
        if position_threshold <= 0.001:
            high_precision = True
        if high_precision:
            if grad_iters is None:
                grad_iters = 200
        # use default values, disable environment collision checking
        base_config_data = load_yaml(join_path(get_task_configs_path(), base_cfg_file))
        config_data = load_yaml(join_path(get_task_configs_path(), particle_file))
        grad_config_data = load_yaml(join_path(get_task_configs_path(), gradient_file))

        if collision_cache is not None:
            base_config_data["world_collision_checker_cfg"]["cache"] = collision_cache
        if n_collision_envs is not None:
            base_config_data["world_collision_checker_cfg"]["n_envs"] = n_collision_envs

        if collision_checker_type is not None:
            base_config_data["world_collision_checker_cfg"]["checker_type"] = collision_checker_type
        if not self_collision_check:
            base_config_data["constraint"]["self_collision_cfg"]["weight"] = 0.0
            self_collision_opt = False
        if not regularization:
            base_config_data["convergence"]["null_space_cfg"]["weight"] = 0.0
            base_config_data["convergence"]["cspace_cfg"]["weight"] = 0.0
            config_data["cost"]["bound_cfg"]["null_space_weight"] = 0.0
            grad_config_data["cost"]["bound_cfg"]["null_space_weight"] = 0.0

        if isinstance(robot_cfg, str):
            robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_cfg))["robot_cfg"]
        if ee_link_name is not None:
            if isinstance(robot_cfg, RobotConfig):
                raise NotImplementedError("ee link cannot be changed after creating RobotConfig")

                robot_cfg.kinematics.ee_link = ee_link_name
            else:
                robot_cfg["kinematics"]["ee_link"] = ee_link_name
        if isinstance(robot_cfg, dict):
            robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)

        if isinstance(world_model, str):
            world_model = load_yaml(join_path(get_world_configs_path(), world_model))
        if world_coll_checker is None and world_model is not None:
            world_cfg = WorldCollisionConfig.load_from_dict(
                base_config_data["world_collision_checker_cfg"], world_model, tensor_args
            )
            world_coll_checker = create_collision_checker(world_cfg)

        if collision_activation_distance is not None:
            config_data["cost"]["primitive_collision_cfg"][
                "activation_distance"
            ] = collision_activation_distance
            grad_config_data["cost"]["primitive_collision_cfg"][
                "activation_distance"
            ] = collision_activation_distance

        if store_debug:
            use_cuda_graph = False
            grad_config_data["lbfgs"]["store_debug"] = store_debug
            config_data["mppi"]["store_debug"] = store_debug
            grad_config_data["lbfgs"]["inner_iters"] = 1
        if use_cuda_graph is not None:
            config_data["mppi"]["use_cuda_graph"] = use_cuda_graph
            grad_config_data["lbfgs"]["use_cuda_graph"] = use_cuda_graph
        if use_fixed_samples is not None:
            config_data["mppi"]["sample_params"]["fixed_samples"] = use_fixed_samples

        if not self_collision_opt:
            config_data["cost"]["self_collision_cfg"]["weight"] = 0.0
            grad_config_data["cost"]["self_collision_cfg"]["weight"] = 0.0
        if grad_iters is not None:
            grad_config_data["lbfgs"]["n_iters"] = grad_iters
        config_data["mppi"]["n_problems"] = 1
        grad_config_data["lbfgs"]["n_problems"] = 1
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

        arm_rollout_mppi = ArmReacher(cfg)
        arm_rollout_grad = ArmReacher(grad_cfg)
        arm_rollout_safety = ArmReacher(grad_cfg)
        aux_rollout = ArmReacher(grad_cfg)

        config_dict = ParallelMPPIConfig.create_data_dict(
            config_data["mppi"], arm_rollout_mppi, tensor_args
        )
        if use_es is not None and use_es:
            mppi_cfg = ParallelESConfig(**config_dict)
            if es_learning_rate is not None:
                mppi_cfg.learning_rate = es_learning_rate
            parallel_mppi = ParallelES(mppi_cfg)
        else:
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
            opts = [parallel_mppi]
        else:
            opts = []
        opts.append(lbfgs)
        cfg = WrapConfig(
            safety_rollout=arm_rollout_safety,
            optimizers=opts,
            compute_metrics=True,
            use_cuda_graph_metrics=grad_config_data["lbfgs"]["use_cuda_graph"],
            sync_cuda_time=sync_cuda_time,
        )
        ik = WrapBase(cfg)
        ik_cfg = IKSolverConfig(
            robot_config=robot_cfg,
            solver=ik,
            num_seeds=num_seeds,
            position_threshold=position_threshold,
            rotation_threshold=rotation_threshold,
            world_coll_checker=world_coll_checker,
            rollout_fn=aux_rollout,
            tensor_args=tensor_args,
            use_cuda_graph=use_cuda_graph,
        )
        return ik_cfg


@dataclass
class IKResult(Sequence):
    js_solution: JointState
    goal_pose: Pose
    solution: T_BDOF
    seed: T_BDOF
    success: T_BValue_bool
    position_error: T_BValue_float

    #: rotation error is computed as \sqrt(q_des^T * q)
    rotation_error: T_BValue_float
    error: T_BValue_float
    solve_time: float
    debug_info: Optional[Any] = None
    goalset_index: Optional[torch.Tensor] = None

    def __getitem__(self, idx):
        success = self.success[idx]

        return IKResult(
            js_solution=self.js_solution[idx],
            goal_pose=self.goal_pose[idx],
            solution=self.solution[idx],
            success=success,
            seed=self.seed[idx],
            position_error=self.position_error[idx],
            rotation_error=self.rotation_error[idx],
            debug_info=self.debug_info,
            goalset_index=None if self.goalset_index is None else self.goalset_index[idx],
        )

    def __len__(self):
        return self.seed.shape[0]

    def get_unique_solution(self, roundoff_decimals: int = 2) -> torch.Tensor:
        in_solution = self.solution[self.success]
        r_sol = torch.round(in_solution, decimals=roundoff_decimals)

        if not (len(in_solution.shape) == 2):
            log_error("Solution shape is not of length 2")

        s, i = torch.unique(r_sol, dim=-2, return_inverse=True)
        sol = in_solution[i[: s.shape[0]]]

        return sol

    def get_batch_unique_solution(self, roundoff_decimals: int = 2) -> List[torch.Tensor]:
        in_solution = self.solution
        r_sol = torch.round(in_solution, decimals=roundoff_decimals)
        if not (len(in_solution.shape) == 3):
            log_error("Solution shape is not of length 3")

        # do a for loop and return list of tensors
        sol = []
        for k in range(in_solution.shape[0]):
            # filter by success:
            in_succ = in_solution[k][self.success[k]]
            r_k = r_sol[k][self.success[k]]
            s, i = torch.unique(r_k, dim=-2, return_inverse=True)
            sol.append(in_succ[i[: s.shape[0]]])
            # sol.append(s)
        return sol


class IKSolver(IKSolverConfig):
    def __init__(self, config: IKSolverConfig) -> None:
        super().__init__(**vars(config))
        # self._solve_
        self.batch_size = -1
        self._num_seeds = self.num_seeds
        self.init_state = JointState.from_position(
            self.solver.rollout_fn.retract_state.unsqueeze(0)
        )
        self.dof = self.solver.safety_rollout.d_action
        self._col = None  # torch.arange(0, 1, device=self.tensor_args.device, dtype=torch.long)

        # self.fixed_seeds = self.solver.safety_rollout.sample_random_actions(100 * 200)
        # create random seeder:
        self.q_sample_gen = HaltonGenerator(
            self.dof,
            self.tensor_args,
            up_bounds=self.solver.safety_rollout.action_bound_highs,
            low_bounds=self.solver.safety_rollout.action_bound_lows,
            seed=1531,
            # store_buffer=1000,
        )

        # load ik nn:

        # store og outer iters:
        self.og_newton_iters = self.solver.newton_optimizer.outer_iters
        self._goal_buffer = Goal()
        self._solve_state = None
        self._kin_list = None
        self._rollout_list = None

    def update_goal_buffer(
        self,
        solve_state: ReacherSolveState,
        goal_pose: Pose,
        retract_config: Optional[T_BDOF] = None,
        link_poses: Optional[Dict[str, Pose]] = None,
    ) -> Goal:
        self._solve_state, self._goal_buffer, update_reference = solve_state.update_goal_buffer(
            goal_pose,
            None,
            retract_config,
            link_poses,
            self._solve_state,
            self._goal_buffer,
            self.tensor_args,
        )

        if update_reference:
            self.reset_shape()
            if self.use_cuda_graph and self._col is not None:
                log_error("changing goal type, breaking previous cuda graph.")
                self.reset_cuda_graph()

            self.solver.update_nproblems(self._solve_state.get_ik_batch_size())
            self._goal_buffer.current_state = self.init_state.repeat_seeds(goal_pose.batch)
            self._col = torch.arange(
                0,
                self._goal_buffer.goal_pose.batch,
                device=self.tensor_args.device,
                dtype=torch.long,
            )

        return self._goal_buffer

    def solve_any(
        self,
        solve_type: ReacherSolveType,
        goal_pose: Pose,
        retract_config: Optional[T_BDOF] = None,
        seed_config: Optional[T_BDOF] = None,
        return_seeds: int = 1,
        num_seeds: Optional[int] = None,
        use_nn_seed: bool = True,
        newton_iters: Optional[int] = None,
        link_poses: Optional[Dict[str, Pose]] = None,
    ) -> IKResult:
        if solve_type == ReacherSolveType.SINGLE:
            return self.solve_single(
                goal_pose,
                retract_config,
                seed_config,
                return_seeds,
                num_seeds,
                use_nn_seed,
                newton_iters,
                link_poses,
            )
        elif solve_type == ReacherSolveType.GOALSET:
            return self.solve_goalset(
                goal_pose,
                retract_config,
                seed_config,
                return_seeds,
                num_seeds,
                use_nn_seed,
                newton_iters,
            )
        elif solve_type == ReacherSolveType.BATCH:
            return self.solve_batch(
                goal_pose,
                retract_config,
                seed_config,
                return_seeds,
                num_seeds,
                use_nn_seed,
                newton_iters,
                link_poses,
            )
        elif solve_type == ReacherSolveType.BATCH_GOALSET:
            return self.solve_batch_goalset(
                goal_pose,
                retract_config,
                seed_config,
                return_seeds,
                num_seeds,
                use_nn_seed,
                newton_iters,
            )
        elif solve_type == ReacherSolveType.BATCH_ENV:
            return self.solve_batch_env(
                goal_pose,
                retract_config,
                seed_config,
                return_seeds,
                num_seeds,
                use_nn_seed,
                newton_iters,
            )
        elif solve_type == ReacherSolveType.BATCH_ENV_GOALSET:
            return self.solve_batch_env_goalset(
                goal_pose,
                retract_config,
                seed_config,
                return_seeds,
                num_seeds,
                use_nn_seed,
                newton_iters,
            )

    def solve_single(
        self,
        goal_pose: Pose,
        retract_config: Optional[T_BDOF] = None,
        seed_config: Optional[T_BDOF] = None,
        return_seeds: int = 1,
        num_seeds: Optional[int] = None,
        use_nn_seed: bool = True,
        newton_iters: Optional[int] = None,
        link_poses: Optional[Dict[str, Pose]] = None,
    ) -> IKResult:
        if num_seeds is None:
            num_seeds = self.num_seeds
        if return_seeds > num_seeds:
            num_seeds = return_seeds

        solve_state = ReacherSolveState(
            ReacherSolveType.SINGLE, num_ik_seeds=num_seeds, batch_size=1, n_envs=1, n_goalset=1
        )

        return self.solve_from_solve_state(
            solve_state,
            goal_pose,
            num_seeds,
            retract_config,
            seed_config,
            return_seeds,
            use_nn_seed,
            newton_iters,
            link_poses=link_poses,
        )

    def solve_goalset(
        self,
        goal_pose: Pose,
        retract_config: Optional[T_BDOF] = None,
        seed_config: Optional[T_BDOF] = None,
        return_seeds: int = 1,
        num_seeds: Optional[int] = None,
        use_nn_seed: bool = True,
        newton_iters: Optional[int] = None,
        link_poses: Optional[Dict[str, Pose]] = None,
    ) -> IKResult:
        if num_seeds is None:
            num_seeds = self.num_seeds
        if return_seeds > num_seeds:
            num_seeds = return_seeds

        solve_state = ReacherSolveState(
            ReacherSolveType.GOALSET,
            num_ik_seeds=num_seeds,
            batch_size=1,
            n_envs=1,
            n_goalset=goal_pose.n_goalset,
        )
        return self.solve_from_solve_state(
            solve_state,
            goal_pose,
            num_seeds,
            retract_config,
            seed_config,
            return_seeds,
            use_nn_seed,
            newton_iters,
            link_poses=link_poses,
        )

    def solve_batch(
        self,
        goal_pose: Pose,
        retract_config: Optional[T_BDOF] = None,
        seed_config: Optional[T_BDOF] = None,
        return_seeds: int = 1,
        num_seeds: Optional[int] = None,
        use_nn_seed: bool = True,
        newton_iters: Optional[int] = None,
        link_poses: Optional[Dict[str, Pose]] = None,
    ) -> IKResult:
        if num_seeds is None:
            num_seeds = self.num_seeds
        if return_seeds > num_seeds:
            num_seeds = return_seeds

        solve_state = ReacherSolveState(
            ReacherSolveType.BATCH,
            num_ik_seeds=num_seeds,
            batch_size=goal_pose.batch,
            n_envs=1,
            n_goalset=1,
        )
        return self.solve_from_solve_state(
            solve_state,
            goal_pose,
            num_seeds,
            retract_config,
            seed_config,
            return_seeds,
            use_nn_seed,
            newton_iters,
            link_poses=link_poses,
        )

    def solve_batch_goalset(
        self,
        goal_pose: Pose,
        retract_config: Optional[T_BDOF] = None,
        seed_config: Optional[T_BDOF] = None,
        return_seeds: int = 1,
        num_seeds: Optional[int] = None,
        use_nn_seed: bool = True,
        newton_iters: Optional[int] = None,
        link_poses: Optional[Dict[str, Pose]] = None,
    ) -> IKResult:
        if num_seeds is None:
            num_seeds = self.num_seeds
        if return_seeds > num_seeds:
            num_seeds = return_seeds

        solve_state = ReacherSolveState(
            ReacherSolveType.BATCH_GOALSET,
            num_ik_seeds=num_seeds,
            batch_size=goal_pose.batch,
            n_envs=1,
            n_goalset=goal_pose.n_goalset,
        )
        return self.solve_from_solve_state(
            solve_state,
            goal_pose,
            num_seeds,
            retract_config,
            seed_config,
            return_seeds,
            use_nn_seed,
            newton_iters,
            link_poses=link_poses,
        )

    def solve_batch_env(
        self,
        goal_pose: Pose,
        retract_config: Optional[T_BDOF] = None,
        seed_config: Optional[T_BDOF] = None,
        return_seeds: int = 1,
        num_seeds: Optional[int] = None,
        use_nn_seed: bool = True,
        newton_iters: Optional[int] = None,
        link_poses: Optional[Dict[str, Pose]] = None,
    ) -> IKResult:
        if num_seeds is None:
            num_seeds = self.num_seeds
        if return_seeds > num_seeds:
            num_seeds = return_seeds

        solve_state = ReacherSolveState(
            ReacherSolveType.BATCH_ENV,
            num_ik_seeds=num_seeds,
            batch_size=goal_pose.batch,
            n_envs=goal_pose.batch,
            n_goalset=1,
        )
        return self.solve_from_solve_state(
            solve_state,
            goal_pose,
            num_seeds,
            retract_config,
            seed_config,
            return_seeds,
            use_nn_seed,
            newton_iters,
            link_poses=link_poses,
        )

    def solve_batch_env_goalset(
        self,
        goal_pose: Pose,
        retract_config: Optional[T_BDOF] = None,
        seed_config: Optional[T_BDOF] = None,
        return_seeds: int = 1,
        num_seeds: Optional[int] = None,
        use_nn_seed: bool = True,
        newton_iters: Optional[int] = None,
        link_poses: Optional[Dict[str, Pose]] = None,
    ) -> IKResult:
        if num_seeds is None:
            num_seeds = self.num_seeds
        if return_seeds > num_seeds:
            num_seeds = return_seeds

        solve_state = ReacherSolveState(
            ReacherSolveType.BATCH_ENV_GOALSET,
            num_ik_seeds=num_seeds,
            batch_size=goal_pose.batch,
            n_envs=goal_pose.batch,
            n_goalset=goal_pose.n_goalset,
        )
        return self.solve_from_solve_state(
            solve_state,
            goal_pose,
            num_seeds,
            retract_config,
            seed_config,
            return_seeds,
            use_nn_seed,
            newton_iters,
            link_poses=link_poses,
        )

    def solve_from_solve_state(
        self,
        solve_state: ReacherSolveState,
        goal_pose: Pose,
        num_seeds: int,
        retract_config: Optional[T_BDOF] = None,
        seed_config: Optional[T_BDOF] = None,
        return_seeds: int = 1,
        use_nn_seed: bool = True,
        newton_iters: Optional[int] = None,
        link_poses: Optional[Dict[str, Pose]] = None,
    ) -> IKResult:
        # create goal buffer:
        goal_buffer = self.update_goal_buffer(solve_state, goal_pose, retract_config, link_poses)
        coord_position_seed = self.get_seed(
            num_seeds, goal_buffer.goal_pose, use_nn_seed, seed_config
        )

        if newton_iters is not None:
            self.solver.newton_optimizer.outer_iters = newton_iters
        self.solver.reset()
        result = self.solver.solve(goal_buffer, coord_position_seed)
        if newton_iters is not None:
            self.solver.newton_optimizer.outer_iters = self.og_newton_iters
        ik_result = self.get_result(num_seeds, result, goal_buffer.goal_pose, return_seeds)
        if ik_result.goalset_index is not None:
            ik_result.goalset_index[ik_result.goalset_index >= goal_pose.n_goalset] = 0

        return ik_result

    @profiler.record_function("ik/get_result")
    def get_result(
        self, num_seeds: int, result: WrapResult, goal_pose: Pose, return_seeds: int
    ) -> IKResult:
        success = self.get_success(result.metrics, num_seeds=num_seeds)
        # if result.metrics.cost is not None:
        #    result.metrics.pose_error += result.metrics.cost * 0.0001
        if result.metrics.null_space_error is not None:
            result.metrics.pose_error += result.metrics.null_space_error
        if result.metrics.cspace_error is not None:
            result.metrics.pose_error += result.metrics.cspace_error

        q_sol, success, position_error, rotation_error, total_error, goalset_index = get_result(
            result.metrics.pose_error,
            result.metrics.position_error,
            result.metrics.rotation_error,
            result.metrics.goalset_index,
            success,
            result.action.position,
            self._col,
            goal_pose.batch,
            return_seeds,
            num_seeds,
        )
        # check if locked joints exist and create js solution:

        new_js = JointState(q_sol, joint_names=self.rollout_fn.kinematics.joint_names)
        sol_js = self.rollout_fn.get_full_dof_from_solution(new_js)
        # reindex success to get successful poses?
        ik_result = IKResult(
            success=success,
            goal_pose=goal_pose,
            solution=q_sol,
            seed=None,
            js_solution=sol_js,
            # seed=coord_position_seed[idx].view(goal_pose.batch, return_seeds, -1).detach(),
            position_error=position_error,
            rotation_error=rotation_error,
            solve_time=result.solve_time,
            error=total_error,
            debug_info={"solver": result.debug},
            goalset_index=goalset_index,
        )
        return ik_result

    @profiler.record_function("ik/get_seed")
    def get_seed(
        self, num_seeds: int, goal_pose: Pose, use_nn_seed, seed_config: Optional[T_BDOF] = None
    ) -> torch.Tensor:
        if seed_config is None:
            coord_position_seed = self.generate_seed(
                num_seeds=num_seeds,
                batch=goal_pose.batch,
                use_nn_seed=use_nn_seed,
                pose=goal_pose,
            )
        elif seed_config.shape[1] < num_seeds:
            coord_position_seed = self.generate_seed(
                num_seeds=num_seeds - seed_config.shape[1],
                batch=goal_pose.batch,
                use_nn_seed=use_nn_seed,
                pose=goal_pose,
            )
            coord_position_seed = torch.cat((seed_config, coord_position_seed), dim=1)
        else:
            coord_position_seed = seed_config
        coord_position_seed = coord_position_seed.view(-1, 1, self.dof)
        return coord_position_seed

    def solve(
        self,
        goal_pose: Pose,
        retract_config: Optional[T_BDOF] = None,
        seed_config: Optional[T_BDOF] = None,
        return_seeds: int = 1,
        num_seeds: Optional[int] = None,
        use_nn_seed: bool = True,
        newton_iters: Optional[int] = None,
    ) -> IKResult:  # pragma : no cover
        """Ik solver

        Args:
            goal_pose (Pose): _description_
            retract_config (Optional[T_BDOF], optional): _description_. Defaults to None.
            seed_config (Optional[T_BDOF], optional): _description_. Defaults to None.
            return_seeds (int, optional): _description_. Defaults to 1.
            num_seeds (Optional[int], optional): _description_. Defaults to None.
            use_nn_seed (bool, optional): _description_. Defaults to True.
            newton_iters (Optional[int], optional): _description_. Defaults to None.

        Returns:
            IKResult: _description_
        """
        log_warn("IKSolver.solve() is deprecated, use solve_single() or others instead")
        if goal_pose.batch == 1 and goal_pose.n_goalset == 1:
            return self.solve_single(
                goal_pose,
                retract_config,
                seed_config,
                return_seeds,
                num_seeds,
                use_nn_seed,
                newton_iters,
            )
        if goal_pose.batch > 1 and goal_pose.n_goalset == 1:
            return self.solve_batch(
                goal_pose,
                retract_config,
                seed_config,
                return_seeds,
                num_seeds,
                use_nn_seed,
                newton_iters,
            )
        if goal_pose.batch > 1 and goal_pose.n_goalset > 1:
            return self.solve_batch_goalset(
                goal_pose,
                retract_config,
                seed_config,
                return_seeds,
                num_seeds,
                use_nn_seed,
                newton_iters,
            )
        if goal_pose.batch == 1 and goal_pose.n_goalset > 1:
            return self.solve_goalset(
                goal_pose,
                retract_config,
                seed_config,
                return_seeds,
                num_seeds,
                use_nn_seed,
                newton_iters,
            )

    def batch_env_solve(
        self,
        goal_pose: Pose,
        retract_config: Optional[T_BDOF] = None,
        seed_config: Optional[T_BDOF] = None,
        return_seeds: int = 1,
        num_seeds: Optional[int] = None,
        use_nn_seed: bool = True,
        newton_iters: Optional[int] = None,
    ) -> IKResult:  # pragma : no cover
        """Ik solver

        Args:
            goal_pose (Pose): _description_
            retract_config (Optional[T_BDOF], optional): _description_. Defaults to None.
            seed_config (Optional[T_BDOF], optional): _description_. Defaults to None.
            return_seeds (int, optional): _description_. Defaults to 1.
            num_seeds (Optional[int], optional): _description_. Defaults to None.
            use_nn_seed (bool, optional): _description_. Defaults to True.
            newton_iters (Optional[int], optional): _description_. Defaults to None.

        Returns:
            IKResult: _description_
        """
        log_warn(
            "IKSolver.batch_env_solve() is deprecated, use solve_batch_env() or others instead"
        )
        if goal_pose.n_goalset == 1:
            return self.solve_batch_env(
                goal_pose,
                retract_config,
                seed_config,
                return_seeds,
                num_seeds,
                use_nn_seed,
                newton_iters,
            )
        if goal_pose.n_goalset > 1:
            return self.solve_batch_env_goalset(
                goal_pose,
                retract_config,
                seed_config,
                return_seeds,
                num_seeds,
                use_nn_seed,
                newton_iters,
            )

    @torch.no_grad()
    @profiler.record_function("ik/get_success")
    def get_success(self, metrics: RolloutMetrics, num_seeds: int) -> torch.Tensor:
        success = get_success(
            metrics.feasible,
            metrics.position_error,
            metrics.rotation_error,
            num_seeds,
            self.position_threshold,
            self.rotation_threshold,
        )

        return success

    @torch.no_grad()
    @profiler.record_function("ik/generate_seed")
    def generate_seed(
        self,
        num_seeds: int,
        batch: int,
        use_nn_seed: bool = False,
        pose: Optional[Pose] = None,
    ) -> torch.Tensor:
        """Generate seeds for a batch. Given a pose set, this will create all
        the seeds: [batch + batch*random_restarts]

        Args:
            batch (int, optional): [description]. Defaults to 1.
            num_seeds (Optional[int], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        num_random_seeds = num_seeds
        seed_list = []
        if use_nn_seed and self.ik_nn_seeder is not None:
            num_random_seeds = num_seeds - 1
            in_data = torch.cat(
                (pose.position, pose.quaternion), dim=-1
            )  # .to(dtype=torch.float32)
            nn_seed = self.ik_nn_seeder(in_data).view(
                batch, 1, self.dof
            )  # .to(dtype=self.tensor_args.dtype)
            seed_list.append(nn_seed)
            # print("using nn seed")
        if num_random_seeds > 0:
            random_seed = self.q_sample_gen.get_gaussian_samples(num_random_seeds * batch).view(
                batch, num_random_seeds, self.dof
            )

            # random_seed = self.fixed_seeds[:num_random_seeds*batch].view(batch, num_random_seeds,
            # self.solver.safety_rollout.d_action)

            seed_list.append(random_seed)
        coord_position_seed = torch.cat(seed_list, dim=1)
        return coord_position_seed

    def update_world(self, world: WorldConfig) -> bool:
        self.world_coll_checker.load_collision_model(world)
        return True

    def reset_seed(self) -> None:
        self.q_sample_gen.reset()

    def check_constraints(self, q: JointState) -> RolloutMetrics:
        metrics = self.rollout_fn.rollout_constraint(q.position.unsqueeze(1))
        return metrics

    def sample_configs(self, n: int, use_batch_env=False) -> torch.Tensor:
        """
        Only works for environment=0
        """
        samples = self.rollout_fn.sample_random_actions(n * self.sample_rejection_ratio)
        metrics = self.rollout_fn.rollout_constraint(
            samples.unsqueeze(1), use_batch_env=use_batch_env
        )
        return samples[metrics.feasible.squeeze()][:n]

    @property
    def kinematics(self) -> CudaRobotModel:
        return self.rollout_fn.dynamics_model.robot_model

    def get_all_rollout_instances(self) -> List[RolloutBase]:
        if self._rollout_list is None:
            self._rollout_list = [self.rollout_fn] + self.solver.get_all_rollout_instances()
        return self._rollout_list

    def get_all_kinematics_instances(self) -> List[CudaRobotModel]:
        if self._kin_list is None:
            self._kin_list = [
                i.dynamics_model.robot_model for i in self.get_all_rollout_instances()
            ]
        return self._kin_list

    def fk(self, q: torch.Tensor) -> CudaRobotModelState:
        return self.kinematics.get_state(q)

    def reset_cuda_graph(self) -> None:
        self.solver.reset_cuda_graph()
        self.rollout_fn.reset_cuda_graph()

    def reset_shape(self):
        self.solver.reset_shape()
        self.rollout_fn.reset_shape()

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

    def get_retract_config(self):
        return self.rollout_fn.dynamics_model.retract_config

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


@torch.jit.script
def get_success(
    feasible,
    position_error,
    rotation_error,
    num_seeds: int,
    position_threshold: float,
    rotation_threshold: float,
):
    feasible = feasible.view(-1, num_seeds)
    converge = torch.logical_and(
        position_error <= position_threshold,
        rotation_error <= rotation_threshold,
    ).view(-1, num_seeds)
    success = torch.logical_and(feasible, converge)
    return success


@torch.jit.script
def get_result(
    pose_error,
    position_error,
    rotation_error,
    goalset_index: Union[torch.Tensor, None],
    success,
    sol_position,
    col,
    batch_size: int,
    return_seeds: int,
    num_seeds: int,
):
    error = pose_error.view(-1, num_seeds)
    error[~success] += 1000.0
    _, idx = torch.topk(error, k=return_seeds, largest=False, dim=-1)
    idx = idx + num_seeds * col.unsqueeze(-1)
    q_sol = sol_position[idx].view(batch_size, return_seeds, -1)

    success = success.view(-1)[idx].view(batch_size, return_seeds)
    position_error = position_error[idx].view(batch_size, return_seeds)
    rotation_error = rotation_error[idx].view(batch_size, return_seeds)
    total_error = position_error + rotation_error
    if goalset_index is not None:
        goalset_index = goalset_index[idx].view(batch_size, return_seeds)
    return q_sol, success, position_error, rotation_error, total_error, goalset_index
