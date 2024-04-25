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
import time
from dataclasses import dataclass
from typing import Dict, Optional, Union

# Third Party
import torch

# CuRobo
from curobo.geom.sdf.utils import create_collision_checker
from curobo.geom.sdf.world import CollisionCheckerType, WorldCollision, WorldCollisionConfig
from curobo.geom.types import WorldConfig
from curobo.opt.newton.lbfgs import LBFGSOpt, LBFGSOptConfig
from curobo.opt.particle.parallel_es import ParallelES, ParallelESConfig
from curobo.opt.particle.parallel_mppi import ParallelMPPI, ParallelMPPIConfig
from curobo.rollout.arm_reacher import ArmReacher, ArmReacherConfig
from curobo.rollout.rollout_base import Goal
from curobo.types.base import TensorDeviceType
from curobo.types.robot import JointState, RobotConfig
from curobo.util.logger import log_error, log_info, log_warn
from curobo.util_file import (
    get_robot_configs_path,
    get_task_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.types import ReacherSolveState, ReacherSolveType
from curobo.wrap.wrap_base import WrapResult
from curobo.wrap.wrap_mpc import WrapConfig, WrapMpc


@dataclass
class MpcSolverConfig:
    solver: WrapMpc
    world_coll_checker: Optional[WorldCollision] = None
    tensor_args: TensorDeviceType = TensorDeviceType()
    use_cuda_graph_full_step: bool = False

    @staticmethod
    def load_from_robot_config(
        robot_cfg: Union[Union[str, dict], RobotConfig],
        world_cfg: Union[Union[str, dict], WorldConfig],
        base_cfg: Optional[dict] = None,
        tensor_args: TensorDeviceType = TensorDeviceType(),
        compute_metrics: bool = True,
        use_cuda_graph: Optional[bool] = None,
        particle_opt_iters: Optional[int] = None,
        self_collision_check: bool = True,
        collision_checker_type: Optional[CollisionCheckerType] = CollisionCheckerType.PRIMITIVE,
        use_es: Optional[bool] = None,
        es_learning_rate: Optional[float] = 0.01,
        use_cuda_graph_metrics: bool = False,
        store_rollouts: bool = True,
        use_cuda_graph_full_step: bool = False,
        sync_cuda_time: bool = True,
        collision_cache: Optional[Dict[str, int]] = None,
        n_collision_envs: Optional[int] = None,
        collision_activation_distance: Optional[float] = None,
        world_coll_checker=None,
        step_dt: Optional[float] = None,
        use_lbfgs: bool = False,
        use_mppi: bool = True,
    ):
        if use_cuda_graph_full_step:
            log_error("use_cuda_graph_full_step currently is not supported")
            raise ValueError("use_cuda_graph_full_step currently is not supported")

        task_file = "particle_mpc.yml"
        config_data = load_yaml(join_path(get_task_configs_path(), task_file))
        config_data["mppi"]["n_problems"] = 1
        if step_dt is not None:
            config_data["model"]["dt_traj_params"]["base_dt"] = step_dt
        if particle_opt_iters is not None:
            config_data["mppi"]["n_iters"] = particle_opt_iters

        if base_cfg is None:
            base_cfg = load_yaml(join_path(get_task_configs_path(), "base_cfg.yml"))
        if collision_cache is not None:
            base_cfg["world_collision_checker_cfg"]["cache"] = collision_cache
        if n_collision_envs is not None:
            base_cfg["world_collision_checker_cfg"]["n_envs"] = n_collision_envs

        if collision_activation_distance is not None:
            config_data["cost"]["primitive_collision_cfg"][
                "activation_distance"
            ] = collision_activation_distance

        if not self_collision_check:
            base_cfg["constraint"]["self_collision_cfg"]["weight"] = 0.0
            config_data["cost"]["self_collision_cfg"]["weight"] = 0.0
        if collision_checker_type is not None:
            base_cfg["world_collision_checker_cfg"]["checker_type"] = collision_checker_type
        if isinstance(robot_cfg, str):
            robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_cfg))
        if isinstance(robot_cfg, dict):
            robot_cfg = RobotConfig.from_dict(robot_cfg, tensor_args)

        if isinstance(world_cfg, str):
            world_cfg = load_yaml(join_path(get_world_configs_path(), world_cfg))

        if world_coll_checker is None and world_cfg is not None:
            world_cfg = WorldCollisionConfig.load_from_dict(
                base_cfg["world_collision_checker_cfg"], world_cfg, tensor_args
            )
            world_coll_checker = create_collision_checker(world_cfg)
        grad_config_data = None
        if use_lbfgs:
            grad_config_data = load_yaml(join_path(get_task_configs_path(), "gradient_mpc.yml"))
            if step_dt is not None:
                grad_config_data["model"]["dt_traj_params"]["base_dt"] = step_dt
                grad_config_data["model"]["dt_traj_params"]["max_dt"] = step_dt

            config_data["model"] = grad_config_data["model"]
            if use_cuda_graph is not None:
                grad_config_data["lbfgs"]["use_cuda_graph"] = use_cuda_graph

        cfg = ArmReacherConfig.from_dict(
            robot_cfg,
            config_data["model"],
            config_data["cost"],
            base_cfg["constraint"],
            base_cfg["convergence"],
            base_cfg["world_collision_checker_cfg"],
            world_cfg,
            world_coll_checker=world_coll_checker,
            tensor_args=tensor_args,
        )

        arm_rollout_mppi = ArmReacher(cfg)
        arm_rollout_safety = ArmReacher(cfg)
        config_data["mppi"]["store_rollouts"] = store_rollouts
        if use_cuda_graph is not None:
            config_data["mppi"]["use_cuda_graph"] = use_cuda_graph
        if use_cuda_graph_full_step:
            config_data["mppi"]["sync_cuda_time"] = False
        config_dict = ParallelMPPIConfig.create_data_dict(
            config_data["mppi"], arm_rollout_mppi, tensor_args
        )
        solvers = []
        parallel_mppi = None
        if use_es is not None and use_es:
            log_warn("ES solver for MPC is highly experimental, not safe to run on real robots")

            mppi_cfg = ParallelESConfig(**config_dict)
            if es_learning_rate is not None:
                mppi_cfg.learning_rate = es_learning_rate
            parallel_mppi = ParallelES(mppi_cfg)
        elif use_mppi:
            mppi_cfg = ParallelMPPIConfig(**config_dict)
            parallel_mppi = ParallelMPPI(mppi_cfg)
        if parallel_mppi is not None:
            solvers.append(parallel_mppi)
        if use_lbfgs:
            log_warn("LBFGS solver for MPC is highly experimental, not safe to run on real robots")
            grad_cfg = ArmReacherConfig.from_dict(
                robot_cfg,
                grad_config_data["model"],
                grad_config_data["cost"],
                base_cfg["constraint"],
                base_cfg["convergence"],
                base_cfg["world_collision_checker_cfg"],
                world_cfg,
                world_coll_checker=world_coll_checker,
                tensor_args=tensor_args,
            )

            arm_rollout_grad = ArmReacher(grad_cfg)
            lbfgs_cfg_dict = LBFGSOptConfig.create_data_dict(
                grad_config_data["lbfgs"], arm_rollout_grad, tensor_args
            )
            lbfgs = LBFGSOpt(LBFGSOptConfig(**lbfgs_cfg_dict))
            solvers.append(lbfgs)

        mpc_cfg = WrapConfig(
            safety_rollout=arm_rollout_safety,
            optimizers=solvers,
            compute_metrics=compute_metrics,
            use_cuda_graph_metrics=use_cuda_graph_metrics,
            sync_cuda_time=sync_cuda_time,
        )
        solver = WrapMpc(mpc_cfg)
        return MpcSolverConfig(
            solver,
            tensor_args=tensor_args,
            use_cuda_graph_full_step=use_cuda_graph_full_step,
            world_coll_checker=world_coll_checker,
        )


class MpcSolver(MpcSolverConfig):
    """Model Predictive Control Solver for Arm Reacher task.

    Args:
        MpcSolverConfig: _description_
    """

    def __init__(self, config: MpcSolverConfig) -> None:
        super().__init__(**vars(config))
        self.tensor_args = self.solver.rollout_fn.tensor_args
        self._goal_buffer = Goal()
        self.batch_size = -1
        self._goal_buffer = None
        self._solve_state = None
        self._col = None
        self._step_goal_buffer = None
        self._cu_state_in = None
        self._cu_seed = None
        self._cu_step_init = None
        self._cu_step_graph = None
        self._cu_result = None

    def _update_batch_size(self, batch_size):
        if self.batch_size != batch_size:
            self.batch_size = batch_size

    def update_goal_buffer(
        self,
        solve_state: ReacherSolveState,
        goal: Goal,
    ) -> Goal:
        self._solve_state, self._goal_buffer, update_reference = solve_state.update_goal(
            goal,
            self._solve_state,
            self._goal_buffer,
            self.tensor_args,
        )
        if update_reference:
            self.solver.update_nproblems(self._solve_state.get_batch_size())
            self.reset()
            self.reset_cuda_graph()
            self._col = torch.arange(
                0, goal.batch, device=self.tensor_args.device, dtype=torch.long
            )
            self._step_goal_buffer = Goal(
                current_state=self._goal_buffer.current_state.clone(),
                batch_current_state_idx=self._goal_buffer.batch_current_state_idx.clone(),
            )
        return self._goal_buffer

    def step(
        self,
        current_state: JointState,
        shift_steps: int = 1,
        seed_traj: Optional[JointState] = None,
        max_attempts: int = 1,
    ):
        converged = True

        for _ in range(max_attempts):
            result = self.step_once(current_state.clone(), shift_steps, seed_traj)
            if (
                torch.count_nonzero(torch.isnan(result.action.position)) == 0
                and torch.max(torch.abs(result.action.position)) < 10.0
                and torch.count_nonzero(~result.metrics.feasible) == 0
            ):
                converged = True
                break
            self.reset()
        if not converged:
            result.action.copy_(current_state)
            log_warn("NOT CONVERGED")

        return result

    def step_once(
        self,
        current_state: JointState,
        shift_steps: int = 1,
        seed_traj: Optional[JointState] = None,
    ) -> WrapResult:
        # Create cuda graph for whole solve step including computation of metrics
        # Including updation of goal buffers

        if self._solve_state is None:
            log_error("Need to first setup solve state before calling solve()")

        if self.use_cuda_graph_full_step:
            st_time = time.time()
            if not self._cu_step_init:
                self._initialize_cuda_graph_step(current_state, shift_steps, seed_traj)
            self._cu_state_in.copy_(current_state)
            if seed_traj is not None:
                self._cu_seed.copy_(seed_traj)
            self._cu_step_graph.replay()
            result = self._cu_result.clone()
            torch.cuda.synchronize()
            result.solve_time = time.time() - st_time
        else:
            self._step_goal_buffer.current_state.copy_(current_state)
            result = self._solve_from_solve_state(
                self._solve_state,
                self._step_goal_buffer,
                shift_steps,
                seed_traj,
            )

        return result

    def _step(
        self,
        current_state: JointState,
        shift_steps: int = 1,
        seed_traj: Optional[JointState] = None,
    ):
        self._step_goal_buffer.current_state.copy_(current_state)
        result = self._solve_from_solve_state(
            self._solve_state,
            self._step_goal_buffer,
            shift_steps,
            seed_traj,
        )

        return result

    def _initialize_cuda_graph_step(
        self,
        current_state: JointState,
        shift_steps: int = 1,
        seed_traj: Optional[JointState] = None,
    ):
        log_info("MpcSolver: Creating Cuda Graph")
        self._cu_state_in = current_state.clone()
        if seed_traj is not None:
            self._cu_seed = seed_traj.clone()
        s = torch.cuda.Stream(device=self.tensor_args.device)
        s.wait_stream(torch.cuda.current_stream(device=self.tensor_args.device))
        with torch.cuda.stream(s):
            for _ in range(3):
                self._cu_result = self._step(
                    self._cu_state_in, shift_steps=shift_steps, seed_traj=self._cu_seed
                )
        torch.cuda.current_stream(device=self.tensor_args.device).wait_stream(s)
        self.reset()
        self._cu_step_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cu_step_graph, stream=s):
            self._cu_result = self._step(
                self._cu_state_in, shift_steps=shift_steps, seed_traj=self._cu_seed
            )
        self._cu_step_init = True

    def setup_solve_single(self, goal: Goal, num_seeds: Optional[int] = None) -> Goal:
        solve_state = ReacherSolveState(
            ReacherSolveType.SINGLE, num_mpc_seeds=num_seeds, batch_size=1, n_envs=1, n_goalset=1
        )
        goal_buffer = self.update_goal_buffer(solve_state, goal)
        return goal_buffer

    def setup_solve_goalset(self, goal: Goal, num_seeds: Optional[int] = None) -> Goal:
        solve_state = ReacherSolveState(
            ReacherSolveType.GOALSET,
            num_mpc_seeds=num_seeds,
            batch_size=1,
            n_envs=1,
            n_goalset=goal.n_goalset,
        )
        goal_buffer = self.update_goal_buffer(solve_state, goal)
        return goal_buffer

    def setup_solve_batch(self, goal: Goal, num_seeds: Optional[int] = None) -> Goal:
        solve_state = ReacherSolveState(
            ReacherSolveType.BATCH,
            num_mpc_seeds=num_seeds,
            batch_size=goal.batch,
            n_envs=1,
            n_goalset=1,
        )
        goal_buffer = self.update_goal_buffer(solve_state, goal)
        return goal_buffer

    def setup_solve_batch_goalset(self, goal: Goal, num_seeds: Optional[int] = None) -> Goal:
        solve_state = ReacherSolveState(
            ReacherSolveType.BATCH_GOALSET,
            num_mpc_seeds=num_seeds,
            batch_size=goal.batch,
            n_envs=1,
            n_goalset=goal.n_goalset,
        )
        goal_buffer = self.update_goal_buffer(solve_state, goal)
        return goal_buffer

    def setup_solve_batch_env(self, goal: Goal, num_seeds: Optional[int] = None) -> Goal:
        solve_state = ReacherSolveState(
            ReacherSolveType.BATCH_ENV,
            num_mpc_seeds=num_seeds,
            batch_size=goal.batch,
            n_envs=1,
            n_goalset=1,
        )
        goal_buffer = self.update_goal_buffer(solve_state, goal)
        return goal_buffer

    def setup_solve_batch_env_goalset(self, goal: Goal, num_seeds: Optional[int] = None) -> Goal:
        solve_state = ReacherSolveState(
            ReacherSolveType.BATCH_ENV_GOALSET,
            num_mpc_seeds=num_seeds,
            batch_size=goal.batch,
            n_envs=1,
            n_goalset=goal.n_goalset,
        )
        goal_buffer = self.update_goal_buffer(solve_state, goal)
        return goal_buffer

    def _solve_from_solve_state(
        self,
        solve_state: ReacherSolveState,
        goal: Goal,
        shift_steps: int = 1,
        seed_traj: Optional[JointState] = None,
    ) -> WrapResult:
        if solve_state.batch_env:
            if solve_state.batch_size > self.world_coll_checker.n_envs:
                raise ValueError("Batch Env is less that goal batch")

        goal_buffer = self.update_goal_buffer(solve_state, goal)
        # NOTE: implement initialization from seed set here:
        if seed_traj is not None:
            self.solver.update_init_seed(seed_traj)

        result = self.solver.solve(goal_buffer, seed_traj, shift_steps)
        result.js_action = self.rollout_fn.get_full_dof_from_solution(result.action)
        return result

    def fn(self):
        # this will run one step of optimization and get new command
        pass

    def update_goal(self, goal: Goal):
        self.solver.update_params(goal)
        return True

    def reset(self):
        # reset warm start
        self.solver.reset()
        pass

    def reset_cuda_graph(self):
        self.solver.reset_cuda_graph()

    @property
    def rollout_fn(self):
        return self.solver.safety_rollout

    def enable_cspace_cost(self, enable=True):
        self.solver.safety_rollout.enable_cspace_cost(enable)
        for opt in self.solver.optimizers:
            opt.rollout_fn.enable_cspace_cost(enable)

    def enable_pose_cost(self, enable=True):
        self.solver.safety_rollout.enable_pose_cost(enable)
        for opt in self.solver.optimizers:
            opt.rollout_fn.enable_pose_cost(enable)

    def get_active_js(
        self,
        in_js: JointState,
    ):
        opt_jnames = self.rollout_fn.joint_names
        opt_js = in_js.get_ordered_joint_state(opt_jnames)
        return opt_js

    @property
    def joint_names(self):
        return self.rollout_fn.joint_names

    def update_world(self, world: WorldConfig):
        self.world_coll_checker.load_collision_model(world)
        return True

    def get_visual_rollouts(self):
        return self.solver.optimizers[0].get_rollouts()

    @property
    def kinematics(self):
        return self.solver.safety_rollout.dynamics_model.robot_model

    @property
    def world_collision(self):
        return self.world_coll_checker
