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

"""
This module contains :meth:`IKSolver` to solve inverse kinematics, along with
:meth:`IKSolverConfig` to configure the solver, and :meth:`IKResult` to store the result. A minimal
example to solve IK is available at :ref:`python_ik_example`.

.. raw:: html

    <p>
    <video autoplay="True" loop="True" muted="True" preload="auto" width="100%"><source src="../videos/ik_obs_clip.mp4" type="video/mp4"></video>
    </p>
This demo is available in :ref:`isaac_ik_reachability`.


"""
from __future__ import annotations

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
from curobo.types.tensor import T_BDOF, T_DOF, T_BValue_bool, T_BValue_float
from curobo.util.logger import log_error, log_warn
from curobo.util.sample_lib import HaltonGenerator
from curobo.util.torch_utils import get_torch_jit_decorator, is_cuda_graph_reset_available
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
    """Configuration for Inverse Kinematics Solver.

    A helper function to load from robot
    configuration is provided as :func:`IKSolverConfig.load_from_robot_config`.

    """

    #: representation of robot, includes kinematic model, link geometry and joint limits.
    robot_config: RobotConfig

    #: Wrapped solver which has many instances of ArmReacher and two optimization
    #: solvers (MPPI, LBFGS) as default.
    solver: WrapBase

    #: Number of seeds to optimize per IK problem in parallel.
    num_seeds: int

    #: Position convergence threshold in meters to use to compute success.
    position_threshold: float

    #: Rotation convergence threshold to use to compute success. Currently this
    #: measures the geodesic distance between reached quaternion and target quaternion.
    #: A good accuracy threshold is 0.05. Check pose_distance_kernel.cu for the exact
    #: implementation.
    rotation_threshold: float

    #: Reference to an instance of ArmReacher rollout class to use for auxillary functions.
    rollout_fn: ArmReacher

    #: Undeveloped functionality to use a neural network as seed for IK.
    ik_nn_seeder: Optional[str] = None

    #: Reference to world collision checker to use for collision avoidance.
    world_coll_checker: Optional[WorldCollision] = None

    #: Rejection ratio for sampling collision-free configurations. These samples are not
    #: used as seeds for solving IK as starting from collision-free seeds did not improve success.
    sample_rejection_ratio: int = 50

    #: Device and floating precision to use for IKSolver.
    tensor_args: TensorDeviceType = TensorDeviceType()

    #: Flag to enable capturing solver iterations as a cuda graph to reduce kernel launch overhead.
    #: Setting this to True can give upto 10x speedup while limiting the IKSolver to solving fixed
    #: batch size of problems.
    use_cuda_graph: bool = True

    #: Seed to use in pseudorandom generator used for creating IK seeds.
    seed: int = 1531

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
        gradient_file: str = "gradient_ik_autotune.yml",
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
        seed: int = 1531,
    ) -> IKSolverConfig:
        """Helper function to load IKSolver configuration from a robot file and world file.

        Use this function to create an instance of IKSolverConfig and load the config into IKSolver.

        Args:
            robot_cfg: representation of robot, includes kinematic model, link geometry, and
                joint limits. This can take as input a cuRobo robot configuration yaml file path or
                a loaded dictionary of the robot configuration file or an instance of RobotConfig.
                Configuration files for some common robots is available at
                :ref:`available_robot_list`. For other robots, follow :ref:`tut_robot_configuration`
                tutorial to create a configuration file.
            world_model: representation of the world for obtaining collision-free IK solutions.
                The world can be represented as cuboids, meshes, from depth camera with nvblox, and
                as an Euclidean Signed Distance Grid (ESDF). This world model can be loaded from a
                dictionary (from disk through yaml) or from :class:`curobo.geom.types.WorldConfig`.
                In an application, if collision computations are being used in more than one
                instance, it's memory efficient to create one instance of
                :class:`curobo.geom.sdf.world.WorldCollision` and share across these class. For
                example, if an instance of IKSolver and MotionGen exists in an application, a
                :class:`curobo.geom.sdf.world.WorldCollision` object can be created in IKSolver
                and then when creating :class:`curobo.wrap.reacher.motion_gen.MotionGenConfig`, this
                ``world_coll_checker`` can be passed as reference. :ref:`world_collision` discusses
                types of obstacles in more detail.
            tensor_args: Device and floating precision to use for IKSolver.
            num_seeds: Number of seeds to optimize per IK problem in parallel.
            position_threshold: Position convergence threshold in meters to use to compute success.
            rotation_threshold: Rotation convergence threshold to use to compute success. See
                :meth:`IKSolverConfig.rotation_threshold` for more details.
            world_coll_checker: Reference to world collision checker to use for collision avoidance.
            base_cfg_file: Base configuration file for IKSolver. This configuration file is used to
                measure convergence and collision-free check after optimization is complete.
            particle_file: Configuration file for particle optimization (uses MPPI as optimizer).
            gradient_file: Configuration file for gradient optimization (uses LBFGS as optimizer).
            use_cuda_graph: Flag to enable capturing solver iterations as a cuda graph to reduce
                kernel launch overhead. Setting this to True can give upto 10x speedup while
                limiting the IKSolver to solving fixed batch size of problems.
            self_collision_check: Flag to enable self-collision checking.
            self_collision_opt: Flag to enable self-collision cost during optimization.
            grad_iters: Number of iterations for gradient optimization.
            use_particle_opt: Flag to enable particle optimization.
            collision_checker_type: Type of collision checker to use for collision checking.
            sync_cuda_time: Flag to enable synchronization of cuda device with host using
                :py:func:`torch.cuda.synchronize` before measuring compute time. If you set this to
                False, then measured time will not include completion of any launched CUDA kernels.
            use_gradient_descent: Flag to enable gradient descent optimization instead of LBFGS.
            collision_cache: Number of objects to cache for collision checking.
            n_collision_envs: Number of collision environments to use for IK. This is useful when
                solving IK for multiple robots in different environments in parallel.
            ee_link_name: Name of end-effector link to use for IK.
            use_es: Flag to enable Evolution Strategies optimization instead of MPPI.
            es_learning_rate: Learning rate for Evolution Strategies optimization.
            use_fixed_samples: Flag to enable fixed samples for MPPI optimization.
            store_debug: Flag to enable storing debug information during optimization. Enabling this
                will store solution and cost at every iteration. This will significantly slow down
                the optimization as CUDA graph capture is disabled. Only use this for debugging.
            regularization: Flag to enable regularization during optimization.
            collision_activation_distance: Distance from obstacle at which to activate collision
                cost. A good value is 0.01 (1cm).
            high_precision: Flag to solve IK at higher pose accuracy. This will increase the number
                of LBFGS iterations from 100 to 200. This flag is set to True when
                position_threshold is less than or equal to 1mm (0.001).
            project_pose_to_goal_frame: Flag to project pose to goal frame when computing distance.
            seed: Seed to use in pseudorandom generator used for creating IK seeds.
        """
        if position_threshold <= 0.001:
            high_precision = True
        if high_precision:
            if grad_iters is None:
                grad_iters = 200
        # use default values, disable environment collision checking
        base_config_data = load_yaml(join_path(get_task_configs_path(), base_cfg_file))
        config_data = load_yaml(join_path(get_task_configs_path(), particle_file))
        grad_config_data = load_yaml(join_path(get_task_configs_path(), gradient_file))

        base_config_data["cost"]["pose_cfg"]["project_distance"] = project_pose_to_goal_frame
        base_config_data["convergence"]["pose_cfg"]["project_distance"] = project_pose_to_goal_frame
        config_data["cost"]["pose_cfg"]["project_distance"] = project_pose_to_goal_frame
        grad_config_data["cost"]["pose_cfg"]["project_distance"] = project_pose_to_goal_frame

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
                log_error("ee link cannot be changed after creating RobotConfig")
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
        safety_cfg = ArmReacherConfig.from_dict(
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

        aux_cfg = ArmReacherConfig.from_dict(
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
        arm_rollout_safety = ArmReacher(safety_cfg)
        aux_rollout = ArmReacher(aux_cfg)

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
            use_cuda_graph_metrics=use_cuda_graph,
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
            seed=seed,
        )
        return ik_cfg


@dataclass
class IKResult(Sequence):
    """Solution of Inverse Kinematics problem."""

    #: Joint state solution for the IK problem.
    js_solution: JointState

    #: Goal pose used in IK problem.
    goal_pose: Pose

    #: Joint configuration Solution as tensor
    solution: T_BDOF

    #: Seed that was selected as the starting point for optimization. This is currently not
    #: available in the result and is set to None.
    seed: T_BDOF

    #: Success tensor for IK problem. If planning for batch, use this to filter js_solution. If
    #: planning for single problem, use this to check if the problem was solved successfully.
    success: T_BValue_bool

    #: Position error between solved pose and goal pose in meters, computed with l-2 norm.
    position_error: T_BValue_float

    #: Rotation error between solved pose and goal pose, computed with geodesic distance. Roughly
    #: \sqrt(q_des^T * q). A good accuracy threshold is 0.05.
    rotation_error: T_BValue_float

    #: Total error for IK problem. This is the sum of position_error and rotation_error.
    error: T_BValue_float

    #: Time taken to solve the IK problem in seconds.
    solve_time: float

    #: Debug information from solver. This can be used to debug solver convergence and tune
    #: weights between cost terms.
    debug_info: Optional[Any] = None

    #: Index of goal in goalset that the IK solver reached. This is useful when solving with
    #: :meth:`IKSolver.solve_goalset` (also :meth:`IKSolver.solve_batch_goalset`,
    #: :meth:`IKSolver.solve_batch_env_goalset`) where the task is to find an IK solution that
    #: reaches 1 pose in set of poses. This index is used to identify which pose was reached by the
    #: solution.
    goalset_index: Optional[torch.Tensor] = None

    def __getitem__(self, idx) -> IKResult:
        """Get IKResult for a single problem in batch.

        Args:
            idx: Index of the problem in batch.

        Returns:
            IKResult for the problem at index idx.
        """

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

    def __len__(self) -> int:
        """Get number of problems in IKResult."""
        return self.seed.shape[0]

    def get_unique_solution(self, roundoff_decimals: int = 2) -> torch.Tensor:
        """Get unique solutions from many feasible solutions for the same problem.

        Use this after solving IK with :meth:`IKSolver.solve_single` with return_seeds > 1. This
        function will return unique solutions from the set of feasible solutions, filering out
        solutions that are within roundoff_decimals of each other.

        Args:
            roundoff_decimals: Number of decimal places to round off the solution to measure
                uniqueness.

        Returns:
            Unique solutions from the set of feasible solutions.
        """
        in_solution = self.solution[self.success]
        r_sol = torch.round(in_solution, decimals=roundoff_decimals)

        if not (len(in_solution.shape) == 2):
            log_error("Solution shape is not of length 2")

        s, i = torch.unique(r_sol, dim=-2, return_inverse=True)
        sol = in_solution[i[: s.shape[0]]]

        return sol

    def get_batch_unique_solution(self, roundoff_decimals: int = 2) -> List[torch.Tensor]:
        """Get unique solutions from many feasible solutions for the same problem in batch.

        Use this after solving IK with :meth:`IKSolver.solve_batch` with return_seeds > 1. This
        function will return unique solutions from the set of feasible solutions, filering out
        solutions that are within roundoff_decimals of each other. Current implementation is not
        efficient as it will run a for loop over the batch as each problem in the batch can have
        different number of unique solutions.

        Args:
            roundoff_decimals: Number of decimal places to round off the solution to measure
                uniqueness.

        Returns:
            List of unique solutions from the set of feasible solutions.
        """
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
        return sol


class IKSolver(IKSolverConfig):
    """Inverse Kinematics Solver for reaching Cartesian Pose with end-effector.

    This also supports reaching poses for multiple links of a robot (e.g., a bimanual robot).
    This solver creates memory buffers on the GPU, captures CUDAGraphs of the operations during the
    very first call to any of the solve functions and reuses them for solving subsequent IK
    problems. As such, changing the number of problems, number of seeds, number of seeds or type of
    IKProblem to solve will cause existing CUDAGraphs to be invalidated, which currently leads to an
    exit with error. Either use multiple instances of IKSolver to solve different types of
    IKProblems or disable `use_cuda_graph` to avoid this issue. Disabling `use_cuda_graph` can lead
    to a 10x slowdown in solving IK problems.
    """

    def __init__(self, config: IKSolverConfig) -> None:
        """Initializer for IK Solver.

        Args:
            config: Configuration for Inverse Kinematics Solver.
        """

        super().__init__(**vars(config))
        self.batch_size = -1
        self._num_seeds = self.num_seeds
        self.init_state = JointState.from_position(
            self.solver.rollout_fn.retract_state.unsqueeze(0)
        )
        self.dof = self.solver.safety_rollout.d_action
        self._col = None

        # create random seeder:
        self.q_sample_gen = HaltonGenerator(
            self.dof,
            self.tensor_args,
            up_bounds=self.solver.safety_rollout.action_bound_highs,
            low_bounds=self.solver.safety_rollout.action_bound_lows,
            seed=self.seed,
        )

        # store og outer iters:
        self.og_newton_iters = self.solver.newton_optimizer.outer_iters
        self._goal_buffer = Goal()
        self._solve_state = None
        self._kin_list = None
        self._rollout_list = None

    def _update_goal_buffer(
        self,
        solve_state: ReacherSolveState,
        goal_pose: Pose,
        retract_config: Optional[T_BDOF] = None,
        link_poses: Optional[Dict[str, Pose]] = None,
    ) -> Goal:
        """Update goal buffer with new goal pose and retract configuration.

        Args:
            solve_state: Current IK problem parameters.
            goal_pose: New goal pose to solve for IK.
            retract_config: Retract configuration to use for IK. If None, the retract configuration
                is set to the retract_config in :meth:`curobo.types.robot.RobotConfig`.
            link_poses: New poses for other links to use for IK. This is useful when solving for
                bimanual robots or when solving for multiple links of the same robot.

        Returns:
            Updated goal buffer. This also updates the internal goal_buffer of IKSolver.
        """
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
                if is_cuda_graph_reset_available():
                    log_warn("changing goal type, breaking previous cuda graph.")
                    self.reset_cuda_graph()
                else:
                    log_error(
                        "changing goal type, cuda graph reset not available, "
                        + "consider updating to cuda >= 12.0"
                    )

            self.solver.update_nproblems(self._solve_state.get_ik_batch_size())
            self._goal_buffer.current_state = self.init_state.repeat_seeds(goal_pose.batch)
            self._col = torch.arange(
                0,
                self._goal_buffer.goal_pose.batch,
                device=self.tensor_args.device,
                dtype=torch.long,
            )

        return self._goal_buffer

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
        """Solve single IK problem.

        To get the closest IK solution from current joint configuration useful for IK servoing, set
        retract_config and seed_config to current joint configuration, also make sure IKSolverConfig
        was created with regularization enabled. If the solution is still not sufficiently close to
        the current joint configuration, increase the weight for `null_space_cfg` in `convergence`
        of `base_cfg.yml` which will select a solution that is closer to the current
        joint configuration after optimization. You can also increase the weight of
        `null_space_weight` in `bound_cfg` of `gradient_ik.yml` to encourage the optimization to
        stay near the current joint configuration during iterations.

        Args:
            goal_pose: Pose to reach with end-effector.
            retract_config: Retract configuration to use as regularization for IK. For this to work,
                :meth:`IKSolverConfig.load_from_robot_config` should have regularization enabled.
                This should be of shape (1, dof), where dof is the number of degrees of freedom in
                the robot. The order of joints should match :meth:`IKSolver.joint_names`.
            seed_config: Initial seed configuration to use for optimization. If None, a random seed
                is generated. The n seeds passed should be of shape (n, 1, dof), where dof is
                the number of degrees of freedom in the robot. The number of seeds do not have to
                match the number of seeds in the IKSolver. The remaining seeds are generated
                randomly. The order of joints should match :meth:`IKSolver.joint_names`.
            return_seeds: Number of solutions to return for the IK problem.
            num_seeds: Number of seeds to optimize per IK problem in parallel. Changing number of
                seeds is not allowed when use_cuda_graph is enabled.
            use_nn_seed: Flag to use neural network as seed for IK. This is not implemented yet.
            newton_iters: Number of iterations to run LBFGS optimization. If None, the number
                of iterations is set to the default value in the configuration (`gradient_ik.yml`).
            link_poses: Poses for other links to use for IK. This is useful when solving for
                bimanual robots or when solving for multiple links of the same robot. The link_poses
                should be a dictionary with link name as key and pose as value.

        Returns:
            :class:`IKResult` object with solution to the IK problem. Use :meth:`IKResult.success` to check
            if the problem was solved successfully.
        """
        if num_seeds is None:
            num_seeds = self.num_seeds
        if return_seeds > num_seeds:
            num_seeds = return_seeds

        solve_state = ReacherSolveState(
            ReacherSolveType.SINGLE, num_ik_seeds=num_seeds, batch_size=1, n_envs=1, n_goalset=1
        )

        return self._solve_from_solve_state(
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
        """Solve IK problem to reach one pose in a set of poses.

        To get the closest IK solution from current joint configuration useful for IK servoing, set
        retract_config and seed_config to current joint configuration, also make sure IKSolverConfig
        was created with regularization enabled. If the solution is still not sufficiently close to
        the current joint configuration, increase the weight for `null_space_cfg` in `convergence`
        of `base_cfg.yml` which will select a solution that is closer to the current
        joint configuration after optimization. You can also increase the weight of
        `null_space_weight` in `bound_cfg` of `gradient_ik.yml` to encourage the optimization to
        stay near the current joint configuration during iterations.

        Args:
            goal_pose: Pose to reach with end-effector.
            retract_config: Retract configuration to use as regularization for IK. For this to work,
                :meth:`IKSolverConfig.load_from_robot_config` should have regularization enabled.
                This should be of shape (1, dof), where dof is the number of degrees of freedom in
                the robot. The order of joints should match :meth:`IKSolver.joint_names`.
            seed_config: Initial seed configuration to use for optimization. If None, a random seed
                is generated. The n seeds passed should be of shape (n, 1, dof), where dof is
                the number of degrees of freedom in the robot. The number of seeds do not have to
                match the number of seeds in the IKSolver. The remaining seeds are generated
                randomly. The order of joints should match :meth:`IKSolver.joint_names`.
            return_seeds: Number of solutions to return for the IK problem.
            num_seeds: Number of seeds to optimize per IK problem in parallel. Changing number of
                seeds is not allowed when use_cuda_graph is enabled.
            use_nn_seed: Flag to use neural network as seed for IK. This is not implemented yet.
            newton_iters: Number of iterations to run LBFGS optimization. If None, the number
                of iterations is set to the default value in the configuration (`gradient_ik.yml`).
            link_poses: Poses for other links to use for IK. This is useful when solving for
                bimanual robots or when solving for multiple links of the same robot. The link_poses
                should be a dictionary with link name as key and pose as value.

        Returns:
            :class:`IKResult` object with solution to the IK problem. Check :meth:`IKResult.goalset_index` to
            identify which pose was reached by the solution. Use :meth:`IKResult.success` to check
            if the problem was solved successfully.
        """
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
        return self._solve_from_solve_state(
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
        """Solve batch of IK problems.

        Changing number of problems (batch size) is not allowed when use_cuda_graph is enabled. The
        number of problems is determined during the first call to this function.

        Args:
            goal_pose: Pose to reach with end-effector.
            retract_config: Retract configuration to use as regularization for IK. For this to work,
                :meth:`IKSolverConfig.load_from_robot_config` should have regularization enabled.
                This should be of shape (batch, dof), where dof is the number of degrees of freedom in
                the robot. The order of joints should match :meth:`IKSolver.joint_names`.
            seed_config: Initial seed configuration to use for optimization. If None, a random seed
                is generated. The n seeds passed should be of shape (n, batch, dof), where dof is
                the number of degrees of freedom in the robot, and batch is number of problems in
                batch. The number of seeds do not have to match the number of seeds in the IKSolver.
                The remaining seeds are generated randomly. The order of joints should match
                :meth:`IKSolver.joint_names`.
            return_seeds: Number of solutions to return per problem in batch.
            num_seeds: Number of seeds to optimize per IK problem in parallel. Changing number of
                seeds is not allowed when use_cuda_graph is enabled.
            use_nn_seed: Flag to use neural network as seed for IK. This is not implemented yet.
            newton_iters: Number of iterations to run LBFGS optimization. If None, the number
                of iterations is set to the default value in the configuration (`gradient_ik.yml`).
            link_poses: Poses for other links to use for IK. This is useful when solving for
                bimanual robots or when solving for multiple links of the same robot. The link_poses
                should be a dictionary with link name as key and pose as value.

        Returns:
            :class:`IKResult` object with solution to the batch of IK problems. Use :meth:`IKResult.success`
            to filter successful solutions.
        """
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
        return self._solve_from_solve_state(
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
        """Solve batch of IK problems to reach one pose in a set of poses for a batch of problems.

        Args:
            goal_pose: Pose to reach with end-effector.
            retract_config: Retract configuration to use as regularization for IK. For this to work,
                :meth:`IKSolverConfig.load_from_robot_config` should have regularization enabled.
                This should be of shape (batch, dof), where dof is the number of degrees of freedom in
                the robot. The order of joints should match :meth:`IKSolver.joint_names`.
            seed_config: Initial seed configuration to use for optimization. If None, a random seed
                is generated. The n seeds passed should be of shape (n, batch, dof), where dof is
                the number of degrees of freedom in the robot, and batch is number of problems in
                batch. The number of seeds do not have to match the number of seeds in the IKSolver.
                The remaining seeds are generated randomly. The order of joints should match
                :meth:`IKSolver.joint_names`.
            return_seeds: Number of solutions to return per problem in batch.
            num_seeds: Number of seeds to optimize per IK problem in parallel. Changing number of
                seeds is not allowed when use_cuda_graph is enabled.
            use_nn_seed: Flag to use neural network as seed for IK. This is not implemented yet.
            newton_iters: Number of iterations to run LBFGS optimization. If None, the number of
                iterations is set to the default value in the configuration (`gradient_ik.yml`).
            link_poses: Poses for other links to use for IK. This is useful when solving for
                bimanual robots or when solving for multiple links of the same robot. The link_poses
                should be a dictionary with link name as key and pose as value.

        Returns:
            :class:`IKResult` object with solution to the batch of IK problems. Use :meth:`IKResult.success`
            to filter successful solutions. Use :meth:`IKResult.goalset_index` to identify which
            pose was reached by each solution.
        """
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
        return self._solve_from_solve_state(
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
        """Solve batch of IK problems with each problem in different world environments.

        Args:
            goal_pose: Pose to reach with end-effector.
            retract_config: Retract configuration to use as regularization for IK. For this to work,
                :meth:`IKSolverConfig.load_from_robot_config` should have regularization enabled.
                This should be of shape (batch, dof), where dof is the number of degrees of freedom
                in the robot. The order of joints should match :meth:`IKSolver.joint_names`.
            seed_config: Initial seed configuration to use for optimization. If None, a random seed
                is generated. The n seeds passed should be of shape (n, batch, dof), where dof is
                the number of degrees of freedom in the robot, and batch is number of problems in
                batch. The number of seeds do not have to match the number of seeds in the IKSolver.
                The remaining seeds are generated randomly. The order of joints should match
                :meth:`IKSolver.joint_names`.
            return_seeds: Number of solutions to return per problem in batch.
            num_seeds: Number of seeds to optimize per IK problem in parallel. Changing number of
                seeds is not allowed when use_cuda_graph is enabled.
            use_nn_seed: Flag to use neural network as seed for IK. This is not implemented yet.
            newton_iters: Number of iterations to run LBFGS optimization. If None, the number of
                iterations is set to the default value in the configuration (`gradient_ik.yml`).
            link_poses: Poses for other links to use for IK. This is useful when solving for
                bimanual robots or when solving for multiple links of the same robot. The link_poses
                should be a dictionary with link name as key and pose as value.

        Returns:
            :class:`IKResult` object with solution to the batch of IK problems. Use :meth:`IKResult.success`
            to filter successful solutions.
        """
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
        return self._solve_from_solve_state(
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
        """Solve batch of goalset IK problems with each problem in different world environments.

        Args:
            goal_pose: Pose to reach with end-effector.
            retract_config: Retract configuration to use as regularization for IK. For this to work,
                :meth:`IKSolverConfig.load_from_robot_config` should have regularization enabled.
                This should be of shape (batch, dof), where dof is the number of degrees of freedom
                in the robot. The order of joints should match :meth:`IKSolver.joint_names`.
            seed_config: Initial seed configuration to use for optimization. If None, a random seed
                is generated. The n seeds passed should be of shape (n, batch, dof), where dof is
                the number of degrees of freedom in the robot, and batch is number of problems in
                batch. The number of seeds do not have to match the number of seeds in the IKSolver.
                The remaining seeds are generated randomly.  The order of joints should match
                :meth:`IKSolver.joint_names`.
            return_seeds: Number of solutions to return per problem in batch.
            num_seeds: Number of seeds to optimize per IK problem in parallel. Changing number of
                seeds is not allowed when use_cuda_graph is enabled.
            use_nn_seed: Flag to use neural network as seed for IK. This is not implemented yet.
            newton_iters: Number of iterations to run LBFGS optimization. If None, the number of
                iterations is set to the default value in the configuration (`gradient_ik.yml`).
            link_poses: Poses for other links to use for IK. This is useful when solving for
                bimanual robots or when solving for multiple links of the same robot. The link_poses
                should be a dictionary with link name as key and pose as value.

        Returns:
            :class:`IKResult` object with solution to the batch of IK problems. Use :meth:`IKResult.success`
            to filter successful solutions. Use :meth:`IKResult.goalset_index` to identify which
            pose was reached by each solution.
        """
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
        return self._solve_from_solve_state(
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

    def _solve_from_solve_state(
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
        """Solve IK problem from ReacherSolveState. Called by all solve functions.

        Args:
            solve_state: ReacherSolveState with information about the type of IK problem to solve.
            goal_pose: Pose to reach with end-effector.
            num_seeds: Number of seeds to optimize per IK problem in parallel.
            retract_config: Retract configuration to use as regularization for IK. For this to work,
                :meth:`IKSolverConfig.load_from_robot_config` should have regularization enabled.
                Shape of retract_config depends on type of IK problem being solved.
            seed_config: Initial seed configuration to use for optimization. If None, a random seed
                is generated. The n seeds passed should be of shape (n, batch, dof), where dof is
                the number of degrees of freedom in the robot, and batch is number of problems in
                batch. The number of seeds do not have to match the number of seeds in the IKSolver.
                The remaining seeds are generated randomly. The order of joints should match
                :meth:`IKSolver.joint_names`. When solving for single IK problem types, batch==1.
            return_seeds: Number of solutions to return per problem in batch.
            use_nn_seed: Flag to use neural network as seed for IK. This is not implemented yet.
            newton_iters: Number of iterations to run LBFGS optimization. If None, the number of
                iterations is set to the default value in the configuration (`gradient_ik.yml`).
            link_poses: Poses for other links to use for IK. This is useful when solving for
                bimanual robots or when solving for multiple links of the same robot. The link_poses
                should be a dictionary with link name as key and pose as value.

        Returns:
            :class:`IKResult` object with solution to the IK problem. Use :meth:`IKResult.success`
            to check if the problem was solved successfully.
        """
        # create goal buffer:
        goal_buffer = self._update_goal_buffer(solve_state, goal_pose, retract_config, link_poses)
        coord_position_seed = self.get_seed(
            num_seeds, goal_buffer.goal_pose, use_nn_seed, seed_config
        )

        if newton_iters is not None:
            self.solver.newton_optimizer.outer_iters = newton_iters
        self.solver.reset()
        result = self.solver.solve(goal_buffer, coord_position_seed)
        if newton_iters is not None:
            self.solver.newton_optimizer.outer_iters = self.og_newton_iters
        ik_result = self._get_result(num_seeds, result, goal_buffer.goal_pose, return_seeds)
        if ik_result.goalset_index is not None:
            ik_result.goalset_index[ik_result.goalset_index >= goal_pose.n_goalset] = 0

        return ik_result

    @profiler.record_function("ik/get_result")
    def _get_result(
        self, num_seeds: int, result: WrapResult, goal_pose: Pose, return_seeds: int
    ) -> IKResult:
        """Get IKResult from WrapResult after optimization.

        Args:
            num_seeds: number of seeds used for optimization.
            result: result from optimization.
            goal_pose: goal poses used for IK problems.
            return_seeds: number of seeds to return per problem.

        Returns:
            IKResult object with solutions to the IK problems.
        """
        success = self._get_success(result.metrics, num_seeds=num_seeds)
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
        """Get seed joint configurations for optimization.

        Args:
            num_seeds: number of seeds to generate.
            goal_pose: goal poses for IK problems. This is used to generate seeds with a
                neural network. Not implemented yet.
            use_nn_seed: flag to use neural network as seed for IK. This is not implemented yet.
            seed_config: seed configuration to use for optimization. If None, random seeds are
                generated. seed config should be of shape (batch, n, dof), where n can be lower
                or equal to num_seeds. The order of joints should match
                :meth:`IKSolver.joint_names`.

        Returns:
            seed joint configurations for optimization.
        """
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
        """Solve IK problem with any solve type."""
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
        """Deprecated API for solving single or batch problems."""
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
        """Deprecated API, use solve_batch_env() or solve_batch_env_goalset() instead."""
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
    def _get_success(self, metrics: RolloutMetrics, num_seeds: int) -> torch.Tensor:
        """Get success of IK optimization.

        Args:
            metrics: RolloutMetrics with feasibility, pose error, and other costs.
            num_seeds: Number of seeds used for IK optimization.

        Returns:
            Success of IK optimization as a boolean tensor of shape (batch, num_seeds).
        """
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
        """Generate seeds for IK optimization.

        Args:
            num_seeds: Number of seeds to generate using pseudo-random generator.
            batch: Number of problems in batch.
            use_nn_seed: Flag to use neural network as seed for IK. This is not implemented yet.
            pose: Pose to use for generating seeds. This is not implemented yet.

        Returns:
            Seed configurations for IK optimization.
        """
        num_random_seeds = num_seeds
        seed_list = []
        if use_nn_seed and self.ik_nn_seeder is not None:
            num_random_seeds = num_seeds - 1
            in_data = torch.cat((pose.position, pose.quaternion), dim=-1)
            nn_seed = self.ik_nn_seeder(in_data).view(batch, 1, self.dof)
            seed_list.append(nn_seed)
        if num_random_seeds > 0:
            random_seed = self.q_sample_gen.get_samples(
                num_random_seeds * batch,
                bounded=True,
            ).view(batch, num_random_seeds, self.dof)
            seed_list.append(random_seed)
        coord_position_seed = torch.cat(seed_list, dim=1)
        return coord_position_seed

    def update_world(self, world: WorldConfig) -> None:
        """Update world in IKSolver.

        If the new world configuration has more obstacles than initial cache, the collision cache
        will be recreated, breaking existing cuda graphs. This will lead to an exit with error if
        use_cuda_graph is enabled.

        Args:
            world: World configuration to update in IKSolver.
        """
        self.world_coll_checker.load_collision_model(world)

    def reset_seed(self) -> None:
        """Reset seed generator in IKSolver."""
        self.q_sample_gen.reset()

    def check_constraints(self, q: JointState) -> RolloutMetrics:
        """Check constraints for joint state.

        Args:
            q: Joint state to check constraints for.

        Returns:
            RolloutMetrics with feasibility of joint state.
        """
        metrics = self.rollout_fn.rollout_constraint(q.position.unsqueeze(1))
        return metrics

    def sample_configs(
        self,
        n: int,
        use_batch_env=False,
        sample_from_ik_seeder: bool = False,
        rejection_ratio: Optional[int] = None,
    ) -> torch.Tensor:
        """Sample n feasible joint configurations. Only samples with environment=0.

        Args:
            n: Number of joint configurations to sample.
            use_batch_env: Flag to sample from batch environments. This is not implemented yet.
            sample_from_ik_seeder: Flag to sample from IK seeder. This is not implemented yet.
            rejection_ratio: Ratio of samples to generate to get n feasible samples. If None, the
                rejection ratio is set to the default value meth:`IKSolver.sample_rejection_ratio`.

        Returns:
            Joint configurations sampled from the IK seeder of shape (n, dof).
        """

        if use_batch_env:
            log_warn(
                "IKSolver.sample_configs() does not work with batch environments,"
                + " sampling only from env=0"
            )
        if rejection_ratio is None:
            rejection_ratio = self.sample_rejection_ratio
        if sample_from_ik_seeder:
            samples = self.q_sample_gen.get_samples(n * rejection_ratio, bounded=True)
        else:
            samples = self.rollout_fn.sample_random_actions(n * rejection_ratio)
        metrics = self.rollout_fn.rollout_constraint(
            samples.unsqueeze(1), use_batch_env=use_batch_env
        )
        return samples[metrics.feasible.squeeze()][:n]

    @property
    def kinematics(self) -> CudaRobotModel:
        """Get kinematics instance in IKSolver."""
        return self.rollout_fn.dynamics_model.robot_model

    def get_all_rollout_instances(self) -> List[ArmReacher]:
        """Get all rollout instances in IKSolver.

        Returns:
            List of all rollout instances in IKSolver.
        """
        if self._rollout_list is None:
            self._rollout_list = [self.rollout_fn] + self.solver.get_all_rollout_instances()
        return self._rollout_list

    def get_all_kinematics_instances(self) -> List[CudaRobotModel]:
        """Deprecated API, use kinematics instead."""
        log_warn("IKSolver.get_all_kinematics_instances() is deprecated, use kinematics instead")

        return [self.kinematics for _ in range(len(self.get_all_rollout_instances))]

    def fk(self, q: torch.Tensor) -> CudaRobotModelState:
        """Forward kinematics for the robot.

        Args:
            q: Joint configuration of the robot, with joint values in order of :meth:`joint_names`.

        Returns:
            :class:`CudaRobotModelState` with link poses, and link spheres for the robot.
        """
        return self.kinematics.get_state(q)

    def reset_cuda_graph(self) -> None:
        """Reset the cuda graph for all rollout instances in IKSolver. Does not work currently."""
        self.solver.reset_cuda_graph()
        self.rollout_fn.reset_cuda_graph()

    def reset_shape(self):
        """Reset the shape of the rollout function and solver to the original shape."""
        self.solver.reset_shape()
        self.rollout_fn.reset_shape()

    def attach_object_to_robot(
        self,
        sphere_radius: float,
        sphere_tensor: Optional[torch.Tensor] = None,
        link_name: str = "attached_object",
    ) -> None:
        """Attach object to robot for collision checking.

        Args:
            sphere_radius: Radius of the sphere to attach to robot.
            sphere_tensor: Tensor of shape (n, 4) to attach to robot, where n is the number of
                spheres for that link. If None, only radius is updated for the existing spheres.
            link_name: Name of the link to attach object spheres to.
        """

        # get single kinematics instance:
        self.kinematics.kinematics_config.attach_object(
            sphere_radius=sphere_radius, sphere_tensor=sphere_tensor, link_name=link_name
        )

    def detach_object_from_robot(self, link_name: str = "attached_object") -> None:
        """Detach all attached objects from robot.

        This currently will reset the spheres for link_name to -10, disabling collision checking
        for that link.

        Args:
            link_name: Name of the link to detach object from.
        """

        self.kinematics.kinematics_config.detach_object(link_name=link_name)

    def get_retract_config(self) -> T_DOF:
        """Get retract configuration from robot model."""
        return self.rollout_fn.dynamics_model.retract_config

    def update_pose_cost_metric(
        self,
        metric: PoseCostMetric,
    ):
        """Update pose cost metric for all rollout instances in IKSolver.

        Args:
            metric: New values for Pose cost metric to update.
        """
        rollouts = self.get_all_rollout_instances()
        [
            rollout.update_pose_cost_metric(metric)
            for rollout in rollouts
            if isinstance(rollout, ArmReacher)
        ]

    def get_full_js(self, active_js: JointState) -> JointState:
        """Get full joint state from controlled joint state, appending locked joints.

        Args:
            active_js: Controlled joint state

        Returns:
            JointState: Full joint state.
        """
        return self.rollout_fn.get_full_dof_from_solution(active_js)

    def get_active_js(
        self,
        in_js: JointState,
    ):
        """Get controlled joint state from input joint state.

        This is used to get the joint state for only joints that are optimization variables. This
        also re-orders the joints to match the order of optimization variables.

        Args:
            in_js: Input joint state.

        Returns:
            JointState: Active joint state.
        """
        opt_jnames = self.rollout_fn.joint_names
        opt_js = in_js.get_ordered_joint_state(opt_jnames)
        return opt_js

    @property
    def joint_names(self) -> List[str]:
        """Get ordered names of all joints used in optimization with IKSolver."""
        return self.rollout_fn.kinematics.joint_names

    def check_valid(self, joint_position: torch.Tensor) -> torch.Tensor:
        """Check if joint position is valid. Also supports batch of joint positions.

        Args:
            joint_position: input position tensor of shape (batch, dof).

        Returns:
            boolean tensor of shape (batch) indicating if the joint position is valid.
        """
        if len(joint_position.shape) == 1:
            joint_position = joint_position.unsqueeze(0)
        if len(joint_position.shape) > 2:
            log_error("joint_position should be of shape (batch, dof)")
        metrics = self.rollout_fn.rollout_constraint(
            joint_position.unsqueeze(1),
            use_batch_env=False,
        )
        feasible = metrics.feasible.squeeze(1)
        return feasible


@get_torch_jit_decorator()
def get_success(
    feasible,
    position_error,
    rotation_error,
    num_seeds: int,
    position_threshold: float,
    rotation_threshold: float,
):
    """JIT compatible function to get the success of IK solutions."""
    feasible = feasible.view(-1, num_seeds)
    converge = torch.logical_and(
        position_error <= position_threshold,
        rotation_error <= rotation_threshold,
    ).view(-1, num_seeds)
    success = torch.logical_and(feasible, converge)
    return success


@get_torch_jit_decorator()
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
    """JIT compatible function to get the best IK solutions."""
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
