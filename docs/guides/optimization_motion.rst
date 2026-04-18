.. _tut_motion_optimization:

Writing Motion Optimization Problems
====================================

In the previous tutorial :ref:`tut_user_rollout_optimization`, we implemented a minimal rollout
class for the Rosenbrock function by hand. That was a good introduction to the
:py:class:`~curobo._src.rollout.rollout_protocol.Rollout` Protocol, but realistic motion-planning
problems involve a robot model, dozens of cost terms, self-collision and scene-collision
checks, and a transition model that turns joint actions into joint trajectories.

Rather than asking you to wire all of that up yourself, cuRobo ships a production-quality
rollout -- :py:class:`~curobo._src.rollout.rollout_robot.RobotRollout` -- that handles the full
stack. This tutorial is a guided tour of that rollout: what it composes, how it is configured,
and how to drive it from your own code. By the end you'll know:

1. What :py:class:`~curobo._src.rollout.rollout_robot.RobotRollout` composes and why.
2. How to build a rollout from a robot YAML through the
   :py:func:`~curobo._src.solver.solver_core_cfg.resolve_yaml_configs` /
   :py:func:`~curobo._src.solver.solver_core_cfg.create_solver_core_cfg` factory pipeline.
3. How the lifecycle hooks (goal updates, batching, resets) fit together across a solve.
4. How to toggle individual cost terms at runtime.
5. Where the high-level solvers (IK, trajectory optimization, MPC) sit on top of this rollout.

Prerequisites
-------------

Before proceeding, make sure you understand:

- The :py:class:`~curobo._src.rollout.rollout_protocol.Rollout` Protocol and the Rollout Flow
  diagram from :ref:`rollout_class_note`.
- The solver families described in :ref:`optimization_solver_note` -- in particular that every
  solver composes a rollout list.
- The basics of rollout authoring from :ref:`tut_user_rollout_optimization`.

What ``RobotRollout`` Composes
------------------------------

:py:class:`~curobo._src.rollout.rollout_robot.RobotRollout` is a standalone class (no base
class, no inheritance) that wires together four concerns:

- A :py:class:`~curobo._src.transition.robot_state_transition.RobotStateTransition` model that
  integrates action sequences into joint trajectories.
- One or more
  :py:class:`~curobo._src.rollout.cost_manager.cost_manager_robot.RobotCostManager` instances
  that evaluate every cost and constraint term used during optimization and metrics
  computation.
- A :py:class:`~curobo._src.geom.collision.SceneCollision` checker for world-collision queries
  that is shared across the cost managers.
- A :py:class:`~curobo._src.util.sampling.sample_buffer.SampleBuffer` Halton sampler that
  produces initial action seeds.

All of these are described by a single flat dataclass,
:py:class:`~curobo._src.rollout.rollout_robot_cfg.RobotRolloutCfg`. The fields mirror the
components one-to-one:

.. graphviz::
    :caption: RobotRolloutCfg fields and the RobotRollout components they configure

    digraph RobotRolloutComposition {
        edge [color="#2B4162", fontsize=10];
        node [shape="box", style="rounded, filled", fontsize=12, color="#cccccc"];

        subgraph cluster_cfg {
            label="RobotRolloutCfg";
            style="rounded, dashed";
            color="#558c8c";

            transition_model_cfg       [label="transition_model_cfg"];
            cost_cfg                   [label="cost_cfg"];
            constraint_cfg             [label="constraint_cfg"];
            hybrid_cost_constraint_cfg [label="hybrid_cost_constraint_cfg"];
            convergence_cfg            [label="convergence_cfg"];
            scene_collision_cfg        [label="scene_collision_cfg"];
        }

        subgraph cluster_rollout {
            label="RobotRollout";
            style="rounded, dashed";
            color="#558c8c";

            transition         [label="transition_model",         color="#76b900", fontcolor="white"];
            metrics_transition [label="metrics_transition_model", color="#76b900", fontcolor="white"];
            cost_mgr           [label="cost_manager",             color="#76b900", fontcolor="white"];
            constraint_mgr     [label="constraint_manager",       color="#76b900", fontcolor="white"];
            hybrid_mgr         [label="hybrid_cost_constraint_manager", color="#76b900", fontcolor="white"];
            convergence_mgr    [label="metrics_convergence_manager",    color="#76b900", fontcolor="white"];
            collision          [label="scene_collision_checker",  color="#76b900", fontcolor="white"];
        }

        transition_model_cfg       -> transition;
        transition_model_cfg       -> metrics_transition;
        cost_cfg                   -> cost_mgr;
        constraint_cfg             -> constraint_mgr;
        hybrid_cost_constraint_cfg -> hybrid_mgr;
        convergence_cfg            -> convergence_mgr;
        scene_collision_cfg        -> collision;
        collision                  -> cost_mgr            [style="dashed", label="shared"];
        collision                  -> constraint_mgr      [style="dashed", label="shared"];
        collision                  -> hybrid_mgr          [style="dashed", label="shared"];
        collision                  -> convergence_mgr     [style="dashed", label="shared"];
    }

Each cost-manager slot (``cost_cfg``, ``constraint_cfg``,
``hybrid_cost_constraint_cfg``, ``convergence_cfg``) is optional. Setting one to ``None``
means that manager is not constructed. The usual configuration uses
``cost_cfg`` + ``convergence_cfg``; constraint-heavy solvers additionally populate
``constraint_cfg``; and tasks that want different cost weights for costs-vs-constraints put
them in ``hybrid_cost_constraint_cfg``.

The transition model
--------------------

The transition model advances joint state under an action sequence. ``RobotRollout`` keeps two
instances of the same transition model: one for the optimizer's forward pass
(``transition_model``) and one for the post-optimization metrics path
(``metrics_transition_model``). They are constructed from the same
:py:class:`~curobo._src.transition.robot_state_transition_cfg.RobotStateTransitionCfg` but can
hold different batch shapes -- the metrics model typically serves a single evaluation after a
solve, while the optimizer model serves many parallel particles per iteration.

Swapping in a different transition model (for example, a custom dynamics model) is done by
setting
:py:attr:`~curobo._src.rollout.rollout_robot_cfg.RobotRolloutCfg.transition_model_config_instance_type`
when constructing ``RobotRolloutCfg`` -- see
:py:meth:`~curobo._src.rollout.rollout_robot_cfg.RobotRolloutCfg.create_with_component_types`.

The cost managers
-----------------

Each cost-manager slot contains a
:py:class:`~curobo._src.rollout.cost_manager.cost_manager_robot_cfg.RobotCostManagerCfg`
dataclass. That dataclass has one optional field per cost *kind*:

- ``self_collision_cfg`` -- self-collision on the robot's sphere approximation.
- ``scene_collision_cfg`` -- world collision against the
  :py:class:`~curobo._src.geom.collision.SceneCollision` checker.
- ``cspace_cfg`` -- joint-space regularization (reaching a target configuration).
- ``start_cspace_dist_cfg`` / ``target_cspace_dist_cfg`` -- c-space distance to the start or
  target state.
- ``tool_pose_cfg`` -- task-space pose error at the tool frame.

Setting any of these to ``None`` disables that cost in the manager. On construction each
populated field is realized as a
:py:class:`~curobo._src.cost.cost_base.BaseCost` instance registered on the manager with a
dedicated CUDA stream/event pair (see :ref:`rollout_class_note` for the broader picture).

At evaluation time, a single call to
:py:meth:`~curobo._src.rollout.cost_manager.cost_manager_robot.RobotCostManager.compute_costs`
runs every enabled component inline (no ``super()`` chain) and returns a
:py:class:`~curobo._src.rollout.metrics.CostCollection`. Individual terms can still be toggled
at runtime via ``enable_cost_component(name)`` / ``disable_cost_component(name)`` -- the
manager remembers the registration but skips disabled components when collecting costs.

Configuring a rollout from YAML
-------------------------------

cuRobo's built-in task configurations live as YAML files under the bundled robot- and
task-config directories (for example ``franka.yml``, ``trajopt.yml``, ``ik.yml``). Rather than
constructing :py:class:`~curobo._src.rollout.rollout_robot_cfg.RobotRolloutCfg` by hand, the
usual path is a two-step factory:

.. code-block:: python

   from curobo._src.solver.solver_core_cfg import (
       create_solver_core_cfg,
       resolve_yaml_configs,
   )
   from curobo._src.types.device_cfg import DeviceCfg
   import torch

   device_cfg = DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)

   # 1. Resolve YAML paths to plain dicts and a loaded RobotCfg.
   (
       robot_config,
       optimizer_dicts,
       metrics_rollout_dict,
       transition_model_dict,
       scene_model_dict,
   ) = resolve_yaml_configs(
       robot="franka.yml",
       optimizer_configs=["mppi_cfg.yml", "lbfgs_cfg.yml"],
       metrics_rollout="metrics_rollout_cfg.yml",
       transition_model="transition_model_cfg.yml",
       scene_model=None,
       device_cfg=device_cfg,
   )

   # 2. Build a SolverCoreCfg.  This assembles one RobotRolloutCfg per optimizer stage
   #    plus a metrics RobotRolloutCfg, validates the resulting dataclasses, and wraps
   #    the scene-collision config, optimizer configs, and device settings into a single
   #    SolverCoreCfg.
   core_cfg = create_solver_core_cfg(
       robot_config=robot_config,
       optimizer_dicts=optimizer_dicts,
       metrics_rollout_dict=metrics_rollout_dict,
       transition_model_dict=transition_model_dict,
       scene_model_dict=scene_model_dict,
       device_cfg=device_cfg,
       use_cuda_graph=True,
   )

At this point ``core_cfg`` contains:

- ``core_cfg.optimizer_rollout_configs`` -- a list of
  :py:class:`~curobo._src.rollout.rollout_robot_cfg.RobotRolloutCfg`, one per optimizer stage.
- ``core_cfg.metrics_rollout_config`` -- a dedicated
  :py:class:`~curobo._src.rollout.rollout_robot_cfg.RobotRolloutCfg` for post-optimization
  metric evaluation.
- ``core_cfg.scene_collision_cfg`` -- the world-collision configuration shared across both.
- ``core_cfg.optimizer_configs`` -- the flat optimizer dataclasses (``MPPICfg``,
  ``LBFGSOptCfg``, etc.) for each stage.

If you want an isolated rollout (no solver), you can also skip ``create_solver_core_cfg`` and
use the lower-level factory
:py:meth:`~curobo._src.rollout.rollout_robot_cfg.RobotRolloutCfg.create_with_component_types`,
which builds a single :py:class:`~curobo._src.rollout.rollout_robot_cfg.RobotRolloutCfg` from a
dict plus an already-loaded :py:class:`~curobo._src.types.robot.RobotCfg`.

Constructing and using the rollout
----------------------------------

Once you have a :py:class:`~curobo._src.rollout.rollout_robot_cfg.RobotRolloutCfg` and a
:py:class:`~curobo._src.geom.collision.SceneCollision` checker, constructing the rollout is a
two-line operation:

.. code-block:: python

   from curobo._src.geom.collision import create_collision_checker
   from curobo._src.rollout.rollout_robot import RobotRollout

   scene_collision_checker = create_collision_checker(core_cfg.scene_collision_cfg)

   # Pick one of the optimizer rollout configs (or the metrics config).
   rollout_cfg = core_cfg.optimizer_rollout_configs[0]
   rollout = RobotRollout(
       config=rollout_cfg,
       scene_collision_checker=scene_collision_checker,
       use_cuda_graph=True,
   )

The rollout now exposes the full
:py:class:`~curobo._src.rollout.rollout_protocol.Rollout` Protocol. You can inspect action
geometry via the standard properties:

.. code-block:: python

   print(rollout.action_dim)         # Number of actuated joints.
   print(rollout.action_horizon)     # Length of an action sequence.
   print(rollout.action_bound_lows)  # Per-joint lower limits, shape (action_dim,).
   print(rollout.action_bound_highs) # Per-joint upper limits.
   print(rollout.dt)                 # Integration timestep in seconds.

and feed it action sequences of shape ``(batch, action_horizon, action_dim)``:

.. code-block:: python

   init_action = rollout.act_sample_gen.get_samples(
       n=batch_size * rollout.action_horizon, bounded=True
   ).view(batch_size, rollout.action_horizon, rollout.action_dim)

   result = rollout.evaluate_action(init_action)
   print(result.costs_and_constraints.get_sum_cost().shape)   # (batch, horizon)
   print(result.state.position.shape)                         # (batch, horizon, n_dof)

``RobotRollout.evaluate_action`` returns a
:py:class:`~curobo._src.rollout.metrics.RolloutResult` with the integrated state, the per-cost
values packed in a :py:class:`~curobo._src.rollout.metrics.CostsAndConstraints`, and the
original action sequence. After a solve finishes, call
:py:meth:`~curobo._src.rollout.rollout_robot.RobotRollout.compute_metrics_from_action` (or
``...from_state``) to get the richer :py:class:`~curobo._src.rollout.metrics.RolloutMetrics`,
which additionally reports feasibility and convergence tolerances.

Lifecycle: goals, batching, and timesteps
-----------------------------------------

Between solves, the rollout's runtime state is updated through four lifecycle hooks on the
Protocol: ``update_params``, ``update_batch_size``, ``update_dt``, and the ``reset*`` family.

Setting a goal
~~~~~~~~~~~~~~

``RobotRollout`` consumes goals through a
:py:class:`~curobo._src.rollout.goal_registry.GoalRegistry`. The registry holds the current
joint state, any target tool poses or joint configurations, and the per-environment index
buffers used for batched multi-environment solves. The solvers build a
:py:class:`~curobo._src.rollout.goal_registry.GoalRegistry` for you; if you are driving the
rollout directly, construct it with the target data your cost manager expects (for example
``tool_pose_cfg`` needs ``goal_tool_poses``, ``cspace_cfg`` needs ``goal_state``):

.. code-block:: python

   from curobo._src.rollout.goal_registry import GoalRegistry

   goal = GoalRegistry(current_js=start_joint_state, goal_tool_poses=target_pose)
   rollout.update_params(goal, num_particles=rollout_cfg_num_particles)

On the first call,
:py:meth:`~curobo._src.rollout.rollout_robot.RobotRollout.update_params` allocates and stores
both a particle-expanded goal (``_num_particles_goal``) and a metrics-sized goal
(``_metrics_goal``). Subsequent calls overwrite tensor *values* in place so the pre-allocated
buffers are reused -- this is what lets CUDA graphs replay correctly across solves.

Changing the batch size and timestep
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:py:meth:`~curobo._src.rollout.rollout_robot.RobotRollout.update_batch_size` resizes every
internal tensor that depends on ``batch_size * num_particles``. It is typically called by the
optimizer before each ``evaluate_action``; you only need to call it yourself if you are driving
the rollout outside of a solver.

:py:meth:`~curobo._src.rollout.rollout_robot.RobotRollout.update_dt` propagates a new
integration timestep to the transition model. Scalars affect every problem uniformly;
per-problem tensors let different environments march at different rates.

Resets
~~~~~~

Three reset hooks clear progressively more state:

- ``reset(reset_problem_ids=None)`` clears cost-manager per-problem buffers so the next solve
  starts with no residual state.
- ``reset_shape()`` drops any cached goal tensors so the next ``update_params`` reallocates
  buffers. Use this when the batch shape changes in a way that
  :py:meth:`~curobo._src.rollout.rollout_robot.RobotRollout.update_batch_size` can't handle on
  its own.
- ``reset_seed()`` resets the Halton sampler -- useful for reproducible experiments.

Toggling costs at runtime
-------------------------

The easiest way to experiment with which costs are active during a solve is to toggle them on
the live cost manager. Every optimization rollout exposes its managers as public attributes:

.. code-block:: python

   rollout.cost_manager.disable_cost_component("scene_collision")
   rollout.cost_manager.enable_cost_component("scene_collision")

The manager remembers the registration; a disabled component is simply skipped during
:py:meth:`~curobo._src.rollout.cost_manager.cost_manager_robot.RobotCostManager.compute_costs`.
You can also disable a whole slot at configuration time by setting the corresponding field
(``self_collision_cfg``, ``tool_pose_cfg``, etc.) to ``None`` on the
:py:class:`~curobo._src.rollout.cost_manager.cost_manager_robot_cfg.RobotCostManagerCfg`.

For adding *new* cost types (for example, a custom energy-penalty cost) without subclassing,
see :ref:`tut_custom_cost`.

Using ``RobotRollout`` via the high-level solvers
-------------------------------------------------

In practice, most users do not construct :py:class:`~curobo._src.rollout.rollout_robot.RobotRollout`
directly. cuRobo provides three public solvers that build
:py:class:`~curobo._src.solver.solver_core.SolverCore` -- and therefore the rollouts behind
it -- from a robot YAML in a few lines:

.. code-block:: python

   from curobo import (
       InverseKinematics, InverseKinematicsCfg,
       TrajectoryOptimizer, TrajectoryOptimizerCfg,
       ModelPredictiveControl, ModelPredictiveControlCfg,
   )

   # Inverse kinematics
   ik = InverseKinematics(InverseKinematicsCfg.create(robot="franka.yml"))
   ik_result = ik.solve_pose(goal_tool_poses=target_poses)

   # Trajectory optimization
   trajopt = TrajectoryOptimizer(TrajectoryOptimizerCfg.create(robot="franka.yml"))
   traj_result = trajopt.solve_pose(goal_tool_poses=target_poses)

   # Model predictive control
   mpc = ModelPredictiveControl(ModelPredictiveControlCfg.create(robot="franka.yml"))
   next_action = mpc.optimize_action_sequence(current_state)

Each of these wraps the same
:py:func:`~curobo._src.solver.solver_core_cfg.resolve_yaml_configs` /
:py:func:`~curobo._src.solver.solver_core_cfg.create_solver_core_cfg` pipeline used above, and
each exposes the underlying rollouts through ``solver.core.optimizer_rollouts`` and
``solver.core.metrics_rollout`` if you need to introspect or modify them between solves. Use
the direct ``RobotRollout`` path only when you need finer-grained control than the high-level
solvers expose.

Conclusion
----------

In this tutorial we:

1. Mapped the fields of :py:class:`~curobo._src.rollout.rollout_robot_cfg.RobotRolloutCfg` to
   the components of :py:class:`~curobo._src.rollout.rollout_robot.RobotRollout` -- a
   transition model, a set of
   :py:class:`~curobo._src.rollout.cost_manager.cost_manager_robot.RobotCostManager` instances,
   and a shared :py:class:`~curobo._src.geom.collision.SceneCollision` checker.
2. Walked through the two-step YAML-to-``SolverCoreCfg`` factory pipeline
   (:py:func:`~curobo._src.solver.solver_core_cfg.resolve_yaml_configs` then
   :py:func:`~curobo._src.solver.solver_core_cfg.create_solver_core_cfg`), and the alternative
   :py:meth:`~curobo._src.rollout.rollout_robot_cfg.RobotRolloutCfg.create_with_component_types`
   factory for single-rollout use cases.
3. Constructed a :py:class:`~curobo._src.rollout.rollout_robot.RobotRollout`, evaluated an
   action sequence, and read the resulting
   :py:class:`~curobo._src.rollout.metrics.RolloutResult` and
   :py:class:`~curobo._src.rollout.metrics.RolloutMetrics`.
4. Walked through the ``update_params`` / ``update_batch_size`` / ``update_dt`` / ``reset*``
   lifecycle and how the cost manager lets you toggle components at runtime.

For more information on specific components, refer to the API documentation:

- :py:class:`~curobo._src.rollout.rollout_robot.RobotRollout`
- :py:class:`~curobo._src.rollout.rollout_robot_cfg.RobotRolloutCfg`
- :py:class:`~curobo._src.transition.robot_state_transition.RobotStateTransition`
- :py:class:`~curobo._src.rollout.cost_manager.cost_manager_robot.RobotCostManager`
- :py:class:`~curobo._src.rollout.cost_manager.cost_manager_robot_cfg.RobotCostManagerCfg`
- :py:class:`~curobo._src.solver.solver_core_cfg.SolverCoreCfg`

Next Steps
----------

- :ref:`tut_custom_cost` -- add a new cost term to
  :py:class:`~curobo._src.rollout.cost_manager.cost_manager_robot.RobotCostManager` without
  subclassing.
- :doc:`/guides/optimization_problem` -- revisit the Rosenbrock example to see how the Protocol
  fits a minimal, textbook problem.
