
.. _rollout_class_note:

Rollout Classes
================================

A rollout in optimal control refers to the computation of costs and constraints from given
optimization variables; these optimization variables are often a sequence of actions that a robot
will take from a given initial state at fixed timesteps ``dt``. The length of the sequence is the
rollout horizon. In some cases, the horizon of the action will be different from the rollout.
For example, if the robot needs to be static at the final timestep, we execute the last action
for a few more timesteps to ensure the robot is static.

.. digraph:: Rollout
   :align: center
   :caption: Rollout Flow

   node [shape="box", style="filled", color="#76b900", fontcolor="white", fontname="sans-serif"];
   edge [color="#2B4162", fontname="sans-serif", fontsize=10];

   subgraph cluster_rollout {
   label="Evaluate Action";
   style="rounded, dashed";
   color="#558c8c";

   action [label="Action Sequence", shape="box", color="#cccccc", style="filled", fontcolor="black"];
   compute_state [label="compute_state_from_action(act_seq)"];
   transition_model [label="Transition Model\n (Optional)", shape="box3d", color="#934337", style="filled", fontcolor="white"];
   state [label="Robot State", shape="box", color="#cccccc", style="filled", fontcolor="black"];

   compute_costs [label="compute_costs_and_constraints(state)"];
   cost_manager [label="CostManager\n (Optional)", shape="box3d", color="#934337", style="filled", fontcolor="white"];
   costs_constraints [label="Costs and Constraints", shape="box", color="#cccccc", style="filled", fontcolor="black"];
   result [label="RolloutResult", shape="box", color="#558c8c", style="filled", fontcolor="white"];
   }
   state_2 [label="Robot State", shape="box", color="#cccccc", style="filled", fontcolor="black"];

   subgraph cluster_metrics {
   label="Compute Metrics";
   style="rounded, dashed";
   color="#558c8c";

   convergence_result [label="RolloutMetrics", shape="box", color="#558c8c", style="filled", fontcolor="white"];
   compute_convergence_metrics [label="compute_convergence_metrics(state)"];
   }

   action -> compute_state;
   compute_state -> state;
   state -> compute_costs;
   compute_costs -> cost_manager[dir="both"];
   compute_state -> transition_model[dir="both"];
   compute_costs -> costs_constraints;
   costs_constraints -> result;
   state_2 -> compute_convergence_metrics [label="Robot State"];
   compute_convergence_metrics -> convergence_result;

In cuRobo, evaluation of action sequences is often done for a batch of action sequences as cuRobo
optimizes over different seeds/variations in parallel to find the best action sequence. For e.g.,
when using sampling based methods like :py:class:`~curobo._src.optim.particle.mppi.MPPI`,
evaluations are done across multiple particles to find the best action sequence. When using
gradient-based methods like :py:class:`~curobo._src.optim.gradient.lbfgs.LBFGSOpt`, evaluations
are done across step sizes (parallel line search). In addition to the batching due to optimization
solvers, we also need to batch across multiple seeds and also across different optimization
targets.

.. _rollout_protocol:

The Rollout Protocol
--------------------

cuRobo rollouts are defined structurally rather than through inheritance. Any class whose public
surface matches the :py:class:`~curobo._src.rollout.rollout_protocol.Rollout`
:class:`~typing.Protocol` can be passed wherever the solvers expect a rollout -- no base class is
required, and there is no ``RolloutBase`` to subclass.

A rollout must expose the following **properties** (all read-only):

- ``action_dim`` -- dimensionality of a single action.
- ``action_horizon`` -- number of timesteps in an action sequence.
- ``action_bound_lows`` / ``action_bound_highs`` -- per-dimension action limits, shape
  ``(action_dim,)``.
- ``dt`` -- integration timestep (seconds).
- ``sum_horizon`` -- whether costs are summed across the horizon before being returned.

and the following **methods**:

- ``evaluate_action(act_seq)`` -- the core forward pass used every optimizer iteration; returns a
  :py:class:`~curobo._src.rollout.metrics.RolloutResult`.
- ``compute_metrics_from_state(state)`` / ``compute_metrics_from_action(act_seq)`` -- compute
  full :py:class:`~curobo._src.rollout.metrics.RolloutMetrics` (costs, constraints, convergence)
  after optimization finishes.
- ``update_params`` / ``update_batch_size`` / ``update_dt`` -- lifecycle hooks the solver calls
  between solves to retarget, resize internal buffers, or change the timestep.
- ``reset`` / ``reset_shape`` / ``reset_seed`` -- clear per-problem state, cached goal shapes, and
  the Halton sampler respectively.

Separating ``evaluate_action`` from ``compute_metrics_from_*`` lets the optimizer hot-loop stay
tight while post-optimization analysis can afford to compute extra information. For inverse
kinematics, for example, ``compute_metrics_from_state`` reports position and rotation error
separately even though the optimizer only needed their weighted sum.

Two reference implementations ship with cuRobo:

- :py:class:`~curobo._src.rollout.rollout_rosenbrock.RosenbrockRollout` -- a minimal
  implementation over the Rosenbrock function, intended as a pedagogical example and as a test
  rollout for the solvers. See :ref:`tut_user_rollout_optimization`.
- :py:class:`~curobo._src.rollout.rollout_robot.RobotRollout` -- the realistic rollout used
  throughout cuRobo, composing a transition model, a
  :py:class:`~curobo._src.rollout.cost_manager.cost_manager_robot.RobotCostManager`, and a scene
  collision checker.


CUDA Graph Acceleration
-----------------------

cuRobo supports `CUDA graphs <https://developer.nvidia.com/blog/cuda-graphs/>`_ to reduce the
overhead of kernel launches during the hot loop. CUDA graphs require the shape of inputs and
outputs to be constant across replays.

CUDA-graph acceleration is a **constructor parameter**, not a subclass or mixin. Pass
``use_cuda_graph=True`` when constructing the rollout:

.. code-block:: python

   from curobo._src.rollout.rollout_rosenbrock import RosenbrockCfg, RosenbrockRollout

   rollout = RosenbrockRollout(config, use_cuda_graph=True)

Internally the rollout wraps the post-optimization hooks
(``compute_metrics_from_state`` and ``compute_metrics_from_action``) in
:py:class:`~curobo._src.util.cuda_graph_util.GraphExecutor` instances. The graph is recorded
lazily on the first call and replayed on every subsequent call. When
``use_cuda_graph=False`` (the default), ``GraphExecutor`` transparently falls back to a direct
function call, so the same rollout class runs with or without graphs.

Two CUDA graphs are captured per rollout when ``use_cuda_graph=True``:

1. One for :py:meth:`~curobo._src.rollout.rollout_protocol.Rollout.compute_metrics_from_state`.
2. One for :py:meth:`~curobo._src.rollout.rollout_protocol.Rollout.compute_metrics_from_action`.

A separate CUDA graph for the inner ``evaluate_action`` loop is captured inside the optimizer
(see :ref:`optimization_solver_note`), since the optimizer's batch size typically differs from
the post-optimization batch size. Because it is also a constructor parameter
(``LBFGSOpt(config, rollout_list, use_cuda_graph=True)``), user rollout code does not need to
opt in; simply satisfying the :py:class:`~curobo._src.rollout.rollout_protocol.Rollout`
Protocol is enough. See :ref:`tut_rosen_rollout_optimization_cuda_graphs` for a worked example.


Cost Managers
--------------

When a rollout has many cost and constraint terms, listing all of them inline in ``evaluate_action``
becomes unwieldy. cuRobo provides a single flat class,
:py:class:`~curobo._src.rollout.cost_manager.cost_manager_robot.RobotCostManager`, that owns every
cost term used by :py:class:`~curobo._src.rollout.rollout_robot.RobotRollout`. There is no cost
manager hierarchy -- ``RobotCostManager`` is not abstract and has no subclasses.

The manager is **config-driven**:
:py:class:`~curobo._src.rollout.cost_manager.cost_manager_robot_cfg.RobotCostManagerCfg` has one
optional field per cost kind:

- ``self_collision_cfg``
- ``scene_collision_cfg``
- ``cspace_cfg``
- ``start_cspace_dist_cfg``
- ``target_cspace_dist_cfg``
- ``tool_pose_cfg``

Setting any of these to ``None`` disables that cost for the rollout. At construction time the
manager instantiates the corresponding :py:class:`~curobo._src.cost.cost_base.BaseCost` objects,
registers them via ``register_cost(name, component)``, and allocates a dedicated CUDA
stream/event pair for each via
:py:mod:`~curobo._src.util.cuda_stream_util` so the enabled costs can be evaluated in parallel.

A single :py:meth:`~curobo._src.rollout.cost_manager.cost_manager_robot.RobotCostManager.compute_costs`
call evaluates every enabled cost component inline (no ``super()`` chain) and returns a
:py:class:`~curobo._src.rollout.metrics.CostCollection`. Individual terms can still be toggled at
runtime through ``enable_cost_component(name)`` / ``disable_cost_component(name)``.

.. digraph:: CostManager
   :align: center
   :caption: Cost Manager Architecture

   node [shape="box", style="filled", color="#76b900", fontcolor="white", fontname="sans-serif"];
   edge [color="#2B4162", fontname="sans-serif", fontsize=10];

   state [label="Robot State", shape="box", color="#cccccc", style="filled", fontcolor="black"];
   manager [label="RobotCostManager"];

   subgraph cluster_costs {
      label="Registered Costs (enabled subset)";
      style="rounded, dashed";
      color="#558c8c";

      cost_a [label="self_collision"];
      cost_b [label="scene_collision"];
      cost_c [label="cspace"];
      cost_d [label="start_cspace_dist"];
      cost_e [label="target_cspace_dist"];
      cost_f [label="tool_pose"];
   }

   collection [label="CostCollection", shape="box", color="#cccccc", style="filled", fontcolor="black"];

   state -> manager;
   manager -> cost_a;
   manager -> cost_b;
   manager -> cost_c;
   manager -> cost_d;
   manager -> cost_e;
   manager -> cost_f;
   cost_a -> collection;
   cost_b -> collection;
   cost_c -> collection;
   cost_d -> collection;
   cost_e -> collection;
   cost_f -> collection;

For a step-by-step guide showing how ``RobotCostManager`` is wired into a rollout, see
:ref:`tut_motion_optimization`.


Available Rollout Classes
-------------------------

cuRobo ships two rollout classes, both of which satisfy the
:py:class:`~curobo._src.rollout.rollout_protocol.Rollout` Protocol.

.. list-table::
   :widths: 30 70
   :header-rows: 1
   :align: left

   * - Rollout Class
     - Description
   * - :py:class:`~curobo._src.rollout.rollout_rosenbrock.RosenbrockRollout`
     - Minimal pedagogical rollout over the Rosenbrock function. Useful for testing optimization
       algorithms and as a starting point for custom rollouts.
   * - :py:class:`~curobo._src.rollout.rollout_robot.RobotRollout`
     - Realistic rollout used across cuRobo's IK, trajectory optimization, and MPC pipelines.
       Composes a
       :py:class:`~curobo._src.transition.robot_state_transition.RobotStateTransition` model,
       a :py:class:`~curobo._src.rollout.cost_manager.cost_manager_robot.RobotCostManager`,
       and a :py:class:`~curobo._src.geom.collision.SceneCollision` checker.

To create your own rollout class and optimize it using cuRobo, see
:ref:`tut_user_rollout_optimization`.
