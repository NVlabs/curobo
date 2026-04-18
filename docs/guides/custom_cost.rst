.. _tut_custom_cost:

Extending RobotCostManager with a Custom Cost
=============================================

:py:class:`~curobo._src.rollout.cost_manager.cost_manager_robot.RobotCostManager` ships with the
six cost terms exposed through
:py:class:`~curobo._src.rollout.cost_manager.cost_manager_robot_cfg.RobotCostManagerCfg`
(self-collision, scene-collision, c-space, start/target c-space distance, tool-pose). When you
need a term that is not in that list, the refactor removed the old "subclass the manager"
path -- there is no ``ArmBaseCostManager`` to extend. Instead, the manager is a plain Python
object you can register new costs on **without subclassing**.

This guide shows the full extension flow:

1. Subclass :py:class:`~curobo._src.cost.cost_base.BaseCost` to implement your cost math.
2. Pair it with a :py:class:`~curobo._src.cost.cost_base_cfg.BaseCostCfg` subclass for its
   runtime parameters (weight, any tunables).
3. Register it on a live
   :py:class:`~curobo._src.rollout.cost_manager.cost_manager_robot.RobotCostManager` via
   :py:meth:`~curobo._src.rollout.cost_manager.cost_manager_robot.RobotCostManager.register_cost`.
4. Call it from your own ``compute_costs``-like code, or re-run the solver's normal evaluation
   loop and read the cost out of the returned
   :py:class:`~curobo._src.rollout.metrics.CostCollection`.

We'll implement a **joint-velocity energy** cost that penalises large joint velocities along
the rollout horizon.

Prerequisites
-------------

You should be comfortable with the material in
:ref:`tut_motion_optimization`, particularly the "Cost Managers" and "Toggling costs at
runtime" sections. The example below assumes you have already constructed a
:py:class:`~curobo._src.rollout.rollout_robot.RobotRollout` as shown there.

Writing a custom ``BaseCost``
-----------------------------

Every cost inherits a tiny contract from
:py:class:`~curobo._src.cost.cost_base.BaseCost`. The important pieces are:

- ``forward(...)`` -- compute the raw cost tensor of shape
  ``(batch, horizon, n_terms)`` and return it (the ``n_terms`` axis is usually 1).
- ``_weight`` -- a tensor of shape ``(n_terms,)`` the manager multiplies into the cost.
- ``enable_cost`` / ``disable_cost`` -- flip ``_cost_enabled`` on or off so the manager can
  skip this component without removing it.
- ``setup_batch_tensors(batch_size, horizon)`` -- called by the manager when shapes change;
  override if you keep per-batch buffers.
- ``reset(reset_problem_ids=None, **kwargs)`` -- called at the start of each solve.

The cost itself is a ``@dataclass`` extending
:py:class:`~curobo._src.cost.cost_base_cfg.BaseCostCfg`. The base dataclass already holds
``weight``, ``device_cfg``, ``class_type``, ``convert_to_binary``, and ``use_grad_input``, so
you typically only add whatever tunables your cost needs.

.. code-block:: python

   from dataclasses import dataclass
   from typing import Type

   import torch

   from curobo._src.cost.cost_base import BaseCost
   from curobo._src.cost.cost_base_cfg import BaseCostCfg


   @dataclass
   class JointVelocityEnergyCostCfg(BaseCostCfg):
       """Configuration for JointVelocityEnergyCost.

       The ``weight`` field inherited from BaseCostCfg is broadcast against every
       degree of freedom; pass a scalar for uniform weighting or a length-n_dof
       list to weight joints individually.
       """

       #: Optional per-joint scaling that multiplies velocity before squaring.
       joint_scale: float = 1.0

       class_type: Type[BaseCost] = None  # set by JointVelocityEnergyCost below.


   class JointVelocityEnergyCost(BaseCost):
       """Sum-of-squares penalty on joint velocities along the rollout horizon."""

       def __init__(self, config: JointVelocityEnergyCostCfg):
           super().__init__(config)
           self._joint_scale = config.joint_scale

       def forward(self, joint_state) -> torch.Tensor:
           """Compute the velocity-energy cost.

           Args:
               joint_state: JointState with a ``velocity`` tensor of shape
                   ``(batch, horizon, n_dof)``.

           Returns:
               Cost tensor of shape ``(batch, horizon, 1)``.
           """
           velocities = joint_state.velocity * self._joint_scale
           # Sum squared velocities across joints -> per-(batch, horizon) cost.
           per_step_energy = (velocities ** 2).sum(dim=-1, keepdim=True)
           # The manager multiplies by self._weight after the fact; keep the math here
           # weight-free so toggling disable_cost() stays well-defined.
           return per_step_energy


   # Wire the config's class_type to the concrete class (the manager uses
   # ``config.class_type(config)`` to instantiate).
   JointVelocityEnergyCostCfg.class_type = JointVelocityEnergyCost

A couple of design notes:

- Do not apply ``self._weight`` inside ``forward``. The manager multiplies the weight on its
  side; keeping the cost weight-free lets ``disable_cost()`` zero out the weight transparently
  and keeps the cost reusable for other callers that want the raw value (for example
  ``CostCollection.add(unweighted, name, weight=...)``).
- Return shape should be ``(batch, horizon, 1)`` for a scalar cost term. The manager's
  :py:class:`~curobo._src.rollout.metrics.CostCollection` expects the trailing 1-dim so it can
  stack multiple single-scalar costs.
- ``setup_batch_tensors(batch_size, horizon)`` is a no-op for this cost because we don't keep
  per-batch buffers; the base class implementation is fine.

Registering with ``RobotCostManager``
-------------------------------------

Once the cost and its config exist, wire them into a live rollout. The manager is exposed on
the rollout as ``rollout.cost_manager`` (and, for post-optimization metrics, as
``rollout.metrics_cost_manager``).
:py:meth:`~curobo._src.rollout.cost_manager.cost_manager_robot.RobotCostManager.register_cost`
takes a name and a constructed ``BaseCost`` instance and allocates the cost its own CUDA
stream/event pair so it can run in parallel with the six built-in costs:

.. code-block:: python

   from curobo._src.types.device_cfg import DeviceCfg

   device_cfg = rollout.device_cfg  # already on CUDA

   velocity_cfg = JointVelocityEnergyCostCfg(
       weight=[0.1],             # scalar weight => shape (1,).
       device_cfg=device_cfg,
       joint_scale=1.0,
   )
   velocity_cost = JointVelocityEnergyCost(velocity_cfg)

   # Register on the optimizer manager...
   rollout.cost_manager.register_cost("joint_velocity_energy", velocity_cost)

   # ...and on the metrics manager if you want the cost to show up in
   # compute_metrics_from_action() too.
   metrics_cost = JointVelocityEnergyCost(velocity_cfg)
   rollout.metrics_cost_manager.register_cost("joint_velocity_energy", metrics_cost)

The two-manager setup is deliberate. The optimizer manager (``rollout.cost_manager``) runs on
every optimizer iteration with the optimizer's batch shape; the metrics manager
(``rollout.metrics_cost_manager``) runs once after the solve with the (smaller) evaluation
batch shape. Registering on both keeps the new cost active everywhere; registering on only
``cost_manager`` means the cost drives the optimizer but is not reported by
:py:meth:`~curobo._src.rollout.rollout_robot.RobotRollout.compute_metrics_from_action`.

Feeding state into the cost
---------------------------

The built-in ``RobotCostManager.compute_costs`` knows how to unpack the
:py:class:`~curobo._src.state.state_robot.RobotState` for each registered cost because every
built-in cost has a known signature.  Custom costs need a little more work -- the simplest
approach is to call your cost yourself after ``evaluate_action`` and add the result to the
returned :py:class:`~curobo._src.rollout.metrics.CostCollection`:

.. code-block:: python

   result = rollout.evaluate_action(action_sequence)
   extra = rollout.cost_manager.get_cost("joint_velocity_energy").forward(
       result.state.joint_state
   )
   result.costs_and_constraints.costs.add(
       extra * rollout.cost_manager.get_cost("joint_velocity_energy")._weight,
       "joint_velocity_energy",
   )

If you want the cost to be driven automatically by the optimizer instead, the cleanest path is
to expose it through the same
:py:class:`~curobo._src.rollout.cost_manager.cost_manager_robot_cfg.RobotCostManagerCfg` that
describes the rest of the manager. Define a ``joint_velocity_energy_cfg`` field on your own
subclass of ``RobotCostManagerCfg`` and extend
:py:meth:`~curobo._src.rollout.cost_manager.cost_manager_robot.RobotCostManager.initialize_from_config`
in a lightweight wrapper class to register the new cost. ``RobotRolloutCfg`` accepts a custom
manager type via its
:py:attr:`~curobo._src.rollout.rollout_robot_cfg.RobotRolloutCfg.cost_manager_config_instance_type`
and
:py:attr:`~curobo._src.rollout.rollout_robot_cfg.RobotRolloutCfg.transition_model_config_instance_type`
slots, and
:py:meth:`~curobo._src.rollout.rollout_robot_cfg.RobotRolloutCfg.create_with_component_types`
takes both as keyword arguments, so the extension is opt-in per rollout. This is more code
than post-hoc ``register_cost`` but keeps the per-iteration call signature inside
``compute_costs``.

Toggling the custom cost at runtime
-----------------------------------

Once registered, the cost uses the exact same lifecycle hooks as the built-in costs:

.. code-block:: python

   # Temporarily disable (e.g., during warm-up).
   rollout.cost_manager.disable_cost_component("joint_velocity_energy")

   # Re-enable later.
   rollout.cost_manager.enable_cost_component("joint_velocity_energy")

   # Inspect which components are live.
   rollout.cost_manager.get_enabled_costs()

Disabling flips ``_cost_enabled`` to False and zeros the stored weight copy, so disabled costs
contribute nothing to
:py:meth:`~curobo._src.rollout.cost_manager.cost_manager_robot.RobotCostManager.compute_costs`
without having to re-register them when you re-enable.

Conclusion
----------

Custom costs are the one inheritance relationship the refactor preserved, because
:py:class:`~curobo._src.cost.cost_base.BaseCost` is a thin public interface rather than a
framework-y god class. The recipe:

1. Subclass :py:class:`~curobo._src.cost.cost_base.BaseCost`, implement ``forward``, and leave
   weight multiplication to the manager.
2. Pair it with a :py:class:`~curobo._src.cost.cost_base_cfg.BaseCostCfg` subclass for runtime
   parameters.
3. Call
   :py:meth:`~curobo._src.rollout.cost_manager.cost_manager_robot.RobotCostManager.register_cost`
   on the live manager (and on the metrics manager if you want the cost reported after the
   solve).
4. Toggle with ``enable_cost_component`` / ``disable_cost_component`` exactly as you would for
   a built-in cost.

When the custom cost stabilises and you want it wired through YAML configs the same way the
built-in costs are, promote it to a
:py:class:`~curobo._src.rollout.cost_manager.cost_manager_robot_cfg.RobotCostManagerCfg`
subclass field and plug that subclass into
:py:meth:`~curobo._src.rollout.rollout_robot_cfg.RobotRolloutCfg.create_with_component_types`.

See Also
--------

- :ref:`rollout_class_note` -- how the cost manager fits inside the rollout.
- :ref:`tut_motion_optimization` -- the production rollout the manager plugs into.
- :ref:`optimization_solver_note` -- the optimizers that consume the combined cost.
