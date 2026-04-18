########################################
cuRobo: CUDA Accelerated Robot Library
########################################

[:ref:`API <api>`] [`GTC Talk <https://www.nvidia.com/gtc/session-catalog/?tab.allsessions=1700692987788001F1cG&search=S62122#/session/1694550714094001UhGc>`_] [`Code <https://github.com/nvlabs/curobo>`_] [:doc:`Paper <technical_reports>`]

A unified, dynamics-aware GPU stack for safe, feasible, and reactive robot
motion generation, scaling from single-arm manipulators to high-DoF humanoids.
cuRobo provides trajectory optimization, GPU-native perception, and whole-body
computation built on PyTorch, CUDA, and Warp.

.. raw:: html

   <figure>
   <video autoplay loop muted playsinline preload="auto" width="100%">
     <source src="videos/tsdf_esdf_dual_rgbd.webm" type="video/webm">
   </video>
   <figcaption>GPU-native perception from dual RGBD streams: TSDF at 5 mm in 1.2 ms, ESDF at 2 cm in 0.2 ms (1.4 ms total).</figcaption>
   </figure>


.. raw:: html

   <figure>
   <video autoplay loop muted playsinline preload="auto" width="100%">
     <source src="videos/humanoid_run.webm" type="video/webm">
   </video>
   <figcaption>Whole-body policy trained in MimicKit (Newton) using retargeted motions from cuRoboV2.</figcaption>
   </figure>


.. include:: news.rst




Documentation
=============

- :doc:`getting-started/index`: Step-by-step tutorials covering installation, robot setup, kinematics, IK, MPC, motion planning, and volumetric mapping.
- :doc:`guides/index`: Writing custom optimization problems, cost terms, and optimizers.
- :doc:`concepts/index`: How cuRobo's rollouts, optimization solvers, and geometric planner work.
- :doc:`reference/index`: API, style guide, runtime configuration, benchmarks, sphere fitting, and self-collision.

Research
========

Read the cuRobo technical reports for algorithm details and benchmarks.

:doc:`technical_reports`

.. toctree::
   :hidden:
   :maxdepth: 2

   getting-started/index
   guides/index
   concepts/index
   reference/index
   technical_reports
