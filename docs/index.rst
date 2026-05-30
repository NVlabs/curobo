########################################
cuRobo: CUDA Accelerated Robot Library
########################################

[:ref:`API <api>`] [`GTC Talk <https://www.nvidia.com/gtc/session-catalog/?tab.allsessions=1700692987788001F1cG&search=S62122#/session/1694550714094001UhGc>`_] [`Code <https://github.com/nvlabs/curobo>`_] [:doc:`Paper <technical_reports>`]

cuRobo is a CUDA-accelerated library for robot motion generation, built on
PyTorch, CUDA, and Warp. It provides GPU-parallel algorithms for forward/inverse
kinematics, collision checking, trajectory optimization, geometric planning,
GPU-native perception, and whole-body motion generation, scaling from single-arm
manipulators to high-DoF humanoids.

.. raw:: html

   <figure>
   <video autoplay loop muted playsinline controls preload="auto" width="100%">
     <source src="videos/dual_rgbd_feature_mapping.webm" type="video/webm">
   </video>
   <figcaption>GPU-native perception from dual RGBD streams, integrating C-RADIO features into TSDF blocks in 2ms.</figcaption>
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
