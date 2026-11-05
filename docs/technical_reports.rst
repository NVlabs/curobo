.. _research_page:

Technical Reports
=========================

cuRoboV2-Fast: Fast Torque-Limited Collision-Free Motion Planning
-----------------------------------------------------------------
*Balakumar Sundaralingam, Adithyavairavan Murali, Stan Birchfield*

[`Project page <https://research.nvidia.com/labs/lpr/curobov2-fast/>`__]

**Abstract**

We present cuRoboV2-Fast, a GPU-accelerated planner that reduces both planning latency and
execution time for collision-free, payload-aware, torque-limited motions. B-spline trajectory
optimization reduces the required iterations, while accelerated robot computations and
differentiable inverse dynamics reduce refinement cost and accommodate changing payloads.
TimeOpt optimizes traversal speed along the fixed path under sampled velocity, acceleration,
jerk, and torque constraints, exploiting local spline support to solve banded Newton systems
efficiently on the GPU. With a 3 kg payload, cuRoboV2-Fast achieves 4.4 ms mean planning time on
RTX 5090, 14.7× faster than cuRoboV2. Our method on Jetson AGX Orin plans faster than
cuRoboV1 + TOPPRA on RTX 5090. A stereo-camera-based deployment on YAM achieves 5.93 s mean
completed pick-and-drop cycles, 34.6% shorter than cuRoboV2.


cuRoboV2: Dynamics-Aware Motion Generation with Depth-Fused Distance Fields for High-DoF Robots
-------------------------------------------------------------------------------------------------
*Balakumar Sundaralingam, Adithyavairavan Murali, Stan Birchfield*

[`Arxiv <https://arxiv.org/abs/2603.05493>`__] [`Code <https://github.com/NVlabs/curobo>`__]

cuRoboV2 extends cuRobo with dynamics-aware motion generation via B-spline trajectory
optimization and differentiable inverse dynamics (RNEA) for torque limit enforcement. It
introduces a GPU-native depth-fused signed distance field (ESDF/TSDF) pipeline, replacing the
external nvblox dependency. cuRoboV2 also scales to high-DoF robots including full humanoids
(up to 48 DoF) through a unified type-generic Warp collision kernel and improved self-collision
algorithms.

.. include:: snippets/citation_v2.rst


cuRobo: Parallelized Collision-Free Minimum-Jerk Robot Motion Generation
-------------------------------------------------------------------------
*Balakumar Sundaralingam,
Siva Kumar Sastry Hari,
Adam Fishman,
Caelan Garrett,
Karl Van Wyk,
Valts Blukis,
Alexander Millane,
Helen Oleynikova,
Ankur Handa,
Fabio Ramos,
Nathan Ratliff,
Dieter Fox*

[`Arxiv <https://arxiv.org/abs/2310.17274>`__] [`GTC Talk <https://www.nvidia.com/gtc/session-catalog/?tab.allsessions=1700692987788001F1cG&search=S62122#/session/1694550714094001UhGc>`_] [`Code <https://github.com/NVlabs/curobo/tree/v0.7.8>`__]


.. raw:: html

        <iframe width="560" height="315" src="https://www.youtube.com/embed/Ux9tFBRLR-A?si=S7-SAAn_TJyaF0bz&amp;rel=0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share; modestbranding" allowfullscreen></iframe>


This paper explores the problem of collision-free motion generation for manipulators by formulating it as a global motion optimization problem. We develop a parallel optimization technique to solve this problem  and demonstrate its effectiveness on massively parallel GPUs. We show that combining simple optimization techniques with many parallel seeds leads to solving difficult motion generation problems within 50ms on average, 60x faster than state-of-the-art (SOTA) trajectory optimization methods. We achieve SOTA performance by combining L-BFGS step direction estimation with a novel parallel noisy line search scheme and a particle-based optimization solver. To further aid trajectory optimization, we develop a parallel geometric planner that is 101x faster than SOTA RRTConnect implementations and also introduce a collision-free IK solver that can solve over 7000 queries/s. We package our contributions into a state of the art GPU accelerated motion generation library, cuRobo and release it to enrich the robotics community.


.. include:: snippets/citation.rst

An initial implementation of cuRobo without minimum jerk optimization was published at ICRA 2023

.. code:: bibtex

    @INPROCEEDINGS{curobo_icra23,
    author={Sundaralingam, Balakumar and Hari, Siva Kumar Sastry and
            Fishman, Adam and Garrett, Caelan and Van Wyk, Karl and Blukis, Valts and
            Millane, Alexander and Oleynikova, Helen and Handa, Ankur and
            Ramos, Fabio and Ratliff, Nathan and Fox, Dieter},
    booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
    title={CuRobo: Parallelized Collision-Free Robot Motion Generation},
    year={2023},
    volume={},
    number={},
    pages={8112-8119},
    doi={10.1109/ICRA48891.2023.10160765}}

