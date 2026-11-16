.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _reference_robot_pose_calibration:

Robot Pose Calibration
======================

``curobo.examples.reference.robot_pose_calibration`` demonstrates robot pose estimation
in an interactive Viser viewer. Observations are synthetic points sampled from the
Franka robot mesh with a known pose offset and added noise.

.. code-block:: bash

   python -m curobo.examples.reference.robot_pose_calibration

.. list-table:: Viewer controls
   :header-rows: 1

   * - Control
     - Behavior
   * - Pose frame
     - Sets the initial robot pose estimate.
   * - Global Calibrate
     - Runs ICP with rotation sampling without requiring an initial guess.
   * - Local Calibrate
     - Refines the current pose using mesh SDFs and Levenberg–Marquardt optimization.

.. list-table:: Arguments
   :header-rows: 1

   * - Argument
     - Default
     - Description
   * - ``--device``
     - ``cuda:0``
     - CUDA device.
   * - ``--n-points``
     - ``500``
     - Number of sampled observation points.
   * - ``--noise``
     - ``0.002``
     - Observation noise standard deviation in meters.
   * - ``--port``
     - ``8080``
     - Viser server port.
