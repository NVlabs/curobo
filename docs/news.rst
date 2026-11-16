
Updates
=======
- November 4, 2026 (v0.9.0): cuRoboV2-Fast, Cartesian waypoint planning,
  GPU bidirectional RRT, torque-limited TimeOpt, C++ deployment, and robot
  pose calibration. With a 3 kg payload, cuRoboV2-Fast achieves 4.4 ms mean
  planning time on RTX 5090, 14.7× faster than cuRoboV2. A stereo-camera-based
  YAM deployment achieves 5.93 s mean completed pick-and-drop cycles,
  34.6% shorter than cuRoboV2. See :doc:`technical_reports` for details.
- July 9, 2026: Add visibility-tested textured mapper outputs, independent RGB and
  neural-feature resolutions, and faster TSDF mesh extraction for real-time visualization.
- June 15, 2026: Add LiDAR TSDF, RGB, and feature fusion, compact mapper block
  checkpointing, and live RealSense RGB-D mapping with MPC.
- April 27, 2026: Add feature channel integration in TSDF mapper, improve TSDF mapper performance (1.5ms to 0.5ms).
- April 2026 (v0.8.0): cuRoboV2, complete rewrite of cuRobo with new features and improvements.
- 2024 (v0.7.x): Sub-millimetre IK precision, mimic joints, ESDF voxel collision checker, Grasp API planning, trajectory re-timing.
- Oct 2023 (v0.6.0): Initial release with Technical Report.

See `CHANGELOG <changelog_link_>`_ for details.
