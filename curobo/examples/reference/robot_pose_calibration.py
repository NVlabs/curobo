# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Interactive Pose Detector Demo with Viser.

Demonstrates PoseDetector (ICP) and SDFPoseDetector (mesh SDF) for robot pose
estimation using an interactive 3D visualization.

Workflow:
1. Load Franka robot mesh at default joint configuration
2. Sample surface points and apply ground truth offset + noise → visualize as pointcloud
3. Use interactive control frame to set initial pose guess
4. "Global Calibrate" button → ICP with rotation sampling (no initial guess needed)
5. "Local Calibrate" button → SDF-based LM refinement from current frame pose

Usage:
    python robot_pose_calibration.py
    python robot_pose_calibration.py --port 8081
"""

from __future__ import annotations

import argparse
import threading

import numpy as np
import torch

from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.perception import (
    DetectorCfg,
    PoseDetector,
    RobotMesh,
    SDFDetectorCfg,
    SDFPoseDetector,
)
from curobo.types import DeviceCfg, Pose
from curobo.viewer import ViserVisualizer


def load_robot(device: str = "cuda:0") -> tuple[Kinematics, torch.Tensor]:
    """Load Franka robot kinematics and default joint angles."""
    device_cfg = DeviceCfg(device=torch.device(device))
    kin_cfg = KinematicsCfg.from_robot_yaml_file(
        file_path="franka.yml",
        device_cfg=device_cfg,
        load_tool_frames_with_mesh=True,
    )
    kinematics = Kinematics(kin_cfg)
    default_joints = kinematics.default_joint_position
    return kinematics, default_joints


def create_simulated_observation(
    robot_mesh: RobotMesh,
    n_points: int,
    pose_offset: Pose,
    noise_std: float = 0.002,
) -> torch.Tensor:
    """Create simulated observation by sampling mesh and applying pose + noise."""
    points, _ = robot_mesh.sample_surface_points(n_points)
    observed_points = pose_offset.transform_points(points)
    if noise_std > 0:
        noise = torch.randn_like(observed_points) * noise_std
        observed_points = observed_points + noise
    return observed_points


def compute_pose_error(estimated: Pose, ground_truth: Pose) -> tuple[float, float]:
    """Compute translation (mm) and rotation (degrees) errors."""
    trans_error = (estimated.position - ground_truth.position).norm().item() * 1000
    q1 = estimated.quaternion[0]
    q2 = ground_truth.quaternion[0]
    dot = torch.abs(torch.sum(q1 * q2))
    dot = torch.clamp(dot, -1.0, 1.0)
    rot_error = 2 * torch.acos(dot).item() * 180 / 3.14159
    return trans_error, rot_error


def main():
    parser = argparse.ArgumentParser(description="Interactive Pose Detector Demo")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument("--n-points", type=int, default=500, help="Observation points")
    parser.add_argument("--noise", type=float, default=0.002, help="Noise std (meters)")
    parser.add_argument("--port", type=int, default=8080, help="Viser port")
    args = parser.parse_args()

    device = args.device
    print(f"\n{'='*60}")
    print("Interactive Pose Detector Demo (Viser)")
    print(f"{'='*60}\n")

    # --- Load robot ---
    print("Loading Franka robot...")
    kinematics, joint_angles = load_robot(device)
    print(f"  Loaded {len(kinematics.joint_names)} joints")

    # --- Create RobotMesh ---
    print("\nCreating robot mesh...")
    robot_mesh = RobotMesh.from_kinematics(kinematics, device=device)
    robot_mesh.update(joint_angles)
    print(f"  Mesh: {robot_mesh.n_vertices:,} vertices, {robot_mesh.n_faces:,} faces")

    # Get trimesh for visualization
    robot_trimesh = robot_mesh.get_trimesh()
    robot_trimesh.visual.vertex_colors = [255, 140, 0, 200]  # Orange

    # --- Define ground truth pose ---
    gt_pose = Pose.from_list(
        [0.1, -0.05, 0.03, 0.9962, 0.0436, 0.0218, 0.0698],
        device_cfg=DeviceCfg(device=torch.device(device)),
    )
    print("\nGround truth pose offset:")
    print(f"  Position: [{gt_pose.position[0,0]:.3f}, {gt_pose.position[0,1]:.3f}, {gt_pose.position[0,2]:.3f}] m")

    # --- Create simulated observation ---
    print(f"\nCreating observation ({args.n_points} points, {args.noise*1000:.1f}mm noise)...")
    observed_points = create_simulated_observation(
        robot_mesh, args.n_points, gt_pose, noise_std=args.noise
    )
    observed_points_np = observed_points.cpu().numpy()
    print(f"  Observed points: {observed_points.shape}")

    # --- Initialize detectors ---
    print("\nInitializing detectors...")
    icp_config = DetectorCfg(
        n_mesh_points_coarse=500,
        n_observed_points_coarse=2000,
        n_mesh_points_fine=2000,
        n_observed_points_fine=5000,
        n_rotation_samples=32,
        device_cfg=DeviceCfg(device=torch.device(device)),
    )
    icp_detector = PoseDetector(robot_mesh, icp_config)

    sdf_config = SDFDetectorCfg(max_iterations=10,
    inner_iterations=2,
    n_points=args.n_points, use_huber=True)
    sdf_detector = SDFPoseDetector(robot_mesh, sdf_config)
    print("  ✓ ICP detector (global search)")
    print("  ✓ SDF detector (local refinement)")

    # --- Initialize Viser ---
    print(f"\nStarting Viser server on port {args.port}...")
    cpu_cfg = DeviceCfg(device=torch.device("cpu"), dtype=torch.float32)
    viser = ViserVisualizer(
        content_path=None,
        device_cfg=cpu_cfg,
        add_robot_to_scene=False,
        connect_ip="0.0.0.0",
        connect_port=args.port,
        initialize_viser=True,
        add_control_frames=False,
        visualize_robot_spheres=False,
    )
    print(f"  ✓ Viser: http://localhost:{args.port}")

    # --- State ---
    state = {"is_calibrating": False, "js_changed": True}

    # --- Add pointcloud (observed points) ---
    # Color: cyan for observed points
    colors = np.tile([0, 200, 200], (len(observed_points_np), 1)).astype(np.uint8)
    viser.add_point_cloud(
        pointcloud=observed_points_np,
        colors=colors,
        name="/observation/pointcloud",
        point_size=0.003,
    )

    # --- Add ground truth marker (small green sphere) ---
    gt_pos = gt_pose.position.squeeze().cpu().numpy()
    viser._server.scene.add_icosphere(
        "/ground_truth",
        radius=0.01,
        color=(0, 255, 0),
        subdivisions=2,
        position=gt_pos,
    )

    # --- Add interactive control frame for robot pose ---
    initial_pose = Pose(
        position=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
    )
    robot_frame = viser.add_control_frame("/robot/base", initial_pose, scale=0.15)

    # --- Add robot mesh ---
    robot_mesh_handle = viser.add_mesh(robot_trimesh, name="/robot/mesh")
    robot_mesh_handle.position = np.array([0.0, 0.0, 0.0])
    robot_mesh_handle.wxyz = np.array([1.0, 0.0, 0.0, 0.0])

    # --- Calibration callbacks ---
    def update_mesh_pose(pose: Pose):
        """Update robot mesh to match calibrated pose."""
        pos = pose.position.squeeze().cpu().numpy()
        quat = pose.quaternion.squeeze().cpu().numpy()
        robot_frame.position = pos
        robot_frame.wxyz = quat
        robot_mesh_handle.position = pos
        robot_mesh_handle.wxyz = quat

    def run_global_calibration():
        """Run ICP with rotation sampling (global search)."""
        if state["is_calibrating"]:
            return
        state["is_calibrating"] = True

        try:
            print("\n[GLOBAL/ICP] Running pose detection...")
            result = icp_detector.detect_from_points(
                observed_points.to(device),
                config=joint_angles,
                initial_pose=None,  # Triggers global search
            )
            trans_err, rot_err = compute_pose_error(result.pose, gt_pose)
            print(f"  Alignment error: {result.alignment_error * 1000:.2f} mm")
            print(f"  Translation error: {trans_err:.2f} mm")
            print(f"  Rotation error: {rot_err:.2f} deg")
            print(f"  Iterations: {result.n_iterations}")

            update_mesh_pose(result.pose)
            print("  ✓ Mesh updated to calibrated pose")

        except Exception as e:
            print(f"  ✗ Calibration failed: {e}")
        finally:
            state["is_calibrating"] = False

    def run_local_calibration():
        """Run SDF-based LM from current frame pose."""
        if state["is_calibrating"]:
            return
        state["is_calibrating"] = True

        try:
            # Get current frame pose
            frame_pos = np.array(robot_frame.position)
            frame_quat = np.array(robot_frame.wxyz)

            initial_pose = Pose(
                position=torch.tensor([frame_pos], dtype=torch.float32, device=device),
                quaternion=torch.tensor([frame_quat], dtype=torch.float32, device=device),
            )

            print("\n[LOCAL/SDF] Running from frame pose...")
            print(f"  Initial: pos=[{frame_pos[0]:.3f}, {frame_pos[1]:.3f}, {frame_pos[2]:.3f}]")
            print("JS changed: ", state["js_changed"])
            result = sdf_detector.detect_from_points(
                observed_points.to(device),
                config=joint_angles if state["js_changed"] else None,
                initial_pose=initial_pose,
            )



            state["js_changed"] = False
            trans_err, rot_err = compute_pose_error(result.pose, gt_pose)
            print(f"  Alignment error: {result.alignment_error * 1000:.2f} mm")
            print(f"  Translation error: {trans_err:.2f} mm")
            print(f"  Rotation error: {rot_err:.2f} deg")
            print(f"  Iterations: {result.n_iterations}")
            print(f"  Confidence: {result.confidence:.2f}")
            print(f"  Compute time: {result.compute_time:.3f} s")

            update_mesh_pose(result.pose)
            print("  ✓ Mesh updated to calibrated pose")

        except Exception as e:
            print(f"  ✗ Calibration failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            state["is_calibrating"] = False

    # --- Add buttons ---
    global_btn = viser._server.gui.add_button("Global Calibrate (ICP)", color="blue")
    global_btn.on_click(
        lambda _: threading.Thread(target=run_global_calibration, daemon=True).start()
    )

    local_btn = viser._server.gui.add_button("Local Calibrate (SDF)", color="green")
    local_btn.on_click(
        lambda _: threading.Thread(target=run_local_calibration, daemon=True).start()
    )

    # --- Sync mesh with frame on drag ---
    @robot_frame.on_update
    def _(_):
        robot_mesh_handle.position = np.array(robot_frame.position)
        robot_mesh_handle.wxyz = np.array(robot_frame.wxyz)

    # --- Instructions ---
    print(f"\n{'='*60}")
    print("INSTRUCTIONS:")
    print("  1. Drag the control frame to set initial robot pose")
    print("  2. Click 'Global Calibrate (ICP)' for global search")
    print("  3. Click 'Local Calibrate (SDF)' for local refinement")
    print("  4. Green sphere = ground truth pose")
    print("  5. Cyan points = observed pointcloud")
    print(f"{'='*60}\n")

    # --- Keep alive ---
    try:
        import time
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
