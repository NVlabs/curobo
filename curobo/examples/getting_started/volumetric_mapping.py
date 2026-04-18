# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Fuse RGB-D frames into a TSDF world model and compute an ESDF for collision-aware planning.

cuRobo builds a persistent volumetric world model from depth observations and
known geometry, then generates a dense Euclidean Signed Distance Field (ESDF)
that enables fast, differentiable collision queries for motion generation.
Depth frames are fused into a block-sparse TSDF via lock-free voxel-centric
integration kernels, while analytic primitives (cuboids, meshes) are stamped
directly into a separate geometry channel. On demand, an ESDF is computed at
task-appropriate resolution using the Parallel Banding Algorithm (PBA+),
providing :math:`O(1)` trilinear distance queries for robot collision spheres.

.. raw:: html

   <figure style="margin:0 0 1.5em;">
     <video autoplay loop muted playsinline style="width:100%;border-radius:6px;">
       <source src="../videos/mapper_tsdf.webm" type="video/webm">
     </video>
     <figcaption style="text-align:center;font-style:italic;margin-top:0.4em;">TSDF Integration</figcaption>
   </figure>

.. raw:: html

   <figure style="margin:0 0 1.5em;">
     <video autoplay loop muted playsinline style="width:100%;border-radius:6px;">
       <source src="../videos/mapper_esdf.webm" type="video/webm">
     </video>
     <figcaption style="text-align:center;font-style:italic;margin-top:0.4em;">ESDF Computation</figcaption>
   </figure>

This tutorial walks through building a 3D map from an RGB-D video sequence and
using it for collision-aware robot motion planning. You will learn how cuRobo's
:class:`~curobo.Mapper` API fuses depth frames into a compact world model and
generates a signed distance field that robot planners can query efficiently.

By the end of this tutorial you will have:

- Fused a sequence of depth images into a block-sparse TSDF
- Stamped a known obstacle (cuboid) into the map as analytic geometry
- Computed a dense ESDF over the workspace
- Rendered a depth image and surface normals from any camera pose
- Extracted a colored triangle mesh of the reconstruction

Step 1: Download the dataset
-------------------------------

This tutorial uses the `Sun3D <http://sun3d.cs.princeton.edu/>`_ indoor RGB-D
dataset, which provides color images, depth maps, and ground-truth camera poses.

Quick start (downloads a single scene, ~1400 MB):

.. code-block:: bash

   wget http://3dvision.princeton.edu/projects/2016/3DMatch/downloads/rgbd-datasets/sun3d-mit_76_studyroom-76-1studyroom2.zip
   mkdir -p datasets/sun3d
   unzip sun3d-mit_76_studyroom-76-1studyroom2.zip -d datasets/sun3d

The extracted directory should look like::

    datasets/sun3d/sun3d-mit_76_studyroom-76-1studyroom2/
        camera-intrinsics.txt
        <sequence_name>/
            000001.color.png
            000001.depth.png
            000001.pose.txt
            ...

Step 2: Run the tutorial
---------------------------

.. code-block:: bash

   python -m curobo.examples.getting_started.volumetric_mapping --root ./datasets/sun3d/sun3d-mit_76_studyroom-76-1studyroom2

To explore the reconstruction interactively, add ``--visualize``. This starts a
`Viser <https://viser.studio>`_ server you can open in your browser at
http://localhost:8080. Drag the gizmo to inspect ESDF slices through the scene.

.. code-block:: bash

   python -m curobo.examples.getting_started.volumetric_mapping --root ./datasets/sun3d/sun3d-mit_76_studyroom-76-1studyroom2 --visualize

Step 3: Check the output
---------------------------

When the tutorial finishes successfully you will see::

    Loading Sun3D dataset from ./datasets/sun3d...
    Found 200 frames
    Mapper initialized: 42.0 MB

    Integrating 200 frames...
    Rendering from first camera pose...
    Saved renders to: ~/.cache/curobo/examples/volumetric_mapping

    Computing ESDF...
    Extracting mesh...
    Saved mesh: output_mesh.ply (150,000 vertices)

The following files are written to ``~/.cache/curobo/examples/volumetric_mapping/``
(override with ``curobo._src.runtime.cache_dir``):

- ``rendered_depth.png``: depth colormap rendered from the first camera pose
- ``rendered_normals.png``: surface normal colormap
- ``rendered_shaded.png``: Phong-shaded surface view
- ``output_mesh.ply``: colored triangle mesh of the full reconstruction
"""

import argparse
import os
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from curobo import runtime
from curobo.content import get_assets_path
from curobo.perception import FilterDepth, Mapper, MapperCfg
from curobo.profiling import CudaEventTimer
from curobo.scene import Cuboid, Mesh, SceneData
from curobo.types import CameraObservation, DeviceCfg, Pose
from curobo.viewer import ViserVisualizer


def create_scene_with_static_obstacles(device: str = "cuda:0") -> SceneData:
    """Create a scene with a cuboid and a mesh to stamp into the TSDF geometry channel.

    These primitives are integrated analytically (not from depth) and do not
    decay, demonstrating how known geometry (e.g., a table, a robot link) can be
    fused alongside depth observations.

    Args:
        device: CUDA device string.

    Returns:
        SceneData containing the static obstacles.
    """
    device_cfg = DeviceCfg(device=torch.device(device))

    scene = SceneData.create_cache(
        num_envs=1,
        device_cfg=device_cfg,
        cuboid_cache=2,
        mesh_cache=2,
    )

    cuboid = Cuboid(
        name="table_top",
        pose=[0.5, 0.0, -0.5, 1.0, 0.0, 0.0, 0.0],
        dims=[0.2, 0.2, 0.2],
    )
    scene.add_obstacle(cuboid, env_idx=0)

    mesh_path = os.path.join(
        get_assets_path(),
        "robot/franka_description/meshes/visual/link1.obj",
    )
    if os.path.exists(mesh_path):
        import trimesh

        mesh_data = trimesh.load(mesh_path, force="mesh")
        robot_link = Mesh(
            name="robot_link",
            vertices=mesh_data.vertices.tolist(),
            faces=mesh_data.faces.flatten().tolist(),
            pose=[0.5, 0.5, -0.5, 1.0, 0.0, 0.0, 0.0],
        )
        scene.add_obstacle(robot_link, env_idx=0)

    return scene


def extract_esdf_slice(
    esdf_grid: torch.Tensor,
    origin: torch.Tensor,
    voxel_size: float,
    slice_pose: np.ndarray,
    slice_size: float = 1.0,
    slice_resolution: int = 128,
) -> np.ndarray:
    """Extract a 2D slice of the ESDF at the given pose.

    Args:
        esdf_grid: (nx, ny, nz) ESDF tensor (VoxelGrid convention: X slowest, Z fastest).
        origin: Grid center (3,).
        voxel_size: Size of each voxel.
        slice_pose: 4x4 homogeneous transform defining slice plane.
        slice_size: Physical size of slice in meters.
        slice_resolution: Number of pixels per side.

    Returns:
        RGB image with blue=inside, white=zero, red=outside.
    """
    device = esdf_grid.device
    nx, ny, nz = esdf_grid.shape

    half = slice_size / 2
    u = torch.linspace(-half, half, slice_resolution, device=device)
    v = torch.linspace(-half, half, slice_resolution, device=device)
    uu, vv = torch.meshgrid(u, v, indexing="xy")

    local_points = torch.stack(
        [
            uu.flatten(),
            vv.flatten(),
            torch.zeros(slice_resolution * slice_resolution, device=device),
            torch.ones(slice_resolution * slice_resolution, device=device),
        ],
        dim=1,
    )

    pose_tensor = torch.tensor(slice_pose, dtype=torch.float32, device=device)
    world_points = (pose_tensor @ local_points.T).T[:, :3]

    # align_corners=True maps [-1,+1] to first/last voxel centers
    local_pts = world_points - origin.to(device)
    half_extent = torch.tensor(
        [(nx - 1) * voxel_size / 2, (ny - 1) * voxel_size / 2, (nz - 1) * voxel_size / 2],
        device=device,
    )
    normalized = local_pts / half_extent

    # grid_sample for 5D expects coords as (z, y, x) matching (N,C,D,H,W) layout
    coords = normalized[:, [2, 1, 0]]
    coords = coords.view(1, 1, slice_resolution, slice_resolution, 3).float()

    esdf_5d = esdf_grid.float().unsqueeze(0).unsqueeze(0)
    sampled = F.grid_sample(
        esdf_5d, coords, mode="bilinear", padding_mode="border", align_corners=True
    )
    values = sampled.squeeze().cpu().numpy()

    max_dist = max(np.max(values), 0.1)
    max_negative_dist = max(np.abs(np.min(values)), 0.05)
    normalized = np.clip(values / max_dist, -1, 1)
    normalized_negative = np.clip(values / max_negative_dist, -1, 1)

    colors = np.zeros((slice_resolution, slice_resolution, 3), dtype=np.uint8)
    neg_mask = normalized_negative < 0
    colors[neg_mask, 0] = ((1 + normalized_negative[neg_mask]) * 255).astype(np.uint8)
    colors[neg_mask, 1] = ((1 + normalized_negative[neg_mask]) * 255).astype(np.uint8)
    colors[neg_mask, 2] = 255
    pos_mask = normalized >= 0
    colors[pos_mask, 0] = 255
    colors[pos_mask, 1] = ((1 - normalized[pos_mask]) * 255).astype(np.uint8)
    colors[pos_mask, 2] = ((1 - normalized[pos_mask]) * 255).astype(np.uint8)

    colors[np.abs(values) < voxel_size * 0.5] = [0, 255, 0]

    return colors


class Sun3dDataset(Dataset):
    """Sun3D RGB-D dataset loader."""

    def __init__(self, root: str, device: str = "cuda", depth_scale: float = 0.001):
        self.root = root
        self.device = device
        self.device_cfg = DeviceCfg(device=torch.device(device))
        self.depth_scale = depth_scale

        self.intrinsics = (
            torch.from_numpy(np.loadtxt(os.path.join(root, "camera-intrinsics.txt")))
            .float()
            .to(device)
        )

        seq_name = sorted(
            d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))
        )[0]
        self.seq_dir = os.path.join(root, seq_name)
        self.frames = sorted({f.split(".")[0] for f in os.listdir(self.seq_dir)})

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx) -> CameraObservation:
        name = self.frames[idx]

        rgb = torch.tensor(
            iio.imread(os.path.join(self.seq_dir, f"{name}.color.png")),
            device=self.device,
        )
        depth = torch.tensor(
            iio.imread(os.path.join(self.seq_dir, f"{name}.depth.png")).astype(np.float32)
            * self.depth_scale,
            device=self.device,
        )
        pose_np = np.loadtxt(os.path.join(self.seq_dir, f"{name}.pose.txt"))

        pose_matrix = torch.tensor(pose_np, device=self.device, dtype=torch.float32)
        y_to_z = Pose.from_list([0, 0, 0, 0.707, 0.707, 0, 0], device_cfg=self.device_cfg)
        transform = torch.eye(4, device=self.device, dtype=torch.float32)
        transform[:3, :3] = y_to_z.get_rotation_matrix()
        world_pose = Pose.from_matrix(transform @ pose_matrix)
        world_pose.position[0, 2] -= 1.25

        return CameraObservation(
            name=f"frame_{idx}",
            rgb_image=rgb,
            depth_image=depth,
            pose=world_pose,
            intrinsics=self.intrinsics,
        )


def main():
    parser = argparse.ArgumentParser(description="Volumetric Mapping")
    parser.add_argument("--root", type=str, required=True, help="Sun3D dataset root path")
    parser.add_argument("--voxel-size", type=float, default=0.02, help="Voxel size (m)")
    parser.add_argument("--max-frames", type=int, default=10000, help="Max frames to process")
    parser.add_argument("--output", type=str, default="output_mesh.ply", help="Output mesh")
    parser.add_argument("--visualize", action="store_true", help="Enable viser visualization")
    parser.add_argument("--num-cameras", type=int, default=1, help="Number of cameras")
    parser.add_argument("--refine-pose", action="store_true", help="Refine pose using ICP")
    parser.add_argument(
        "--estimate-pose",
        action="store_true",
        help="Estimate pose from first frame (identity init)",
    )
    args = parser.parse_args()

    print(f"Loading Sun3D dataset from {args.root}...")
    dataset = Sun3dDataset(args.root)
    print(f"Found {len(dataset)} frames")

    # truncation_distance controls how many voxels around each surface are
    # updated per frame. 6x the voxel size is a common default; wider bands
    # smooth noise but cost more memory and integration time.
    config = MapperCfg(
        voxel_size=args.voxel_size,
        extent_meters_xyz=(11.0, 7.0, 5.0),
        truncation_distance=args.voxel_size * 6,
        depth_maximum_distance=10.0,
        depth_minimum_distance=0.05,
        minimum_tsdf_weight=0.01,
        rgb_scale=1,
        decay_factor=1.0,
        frustum_decay_factor=1.0,
        # enable_static=True allocates a separate geometry channel for analytic
        # primitives (cuboids, meshes) that never decay with time.
        enable_static=True,
        static_obstacle_color=(0, 100, 100),
        roughness=5.0,
        num_cameras=args.num_cameras,
        integration_method="voxel_project",
    )
    mapper = Mapper(config)
    print(f"Mapper initialized: {mapper.memory_usage_mb():.1f} MB")

    visualizer = None
    if args.visualize:
        visualizer = ViserVisualizer(connect_port=8080)
        print("Visualization: http://localhost:8080")

        min_corner, max_corner = config.get_grid_bounds()
        min_c = np.array(min_corner, dtype=np.float32)
        max_c = np.array(max_corner, dtype=np.float32)

        corners = np.array(
            [
                [min_c[0], min_c[1], min_c[2]],
                [max_c[0], min_c[1], min_c[2]],
                [max_c[0], max_c[1], min_c[2]],
                [min_c[0], max_c[1], min_c[2]],
                [min_c[0], min_c[1], max_c[2]],
                [max_c[0], min_c[1], max_c[2]],
                [max_c[0], max_c[1], max_c[2]],
                [min_c[0], max_c[1], max_c[2]],
            ],
            dtype=np.float32,
        )

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        lines = np.array([[corners[i], corners[j]] for i, j in edges], dtype=np.float32)
        yellow = np.array([255, 255, 0], dtype=np.uint8)
        visualizer._server.scene.add_line_segments(
            "/extent_box",
            points=lines,
            colors=yellow,
            line_width=3.0,
        )

    max_frames = min(args.max_frames, len(dataset))
    print(f"\nIntegrating {max_frames} frames...")

    depth_filter = FilterDepth(
        image_shape=(480, 640),
        depth_minimum_distance=mapper.config.depth_minimum_distance,
        depth_maximum_distance=mapper.config.depth_maximum_distance,
        flying_pixel_threshold=0.5,
        bilateral_kernel_size=3,
    )
    last_pose = None

    n_cams = args.num_cameras
    pbar = tqdm(range(0, max_frames, n_cams))
    for batch_start in pbar:
        batch_end = min(batch_start + n_cams, max_frames)
        batch_obs = []
        t_load = 0.0
        t_filter = 0.0
        for i in range(batch_start, batch_end):
            t0 = time.perf_counter()
            obs = dataset[i]
            t_load += time.perf_counter() - t0

            t0 = time.perf_counter()
            obs.depth_image = torch.nan_to_num(obs.depth_image, nan=0.0)
            filtered, _ = depth_filter(obs.depth_image.unsqueeze(0))
            obs.depth_image = filtered[0]
            t_filter += time.perf_counter() - t0

            if args.refine_pose or args.estimate_pose:
                if i < 5:
                    last_pose = obs.pose
                else:
                    init_pose = last_pose if args.estimate_pose else obs.pose
                    new_pose, error, iters = mapper.refine_pose(
                        obs.depth_image, obs.intrinsics, init_pose
                    )
                    print("Pose refinement error(mm):", error * 1000.0)
                    if error * 1000.0 < 2 * args.voxel_size * 1000.0:
                        obs.pose = new_pose
                        last_pose = new_pose
                    else:
                        print("Pose refinement failed. Ignoring frame.")
                        last_pose = new_pose
                        continue

            batch_obs.append(obs)

        if not batch_obs:
            continue

        # Pad incomplete final batch with zero-depth frames (rejected by the
        # kernel's depth_min check, so they contribute nothing to the TSDF).
        if len(batch_obs) < n_cams:
            ref = batch_obs[0]
            empty = CameraObservation(
                depth_image=torch.zeros_like(ref.depth_image),
                rgb_image=torch.zeros_like(ref.rgb_image),
                pose=ref.pose,
                intrinsics=ref.intrinsics,
            )
            while len(batch_obs) < n_cams:
                batch_obs.append(empty)

        # Stack into a single batched CameraObservation
        batched = CameraObservation(
            depth_image=torch.stack([o.depth_image for o in batch_obs]),
            rgb_image=torch.stack([o.rgb_image for o in batch_obs]),
            pose=Pose(
                position=torch.cat([o.pose.position.view(1, 3) for o in batch_obs]),
                quaternion=torch.cat([o.pose.quaternion.view(1, 4) for o in batch_obs]),
            ),
            intrinsics=torch.stack([o.intrinsics for o in batch_obs]),
        )
        timer = CudaEventTimer().start()
        mapper.integrate(batched)
        dt_s = timer.stop()
        pbar.set_postfix(
            load_ms=f"{t_load * 1000:.0f}",
            filter_ms=f"{t_filter * 1000:.0f}",
            integrate_ms=f"{dt_s * 1000:.1f}",
            n_cams=n_cams,
        )

        if visualizer and (batch_end) % 20 < n_cams:
            centers, colors = mapper.integrator.extract_occupied_voxels(surface_only=False)
            if centers is not None:
                if len(centers) > 100_000:
                    scale = int(len(centers) / 100_000)
                    if scale > 1:
                        centers = centers[::scale]
                        colors = colors[::scale]

                visualizer.add_point_cloud(
                    pointcloud=centers.cpu().numpy(),
                    colors=colors.cpu().numpy(),
                    point_size=args.voxel_size,
                    name="/reconstruction",
                )

            for cam_i, vis_obs in enumerate(batch_obs):
                visualizer.add_frame(
                    f"/cameras/frame_{cam_i}", vis_obs.pose, scale=0.1,
                )

                img_shape = vis_obs.depth_image.shape[:2]
                fx = vis_obs.intrinsics[0, 0].item()
                render_width = 0.3 * img_shape[1] / fx
                render_height = render_width * img_shape[0] / img_shape[1]

                visualizer.add_image(
                    image=vis_obs.rgb_image.cpu().numpy(),
                    render_width=render_width,
                    render_height=render_height,
                    pose=vis_obs.pose,
                    name=f"/cameras/image_{cam_i}",
                )

    # Stamp known geometry into the static TSDF channel. Unlike depth frames,
    # these primitives are integrated analytically and do not decay over time,
    # so they remain in the map even when not visible to the camera.
    scene = create_scene_with_static_obstacles(device="cuda:0")
    mapper.update_static_obstacles(scene)
    print("Static obstacles stamped into TSDF geometry channel")

    if visualizer:
        centers, colors = mapper.integrator.extract_occupied_voxels(surface_only=False)
        if centers is not None:
            print(f"Extracted {len(centers)} voxels (with static obstacles)")
            if len(centers) > 100_000:
                scale = int(len(centers) / 100_000)
                if scale > 1:
                    centers = centers[::scale]
                    colors = colors[::scale]
            visualizer.add_point_cloud(
                pointcloud=centers.cpu().numpy(),
                colors=colors.cpu().numpy(),
                point_size=args.voxel_size,
                name="/reconstruction",
            )

    # Render from first camera pose
    print("\nRendering from first camera pose. First call will take minutes to compile kernel.")
    first_obs = dataset[0]
    img_shape = first_obs.depth_image.shape[:2]

    depth, normals, mask = mapper.render(
        intrinsics=first_obs.intrinsics,
        pose=first_obs.pose,
        image_shape=img_shape,
    )
    print(f"Rendered: {mask.sum().item():,} valid pixels")

    depth_colormap = mapper.render_depth_colormap(
        first_obs.intrinsics, first_obs.pose, img_shape
    )
    normal_colormap = mapper.render_normal_colormap(
        first_obs.intrinsics, first_obs.pose, img_shape
    )
    shaded = mapper.render_shaded(first_obs.intrinsics, first_obs.pose, img_shape)

    if visualizer:
        server = visualizer._server
        with server.gui.add_folder("Rendered Views"):
            server.gui.add_image(depth_colormap.cpu().numpy(), label="Depth")
            server.gui.add_image(normal_colormap.cpu().numpy(), label="Normals")
            server.gui.add_image(shaded.cpu().numpy(), label="Shaded")
    else:
        out = Path(runtime.cache_dir) / "examples" / "volumetric_mapping"
        out.mkdir(parents=True, exist_ok=True)
        iio.imwrite(str(out / "rendered_depth.png"), depth_colormap.cpu().numpy())
        iio.imwrite(str(out / "rendered_normals.png"), normal_colormap.cpu().numpy())
        iio.imwrite(str(out / "rendered_shaded.png"), shaded.cpu().numpy())
        print(f"Saved renders to: {out}")

    # Compute the ESDF over the workspace. The ESDF is generated at a coarser
    # resolution than the TSDF, fine enough for robot collision spheres but
    # cheap enough to recompute after each depth update. At query time, cuRobo
    # uses trilinear interpolation over this grid for O(1) distance lookups.
    print("\nComputing ESDF...")
    voxel_grid = mapper.compute_esdf()
    if voxel_grid.feature_tensor is not None:
        print(
            f"  ESDF shape: {voxel_grid.feature_tensor.shape}, "
            f"voxel_size: {voxel_grid.voxel_size:.4f}m"
        )

    # Extract and save mesh
    print("\nExtracting mesh...")
    mesh = mapper.extract_mesh(surface_only=False)
    if mesh.vertices is not None and len(mesh.vertices) > 0:
        print(f"Saving mesh to {args.output}")
        mesh.save_as_mesh(args.output)
        print(f"Saved mesh: {args.output} ({len(mesh.vertices):,} vertices)")

    else:
        print("No mesh extracted (empty reconstruction)")

    # Keep viser running with interactive ESDF slice
    if visualizer:
        server = visualizer._server
        slice_size = min(config.extent_meters_xyz)

        slice_gizmo = server.scene.add_transform_controls(
            "/esdf_slice_gizmo",
            scale=0.2,
            position=(0.0, 0.0, 0.0),
        )

        def update_esdf_slice():
            """Update ESDF slice visualization at gizmo position and orientation."""
            import viser.transforms as vtf

            if voxel_grid.feature_tensor is None:
                return

            position = np.array(slice_gizmo.position)
            orientation = slice_gizmo.wxyz
            slice_pose = np.eye(4, dtype=np.float32)
            slice_pose[:3, 3] = position
            slice_pose[:3, :3] = vtf.SO3(orientation).as_matrix()

            grid_center = torch.tensor(
                voxel_grid.pose[:3],
                dtype=torch.float32,
                device=voxel_grid.feature_tensor.device,
            )
            colors = extract_esdf_slice(
                esdf_grid=voxel_grid.feature_tensor,
                origin=grid_center,
                voxel_size=voxel_grid.voxel_size,
                slice_pose=slice_pose,
                slice_size=slice_size,
                slice_resolution=256,
            )
            server.scene.add_image(
                name="/esdf_slice_gizmo/slice_image",
                image=colors,
                render_width=slice_size,
                render_height=slice_size,
            )

        @slice_gizmo.on_update
        def _on_slice_update(_):
            update_esdf_slice()

        update_esdf_slice()

        print(
            "\nVisualization running. Drag the gizmo to move ESDF slice. "
            "Press Ctrl+C to exit."
        )
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
