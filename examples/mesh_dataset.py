#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
import math
import os
import sys
from typing import Optional

import numpy as np
import pyrender
import torch
import torch.nn.functional as F
import trimesh
from torch.utils.data.dataset import Dataset


def fov_and_size_to_intrinsics(fov, img_size, device="cpu"):
    img_h, img_w = img_size
    fx = img_w / (2 * math.tan(math.radians(fov) / 2))
    fy = img_h / (2 * math.tan(math.radians(fov) / 2))

    intrinsics = torch.tensor(
        [[fx, 0, img_h / 2], [0, fy, img_w / 2], [0, 0, 1]],
        dtype=torch.float,
        device=device,
    )
    return intrinsics


def lookat_to_cam_pose(eyes, ats, ups=[[0, 0, 1]], device="cpu", mode="opengl"):
    if not isinstance(eyes, torch.Tensor):
        eyes = torch.tensor(eyes, device=device, dtype=torch.float32)
    if not isinstance(ats, torch.Tensor):
        ats = torch.tensor(ats, device=device, dtype=torch.float32)
    if not isinstance(ups, torch.Tensor):
        ups = torch.tensor(ups, device=device, dtype=torch.float32)

    batch_size = eyes.shape[0]

    camera_view = F.normalize(ats - eyes, dim=1)
    camera_right = F.normalize(torch.cross(camera_view, ups, dim=1), dim=1)
    camera_up = F.normalize(torch.cross(camera_right, camera_view, dim=1), dim=1)

    # rotation matrix from opencv conventions
    T = torch.zeros((batch_size, 4, 4))
    if mode == "opengl":
        T[:, :3, :] = torch.stack([camera_right, camera_up, -camera_view, eyes], dim=2)
    elif mode == "opencv":
        T[:, :3, :] = torch.stack([camera_right, -camera_up, camera_view, eyes], dim=2)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    T[:, 3, 3] = 1.0
    return T.float()


def sample_sphere_points(N, radius, device="cuda"):
    latitude = (torch.rand(size=(N, 1), device=device) - 0.5) * torch.pi
    longitude = (torch.rand(size=(N, 1), device=device) - 0.5) * torch.pi * 2
    x = torch.cos(latitude) * torch.cos(longitude)
    y = torch.sin(latitude) * torch.cos(longitude)
    z = torch.sin(longitude)
    pc = torch.cat([x, y, z], dim=1) * radius
    return pc


def sample_sphere_poses(N, origin, radius, device="cuda"):
    eyes = sample_sphere_points(N, radius, device)
    if not isinstance(origin, torch.Tensor):
        origin = torch.tensor(origin).float().to(device)
    ats = origin[None, :].repeat((N, 1))
    poses_gl = lookat_to_cam_pose(eyes, ats, device=device, mode="opengl")
    poses_cv = lookat_to_cam_pose(eyes, ats, device=device, mode="opencv")
    return poses_gl, poses_cv


def compute_origin_and_radius(trimesh_scene):
    low, high = trimesh_scene.bounds
    center = high + low / 2
    low = low - center
    high = high - center
    radius = max(np.sqrt((high**2).sum()), np.sqrt((low**2).sum()))
    return center, radius


def render_batch(trimesh_mesh, camera_poses, fov, image_size):
    camera_poses = camera_poses.detach().cpu().numpy()
    mesh = pyrender.Mesh.from_trimesh(trimesh_mesh)
    scene = pyrender.Scene()
    scene.add(mesh)
    camera = pyrender.PerspectiveCamera(yfov=fov, aspectRatio=1.0)
    camera = pyrender.Node(camera=camera, matrix=np.eye(4))
    scene.add_node(camera)

    light = pyrender.SpotLight(
        color=np.ones(3),
        intensity=1.0,
        innerConeAngle=np.pi / 16.0,
        outerConeAngle=np.pi / 2.0,
    )
    light = pyrender.Node(light=light, matrix=np.eye(4))
    scene.add_node(light)
    r = pyrender.OffscreenRenderer(image_size, image_size)

    colors = []
    depths = []
    for camera_pose in camera_poses:
        scene.set_pose(camera, camera_pose)
        scene.set_pose(light, camera_pose)
        color, depth = r.render(scene)
        colors.append(color)
        depths.append(depth)

    return np.asarray(colors), np.asarray(depths)


"""
MeshDataset takes a path to a mesh as input and uses PyRender to render images of the mesh
from a sphere centered around the scene.
"""


class MeshDataset(Dataset):
    def __init__(
        self,
        mesh_file: str = None,
        n_frames: int = 10,
        image_size: float = 256,
        save_data_dir: Optional[str] = None,
        trimesh_mesh: Optional[trimesh.Trimesh] = None,
        fov_deg: int = 60,
        # visible_point: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.mesh_file = mesh_file
        self.n_frames = n_frames
        if trimesh_mesh is None:
            self.trimesh_mesh = trimesh.load(self.mesh_file)
        else:
            self.trimesh_mesh = trimesh_mesh
        self.image_size = image_size

        origin, radius = compute_origin_and_radius(self.trimesh_mesh)
        self.fov_deg = fov_deg
        sphere_radius = radius * 2.0
        self.camera_poses_gl, self.camera_poses_cv = sample_sphere_poses(
            n_frames, origin, sphere_radius, "cuda"
        )
        self.colors, self.depths = render_batch(
            self.trimesh_mesh,
            self.camera_poses_gl,
            fov=math.radians(self.fov_deg),
            image_size=self.image_size,
        )
        self.intrinsics = fov_and_size_to_intrinsics(
            self.fov_deg, (self.image_size, self.image_size), device="cuda"
        )

        if save_data_dir is not None:
            self.save_as_sun3d_dataset(save_data_dir)
            # sys.exit(0)

    def save_as_sun3d_dataset(self, output_dir):
        import imageio
        from transforms3d.quaternions import quat2mat

        os.makedirs(output_dir, exist_ok=True)
        K = self.intrinsics.detach().cpu().numpy().tolist()
        intrinsics_text = f"""{K[0][0]} {K[0][1]} {K[0][2]}
            {K[1][0]} {K[1][1]} {K[1][2]}
            {K[2][0]} {K[2][1]} {K[2][2]}"""
        with open(f"{output_dir}/camera-intrinsics.txt", "w") as fp:
            fp.write(intrinsics_text)

        seqdir = f"{output_dir}/seq-01"
        os.makedirs(seqdir, exist_ok=True)

        for i in range(len(self)):
            data = self[i]
            rgb = data["rgba"][:3, :, :].detach().cpu().permute(1, 2, 0).numpy()
            depth = data["depth"]
            depth = (depth * 1000).detach().cpu().numpy().astype(np.uint16)
            nvblox_pose = data["pose"]

            eigen_quat = [0.707106769, 0.707106769, 0, 0]
            sun3d_to_nvblox_T = torch.eye(4)
            sun3d_to_nvblox_T[:3, :3] = torch.tensor(quat2mat(eigen_quat))

            sun3d_pose = torch.linalg.inv(sun3d_to_nvblox_T) @ nvblox_pose
            P = sun3d_pose.detach().cpu().numpy().tolist()

            pose_text = f"""{P[0][0]} {P[0][1]} {P[0][2]} {P[0][3]}
                {P[1][0]} {P[1][1]} {P[1][2]} {P[1][3]}
                {P[2][0]} {P[2][1]} {P[2][2]} {P[2][3]}
                {P[3][0]} {P[3][1]} {P[3][2]} {P[3][3]}"""

            framename = f"frame-{str(i).zfill(6)}"
            imageio.imwrite(f"{seqdir}/{framename}.color.png", rgb)
            imageio.imwrite(f"{seqdir}/{framename}.depth.png", depth)
            with open(f"{seqdir}/{framename}.pose.txt", "w") as fp:
                fp.write(pose_text)

    def __len__(self):
        return self.n_frames

    def __getitem__(self, index):
        rgb_np = self.colors[index]
        depth_np = self.depths[index]
        a_np = (depth_np > 0).astype(np.uint8) * 255
        rgba_np = np.concatenate([rgb_np, a_np[:, :, None]], axis=2)
        pose = self.camera_poses_cv[index]
        intrinsics = self.intrinsics

        depth_np = depth_np.astype(np.float32)
        rgba = torch.from_numpy(rgba_np).permute((2, 0, 1))
        depth = torch.from_numpy(depth_np).float()

        return {"rgba": rgba, "depth": depth, "pose": pose, "intrinsics": intrinsics}
