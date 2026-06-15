#!/usr/bin/env python3
"""Fuse Oxford Spires LiDAR clouds into a cuRobo block-sparse TSDF.

The default path builds a LiDAR-only RGB TSDF. Pass ``--enable-features`` to
also sample C-RADIO features from timestamp-matched RGB cameras, fuse them into
``LidarObservation.feature_grid``, and expose a Viser text-query overlay.

.. raw:: html

   <figure style="margin:0 0 1.5em;">
     <video autoplay loop muted playsinline controls style="width:100%;border-radius:6px;">
       <source src="../videos/tsdf_lidar_mapping.webm" type="video/webm">
     </video>
     <figcaption style="text-align:center;font-style:italic;margin-top:0.4em;">LiDAR volumetric mapping</figcaption>
   </figure>

Dataset
-------

This example uses one sequence from the Oxford Spires dataset:

    https://huggingface.co/datasets/ori-drs/oxford_spires_dataset

Download only that sequence into ``data/`` from this cuRobo checkout:

.. code-block:: bash

   python -m pip install -U "huggingface_hub[cli]"
   # Optional: run this if Hugging Face asks for authentication.
   hf auth login
   hf download ori-drs/oxford_spires_dataset --repo-type dataset --local-dir data \
     --include 'sequences/2024-03-18-christ-church-03/processed/colmap/images.zip' \
     --include 'sequences/2024-03-18-christ-church-03/processed/colmap/transforms_colmap*.json' \
     --include 'sequences/2024-03-18-christ-church-03/processed/vilens-slam/undist-clouds.zip' \
     --include 'sequences/2024-03-18-christ-church-03/processed/vilens-slam/slam-poses.csv' \
     --include 'sequences/2024-03-18-christ-church-03/processed/trajectory/*' \
     --include 'calibration/*' \
     --include 'calibration/calibration-sequences-2024-03-19/*'
   python -m zipfile -e data/sequences/2024-03-18-christ-church-03/processed/colmap/images.zip \
     data/sequences/2024-03-18-christ-church-03/processed/colmap
   python -m zipfile -e data/sequences/2024-03-18-christ-church-03/processed/vilens-slam/undist-clouds.zip \
     data/sequences/2024-03-18-christ-church-03/processed/vilens-slam

The ``hf download`` command uses include patterns to fetch only
``sequences/2024-03-18-christ-church-03`` plus required calibration files,
not the full Hugging Face dataset. The default mapping command then finds the
sequence at ``data/sequences/2024-03-18-christ-church-03``.

Run LiDAR volumetric mapping:

.. code-block:: bash

   python curobo/examples/reference/lidar_volumetric_mapping.py

Run feature mapping and text queries:

.. code-block:: bash

   python curobo/examples/reference/lidar_volumetric_mapping.py --enable-features --max-frames 10

.. raw:: html

   <figure style="margin:0 0 1.5em;">
     <video autoplay loop muted playsinline controls style="width:100%;border-radius:6px;">
       <source src="../videos/tsdf_lidar_features.webm" type="video/webm">
     </video>
     <figcaption style="text-align:center;font-style:italic;margin-top:0.4em;">C-RADIO feature mapping and text query</figcaption>
   </figure>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


OXFORD_SPIRES_SEQUENCE = "2024-03-18-christ-church-03"
DEFAULT_SEQUENCE_DIR = Path(f"data/sequences/{OXFORD_SPIRES_SEQUENCE}")
DEFAULT_TSDF_OUTPUT = Path("processed/mapper/lidar_tsdf_blocks.pt")
DEFAULT_FEATURE_OUTPUT = Path("processed/mapper/lidar_feature_tsdf_blocks.pt")
OXFORD_SPIRES_REPO_ID = "ori-drs/oxford_spires_dataset"
RADIO_MODEL_NAME = "c-radio_v3-B"
TEXT_ADAPTOR_NAME = "siglip2"
TEXT_TOP_K = 500
TEXT_MIN_SCORE = 0.05


def default_curobo_root() -> Path:
    curobo_root = os.environ.get("CUROBO_ROOT")
    if curobo_root:
        return Path(curobo_root).expanduser()
    repo_root = Path(__file__).resolve().parents[3]
    if (repo_root / "pyproject.toml").exists():
        return repo_root
    return Path.cwd()


DEFAULT_CUROBO_ROOT = default_curobo_root()
CLOUD_RE = re.compile(r"cloud_(\d+)_(\d+)\.pcd$")
BASE_T_LIDAR_T_XYZ_Q_XYZW = np.array(
    [0.0, 0.0, 0.124, 0.0, 0.0, 1.0, 0.0],
    dtype=np.float32,
)
CAMERA_ORDER = (2, 0, 1)
IMAGE_MATCH_TOLERANCE_NS = 50_000_000
VISER_PORT_RANGE = range(8080, 8090)
VISER_MAX_TSDF_POINTS = 300_000
VISER_TSDF_SURFACE_THRESHOLD_VOX = 1.5
VISER_TSDF_POINTCLOUD_NAME = "/tsdf_surface_voxels_camera_rgb"
VISER_FEATURE_POINTCLOUD_NAME = "/tsdf_surface_voxels_feature_pca"
VISER_TEXT_MATCH_POINTCLOUD_NAME = "/tsdf_surface_voxels_text_query"
VISER_RGB_MAX_IMAGE_WIDTH = 480
VISER_RGB_IMAGE_DISTANCE_M = 0.75
VISER_RGB_RENDER_WIDTH_M = 1.0
VISER_RGB_CAMERA_AXES_LENGTH = 0.45
VISER_MAX_FEATURE_POINTS = 300_000
VISER_MAX_TEXT_MATCH_POINTS = 500_000
VISER_ESDF_SLICE_NAME = "/esdf/xy_slice"
DEFAULT_ESDF_EXTENT_M = 5.0
DEFAULT_ESDF_VOXEL_SIZE_M = 0.05
DEFAULT_ESDF_SLICE_RESOLUTION = 256
ESDF_ORIGIN_XYZ = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class TimedPose:
    sec: int
    nsec: int
    position: np.ndarray
    quaternion_wxyz: np.ndarray


@dataclass(frozen=True)
class Frame:
    cloud_path: Path
    pose: TimedPose


@dataclass(frozen=True)
class Bounds:
    min_xyz: np.ndarray
    max_xyz: np.ndarray
    elevation_min_rad: float
    elevation_max_rad: float

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.min_xyz + self.max_xyz)

    @property
    def extent_xyz(self) -> np.ndarray:
        return self.max_xyz - self.min_xyz


@dataclass(frozen=True)
class CameraCalibration:
    name: str
    image_dir: Path
    image_width: int
    image_height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: np.ndarray
    T_cam_base: np.ndarray
    image_timestamps_ns: list[int]
    image_paths: list[Path]


@dataclass(frozen=True)
class RangeImageResult:
    range_image: np.ndarray
    rgb_image: np.ndarray
    projected_pixels: int
    camera_rgb_pixels: int
    sample_points_base: np.ndarray
    sample_colors: np.ndarray
    feature_grid: np.ndarray | None = None
    feature_pixels: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LiDAR volumetric mapping with optional RGB feature fusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""\
Install the Hugging Face CLI if needed:
  python -m pip install -U "huggingface_hub[cli]"

Optional if Hugging Face asks for authentication:
  hf auth login

Download only the Oxford Spires sequence used by this example first:
  hf download {OXFORD_SPIRES_REPO_ID} --repo-type dataset --local-dir data \\
    --include 'sequences/{OXFORD_SPIRES_SEQUENCE}/processed/colmap/images.zip' \\
    --include 'sequences/{OXFORD_SPIRES_SEQUENCE}/processed/colmap/transforms_colmap*.json' \\
    --include 'sequences/{OXFORD_SPIRES_SEQUENCE}/processed/vilens-slam/undist-clouds.zip' \\
    --include 'sequences/{OXFORD_SPIRES_SEQUENCE}/processed/vilens-slam/slam-poses.csv' \\
    --include 'sequences/{OXFORD_SPIRES_SEQUENCE}/processed/trajectory/*' \\
    --include 'calibration/*' \\
    --include 'calibration/calibration-sequences-2024-03-19/*'
  python -m zipfile -e data/sequences/{OXFORD_SPIRES_SEQUENCE}/processed/colmap/images.zip \\
    data/sequences/{OXFORD_SPIRES_SEQUENCE}/processed/colmap
  python -m zipfile -e data/sequences/{OXFORD_SPIRES_SEQUENCE}/processed/vilens-slam/undist-clouds.zip \\
    data/sequences/{OXFORD_SPIRES_SEQUENCE}/processed/vilens-slam

Dataset source:
  https://huggingface.co/datasets/{OXFORD_SPIRES_REPO_ID}

This fetches only:
  data/sequences/{OXFORD_SPIRES_SEQUENCE}
  data/calibration

Run LiDAR volumetric mapping:
  python curobo/examples/reference/lidar_volumetric_mapping.py

Run feature mapping and text queries:
  python curobo/examples/reference/lidar_volumetric_mapping.py --enable-features --max-frames 10
""",
    )
    parser.add_argument("--sequence-dir", type=Path, default=DEFAULT_SEQUENCE_DIR)
    parser.add_argument("--curobo-root", type=Path, default=DEFAULT_CUROBO_ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=64, help="Use 0 for all frames.")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--lidar-height", type=int, default=64)
    parser.add_argument("--lidar-width", type=int, default=1024)
    parser.add_argument("--range-min", type=float, default=0.1)
    parser.add_argument("--range-max", type=float, default=20.0)
    parser.add_argument("--elevation-min-deg", type=float, default=None)
    parser.add_argument("--elevation-max-deg", type=float, default=None)
    parser.add_argument("--voxel-size", type=float, default=0.05)
    parser.add_argument("--truncation-distance", type=float, default=None)
    parser.add_argument(
        "--esdf-every",
        type=int,
        default=1,
        help="Compute and visualize an ESDF slice every N integrated frames. Use 0 to disable.",
    )
    parser.add_argument(
        "--esdf-extent-meters",
        type=float,
        default=DEFAULT_ESDF_EXTENT_M,
        help="Side length of the cubic ESDF volume in meters.",
    )
    parser.add_argument(
        "--esdf-voxel-size",
        type=float,
        default=DEFAULT_ESDF_VOXEL_SIZE_M,
        help="ESDF voxel size in meters.",
    )
    parser.add_argument(
        "--esdf-slice-resolution",
        type=int,
        default=DEFAULT_ESDF_SLICE_RESOLUTION,
        help="Pixel width and height of the Viser ESDF slice image.",
    )
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument(
        "--max-blocks",
        type=int,
        default=None,
        help="Rejected for this cuRobo checkout; MapperCfg derives max_blocks from extent.",
    )
    parser.add_argument("--roughness", type=float, default=2.0)
    parser.add_argument("--padding-m", type=float, default=1.0)
    parser.add_argument("--grid-center", type=float, nargs=3, default=None)
    parser.add_argument("--extent-meters-xyz", type=float, nargs=3, default=None)
    parser.add_argument(
        "--color-mode",
        choices=("intensity", "height", "constant"),
        default="intensity",
    )
    parser.add_argument(
        "--enable-features",
        action="store_true",
        help="Fuse C-RADIO patch features and enable Viser text-query overlays.",
    )
    parser.add_argument("--feature-model", default=RADIO_MODEL_NAME)
    parser.add_argument("--status-every", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.esdf_every < 0:
        parser.error("--esdf-every must be non-negative")
    if args.esdf_extent_meters <= 0:
        parser.error("--esdf-extent-meters must be positive")
    if args.esdf_voxel_size <= 0:
        parser.error("--esdf-voxel-size must be positive")
    if args.esdf_slice_resolution <= 0:
        parser.error("--esdf-slice-resolution must be positive")
    if args.output is None:
        default_output = DEFAULT_FEATURE_OUTPUT if args.enable_features else DEFAULT_TSDF_OUTPUT
        args.output = args.sequence_dir / default_output
    return args


def load_poses(csv_path: Path) -> dict[tuple[int, int], TimedPose]:
    poses: dict[tuple[int, int], TimedPose] = {}
    with csv_path.open(newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].lstrip().startswith("#"):
                continue
            sec = int(row[1])
            nsec = int(row[2])
            position = np.array([float(row[3]), float(row[4]), float(row[5])], dtype=np.float32)
            qx, qy, qz, qw = (float(row[6]), float(row[7]), float(row[8]), float(row[9]))
            quaternion_wxyz = normalize_quat(np.array([qw, qx, qy, qz], dtype=np.float32))
            poses[(sec, nsec)] = TimedPose(sec, nsec, position, quaternion_wxyz)
    return poses


def select_frames(args: argparse.Namespace) -> list[Frame]:
    cloud_dir = args.sequence_dir / "processed/vilens-slam/undist-clouds"
    pose_csv = args.sequence_dir / "processed/vilens-slam/slam-poses.csv"
    poses = load_poses(pose_csv)
    frames: list[Frame] = []
    for path in sorted(cloud_dir.glob("cloud_*.pcd")):
        match = CLOUD_RE.match(path.name)
        if match is None:
            continue
        key = (int(match.group(1)), int(match.group(2)))
        pose = poses.get(key)
        if pose is not None:
            frames.append(Frame(path, pose))

    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")

    frames = frames[args.start_index :: args.frame_stride]
    if args.max_frames > 0:
        frames = frames[: args.max_frames]
    if not frames:
        raise RuntimeError("No LiDAR frames matched the selected slice.")
    return frames


def sequence_data_root(sequence_dir: Path) -> Path:
    return sequence_dir.parents[1]


def timestamp_ns(sec: int, nsec: int) -> int:
    return int(sec) * 1_000_000_000 + int(nsec)


def timestamp_ns_from_image_path(path: Path) -> int:
    sec, nsec = path.stem.split(".", 1)
    return timestamp_ns(int(sec), int(nsec))


def load_camera_calibrations(sequence_dir: Path) -> list[CameraCalibration]:
    import yaml

    calibration_dir = sequence_data_root(sequence_dir) / "calibration"
    extrinsics = yaml.safe_load((calibration_dir / "cam-lidar-imu.yaml").read_text())
    T_base_lidar = transform_from_t_xyz_q_xyzw(BASE_T_LIDAR_T_XYZ_Q_XYZW)
    T_lidar_base = invert_transform(T_base_lidar)

    cameras_by_id: dict[int, CameraCalibration] = {}
    for camera_idx in range(3):
        cam_name = f"cam{camera_idx}"
        intrinsics = yaml.safe_load((calibration_dir / f"{cam_name}.yaml").read_text())
        K = intrinsics["camera_matrix"]["data"]
        distortion = np.array(intrinsics["distortion_coefficients"]["data"], dtype=np.float32)
        T_cam_lidar = np.array(extrinsics[cam_name]["T_cam_lidar"], dtype=np.float32)
        image_dir = (
            sequence_dir
            / "processed/colmap"
            / f"alphasense_driver_ros_{cam_name}_debayered_image_compressed"
        )
        image_paths = sorted(image_dir.glob("*.jpg"), key=timestamp_ns_from_image_path)
        if not image_paths:
            raise RuntimeError(f"No RGB images found for {cam_name}: {image_dir}")
        cameras_by_id[camera_idx] = CameraCalibration(
            name=cam_name,
            image_dir=image_dir,
            image_width=int(intrinsics["image_width"]),
            image_height=int(intrinsics["image_height"]),
            fx=float(K[0]),
            fy=float(K[4]),
            cx=float(K[2]),
            cy=float(K[5]),
            distortion=distortion,
            T_cam_base=T_cam_lidar @ T_lidar_base,
            image_timestamps_ns=[timestamp_ns_from_image_path(p) for p in image_paths],
            image_paths=image_paths,
        )
    return [cameras_by_id[i] for i in CAMERA_ORDER]


def find_camera_image(camera: CameraCalibration, pose: TimedPose) -> Path | None:
    target = timestamp_ns(pose.sec, pose.nsec)
    timestamps = camera.image_timestamps_ns
    lo = 0
    hi = len(timestamps)
    while lo < hi:
        mid = (lo + hi) // 2
        if timestamps[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    best_idx = None
    best_delta = None
    for idx in (lo - 1, lo, lo + 1):
        if idx < 0 or idx >= len(timestamps):
            continue
        delta = abs(timestamps[idx] - target)
        if best_delta is None or delta < best_delta:
            best_delta = delta
            best_idx = idx
    if best_idx is None or best_delta is None or best_delta > IMAGE_MATCH_TOLERANCE_NS:
        return None
    return camera.image_paths[best_idx]


def load_rgb_image(path: Path) -> np.ndarray:
    import imageio.v3 as iio

    image = iio.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.shape[-1] > 3:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def read_binary_pcd_xyz_intensity(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    header: dict[str, list[str]] = {}
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"{path} ended before DATA line")
            text = line.decode("ascii", errors="strict").strip()
            if text.startswith("#") or not text:
                continue
            parts = text.split()
            key = parts[0].upper()
            header[key] = parts[1:]
            if key == "DATA":
                break
        data = f.read()

    if header.get("DATA") != ["binary"]:
        raise ValueError(f"{path} must be binary PCD, got DATA {' '.join(header.get('DATA', []))}")
    fields = header.get("FIELDS", [])
    sizes = [int(x) for x in header.get("SIZE", [])]
    types = header.get("TYPE", [])
    counts = [int(x) for x in header.get("COUNT", ["1"] * len(fields))]
    points = int(header["POINTS"][0])

    if not fields or not sizes or not types:
        raise ValueError(f"{path} is missing FIELDS/SIZE/TYPE header entries")
    if counts != [1] * len(fields) or sizes != [4] * len(fields) or types != ["F"] * len(fields):
        raise ValueError(
            f"{path} uses an unsupported PCD layout; expected float32 COUNT 1 fields"
        )
    if not {"x", "y", "z"}.issubset(fields):
        raise ValueError(f"{path} is missing x/y/z fields")

    values = np.frombuffer(data, dtype="<f4")
    expected = points * len(fields)
    if values.size != expected:
        raise ValueError(f"{path} has {values.size} float32 values, expected {expected}")
    table = values.reshape(points, len(fields))
    x_idx, y_idx, z_idx = (fields.index("x"), fields.index("y"), fields.index("z"))
    xyz = table[:, [x_idx, y_idx, z_idx]].astype(np.float32, copy=False)
    intensity = table[:, fields.index("intensity")] if "intensity" in fields else None
    return xyz, intensity


def normalize_quat(q_wxyz: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q_wxyz))
    if norm <= 0.0:
        raise ValueError("zero-length quaternion")
    return (q_wxyz / norm).astype(np.float32)


def quat_wxyz_to_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quat(q_wxyz)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def quat_wxyz_from_matrix(rotation: np.ndarray) -> np.ndarray:
    R = rotation.astype(np.float64, copy=False)
    trace = float(np.trace(R))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (R[2, 1] - R[1, 2]) / scale,
                (R[0, 2] - R[2, 0]) / scale,
                (R[1, 0] - R[0, 1]) / scale,
            ],
            dtype=np.float32,
        )
    else:
        axis = int(np.argmax(np.diag(R)))
        if axis == 0:
            scale = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            quat = np.array(
                [
                    (R[2, 1] - R[1, 2]) / scale,
                    0.25 * scale,
                    (R[0, 1] + R[1, 0]) / scale,
                    (R[0, 2] + R[2, 0]) / scale,
                ],
                dtype=np.float32,
            )
        elif axis == 1:
            scale = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            quat = np.array(
                [
                    (R[0, 2] - R[2, 0]) / scale,
                    (R[0, 1] + R[1, 0]) / scale,
                    0.25 * scale,
                    (R[1, 2] + R[2, 1]) / scale,
                ],
                dtype=np.float32,
            )
        else:
            scale = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            quat = np.array(
                [
                    (R[1, 0] - R[0, 1]) / scale,
                    (R[0, 2] + R[2, 0]) / scale,
                    (R[1, 2] + R[2, 1]) / scale,
                    0.25 * scale,
                ],
                dtype=np.float32,
            )
    return normalize_quat(quat)


def transform_from_t_xyz_q_xyzw(t_xyz_q_xyzw: np.ndarray) -> np.ndarray:
    q_wxyz = np.array(
        [
            t_xyz_q_xyzw[6],
            t_xyz_q_xyzw[3],
            t_xyz_q_xyzw[4],
            t_xyz_q_xyzw[5],
        ],
        dtype=np.float32,
    )
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = quat_wxyz_to_matrix(q_wxyz)
    T[:3, 3] = t_xyz_q_xyzw[:3]
    return T


def transform_from_timed_pose(pose: TimedPose) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = quat_wxyz_to_matrix(pose.quaternion_wxyz)
    T[:3, 3] = pose.position
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    T_inv = np.eye(4, dtype=np.float32)
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -(R.T @ t)
    return T_inv


def finite_range_mask(xyz: np.ndarray, range_min: float, range_max: float) -> tuple[np.ndarray, np.ndarray]:
    ranges = np.linalg.norm(xyz, axis=1).astype(np.float32)
    mask = np.isfinite(xyz).all(axis=1) & np.isfinite(ranges)
    mask &= (ranges >= range_min) & (ranges <= range_max)
    return mask, ranges


def transform_points(points_lidar: np.ndarray, pose: TimedPose) -> np.ndarray:
    rotation = quat_wxyz_to_matrix(pose.quaternion_wxyz)
    return points_lidar @ rotation.T + pose.position


def transform_points_h(points: np.ndarray, T_dst_src: np.ndarray) -> np.ndarray:
    return points @ T_dst_src[:3, :3].T + T_dst_src[:3, 3]


def camera_world_transform(frame: Frame, camera: CameraCalibration) -> np.ndarray:
    T_world_base = transform_from_timed_pose(frame.pose)
    T_base_cam = invert_transform(camera.T_cam_base)
    return T_world_base @ T_base_cam


def downsample_image_for_viser(image: np.ndarray, max_width: int) -> np.ndarray:
    if image.shape[1] <= max_width:
        return image
    step = max(1, int(math.ceil(image.shape[1] / max_width)))
    return np.ascontiguousarray(image[::step, ::step])


def project_fisheye(
    points_cam: np.ndarray,
    camera: CameraCalibration,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = points_cam[:, 2]
    x = points_cam[:, 0] / np.clip(z, 1e-7, None)
    y = points_cam[:, 1] / np.clip(z, 1e-7, None)
    r = np.sqrt(x * x + y * y)
    theta = np.arctan(r)
    theta2 = theta * theta
    k1, k2, k3, k4 = camera.distortion
    theta_d = theta * (
        1.0
        + k1 * theta2
        + k2 * theta2 * theta2
        + k3 * theta2 * theta2 * theta2
        + k4 * theta2 * theta2 * theta2 * theta2
    )
    scale = np.ones_like(r, dtype=np.float32)
    nonzero = r > 1e-8
    scale[nonzero] = theta_d[nonzero] / r[nonzero]
    u = camera.fx * x * scale + camera.cx
    v = camera.fy * y * scale + camera.cy
    valid = (
        (z > 0.0)
        & np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0.0)
        & (u < camera.image_width)
        & (v >= 0.0)
        & (v < camera.image_height)
    )
    return u, v, valid


def colorize_from_registered_cameras(
    points_base: np.ndarray,
    frame: Frame,
    cameras: list[CameraCalibration],
    fallback_colors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    colors = fallback_colors.copy()
    has_camera_rgb = np.zeros(points_base.shape[0], dtype=bool)
    counts: dict[str, int] = {}

    for camera in cameras:
        image_path = find_camera_image(camera, frame.pose)
        if image_path is None:
            counts[camera.name] = 0
            continue
        image = load_rgb_image(image_path)
        points_cam = transform_points_h(points_base, camera.T_cam_base)
        u, v, valid = project_fisheye(points_cam, camera)
        valid &= ~has_camera_rgb
        if not valid.any():
            counts[camera.name] = 0
            continue
        uu = np.rint(u[valid]).astype(np.int64)
        vv = np.rint(v[valid]).astype(np.int64)
        uu = np.clip(uu, 0, camera.image_width - 1)
        vv = np.clip(vv, 0, camera.image_height - 1)
        colors[valid] = image[vv, uu]
        has_camera_rgb[valid] = True
        counts[camera.name] = int(valid.sum())

    return colors, has_camera_rgb, counts


def extract_camera_feature_grid(feature_model, image: np.ndarray, torch_module, device: str):
    image_t = torch_module.from_numpy(image).to(device=device)
    if str(device).startswith("cuda"):
        with torch_module.autocast("cuda", dtype=torch_module.bfloat16):
            feats = feature_model.extract_patch_features(image_t)
    else:
        feats = feature_model.extract_patch_features(image_t)
    return feats.contiguous()


def empty_cuda_cache(torch_module) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.empty_cache()


def colorize_and_featurize_from_registered_cameras(
    points_base: np.ndarray,
    frame: Frame,
    cameras: list[CameraCalibration],
    fallback_colors: np.ndarray,
    feature_model,
    torch_module,
    device: str,
    feature_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int], dict[str, int]]:
    colors = fallback_colors.copy()
    features = np.zeros((points_base.shape[0], feature_dim), dtype=np.float16)
    has_camera_rgb = np.zeros(points_base.shape[0], dtype=bool)
    has_feature = np.zeros(points_base.shape[0], dtype=bool)
    rgb_counts: dict[str, int] = {}
    feature_counts: dict[str, int] = {}

    for camera in cameras:
        image_path = find_camera_image(camera, frame.pose)
        if image_path is None:
            rgb_counts[camera.name] = 0
            feature_counts[camera.name] = 0
            continue

        image = load_rgb_image(image_path)
        feature_grid_t = extract_camera_feature_grid(feature_model, image, torch_module, device)
        feature_grid = feature_grid_t.detach().cpu().numpy().astype(np.float16, copy=False)
        feature_h, feature_w = feature_grid.shape[:2]

        points_cam = transform_points_h(points_base, camera.T_cam_base)
        u, v, valid = project_fisheye(points_cam, camera)
        valid &= ~has_camera_rgb
        if not valid.any():
            rgb_counts[camera.name] = 0
            feature_counts[camera.name] = 0
            continue

        uu = np.clip(np.rint(u[valid]).astype(np.int64), 0, camera.image_width - 1)
        vv = np.clip(np.rint(v[valid]).astype(np.int64), 0, camera.image_height - 1)
        colors[valid] = image[vv, uu]
        has_camera_rgb[valid] = True
        rgb_counts[camera.name] = int(valid.sum())

        gx = np.clip(
            np.floor(u[valid] * (feature_w / camera.image_width)).astype(np.int64),
            0,
            feature_w - 1,
        )
        gy = np.clip(
            np.floor(v[valid] * (feature_h / camera.image_height)).astype(np.int64),
            0,
            feature_h - 1,
        )
        features[valid] = feature_grid[gy, gx]
        has_feature[valid] = True
        feature_counts[camera.name] = int(valid.sum())

    return colors, has_camera_rgb, features, has_feature, rgb_counts, feature_counts


def infer_bounds(
    frames: Iterable[Frame],
    range_min: float,
    range_max: float,
    padding_m: float,
    elevation_min_deg: float | None,
    elevation_max_deg: float | None,
) -> Bounds:
    min_xyz = np.full(3, np.inf, dtype=np.float32)
    max_xyz = np.full(3, -np.inf, dtype=np.float32)
    elev_min = np.inf
    elev_max = -np.inf

    for frame in frames:
        xyz, _ = read_binary_pcd_xyz_intensity(frame.cloud_path)
        mask, ranges = finite_range_mask(xyz, range_min, range_max)
        if not mask.any():
            continue
        xyz_valid = xyz[mask]
        world = transform_points(xyz_valid, frame.pose)
        min_xyz = np.minimum(min_xyz, world.min(axis=0))
        max_xyz = np.maximum(max_xyz, world.max(axis=0))

        if elevation_min_deg is None or elevation_max_deg is None:
            xy_norm = np.linalg.norm(xyz_valid[:, :2], axis=1)
            elev = np.arctan2(xyz_valid[:, 2], xy_norm)
            elev_min = min(elev_min, float(elev.min()))
            elev_max = max(elev_max, float(elev.max()))

    if not np.isfinite(min_xyz).all():
        raise RuntimeError("No finite in-range points found while inferring bounds.")

    min_xyz -= padding_m
    max_xyz += padding_m
    if elevation_min_deg is not None:
        elev_min = math.radians(elevation_min_deg)
    else:
        elev_min -= math.radians(1.0)
    if elevation_max_deg is not None:
        elev_max = math.radians(elevation_max_deg)
    else:
        elev_max += math.radians(1.0)
    if elev_min >= elev_max:
        raise ValueError("elevation minimum must be below maximum")
    return Bounds(min_xyz, max_xyz, float(elev_min), float(elev_max))


def grid_shape_nzyx(extent_xyz: np.ndarray, voxel_size: float) -> tuple[int, int, int]:
    nx, ny, nz = np.ceil(extent_xyz / voxel_size).astype(np.int64).tolist()
    return (int(nz), int(ny), int(nx))


def make_range_image(
    xyz: np.ndarray,
    intensity: np.ndarray | None,
    *,
    image_height: int,
    image_width: int,
    range_min: float,
    range_max: float,
    elevation_min_rad: float,
    elevation_max_rad: float,
    color_mode: str,
    point_colors: np.ndarray | None = None,
    camera_color_mask: np.ndarray | None = None,
    point_features: np.ndarray | None = None,
    feature_mask: np.ndarray | None = None,
) -> RangeImageResult:
    mask, ranges = finite_range_mask(xyz, range_min, range_max)
    xy_norm = np.linalg.norm(xyz[:, :2], axis=1)
    elevation = np.arctan2(xyz[:, 2], xy_norm)
    mask &= (elevation >= elevation_min_rad) & (elevation <= elevation_max_rad)

    if not mask.any():
        feature_dim = 0 if point_features is None else int(point_features.shape[1])
        return RangeImageResult(
            range_image=np.zeros((image_height, image_width), dtype=np.float32),
            rgb_image=np.zeros((image_height, image_width, 3), dtype=np.uint8),
            projected_pixels=0,
            camera_rgb_pixels=0,
            sample_points_base=np.zeros((0, 3), dtype=np.float32),
            sample_colors=np.zeros((0, 3), dtype=np.uint8),
            feature_grid=(
                None
                if point_features is None
                else np.zeros((image_height, image_width, feature_dim), dtype=np.float16)
            ),
        )

    points = xyz[mask]
    ranges = ranges[mask]
    elevation = elevation[mask]
    azimuth = np.arctan2(points[:, 1], points[:, 0])
    u = np.mod(np.rint((azimuth + math.pi) * (image_width / (2.0 * math.pi))), image_width)
    if image_height == 1:
        v = np.zeros_like(u)
    else:
        v = np.rint(
            (elevation_max_rad - elevation)
            * ((image_height - 1.0) / (elevation_max_rad - elevation_min_rad))
        )
    u_i = u.astype(np.int64)
    v_i = np.clip(v.astype(np.int64), 0, image_height - 1)
    flat = v_i * image_width + u_i

    order = np.lexsort((ranges, flat))
    flat_sorted = flat[order]
    first = np.empty(flat_sorted.shape[0], dtype=bool)
    first[0] = True
    first[1:] = flat_sorted[1:] != flat_sorted[:-1]
    winners = order[first]

    range_flat = np.zeros(image_height * image_width, dtype=np.float32)
    range_flat[flat[winners]] = ranges[winners]

    rgb_flat = np.zeros((image_height * image_width, 3), dtype=np.uint8)
    if point_colors is None:
        colors = colorize_points(
            points[winners],
            intensity[mask][winners] if intensity is not None else None,
            color_mode,
        )
        camera_rgb_pixels = 0
    else:
        colors = point_colors[mask][winners]
        camera_rgb_pixels = (
            int(camera_color_mask[mask][winners].sum())
            if camera_color_mask is not None
            else 0
        )
    rgb_flat[flat[winners]] = colors
    feature_grid = None
    feature_pixels = 0
    if point_features is not None:
        feature_dim = int(point_features.shape[1])
        feature_flat = np.zeros((image_height * image_width, feature_dim), dtype=np.float16)
        winner_features = point_features[mask][winners].copy()
        if feature_mask is not None:
            winner_feature_mask = feature_mask[mask][winners]
            feature_pixels = int(winner_feature_mask.sum())
            if feature_pixels > 0:
                mean_feature = winner_features[winner_feature_mask].astype(np.float32).mean(axis=0)
                winner_features[~winner_feature_mask] = mean_feature.astype(np.float16)
        feature_flat[flat[winners]] = winner_features
        feature_grid = feature_flat.reshape(image_height, image_width, feature_dim)
    return RangeImageResult(
        range_image=range_flat.reshape(image_height, image_width),
        rgb_image=rgb_flat.reshape(image_height, image_width, 3),
        projected_pixels=int(winners.size),
        camera_rgb_pixels=camera_rgb_pixels,
        sample_points_base=points[winners],
        sample_colors=colors,
        feature_grid=feature_grid,
        feature_pixels=feature_pixels,
    )


def colorize_points(
    points: np.ndarray,
    intensity: np.ndarray | None,
    color_mode: str,
) -> np.ndarray:
    if color_mode == "constant":
        return np.full((points.shape[0], 3), 180, dtype=np.uint8)
    if color_mode == "height":
        values = points[:, 2]
    elif intensity is not None:
        values = intensity
    else:
        values = np.linalg.norm(points, axis=1)

    finite = np.isfinite(values)
    if not finite.any():
        scaled = np.full(values.shape, 180.0, dtype=np.float32)
    else:
        lo, hi = np.percentile(values[finite], [1.0, 99.0])
        if hi <= lo:
            scaled = np.full(values.shape, 180.0, dtype=np.float32)
        else:
            scaled = np.clip((values - lo) / (hi - lo), 0.0, 1.0) * 255.0
    gray = scaled.astype(np.uint8)
    return np.repeat(gray[:, None], 3, axis=1)


def downsample_points(
    points: np.ndarray,
    colors: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] <= max_points:
        return points, colors
    stride = max(1, int(math.ceil(points.shape[0] / max_points)))
    return points[::stride][:max_points], colors[::stride][:max_points]


def box_line_segments(center: np.ndarray, extent: np.ndarray) -> np.ndarray:
    half = 0.5 * extent
    corners = np.array(
        [
            [center[0] - half[0], center[1] - half[1], center[2] - half[2]],
            [center[0] + half[0], center[1] - half[1], center[2] - half[2]],
            [center[0] + half[0], center[1] + half[1], center[2] - half[2]],
            [center[0] - half[0], center[1] + half[1], center[2] - half[2]],
            [center[0] - half[0], center[1] - half[1], center[2] + half[2]],
            [center[0] + half[0], center[1] - half[1], center[2] + half[2]],
            [center[0] + half[0], center[1] + half[1], center[2] + half[2]],
            [center[0] - half[0], center[1] + half[1], center[2] + half[2]],
        ],
        dtype=np.float32,
    )
    edges = np.array(
        [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
        ],
        dtype=np.int64,
    )
    return corners[edges]


def tsdf_surface_voxels_with_blocks(mapper) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    blocks = mapper.tsdf.export_blocks()
    coords = blocks["active_block_coords"].to(device="cpu").numpy().astype(np.float32)
    if coords.shape[0] == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros((0,), dtype=np.int64),
            blocks,
        )

    block_data = blocks["block_data"].to(device="cpu").float().numpy()
    sdf = block_data[..., 0]
    weight = block_data[..., 1]
    threshold = float(mapper.config.voxel_size) * VISER_TSDF_SURFACE_THRESHOLD_VOX
    mask = (weight > float(mapper.config.minimum_tsdf_weight)) & (np.abs(sdf) <= threshold)
    if not mask.any():
        mask = (weight > 0.0) & (sdf <= 0.0)
    if not mask.any():
        mask = weight > 0.0
    if not mask.any():
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
            np.zeros((0,), dtype=np.int64),
            blocks,
        )

    block_idx, local_idx = np.nonzero(mask)
    block_size = int(mapper.config.block_size)
    voxel_size = float(mapper.config.voxel_size)
    nz, ny, nx = mapper.config.grid_shape
    block_offsets = np.array(
        [
            math.ceil(nx / block_size) // 2,
            math.ceil(ny / block_size) // 2,
            math.ceil(nz / block_size) // 2,
        ],
        dtype=np.float32,
    )
    grid_shape_xyz = np.array([nx, ny, nz], dtype=np.float32)
    lx = local_idx % block_size
    ly = (local_idx // block_size) % block_size
    lz = local_idx // (block_size * block_size)
    local_xyz = np.stack([lx, ly, lz], axis=1).astype(np.float32)
    voxel_xyz = (coords[block_idx] + block_offsets) * block_size + local_xyz
    centers = (
        (voxel_xyz + 0.5 - 0.5 * grid_shape_xyz) * voxel_size
        + mapper.config.grid_center.detach().to(device="cpu").numpy()
    ).astype(np.float32)

    rgbw = blocks["block_rgb"].to(device="cpu").float().numpy()
    weight_rgb = np.clip(rgbw[block_idx, 3:4], 1e-6, None)
    colors = np.clip((rgbw[block_idx, :3] / weight_rgb) * 255.0, 0.0, 255.0).astype(np.uint8)
    return centers, colors, block_idx.astype(np.int64), blocks


def tsdf_surface_voxel_preview(mapper) -> tuple[np.ndarray, np.ndarray]:
    centers, colors, _, _ = tsdf_surface_voxels_with_blocks(mapper)
    return centers, colors


def write_metadata(path: Path, payload: dict) -> None:
    metadata_path = path.with_suffix(path.suffix + ".json")
    metadata_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def start_viser():
    from curobo.viewer import ViserVisualizer

    visualizer = None
    port = None
    last_error: Exception | None = None
    for candidate_port in VISER_PORT_RANGE:
        try:
            visualizer = ViserVisualizer(
                connect_port=candidate_port,
                add_robot_to_scene=False,
                add_control_frames=False,
            )
            port = int(visualizer._server.get_port())
            break
        except Exception as exc:
            last_error = exc
    if visualizer is None or port is None:
        raise RuntimeError(f"Could not start Viser: {last_error}")
    print(f"Viser running at http://localhost:{port}")
    return visualizer, port


def add_viser_context(
    visualizer,
    frames: list[Frame],
    grid_center: np.ndarray,
    extent_xyz: np.ndarray,
) -> None:
    trajectory = np.stack([frame.pose.position for frame in frames]).astype(np.float32)
    if trajectory.shape[0] >= 2:
        segments = np.stack([trajectory[:-1], trajectory[1:]], axis=1)
        segment_colors = np.full((*segments.shape[:2], 3), [255, 220, 0], dtype=np.uint8)
        visualizer._server.scene.add_line_segments(
            "/trajectory",
            points=segments,
            colors=segment_colors,
            line_width=4.0,
        )
    visualizer._server.scene.add_batched_axes(
        name="/sensor_poses",
        batched_positions=trajectory[:: max(1, len(trajectory) // 32)],
        batched_wxyzs=np.stack([frame.pose.quaternion_wxyz for frame in frames])[
            :: max(1, len(trajectory) // 32)
        ],
        axes_length=0.6,
        axes_radius=0.015,
    )

    box_segments = box_line_segments(grid_center, extent_xyz)
    box_colors = np.full((*box_segments.shape[:2], 3), [80, 180, 255], dtype=np.uint8)
    visualizer._server.scene.add_line_segments(
        "/tsdf_bounds",
        points=box_segments,
        colors=box_colors,
        line_width=2.0,
    )


def update_viser_tsdf_voxels(visualizer, mapper, voxel_size: float) -> int:
    tsdf_points, tsdf_colors = tsdf_surface_voxel_preview(mapper)
    tsdf_points, tsdf_colors = downsample_points(
        tsdf_points,
        tsdf_colors,
        VISER_MAX_TSDF_POINTS,
    )

    if tsdf_points.size:
        visualizer.add_point_cloud(
            pointcloud=tsdf_points,
            colors=tsdf_colors,
            point_size=max(0.025, voxel_size * 1.5),
            name=VISER_TSDF_POINTCLOUD_NAME,
        )
    return int(tsdf_points.shape[0])


def pose_from_transform(T: np.ndarray, Pose, torch_module):
    position = torch_module.tensor(T[:3, 3][None], dtype=torch_module.float32)
    quaternion = torch_module.tensor(
        quat_wxyz_from_matrix(T[:3, :3])[None],
        dtype=torch_module.float32,
    )
    return Pose(position=position, quaternion=quaternion, normalize_rotation=True)


def synchronize_if_cuda(torch_module, device) -> None:
    if device.type == "cuda":
        torch_module.cuda.synchronize(device)


def esdf_xy_slice_image(
    voxel_grid,
    slice_size_m: float,
    slice_resolution: int,
    torch_module,
) -> np.ndarray | None:
    if voxel_grid.feature_tensor is None:
        return None

    from torch.nn import functional as F

    esdf_grid = voxel_grid.feature_tensor
    device = esdf_grid.device
    nx, ny, nz = esdf_grid.shape
    origin = torch_module.tensor(
        voxel_grid.pose[:3],
        dtype=torch_module.float32,
        device=device,
    )

    half = slice_size_m / 2.0
    u = torch_module.linspace(-half, half, slice_resolution, device=device)
    v = torch_module.linspace(-half, half, slice_resolution, device=device)
    uu, vv = torch_module.meshgrid(u, v, indexing="xy")
    flat_count = slice_resolution * slice_resolution
    world_points = torch_module.stack(
        [
            origin[0] + uu.flatten(),
            origin[1] + vv.flatten(),
            origin[2].expand(flat_count),
        ],
        dim=1,
    )

    local_points = world_points - origin
    half_extent = torch_module.tensor(
        [
            (nx - 1) * voxel_grid.voxel_size / 2.0,
            (ny - 1) * voxel_grid.voxel_size / 2.0,
            (nz - 1) * voxel_grid.voxel_size / 2.0,
        ],
        dtype=torch_module.float32,
        device=device,
    ).clamp_min(1e-6)
    normalized = local_points / half_extent

    coords = normalized[:, [2, 1, 0]].view(1, 1, slice_resolution, slice_resolution, 3)
    esdf_5d = esdf_grid.float().unsqueeze(0).unsqueeze(0)
    sampled = F.grid_sample(
        esdf_5d,
        coords.float(),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    values = sampled.squeeze().view(slice_resolution, slice_resolution).cpu().numpy()

    max_dist = max(float(np.max(values)), 0.1)
    max_negative_dist = max(float(np.abs(np.min(values))), 0.05)
    normalized_positive = np.clip(values / max_dist, -1.0, 1.0)
    normalized_negative = np.clip(values / max_negative_dist, -1.0, 1.0)

    colors = np.zeros((slice_resolution, slice_resolution, 3), dtype=np.uint8)
    neg_mask = normalized_negative < 0
    colors[neg_mask, 0] = ((1 + normalized_negative[neg_mask]) * 255).astype(np.uint8)
    colors[neg_mask, 1] = ((1 + normalized_negative[neg_mask]) * 255).astype(np.uint8)
    colors[neg_mask, 2] = 255

    pos_mask = normalized_positive >= 0
    colors[pos_mask, 0] = 255
    colors[pos_mask, 1] = ((1 - normalized_positive[pos_mask]) * 255).astype(np.uint8)
    colors[pos_mask, 2] = ((1 - normalized_positive[pos_mask]) * 255).astype(np.uint8)
    colors[np.abs(values) < voxel_grid.voxel_size * 0.5] = [0, 255, 0]
    return colors


def update_viser_esdf_slice(
    visualizer,
    voxel_grid,
    Pose,
    torch_module,
    slice_size_m: float,
    slice_resolution: int,
) -> tuple[int, int, int] | None:
    image = esdf_xy_slice_image(
        voxel_grid,
        slice_size_m=slice_size_m,
        slice_resolution=slice_resolution,
        torch_module=torch_module,
    )
    if image is None:
        return None

    T_world_slice = np.eye(4, dtype=np.float32)
    T_world_slice[:3, 3] = np.asarray(voxel_grid.pose[:3], dtype=np.float32)
    visualizer.add_image(
        image=image,
        render_width=slice_size_m,
        render_height=slice_size_m,
        pose=pose_from_transform(T_world_slice, Pose, torch_module),
        name=VISER_ESDF_SLICE_NAME,
    )
    return tuple(int(x) for x in voxel_grid.feature_tensor.shape)


def update_viser_rgb_cameras(visualizer, frame: Frame, cameras: list[CameraCalibration], Pose, torch_module) -> int:
    shown = 0
    for camera in cameras:
        image_path = find_camera_image(camera, frame.pose)
        if image_path is None:
            continue

        T_world_cam = camera_world_transform(frame, camera)
        camera_quat = quat_wxyz_from_matrix(T_world_cam[:3, :3])
        visualizer._server.scene.add_frame(
            f"/rgb/{camera.name}/camera",
            position=T_world_cam[:3, 3],
            wxyz=camera_quat,
            axes_length=VISER_RGB_CAMERA_AXES_LENGTH,
            axes_radius=0.015,
        )

        image = downsample_image_for_viser(
            load_rgb_image(image_path),
            VISER_RGB_MAX_IMAGE_WIDTH,
        )
        T_world_image = T_world_cam.copy()
        T_world_image[:3, 3] += T_world_image[:3, 2] * VISER_RGB_IMAGE_DISTANCE_M
        image_pose = pose_from_transform(T_world_image, Pose, torch_module)
        render_height = VISER_RGB_RENDER_WIDTH_M * float(image.shape[0]) / float(image.shape[1])
        visualizer.add_image(
            image=image,
            render_width=VISER_RGB_RENDER_WIDTH_M,
            render_height=render_height,
            pose=image_pose,
            name=f"/rgb/{camera.name}/image",
        )
        shown += 1
    return shown


def update_viser_feature_voxels(
    visualizer,
    mapper,
    voxel_size: float,
    pca_basis,
    pca_colorize_tensor,
) -> tuple[int, object]:
    points, _, block_idx, blocks = tsdf_surface_voxels_with_blocks(mapper)
    if points.shape[0] == 0 or "block_features" not in blocks:
        return 0, pca_basis

    block_features = blocks["block_features"].float()
    block_weight = blocks["block_feature_weight"].float().clamp(min=1e-6).unsqueeze(1)
    normalized_features = block_features / block_weight
    block_colors, pca_basis = pca_colorize_tensor(normalized_features, prev_basis=pca_basis)
    colors = block_colors[block_idx].cpu().numpy()
    points, colors = downsample_points(points, colors, VISER_MAX_FEATURE_POINTS)

    visualizer.add_point_cloud(
        pointcloud=points,
        colors=colors,
        point_size=max(0.025, voxel_size * 1.0),
        name=VISER_FEATURE_POINTCLOUD_NAME,
    )
    return int(points.shape[0]), pca_basis


def _clear_text_match_layer(visualizer, voxel_size: float) -> None:
    visualizer.add_point_cloud(
        pointcloud=np.zeros((0, 3), dtype=np.float32),
        colors=np.zeros((0, 3), dtype=np.uint8),
        point_size=max(0.025, voxel_size * 1.5),
        name=VISER_TEXT_MATCH_POINTCLOUD_NAME,
    )


def setup_text_query_gui(visualizer, mapper, feature_model, args: argparse.Namespace, torch_module):
    server = visualizer._server
    state = {
        "has_voxels": False,
        "text_prompt": None,
        "text_query": None,
        "last_match_count": 0,
    }

    with server.gui.add_folder("Text Query"):
        text_input = server.gui.add_text("prompt", initial_value="building")
        text_top_k = server.gui.add_slider(
            "top_k",
            min=10,
            max=5000,
            step=10,
            initial_value=TEXT_TOP_K,
        )
        text_min_score = server.gui.add_slider(
            "min_score",
            min=0.0,
            max=1.0,
            step=0.005,
            initial_value=TEXT_MIN_SCORE,
        )
        text_live_cb = server.gui.add_checkbox("live_update", initial_value=True)
        text_search = server.gui.add_button("Search")
        text_clear = server.gui.add_button("Clear")
        text_status = server.gui.add_text(
            "status",
            initial_value="Waiting for voxels...",
            disabled=True,
        )

    def _get_live_text_query():
        prompt = text_input.value.strip()
        if not prompt:
            state["text_prompt"] = None
            state["text_query"] = None
            return None, None
        if state["text_prompt"] != prompt or state["text_query"] is None:
            state["text_prompt"] = prompt
            state["text_query"] = feature_model.encode_text(prompt)[0]
        return prompt, state["text_query"]

    def _current_text_min_score():
        value = float(text_min_score.value)
        return value if value > 0.0 else None

    def run_text_match() -> int:
        if not state["has_voxels"]:
            text_status.value = "Waiting for voxels..."
            return 0

        prompt, text_query = _get_live_text_query()
        if text_query is None:
            _clear_text_match_layer(visualizer, args.voxel_size)
            state["last_match_count"] = 0
            text_status.value = "Enter a prompt to highlight matches."
            return 0

        min_score = _current_text_min_score()
        matched = mapper.extract_matching_feature_voxels(
            feature_vector=text_query,
            top_k=int(text_top_k.value),
            surface_only=True,
            minimum_score=min_score,
            feature_projector=feature_model.project_features,
        )
        if matched.block_pool_idx.numel() == 0:
            _clear_text_match_layer(visualizer, args.voxel_size)
            state["last_match_count"] = 0
            threshold_str = "off" if min_score is None else f"{min_score:.3f}"
            text_status.value = f"text='{prompt}', no matches (min_score={threshold_str})"
            del matched
            empty_cuda_cache(torch_module)
            return 0

        centers = matched.voxels.centers
        step = max(1, int(math.ceil(len(centers) / VISER_MAX_TEXT_MATCH_POINTS)))
        points = centers[::step].cpu().numpy()
        colors = np.zeros((len(points), 3), dtype=np.uint8)
        colors[:, 0] = 255
        colors[:, 2] = 255
        visualizer.add_point_cloud(
            pointcloud=points,
            colors=colors,
            point_size=max(0.035, args.voxel_size * 2.0),
            name=VISER_TEXT_MATCH_POINTCLOUD_NAME,
        )

        top_score = float(matched.block_scores[0].item())
        threshold_str = "off" if min_score is None else f"{min_score:.3f}"
        state["last_match_count"] = int(len(centers))
        text_status.value = (
            f"text='{prompt}', matched={len(centers)} "
            f"({matched.block_scores.numel()} blocks, top={top_score:.3f}, "
            f"min_score={threshold_str})"
        )
        del matched, centers, points
        empty_cuda_cache(torch_module)
        return int(state["last_match_count"])

    def refresh_after_integration() -> int:
        state["has_voxels"] = True
        if text_live_cb.value:
            return run_text_match()
        return int(state["last_match_count"])

    @text_search.on_click
    def _on_text_search(_):
        run_text_match()

    @text_clear.on_click
    def _on_text_clear(_):
        _clear_text_match_layer(visualizer, args.voxel_size)
        state["last_match_count"] = 0
        text_status.value = "Text highlight cleared."

    @text_top_k.on_update
    def _on_text_top_k(_):
        if text_live_cb.value:
            run_text_match()

    @text_min_score.on_update
    def _on_text_min_score(_):
        if text_live_cb.value:
            run_text_match()

    @text_live_cb.on_update
    def _on_text_live(_):
        if text_live_cb.value:
            run_text_match()

    return refresh_after_integration


def keep_viser_alive(
    port: int,
    *,
    enable_features: bool = False,
    enable_esdf: bool = False,
) -> None:
    layers = [VISER_TSDF_POINTCLOUD_NAME, "/rgb/<camera>/image"]
    if enable_esdf:
        layers.append(VISER_ESDF_SLICE_NAME)
    if enable_features:
        layers.extend([VISER_FEATURE_POINTCLOUD_NAME, VISER_TEXT_MATCH_POINTCLOUD_NAME])
    print("Viser layers update during integration: " + ", ".join(layers))
    print(f"Viser running at http://localhost:{port}")
    print("Press Ctrl+C to stop the viewer.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


def run_mapper(args: argparse.Namespace, frames: list[Frame], bounds: Bounds) -> None:
    if args.max_blocks is not None:
        raise ValueError(
            "This cuRobo checkout exposes MapperCfg.max_blocks as a computed property. "
            "Control the block budget with --extent-meters-xyz, --voxel-size, "
            "--truncation-distance, --block-size, or --roughness instead."
        )

    sys.path.insert(0, str(args.curobo_root))

    import torch

    if args.enable_features:
        from curobo.examples.getting_started.feature_mapping import (
            CRadioInference,
            pca_colorize_tensor,
        )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"{args.device} requested, but torch.cuda.is_available() is false. "
            "Run --dry-run on CPU-only hosts, or run integration where CUDA is visible."
        )

    cameras = load_camera_calibrations(args.sequence_dir)

    from curobo._src.perception.mapper.mapper import Mapper
    from curobo._src.perception.mapper.mapper_cfg import MapperCfg
    from curobo._src.types.lidar import LidarObservation
    from curobo._src.types.pose import Pose
    from curobo._src.util.warp import init_warp

    feature_model = None
    feature_dim = 0
    feature_shape_hw: tuple[int, int] | None = None
    if args.enable_features:
        print(f"Loading C-RADIO ({args.feature_model}) via NVlabs/RADIO torch.hub...")
        feature_model = CRadioInference(
            model_name=args.feature_model,
            device=args.device,
            text_adaptor_name=TEXT_ADAPTOR_NAME,
        )
        print(f"Text query enabled with adaptor='{TEXT_ADAPTOR_NAME}'.")
        probe_image_path = find_camera_image(cameras[0], frames[0].pose)
        if probe_image_path is None:
            raise RuntimeError("Could not find a timestamp-matched RGB image for feature probing.")
        probe_image = load_rgb_image(probe_image_path)
        probe_features = extract_camera_feature_grid(feature_model, probe_image, torch, args.device)
        feature_shape_hw = (int(probe_features.shape[0]), int(probe_features.shape[1]))
        feature_dim = int(probe_features.shape[-1])
        print(f"Feature shape from {args.feature_model}: {feature_shape_hw} x {feature_dim}")
        del probe_features
        empty_cuda_cache(torch)

    init_warp()
    device = torch.device(args.device)
    extent_xyz = np.asarray(args.extent_meters_xyz, dtype=np.float32) if args.extent_meters_xyz else bounds.extent_xyz
    grid_center = np.asarray(args.grid_center, dtype=np.float32) if args.grid_center else bounds.center
    truncation_distance = args.truncation_distance or (3.0 * args.voxel_size)

    cfg_kwargs = {
        "extent_meters_xyz": tuple(float(x) for x in extent_xyz),
        "grid_center": torch.tensor(grid_center, dtype=torch.float32),
        "voxel_size": float(args.voxel_size),
        "truncation_distance": float(truncation_distance),
        "depth_minimum_distance": float(args.range_min),
        "depth_maximum_distance": float(args.range_max),
        "image_height": 1,
        "image_width": 1,
        "num_cameras": 1,
        "lidar_num_sensors": 1,
        "lidar_image_height": int(args.lidar_height),
        "lidar_image_width": int(args.lidar_width),
        "block_size": int(args.block_size),
        "roughness": float(args.roughness),
        "extent_esdf_meters_xyz": (float(args.esdf_extent_meters),) * 3,
        "esdf_voxel_size": float(args.esdf_voxel_size),
        "device": args.device,
        "minimum_tsdf_weight": 0.0001,
    }
    if args.enable_features:
        cfg_kwargs.update(
            {
                "lidar_feature_grid_height": int(args.lidar_height),
                "lidar_feature_grid_width": int(args.lidar_width),
                "feature_dim": feature_dim,
                "feature_grid_height": int(args.lidar_height),
                "feature_grid_width": int(args.lidar_width),
                "feature_integration_kernel": "grouped",
                "max_support_pixels_per_block_lidar": 32,
            }
        )
    cfg = MapperCfg(**cfg_kwargs)
    mapper = Mapper(cfg)
    print(f"Mapper initialized: {mapper.memory_usage_mb():.1f} MB")
    if args.esdf_every > 0:
        esdf_grid_shape = tuple(
            int(math.ceil(args.esdf_extent_meters / args.esdf_voxel_size)) for _ in range(3)
        )
        print(
            "ESDF slice timing: "
            f"every {args.esdf_every} frame(s), volume={(args.esdf_extent_meters,) * 3} m, "
            f"voxel_size={args.esdf_voxel_size:.4f} m, grid_shape={esdf_grid_shape}, "
            f"origin={ESDF_ORIGIN_XYZ}"
        )
    else:
        print("ESDF slice timing disabled.")
    esdf_origin = torch.tensor(ESDF_ORIGIN_XYZ, dtype=torch.float32, device=device)
    valid_range = torch.tensor(
        [[args.range_min, args.range_max]],
        dtype=torch.float32,
        device=device,
    )
    elevation_range = torch.tensor(
        [[bounds.elevation_min_rad, bounds.elevation_max_rad]],
        dtype=torch.float32,
        device=device,
    )
    visualizer, viser_port = start_viser()
    add_viser_context(visualizer, frames, grid_center, extent_xyz)
    refresh_text_query = (
        setup_text_query_gui(visualizer, mapper, feature_model, args, torch)
        if args.enable_features
        else None
    )
    pca_basis = None

    integrated = 0
    projected_points = 0
    camera_rgb_pixels = 0
    camera_feature_pixels = 0
    camera_rgb_counts = {camera.name: 0 for camera in cameras}
    camera_feature_counts = {camera.name: 0 for camera in cameras}
    for i, frame in enumerate(frames):
        xyz, intensity = read_binary_pcd_xyz_intensity(frame.cloud_path)
        fallback_colors = colorize_points(xyz, intensity, args.color_mode)
        point_features = None
        feature_mask = None
        if args.enable_features:
            (
                point_colors,
                camera_color_mask,
                point_features,
                feature_mask,
                frame_camera_counts,
                frame_feature_counts,
            ) = colorize_and_featurize_from_registered_cameras(
                xyz,
                frame,
                cameras,
                fallback_colors,
                feature_model,
                torch,
                args.device,
                feature_dim,
            )
        else:
            point_colors, camera_color_mask, frame_camera_counts = colorize_from_registered_cameras(
                xyz,
                frame,
                cameras,
                fallback_colors,
            )
            frame_feature_counts = {}
        range_result = make_range_image(
            xyz,
            intensity,
            image_height=args.lidar_height,
            image_width=args.lidar_width,
            range_min=args.range_min,
            range_max=args.range_max,
            elevation_min_rad=bounds.elevation_min_rad,
            elevation_max_rad=bounds.elevation_max_rad,
            color_mode=args.color_mode,
            point_colors=point_colors,
            camera_color_mask=camera_color_mask,
            point_features=point_features,
            feature_mask=feature_mask,
        )
        projected_points += range_result.projected_pixels
        camera_rgb_pixels += range_result.camera_rgb_pixels
        camera_feature_pixels += range_result.feature_pixels
        for camera_name, count in frame_camera_counts.items():
            camera_rgb_counts[camera_name] = camera_rgb_counts.get(camera_name, 0) + count
        for camera_name, count in frame_feature_counts.items():
            camera_feature_counts[camera_name] = camera_feature_counts.get(camera_name, 0) + count
        if range_result.projected_pixels == 0:
            continue

        position = torch.tensor(frame.pose.position[None], dtype=torch.float32, device=device)
        quaternion = torch.tensor(frame.pose.quaternion_wxyz[None], dtype=torch.float32, device=device)
        observation_kwargs = {
            "range_image": torch.from_numpy(range_result.range_image[None]).to(device=device),
            "rgb_image": torch.from_numpy(range_result.rgb_image[None]).to(device=device),
            "pose": Pose(position=position, quaternion=quaternion, normalize_rotation=True),
            "valid_range_m": valid_range,
            "elevation_range_rad": elevation_range,
        }
        if args.enable_features and range_result.feature_grid is not None:
            observation_kwargs["feature_grid"] = torch.from_numpy(
                range_result.feature_grid[None]
            ).to(device=device)
        observation = LidarObservation(**observation_kwargs)

        synchronize_if_cuda(torch, device)
        tsdf_t0 = time.perf_counter()
        mapper.integrate(lidar_observation=observation)
        synchronize_if_cuda(torch, device)
        tsdf_ms = (time.perf_counter() - tsdf_t0) * 1000.0

        integrated += 1
        esdf_ms = None
        esdf_slice_ms = None
        esdf_shape = None
        if args.esdf_every > 0 and integrated % args.esdf_every == 0:
            synchronize_if_cuda(torch, device)
            esdf_t0 = time.perf_counter()
            voxel_grid = mapper.compute_esdf(esdf_origin=esdf_origin)
            synchronize_if_cuda(torch, device)
            esdf_ms = (time.perf_counter() - esdf_t0) * 1000.0

            slice_t0 = time.perf_counter()
            esdf_shape = update_viser_esdf_slice(
                visualizer,
                voxel_grid,
                Pose,
                torch,
                slice_size_m=float(args.esdf_extent_meters),
                slice_resolution=int(args.esdf_slice_resolution),
            )
            synchronize_if_cuda(torch, device)
            esdf_slice_ms = (time.perf_counter() - slice_t0) * 1000.0

        viser_voxels = update_viser_tsdf_voxels(visualizer, mapper, args.voxel_size)
        feature_voxels = 0
        text_matches = 0
        if args.enable_features:
            feature_voxels, pca_basis = update_viser_feature_voxels(
                visualizer,
                mapper,
                args.voxel_size,
                pca_basis,
                pca_colorize_tensor,
            )
            text_matches = refresh_text_query() if refresh_text_query is not None else 0
        viser_rgb_images = update_viser_rgb_cameras(visualizer, frame, cameras, Pose, torch)
        message = (
            f"viser_update={integrated:04d}/{len(frames)} "
            f"tsdf_ms={tsdf_ms:.2f} "
            f"tsdf_surface_voxels={viser_voxels} "
            f"rgb_images={viser_rgb_images}"
        )
        if esdf_ms is not None:
            message += f" esdf_ms={esdf_ms:.2f}"
        if esdf_shape is not None:
            message += f" esdf_shape={esdf_shape}"
        if esdf_slice_ms is not None:
            message += f" esdf_slice_ms={esdf_slice_ms:.2f}"
        if args.enable_features:
            message += f" feature_voxels={feature_voxels} text_matches={text_matches}"
        print(message)

        if args.status_every > 0 and (integrated == 1 or integrated % args.status_every == 0):
            stats = mapper.get_stats(scan_pool=False)
            lidar_stats = stats.get("last_lidar_integration", {})
            status = (
                f"integrated={integrated:04d}/{len(frames)} "
                f"projected_pixels={range_result.projected_pixels} "
                f"camera_rgb_pixels={range_result.camera_rgb_pixels} "
                f"active_blocks={stats.get('active_blocks')} "
                f"visible_blocks={lidar_stats.get('num_visible_blocks')}"
            )
            if args.enable_features:
                status += f" feature_pixels={range_result.feature_pixels}"
            print(status)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mapper.save_blocks(args.output)
    stats = mapper.get_stats(scan_pool=True, scan_hash=False)
    metadata = build_metadata(
        args,
        frames,
        bounds,
        stats,
        projected_points,
        integrated,
        camera_rgb_pixels,
        camera_rgb_counts,
        feature_pixels=camera_feature_pixels if args.enable_features else None,
        camera_feature_counts=camera_feature_counts if args.enable_features else None,
        feature_dim=feature_dim if args.enable_features else None,
        feature_shape_hw=feature_shape_hw,
    )
    write_metadata(args.output, metadata)
    print(f"saved {args.output}")
    print(f"saved {args.output.with_suffix(args.output.suffix + '.json')}")
    print(
        json.dumps(
            {
                k: stats[k]
                for k in ("frame_count", "active_blocks", "num_allocated", "memory_mb")
                if k in stats
            },
            indent=2,
        )
    )
    keep_viser_alive(
        viser_port,
        enable_features=args.enable_features,
        enable_esdf=args.esdf_every > 0,
    )


def build_metadata(
    args: argparse.Namespace,
    frames: list[Frame],
    bounds: Bounds,
    stats: dict,
    projected_points: int,
    integrated_frames: int,
    camera_rgb_pixels: int,
    camera_rgb_counts: dict[str, int],
    feature_pixels: int | None = None,
    camera_feature_counts: dict[str, int] | None = None,
    feature_dim: int | None = None,
    feature_shape_hw: tuple[int, int] | None = None,
) -> dict:
    extent_xyz = np.asarray(args.extent_meters_xyz, dtype=np.float32) if args.extent_meters_xyz else bounds.extent_xyz
    center = np.asarray(args.grid_center, dtype=np.float32) if args.grid_center else bounds.center
    payload = {
        "sequence_dir": str(args.sequence_dir),
        "num_selected_frames": len(frames),
        "num_integrated_frames": integrated_frames,
        "projected_lidar_pixels_total": projected_points,
        "camera_rgb_lidar_pixels_total": camera_rgb_pixels,
        "camera_rgb_samples_by_camera": camera_rgb_counts,
        "camera_rgb_match_tolerance_ns": IMAGE_MATCH_TOLERANCE_NS,
        "first_cloud": frames[0].cloud_path.name,
        "last_cloud": frames[-1].cloud_path.name,
        "range_m": [args.range_min, args.range_max],
        "elevation_range_deg": [
            math.degrees(bounds.elevation_min_rad),
            math.degrees(bounds.elevation_max_rad),
        ],
        "lidar_image_shape_hw": [args.lidar_height, args.lidar_width],
        "grid_center_xyz": center.astype(float).tolist(),
        "extent_meters_xyz": extent_xyz.astype(float).tolist(),
        "grid_shape_nzyx": list(grid_shape_nzyx(extent_xyz, args.voxel_size)),
        "voxel_size": args.voxel_size,
        "esdf_origin_xyz": list(ESDF_ORIGIN_XYZ),
        "esdf_extent_meters_xyz": [args.esdf_extent_meters] * 3,
        "esdf_voxel_size": args.esdf_voxel_size,
        "esdf_every": args.esdf_every,
        "esdf_slice_resolution": args.esdf_slice_resolution,
        "truncation_distance": args.truncation_distance or (3.0 * args.voxel_size),
        "block_size": args.block_size,
        "max_blocks": args.max_blocks,
        "roughness": args.roughness,
        "mapper_stats": stats,
    }
    if args.enable_features:
        payload.update(
            {
                "feature_model": args.feature_model,
                "text_adaptor": TEXT_ADAPTOR_NAME,
                "feature_dim": feature_dim,
                "feature_grid_shape_hw": (
                    None if feature_shape_hw is None else list(feature_shape_hw)
                ),
                "lidar_feature_grid_shape_hw": [args.lidar_height, args.lidar_width],
                "camera_feature_lidar_pixels_total": feature_pixels,
                "camera_feature_samples_by_camera": camera_feature_counts,
                "viser_feature_layer": VISER_FEATURE_POINTCLOUD_NAME,
                "viser_text_query_layer": VISER_TEXT_MATCH_POINTCLOUD_NAME,
            }
        )
    return payload


def print_dry_run(args: argparse.Namespace, frames: list[Frame], bounds: Bounds) -> None:
    extent_xyz = np.asarray(args.extent_meters_xyz, dtype=np.float32) if args.extent_meters_xyz else bounds.extent_xyz
    center = np.asarray(args.grid_center, dtype=np.float32) if args.grid_center else bounds.center
    sample_xyz, sample_intensity = read_binary_pcd_xyz_intensity(frames[0].cloud_path)
    cameras = load_camera_calibrations(args.sequence_dir)
    fallback_colors = colorize_points(sample_xyz, sample_intensity, args.color_mode)
    point_colors, camera_color_mask, camera_counts = colorize_from_registered_cameras(
        sample_xyz,
        frames[0],
        cameras,
        fallback_colors,
    )
    sample_result = make_range_image(
        sample_xyz,
        sample_intensity,
        image_height=args.lidar_height,
        image_width=args.lidar_width,
        range_min=args.range_min,
        range_max=args.range_max,
        elevation_min_rad=bounds.elevation_min_rad,
        elevation_max_rad=bounds.elevation_max_rad,
        color_mode=args.color_mode,
        point_colors=point_colors,
        camera_color_mask=camera_color_mask,
    )
    print(
        json.dumps(
            {
                "sequence_dir": str(args.sequence_dir),
                "frames": len(frames),
                "first_cloud": frames[0].cloud_path.name,
                "last_cloud": frames[-1].cloud_path.name,
                "range_m": [args.range_min, args.range_max],
                "elevation_range_deg": [
                    math.degrees(bounds.elevation_min_rad),
                    math.degrees(bounds.elevation_max_rad),
                ],
                "sample_projected_pixels": sample_result.projected_pixels,
                "sample_camera_rgb_pixels": sample_result.camera_rgb_pixels,
                "sample_camera_rgb_samples_by_camera": camera_counts,
                "sample_nonzero_range_pixels": int(np.count_nonzero(sample_result.range_image)),
                "enable_features": args.enable_features,
                "feature_model": args.feature_model if args.enable_features else None,
                "text_adaptor": TEXT_ADAPTOR_NAME if args.enable_features else None,
                "feature_grid_registered_to": (
                    "lidar range image" if args.enable_features else None
                ),
                "grid_center_xyz": center.astype(float).tolist(),
                "extent_meters_xyz": extent_xyz.astype(float).tolist(),
                "grid_shape_nzyx": list(grid_shape_nzyx(extent_xyz, args.voxel_size)),
                "voxel_size": args.voxel_size,
                "esdf_origin_xyz": list(ESDF_ORIGIN_XYZ),
                "esdf_extent_meters_xyz": [args.esdf_extent_meters] * 3,
                "esdf_voxel_size": args.esdf_voxel_size,
                "esdf_every": args.esdf_every,
                "esdf_slice_resolution": args.esdf_slice_resolution,
                "block_size": args.block_size,
                "output": str(args.output),
            },
            indent=2,
        )
    )


def main() -> None:
    args = parse_args()
    frames = select_frames(args)
    bounds = infer_bounds(
        frames,
        range_min=args.range_min,
        range_max=args.range_max,
        padding_m=args.padding_m,
        elevation_min_deg=args.elevation_min_deg,
        elevation_max_deg=args.elevation_max_deg,
    )
    if args.dry_run:
        print_dry_run(args, frames, bounds)
        return
    run_mapper(args, frames, bounds)


if __name__ == "__main__":
    main()
