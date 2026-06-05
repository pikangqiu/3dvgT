"""Create camera-depth supervision from nuScenes LiDAR samples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LidarDepthReport:
    manifest_path: Path
    output_manifest_path: Path | None
    camera_name: str
    sample_count: int
    depth_maps_written: int
    camera_names: tuple[str, ...] = ()


def materialize_lidar_depth_manifest(
    nusc: Any,
    manifest_path: Path,
    *,
    camera_name: str = "CAM_FRONT",
    camera_names: tuple[str, ...] | list[str] | None = None,
    depth_dir: Path = Path("lidar_depth"),
    output_manifest_path: Path | None = None,
    max_depth_m: float = 80.0,
    overwrite: bool = False,
) -> LidarDepthReport:
    """Project nuScenes LIDAR_TOP points into one or more cameras and update records."""

    base = manifest_path.parent
    records = _read_jsonl_records(manifest_path)
    written = 0
    selected_camera_names = _selected_camera_names(camera_name, camera_names)

    for record in records:
        sample_token = str(record["token"])
        depth_paths: dict[str, str] = {}
        for selected_camera_name in selected_camera_names:
            relative_depth_path = depth_dir / f"{sample_token}_{selected_camera_name}.png"
            depth_paths[selected_camera_name] = str(relative_depth_path)
            depth_path = _resolve(base, str(relative_depth_path))
            if overwrite or not depth_path.exists():
                image = render_nuscenes_lidar_depth(
                    nusc,
                    sample_token=sample_token,
                    camera_name=selected_camera_name,
                    max_depth_m=max_depth_m,
                )
                depth_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(depth_path)
                written += 1
        record["lidar_depth_paths"] = depth_paths
        record["lidar_depth_path"] = depth_paths[selected_camera_names[0]]

    if output_manifest_path is not None:
        _write_jsonl_records(records, output_manifest_path)

    return LidarDepthReport(
        manifest_path=manifest_path,
        output_manifest_path=output_manifest_path,
        camera_name=",".join(selected_camera_names),
        sample_count=len(records),
        depth_maps_written=written,
        camera_names=selected_camera_names,
    )


def render_nuscenes_lidar_depth(
    nusc: Any,
    *,
    sample_token: str,
    camera_name: str = "CAM_FRONT",
    max_depth_m: float = 80.0,
):
    """Render an 8-bit normalized depth image for a nuScenes sample camera."""

    from PIL import Image

    sample = nusc.get("sample", sample_token)
    if "LIDAR_TOP" not in sample["data"]:
        raise KeyError(f"sample {sample_token} has no LIDAR_TOP data")
    if camera_name not in sample["data"]:
        raise KeyError(f"sample {sample_token} has no {camera_name} data")

    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    camera_sd = nusc.get("sample_data", sample["data"][camera_name])
    camera_cs = nusc.get("calibrated_sensor", camera_sd["calibrated_sensor_token"])
    points_lidar = _load_lidar_points(Path(nusc.dataroot) / lidar_sd["filename"])
    points_camera = transform_lidar_points_to_camera(nusc, points_lidar, lidar_sd, camera_sd)
    width, height = _camera_image_size(nusc, camera_sd)
    depth = rasterize_camera_depth(
        points_camera=points_camera,
        camera_intrinsic=np.asarray(camera_cs["camera_intrinsic"], dtype=np.float32),
        image_width=width,
        image_height=height,
        max_depth_m=max_depth_m,
    )
    return Image.fromarray(depth, mode="L")


def transform_lidar_points_to_camera(
    nusc: Any,
    points_lidar: np.ndarray,
    lidar_sample_data: dict,
    camera_sample_data: dict,
) -> np.ndarray:
    """Transform 3xN LiDAR points into a nuScenes camera coordinate frame."""

    lidar_cs = nusc.get("calibrated_sensor", lidar_sample_data["calibrated_sensor_token"])
    lidar_pose = nusc.get("ego_pose", lidar_sample_data["ego_pose_token"])
    camera_cs = nusc.get("calibrated_sensor", camera_sample_data["calibrated_sensor_token"])
    camera_pose = nusc.get("ego_pose", camera_sample_data["ego_pose_token"])

    points = _rotate(points_lidar, lidar_cs["rotation"]) + _column(lidar_cs["translation"])
    points = _rotate(points, lidar_pose["rotation"]) + _column(lidar_pose["translation"])
    points = _inverse_rotate(points - _column(camera_pose["translation"]), camera_pose["rotation"])
    points = _inverse_rotate(points - _column(camera_cs["translation"]), camera_cs["rotation"])
    return points


def rasterize_camera_depth(
    *,
    points_camera: np.ndarray,
    camera_intrinsic: np.ndarray,
    image_width: int,
    image_height: int,
    max_depth_m: float = 80.0,
) -> np.ndarray:
    """Project 3D camera-frame points into a nearest-depth 8-bit image."""

    if points_camera.shape[0] != 3:
        raise ValueError("points_camera must have shape 3xN")

    z = points_camera[2]
    positive = z > 1e-6
    points = points_camera[:, positive]
    z = z[positive]
    if points.size == 0:
        return np.zeros((image_height, image_width), dtype=np.uint8)

    projected = camera_intrinsic @ points
    xs = np.round(projected[0] / projected[2]).astype(np.int32)
    ys = np.round(projected[1] / projected[2]).astype(np.int32)
    inside = (xs >= 0) & (xs < image_width) & (ys >= 0) & (ys < image_height)
    xs = xs[inside]
    ys = ys[inside]
    z = z[inside]
    if z.size == 0:
        return np.zeros((image_height, image_width), dtype=np.uint8)

    nearest = np.full((image_height, image_width), np.inf, dtype=np.float32)
    for x, y, depth in zip(xs, ys, z, strict=True):
        if depth < nearest[y, x]:
            nearest[y, x] = depth

    normalized = np.zeros((image_height, image_width), dtype=np.uint8)
    valid = np.isfinite(nearest)
    clipped = np.clip(nearest[valid], 0.0, max_depth_m)
    normalized[valid] = np.rint((clipped / max_depth_m) * 255.0).astype(np.uint8)
    return normalized


def _load_lidar_points(path: Path) -> np.ndarray:
    from nuscenes.utils.data_classes import LidarPointCloud

    return LidarPointCloud.from_file(str(path)).points[:3]


def _camera_image_size(nusc: Any, camera_sample_data: dict) -> tuple[int, int]:
    width = camera_sample_data.get("width")
    height = camera_sample_data.get("height")
    if width is not None and height is not None:
        return int(width), int(height)

    from PIL import Image

    image_path = Path(nusc.dataroot) / camera_sample_data["filename"]
    with Image.open(image_path) as image:
        return image.size


def _rotate(points: np.ndarray, rotation: list[float]) -> np.ndarray:
    from pyquaternion import Quaternion

    return Quaternion(rotation).rotation_matrix @ points


def _inverse_rotate(points: np.ndarray, rotation: list[float]) -> np.ndarray:
    from pyquaternion import Quaternion

    return Quaternion(rotation).inverse.rotation_matrix @ points


def _column(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32).reshape(3, 1)


def _read_jsonl_records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl_records(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _selected_camera_names(
    camera_name: str,
    camera_names: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    if camera_names is None:
        camera_names = (camera_name,)
    names: list[str] = []
    for value in camera_names:
        names.extend(part.strip() for part in str(value).split(",") if part.strip())
    if not names:
        raise ValueError("at least one camera name is required")
    return tuple(dict.fromkeys(names))
