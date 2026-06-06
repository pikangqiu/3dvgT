"""Create pointmap supervision from nuScenes LiDAR samples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vggt_project.data.nuscenes_depth import (
    _camera_image_size,
    _column,
    _load_lidar_points,
    _resolve,
    _rotate,
    _selected_camera_names,
    transform_lidar_points_to_camera,
)


@dataclass(frozen=True)
class LidarPointmapReport:
    manifest_path: Path
    output_manifest_path: Path | None
    sample_count: int
    pointmaps_written: int
    camera_names: tuple[str, ...] = ()


def materialize_lidar_pointmap_manifest(
    nusc: Any,
    manifest_path: Path,
    *,
    pointmap_dir: Path = Path("pointmaps"),
    output_manifest_path: Path | None = None,
    max_points: int = 4096,
    overwrite: bool = False,
) -> LidarPointmapReport:
    """Create per-sample ego-frame pointmap targets and update manifest records."""

    base = manifest_path.parent
    records = _read_jsonl_records(manifest_path)
    written = 0

    for record in records:
        sample_token = str(record["token"])
        relative_pointmap_path = pointmap_dir / f"{sample_token}_LIDAR_TOP.npy"
        record["pointmap_path"] = str(relative_pointmap_path)
        pointmap_path = _resolve(base, str(relative_pointmap_path))
        if overwrite or not pointmap_path.exists():
            pointmap = render_nuscenes_lidar_pointmap(
                nusc,
                sample_token=sample_token,
                max_points=max_points,
            )
            pointmap_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(pointmap_path, pointmap)
            written += 1

    if output_manifest_path is not None:
        _write_jsonl_records(records, output_manifest_path)

    return LidarPointmapReport(
        manifest_path=manifest_path,
        output_manifest_path=output_manifest_path,
        sample_count=len(records),
        pointmaps_written=written,
    )


def materialize_camera_lidar_pointmap_manifest(
    nusc: Any,
    manifest_path: Path,
    *,
    camera_name: str = "CAM_FRONT",
    camera_names: tuple[str, ...] | list[str] | None = None,
    pointmap_dir: Path = Path("camera_pointmaps"),
    output_manifest_path: Path | None = None,
    max_points: int = 4096,
    overwrite: bool = False,
) -> LidarPointmapReport:
    """Create per-camera camera-frame pointmap targets and update manifest records."""

    base = manifest_path.parent
    records = _read_jsonl_records(manifest_path)
    written = 0
    selected_camera_names = _selected_camera_names(camera_name, camera_names)

    for record in records:
        sample_token = str(record["token"])
        pointmap_paths: dict[str, str] = {}
        for selected_camera_name in selected_camera_names:
            relative_pointmap_path = pointmap_dir / f"{sample_token}_{selected_camera_name}.npy"
            pointmap_paths[selected_camera_name] = str(relative_pointmap_path)
            pointmap_path = _resolve(base, str(relative_pointmap_path))
            if overwrite or not pointmap_path.exists():
                pointmap = render_nuscenes_camera_lidar_pointmap(
                    nusc,
                    sample_token=sample_token,
                    camera_name=selected_camera_name,
                    max_points=max_points,
                )
                pointmap_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(pointmap_path, pointmap)
                written += 1
        record["pointmap_paths"] = pointmap_paths

    if output_manifest_path is not None:
        _write_jsonl_records(records, output_manifest_path)

    return LidarPointmapReport(
        manifest_path=manifest_path,
        output_manifest_path=output_manifest_path,
        sample_count=len(records),
        pointmaps_written=written,
        camera_names=selected_camera_names,
    )


def render_nuscenes_lidar_pointmap(
    nusc: Any,
    *,
    sample_token: str,
    max_points: int = 4096,
) -> np.ndarray:
    """Return an ego-frame Nx3 pointmap from the sample's LIDAR_TOP sweep."""

    sample = nusc.get("sample", sample_token)
    if "LIDAR_TOP" not in sample["data"]:
        raise KeyError(f"sample {sample_token} has no LIDAR_TOP data")

    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    points_lidar = _load_lidar_points(Path(nusc.dataroot) / lidar_sd["filename"])
    pointmap = lidar_points_to_ego_pointmap(points_lidar, lidar_cs)
    return subsample_pointmap(pointmap, max_points=max_points)


def render_nuscenes_camera_lidar_pointmap(
    nusc: Any,
    *,
    sample_token: str,
    camera_name: str = "CAM_FRONT",
    max_points: int = 4096,
) -> np.ndarray:
    """Return a visible camera-frame Nx3 pointmap for one nuScenes sample camera."""

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
    return camera_points_to_visible_pointmap(
        points_camera=points_camera,
        camera_intrinsic=np.asarray(camera_cs["camera_intrinsic"], dtype=np.float32),
        image_width=width,
        image_height=height,
        max_points=max_points,
    )


def lidar_points_to_ego_pointmap(points_lidar: np.ndarray, lidar_calibrated_sensor: dict) -> np.ndarray:
    """Transform 3xN LiDAR-frame points into an ego-frame Nx3 pointmap."""

    if points_lidar.shape[0] != 3:
        raise ValueError("points_lidar must have shape 3xN")
    points_ego = (
        _rotate(points_lidar, lidar_calibrated_sensor["rotation"])
        + _column(lidar_calibrated_sensor["translation"])
    )
    return points_ego.T.astype(np.float32, copy=False)


def camera_points_to_visible_pointmap(
    *,
    points_camera: np.ndarray,
    camera_intrinsic: np.ndarray,
    image_width: int,
    image_height: int,
    max_points: int = 4096,
) -> np.ndarray:
    """Filter camera-frame points to positive-depth points visible in the image."""

    if points_camera.shape[0] != 3:
        raise ValueError("points_camera must have shape 3xN")
    z = points_camera[2]
    positive = z > 1e-6
    points = points_camera[:, positive]
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    projected = camera_intrinsic @ points
    xs = np.round(projected[0] / projected[2]).astype(np.int32)
    ys = np.round(projected[1] / projected[2]).astype(np.int32)
    inside = (xs >= 0) & (xs < image_width) & (ys >= 0) & (ys < image_height)
    return subsample_pointmap(points[:, inside].T, max_points=max_points)


def subsample_pointmap(pointmap: np.ndarray, max_points: int) -> np.ndarray:
    """Deterministically truncate a pointmap for scaffold-sized supervision."""

    if pointmap.ndim != 2 or pointmap.shape[1] != 3:
        raise ValueError(f"pointmap must have shape Nx3, got {pointmap.shape}")
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    return np.asarray(pointmap[:max_points], dtype=np.float32)


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
