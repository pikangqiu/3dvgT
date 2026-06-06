"""Create BEV occupancy supervision from nuScenes LiDAR samples."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vggt_project.data.nuscenes_depth import _load_lidar_points, _resolve
from vggt_project.data.nuscenes_pointmap import lidar_points_to_ego_pointmap


@dataclass(frozen=True)
class LidarOccupancyReport:
    manifest_path: Path
    output_manifest_path: Path | None
    sample_count: int
    occupancy_maps_written: int


def materialize_lidar_occupancy_manifest(
    nusc: Any,
    manifest_path: Path,
    *,
    occupancy_dir: Path = Path("occupancy"),
    output_manifest_path: Path | None = None,
    x_range: tuple[float, float] = (-50.0, 50.0),
    y_range: tuple[float, float] = (-50.0, 50.0),
    z_range: tuple[float, float] = (-5.0, 5.0),
    grid_size: tuple[int, int] = (200, 200),
    overwrite: bool = False,
) -> LidarOccupancyReport:
    """Create per-sample BEV occupancy targets and update manifest records."""

    base = manifest_path.parent
    records = _read_jsonl_records(manifest_path)
    written = 0

    for record in records:
        sample_token = str(record["token"])
        relative_occupancy_path = occupancy_dir / f"{sample_token}_LIDAR_TOP.npy"
        record["occupancy_path"] = str(relative_occupancy_path)
        occupancy_path = _resolve(base, str(relative_occupancy_path))
        if overwrite or not occupancy_path.exists():
            occupancy = render_nuscenes_lidar_occupancy(
                nusc,
                sample_token=sample_token,
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                grid_size=grid_size,
            )
            occupancy_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(occupancy_path, occupancy)
            written += 1

    if output_manifest_path is not None:
        _write_jsonl_records(records, output_manifest_path)

    return LidarOccupancyReport(
        manifest_path=manifest_path,
        output_manifest_path=output_manifest_path,
        sample_count=len(records),
        occupancy_maps_written=written,
    )


def render_nuscenes_lidar_occupancy(
    nusc: Any,
    *,
    sample_token: str,
    x_range: tuple[float, float] = (-50.0, 50.0),
    y_range: tuple[float, float] = (-50.0, 50.0),
    z_range: tuple[float, float] = (-5.0, 5.0),
    grid_size: tuple[int, int] = (200, 200),
) -> np.ndarray:
    """Return an ego-frame BEV occupancy grid from one sample's LIDAR_TOP sweep."""

    sample = nusc.get("sample", sample_token)
    if "LIDAR_TOP" not in sample["data"]:
        raise KeyError(f"sample {sample_token} has no LIDAR_TOP data")

    lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    lidar_cs = nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
    points_lidar = _load_lidar_points(Path(nusc.dataroot) / lidar_sd["filename"])
    points_ego = lidar_points_to_ego_pointmap(points_lidar, lidar_cs)
    return lidar_points_to_bev_occupancy(
        points_ego,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        grid_size=grid_size,
    )


def lidar_points_to_bev_occupancy(
    points_ego: np.ndarray,
    *,
    x_range: tuple[float, float] = (-50.0, 50.0),
    y_range: tuple[float, float] = (-50.0, 50.0),
    z_range: tuple[float, float] = (-5.0, 5.0),
    grid_size: tuple[int, int] = (200, 200),
) -> np.ndarray:
    """Rasterize ego-frame Nx3 LiDAR points into a binary BEV occupancy grid."""

    points = np.asarray(points_ego, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_ego must have shape Nx3, got {points.shape}")
    height, width = _validate_grid_size(grid_size)
    x_min, x_max = _validate_range("x_range", x_range)
    y_min, y_max = _validate_range("y_range", y_range)
    z_min, z_max = _validate_range("z_range", z_range)

    inside = (
        (points[:, 0] >= x_min)
        & (points[:, 0] < x_max)
        & (points[:, 1] >= y_min)
        & (points[:, 1] < y_max)
        & (points[:, 2] >= z_min)
        & (points[:, 2] < z_max)
    )
    occupancy = np.zeros((height, width), dtype=np.float32)
    if not inside.any():
        return occupancy

    selected = points[inside]
    x_bins = np.floor((selected[:, 0] - x_min) / (x_max - x_min) * width).astype(np.int32)
    y_bins = np.floor((selected[:, 1] - y_min) / (y_max - y_min) * height).astype(np.int32)
    x_bins = np.clip(x_bins, 0, width - 1)
    y_bins = np.clip(y_bins, 0, height - 1)
    occupancy[y_bins, x_bins] = 1.0
    return occupancy


def _validate_range(name: str, value: tuple[float, float]) -> tuple[float, float]:
    lower, upper = float(value[0]), float(value[1])
    if lower >= upper:
        raise ValueError(f"{name} lower bound must be less than upper bound")
    return lower, upper


def _validate_grid_size(value: tuple[int, int]) -> tuple[int, int]:
    height, width = int(value[0]), int(value[1])
    if height <= 0 or width <= 0:
        raise ValueError("grid_size values must be positive")
    return height, width


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
