"""Compose nuScenes manifest supervision generation steps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vggt_project.data.nuscenes_depth import materialize_lidar_depth_manifest
from vggt_project.data.nuscenes_pointmap import materialize_lidar_pointmap_manifest


@dataclass(frozen=True)
class LidarSupervisionReport:
    manifest_path: Path
    depth_manifest_path: Path
    output_manifest_path: Path
    sample_count: int
    depth_maps_written: int
    pointmaps_written: int


def materialize_lidar_supervision_manifest(
    nusc: Any,
    manifest_path: Path,
    *,
    output_manifest_path: Path,
    depth_manifest_path: Path | None = None,
    camera_name: str = "CAM_FRONT",
    camera_names: tuple[str, ...] | list[str] | None = None,
    depth_dir: Path = Path("lidar_depth"),
    pointmap_dir: Path = Path("pointmaps"),
    max_depth_m: float = 80.0,
    max_points: int = 4096,
    overwrite: bool = False,
) -> LidarSupervisionReport:
    """Generate depth and pointmap targets, returning a final supervised manifest."""

    depth_manifest = depth_manifest_path or _default_depth_manifest(output_manifest_path)
    depth_report = materialize_lidar_depth_manifest(
        nusc,
        manifest_path,
        camera_name=camera_name,
        camera_names=camera_names,
        depth_dir=depth_dir,
        output_manifest_path=depth_manifest,
        max_depth_m=max_depth_m,
        overwrite=overwrite,
    )
    pointmap_report = materialize_lidar_pointmap_manifest(
        nusc,
        depth_manifest,
        pointmap_dir=pointmap_dir,
        output_manifest_path=output_manifest_path,
        max_points=max_points,
        overwrite=overwrite,
    )
    return LidarSupervisionReport(
        manifest_path=manifest_path,
        depth_manifest_path=depth_manifest,
        output_manifest_path=output_manifest_path,
        sample_count=pointmap_report.sample_count,
        depth_maps_written=depth_report.depth_maps_written,
        pointmaps_written=pointmap_report.pointmaps_written,
    )


def _default_depth_manifest(output_manifest_path: Path) -> Path:
    return output_manifest_path.with_name(f"{output_manifest_path.stem}.depth.jsonl")
