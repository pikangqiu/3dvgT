"""Validation helpers for generated real-data manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vggt_project.data.manifest import load_manifest


@dataclass(frozen=True)
class MissingManifestPath:
    sample_token: str
    field: str
    path: Path


@dataclass(frozen=True)
class ManifestValidationReport:
    manifest_path: Path
    sample_count: int
    missing_paths: tuple[MissingManifestPath, ...]

    @property
    def ready(self) -> bool:
        return self.sample_count > 0 and not self.missing_paths


def validate_manifest_paths(path: Path) -> ManifestValidationReport:
    """Validate that every file referenced by a manifest exists."""

    samples = load_manifest(path)
    missing: list[MissingManifestPath] = []
    for sample in samples:
        for camera in sample.cameras:
            _append_missing(missing, sample.token, "camera.image_path", camera.image_path)
        _append_missing(missing, sample.token, "satellite_patch_path", sample.satellite_patch_path)
        _append_missing(missing, sample.token, "valid_area_mask_path", sample.valid_area_mask_path)
        _append_missing(missing, sample.token, "lidar_depth_path", sample.lidar_depth_path)
        if sample.lidar_depth_paths is not None:
            for camera_name, depth_path in sample.lidar_depth_paths.items():
                _append_missing(missing, sample.token, f"lidar_depth_paths.{camera_name}", depth_path)
        _append_missing(missing, sample.token, "occupancy_path", sample.occupancy_path)
        _append_missing(missing, sample.token, "pointmap_path", sample.pointmap_path)
        if sample.pointmap_paths is not None:
            for camera_name, pointmap_path in sample.pointmap_paths.items():
                _append_missing(missing, sample.token, f"pointmap_paths.{camera_name}", pointmap_path)
        _append_missing(missing, sample.token, "vector_map_path", sample.vector_map_path)

    return ManifestValidationReport(
        manifest_path=path,
        sample_count=len(samples),
        missing_paths=tuple(missing),
    )


def _append_missing(
    missing: list[MissingManifestPath],
    sample_token: str,
    field: str,
    path: Path | None,
) -> None:
    if path is not None and not path.exists():
        missing.append(MissingManifestPath(sample_token=sample_token, field=field, path=path))
