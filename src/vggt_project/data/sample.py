"""Typed sample contracts for satellite/BEV-guided reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CameraFrame:
    """One calibrated nuScenes camera image."""

    image_path: Path
    camera_name: str
    intrinsics_frame: str
    extrinsics_source_frame: str
    extrinsics_target_frame: str


@dataclass(frozen=True)
class AlignedNuScenesSample:
    """Inputs expected by the reconstruction-first model.

    Coordinate-frame fields are explicit because this project depends on
    correct transitions between camera, ego, BEV, satellite, and gravity frames.
    """

    token: str
    scene_token: str
    timestamp_us: int
    cameras: tuple[CameraFrame, ...]
    ego_pose_frame: str
    bev_frame: str
    gravity_frame: str
    satellite_patch_path: Path
    satellite_frame: str
    valid_area_mask_path: Path | None = None
    lidar_depth_path: Path | None = None
    lidar_depth_paths: dict[str, Path] | None = None
    occupancy_path: Path | None = None
    pointmap_path: Path | None = None
    pointmap_paths: dict[str, Path] | None = None
    vector_map_path: Path | None = None
    ego_translation: tuple[float, float, float] | None = None
    ego_rotation: tuple[float, float, float, float] | None = None
    camera_local_camera_to_gravity_poses: dict[str, tuple[float, float, float, float]] | None = None
    map_location: str | None = None
