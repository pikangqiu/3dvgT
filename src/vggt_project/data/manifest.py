"""Manifest loader for real-data experiments.

The manifest is a JSONL bridge between nuScenes preprocessing and training.
Preprocessing can create one line per sample without forcing the model code to
know every detail of the nuScenes SDK.
"""

from __future__ import annotations

import json
from pathlib import Path

from vggt_project.data.sample import AlignedNuScenesSample, CameraFrame


def _resolve(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def load_manifest(path: Path) -> list[AlignedNuScenesSample]:
    """Load a JSONL manifest into typed sample contracts."""

    base = path.parent
    samples: list[AlignedNuScenesSample] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        camera_paths = record.get("camera_paths", [])
        cameras = tuple(
            CameraFrame(
                image_path=_resolve(base, camera_path),
                camera_name=record.get("camera_names", [])[index]
                if index < len(record.get("camera_names", []))
                else f"camera_{index}",
                intrinsics_frame=record.get("intrinsics_frame", "camera"),
                extrinsics_source_frame=record.get("extrinsics_source_frame", "camera"),
                extrinsics_target_frame=record.get("extrinsics_target_frame", "ego"),
            )
            for index, camera_path in enumerate(camera_paths)
        )
        if not cameras:
            raise ValueError(f"manifest line {line_number} has no camera_paths")

        satellite_patch_path = _resolve(base, record["satellite_patch_path"])
        assert satellite_patch_path is not None

        samples.append(
            AlignedNuScenesSample(
                token=record["token"],
                scene_token=record["scene_token"],
                timestamp_us=int(record["timestamp_us"]),
                cameras=cameras,
                ego_pose_frame=record.get("ego_pose_frame", "ego"),
                bev_frame=record.get("bev_frame", "bev"),
                gravity_frame=record.get("gravity_frame", "gravity"),
                satellite_patch_path=satellite_patch_path,
                satellite_frame=record.get("satellite_frame", "satellite"),
                valid_area_mask_path=_resolve(base, record.get("valid_area_mask_path")),
                lidar_depth_path=_resolve(base, record.get("lidar_depth_path")),
                pointmap_path=_resolve(base, record.get("pointmap_path")),
                vector_map_path=_resolve(base, record.get("vector_map_path")),
                ego_translation=_tuple_or_none(record.get("ego_translation"), 3),
                ego_rotation=_tuple_or_none(record.get("ego_rotation"), 4),
                map_location=record.get("map_location"),
            )
        )
    return samples


def _tuple_or_none(value: list | tuple | None, length: int) -> tuple[float, ...] | None:
    if value is None:
        return None
    if len(value) != length:
        raise ValueError(f"expected sequence of length {length}, got {len(value)}")
    return tuple(float(item) for item in value)
