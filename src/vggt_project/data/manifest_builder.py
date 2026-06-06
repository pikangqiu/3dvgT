"""Build JSONL manifest records from nuScenes-style samples."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


CAMERA_ORDER = (
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
    "CAM_FRONT",
    "CAM_FRONT_LEFT",
    "CAM_FRONT_RIGHT",
)


def build_manifest_records(nusc: Any, satellite_patch_dir: Path) -> Iterator[dict]:
    """Yield project manifest records from a nuScenes SDK object."""

    for sample in nusc.sample:
        camera_names = sorted(
            name for name in sample["data"] if name.startswith("CAM_")
        )
        camera_paths = []
        for camera_name in camera_names:
            sample_data = nusc.get("sample_data", sample["data"][camera_name])
            camera_paths.append(sample_data["filename"])
        ego_pose = _sample_ego_pose(nusc, sample)
        map_location = _sample_map_location(nusc, sample)

        record = {
            "token": sample["token"],
            "scene_token": sample["scene_token"],
            "timestamp_us": int(sample["timestamp"]),
            "camera_names": camera_names,
            "camera_paths": camera_paths,
            "satellite_patch_path": str(satellite_patch_dir / f"{sample['token']}.png"),
            "ego_pose_frame": "ego",
            "bev_frame": "bev",
            "gravity_frame": "gravity",
            "satellite_frame": "satellite",
        }
        if ego_pose is not None:
            record["ego_translation"] = [float(value) for value in ego_pose["translation"]]
            record["ego_rotation"] = [float(value) for value in ego_pose["rotation"]]
        camera_poses = _camera_ego_pose_rotations(nusc, sample, camera_names)
        if camera_poses:
            record["camera_local_camera_to_gravity_poses"] = camera_poses
        if map_location is not None:
            record["map_location"] = map_location
        yield record


def write_manifest(records: Iterable[dict], output_path: Path) -> None:
    """Write records as JSONL."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def _sample_ego_pose(nusc: Any, sample: dict) -> dict | None:
    sample_data_token = sample["data"].get("LIDAR_TOP")
    if sample_data_token is None:
        camera_tokens = [token for name, token in sample["data"].items() if name.startswith("CAM_")]
        sample_data_token = camera_tokens[0] if camera_tokens else None
    if sample_data_token is None:
        return None
    sample_data = nusc.get("sample_data", sample_data_token)
    ego_pose_token = sample_data.get("ego_pose_token")
    if ego_pose_token is None:
        return None
    return nusc.get("ego_pose", ego_pose_token)


def _camera_ego_pose_rotations(nusc: Any, sample: dict, camera_names: list[str]) -> dict[str, list[float]]:
    rotations: dict[str, list[float]] = {}
    for camera_name in camera_names:
        sample_data_token = sample["data"].get(camera_name)
        if sample_data_token is None:
            continue
        try:
            sample_data = nusc.get("sample_data", sample_data_token)
            ego_pose = nusc.get("ego_pose", sample_data["ego_pose_token"])
        except (KeyError, TypeError):
            continue
        rotation = ego_pose.get("rotation")
        if rotation is None:
            continue
        rotations[camera_name] = [float(value) for value in rotation]
    return rotations


def _sample_map_location(nusc: Any, sample: dict) -> str | None:
    scene_token = sample.get("scene_token")
    if scene_token is None:
        return None
    try:
        scene = nusc.get("scene", scene_token)
        log = nusc.get("log", scene["log_token"])
    except (KeyError, TypeError):
        return None
    return log.get("location")
