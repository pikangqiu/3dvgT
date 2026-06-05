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

        yield {
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


def write_manifest(records: Iterable[dict], output_path: Path) -> None:
    """Write records as JSONL."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")

