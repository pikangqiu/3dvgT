"""Create camera pose supervision fields from nuScenes calibration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class CameraPoseReport:
    manifest_path: Path
    output_manifest_path: Path | None
    sample_count: int
    pose_targets_written: int
    camera_names: tuple[str, ...]


def materialize_camera_pose_manifest(
    nusc: Any,
    manifest_path: Path,
    *,
    output_manifest_path: Path | None = None,
    camera_name: str = "CAM_FRONT",
    camera_names: tuple[str, ...] | list[str] | None = None,
) -> CameraPoseReport:
    """Update manifest records with per-camera local pose quaternion targets.

    The current scaffold uses nuScenes calibrated sensor rotations as explicit
    camera-local supervision. Full G3T pose targets can later replace this field
    without changing the train/eval batch contract.
    """

    records = _read_jsonl_records(manifest_path)
    selected_camera_names = _camera_names_for_records(records, camera_name, camera_names)
    pose_targets_written = 0

    for record in records:
        sample = nusc.get("sample", str(record["token"]))
        poses: dict[str, list[float]] = {}
        for selected_camera_name in selected_camera_names:
            if selected_camera_name not in sample["data"]:
                continue
            sample_data = nusc.get("sample_data", sample["data"][selected_camera_name])
            calibrated_sensor = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
            poses[selected_camera_name] = [float(value) for value in calibrated_sensor["rotation"]]
        record["camera_local_camera_to_gravity_poses"] = poses
        pose_targets_written += len(poses)

    if output_manifest_path is not None:
        _write_jsonl_records(records, output_manifest_path)

    return CameraPoseReport(
        manifest_path=manifest_path,
        output_manifest_path=output_manifest_path,
        sample_count=len(records),
        pose_targets_written=pose_targets_written,
        camera_names=selected_camera_names,
    )


def _camera_names_for_records(
    records: list[dict],
    camera_name: str,
    camera_names: tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    if camera_names is not None:
        return _selected_camera_names(camera_name, camera_names)
    names: list[str] = []
    for record in records:
        names.extend(str(name) for name in record.get("camera_names", []))
    if names:
        return tuple(dict.fromkeys(names))
    return _selected_camera_names(camera_name, None)


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
