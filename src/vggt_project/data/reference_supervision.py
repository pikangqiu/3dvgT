"""Materialize dense G3T/VGGT reference predictions into manifest targets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ReferenceSupervisionReport:
    manifest_path: Path
    output_manifest_path: Path | None
    sample_count: int
    depth_maps_written: int
    pointmaps_written: int
    pose_targets_written: int


def materialize_reference_prediction_manifest(
    manifest_path: Path,
    *,
    prediction_fn: Callable[[dict, int], dict],
    target_dir: Path = Path("reference_targets"),
    output_manifest_path: Path | None = None,
    max_points: int = 4096,
    overwrite: bool = False,
) -> ReferenceSupervisionReport:
    """Write dense reference depth, pointmap, and pose predictions as manifest targets."""

    import numpy as np

    if max_points <= 0:
        raise ValueError("max_points must be positive")

    base = manifest_path.parent
    records = _read_jsonl_records(manifest_path)
    depth_written = 0
    pointmaps_written = 0
    pose_targets_written = 0

    for index, record in enumerate(records):
        token = str(record["token"])
        camera_names = _camera_names(record)
        prediction = prediction_fn(record, index)
        depths = _camera_depths_from_prediction(prediction, expected_cameras=len(camera_names))
        pointmaps = _camera_pointmaps_from_prediction(prediction, expected_cameras=len(camera_names))
        poses = _camera_poses_from_prediction(prediction, expected_cameras=len(camera_names))

        if depths is not None:
            depth_paths: dict[str, str] = {}
            for camera_index, camera_name in enumerate(camera_names):
                relative_path = target_dir / "depth" / f"{token}_{camera_name}.npy"
                absolute_path = _resolve(base, relative_path)
                depth_paths[camera_name] = str(relative_path)
                if overwrite or not absolute_path.exists():
                    absolute_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(absolute_path, depths[camera_index].astype(np.float32, copy=False))
                    depth_written += 1
            record["lidar_depth_paths"] = depth_paths
            record["lidar_depth_path"] = depth_paths[camera_names[0]]

        if pointmaps is not None:
            pointmap_paths: dict[str, str] = {}
            for camera_index, camera_name in enumerate(camera_names):
                relative_path = target_dir / "pointmaps" / f"{token}_{camera_name}.npy"
                absolute_path = _resolve(base, relative_path)
                pointmap_paths[camera_name] = str(relative_path)
                if overwrite or not absolute_path.exists():
                    absolute_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(absolute_path, _truncate_pointmap(pointmaps[camera_index], max_points=max_points))
                    pointmaps_written += 1
            record["pointmap_paths"] = pointmap_paths

        if poses is not None:
            pose_mapping = {
                camera_name: _normalize_quaternion_list(poses[camera_index])
                for camera_index, camera_name in enumerate(camera_names)
            }
            record["camera_local_camera_to_gravity_poses"] = pose_mapping
            pose_targets_written += len(pose_mapping)

    if output_manifest_path is not None:
        _write_jsonl_records(records, output_manifest_path)

    return ReferenceSupervisionReport(
        manifest_path=manifest_path,
        output_manifest_path=output_manifest_path,
        sample_count=len(records),
        depth_maps_written=depth_written,
        pointmaps_written=pointmaps_written,
        pose_targets_written=pose_targets_written,
    )


def _camera_names(record: dict) -> tuple[str, ...]:
    names = record.get("camera_names")
    if names:
        return tuple(str(name) for name in names)
    return tuple(f"camera_{index}" for index, _ in enumerate(record.get("camera_paths", ())))


def _camera_depths_from_prediction(prediction: dict, *, expected_cameras: int):
    depth = prediction.get("camera_depths")
    if depth is None:
        depth = prediction.get("depth")
    if depth is None:
        return None
    array = _to_numpy(depth)
    if array.ndim == 5 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array.transpose(0, 3, 1, 2)
    elif array.ndim == 3:
        array = array[:, None, :, :]
    if array.ndim != 4 or array.shape[1] != 1:
        raise ValueError(f"reference depth must have shape Sx1xHxW or SxHxWx1, got {array.shape}")
    return _require_camera_count(array, expected_cameras, "reference depth")


def _camera_pointmaps_from_prediction(prediction: dict, *, expected_cameras: int):
    pointmaps = prediction.get("camera_pointmaps")
    if pointmaps is None:
        pointmaps = prediction.get("world_points")
    if pointmaps is None:
        return None
    array = _to_numpy(pointmaps)
    if array.ndim == 5 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 4 and array.shape[-1] == 3:
        array = array.reshape(array.shape[0], -1, 3)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"reference pointmaps must have shape SxNx3 or SxHxWx3, got {array.shape}")
    return _require_camera_count(array, expected_cameras, "reference pointmaps")


def _camera_poses_from_prediction(prediction: dict, *, expected_cameras: int):
    poses = prediction.get("camera_local_camera_to_gravity_poses")
    if poses is not None:
        array = _to_numpy(poses)
    elif prediction.get("local_pose_enc") is not None:
        array = _to_numpy(prediction["local_pose_enc"])[..., :4]
    elif prediction.get("pose_enc") is not None:
        pose_enc = _to_numpy(prediction["pose_enc"])
        array = pose_enc[..., 3:7] if pose_enc.shape[-1] >= 7 else pose_enc[..., :4]
    else:
        return None
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[-1] != 4:
        raise ValueError(f"reference poses must have shape Sx4, got {array.shape}")
    return _require_camera_count(array, expected_cameras, "reference poses")


def _require_camera_count(array, expected_cameras: int, label: str):
    if array.shape[0] < expected_cameras:
        raise ValueError(f"{label} has {array.shape[0]} cameras, expected {expected_cameras}")
    return array[:expected_cameras]


def _truncate_pointmap(pointmap, *, max_points: int):
    import numpy as np

    array = np.asarray(pointmap, dtype=np.float32)
    if array.shape[0] >= max_points:
        return array[:max_points]
    return array


def _normalize_quaternion_list(quaternion) -> list[float]:
    import numpy as np

    array = np.asarray(quaternion, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-8:
        return [1.0, 0.0, 0.0, 0.0]
    return [float(value) for value in (array / norm)]


def _to_numpy(value):
    import numpy as np

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl_records(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def _resolve(base: Path, path: Path) -> Path:
    return path if path.is_absolute() else base / path
