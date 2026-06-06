"""Attach public Occ3D/OpenOccupancy occupancy labels to project manifests."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any


@dataclass(frozen=True)
class Occ3DLabelAttachReport:
    manifest_path: Path
    output_manifest_path: Path
    occ3d_root: Path
    sample_count: int
    labels_attached: int
    missing_labels: tuple[str, ...]


def attach_occ3d_label_manifest(
    manifest_path: Path,
    *,
    occ3d_root: Path = Path("data/occ3d"),
    output_manifest_path: Path,
    target_field: str = "occupancy_path",
    split: str = "trainval",
    scene_name_resolver: Callable[[str], str] | None = None,
) -> Occ3DLabelAttachReport:
    """Write a manifest with Occ3D label paths attached as occupancy targets."""

    records = _read_jsonl_records(manifest_path)
    output_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    missing: list[str] = []
    attached = 0

    for index, record in enumerate(records):
        sample_token = _sample_token(record, index)
        scene_name = _scene_name(record, scene_name_resolver=scene_name_resolver)
        label_path = _find_label_path(
            occ3d_root=occ3d_root,
            split=split,
            scene_name=scene_name,
            sample_token=sample_token,
        )
        if label_path is None:
            missing.append(f"{scene_name}/{sample_token}/labels.npz")
            continue
        record["scene_name"] = scene_name
        record[target_field] = _relative_to_output(label_path, output_manifest_path.parent)
        attached += 1

    if missing:
        raise FileNotFoundError(
            "missing Occ3D labels: "
            + ", ".join(missing[:5])
            + (f" ... and {len(missing) - 5} more" if len(missing) > 5 else "")
        )

    _write_jsonl_records(records, output_manifest_path)
    return Occ3DLabelAttachReport(
        manifest_path=manifest_path,
        output_manifest_path=output_manifest_path,
        occ3d_root=occ3d_root,
        sample_count=len(records),
        labels_attached=attached,
        missing_labels=tuple(missing),
    )


def _sample_token(record: dict[str, Any], index: int) -> str:
    token = record.get("token")
    if not token:
        raise ValueError(f"record {index} missing required field: token")
    return str(token)


def _scene_name(
    record: dict[str, Any],
    *,
    scene_name_resolver: Callable[[str], str] | None,
) -> str:
    scene_name = record.get("scene_name")
    if scene_name:
        return str(scene_name)
    scene_token = record.get("scene_token")
    if scene_token and scene_name_resolver is not None:
        return scene_name_resolver(str(scene_token))
    raise ValueError("record missing scene_name; provide scene_name_resolver for scene_token manifests")


def _find_label_path(
    *,
    occ3d_root: Path,
    split: str,
    scene_name: str,
    sample_token: str,
) -> Path | None:
    candidates = [
        occ3d_root / "occ3d-nuscenes" / split / "gts" / scene_name / sample_token / "labels.npz",
        occ3d_root / "occ3d-nuscenes" / "gts" / scene_name / sample_token / "labels.npz",
        occ3d_root / split / "gts" / scene_name / sample_token / "labels.npz",
        occ3d_root / "gts" / scene_name / sample_token / "labels.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _relative_to_output(path: Path, output_base: Path) -> str:
    if path.is_absolute():
        return os.path.relpath(path, output_base)
    return os.path.relpath(path.resolve(), output_base.resolve())


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl_records(records: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")
