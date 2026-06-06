"""Validate public occupancy benchmark manifest alignment before evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PublicOccupancyManifestReport:
    manifest_path: Path
    target_field: str
    token_field: str
    scene_field: str
    expected_split: str | None
    ready: bool
    sample_count: int
    scene_count: int
    label_root_matches: int
    sample_tokens: tuple[str, ...]
    scene_names: tuple[str, ...]
    errors: tuple[str, ...]

    def to_json(self) -> str:
        payload = asdict(self)
        payload["manifest_path"] = str(self.manifest_path)
        return json.dumps(payload, indent=2, sort_keys=True)


def validate_public_occupancy_manifest(
    manifest_path: Path,
    *,
    target_field: str = "occupancy_path",
    token_field: str = "token",
    scene_field: str = "scene_name",
    expected_split: str | None = None,
) -> PublicOccupancyManifestReport:
    """Validate that a manifest is aligned with public Occ3D/OpenOccupancy labels."""

    errors: list[str] = []
    records = _read_jsonl(manifest_path, errors=errors)
    if not records and not errors:
        errors.append(f"manifest is empty: {manifest_path}")

    sample_tokens: list[str] = []
    scene_names: list[str] = []
    seen_tokens: set[str] = set()
    label_root_matches = 0

    for index, record in enumerate(records):
        token = _required_str(record, token_field, index, errors)
        scene_name = _required_str(record, scene_field, index, errors)
        target_value = _required_str(record, target_field, index, errors)
        if token:
            if token in seen_tokens:
                errors.append(f"record {index} duplicates sample token: {token}")
            seen_tokens.add(token)
            sample_tokens.append(token)
        if scene_name:
            scene_names.append(scene_name)
        if not target_value:
            continue

        target_path = Path(target_value)
        resolved = target_path if target_path.is_absolute() else manifest_path.parent / target_path
        if not resolved.exists():
            errors.append(f"record {index} {target_field} does not exist: {target_value}")
        if resolved.name != "labels.npz":
            errors.append(f"record {index} {target_field} must point to labels.npz: {target_value}")

        parts = resolved.parts
        has_public_root = "occ3d-nuscenes" in parts and "gts" in parts
        split_matches = expected_split is None or expected_split in parts
        if not has_public_root:
            errors.append(f"record {index} {target_field} is not under occ3d-nuscenes/gts: {target_value}")
        if expected_split is not None and not split_matches:
            errors.append(f"record {index} {target_field} is not under expected public split: {expected_split}")
        if scene_name and scene_name not in parts:
            errors.append(f"record {index} {target_field} path does not contain scene_name: {scene_name}")
        if token and token not in parts:
            errors.append(f"record {index} {target_field} path does not contain sample token: {token}")
        if has_public_root and split_matches:
            label_root_matches += 1

    return PublicOccupancyManifestReport(
        manifest_path=manifest_path,
        target_field=target_field,
        token_field=token_field,
        scene_field=scene_field,
        expected_split=expected_split,
        ready=not errors,
        sample_count=len(records),
        scene_count=len(set(scene_names)),
        label_root_matches=label_root_matches,
        sample_tokens=tuple(sample_tokens),
        scene_names=tuple(sorted(set(scene_names))),
        errors=tuple(errors),
    )


def format_public_occupancy_manifest_report(report: PublicOccupancyManifestReport) -> str:
    """Render a compact public occupancy manifest validation report."""

    lines = [
        f"public_occupancy_manifest_ready: {str(report.ready).lower()}",
        f"manifest: {report.manifest_path}",
        f"target_field: {report.target_field}",
        f"expected_split: {report.expected_split or '<none>'}",
        f"samples: {report.sample_count}",
        f"scenes: {report.scene_count}",
        f"label_root_matches: {report.label_root_matches}",
        "scene_names:",
    ]
    lines.extend(f"- {scene_name}" for scene_name in report.scene_names) if report.scene_names else lines.append("- none")
    if report.errors:
        lines.append("errors:")
        lines.extend(f"- {error}" for error in report.errors)
    return "\n".join(lines)


def _read_jsonl(path: Path, *, errors: list[str]) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        errors.append(f"manifest does not exist: {path}")
        return []
    records: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"record {index} is not valid JSON: {error}")
            continue
        if not isinstance(record, dict):
            errors.append(f"record {index} must be a JSON object")
            continue
        records.append(record)
    return records


def _required_str(record: dict[str, Any], field: str, index: int, errors: list[str]) -> str | None:
    value = record.get(field)
    if value is None or value == "":
        errors.append(f"record {index} missing required field: {field}")
        return None
    return str(value)
