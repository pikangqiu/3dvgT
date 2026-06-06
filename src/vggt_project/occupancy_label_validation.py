"""Validate semantic occupancy label manifests before public benchmark evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vggt_project.occupancy_benchmark import _flatten_array, _load_array, _read_jsonl, _required_path


@dataclass(frozen=True)
class OccupancyLabelValidationReport:
    manifest_path: Path
    target_field: str
    num_classes: int
    ignore_index: int | None
    ready: bool
    sample_count: int
    voxel_count: int
    ignored_count: int
    class_histogram: dict[str, int]
    sample_tokens: tuple[str, ...]
    errors: tuple[str, ...]

    def to_json(self) -> str:
        payload = asdict(self)
        payload["manifest_path"] = str(self.manifest_path)
        return json.dumps(payload, indent=2, sort_keys=True)


def validate_occupancy_label_manifest(
    manifest_path: Path,
    *,
    target_field: str = "occupancy_path",
    num_classes: int = 18,
    ignore_index: int | None = None,
) -> OccupancyLabelValidationReport:
    """Validate label class ids and summarize class counts for a benchmark manifest."""

    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    records = _read_jsonl(manifest_path)
    errors: list[str] = []
    histogram = {str(index): 0 for index in range(num_classes)}
    ignored_count = 0
    voxel_count = 0
    sample_tokens: list[str] = []
    base = manifest_path.parent

    for record_index, record in enumerate(records):
        try:
            label_path = _required_path(record, target_field, record_index, base)
            labels = _flatten_array(_load_array(label_path))
        except Exception as error:
            errors.append(f"record {record_index} label load failed: {error}")
            continue

        for value in labels:
            label = int(value)
            if ignore_index is not None and label == ignore_index:
                ignored_count += 1
                continue
            if label < 0 or label >= num_classes:
                errors.append(
                    f"record {record_index} has out-of-range class id {label}; "
                    f"expected 0..{num_classes - 1}"
                )
                continue
            histogram[str(label)] += 1
            voxel_count += 1
        sample_tokens.append(str(record.get("token", record_index)))

    if not records:
        errors.append(f"manifest is empty: {manifest_path}")

    return OccupancyLabelValidationReport(
        manifest_path=manifest_path,
        target_field=target_field,
        num_classes=num_classes,
        ignore_index=ignore_index,
        ready=not errors,
        sample_count=len(records),
        voxel_count=voxel_count,
        ignored_count=ignored_count,
        class_histogram={key: value for key, value in histogram.items() if value > 0},
        sample_tokens=tuple(sample_tokens),
        errors=tuple(errors),
    )


def format_occupancy_label_validation_report(report: OccupancyLabelValidationReport) -> str:
    """Render a compact text report for CLI use."""

    lines = [
        f"occupancy_labels_ready: {str(report.ready).lower()}",
        f"manifest: {report.manifest_path}",
        f"samples: {report.sample_count}",
        f"target_field: {report.target_field}",
        f"num_classes: {report.num_classes}",
        f"ignore_index: {report.ignore_index}",
        f"voxels: {report.voxel_count}",
        f"ignored: {report.ignored_count}",
        "class_histogram:",
    ]
    if report.class_histogram:
        lines.extend(f"- {class_id}: {count}" for class_id, count in sorted(report.class_histogram.items()))
    else:
        lines.append("- none")
    lines.append("sample_tokens:")
    lines.extend(f"- {token}" for token in report.sample_tokens) if report.sample_tokens else lines.append("- none")
    if report.errors:
        lines.append("errors:")
        lines.extend(f"- {error}" for error in report.errors)
    return "\n".join(lines)
