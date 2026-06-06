"""Lightweight occupancy benchmark metrics for exported prediction arrays."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class OccupancyBenchmarkReport:
    manifest_path: Path
    sample_count: int
    sample_tokens: tuple[str, ...]
    prediction_field: str
    target_field: str
    num_classes: int
    ignore_index: int | None
    class_iou: dict[str, float]
    occupancy_miou: float

    def to_json(self) -> str:
        payload = asdict(self)
        payload["manifest_path"] = str(self.manifest_path)
        return json.dumps(payload, indent=2, sort_keys=True)


def evaluate_occupancy_manifest(
    manifest_path: Path,
    *,
    prediction_field: str = "predicted_occupancy_path",
    target_field: str = "occupancy_path",
    num_classes: int = 2,
    ignore_index: int | None = None,
    binary_threshold: float = 0.5,
) -> OccupancyBenchmarkReport:
    """Evaluate semantic occupancy predictions referenced by a JSONL manifest."""

    if num_classes <= 0:
        raise ValueError("num_classes must be positive")

    records = _read_jsonl(manifest_path)
    if not records:
        raise ValueError(f"manifest is empty: {manifest_path}")

    intersections = [0 for _ in range(num_classes)]
    unions = [0 for _ in range(num_classes)]
    sample_tokens: list[str] = []
    base = manifest_path.parent

    for index, record in enumerate(records):
        prediction_path = _required_path(record, prediction_field, index, base)
        target_path = _required_path(record, target_field, index, base)
        prediction = _load_array(prediction_path)
        target = _load_array(target_path)
        prediction_flat = _flatten_array(prediction)
        target_flat = _flatten_array(target)
        if len(prediction_flat) != len(target_flat):
            raise ValueError(
                f"prediction/target shape mismatch for record {index}: "
                f"{prediction_path} has {len(prediction_flat)} values, {target_path} has {len(target_flat)}"
            )
        if num_classes == 2:
            prediction_labels = [1 if float(value) >= binary_threshold else 0 for value in prediction_flat]
            target_labels = [1 if float(value) >= binary_threshold else 0 for value in target_flat]
        else:
            prediction_labels = [int(value) for value in prediction_flat]
            target_labels = [int(value) for value in target_flat]

        for predicted, actual in zip(prediction_labels, target_labels):
            if ignore_index is not None and actual == ignore_index:
                continue
            for class_index in range(num_classes):
                predicted_is_class = predicted == class_index
                actual_is_class = actual == class_index
                if predicted_is_class and actual_is_class:
                    intersections[class_index] += 1
                if predicted_is_class or actual_is_class:
                    unions[class_index] += 1
        sample_tokens.append(str(record.get("token", index)))

    class_iou = {
        str(class_index): (intersections[class_index] / unions[class_index] if unions[class_index] else 0.0)
        for class_index in range(num_classes)
    }
    occupancy_miou = sum(class_iou.values()) / num_classes
    return OccupancyBenchmarkReport(
        manifest_path=manifest_path,
        sample_count=len(records),
        sample_tokens=tuple(sample_tokens),
        prediction_field=prediction_field,
        target_field=target_field,
        num_classes=num_classes,
        ignore_index=ignore_index,
        class_iou=class_iou,
        occupancy_miou=occupancy_miou,
    )


def format_occupancy_benchmark_report(report: OccupancyBenchmarkReport) -> str:
    """Render an occupancy benchmark report for CLI output."""

    lines = [
        f"manifest: {report.manifest_path}",
        f"samples: {report.sample_count}",
        f"num_classes: {report.num_classes}",
        f"ignore_index: {report.ignore_index}",
        f"occupancy_miou: {report.occupancy_miou:.6f}",
        "class_iou:",
    ]
    lines.extend(f"- {class_name}: {score:.6f}" for class_name, score in sorted(report.class_iou.items()))
    lines.append("sample_tokens:")
    lines.extend(f"- {token}" for token in report.sample_tokens)
    return "\n".join(lines)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _required_path(record: dict[str, Any], field: str, index: int, base: Path) -> Path:
    value = record.get(field)
    if not value:
        raise ValueError(f"record {index} missing required field: {field}")
    path = Path(str(value))
    return path if path.is_absolute() else base / path


def _load_array(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".npy", ".npz"}:
        import numpy as np

        loaded = np.load(path)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            for key in ("occupancy", "labels", "prediction", "arr_0"):
                if key in loaded:
                    return loaded[key]
            raise ValueError(f"npz occupancy file must contain occupancy, labels, prediction, or arr_0: {path}")
        return loaded
    if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        from PIL import Image

        return Image.open(path).convert("L")
    raise ValueError(f"unsupported occupancy array format: {path}")


def _flatten_array(value: Any) -> list[Any]:
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    if hasattr(value, "getdata"):
        value = list(value.getdata())
    flattened: list[Any] = []

    def visit(item: Any) -> None:
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        else:
            flattened.append(item)

    visit(value)
    return flattened
