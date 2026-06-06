"""Materialize model occupancy predictions into benchmark manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class OccupancyPredictionReport:
    manifest_path: Path
    output_manifest_path: Path | None
    sample_count: int
    prediction_maps_written: int


def materialize_occupancy_prediction_manifest(
    manifest_path: Path,
    *,
    prediction_fn: Callable[[dict, int], dict],
    target_dir: Path = Path("occupancy_predictions"),
    output_manifest_path: Path | None = None,
    prediction_field: str = "predicted_occupancy_path",
    source_key: str = "bev_occupancy",
    array_format: str = "npy",
    binary_threshold: float = 0.5,
    overwrite: bool = False,
) -> OccupancyPredictionReport:
    """Write occupancy predictions and add prediction paths to manifest records."""

    if array_format not in {"json", "npy"}:
        raise ValueError("array_format must be 'json' or 'npy'")

    output_base = output_manifest_path.parent if output_manifest_path is not None else manifest_path.parent
    records = _read_jsonl_records(manifest_path)
    written = 0

    for index, record in enumerate(records):
        token = str(record["token"])
        prediction = prediction_fn(record, index)
        if source_key not in prediction:
            raise ValueError(f"prediction missing required occupancy key: {source_key}")
        labels = _binary_labels(_remove_batch_dimension(prediction[source_key]), threshold=binary_threshold)
        relative_path = target_dir / f"{token}.{array_format}"
        absolute_path = _resolve(output_base, relative_path)
        record[prediction_field] = str(relative_path) if not relative_path.is_absolute() else str(absolute_path)
        if overwrite or not absolute_path.exists():
            absolute_path.parent.mkdir(parents=True, exist_ok=True)
            _write_array(labels, absolute_path, array_format=array_format)
            written += 1

    if output_manifest_path is not None:
        _write_jsonl_records(records, output_manifest_path)

    return OccupancyPredictionReport(
        manifest_path=manifest_path,
        output_manifest_path=output_manifest_path,
        sample_count=len(records),
        prediction_maps_written=written,
    )


def _remove_batch_dimension(value: Any) -> Any:
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 0 and int(shape[0]) == 1:
        return value[0]
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        return value[0]
    return value


def _binary_labels(value: Any, *, threshold: float) -> Any:
    value = _to_plain_list(value)

    def visit(item: Any) -> Any:
        if isinstance(item, (list, tuple)):
            return [visit(child) for child in item]
        return 1 if float(item) >= threshold else 0

    return visit(value)


def _to_plain_list(value: Any) -> Any:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach().cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    return value


def _write_array(value: Any, path: Path, *, array_format: str) -> None:
    if array_format == "json":
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        return
    import numpy as np

    np.save(path, np.asarray(value, dtype=np.int64))


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
