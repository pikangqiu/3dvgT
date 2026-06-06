"""Verify training/evaluation artifacts after a real run."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vggt_project.checkpoint_inspection import CHECKPOINT_SUFFIXES


@dataclass(frozen=True)
class TrainingArtifactReport:
    ready: bool
    checkpoint_path: Path
    experiment_report_path: Path | None
    train_metrics_path: Path | None
    eval_metrics_path: Path | None
    occupancy_report_path: Path | None
    present_artifacts: tuple[str, ...]
    missing_artifacts: tuple[str, ...]
    errors: tuple[str, ...]
    train_metrics: tuple[str, ...]
    eval_metrics: tuple[str, ...]
    occupancy_metrics: tuple[str, ...]

    def to_json(self) -> str:
        payload = asdict(self)
        payload["checkpoint_path"] = str(self.checkpoint_path)
        payload["experiment_report_path"] = (
            str(self.experiment_report_path) if self.experiment_report_path is not None else None
        )
        payload["train_metrics_path"] = str(self.train_metrics_path) if self.train_metrics_path is not None else None
        payload["eval_metrics_path"] = str(self.eval_metrics_path) if self.eval_metrics_path is not None else None
        payload["occupancy_report_path"] = (
            str(self.occupancy_report_path) if self.occupancy_report_path is not None else None
        )
        return json.dumps(payload, indent=2, sort_keys=True)


def verify_training_artifacts(
    *,
    checkpoint_path: Path,
    experiment_report_path: Path | None = None,
    train_metrics_path: Path | None = None,
    eval_metrics_path: Path | None = None,
    occupancy_report_path: Path | None = None,
    required_train_metrics: tuple[str, ...] = ("loss",),
    required_eval_metrics: tuple[str, ...] = ("loss",),
    require_occupancy_report: bool = False,
) -> TrainingArtifactReport:
    """Return whether a completed run has the expected result artifacts."""

    present: list[str] = []
    missing: list[str] = []
    errors: list[str] = []

    _check_checkpoint(checkpoint_path, present=present, missing=missing, errors=errors)
    experiment_payload: dict[str, Any] = {}
    train_payload: dict[str, Any] = {}
    eval_payload: dict[str, Any] = {}
    if experiment_report_path is not None:
        experiment_payload = _check_json_artifact(
            experiment_report_path,
            artifact_name="experiment_report",
            present=present,
            missing=missing,
            errors=errors,
        )
    else:
        if train_metrics_path is None:
            missing.append("train_metrics")
        else:
            train_payload = _check_json_artifact(
                train_metrics_path,
                artifact_name="train_metrics",
                present=present,
                missing=missing,
                errors=errors,
            )
        if eval_metrics_path is None:
            missing.append("eval_metrics")
        else:
            eval_payload = _check_json_artifact(
                eval_metrics_path,
                artifact_name="eval_metrics",
                present=present,
                missing=missing,
                errors=errors,
            )
    occupancy_payload: dict[str, Any] | None = None
    if occupancy_report_path is not None:
        occupancy_payload = _check_json_artifact(
            occupancy_report_path,
            artifact_name="occupancy_report",
            present=present,
            missing=missing,
            errors=errors,
        )
    elif require_occupancy_report:
        missing.append("occupancy_report")

    train_metrics = (
        _metric_keys(experiment_payload, "train_metrics")
        if experiment_payload
        else tuple(sorted(str(key) for key in train_payload))
    )
    eval_metrics = (
        _metric_keys(experiment_payload, "eval_metrics")
        if experiment_payload
        else tuple(sorted(str(key) for key in eval_payload))
    )
    occupancy_metrics = tuple(sorted(occupancy_payload.keys())) if occupancy_payload else ()

    _require_metrics(
        "experiment_report.train_metrics",
        train_metrics,
        required_train_metrics,
        errors=errors,
    )
    _require_metrics(
        "experiment_report.eval_metrics",
        eval_metrics,
        required_eval_metrics,
        errors=errors,
    )
    if occupancy_payload is not None:
        _require_metrics(
            "occupancy_report",
            occupancy_metrics,
            ("occupancy_miou", "class_iou"),
            errors=errors,
        )

    ready = not missing and not errors
    return TrainingArtifactReport(
        ready=ready,
        checkpoint_path=checkpoint_path,
        experiment_report_path=experiment_report_path,
        train_metrics_path=train_metrics_path,
        eval_metrics_path=eval_metrics_path,
        occupancy_report_path=occupancy_report_path,
        present_artifacts=tuple(present),
        missing_artifacts=tuple(missing),
        errors=tuple(errors),
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        occupancy_metrics=occupancy_metrics,
    )


def format_training_artifact_report(report: TrainingArtifactReport) -> str:
    """Render a compact text report for CLI use."""

    lines = [
        f"training_artifacts_ready: {str(report.ready).lower()}",
        f"checkpoint: {report.checkpoint_path}",
        f"experiment_report: {report.experiment_report_path or '<none>'}",
        f"train_metrics_path: {report.train_metrics_path or '<none>'}",
        f"eval_metrics_path: {report.eval_metrics_path or '<none>'}",
        f"occupancy_report: {report.occupancy_report_path or '<none>'}",
        "present_artifacts:",
    ]
    lines.extend(f"- {name}" for name in report.present_artifacts)
    lines.append("missing_artifacts:")
    lines.extend(f"- {name}" for name in report.missing_artifacts) if report.missing_artifacts else lines.append("- none")
    lines.append("train_metrics:")
    lines.extend(f"- {name}" for name in report.train_metrics) if report.train_metrics else lines.append("- none")
    lines.append("eval_metrics:")
    lines.extend(f"- {name}" for name in report.eval_metrics) if report.eval_metrics else lines.append("- none")
    if report.occupancy_report_path is not None:
        lines.append("occupancy_metrics:")
        lines.extend(f"- {name}" for name in report.occupancy_metrics) if report.occupancy_metrics else lines.append("- none")
    if report.errors:
        lines.append("errors:")
        lines.extend(f"- {error}" for error in report.errors)
    return "\n".join(lines)


def _check_checkpoint(
    path: Path,
    *,
    present: list[str],
    missing: list[str],
    errors: list[str],
) -> None:
    if not path.exists():
        missing.append("checkpoint")
        return
    if not path.is_file():
        errors.append(f"checkpoint is not a file: {path}")
        return
    if path.suffix.lower() not in CHECKPOINT_SUFFIXES:
        errors.append(f"checkpoint suffix must be one of {', '.join(CHECKPOINT_SUFFIXES)}: {path}")
        return
    present.append("checkpoint")


def _check_json_artifact(
    path: Path,
    *,
    artifact_name: str,
    present: list[str],
    missing: list[str],
    errors: list[str],
) -> dict[str, Any]:
    if not path.exists():
        missing.append(artifact_name)
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        errors.append(f"{artifact_name} is not valid JSON: {error}")
        return {}
    if not isinstance(payload, dict):
        errors.append(f"{artifact_name} must be a JSON object: {path}")
        return {}
    present.append(artifact_name)
    return payload


def _metric_keys(payload: dict[str, Any], field: str) -> tuple[str, ...]:
    metrics = payload.get(field)
    if not isinstance(metrics, dict):
        return ()
    return tuple(sorted(str(key) for key in metrics))


def _require_metrics(
    label: str,
    present_metrics: tuple[str, ...],
    required_metrics: tuple[str, ...],
    *,
    errors: list[str],
) -> None:
    present = set(present_metrics)
    for metric in required_metrics:
        if metric not in present:
            errors.append(f"{label} missing required metric: {metric}")
