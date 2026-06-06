"""Verify a saved evidence bundle after a real training run."""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class RealRunEvidenceReport:
    ready: bool
    preflight_report_path: Path
    artifact_report_path: Path
    environment_report_path: Path | None
    model_device_report_path: Path | None
    git_commit: str
    expected_git_commit: str | None
    clean_worktree: bool
    preflight_ready: bool
    artifacts_ready: bool
    environment_ready: bool
    model_device_ready: bool
    errors: tuple[str, ...]

    def to_json(self) -> str:
        payload = asdict(self)
        payload["preflight_report_path"] = str(self.preflight_report_path)
        payload["artifact_report_path"] = str(self.artifact_report_path)
        payload["environment_report_path"] = (
            str(self.environment_report_path) if self.environment_report_path is not None else None
        )
        payload["model_device_report_path"] = (
            str(self.model_device_report_path) if self.model_device_report_path is not None else None
        )
        return json.dumps(payload, indent=2, sort_keys=True)


def verify_real_run_evidence(
    *,
    preflight_report_path: Path,
    artifact_report_path: Path,
    environment_report_path: Path | None = None,
    model_device_report_path: Path | None = None,
    expected_git_commit: str | None = None,
    require_clean_worktree: bool = False,
    git_commit_probe: Callable[[], str] | None = None,
    git_status_probe: Callable[[], str] | None = None,
) -> RealRunEvidenceReport:
    """Verify preflight, artifact, and git evidence for a completed run."""

    errors: list[str] = []
    preflight_payload = _read_json_object(preflight_report_path, "preflight_report", errors)
    artifact_payload = _read_json_object(artifact_report_path, "artifact_report", errors)
    environment_payload = (
        _read_json_object(environment_report_path, "environment_report", errors)
        if environment_report_path is not None
        else {}
    )
    model_device_payload = (
        _read_json_object(model_device_report_path, "model_device_report", errors)
        if model_device_report_path is not None
        else {}
    )
    preflight_ready = preflight_payload.get("ready_for_real_training") is True
    artifacts_ready = artifact_payload.get("ready") is True
    environment_ready = environment_payload.get("ready") is True if environment_report_path is not None else True
    model_device_ready = model_device_payload.get("ready") is True if model_device_report_path is not None else True
    if preflight_payload and not preflight_ready:
        errors.append("preflight report is not ready")
    if artifact_payload and not artifacts_ready:
        errors.append("artifact report is not ready")
    if environment_payload and not environment_ready:
        errors.append("environment report is not ready")
    if model_device_payload and not model_device_ready:
        errors.append("model device report is not ready")

    git_commit = _probe_git_commit(git_commit_probe, errors)
    if expected_git_commit is not None and git_commit != expected_git_commit:
        errors.append(f"git commit mismatch: expected {expected_git_commit}, got {git_commit or '<unknown>'}")

    git_status = _probe_git_status(git_status_probe, errors)
    clean_worktree = git_status == ""
    if require_clean_worktree and not clean_worktree:
        errors.append("git worktree is not clean")

    return RealRunEvidenceReport(
        ready=not errors,
        preflight_report_path=preflight_report_path,
        artifact_report_path=artifact_report_path,
        environment_report_path=environment_report_path,
        model_device_report_path=model_device_report_path,
        git_commit=git_commit,
        expected_git_commit=expected_git_commit,
        clean_worktree=clean_worktree,
        preflight_ready=preflight_ready,
        artifacts_ready=artifacts_ready,
        environment_ready=environment_ready,
        model_device_ready=model_device_ready,
        errors=tuple(errors),
    )


def format_real_run_evidence_report(report: RealRunEvidenceReport) -> str:
    """Render a compact human-readable evidence report."""

    lines = [
        f"real_run_evidence_ready: {str(report.ready).lower()}",
        f"preflight_report: {report.preflight_report_path}",
        f"artifact_report: {report.artifact_report_path}",
        f"environment_report: {report.environment_report_path or '<none>'}",
        f"model_device_report: {report.model_device_report_path or '<none>'}",
        f"git_commit: {report.git_commit or '<unknown>'}",
        f"expected_git_commit: {report.expected_git_commit or '<none>'}",
        f"clean_worktree: {str(report.clean_worktree).lower()}",
        f"preflight_ready: {str(report.preflight_ready).lower()}",
        f"artifacts_ready: {str(report.artifacts_ready).lower()}",
        f"environment_ready: {str(report.environment_ready).lower()}",
        f"model_device_ready: {str(report.model_device_ready).lower()}",
    ]
    if report.errors:
        lines.append("errors:")
        lines.extend(f"- {error}" for error in report.errors)
    return "\n".join(lines)


def _read_json_object(path: Path, label: str, errors: list[str]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"{label} does not exist: {path}")
        return {}
    except json.JSONDecodeError as error:
        errors.append(f"{label} is not valid JSON: {error}")
        return {}
    if not isinstance(payload, dict):
        errors.append(f"{label} must be a JSON object: {path}")
        return {}
    return payload


def _probe_git_commit(probe: Callable[[], str] | None, errors: list[str]) -> str:
    if probe is not None:
        return probe().strip()
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        errors.append(f"could not read git commit: {error}")
        return ""


def _probe_git_status(probe: Callable[[], str] | None, errors: list[str]) -> str:
    if probe is not None:
        return probe().strip()
    try:
        return subprocess.check_output(["git", "status", "--short"], text=True).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        errors.append(f"could not read git status: {error}")
        return ""
