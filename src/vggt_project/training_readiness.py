"""Training readiness checks for real experiment setup."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from vggt_project.experiments import ExperimentRunConfig


@dataclass(frozen=True)
class DependencyStatus:
    name: str
    available: bool
    version: str | None = None


@dataclass(frozen=True)
class TrainingReadinessReport:
    ready: bool
    mode: str
    device: str | None
    device_available: bool
    missing_paths: dict[str, str]
    dependencies: tuple[DependencyStatus, ...]

    @property
    def missing_dependencies(self) -> tuple[str, ...]:
        return tuple(status.name for status in self.dependencies if not status.available)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def check_training_readiness(
    config: ExperimentRunConfig,
    *,
    dependency_probe: Callable[[], tuple[DependencyStatus, ...]] | None = None,
    device_probe: Callable[[str | None], bool] | None = None,
) -> TrainingReadinessReport:
    """Check config paths, dependencies, and requested device availability."""

    missing_paths = _missing_config_paths(config)
    dependency_probe = dependency_probe or probe_dependencies
    device_probe = device_probe or probe_device_available
    dependencies = dependency_probe()
    device_available = device_probe(config.device)
    ready = not missing_paths and not any(
        not dependency.available for dependency in dependencies
    ) and device_available
    return TrainingReadinessReport(
        ready=ready,
        mode=config.training_mode,
        device=config.device,
        device_available=device_available,
        missing_paths=missing_paths,
        dependencies=dependencies,
    )


def probe_dependencies() -> tuple[DependencyStatus, ...]:
    """Probe core packages needed for scaffold training."""

    return tuple(
        _dependency_status(name)
        for name in ("torch", "PIL", "numpy", "yaml")
    )


def probe_device_available(device: str | None) -> bool:
    """Return whether the configured device is usable in the current torch env."""

    if device is None:
        return True
    try:
        import torch
    except ModuleNotFoundError:
        return False
    if device == "cpu":
        return True
    if device == "cuda":
        return bool(torch.cuda.is_available())
    if device == "mps":
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    try:
        torch.device(device)
    except (RuntimeError, TypeError):
        return False
    return True


def format_training_readiness_report(report: TrainingReadinessReport) -> str:
    """Render a compact human-readable readiness report."""

    lines = [
        f"ready: {str(report.ready).lower()}",
        f"mode: {report.mode}",
        f"device: {report.device or 'auto'}",
        f"device_available: {str(report.device_available).lower()}",
        "dependencies:",
    ]
    for dependency in report.dependencies:
        version = f" ({dependency.version})" if dependency.version else ""
        status = "ready" if dependency.available else "missing"
        lines.append(f"- {dependency.name}: {status}{version}")
    lines.append("paths:")
    if not report.missing_paths:
        lines.append("- all configured paths exist")
    else:
        for name, path in report.missing_paths.items():
            lines.append(f"- {name}: missing {path}")
    return "\n".join(lines)


def _missing_config_paths(config: ExperimentRunConfig) -> dict[str, str]:
    missing: dict[str, str] = {}
    for name, path in {
        "manifest_path": config.manifest_path,
        "train_manifest_path": config.train_manifest_path,
        "eval_manifest_path": config.eval_manifest_path,
    }.items():
        if path is not None and not Path(path).exists():
            missing[name] = str(path)
    return missing


def _dependency_status(name: str) -> DependencyStatus:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return DependencyStatus(name=name, available=False)
    version = _dependency_version(name)
    return DependencyStatus(name=name, available=True, version=version)


def _dependency_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except Exception:
        return None
    return getattr(module, "__version__", None)
