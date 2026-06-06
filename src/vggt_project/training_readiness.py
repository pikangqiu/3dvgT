"""Training readiness checks for real experiment setup."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from vggt_project.data.satellite_crops import validate_satellite_raster_config
from vggt_project.experiments import ExperimentRunConfig
from vggt_project.models.factory import FINE_TUNING_POLICIES


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
    config_errors: tuple[str, ...]
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
    config_errors = _config_errors(config)
    ready = not missing_paths and not config_errors and not any(
        not dependency.available for dependency in dependencies
    ) and device_available
    return TrainingReadinessReport(
        ready=ready,
        mode=config.training_mode,
        device=config.device,
        device_available=device_available,
        missing_paths=missing_paths,
        config_errors=config_errors,
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
    lines.append("config_errors:")
    if not report.config_errors:
        lines.append("- none")
    else:
        for error in report.config_errors:
            lines.append(f"- {error}")
    return "\n".join(lines)


def _missing_config_paths(config: ExperimentRunConfig) -> dict[str, str]:
    missing: dict[str, str] = {}
    for name, path in {
        "manifest_path": config.manifest_path,
        "train_manifest_path": config.train_manifest_path,
        "eval_manifest_path": config.eval_manifest_path,
        "satellite_raster_config_path": config.satellite_raster_config_path,
        "adapter_module_path": config.adapter_module_path,
        "weights_path": config.weights_path,
        "reference_root": config.reference_root if config.use_reference_adapter else None,
    }.items():
        if path is not None and not Path(path).exists():
            missing[name] = str(path)
    return missing


def _config_errors(config: ExperimentRunConfig) -> tuple[str, ...]:
    errors = list(_model_config_errors(config))
    if config.satellite_raster_config_path is None:
        return tuple(errors)
    config_path = Path(config.satellite_raster_config_path)
    if not config_path.exists():
        return tuple(errors)

    manifest_path = _existing_manifest_for_satellite_check(config)
    report = validate_satellite_raster_config(config_path, manifest_path=manifest_path)
    for location in report.missing_manifest_locations:
        errors.append(f"satellite_raster_config missing map_location {location}")
    for path in report.missing_raster_paths:
        errors.append(f"satellite raster missing {path}")
    for issue in report.invalid_specs:
        errors.append(f"satellite_raster_config {issue.map_location}.{issue.field}: {issue.reason}")
    return tuple(errors)


def _model_config_errors(config: ExperimentRunConfig) -> tuple[str, ...]:
    errors: list[str] = []
    policy = config.fine_tuning_policy.lower().replace("-", "_")
    if policy == "all":
        policy = "full"
    if policy == "freeze_backbone":
        policy = "frozen_backbone"
    if policy not in FINE_TUNING_POLICIES:
        errors.append(
            "fine_tuning_policy must be one of "
            + ", ".join(FINE_TUNING_POLICIES)
            + f"; got {config.fine_tuning_policy}"
        )
    if config.model_family in {"scaffold", ""}:
        return tuple(errors)
    if config.model_family not in {"external", "g3t", "vggt", "g3t-vggt"}:
        errors.append(f"unsupported model family {config.model_family}")
        return tuple(errors)
    if config.adapter_module_path is None:
        errors.append(f"model family {config.model_family} requires adapter_module_path")
    if config.use_reference_adapter and config.reference_root is None:
        errors.append("use_reference_adapter requires reference_root")
    return tuple(errors)


def _existing_manifest_for_satellite_check(config: ExperimentRunConfig) -> Path | None:
    for path in (config.train_manifest_path, config.manifest_path, config.eval_manifest_path):
        if path is not None and Path(path).exists():
            return Path(path)
    return None


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
