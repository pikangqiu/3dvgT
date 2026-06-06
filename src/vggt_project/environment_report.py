"""Capture a reproducible runtime environment snapshot for real runs."""

from __future__ import annotations

import importlib.util
import json
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Callable


DEFAULT_DEPENDENCIES = ("torch", "PIL", "numpy", "yaml")


@dataclass(frozen=True)
class DependencySnapshot:
    name: str
    available: bool
    version: str | None = None


@dataclass(frozen=True)
class TorchRuntimeSnapshot:
    importable: bool
    version: str | None
    cuda_available: bool
    cuda_device_count: int
    mps_available: bool


@dataclass(frozen=True)
class EnvironmentReport:
    ready: bool
    python_version: str
    python_executable: str
    platform: str
    machine: str
    dependencies: tuple[DependencySnapshot, ...]
    torch_runtime: TorchRuntimeSnapshot

    @property
    def missing_dependencies(self) -> tuple[str, ...]:
        return tuple(dependency.name for dependency in self.dependencies if not dependency.available)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def collect_environment_report(
    *,
    dependency_probe: Callable[[tuple[str, ...]], tuple[DependencySnapshot, ...]] | None = None,
    torch_runtime_probe: Callable[[], TorchRuntimeSnapshot] | None = None,
    dependency_names: tuple[str, ...] = DEFAULT_DEPENDENCIES,
) -> EnvironmentReport:
    """Collect package, interpreter, platform, and accelerator readiness."""

    dependency_probe = dependency_probe or probe_dependencies
    torch_runtime_probe = torch_runtime_probe or probe_torch_runtime
    dependencies = dependency_probe(dependency_names)
    torch_runtime = torch_runtime_probe()
    ready = not any(not dependency.available for dependency in dependencies)
    return EnvironmentReport(
        ready=ready,
        python_version=sys.version.split()[0],
        python_executable=sys.executable,
        platform=platform.platform(),
        machine=platform.machine(),
        dependencies=dependencies,
        torch_runtime=torch_runtime,
    )


def probe_dependencies(names: tuple[str, ...] = DEFAULT_DEPENDENCIES) -> tuple[DependencySnapshot, ...]:
    """Probe importability and versions for runtime dependencies."""

    return tuple(_dependency_snapshot(name) for name in names)


def probe_torch_runtime() -> TorchRuntimeSnapshot:
    """Return torch accelerator availability without requiring CUDA/MPS to exist."""

    try:
        import torch
    except Exception:
        return TorchRuntimeSnapshot(
            importable=False,
            version=None,
            cuda_available=False,
            cuda_device_count=0,
            mps_available=False,
        )

    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())
    return TorchRuntimeSnapshot(
        importable=True,
        version=getattr(torch, "__version__", None),
        cuda_available=bool(torch.cuda.is_available()),
        cuda_device_count=int(torch.cuda.device_count()),
        mps_available=mps_available,
    )


def format_environment_report(report: EnvironmentReport) -> str:
    """Render a compact human-readable environment report."""

    lines = [
        f"environment_ready: {str(report.ready).lower()}",
        f"python_version: {report.python_version}",
        f"python_executable: {report.python_executable}",
        f"platform: {report.platform}",
        f"machine: {report.machine}",
        "dependencies:",
    ]
    for dependency in report.dependencies:
        status = "ready" if dependency.available else "missing"
        version = f" ({dependency.version})" if dependency.version else ""
        lines.append(f"- {dependency.name}: {status}{version}")
    lines.extend(
        [
            "torch_runtime:",
            f"- importable: {str(report.torch_runtime.importable).lower()}",
            f"- version: {report.torch_runtime.version or '<unknown>'}",
            f"- cuda_available: {str(report.torch_runtime.cuda_available).lower()}",
            f"- cuda_device_count: {report.torch_runtime.cuda_device_count}",
            f"- mps_available: {str(report.torch_runtime.mps_available).lower()}",
        ]
    )
    return "\n".join(lines)


def _dependency_snapshot(name: str) -> DependencySnapshot:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return DependencySnapshot(name=name, available=False)
    return DependencySnapshot(name=name, available=True, version=_dependency_version(name))


def _dependency_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except Exception:
        return None
    return getattr(module, "__version__", None)
