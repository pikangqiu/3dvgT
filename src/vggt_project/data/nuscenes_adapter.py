"""nuScenes dataset adapter scaffolding.

This module intentionally starts with filesystem/layout inspection. The real
sample loader should be added only after the dataset root, satellite source, and
target supervision are fixed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NuScenesAdapterConfig:
    """Configuration for locating a nuScenes split."""

    root: Path = Path("data/nuscenes")
    version: str = "v1.0-mini"


@dataclass(frozen=True)
class NuScenesRootStatus:
    """Filesystem readiness report for a nuScenes root."""

    root: Path
    version: str
    ready: bool
    missing: tuple[str, ...]
    expected_layout: tuple[str, ...]


def expected_nuscenes_layout(version: str) -> tuple[str, ...]:
    """Return required top-level entries for the current project stage."""

    return ("samples", "sweeps", "maps", version)


def inspect_nuscenes_root(config: NuScenesAdapterConfig) -> NuScenesRootStatus:
    """Inspect whether the nuScenes root has the minimum expected layout."""

    expected = expected_nuscenes_layout(config.version)
    root = config.root
    if not root.exists():
        missing = ("root",) + expected
    else:
        missing = tuple(name for name in expected if not (root / name).exists())

    return NuScenesRootStatus(
        root=root,
        version=config.version,
        ready=not missing,
        missing=missing,
        expected_layout=expected,
    )

