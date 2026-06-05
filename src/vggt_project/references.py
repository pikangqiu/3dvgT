"""Utilities for validating external reference repositories and papers."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from vggt_project.config import ReferencePaths


@dataclass(frozen=True)
class ReferenceStatus:
    """Current availability of a local reference."""

    name: str
    path: Path
    exists: bool
    details: str


def _git_short_head(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def collect_reference_status(paths: ReferencePaths | None = None) -> list[ReferenceStatus]:
    """Return status records for all known references."""

    resolved = (paths or ReferencePaths()).resolve()

    g3t_head = _git_short_head(resolved.g3t) if resolved.g3t.exists() else None
    statuses = [
        ReferenceStatus(
            name="g3t",
            path=resolved.g3t,
            exists=resolved.g3t.exists(),
            details=f"git HEAD {g3t_head}" if g3t_head else "missing or not a git repository",
        ),
        ReferenceStatus(
            name="look_from_above_notes",
            path=resolved.look_from_above_notes,
            exists=resolved.look_from_above_notes.exists(),
            details="paper-derived notes; public code repository not identified",
        ),
        ReferenceStatus(
            name="look_from_above_paper",
            path=resolved.look_from_above_paper,
            exists=resolved.look_from_above_paper.exists(),
            details="local PDF reference",
        ),
    ]
    return statuses


def require_core_references(paths: ReferencePaths | None = None) -> None:
    """Raise if the references needed for current scaffold work are missing."""

    missing = [status for status in collect_reference_status(paths) if not status.exists]
    if missing:
        joined = ", ".join(f"{item.name} at {item.path}" for item in missing)
        raise FileNotFoundError(f"Missing project references: {joined}")

