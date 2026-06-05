"""Project completeness audit for the research scaffold."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectAuditItem:
    name: str
    ready: bool
    evidence: tuple[str, ...]
    missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectAuditReport:
    items: tuple[ProjectAuditItem, ...]
    real_training_complete: bool
    remaining_gaps: tuple[str, ...]

    @property
    def scaffold_ready(self) -> bool:
        return all(item.ready for item in self.items)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def audit_project_files(root: Path = Path(".")) -> ProjectAuditReport:
    """Audit whether the repository contains the requested scaffold areas."""

    resolved = root.resolve()
    item_specs = {
        "data_processing": (
            "src/vggt_project/data/manifest.py",
            "src/vggt_project/data/manifest_builder.py",
            "src/vggt_project/data/manifest_tensor_dataset.py",
            "src/vggt_project/data/nuscenes_depth.py",
            "scripts/generate_manifest.py",
            "scripts/validate_manifest.py",
            "scripts/materialize_manifest_assets.py",
            "scripts/generate_lidar_depth_targets.py",
        ),
        "model_framework": (
            "src/vggt_project/models/interfaces.py",
            "src/vggt_project/models/scaffold.py",
        ),
        "losses": ("src/vggt_project/losses.py",),
        "train_loop": (
            "src/vggt_project/training.py",
            "scripts/train.py",
            "scripts/run_smoke_pipeline.py",
        ),
        "eval_loop": (
            "src/vggt_project/evaluation.py",
            "scripts/evaluate.py",
        ),
        "environment": (
            "requirements.txt",
            "environment.yml",
            "pyproject.toml",
            "scripts/setup_env.sh",
        ),
        "weights": ("scripts/download_weights.py",),
        "dataset_setup": (
            "scripts/prepare_nuscenes.sh",
            "scripts/check_nuscenes.py",
            "docs/datasets.md",
        ),
        "reference_setup": (
            "scripts/setup_references.py",
            "scripts/check_references.py",
            "refs/README.md",
        ),
        "benchmarks": (
            "docs/research_survey.md",
            "refs/benchmarks/README.md",
        ),
    }

    items = tuple(_audit_item(resolved, name, paths) for name, paths in item_specs.items())
    remaining_gaps = (
        "real satellite patch extraction/alignment is not implemented",
        "real pointmap/pose/occupancy supervision generation is not implemented",
        "G3T/VGGT head adapter and fine-tuning path are not implemented",
        "multi-camera depth/pointmap target wiring is not implemented",
        "GitHub upload still requires gh authentication and remote creation",
    )
    return ProjectAuditReport(
        items=items,
        real_training_complete=False,
        remaining_gaps=remaining_gaps,
    )


def format_audit_report(report: ProjectAuditReport) -> str:
    """Return a compact human-readable audit report."""

    lines = [
        f"scaffold_ready: {str(report.scaffold_ready).lower()}",
        f"real_training_complete: {str(report.real_training_complete).lower()}",
        "items:",
    ]
    for item in report.items:
        marker = "ready" if item.ready else "missing"
        lines.append(f"- {item.name}: {marker}")
        if item.missing:
            lines.append(f"  missing: {', '.join(item.missing)}")
    lines.append("remaining_gaps:")
    for gap in report.remaining_gaps:
        lines.append(f"- {gap}")
    return "\n".join(lines)


def _audit_item(root: Path, name: str, paths: tuple[str, ...]) -> ProjectAuditItem:
    missing = tuple(path for path in paths if not (root / path).exists())
    return ProjectAuditItem(
        name=name,
        ready=not missing,
        evidence=paths,
        missing=missing,
    )
