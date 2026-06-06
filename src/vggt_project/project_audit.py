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
            "src/vggt_project/data/nuscenes_pointmap.py",
            "src/vggt_project/data/supervision_pipeline.py",
            "src/vggt_project/data/manifest_split.py",
            "scripts/generate_manifest.py",
            "scripts/validate_manifest.py",
            "scripts/split_manifest.py",
            "scripts/materialize_manifest_assets.py",
            "scripts/materialize_satellite_crops.py",
            "scripts/check_satellite_rasters.py",
            "scripts/generate_lidar_depth_targets.py",
            "scripts/generate_lidar_pointmap_targets.py",
            "scripts/generate_camera_lidar_pointmap_targets.py",
            "scripts/generate_lidar_supervision.py",
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
        "experiment_config": (
            "configs/reconstruction_first.yaml",
            "src/vggt_project/experiments.py",
            "scripts/run_experiment.py",
        ),
        "environment": (
            "requirements.txt",
            "environment.yml",
            "pyproject.toml",
            "scripts/setup_env.sh",
            "src/vggt_project/training_readiness.py",
            "scripts/check_training_readiness.py",
        ),
        "weights": ("scripts/download_weights.py",),
        "dataset_setup": (
            "scripts/prepare_nuscenes.sh",
            "scripts/check_nuscenes.py",
            "configs/satellite_rasters.example.json",
            "docs/datasets.md",
        ),
        "reference_setup": (
            "scripts/setup_references.py",
            "scripts/check_references.py",
            "refs/README.md",
        ),
        "github_ci": (".github/workflows/ci.yml",),
        "github_publish": (
            "scripts/check_github_publish.py",
            "scripts/publish_github.sh",
        ),
        "benchmarks": (
            "docs/research_survey.md",
            "refs/benchmarks/README.md",
        ),
    }

    items = tuple(_audit_item(resolved, name, paths) for name, paths in item_specs.items())
    remaining_gaps = (
        "real satellite patch extraction requires user-provided satellite rasters/config, though config validation and crop materialization are now scripted",
        "G3T/VGGT dense camera-level pointmap/pose target generation and occupancy target generation are not implemented, though LiDAR camera-frame pointmap target generation is now scripted",
        "G3T/VGGT head adapter and fine-tuning path are not implemented",
        "camera-specific G3T/VGGT pose heads are not implemented, though scaffold camera-depth and camera-pointmap heads are wired",
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
