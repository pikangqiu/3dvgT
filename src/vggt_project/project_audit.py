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
    next_actions: tuple[str, ...]

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
            "src/vggt_project/data/manifest_preview.py",
            "src/vggt_project/data/manifest_tensor_dataset.py",
            "src/vggt_project/data/nuscenes_depth.py",
            "src/vggt_project/data/nuscenes_pointmap.py",
            "src/vggt_project/data/nuscenes_occupancy.py",
            "src/vggt_project/data/nuscenes_pose.py",
            "src/vggt_project/data/reference_supervision.py",
            "src/vggt_project/data/supervision_pipeline.py",
            "src/vggt_project/data/manifest_split.py",
            "scripts/generate_manifest.py",
            "scripts/validate_manifest.py",
            "scripts/inspect_manifest_sample.py",
            "scripts/split_manifest.py",
            "scripts/materialize_manifest_assets.py",
            "scripts/materialize_satellite_crops.py",
            "scripts/check_satellite_rasters.py",
            "scripts/generate_lidar_depth_targets.py",
            "scripts/generate_lidar_pointmap_targets.py",
            "scripts/generate_camera_lidar_pointmap_targets.py",
            "scripts/generate_lidar_occupancy_targets.py",
            "scripts/generate_camera_pose_targets.py",
            "scripts/generate_lidar_supervision.py",
            "scripts/generate_reference_supervision_targets.py",
        ),
        "model_framework": (
            "adapters/g3t_vggt_adapter.py",
            "src/vggt_project/models/adapter_contract.py",
            "src/vggt_project/models/interfaces.py",
            "src/vggt_project/models/factory.py",
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
            "configs/reconstruction_first.json",
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
            "scripts/check_model_adapter.py",
            "src/vggt_project/training_plan.py",
            "src/vggt_project/training_bootstrap.py",
            "src/vggt_project/training_launch.py",
            "scripts/plan_training_run.py",
            "scripts/bootstrap_training_run.py",
            "scripts/report_training_launch.py",
            "scripts/check_external_assets.py",
            "scripts/report_environment.py",
            "scripts/report_real_training_preflight.py",
            "scripts/probe_manifest_forward.py",
            "scripts/verify_training_artifacts.py",
            "scripts/verify_real_run_evidence.py",
            "src/vggt_project/training_artifacts.py",
            "src/vggt_project/real_run_evidence.py",
            "scripts/export_occupancy_predictions.py",
        ),
        "weights": (
            "scripts/prepare_model_weights.sh",
            "scripts/configure_model_weights.py",
            "scripts/download_weights.py",
            "scripts/inspect_checkpoint.py",
            "src/vggt_project/checkpoint_inspection.py",
        ),
        "dataset_setup": (
            "scripts/prepare_nuscenes.sh",
            "scripts/prepare_occ3d.sh",
            "scripts/attach_occ3d_labels.py",
            "scripts/prepare_satellite_rasters.sh",
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
            "src/vggt_project/experiment_protocol.py",
            "scripts/list_experiment_protocol.py",
            "src/vggt_project/occupancy_benchmark.py",
            "src/vggt_project/public_occupancy_manifest.py",
            "src/vggt_project/occupancy_label_validation.py",
            "src/vggt_project/data/occ3d_labels.py",
            "src/vggt_project/data/occupancy_predictions.py",
            "scripts/attach_occ3d_labels.py",
            "scripts/validate_public_occupancy_manifest.py",
            "scripts/validate_occupancy_labels.py",
            "scripts/export_occupancy_predictions.py",
            "scripts/evaluate_occupancy_benchmark.py",
        ),
    }

    items = tuple(_audit_item(resolved, name, paths) for name, paths in item_specs.items())
    remaining_gaps = (
        "real satellite patch extraction requires user-provided satellite rasters/config, though config validation and crop materialization are now scripted",
        "dense G3T/VGGT reference depth, pointmap, and pose target materialization, LiDAR-derived occupancy proxy generation, Occ3D label attachment, public occupancy manifest split validation, occupancy label class-id validation, occupancy prediction export, semantic occupancy metric evaluation, and evaluator-output artifact validation are scripted, but real public checkpoint/GPU validation plus actual-data evaluator execution are not complete yet",
        "G3T/VGGT adapter template, reference-output mapping, local reference builder, config-level reference instantiation, reference constructor kwargs, reference checkpoint loading hooks, configurable fine-tuning policies, and manifest forward probing are implemented, but full real-asset/head-call validation is not complete yet",
        "camera-specific scaffold pose heads and calibration-derived manifest pose targets are wired, but concrete real-checkpoint fine-tuning validation for G3T/VGGT pose heads is not complete yet",
    )
    next_actions = (
        "PYTHONPATH=src python3 scripts/report_real_training_preflight.py --config configs/reconstruction_first.json --json",
        "PYTHONPATH=src python3 scripts/report_training_launch.py --config configs/reconstruction_first.json --json",
        "PYTHONPATH=src python3 scripts/check_external_assets.py --config configs/reconstruction_first.json",
        "PYTHONPATH=src python3 scripts/plan_training_run.py --config configs/reconstruction_first.json",
        "Populate data/nuscenes and run bash scripts/prepare_satellite_rasters.sh, then edit data/satellite_rasters/config.json and run PYTHONPATH=src python3 scripts/check_training_readiness.py --config configs/reconstruction_first.json",
        "For public occupancy comparison, run bash scripts/prepare_occ3d.sh, attach labels with PYTHONPATH=src python3 scripts/attach_occ3d_labels.py --manifest data/manifests/nuscenes-mini.val.jsonl --occ3d-root data/occ3d --output data/manifests/nuscenes-mini.val.occ3d.jsonl --nuscenes-root data/nuscenes --nuscenes-version v1.0-trainval, validate split alignment with PYTHONPATH=src python3 scripts/validate_public_occupancy_manifest.py --manifest data/manifests/nuscenes-mini.val.occ3d.jsonl --expected-split trainval, validate class ids with PYTHONPATH=src python3 scripts/validate_occupancy_labels.py --manifest data/manifests/nuscenes-mini.val.occ3d.jsonl --num-classes 18, export prediction arrays with PYTHONPATH=src python3 scripts/export_occupancy_predictions.py --config configs/reconstruction_first.json --manifest data/manifests/nuscenes-mini.val.occ3d.jsonl --output data/manifests/nuscenes-mini.val.occ3d.predictions.jsonl, then run PYTHONPATH=src python3 scripts/evaluate_occupancy_benchmark.py --manifest data/manifests/nuscenes-mini.val.occ3d.predictions.jsonl --num-classes 18 --json --output outputs/manifest-smoke/occupancy_benchmark.json while keeping semantic metrics separate from local LiDAR-proxy bev_occupancy_iou",
        "Run bash scripts/prepare_model_weights.sh, inspect a concrete checkpoint, and write it with PYTHONPATH=src python3 scripts/configure_model_weights.py --weights-path <checkpoint>",
        "After train/eval/benchmark, run PYTHONPATH=src python3 scripts/verify_training_artifacts.py --checkpoint outputs/manifest-smoke/manifest_smoke_scaffold.pt --train-metrics outputs/manifest-smoke/train_metrics.json --eval-metrics outputs/manifest-smoke/eval_metrics.json --occupancy-report outputs/manifest-smoke/occupancy_benchmark.json --required-eval-metric loss --required-eval-metric depth_mae --required-occupancy-class-count 18",
        "Save environment, preflight, and artifact verifier JSON with PYTHONPATH=src python3 scripts/report_environment.py --output outputs/manifest-smoke/environment.json --json, then run PYTHONPATH=src python3 scripts/verify_real_run_evidence.py --preflight-report outputs/manifest-smoke/real_training_preflight.json --artifact-report outputs/manifest-smoke/training_artifacts.json --environment-report outputs/manifest-smoke/environment.json --output outputs/manifest-smoke/real_run_evidence.json --require-clean-worktree --expected-git-commit $(git rev-parse HEAD) --json",
        "Run PYTHONPATH=src python3 scripts/check_model_adapter.py --config configs/reconstruction_first.json and PYTHONPATH=src python3 scripts/probe_manifest_forward.py --config configs/reconstruction_first.json after the reference adapter, weights, and manifests are configured",
    )
    return ProjectAuditReport(
        items=items,
        real_training_complete=False,
        remaining_gaps=remaining_gaps,
        next_actions=next_actions,
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
    lines.append("next_actions:")
    for action in report.next_actions:
        lines.append(f"- {action}")
    return "\n".join(lines)


def _audit_item(root: Path, name: str, paths: tuple[str, ...]) -> ProjectAuditItem:
    missing = tuple(path for path in paths if not (root / path).exists())
    return ProjectAuditItem(
        name=name,
        ready=not missing,
        evidence=paths,
        missing=missing,
    )
