"""Generate an executable command plan for a real training run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from vggt_project.experiments import DEFAULT_EXPERIMENT_CONFIG_PATH, ExperimentRunConfig


@dataclass(frozen=True)
class TrainingPlanStep:
    name: str
    command: str
    output_path: Path | None = None
    ready: bool = False
    note: str = ""


@dataclass(frozen=True)
class TrainingRunPlan:
    steps: tuple[TrainingPlanStep, ...]
    missing_outputs: tuple[str, ...]
    ready_to_train: bool

    def to_json(self) -> str:
        """Render the training plan as machine-readable JSON."""

        return json.dumps(
            {
                "missing_outputs": list(self.missing_outputs),
                "ready_to_train": self.ready_to_train,
                "steps": [
                    {
                        "name": step.name,
                        "command": step.command,
                        "output_path": str(step.output_path) if step.output_path is not None else None,
                        "ready": step.ready,
                        "note": step.note,
                    }
                    for step in self.steps
                ],
            },
            indent=2,
            sort_keys=True,
        )


def build_training_run_plan(
    config: ExperimentRunConfig,
    *,
    config_path: Path = DEFAULT_EXPERIMENT_CONFIG_PATH,
    nuscenes_root: Path = Path("data/nuscenes"),
    nuscenes_version: str = "v1.0-mini",
    raw_manifest_path: Path = Path("data/manifests/nuscenes-mini.jsonl"),
    satellite_manifest_path: Path = Path("data/manifests/nuscenes-mini.satellite.jsonl"),
    smoke_manifest_path: Path = Path("data/manifests/nuscenes-mini.smoke.jsonl"),
    pose_manifest_path: Path = Path("data/manifests/nuscenes-mini.pose.jsonl"),
    occupancy_manifest_path: Path = Path("data/manifests/nuscenes-mini.occupancy.jsonl"),
    path_exists: Callable[[Path], bool] | None = None,
) -> TrainingRunPlan:
    """Return the ordered preprocessing/train commands for the configured run."""

    path_exists = path_exists or Path.exists
    supervised_manifest = config.manifest_path or Path("data/manifests/nuscenes-mini.supervised.jsonl")
    train_manifest = config.train_manifest_path or supervised_manifest
    eval_manifest = config.eval_manifest_path or supervised_manifest
    occ3d_eval_manifest = eval_manifest.with_suffix(".occ3d.jsonl")
    occupancy_prediction_manifest = occ3d_eval_manifest.with_suffix(".predictions.jsonl")
    train_metrics = config.output_dir / "train_metrics.json"
    eval_metrics = config.output_dir / "eval_metrics.json"
    occupancy_report = config.output_dir / "occupancy_benchmark.json"
    satellite_config = config.satellite_raster_config_path
    point_count = config.point_count

    satellite_input_manifest = satellite_manifest_path if satellite_config else smoke_manifest_path
    pose_input_manifest = pose_manifest_path
    occupancy_input_manifest = occupancy_manifest_path
    split_ready = path_exists(train_manifest) and path_exists(eval_manifest)
    reference_supervision_requested = _reference_supervision_requested(config)

    steps = [
        TrainingPlanStep(
            name="check_nuscenes_layout",
            command=f"PYTHONPATH=src python scripts/check_nuscenes.py --root {nuscenes_root} --version {nuscenes_version}",
            ready=path_exists(nuscenes_root),
            note="Confirms nuScenes metadata and sample folders are present.",
        ),
        TrainingPlanStep(
            name="generate_base_manifest",
            command=(
                "PYTHONPATH=src python scripts/generate_manifest.py "
                f"--root {nuscenes_root} --version {nuscenes_version} "
                "--satellite-patch-dir satellite "
                f"--output {raw_manifest_path}"
            ),
            output_path=raw_manifest_path,
            ready=path_exists(raw_manifest_path),
            note="Creates one JSONL record per nuScenes sample.",
        ),
    ]

    if satellite_config is not None:
        steps.extend(
            [
                TrainingPlanStep(
                    name="check_satellite_rasters",
                    command=(
                        "PYTHONPATH=src python scripts/check_satellite_rasters.py "
                        f"--config {satellite_config} --manifest {raw_manifest_path}"
                    ),
                    output_path=satellite_config,
                    ready=path_exists(satellite_config),
                    note="Validates local satellite raster paths and map-location coverage.",
                ),
                TrainingPlanStep(
                    name="materialize_satellite_crops",
                    command=(
                        "PYTHONPATH=src python scripts/materialize_satellite_crops.py "
                        f"{raw_manifest_path} --config {satellite_config} --output {satellite_manifest_path}"
                    ),
                    output_path=satellite_manifest_path,
                    ready=path_exists(satellite_manifest_path),
                    note="Writes metric-aligned satellite patch images.",
                ),
            ]
        )
    else:
        steps.append(
            TrainingPlanStep(
                name="materialize_smoke_satellite_assets",
                command=(
                    "PYTHONPATH=src python scripts/materialize_manifest_assets.py "
                    f"{raw_manifest_path} --create-valid-masks --output {smoke_manifest_path}"
                ),
                output_path=smoke_manifest_path,
                ready=path_exists(smoke_manifest_path),
                note="Fallback smoke path when real satellite rasters are not configured.",
            )
        )

    steps.extend(
        [
            TrainingPlanStep(
                name="generate_camera_pose_targets",
                command=(
                    "PYTHONPATH=src python scripts/generate_camera_pose_targets.py "
                    f"{satellite_input_manifest} --root {nuscenes_root} --version {nuscenes_version} "
                    f"--camera CAM_FRONT --camera CAM_BACK --output {pose_manifest_path}"
                ),
                output_path=pose_manifest_path,
                ready=path_exists(pose_manifest_path),
                note="Adds calibration-derived per-camera pose targets to manifest records.",
            ),
            TrainingPlanStep(
                name="generate_lidar_supervision",
                command=(
                    "PYTHONPATH=src python scripts/generate_lidar_supervision.py "
                    f"{pose_input_manifest} --root {nuscenes_root} --version {nuscenes_version} "
                    "--camera CAM_FRONT --camera CAM_BACK --pointmap-target-frame camera "
                    f"--pointmap-dir camera_pointmaps --max-points {point_count} --output {supervised_manifest}"
                ),
                output_path=supervised_manifest,
                ready=path_exists(supervised_manifest),
                note="Adds multi-camera depth and camera-frame pointmap targets.",
            ),
            TrainingPlanStep(
                name="generate_lidar_occupancy_targets",
                command=(
                    "PYTHONPATH=src python scripts/generate_lidar_occupancy_targets.py "
                    f"{supervised_manifest} --root {nuscenes_root} --version {nuscenes_version} "
                    f"--occupancy-dir occupancy --output {occupancy_manifest_path}"
                ),
                output_path=occupancy_manifest_path,
                ready=path_exists(occupancy_manifest_path),
                note="Adds LiDAR-derived BEV occupancy targets for optional geometry supervision.",
            ),
            TrainingPlanStep(
                name="optional_generate_reference_supervision",
                command=(
                    "PYTHONPATH=src python scripts/generate_reference_supervision_targets.py "
                    f"--config {config_path} --manifest {occupancy_input_manifest} "
                    f"--output {occupancy_input_manifest} --target-dir reference_targets --max-points {point_count}"
                ),
                ready=not reference_supervision_requested,
                note=(
                    "Optionally replaces sparse LiDAR targets with dense configured G3T/VGGT reference predictions."
                    if reference_supervision_requested
                    else "Skipped unless a G3T/VGGT reference adapter is configured."
                ),
            ),
            TrainingPlanStep(
                name="inspect_manifest_sample",
                command=(
                    "PYTHONPATH=src python scripts/inspect_manifest_sample.py "
                    f"{occupancy_input_manifest} --output-dir outputs/manifest-preview --sample-index 0"
                ),
                ready=path_exists(occupancy_manifest_path),
                note="Writes a JSON summary and contact sheet for camera/satellite/target sanity checks.",
            ),
            TrainingPlanStep(
                name="split_manifest",
                command=(
                    "PYTHONPATH=src python scripts/split_manifest.py "
                    f"{occupancy_input_manifest} --train-output {train_manifest} --eval-output {eval_manifest} "
                    "--eval-fraction 0.2 --seed 0"
                ),
                ready=split_ready,
                note="Creates scene-disjoint train/eval manifests.",
            ),
            TrainingPlanStep(
                name="validate_train_manifest",
                command=f"PYTHONPATH=src python scripts/validate_manifest.py {train_manifest}",
                ready=False,
                note="Validates that train manifest file references exist before DataLoader construction.",
            ),
            TrainingPlanStep(
                name="validate_eval_manifest",
                command=f"PYTHONPATH=src python scripts/validate_manifest.py {eval_manifest}",
                ready=False,
                note="Validates that eval manifest file references exist before evaluation.",
            ),
        ]
    )
    if config.weights_path is not None:
        steps.append(
            TrainingPlanStep(
                name="inspect_checkpoint",
                command=f"PYTHONPATH=src python scripts/inspect_checkpoint.py {config.weights_path}",
                ready=False,
                note="Inspects configured weights before readiness and adapter checks.",
            )
        )
    steps.extend(
        [
            TrainingPlanStep(
                name="check_training_readiness",
                command=f"PYTHONPATH=src python scripts/check_training_readiness.py --config {config_path}",
                ready=False,
                note="Verifies dependencies, device, paths, satellite config, checkpoint loadability, and adapter config.",
            ),
            TrainingPlanStep(
                name="check_model_adapter",
                command=f"PYTHONPATH=src python scripts/check_model_adapter.py --config {config_path}",
                ready=False,
                note="Builds the configured model and verifies camera-aware reconstruction outputs.",
            ),
            TrainingPlanStep(
                name="probe_manifest_forward",
                command=f"PYTHONPATH=src python scripts/probe_manifest_forward.py --config {config_path}",
                ready=False,
                note="Runs one real manifest sample through the configured model before train/eval launch.",
            ),
            TrainingPlanStep(
                name="train",
                command=(
                    "PYTHONPATH=src python scripts/train.py "
                    f"--config {config_path} --metrics-output {train_metrics}"
                ),
                ready=False,
                note="Launches the configured training run.",
            ),
            TrainingPlanStep(
                name="evaluate",
                command=(
                    "PYTHONPATH=src python scripts/evaluate.py "
                    f"--config {config_path} --metrics-output {eval_metrics}"
                ),
                ready=False,
                note="Runs evaluation on the configured eval manifest.",
            ),
            TrainingPlanStep(
                name="optional_attach_occ3d_labels",
                command=(
                    "PYTHONPATH=src python scripts/attach_occ3d_labels.py "
                    f"--manifest {eval_manifest} --occ3d-root data/occ3d --output {occ3d_eval_manifest} "
                    f"--nuscenes-root {nuscenes_root} --nuscenes-version {nuscenes_version}"
                ),
                output_path=occ3d_eval_manifest,
                ready=path_exists(occ3d_eval_manifest),
                note="Optionally attaches public Occ3D/OpenOccupancy semantic labels for benchmark evaluation.",
            ),
            TrainingPlanStep(
                name="export_occupancy_predictions",
                command=(
                    "PYTHONPATH=src python scripts/export_occupancy_predictions.py "
                    f"--config {config_path} --manifest {occ3d_eval_manifest} --output {occupancy_prediction_manifest}"
                ),
                ready=False,
                note="Exports BEV occupancy predictions into the public-label manifest for semantic occupancy metrics.",
            ),
            TrainingPlanStep(
                name="evaluate_occupancy_benchmark",
                command=(
                    "PYTHONPATH=src python scripts/evaluate_occupancy_benchmark.py "
                    f"--manifest {occupancy_prediction_manifest} --num-classes 18 --json --output {occupancy_report}"
                ),
                ready=False,
                note="Computes class IoU and occupancy_miou from exported prediction/target arrays.",
            ),
            TrainingPlanStep(
                name="verify_training_artifacts",
                command=(
                    "PYTHONPATH=src python scripts/verify_training_artifacts.py "
                    f"--checkpoint {config.checkpoint} --train-metrics {train_metrics} --eval-metrics {eval_metrics} "
                    f"--occupancy-report {occupancy_report} --required-eval-metric loss "
                    "--required-eval-metric depth_mae"
                ),
                ready=False,
                note="Verifies checkpoint, train/eval report, and optional occupancy benchmark artifacts before recording results.",
            ),
        ]
    )

    required_ready = path_exists(supervised_manifest) and split_ready
    if required_ready:
        missing_outputs = ()
    else:
        missing = [
            f"{step.name}: {step.output_path}"
            for step in steps
            if step.output_path is not None and not step.ready
        ]
        for path in (train_manifest, eval_manifest):
            if not path_exists(path):
                missing.append(str(path))
        missing_outputs = tuple(dict.fromkeys(missing))
    return TrainingRunPlan(
        steps=tuple(steps),
        missing_outputs=missing_outputs,
        ready_to_train=required_ready,
    )


def format_training_run_plan(plan: TrainingRunPlan) -> str:
    """Render the plan as a compact CLI-friendly checklist."""

    lines = [f"ready_to_train: {str(plan.ready_to_train).lower()}", "steps:"]
    for index, step in enumerate(plan.steps, start=1):
        status = "ready" if step.ready else "pending"
        lines.append(f"{index}. {step.name}: {status}")
        if step.note:
            lines.append(f"   note: {step.note}")
        if step.output_path is not None:
            lines.append(f"   output: {step.output_path}")
        lines.append(f"   run: {step.command}")
    lines.append("missing_outputs:")
    if plan.missing_outputs:
        for output in plan.missing_outputs:
            lines.append(f"- {output}")
    else:
        lines.append("- none")
    return "\n".join(lines)


def _reference_supervision_requested(config: ExperimentRunConfig) -> bool:
    if config.use_reference_adapter:
        return True
    return config.model_family in {"g3t", "vggt", "g3t-vggt"}
