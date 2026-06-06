"""Generate an executable command plan for a real training run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from vggt_project.experiments import ExperimentRunConfig


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


def build_training_run_plan(
    config: ExperimentRunConfig,
    *,
    config_path: Path = Path("configs/reconstruction_first.yaml"),
    nuscenes_root: Path = Path("data/nuscenes"),
    nuscenes_version: str = "v1.0-mini",
    raw_manifest_path: Path = Path("data/manifests/nuscenes-mini.jsonl"),
    satellite_manifest_path: Path = Path("data/manifests/nuscenes-mini.satellite.jsonl"),
    smoke_manifest_path: Path = Path("data/manifests/nuscenes-mini.smoke.jsonl"),
    pose_manifest_path: Path = Path("data/manifests/nuscenes-mini.pose.jsonl"),
    path_exists: Callable[[Path], bool] | None = None,
) -> TrainingRunPlan:
    """Return the ordered preprocessing/train commands for the configured run."""

    path_exists = path_exists or Path.exists
    supervised_manifest = config.manifest_path or Path("data/manifests/nuscenes-mini.supervised.jsonl")
    train_manifest = config.train_manifest_path or supervised_manifest
    eval_manifest = config.eval_manifest_path or supervised_manifest
    satellite_config = config.satellite_raster_config_path
    point_count = config.point_count

    satellite_input_manifest = satellite_manifest_path if satellite_config else smoke_manifest_path
    pose_input_manifest = pose_manifest_path
    split_ready = path_exists(train_manifest) and path_exists(eval_manifest)

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
                name="inspect_manifest_sample",
                command=(
                    "PYTHONPATH=src python scripts/inspect_manifest_sample.py "
                    f"{supervised_manifest} --output-dir outputs/manifest-preview --sample-index 0"
                ),
                ready=path_exists(supervised_manifest),
                note="Writes a JSON summary and contact sheet for camera/satellite/target sanity checks.",
            ),
            TrainingPlanStep(
                name="split_manifest",
                command=(
                    "PYTHONPATH=src python scripts/split_manifest.py "
                    f"{supervised_manifest} --train-output {train_manifest} --eval-output {eval_manifest} "
                    "--eval-fraction 0.2 --seed 0"
                ),
                ready=split_ready,
                note="Creates scene-disjoint train/eval manifests.",
            ),
            TrainingPlanStep(
                name="check_training_readiness",
                command=f"PYTHONPATH=src python scripts/check_training_readiness.py --config {config_path}",
                ready=split_ready,
                note="Verifies dependencies, device, paths, satellite config, and adapter config.",
            ),
            TrainingPlanStep(
                name="train",
                command=f"PYTHONPATH=src python scripts/train.py --config {config_path}",
                ready=split_ready,
                note="Launches the configured training run.",
            ),
            TrainingPlanStep(
                name="evaluate",
                command=f"PYTHONPATH=src python scripts/evaluate.py --config {config_path}",
                ready=split_ready,
                note="Runs evaluation on the configured eval manifest.",
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
