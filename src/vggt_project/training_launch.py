"""Combined launch readiness packet for real training runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from vggt_project.experiments import DEFAULT_EXPERIMENT_CONFIG_PATH, ExperimentRunConfig
from vggt_project.training_plan import TrainingRunPlan, build_training_run_plan
from vggt_project.training_readiness import (
    DependencyStatus,
    TrainingReadinessReport,
    check_training_readiness,
)


@dataclass(frozen=True)
class TrainingLaunchPacket:
    ready_to_launch: bool
    readiness: TrainingReadinessReport
    plan: TrainingRunPlan
    blockers: tuple[str, ...]
    next_commands: tuple[str, ...]
    remediation_commands: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(
            {
                "ready_to_launch": self.ready_to_launch,
                "readiness": json.loads(self.readiness.to_json()),
                "plan": json.loads(self.plan.to_json()),
                "blockers": list(self.blockers),
                "next_commands": list(self.next_commands),
                "remediation_commands": list(self.remediation_commands),
            },
            indent=2,
            sort_keys=True,
        )


def build_training_launch_packet(
    config: ExperimentRunConfig,
    *,
    config_path: Path = DEFAULT_EXPERIMENT_CONFIG_PATH,
    nuscenes_root: Path = Path("data/nuscenes"),
    nuscenes_version: str = "v1.0-mini",
    raw_manifest_path: Path = Path("data/manifests/nuscenes-mini.jsonl"),
    satellite_manifest_path: Path = Path("data/manifests/nuscenes-mini.satellite.jsonl"),
    smoke_manifest_path: Path = Path("data/manifests/nuscenes-mini.smoke.jsonl"),
    dependency_probe: Callable[[], tuple[DependencyStatus, ...]] | None = None,
    device_probe: Callable[[str | None], bool] | None = None,
    plan_path_exists: Callable[[Path], bool] | None = None,
) -> TrainingLaunchPacket:
    """Build one machine-readable report spanning readiness and plan state."""

    readiness = check_training_readiness(
        config,
        dependency_probe=dependency_probe,
        device_probe=device_probe,
    )
    plan = build_training_run_plan(
        config,
        config_path=config_path,
        nuscenes_root=nuscenes_root,
        nuscenes_version=nuscenes_version,
        raw_manifest_path=raw_manifest_path,
        satellite_manifest_path=satellite_manifest_path,
        smoke_manifest_path=smoke_manifest_path,
        path_exists=plan_path_exists,
    )
    blockers = _collect_blockers(readiness, plan)
    next_commands = tuple(step.command for step in plan.steps if not step.ready)[:5]
    remediation_commands = _build_remediation_commands(readiness, plan)
    return TrainingLaunchPacket(
        ready_to_launch=readiness.ready and plan.ready_to_train,
        readiness=readiness,
        plan=plan,
        blockers=blockers,
        next_commands=next_commands,
        remediation_commands=remediation_commands,
    )


def format_training_launch_packet(packet: TrainingLaunchPacket) -> str:
    """Render the launch packet as a compact human-readable checklist."""

    lines = [f"ready_to_launch: {str(packet.ready_to_launch).lower()}", "blockers:"]
    if packet.blockers:
        lines.extend(f"- {blocker}" for blocker in packet.blockers)
    else:
        lines.append("- none")
    lines.append("remediation_commands:")
    if packet.remediation_commands:
        lines.extend(f"- {command}" for command in packet.remediation_commands)
    else:
        lines.append("- none")
    lines.append("next_commands:")
    if packet.next_commands:
        lines.extend(f"- {command}" for command in packet.next_commands)
    else:
        lines.append("- none")
    return "\n".join(lines)


def _collect_blockers(
    readiness: TrainingReadinessReport,
    plan: TrainingRunPlan,
) -> tuple[str, ...]:
    blockers: list[str] = []
    blockers.extend(f"missing_path: {name}" for name in readiness.missing_paths)
    blockers.extend(f"missing_dependency: {name}" for name in readiness.missing_dependencies)
    blockers.extend(f"config_error: {error}" for error in readiness.config_errors)
    blockers.extend(f"plan_missing_output: {missing}" for missing in plan.missing_outputs)
    return tuple(dict.fromkeys(blockers))


def _build_remediation_commands(
    readiness: TrainingReadinessReport,
    plan: TrainingRunPlan,
) -> tuple[str, ...]:
    commands: list[str] = []
    if readiness.missing_dependencies:
        commands.extend(
            (
                "bash scripts/setup_env.sh .venv",
                "source .venv/bin/activate",
                "export MPLCONFIGDIR=.venv/.matplotlib",
                "PYTHONPATH=src python scripts/report_training_launch.py --json",
            )
        )
    if "satellite_raster_config_path" in readiness.missing_paths:
        commands.extend(
            (
                "bash scripts/prepare_satellite_rasters.sh",
            )
        )
    if "manifest_path" in readiness.missing_paths or "train_manifest_path" in readiness.missing_paths:
        commands.append("bash scripts/prepare_nuscenes.sh")
        commands.extend(
            step.command
            for step in plan.steps
            if not step.ready and step.name not in _LAUNCH_ONLY_STEPS
        )
    if "weights_path" in readiness.missing_paths or any("weights" in error for error in readiness.config_errors):
        commands.append("bash scripts/prepare_model_weights.sh")
    return tuple(dict.fromkeys(commands))


_LAUNCH_ONLY_STEPS = {
    "train",
    "evaluate",
    "export_occupancy_predictions",
    "evaluate_occupancy_benchmark",
}
