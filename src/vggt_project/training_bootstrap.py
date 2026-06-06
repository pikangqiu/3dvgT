"""Execute or dry-run the ordered real-training preparation plan."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Callable

from vggt_project.training_plan import TrainingRunPlan


@dataclass(frozen=True)
class TrainingBootstrapStepResult:
    name: str
    command: str
    status: str
    exit_code: int | None = None


@dataclass(frozen=True)
class TrainingBootstrapReport:
    executed: bool
    exit_code: int
    steps: tuple[TrainingBootstrapStepResult, ...]


def run_training_bootstrap(
    plan: TrainingRunPlan,
    *,
    execute: bool = False,
    until: str | None = None,
    include_ready: bool = False,
    command_runner: Callable[[str], int] | None = None,
) -> TrainingBootstrapReport:
    """Run plan commands, or report what would run in dry-run mode."""

    command_runner = command_runner or _run_shell_command
    results: list[TrainingBootstrapStepResult] = []

    for step in plan.steps:
        if not execute:
            results.append(
                TrainingBootstrapStepResult(
                    name=step.name,
                    command=step.command,
                    status="dry-run",
                )
            )
            if until == step.name:
                break
            continue

        if step.ready and not include_ready:
            results.append(
                TrainingBootstrapStepResult(
                    name=step.name,
                    command=step.command,
                    status="skipped-ready",
                    exit_code=0,
                )
            )
            if until == step.name:
                break
            continue

        exit_code = command_runner(step.command)
        status = "passed" if exit_code == 0 else "failed"
        results.append(
            TrainingBootstrapStepResult(
                name=step.name,
                command=step.command,
                status=status,
                exit_code=exit_code,
            )
        )
        if exit_code != 0:
            return TrainingBootstrapReport(executed=True, exit_code=exit_code, steps=tuple(results))
        if until == step.name:
            break

    return TrainingBootstrapReport(executed=execute, exit_code=0, steps=tuple(results))


def format_training_bootstrap_report(report: TrainingBootstrapReport) -> str:
    """Render bootstrap execution status for CLI use."""

    lines = [
        f"executed: {str(report.executed).lower()}",
        f"exit_code: {report.exit_code}",
        "steps:",
    ]
    for index, step in enumerate(report.steps, start=1):
        lines.append(f"{index}. {step.name}: {step.status}")
        if step.exit_code is not None:
            lines.append(f"   exit_code: {step.exit_code}")
        lines.append(f"   run: {step.command}")
    return "\n".join(lines)


def _run_shell_command(command: str) -> int:
    return subprocess.run(command, shell=True, check=False).returncode
