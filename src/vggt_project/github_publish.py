"""GitHub publish preflight checks."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass


CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class GitHubPublishPreflightReport:
    clean_worktree: bool
    current_branch: str
    has_origin: bool
    origin_url: str | None
    gh_authenticated: bool
    next_steps: tuple[str, ...]

    @property
    def ready_to_publish(self) -> bool:
        return self.clean_worktree and self.gh_authenticated


def run_publish_preflight(
    runner: CommandRunner | None = None,
) -> GitHubPublishPreflightReport:
    """Check whether the repository is ready for `scripts/publish_github.sh`."""

    run = runner or _run_command
    status = run(["git", "status", "--porcelain"])
    branch = run(["git", "branch", "--show-current"])
    origin = run(["git", "remote", "get-url", "origin"])
    auth = run(["gh", "auth", "status"])

    clean_worktree = status.returncode == 0 and not status.stdout.strip()
    current_branch = branch.stdout.strip() if branch.returncode == 0 else ""
    has_origin = origin.returncode == 0 and bool(origin.stdout.strip())
    origin_url = origin.stdout.strip() if has_origin else None
    gh_authenticated = auth.returncode == 0

    next_steps: list[str] = []
    if not clean_worktree:
        next_steps.append("commit or stash local changes before publishing")
    if not current_branch:
        next_steps.append("create or check out a branch before publishing")
    if not gh_authenticated:
        next_steps.append("gh auth login")
    if gh_authenticated and not has_origin:
        next_steps.append("run: bash scripts/publish_github.sh VggT")
    if gh_authenticated and has_origin:
        next_steps.append(f"run: git push -u origin {current_branch}")

    return GitHubPublishPreflightReport(
        clean_worktree=clean_worktree,
        current_branch=current_branch,
        has_origin=has_origin,
        origin_url=origin_url,
        gh_authenticated=gh_authenticated,
        next_steps=tuple(next_steps),
    )


def format_publish_preflight(report: GitHubPublishPreflightReport) -> str:
    """Return a compact text report."""

    lines = [
        f"clean_worktree: {str(report.clean_worktree).lower()}",
        f"current_branch: {report.current_branch or '<none>'}",
        f"has_origin: {str(report.has_origin).lower()}",
        f"origin_url: {report.origin_url or '<none>'}",
        f"gh_authenticated: {str(report.gh_authenticated).lower()}",
        f"ready_to_publish: {str(report.ready_to_publish).lower()}",
        "next_steps:",
    ]
    for step in report.next_steps:
        lines.append(f"- {step}")
    return "\n".join(lines)


def _run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=False)
