#!/usr/bin/env python3
"""Verify saved preflight and artifact reports for a real training run."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt_project.real_run_evidence import (
    format_real_run_evidence_report,
    verify_real_run_evidence,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--artifact-report", type=Path, required=True)
    parser.add_argument("--environment-report", type=Path, default=None)
    parser.add_argument("--expected-git-commit", default=None)
    parser.add_argument("--require-clean-worktree", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = verify_real_run_evidence(
        preflight_report_path=args.preflight_report,
        artifact_report_path=args.artifact_report,
        environment_report_path=args.environment_report,
        expected_git_commit=args.expected_git_commit,
        require_clean_worktree=args.require_clean_worktree,
    )
    payload = report.to_json()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if args.json:
        print(payload)
    else:
        print(format_real_run_evidence_report(report))
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
