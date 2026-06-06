#!/usr/bin/env python3
"""Verify checkpoint, experiment report, and optional benchmark reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt_project.training_artifacts import (
    format_training_artifact_report,
    verify_training_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--experiment-report", type=Path, default=None)
    parser.add_argument("--train-metrics", type=Path, default=None)
    parser.add_argument("--eval-metrics", type=Path, default=None)
    parser.add_argument("--occupancy-report", type=Path, default=None)
    parser.add_argument("--required-train-metric", action="append", default=None)
    parser.add_argument("--required-eval-metric", action="append", default=None)
    parser.add_argument("--require-occupancy-report", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = verify_training_artifacts(
        checkpoint_path=args.checkpoint,
        experiment_report_path=args.experiment_report,
        train_metrics_path=args.train_metrics,
        eval_metrics_path=args.eval_metrics,
        occupancy_report_path=args.occupancy_report,
        required_train_metrics=tuple(args.required_train_metric or ("loss",)),
        required_eval_metrics=tuple(args.required_eval_metric or ("loss",)),
        require_occupancy_report=args.require_occupancy_report,
    )

    if args.json:
        print(report.to_json())
    else:
        print(format_training_artifact_report(report))
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
