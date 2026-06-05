#!/usr/bin/env python3
"""Run scaffold train+eval from an experiment config and write a report."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt_project.experiments import load_experiment_config, run_experiment_from_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/reconstruction_first.yaml"))
    parser.add_argument("--report", type=Path, default=Path("outputs/experiment_report.json"))
    args = parser.parse_args()

    report = run_experiment_from_config(
        load_experiment_config(args.config),
        report_path=args.report,
    )
    print(f"report: {args.report}")
    print(f"mode: {report['mode']}")
    for key, value in report["train_metrics"].items():
        print(f"train.{key}: {value}")
    for key, value in report["eval_metrics"].items():
        print(f"eval.{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
