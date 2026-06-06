#!/usr/bin/env python3
"""Check whether the current environment/config is ready for training."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt_project.experiments import DEFAULT_EXPERIMENT_CONFIG_PATH, load_experiment_config
from vggt_project.training_readiness import (
    check_training_readiness,
    format_training_readiness_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG_PATH)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        config = load_experiment_config(args.config)
    except RuntimeError as error:
        print(f"ready: false")
        print(f"config_error: {error}")
        return 1

    report = check_training_readiness(config)
    if args.json:
        print(report.to_json())
    else:
        print(format_training_readiness_report(report))
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
