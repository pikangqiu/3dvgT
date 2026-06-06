#!/usr/bin/env python3
"""Dry-run or execute the ordered commands for preparing a real training run."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt_project.experiments import load_experiment_config
from vggt_project.training_bootstrap import (
    format_training_bootstrap_report,
    run_training_bootstrap,
)
from vggt_project.training_plan import build_training_run_plan, format_training_run_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/reconstruction_first.yaml"))
    parser.add_argument("--root", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--raw-manifest", type=Path, default=Path("data/manifests/nuscenes-mini.jsonl"))
    parser.add_argument(
        "--satellite-manifest",
        type=Path,
        default=Path("data/manifests/nuscenes-mini.satellite.jsonl"),
    )
    parser.add_argument("--smoke-manifest", type=Path, default=Path("data/manifests/nuscenes-mini.smoke.jsonl"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--include-ready", action="store_true")
    parser.add_argument("--until", default=None)
    parser.add_argument("--show-plan", action="store_true")
    args = parser.parse_args()

    try:
        config = load_experiment_config(args.config)
    except RuntimeError as error:
        print("executed: false")
        print("exit_code: 1")
        print(f"config_error: {error}")
        return 1

    plan = build_training_run_plan(
        config,
        config_path=args.config,
        nuscenes_root=args.root,
        nuscenes_version=args.version,
        raw_manifest_path=args.raw_manifest,
        satellite_manifest_path=args.satellite_manifest,
        smoke_manifest_path=args.smoke_manifest,
    )
    if args.show_plan:
        print(format_training_run_plan(plan))
        print()

    report = run_training_bootstrap(
        plan,
        execute=args.execute,
        until=args.until,
        include_ready=args.include_ready,
    )
    print(format_training_bootstrap_report(report))
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
