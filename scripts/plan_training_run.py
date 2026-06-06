#!/usr/bin/env python3
"""Print the ordered commands needed to prepare and launch a training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vggt_project.experiments import load_experiment_config
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
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        config = load_experiment_config(args.config)
    except RuntimeError as error:
        if args.json:
            print(
                json.dumps(
                    {
                        "config_error": str(error),
                        "missing_outputs": [],
                        "ready_to_train": False,
                        "steps": [],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        print("ready_to_train: false")
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
    if args.json:
        print(plan.to_json())
    else:
        print(format_training_run_plan(plan))
    return 0 if plan.ready_to_train else 1


if __name__ == "__main__":
    raise SystemExit(main())
