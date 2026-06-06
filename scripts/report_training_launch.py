#!/usr/bin/env python3
"""Report whether the configured real training run is launchable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vggt_project.experiments import DEFAULT_EXPERIMENT_CONFIG_PATH, load_experiment_config
from vggt_project.training_launch import (
    build_training_launch_packet,
    format_training_launch_packet,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG_PATH)
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
                        "ready_to_launch": False,
                        "config_error": str(error),
                        "readiness": None,
                        "plan": None,
                        "blockers": [f"config_error: {error}"],
                        "next_commands": [],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 1
        print("ready_to_launch: false")
        print(f"config_error: {error}")
        return 1

    packet = build_training_launch_packet(
        config,
        config_path=args.config,
        nuscenes_root=args.root,
        nuscenes_version=args.version,
        raw_manifest_path=args.raw_manifest,
        satellite_manifest_path=args.satellite_manifest,
        smoke_manifest_path=args.smoke_manifest,
    )
    if args.json:
        print(packet.to_json())
    else:
        print(format_training_launch_packet(packet))
    return 0 if packet.ready_to_launch else 1


if __name__ == "__main__":
    raise SystemExit(main())
