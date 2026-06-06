#!/usr/bin/env python3
"""Combine external asset and launch checks for a real training run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from check_external_assets import build_external_asset_report, format_external_asset_report

from vggt_project.experiments import DEFAULT_EXPERIMENT_CONFIG_PATH, load_experiment_config
from vggt_project.training_launch import (
    build_training_launch_packet,
    format_training_launch_packet,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG_PATH)
    parser.add_argument("--nuscenes-root", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--nuscenes-version", default="v1.0-mini")
    parser.add_argument("--satellite-config", type=Path, default=Path("data/satellite_rasters/config.json"))
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument("--occ3d-root", type=Path, default=Path("data/occ3d"))
    parser.add_argument("--raw-manifest", type=Path, default=Path("data/manifests/nuscenes-mini.jsonl"))
    parser.add_argument(
        "--satellite-manifest",
        type=Path,
        default=Path("data/manifests/nuscenes-mini.satellite.jsonl"),
    )
    parser.add_argument("--smoke-manifest", type=Path, default=Path("data/manifests/nuscenes-mini.smoke.jsonl"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        config = load_experiment_config(args.config)
    except RuntimeError as error:
        payload = {
            "ready_for_real_training": False,
            "config_error": str(error),
            "external_assets": None,
            "launch": None,
            "next_actions": [],
        }
        serialized = json.dumps(payload, indent=2, sort_keys=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n", encoding="utf-8")
        if args.json:
            print(serialized)
        else:
            print("ready_for_real_training: false")
            print(f"config_error: {error}")
        return 1

    external_assets = build_external_asset_report(
        config_path=args.config,
        nuscenes_root=args.nuscenes_root,
        nuscenes_version=args.nuscenes_version,
        satellite_config=args.satellite_config,
        weights_path=args.weights_path,
        occ3d_root=args.occ3d_root,
    )
    launch = build_training_launch_packet(
        config,
        config_path=args.config,
        nuscenes_root=args.nuscenes_root,
        nuscenes_version=args.nuscenes_version,
        raw_manifest_path=args.raw_manifest,
        satellite_manifest_path=args.satellite_manifest,
        smoke_manifest_path=args.smoke_manifest,
    )
    ready = external_assets.required_ready and launch.ready_to_launch
    next_actions = _dedupe(
        tuple(external_assets.next_actions)
        + tuple(launch.remediation_commands)
        + tuple(launch.next_commands)
    )

    payload = {
        "ready_for_real_training": ready,
        "external_assets": json.loads(external_assets.to_json()),
        "launch": json.loads(launch.to_json()),
        "next_actions": list(next_actions),
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    if args.json:
        print(serialized)
    else:
        print(f"ready_for_real_training: {str(ready).lower()}")
        print("external_assets:")
        print(format_external_asset_report(external_assets))
        print("launch:")
        print(format_training_launch_packet(launch))
        print("next_actions:")
        if next_actions:
            for action in next_actions:
                print(f"- {action}")
        else:
            print("- none")
    return 0 if ready else 1


def _dedupe(commands: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(command for command in commands if command))


if __name__ == "__main__":
    raise SystemExit(main())
