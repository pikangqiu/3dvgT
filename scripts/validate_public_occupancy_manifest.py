#!/usr/bin/env python3
"""Validate a public occupancy benchmark manifest before prediction export."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt_project.public_occupancy_manifest import (
    format_public_occupancy_manifest_report,
    validate_public_occupancy_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-field", default="occupancy_path")
    parser.add_argument("--token-field", default="token")
    parser.add_argument("--scene-field", default="scene_name")
    parser.add_argument("--expected-split", default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = validate_public_occupancy_manifest(
        args.manifest,
        target_field=args.target_field,
        token_field=args.token_field,
        scene_field=args.scene_field,
        expected_split=args.expected_split,
    )
    if args.json:
        print(report.to_json())
    else:
        print(format_public_occupancy_manifest_report(report))
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
