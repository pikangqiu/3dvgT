#!/usr/bin/env python3
"""Validate occupancy label class ids before benchmark evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt_project.occupancy_label_validation import (
    format_occupancy_label_validation_report,
    validate_occupancy_label_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-field", default="occupancy_path")
    parser.add_argument("--num-classes", type=int, default=18)
    parser.add_argument("--ignore-index", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = validate_occupancy_label_manifest(
        args.manifest,
        target_field=args.target_field,
        num_classes=args.num_classes,
        ignore_index=args.ignore_index,
    )
    if args.json:
        print(report.to_json())
    else:
        print(format_occupancy_label_validation_report(report))
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
