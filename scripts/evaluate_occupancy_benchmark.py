#!/usr/bin/env python3
"""Evaluate semantic occupancy predictions referenced by a manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt_project.occupancy_benchmark import (
    evaluate_occupancy_manifest,
    format_occupancy_benchmark_report,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prediction-field", default="predicted_occupancy_path")
    parser.add_argument("--target-field", default="occupancy_path")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--ignore-index", type=int, default=None)
    parser.add_argument("--binary-threshold", type=float, default=0.5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        report = evaluate_occupancy_manifest(
            args.manifest,
            prediction_field=args.prediction_field,
            target_field=args.target_field,
            num_classes=args.num_classes,
            ignore_index=args.ignore_index,
            binary_threshold=args.binary_threshold,
        )
    except Exception as error:
        if args.json:
            import json

            print(
                json.dumps(
                    {
                        "manifest_path": str(args.manifest),
                        "sample_count": 0,
                        "occupancy_miou": 0.0,
                        "class_iou": {},
                        "errors": [str(error)],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        else:
            print("occupancy_benchmark_ready: false")
            print(f"error: {error}")
        return 1

    if args.json:
        print(report.to_json())
    else:
        print(format_occupancy_benchmark_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
