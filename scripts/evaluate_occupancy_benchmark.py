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
    parser.add_argument("--output", type=Path, default=None)
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

            payload = json.dumps(
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
            _write_output(args.output, payload)
            print(payload)
        else:
            print("occupancy_benchmark_ready: false")
            print(f"error: {error}")
        return 1

    if args.json:
        payload = report.to_json()
    else:
        payload = format_occupancy_benchmark_report(report)
    _write_output(args.output, payload)
    print(payload)
    return 0


def _write_output(path: Path | None, text: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
