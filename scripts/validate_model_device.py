#!/usr/bin/env python3
"""Validate configured model, optional weights, and requested device."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt_project.experiments import DEFAULT_EXPERIMENT_CONFIG_PATH, load_experiment_config
from vggt_project.model_device_validation import (
    format_model_device_validation_report,
    validate_model_device,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG_PATH)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--camera-count", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--require-weights", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        config = load_experiment_config(args.config)
    except RuntimeError as error:
        print("ready: false")
        print(f"config_error: {error}")
        return 1

    report = validate_model_device(
        config,
        device=args.device,
        require_weights=args.require_weights,
        batch_size=args.batch_size,
        camera_count=args.camera_count,
        image_size=args.image_size,
    )
    payload = report.to_json()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if args.json:
        print(payload)
    else:
        print(format_model_device_validation_report(report))
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
