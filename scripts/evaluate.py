#!/usr/bin/env python3
"""Evaluate entrypoint for the current scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt_project.experiments import evaluate_from_config, load_experiment_config
from vggt_project.evaluation import evaluate_manifest_smoke, evaluate_synthetic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--mode", choices=["synthetic", "manifest-smoke"], default="synthetic")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/synthetic/synthetic_scaffold.pt"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--point-count", type=int, default=128)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.config is not None:
        metrics = evaluate_from_config(load_experiment_config(args.config))
    elif args.mode == "synthetic":
        metrics = evaluate_synthetic(
            checkpoint=args.checkpoint,
            batch_size=args.batch_size,
            device=args.device,
        )
    else:
        if args.manifest is None:
            parser.error("--manifest is required for --mode manifest-smoke")
        metrics = evaluate_manifest_smoke(
            checkpoint=args.checkpoint,
            manifest_path=args.manifest,
            batch_size=args.batch_size,
            image_size=args.image_size,
            point_count=args.point_count,
            device=args.device,
        )
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
