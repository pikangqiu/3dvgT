#!/usr/bin/env python3
"""Train entrypoint for the current scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt_project.training import train_manifest_smoke, train_synthetic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "manifest-smoke"], default="synthetic")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    if args.mode == "synthetic":
        metrics = train_synthetic(
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
    else:
        if args.manifest is None:
            parser.error("--manifest is required for manifest-smoke mode")
        metrics = train_manifest_smoke(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
    for key, value in metrics.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
