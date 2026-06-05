#!/usr/bin/env python3
"""Run a local end-to-end train/eval smoke pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt_project.smoke_pipeline import run_manifest_smoke_pipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/smoke-pipeline"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--point-count", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    report = run_manifest_smoke_pipeline(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        point_count=args.point_count,
        learning_rate=args.learning_rate,
    )

    print(f"manifest: {report.manifest_path}")
    print(f"checkpoint: {report.checkpoint_path}")
    print("train_metrics:")
    for key, value in report.train_metrics.items():
        print(f"  {key}: {value}")
    print("eval_metrics:")
    for key, value in report.eval_metrics.items():
        print(f"  {key}: {value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
