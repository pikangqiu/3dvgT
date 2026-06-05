#!/usr/bin/env python3
"""Evaluate entrypoint for the current scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt_project.evaluation import evaluate_synthetic


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic"], default="synthetic")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/synthetic/synthetic_scaffold.pt"))
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    metrics = evaluate_synthetic(checkpoint=args.checkpoint, batch_size=args.batch_size)
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

