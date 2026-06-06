#!/usr/bin/env python3
"""Inspect downloaded model checkpoint key structure before adapter loading."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt_project.checkpoint_inspection import (
    format_checkpoint_summary,
    load_checkpoint_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = load_checkpoint_summary(args.checkpoint, sample_limit=args.sample_limit)
    if args.json:
        print(summary.to_json())
    else:
        print(format_checkpoint_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
