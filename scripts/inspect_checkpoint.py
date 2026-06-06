#!/usr/bin/env python3
"""Inspect downloaded model checkpoint key structure before adapter loading."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from vggt_project.checkpoint_inspection import (
    find_checkpoint_candidates,
    format_checkpoint_summary,
    load_checkpoint_summary,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--inspect-all", action="store_true")
    args = parser.parse_args()

    candidates = find_checkpoint_candidates(args.checkpoint)
    if args.checkpoint.is_dir() and not args.inspect_all:
        print(f"checkpoint_candidates: {len(candidates)}")
        for candidate in candidates:
            print(f"- {candidate}")
        return 0 if candidates else 1

    if not candidates:
        print(f"checkpoint_candidates: 0")
        print(f"error: no .pt/.pth/.bin checkpoint files found at {args.checkpoint}")
        return 1

    summaries = [
        (candidate, load_checkpoint_summary(candidate, sample_limit=args.sample_limit))
        for candidate in candidates
    ]
    if args.json:
        if len(summaries) == 1 and args.checkpoint.is_file():
            print(summaries[0][1].to_json())
            return 0
        print(
            json.dumps(
                [
                    {
                        "path": str(candidate),
                        "summary": asdict(summary),
                    }
                    for candidate, summary in summaries
                ],
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    for index, (candidate, summary) in enumerate(summaries):
        if index:
            print()
        print(f"checkpoint: {candidate}")
        print(format_checkpoint_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
