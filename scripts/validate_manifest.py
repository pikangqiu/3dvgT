#!/usr/bin/env python3
"""Validate paths referenced by a project JSONL manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt_project.data import validate_manifest_paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--max-missing", type=int, default=20)
    args = parser.parse_args()

    report = validate_manifest_paths(args.manifest)
    print(f"manifest: {report.manifest_path}")
    print(f"samples: {report.sample_count}")
    if report.ready:
        print("status: ready")
        return 0

    if report.sample_count == 0:
        print("status: empty")
        return 1

    print("status: missing")
    print(f"missing_count: {len(report.missing_paths)}")
    for item in report.missing_paths[: args.max_missing]:
        print(f"{item.sample_token}\t{item.field}\t{item.path}")
    if len(report.missing_paths) > args.max_missing:
        print(f"... {len(report.missing_paths) - args.max_missing} more")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
