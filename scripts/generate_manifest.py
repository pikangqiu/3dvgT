#!/usr/bin/env python3
"""Generate a project JSONL manifest from nuScenes metadata."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt_project.data.manifest_builder import build_manifest_records, write_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--satellite-patch-dir", type=Path, default=Path("satellite"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/nuscenes-mini.jsonl"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=args.version, dataroot=str(args.root), verbose=True)
    records = build_manifest_records(nusc=nusc, satellite_patch_dir=args.satellite_patch_dir)
    if args.limit is not None:
        records = _take(records, args.limit)
    write_manifest(records, args.output)
    print(f"Wrote manifest to {args.output}")
    return 0


def _take(records, limit: int):
    for index, record in enumerate(records):
        if index >= limit:
            break
        yield record


if __name__ == "__main__":
    raise SystemExit(main())

