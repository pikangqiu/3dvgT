#!/usr/bin/env python3
"""Inspect whether a nuScenes root has the minimum layout for this project."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt_project.data import NuScenesAdapterConfig, inspect_nuscenes_root


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--version", default="v1.0-mini")
    args = parser.parse_args()

    status = inspect_nuscenes_root(NuScenesAdapterConfig(root=args.root, version=args.version))
    print(f"root: {status.root}")
    print(f"version: {status.version}")
    print(f"expected_layout: {', '.join(status.expected_layout)}")
    if status.ready:
        print("status: ready")
        return 0

    print("status: missing")
    print(f"missing: {', '.join(status.missing)}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

