#!/usr/bin/env python3
"""Crop manifest satellite patches from local raster imagery."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt_project.data.satellite_crops import materialize_satellite_crops


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patch-size-px", type=int, default=224)
    parser.add_argument("--output-dir", type=Path, default=Path("satellite"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = materialize_satellite_crops(
        manifest_path=args.manifest,
        config_path=args.config,
        output_manifest_path=args.output,
        patch_size_px=args.patch_size_px,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(f"manifest: {report.manifest_path}")
    print(f"output_manifest: {report.output_manifest_path}")
    print(f"samples: {report.sample_count}")
    print(f"crops_written: {report.crops_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
