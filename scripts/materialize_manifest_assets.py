#!/usr/bin/env python3
"""Create smoke-test assets referenced by a project manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt_project.data.manifest_assets import materialize_manifest_assets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--skip-satellite-placeholders", action="store_true")
    parser.add_argument("--create-valid-masks", action="store_true")
    parser.add_argument("--valid-mask-dir", type=Path, default=Path("valid_masks"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = materialize_manifest_assets(
        args.manifest,
        patch_size=args.patch_size,
        create_satellite_placeholders=not args.skip_satellite_placeholders,
        create_valid_masks=args.create_valid_masks,
        valid_mask_dir=args.valid_mask_dir,
        output_manifest_path=args.output,
        overwrite=args.overwrite,
    )

    print(f"manifest: {report.manifest_path}")
    if report.output_manifest_path is not None:
        print(f"output_manifest: {report.output_manifest_path}")
    print(f"samples: {report.sample_count}")
    print(f"satellite_placeholders_written: {report.satellite_placeholders_written}")
    print(f"valid_masks_written: {report.valid_masks_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
