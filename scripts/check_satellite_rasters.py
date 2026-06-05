#!/usr/bin/env python3
"""Validate local satellite raster config and optional manifest coverage."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt_project.data.satellite_crops import validate_satellite_raster_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    report = validate_satellite_raster_config(args.config, manifest_path=args.manifest)
    print(f"ready: {str(report.ready).lower()}")
    print(f"config: {report.config_path}")
    if report.manifest_path is not None:
        print(f"manifest: {report.manifest_path}")
        print(f"manifest_map_locations: {', '.join(report.manifest_map_locations) or '<none>'}")
    print(f"config_map_locations: {', '.join(report.map_locations) or '<none>'}")
    for location in report.missing_manifest_locations:
        print(f"missing_manifest_location: {location}")
    for path in report.missing_raster_paths:
        print(f"missing_raster_path: {path}")
    for issue in report.invalid_specs:
        print(f"invalid_spec: {issue.map_location}.{issue.field}: {issue.reason}")
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
