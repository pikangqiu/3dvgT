#!/usr/bin/env python3
"""Generate nuScenes LiDAR BEV occupancy targets for a manifest."""

from __future__ import annotations

import argparse
from pathlib import Path


def _range_pair(value: str) -> tuple[float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("range values must be formatted as min,max")
    lower, upper = float(parts[0]), float(parts[1])
    if lower >= upper:
        raise argparse.ArgumentTypeError("range lower bound must be less than upper bound")
    return lower, upper


def _grid_size(value: str) -> tuple[int, int]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("grid size must be formatted as height,width")
    height, width = int(parts[0]), int(parts[1])
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError("grid size values must be positive")
    return height, width


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--occupancy-dir", type=Path, default=Path("occupancy"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--x-range", type=_range_pair, default=(-50.0, 50.0), help="BEV x range as min,max meters")
    parser.add_argument("--y-range", type=_range_pair, default=(-50.0, 50.0), help="BEV y range as min,max meters")
    parser.add_argument("--z-range", type=_range_pair, default=(-5.0, 5.0), help="LiDAR z range as min,max meters")
    parser.add_argument("--grid-size", type=_grid_size, default=(200, 200), help="BEV grid as height,width cells")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from nuscenes.nuscenes import NuScenes
    from vggt_project.data.nuscenes_occupancy import materialize_lidar_occupancy_manifest

    nusc = NuScenes(version=args.version, dataroot=str(args.root), verbose=True)
    report = materialize_lidar_occupancy_manifest(
        nusc,
        args.manifest,
        occupancy_dir=args.occupancy_dir,
        output_manifest_path=args.output,
        x_range=args.x_range,
        y_range=args.y_range,
        z_range=args.z_range,
        grid_size=args.grid_size,
        overwrite=args.overwrite,
    )

    print(f"manifest: {report.manifest_path}")
    if report.output_manifest_path is not None:
        print(f"output_manifest: {report.output_manifest_path}")
    print(f"samples: {report.sample_count}")
    print(f"occupancy_maps_written: {report.occupancy_maps_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
