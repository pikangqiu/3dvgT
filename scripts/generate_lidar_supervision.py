#!/usr/bin/env python3
"""Generate nuScenes LiDAR depth and pointmap targets for a manifest."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth-manifest", type=Path, default=None)
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        help="Camera to project into. Repeat or pass comma-separated names for multi-camera depth.",
    )
    parser.add_argument("--depth-dir", type=Path, default=Path("lidar_depth"))
    parser.add_argument("--pointmap-dir", type=Path, default=Path("pointmaps"))
    parser.add_argument("--max-depth-m", type=float, default=80.0)
    parser.add_argument("--max-points", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from nuscenes.nuscenes import NuScenes
    from vggt_project.data.supervision_pipeline import materialize_lidar_supervision_manifest

    nusc = NuScenes(version=args.version, dataroot=str(args.root), verbose=True)
    report = materialize_lidar_supervision_manifest(
        nusc,
        args.manifest,
        output_manifest_path=args.output,
        depth_manifest_path=args.depth_manifest,
        camera_names=args.camera,
        depth_dir=args.depth_dir,
        pointmap_dir=args.pointmap_dir,
        max_depth_m=args.max_depth_m,
        max_points=args.max_points,
        overwrite=args.overwrite,
    )

    print(f"manifest: {report.manifest_path}")
    print(f"depth_manifest: {report.depth_manifest_path}")
    print(f"output_manifest: {report.output_manifest_path}")
    print(f"samples: {report.sample_count}")
    print(f"depth_maps_written: {report.depth_maps_written}")
    print(f"pointmaps_written: {report.pointmaps_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
