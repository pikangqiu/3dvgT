#!/usr/bin/env python3
"""Generate nuScenes camera-frame LiDAR pointmap targets for a manifest."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        help="Camera to project into. Repeat or pass comma-separated names for multi-camera pointmaps.",
    )
    parser.add_argument("--pointmap-dir", type=Path, default=Path("camera_pointmaps"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-points", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from nuscenes.nuscenes import NuScenes
    from vggt_project.data.nuscenes_pointmap import materialize_camera_lidar_pointmap_manifest

    nusc = NuScenes(version=args.version, dataroot=str(args.root), verbose=True)
    report = materialize_camera_lidar_pointmap_manifest(
        nusc,
        args.manifest,
        camera_names=args.camera,
        pointmap_dir=args.pointmap_dir,
        output_manifest_path=args.output,
        max_points=args.max_points,
        overwrite=args.overwrite,
    )

    print(f"manifest: {report.manifest_path}")
    if report.output_manifest_path is not None:
        print(f"output_manifest: {report.output_manifest_path}")
    print(f"samples: {report.sample_count}")
    print(f"cameras: {','.join(report.camera_names)}")
    print(f"pointmaps_written: {report.pointmaps_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
