#!/usr/bin/env python3
"""Generate nuScenes LiDAR-projected depth targets for a manifest."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--root", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--camera", default="CAM_FRONT")
    parser.add_argument("--depth-dir", type=Path, default=Path("lidar_depth"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-depth-m", type=float, default=80.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from nuscenes.nuscenes import NuScenes
    from vggt_project.data.nuscenes_depth import materialize_lidar_depth_manifest

    nusc = NuScenes(version=args.version, dataroot=str(args.root), verbose=True)
    report = materialize_lidar_depth_manifest(
        nusc,
        args.manifest,
        camera_name=args.camera,
        depth_dir=args.depth_dir,
        output_manifest_path=args.output,
        max_depth_m=args.max_depth_m,
        overwrite=args.overwrite,
    )

    print(f"manifest: {report.manifest_path}")
    if report.output_manifest_path is not None:
        print(f"output_manifest: {report.output_manifest_path}")
    print(f"camera: {report.camera_name}")
    print(f"samples: {report.sample_count}")
    print(f"depth_maps_written: {report.depth_maps_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
