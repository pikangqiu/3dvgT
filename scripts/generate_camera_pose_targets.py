#!/usr/bin/env python3
"""Generate nuScenes camera pose target fields for a manifest."""

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
        help="Camera to write pose targets for. Repeat or pass comma-separated names.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from nuscenes.nuscenes import NuScenes
    from vggt_project.data.nuscenes_pose import materialize_camera_pose_manifest

    nusc = NuScenes(version=args.version, dataroot=str(args.root), verbose=True)
    report = materialize_camera_pose_manifest(
        nusc,
        args.manifest,
        output_manifest_path=args.output,
        camera_names=args.camera,
    )

    print(f"manifest: {report.manifest_path}")
    print(f"output_manifest: {report.output_manifest_path}")
    print(f"samples: {report.sample_count}")
    print(f"cameras: {','.join(report.camera_names)}")
    print(f"pose_targets_written: {report.pose_targets_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
