#!/usr/bin/env python3
"""Attach Occ3D/OpenOccupancy label paths to a project JSONL manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt_project.data.occ3d_labels import attach_occ3d_label_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--occ3d-root", type=Path, default=Path("data/occ3d"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-field", default="occupancy_path")
    parser.add_argument("--split", default="trainval")
    parser.add_argument("--nuscenes-root", type=Path, default=None)
    parser.add_argument("--nuscenes-version", default="v1.0-trainval")
    args = parser.parse_args()

    resolver = None
    if args.nuscenes_root is not None:
        resolver = _build_scene_name_resolver(
            root=args.nuscenes_root,
            version=args.nuscenes_version,
        )

    try:
        report = attach_occ3d_label_manifest(
            args.manifest,
            occ3d_root=args.occ3d_root,
            output_manifest_path=args.output,
            target_field=args.target_field,
            split=args.split,
            scene_name_resolver=resolver,
        )
    except Exception as error:
        print("occ3d_labels_ready: false")
        print(f"error: {error}")
        return 1

    print("occ3d_labels_ready: true")
    print(f"manifest: {report.manifest_path}")
    print(f"output_manifest: {report.output_manifest_path}")
    print(f"occ3d_root: {report.occ3d_root}")
    print(f"samples: {report.sample_count}")
    print(f"labels_attached: {report.labels_attached}")
    print(
        "next: "
        f"PYTHONPATH=src python scripts/export_occupancy_predictions.py --manifest {args.output}"
    )
    return 0


def _build_scene_name_resolver(*, root: Path, version: str):
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=version, dataroot=str(root), verbose=False)
    by_token = {scene["token"]: scene["name"] for scene in nusc.scene}

    def resolve(scene_token: str) -> str:
        try:
            return by_token[scene_token]
        except KeyError as error:
            raise KeyError(f"scene token not found in nuScenes metadata: {scene_token}") from error

    return resolve


if __name__ == "__main__":
    raise SystemExit(main())
