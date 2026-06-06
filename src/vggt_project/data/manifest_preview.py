"""Pre-training manifest sample preview utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vggt_project.data.manifest import load_manifest
from vggt_project.data.sample import AlignedNuScenesSample


@dataclass(frozen=True)
class ManifestSamplePreview:
    token: str
    summary_path: Path
    contact_sheet_path: Path
    summary: dict[str, Any]


def build_manifest_sample_preview(
    manifest_path: Path,
    output_dir: Path,
    *,
    sample_index: int = 0,
    tile_size: int = 160,
) -> ManifestSamplePreview:
    """Write a JSON summary and contact-sheet image for one manifest sample."""

    samples = load_manifest(manifest_path)
    if sample_index < 0 or sample_index >= len(samples):
        raise IndexError(f"sample_index {sample_index} is outside manifest size {len(samples)}")

    sample = samples[sample_index]
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_manifest_sample(sample)
    safe_token = _safe_name(sample.token)
    summary_path = output_dir / f"{safe_token}_summary.json"
    contact_sheet_path = output_dir / f"{safe_token}_preview.png"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_contact_sheet(sample, contact_sheet_path, tile_size=tile_size)
    return ManifestSamplePreview(
        token=sample.token,
        summary_path=summary_path,
        contact_sheet_path=contact_sheet_path,
        summary=summary,
    )


def summarize_manifest_sample(sample: AlignedNuScenesSample) -> dict[str, Any]:
    """Return the frame, input, and target fields that should be checked before training."""

    return {
        "token": sample.token,
        "scene_token": sample.scene_token,
        "timestamp_us": sample.timestamp_us,
        "map_location": sample.map_location,
        "camera_count": len(sample.cameras),
        "camera_names": [camera.camera_name for camera in sample.cameras],
        "camera_paths": [str(camera.image_path) for camera in sample.cameras],
        "frames": {
            "ego_pose": sample.ego_pose_frame,
            "bev": sample.bev_frame,
            "gravity": sample.gravity_frame,
            "satellite": sample.satellite_frame,
        },
        "has_satellite_patch": sample.satellite_patch_path.exists(),
        "satellite_patch_path": str(sample.satellite_patch_path),
        "has_valid_area_mask": sample.valid_area_mask_path is not None and sample.valid_area_mask_path.exists(),
        "valid_area_mask_path": str(sample.valid_area_mask_path) if sample.valid_area_mask_path else None,
        "has_sample_depth_target": sample.lidar_depth_path is not None and sample.lidar_depth_path.exists(),
        "depth_target_cameras": sorted((sample.lidar_depth_paths or {}).keys()),
        "has_sample_pointmap_target": sample.pointmap_path is not None and sample.pointmap_path.exists(),
        "pointmap_target_cameras": sorted((sample.pointmap_paths or {}).keys()),
        "ego_translation": sample.ego_translation,
        "ego_rotation": sample.ego_rotation,
    }


def _write_contact_sheet(sample: AlignedNuScenesSample, path: Path, *, tile_size: int) -> None:
    from PIL import Image

    tiles = [_load_rgb_tile(camera.image_path, tile_size) for camera in sample.cameras]
    tiles.append(_load_rgb_tile(sample.satellite_patch_path, tile_size))
    if sample.valid_area_mask_path is not None:
        tiles.append(_load_gray_tile(sample.valid_area_mask_path, tile_size))
    if sample.lidar_depth_path is not None:
        tiles.append(_load_gray_tile(sample.lidar_depth_path, tile_size))
    for camera in sample.cameras:
        if sample.lidar_depth_paths and camera.camera_name in sample.lidar_depth_paths:
            tiles.append(_load_gray_tile(sample.lidar_depth_paths[camera.camera_name], tile_size))

    columns = min(3, max(1, len(tiles)))
    rows = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile_size, rows * tile_size), color=(0, 0, 0))
    for index, tile in enumerate(tiles):
        x = (index % columns) * tile_size
        y = (index // columns) * tile_size
        sheet.paste(tile, (x, y))
    sheet.save(path)


def _load_rgb_tile(path: Path, tile_size: int):
    from PIL import Image

    return Image.open(path).convert("RGB").resize((tile_size, tile_size))


def _load_gray_tile(path: Path, tile_size: int):
    from PIL import Image, ImageOps

    image = Image.open(path).convert("L").resize((tile_size, tile_size))
    return ImageOps.colorize(image, black=(0, 0, 0), white=(255, 255, 255))


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in ("-", "_") else "_" for character in value)
