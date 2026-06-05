"""Materialize aligned satellite crops from local raster images."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SatelliteCropReport:
    manifest_path: Path
    output_manifest_path: Path
    sample_count: int
    crops_written: int


def materialize_satellite_crops(
    manifest_path: Path,
    config_path: Path,
    output_manifest_path: Path,
    *,
    patch_size_px: int = 224,
    output_dir: Path = Path("satellite"),
    overwrite: bool = False,
) -> SatelliteCropReport:
    """Crop satellite patches from local rasters using manifest ego pose metadata."""

    base = manifest_path.parent
    records = _read_jsonl_records(manifest_path)
    config = _load_config(config_path)
    crops_written = 0

    for record in records:
        map_location = record.get("map_location")
        ego_translation = record.get("ego_translation")
        if map_location is None or ego_translation is None:
            raise ValueError(f"sample {record.get('token')} lacks map_location or ego_translation")
        if map_location not in config:
            raise KeyError(f"no satellite raster config for map_location {map_location}")

        crop_relative_path = output_dir / f"{record['token']}.png"
        crop_path = base / crop_relative_path
        if overwrite or not crop_path.exists():
            _write_crop(
                base=base,
                crop_path=crop_path,
                raster_spec=config[map_location],
                ego_x_m=float(ego_translation[0]),
                ego_y_m=float(ego_translation[1]),
                patch_size_px=patch_size_px,
            )
            crops_written += 1
        record["satellite_patch_path"] = str(crop_relative_path)
        record["satellite_crop_source"] = map_location

    _write_jsonl_records(records, output_manifest_path)
    return SatelliteCropReport(
        manifest_path=manifest_path,
        output_manifest_path=output_manifest_path,
        sample_count=len(records),
        crops_written=crops_written,
    )


def _write_crop(
    *,
    base: Path,
    crop_path: Path,
    raster_spec: dict,
    ego_x_m: float,
    ego_y_m: float,
    patch_size_px: int,
) -> None:
    from PIL import Image

    raster_path = _resolve(base, raster_spec["raster_path"])
    origin_ego_x, origin_ego_y = raster_spec["origin_ego_xy_m"]
    origin_px_x, origin_px_y = raster_spec["origin_pixel_xy"]
    meters_per_pixel = float(raster_spec["meters_per_pixel"])

    center_x = float(origin_px_x) + (ego_x_m - float(origin_ego_x)) / meters_per_pixel
    center_y = float(origin_px_y) + (ego_y_m - float(origin_ego_y)) / meters_per_pixel
    half = patch_size_px / 2.0
    box = (
        round(center_x - half),
        round(center_y - half),
        round(center_x + half),
        round(center_y + half),
    )

    crop_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(raster_path).convert("RGB") as image:
        image.crop(box).save(crop_path)


def _load_config(config_path: Path) -> dict:
    config_base = config_path.parent
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return {
        location: {
            **spec,
            "raster_path": str(_resolve(config_base, spec["raster_path"])),
        }
        for location, spec in raw.items()
    }


def _read_jsonl_records(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl_records(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path
