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


@dataclass(frozen=True)
class InvalidSatelliteRasterSpec:
    map_location: str
    field: str
    reason: str


@dataclass(frozen=True)
class SatelliteRasterConfigReport:
    config_path: Path
    manifest_path: Path | None
    map_locations: tuple[str, ...]
    manifest_map_locations: tuple[str, ...]
    missing_manifest_locations: tuple[str, ...]
    missing_raster_paths: tuple[Path, ...]
    invalid_specs: tuple[InvalidSatelliteRasterSpec, ...]

    @property
    def ready(self) -> bool:
        return (
            not self.missing_manifest_locations
            and not self.missing_raster_paths
            and not self.invalid_specs
        )


def validate_satellite_raster_config(
    config_path: Path,
    *,
    manifest_path: Path | None = None,
) -> SatelliteRasterConfigReport:
    """Validate local satellite raster config paths and map-location coverage."""

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("satellite raster config must be a JSON object keyed by map_location")

    config_base = config_path.parent
    invalid_specs: list[InvalidSatelliteRasterSpec] = []
    missing_raster_paths: list[Path] = []
    for map_location, spec in raw.items():
        if not isinstance(spec, dict):
            invalid_specs.append(
                InvalidSatelliteRasterSpec(str(map_location), "<spec>", "must be a JSON object")
            )
            continue
        invalid_specs.extend(_validate_raster_spec(str(map_location), spec))
        raster_path = spec.get("raster_path")
        if isinstance(raster_path, str):
            resolved = _resolve(config_base, raster_path)
            if not resolved.exists():
                missing_raster_paths.append(resolved)

    manifest_locations = _manifest_map_locations(manifest_path) if manifest_path is not None else ()
    config_locations = tuple(str(location) for location in raw.keys())
    missing_manifest_locations = tuple(
        location for location in manifest_locations if location not in config_locations
    )

    return SatelliteRasterConfigReport(
        config_path=config_path,
        manifest_path=manifest_path,
        map_locations=config_locations,
        manifest_map_locations=manifest_locations,
        missing_manifest_locations=missing_manifest_locations,
        missing_raster_paths=tuple(missing_raster_paths),
        invalid_specs=tuple(invalid_specs),
    )


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
    validation = validate_satellite_raster_config(config_path, manifest_path=manifest_path)
    if not validation.ready:
        raise ValueError(_format_validation_error(validation))
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


def _validate_raster_spec(map_location: str, spec: dict) -> list[InvalidSatelliteRasterSpec]:
    invalid: list[InvalidSatelliteRasterSpec] = []
    if not isinstance(spec.get("raster_path"), str) or not spec.get("raster_path"):
        invalid.append(InvalidSatelliteRasterSpec(map_location, "raster_path", "must be a non-empty string"))
    _validate_pair(invalid, map_location, spec, "origin_ego_xy_m")
    _validate_pair(invalid, map_location, spec, "origin_pixel_xy")
    meters_per_pixel = spec.get("meters_per_pixel")
    try:
        meters_per_pixel_value = float(meters_per_pixel)
    except (TypeError, ValueError):
        invalid.append(InvalidSatelliteRasterSpec(map_location, "meters_per_pixel", "must be numeric"))
    else:
        if meters_per_pixel_value <= 0.0:
            invalid.append(InvalidSatelliteRasterSpec(map_location, "meters_per_pixel", "must be > 0"))
    return invalid


def _validate_pair(
    invalid: list[InvalidSatelliteRasterSpec],
    map_location: str,
    spec: dict,
    field: str,
) -> None:
    value = spec.get(field)
    if not isinstance(value, list | tuple) or len(value) != 2:
        invalid.append(InvalidSatelliteRasterSpec(map_location, field, "must be a 2-value array"))
        return
    for item in value:
        try:
            float(item)
        except (TypeError, ValueError):
            invalid.append(InvalidSatelliteRasterSpec(map_location, field, "values must be numeric"))
            return


def _manifest_map_locations(manifest_path: Path | None) -> tuple[str, ...]:
    if manifest_path is None:
        return ()
    records = _read_jsonl_records(manifest_path)
    locations = {
        str(record["map_location"])
        for record in records
        if record.get("map_location") is not None
    }
    return tuple(sorted(locations))


def _format_validation_error(report: SatelliteRasterConfigReport) -> str:
    issues: list[str] = []
    if report.missing_manifest_locations:
        issues.append(f"missing map_location configs: {', '.join(report.missing_manifest_locations)}")
    if report.missing_raster_paths:
        issues.append(
            "missing raster paths: " + ", ".join(str(path) for path in report.missing_raster_paths)
        )
    if report.invalid_specs:
        issues.extend(
            f"{issue.map_location}.{issue.field}: {issue.reason}"
            for issue in report.invalid_specs
        )
    return "satellite raster config is not ready; " + "; ".join(issues)


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
