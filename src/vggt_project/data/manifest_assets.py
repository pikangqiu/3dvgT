"""Materialize file assets referenced by project manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ManifestAssetReport:
    manifest_path: Path
    output_manifest_path: Path | None
    sample_count: int
    satellite_placeholders_written: int
    valid_masks_written: int


def materialize_manifest_assets(
    manifest_path: Path,
    *,
    patch_size: int = 224,
    create_satellite_placeholders: bool = True,
    create_valid_masks: bool = False,
    valid_mask_dir: Path = Path("valid_masks"),
    output_manifest_path: Path | None = None,
    overwrite: bool = False,
) -> ManifestAssetReport:
    """Create smoke-test assets referenced by a manifest.

    Placeholder satellite patches are only for pipeline bring-up. Real
    experiments must replace them with geospatially aligned satellite crops.
    """

    base = manifest_path.parent
    records = _read_jsonl_records(manifest_path)
    satellite_count = 0
    mask_count = 0

    for record in records:
        if create_satellite_placeholders:
            satellite_path = _resolve(base, record["satellite_patch_path"])
            if overwrite or not satellite_path.exists():
                _write_placeholder_satellite(
                    satellite_path,
                    token=str(record["token"]),
                    patch_size=patch_size,
                )
                satellite_count += 1

        if create_valid_masks:
            if "valid_area_mask_path" not in record:
                record["valid_area_mask_path"] = str(valid_mask_dir / f"{record['token']}.png")
            mask_path = _resolve(base, record["valid_area_mask_path"])
            if overwrite or not mask_path.exists():
                _write_valid_mask(mask_path, patch_size=patch_size)
                mask_count += 1

    if output_manifest_path is not None:
        _write_jsonl_records(records, output_manifest_path)

    return ManifestAssetReport(
        manifest_path=manifest_path,
        output_manifest_path=output_manifest_path,
        sample_count=len(records),
        satellite_placeholders_written=satellite_count,
        valid_masks_written=mask_count,
    )


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


def _write_placeholder_satellite(path: Path, *, token: str, patch_size: int) -> None:
    from PIL import Image, ImageDraw

    path.parent.mkdir(parents=True, exist_ok=True)
    color = _token_color(token)
    image = Image.new("RGB", (patch_size, patch_size), color=color)
    draw = ImageDraw.Draw(image)
    grid_color = tuple(max(channel - 28, 0) for channel in color)
    step = max(patch_size // 8, 1)
    for offset in range(0, patch_size, step):
        draw.line((offset, 0, offset, patch_size), fill=grid_color)
        draw.line((0, offset, patch_size, offset), fill=grid_color)
    draw.rectangle((0, 0, patch_size - 1, patch_size - 1), outline=(255, 255, 255))
    image.save(path)


def _write_valid_mask(path: Path, *, patch_size: int) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (patch_size, patch_size), color=255).save(path)


def _token_color(token: str) -> tuple[int, int, int]:
    value = sum((index + 1) * ord(char) for index, char in enumerate(token))
    return (
        48 + value % 96,
        64 + (value // 7) % 96,
        80 + (value // 13) % 96,
    )
