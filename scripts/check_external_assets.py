#!/usr/bin/env python3
"""Check external assets needed before a real training launch."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from vggt_project.checkpoint_inspection import CHECKPOINT_SUFFIXES
from vggt_project.data.nuscenes_adapter import NuScenesAdapterConfig, inspect_nuscenes_root
from vggt_project.experiments import DEFAULT_EXPERIMENT_CONFIG_PATH, load_experiment_config


@dataclass(frozen=True)
class ExternalAssetStatus:
    name: str
    ready: bool
    required: bool
    path: str
    detail: str


@dataclass(frozen=True)
class ExternalAssetReport:
    assets: tuple[ExternalAssetStatus, ...]
    next_actions: tuple[str, ...]

    @property
    def required_ready(self) -> bool:
        return all(asset.ready for asset in self.assets if asset.required)

    def to_json(self) -> str:
        return json.dumps(
            {
                "required_ready": self.required_ready,
                "assets": [asdict(asset) for asset in self.assets],
                "next_actions": list(self.next_actions),
            },
            indent=2,
            sort_keys=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG_PATH)
    parser.add_argument("--nuscenes-root", type=Path, default=Path("data/nuscenes"))
    parser.add_argument("--nuscenes-version", default="v1.0-mini")
    parser.add_argument("--satellite-config", type=Path, default=Path("data/satellite_rasters/config.json"))
    parser.add_argument("--weights-path", type=Path, default=None)
    parser.add_argument("--occ3d-root", type=Path, default=Path("data/occ3d"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    report = build_external_asset_report(
        config_path=args.config,
        nuscenes_root=args.nuscenes_root,
        nuscenes_version=args.nuscenes_version,
        satellite_config=args.satellite_config,
        weights_path=args.weights_path,
        occ3d_root=args.occ3d_root,
    )
    payload = report.to_json()
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    if args.json:
        print(payload)
    else:
        print(format_external_asset_report(report))
    return 0 if report.required_ready else 1


def build_external_asset_report(
    *,
    config_path: Path,
    nuscenes_root: Path,
    nuscenes_version: str,
    satellite_config: Path,
    weights_path: Path | None,
    occ3d_root: Path,
) -> ExternalAssetReport:
    """Build a lightweight report for external training assets."""

    resolved_weights = weights_path if weights_path is not None else _weights_from_config(config_path)
    assets = (
        _nuscenes_status(nuscenes_root, nuscenes_version),
        _satellite_status(satellite_config),
        _weights_status(resolved_weights),
        _occ3d_status(occ3d_root),
    )
    next_actions = _next_actions(assets)
    return ExternalAssetReport(assets=assets, next_actions=next_actions)


def format_external_asset_report(report: ExternalAssetReport) -> str:
    """Render a human-readable asset report."""

    lines = [f"required_ready: {str(report.required_ready).lower()}", "assets:"]
    for asset in report.assets:
        required = "required" if asset.required else "optional"
        status = "ready" if asset.ready else "missing"
        lines.append(f"- {asset.name}: {status}; {required}; {asset.path}; {asset.detail}")
    lines.append("next_actions:")
    if report.next_actions:
        lines.extend(f"- {action}" for action in report.next_actions)
    else:
        lines.append("- none")
    return "\n".join(lines)


def _nuscenes_status(root: Path, version: str) -> ExternalAssetStatus:
    status = inspect_nuscenes_root(NuScenesAdapterConfig(root=root, version=version))
    detail = "layout ready" if status.ready else "missing " + ", ".join(status.missing)
    return ExternalAssetStatus(
        name="nuscenes",
        ready=status.ready,
        required=True,
        path=str(root),
        detail=detail,
    )


def _satellite_status(config_path: Path) -> ExternalAssetStatus:
    ready = config_path.is_file()
    detail = "config exists" if ready else "config missing"
    return ExternalAssetStatus(
        name="satellite_rasters",
        ready=ready,
        required=True,
        path=str(config_path),
        detail=detail,
    )


def _weights_status(weights_path: Path | None) -> ExternalAssetStatus:
    if weights_path is None:
        return ExternalAssetStatus(
            name="model_weights",
            ready=False,
            required=True,
            path="<unset>",
            detail="runtime.model.weights_path is unset",
        )
    if weights_path.exists() and weights_path.is_file() and weights_path.suffix.lower() in CHECKPOINT_SUFFIXES:
        return ExternalAssetStatus(
            name="model_weights",
            ready=True,
            required=True,
            path=str(weights_path),
            detail="checkpoint file exists",
        )
    if weights_path.exists() and weights_path.is_dir():
        detail = "weights_path is a directory; choose one concrete checkpoint file"
    elif weights_path.suffix.lower() not in CHECKPOINT_SUFFIXES:
        detail = "weights_path must end with .pt, .pth, or .bin"
    else:
        detail = "checkpoint file missing"
    return ExternalAssetStatus(
        name="model_weights",
        ready=False,
        required=True,
        path=str(weights_path),
        detail=detail,
    )


def _occ3d_status(root: Path) -> ExternalAssetStatus:
    label_root = root / "occ3d-nuscenes"
    ready = (label_root / "gts").exists() and (label_root / "infos").exists()
    detail = "benchmark labels ready" if ready else "optional benchmark labels missing"
    return ExternalAssetStatus(
        name="occ3d",
        ready=ready,
        required=False,
        path=str(root),
        detail=detail,
    )


def _weights_from_config(config_path: Path) -> Path | None:
    try:
        config = load_experiment_config(config_path)
    except Exception:
        return None
    return config.weights_path


def _next_actions(assets: tuple[ExternalAssetStatus, ...]) -> tuple[str, ...]:
    actions: list[str] = []
    by_name = {asset.name: asset for asset in assets}
    if not by_name["nuscenes"].ready:
        actions.append("bash scripts/prepare_nuscenes.sh")
    if not by_name["satellite_rasters"].ready:
        actions.append("bash scripts/prepare_satellite_rasters.sh")
    if not by_name["model_weights"].ready:
        actions.append("bash scripts/prepare_model_weights.sh")
        actions.append(
            "PYTHONPATH=src python scripts/configure_model_weights.py --config configs/reconstruction_first.json --weights-path <checkpoint>"
        )
    if not by_name["occ3d"].ready:
        actions.append("bash scripts/prepare_occ3d.sh")
    return tuple(actions)


if __name__ == "__main__":
    raise SystemExit(main())
