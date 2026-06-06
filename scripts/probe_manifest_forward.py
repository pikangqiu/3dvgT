#!/usr/bin/env python3
"""Run one manifest sample through the configured reconstruction model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vggt_project.experiments import DEFAULT_EXPERIMENT_CONFIG_PATH, load_experiment_config
from vggt_project.models.factory import (
    ModelBuildConfig,
    build_reconstruction_model,
    validate_reconstruction_prediction,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG_PATH)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--point-count", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        config = load_experiment_config(args.config)
    except RuntimeError as error:
        report = _failure_report(
            manifest_path=None,
            errors=(f"config error: {error}",),
            next_actions=(f"PYTHONPATH=src python3 scripts/report_real_training_preflight.py --config {args.config} --json",),
        )
        _print_report(report, as_json=args.json)
        return 1

    manifest_path = args.manifest or config.eval_manifest_path or config.train_manifest_path or config.manifest_path
    if manifest_path is None:
        report = _failure_report(
            manifest_path=None,
            errors=("manifest path is not configured",),
            next_actions=(f"PYTHONPATH=src python3 scripts/plan_training_run.py --config {args.config}",),
        )
        _print_report(report, as_json=args.json)
        return 1
    if not manifest_path.exists():
        report = _failure_report(
            manifest_path=manifest_path,
            errors=(f"manifest does not exist: {manifest_path}",),
            next_actions=(f"PYTHONPATH=src python3 scripts/plan_training_run.py --config {args.config}",),
        )
        _print_report(report, as_json=args.json)
        return 1

    try:
        report = _probe_forward(
            config=config,
            config_path=args.config,
            manifest_path=manifest_path,
            sample_index=args.sample_index,
            device_name=args.device,
            image_size=args.image_size,
            point_count=args.point_count,
        )
    except Exception as error:
        report = _failure_report(
            manifest_path=manifest_path,
            errors=(str(error),),
            next_actions=(
                f"PYTHONPATH=src python3 scripts/check_training_readiness.py --config {args.config}",
                f"PYTHONPATH=src python3 scripts/check_model_adapter.py --config {args.config}",
            ),
        )
        _print_report(report, as_json=args.json)
        return 1

    _print_report(report, as_json=args.json)
    return 0 if report["forward_ready"] else 1


def _probe_forward(
    *,
    config,
    config_path: Path,
    manifest_path: Path,
    sample_index: int,
    device_name: str | None,
    image_size: int | None,
    point_count: int | None,
) -> dict[str, Any]:
    from vggt_project.data.manifest import load_manifest

    records = load_manifest(manifest_path)
    if len(records) == 0:
        raise ValueError(f"manifest is empty: {manifest_path}")
    if sample_index < 0 or sample_index >= len(records):
        raise ValueError(f"sample-index {sample_index} is outside manifest length {len(records)}")

    import torch

    from vggt_project.data.manifest_tensor_dataset import ManifestTensorDataset

    resolved_image_size = image_size or config.image_size
    resolved_point_count = point_count or config.point_count
    dataset = ManifestTensorDataset(
        manifest_path,
        image_size=resolved_image_size,
        point_count=resolved_point_count,
    )

    device = torch.device(device_name or config.device or _default_device(torch))
    model = build_reconstruction_model(
        ModelBuildConfig(
            family=config.model_family,
            adapter_module_path=config.adapter_module_path,
            weights_path=config.weights_path,
            strict_weights=config.strict_weights,
            freeze_backbone=config.freeze_backbone,
            fine_tuning_policy=config.fine_tuning_policy,
            use_reference_adapter=config.use_reference_adapter,
            reference_root=config.reference_root,
            reference_model=config.reference_model,
            reference_model_kwargs=config.reference_model_kwargs,
            point_count=resolved_point_count,
        )
    ).to(device)
    model.eval()

    sample = dataset[sample_index]
    batch = {
        key: value.unsqueeze(0).to(device)
        for key, value in sample.items()
        if hasattr(value, "unsqueeze")
    }
    with torch.no_grad():
        prediction = model(batch)
    validate_reconstruction_prediction(prediction)

    return {
        "forward_ready": True,
        "config_path": str(config_path),
        "manifest_path": str(manifest_path),
        "sample_index": sample_index,
        "sample_token": str(sample.get("sample_token", "")),
        "dataset_length": len(dataset),
        "model_family": config.model_family,
        "device": str(device),
        "input_shapes": _shape_mapping(batch),
        "prediction_keys": sorted(str(key) for key in prediction.keys()),
        "prediction_shapes": _shape_mapping(prediction),
        "errors": [],
        "next_actions": (
            f"PYTHONPATH=src python3 scripts/train.py --config {config_path}",
            f"PYTHONPATH=src python3 scripts/evaluate.py --config {config_path}",
        ),
    }


def _failure_report(
    *,
    manifest_path: Path | None,
    errors: tuple[str, ...],
    next_actions: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "forward_ready": False,
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "sample_index": None,
        "sample_token": None,
        "dataset_length": 0,
        "model_family": None,
        "device": None,
        "input_shapes": {},
        "prediction_keys": [],
        "prediction_shapes": {},
        "errors": list(errors),
        "next_actions": list(next_actions),
    }


def _print_report(report: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print(f"forward_ready: {str(report['forward_ready']).lower()}")
    print(f"manifest_path: {report['manifest_path']}")
    print(f"sample_index: {report['sample_index']}")
    print(f"sample_token: {report['sample_token']}")
    print(f"model_family: {report['model_family']}")
    print(f"device: {report['device']}")
    print("input_shapes:")
    _print_mapping(report["input_shapes"])
    print("prediction_shapes:")
    _print_mapping(report["prediction_shapes"])
    print("errors:")
    _print_list(report["errors"])
    print("next_actions:")
    _print_list(report["next_actions"])


def _shape_mapping(values: dict[str, Any]) -> dict[str, list[int]]:
    shapes = {}
    for key, value in values.items():
        shape = getattr(value, "shape", None)
        if shape is not None:
            shapes[str(key)] = [int(dim) for dim in shape]
    return dict(sorted(shapes.items()))


def _print_mapping(values: dict[str, Any]) -> None:
    if not values:
        print("- none")
        return
    for key, value in sorted(values.items()):
        print(f"- {key}: {value}")


def _print_list(values: list[str] | tuple[str, ...]) -> None:
    if values:
        for value in values:
            print(f"- {value}")
    else:
        print("- none")


def _default_device(torch_module) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    mps = getattr(torch_module.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    raise SystemExit(main())
