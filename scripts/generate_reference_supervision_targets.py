#!/usr/bin/env python3
"""Generate dense reference-model supervision targets for a project manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt_project.experiments import DEFAULT_EXPERIMENT_CONFIG_PATH


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG_PATH)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--target-dir", type=Path, default=Path("reference_targets"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--point-count", type=int, default=None)
    parser.add_argument("--max-points", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    from vggt_project.data.reference_supervision import materialize_reference_prediction_manifest
    from vggt_project.data.manifest_tensor_dataset import ManifestTensorDataset
    from vggt_project.experiments import load_experiment_config
    from vggt_project.models.factory import ModelBuildConfig, build_reconstruction_model

    import torch

    config = load_experiment_config(args.config)
    manifest_path = args.manifest or config.train_manifest_path or config.manifest_path
    if manifest_path is None:
        raise ValueError("--manifest or runtime.data.train_manifest_path/runtime.data.manifest_path is required")

    image_size = args.image_size or config.image_size
    point_count = args.point_count or config.point_count
    output_path = args.output or manifest_path.with_suffix(".reference.jsonl")
    device = args.device or config.device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ManifestTensorDataset(manifest_path, image_size=image_size, point_count=point_count)
    model = build_reconstruction_model(
        ModelBuildConfig(
            family=config.model_family,
            adapter_module_path=config.adapter_module_path,
            weights_path=config.weights_path,
            strict_weights=config.strict_weights,
            freeze_backbone=config.freeze_backbone,
            use_reference_adapter=config.use_reference_adapter,
            reference_root=config.reference_root,
            reference_model=config.reference_model,
            reference_model_kwargs=config.reference_model_kwargs,
            point_count=point_count,
        )
    ).to(device)
    model.eval()

    def predict_record(record: dict, index: int) -> dict:
        del record
        batch = {
            key: value.unsqueeze(0).to(device)
            for key, value in dataset[index].items()
            if hasattr(value, "unsqueeze")
        }
        with torch.no_grad():
            prediction = model(batch)
        return _remove_batch_dimension(prediction)

    report = materialize_reference_prediction_manifest(
        manifest_path,
        prediction_fn=predict_record,
        target_dir=args.target_dir,
        output_manifest_path=output_path,
        max_points=args.max_points,
        overwrite=args.overwrite,
    )

    print(f"manifest: {report.manifest_path}")
    if report.output_manifest_path is not None:
        print(f"output_manifest: {report.output_manifest_path}")
    print(f"samples: {report.sample_count}")
    print(f"depth_maps_written: {report.depth_maps_written}")
    print(f"pointmaps_written: {report.pointmaps_written}")
    print(f"pose_targets_written: {report.pose_targets_written}")
    return 0


def _remove_batch_dimension(prediction: dict) -> dict:
    squeezed = {}
    for key, value in prediction.items():
        if hasattr(value, "detach") and getattr(value, "ndim", 0) > 0 and value.shape[0] == 1:
            squeezed[key] = value[0]
        else:
            squeezed[key] = value
    return squeezed


if __name__ == "__main__":
    raise SystemExit(main())
