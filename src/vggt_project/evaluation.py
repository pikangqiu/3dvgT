"""Evaluation loop for the scaffold model."""

from __future__ import annotations

from pathlib import Path

from vggt_project.data.manifest_tensor_dataset import ManifestTensorDataset
from vggt_project.data.synthetic import SyntheticSpec, make_synthetic_dataset, tensor_tuple_to_batch
from vggt_project.losses import detach_float_metrics, reconstruction_losses
from vggt_project.metrics import reconstruction_metrics
from vggt_project.models.scaffold import SatelliteBEVG3TScaffold


def evaluate_synthetic(checkpoint: Path, batch_size: int = 4, device: str | None = None) -> dict[str, float]:
    """Evaluate the scaffold checkpoint on synthetic data."""

    import torch
    from torch.utils.data import DataLoader

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    spec = SyntheticSpec()
    model = SatelliteBEVG3TScaffold.build(
        bev_channels=spec.bev_channels,
        satellite_channels=spec.satellite_channels,
        point_count=spec.point_count,
    ).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    accum: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for tensors in DataLoader(make_synthetic_dataset(spec), batch_size=batch_size):
            batch = {
                key: value.to(device)
                for key, value in tensor_tuple_to_batch(tensors).items()
            }
            prediction = model(batch)
            metrics = detach_float_metrics(reconstruction_losses(prediction, batch))
            metrics.update(reconstruction_metrics(prediction, batch))
            for key, value in metrics.items():
                accum[key] = accum.get(key, 0.0) + value
            count += 1

    return {key: value / max(count, 1) for key, value in accum.items()}


def evaluate_manifest_smoke(
    checkpoint: Path,
    manifest_path: Path,
    batch_size: int = 1,
    image_size: int = 32,
    point_count: int = 128,
    device: str | None = None,
) -> dict[str, float]:
    """Evaluate a scaffold checkpoint on real files listed in a manifest."""

    import torch
    from torch.utils.data import DataLoader

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ManifestTensorDataset(
        manifest_path=manifest_path,
        image_size=image_size,
        point_count=point_count,
    )
    model = SatelliteBEVG3TScaffold.build(point_count=point_count).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    accum: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size):
            batch = {
                key: value.to(device)
                for key, value in batch.items()
                if hasattr(value, "to")
            }
            prediction = model(batch)
            metrics = detach_float_metrics(reconstruction_losses(prediction, batch))
            metrics.update(reconstruction_metrics(prediction, batch))
            for key, value in metrics.items():
                accum[key] = accum.get(key, 0.0) + value
            count += 1

    return {key: value / max(count, 1) for key, value in accum.items()}
