"""Training loop for the scaffold model."""

from __future__ import annotations

from pathlib import Path

from vggt_project.data.synthetic import SyntheticSpec, make_synthetic_dataset, tensor_tuple_to_batch
from vggt_project.losses import detach_float_metrics, reconstruction_losses
from vggt_project.models.scaffold import SatelliteBEVG3TScaffold


def train_synthetic(
    output_dir: Path,
    epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    device: str | None = None,
) -> dict[str, float]:
    """Run a small synthetic training job to verify plumbing."""

    import torch
    from torch.utils.data import DataLoader

    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    spec = SyntheticSpec()
    loader = DataLoader(make_synthetic_dataset(spec), batch_size=batch_size, shuffle=True)
    model = SatelliteBEVG3TScaffold.build(
        bev_channels=spec.bev_channels,
        satellite_channels=spec.satellite_channels,
        point_count=spec.point_count,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    last_metrics: dict[str, float] = {}
    for _epoch in range(epochs):
        model.train()
        for tensors in loader:
            batch = {
                key: value.to(device)
                for key, value in tensor_tuple_to_batch(tensors).items()
            }
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            losses = reconstruction_losses(prediction, batch)
            losses["loss"].backward()
            optimizer.step()
            last_metrics = detach_float_metrics(losses)

    checkpoint_path = output_dir / "synthetic_scaffold.pt"
    torch.save({"model": model.state_dict(), "metrics": last_metrics}, checkpoint_path)
    last_metrics["checkpoint"] = str(checkpoint_path)
    return last_metrics

