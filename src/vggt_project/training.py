"""Training loop for the scaffold model."""

from __future__ import annotations

from pathlib import Path

from vggt_project.data.manifest_tensor_dataset import ManifestTensorDataset
from vggt_project.data.synthetic import SyntheticSpec, make_synthetic_dataset, tensor_tuple_to_batch
from vggt_project.losses import detach_float_metrics, reconstruction_losses
from vggt_project.models.factory import (
    ModelBuildConfig,
    build_reconstruction_model,
    validate_reconstruction_prediction,
)


def train_synthetic(
    output_dir: Path,
    epochs: int = 1,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    device: str | None = None,
    seed: int | None = None,
    model_family: str = "scaffold",
    adapter_module_path: Path | None = None,
    weights_path: Path | None = None,
    strict_weights: bool = True,
    freeze_backbone: bool = False,
    fine_tuning_policy: str = "full",
    use_reference_adapter: bool = False,
    reference_root: Path | None = None,
    reference_model: str = "g3t",
    reference_model_kwargs: dict | None = None,
) -> dict[str, float]:
    """Run a small synthetic training job to verify plumbing."""

    import torch
    from torch.utils.data import DataLoader

    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    generator = _seed_training(torch, seed)

    spec = SyntheticSpec()
    loader = DataLoader(
        make_synthetic_dataset(spec),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    model = build_reconstruction_model(
        ModelBuildConfig(
            family=model_family,
            adapter_module_path=adapter_module_path,
            weights_path=weights_path,
            strict_weights=strict_weights,
            freeze_backbone=freeze_backbone,
            fine_tuning_policy=fine_tuning_policy,
            use_reference_adapter=use_reference_adapter,
            reference_root=reference_root,
            reference_model=reference_model,
            reference_model_kwargs=reference_model_kwargs or {},
            bev_channels=spec.bev_channels,
            satellite_channels=spec.satellite_channels,
            point_count=spec.point_count,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(_trainable_parameters(model), lr=learning_rate)

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
            validate_reconstruction_prediction(prediction)
            losses = reconstruction_losses(prediction, batch)
            losses["loss"].backward()
            optimizer.step()
            last_metrics = detach_float_metrics(losses)

    checkpoint_path = output_dir / "synthetic_scaffold.pt"
    torch.save({"model": model.state_dict(), "metrics": last_metrics}, checkpoint_path)
    last_metrics["checkpoint"] = str(checkpoint_path)
    return last_metrics


def train_manifest_smoke(
    manifest_path: Path,
    output_dir: Path,
    epochs: int = 1,
    batch_size: int = 1,
    learning_rate: float = 1e-3,
    image_size: int = 32,
    point_count: int = 128,
    device: str | None = None,
    seed: int | None = None,
    model_family: str = "scaffold",
    adapter_module_path: Path | None = None,
    weights_path: Path | None = None,
    strict_weights: bool = True,
    freeze_backbone: bool = False,
    fine_tuning_policy: str = "full",
    use_reference_adapter: bool = False,
    reference_root: Path | None = None,
    reference_model: str = "g3t",
    reference_model_kwargs: dict | None = None,
) -> dict[str, float]:
    """Run smoke training from real image files listed in a manifest."""

    import torch
    from torch.utils.data import DataLoader

    output_dir.mkdir(parents=True, exist_ok=True)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    generator = _seed_training(torch, seed)
    dataset = ManifestTensorDataset(
        manifest_path=manifest_path,
        image_size=image_size,
        point_count=point_count,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    model = build_reconstruction_model(
        ModelBuildConfig(
            family=model_family,
            adapter_module_path=adapter_module_path,
            weights_path=weights_path,
            strict_weights=strict_weights,
            freeze_backbone=freeze_backbone,
            fine_tuning_policy=fine_tuning_policy,
            use_reference_adapter=use_reference_adapter,
            reference_root=reference_root,
            reference_model=reference_model,
            reference_model_kwargs=reference_model_kwargs or {},
            point_count=point_count,
        )
    ).to(device)
    optimizer = torch.optim.AdamW(_trainable_parameters(model), lr=learning_rate)

    last_metrics: dict[str, float] = {}
    for _epoch in range(epochs):
        model.train()
        for batch in loader:
            batch = {
                key: value.to(device)
                for key, value in batch.items()
                if hasattr(value, "to")
            }
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            validate_reconstruction_prediction(prediction)
            losses = reconstruction_losses(prediction, batch)
            losses["loss"].backward()
            optimizer.step()
            last_metrics = detach_float_metrics(losses)

    checkpoint_path = output_dir / "manifest_smoke_scaffold.pt"
    torch.save({"model": model.state_dict(), "metrics": last_metrics}, checkpoint_path)
    last_metrics["checkpoint"] = str(checkpoint_path)
    return last_metrics


def _seed_training(torch, seed: int | None):
    if seed is None:
        return None
    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _trainable_parameters(model):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("model has no trainable parameters; disable freeze_backbone or expose trainable heads")
    return parameters
