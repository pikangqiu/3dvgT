"""Model construction helpers for scaffold and external G3T/VGGT adapters."""

from __future__ import annotations

import importlib.util
import inspect
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from vggt_project.models.scaffold import SatelliteBEVG3TScaffold

REQUIRED_PREDICTION_KEYS = (
    "gravity_aligned_pointmap",
    "depth",
    "local_camera_to_gravity_pose",
    "relative_yaw_translation",
)


@dataclass(frozen=True)
class ModelBuildConfig:
    family: str = "scaffold"
    adapter_module_path: Path | None = None
    weights_path: Path | None = None
    strict_weights: bool = True
    freeze_backbone: bool = False
    use_reference_adapter: bool = False
    reference_root: Path | None = None
    reference_model: str = "g3t"
    bev_channels: int = 8
    satellite_channels: int = 3
    latent_dim: int = 128
    point_count: int = 128


def build_reconstruction_model(config: ModelBuildConfig):
    """Build the configured reconstruction model."""

    if config.family == "scaffold":
        return SatelliteBEVG3TScaffold.build(
            bev_channels=config.bev_channels,
            satellite_channels=config.satellite_channels,
            latent_dim=config.latent_dim,
            point_count=config.point_count,
        )
    if config.family in {"external", "g3t", "vggt", "g3t-vggt"}:
        return _build_external_adapter(config)
    raise ValueError(f"Unsupported model family: {config.family}")


def validate_reconstruction_prediction(prediction: dict) -> None:
    """Validate the output contract required by losses and metrics."""

    missing = tuple(key for key in REQUIRED_PREDICTION_KEYS if key not in prediction)
    if missing:
        raise ValueError(f"reconstruction prediction missing required keys: {', '.join(missing)}")


def _build_external_adapter(config: ModelBuildConfig):
    if config.adapter_module_path is None:
        raise ValueError("adapter_module_path is required for external/G3T/VGGT model families")
    module = _load_adapter_module(config.adapter_module_path)
    build_model = getattr(module, "build_model", None)
    if build_model is None:
        raise ValueError(f"adapter module must define build_model(...): {config.adapter_module_path}")
    model = _call_adapter_build_model(build_model, config)
    if config.weights_path is not None:
        _load_weights(model, config.weights_path, strict=config.strict_weights)
    if config.freeze_backbone:
        _freeze_backbone(model)
    return model


def _call_adapter_build_model(build_model: Any, config: ModelBuildConfig):
    kwargs = {
        "point_count": config.point_count,
        "bev_channels": config.bev_channels,
        "satellite_channels": config.satellite_channels,
        "latent_dim": config.latent_dim,
        "use_reference_adapter": config.use_reference_adapter,
        "reference_root": config.reference_root,
        "reference_model": config.reference_model,
    }
    signature = inspect.signature(build_model)
    accepts_kwargs = any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
    if accepts_kwargs:
        return build_model(**kwargs)
    accepted = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return build_model(**accepted)


def _load_adapter_module(path: Path) -> ModuleType:
    if not path.exists():
        raise FileNotFoundError(f"adapter_module_path does not exist: {path}")
    spec = importlib.util.spec_from_file_location("vggt_project_external_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load adapter module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_weights(model: Any, weights_path: Path, *, strict: bool) -> None:
    if not weights_path.exists():
        raise FileNotFoundError(f"weights_path does not exist: {weights_path}")

    import torch

    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=strict)


def _freeze_backbone(model: Any) -> None:
    freeze_hook = getattr(model, "freeze_backbone", None)
    if callable(freeze_hook):
        freeze_hook()
        return
    for parameter in model.parameters():
        parameter.requires_grad = False
