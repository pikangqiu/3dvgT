"""Validate configured model, optional weights, and requested device."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

from vggt_project.experiments import ExperimentRunConfig
from vggt_project.models.factory import ModelBuildConfig, build_reconstruction_model


@dataclass(frozen=True)
class ModelDeviceValidationReport:
    ready: bool
    device: str
    requested_device: str
    weights_required: bool
    weights_path: str | None
    prediction_keys: tuple[str, ...]
    prediction_devices: dict[str, str]
    parameter_devices: tuple[str, ...]
    total_parameter_tensors: int
    trainable_parameter_tensors: int
    errors: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def validate_model_device(
    config: ExperimentRunConfig,
    *,
    device: str | None = None,
    require_weights: bool = False,
    batch_size: int = 1,
    camera_count: int = 2,
    image_size: int | None = None,
) -> ModelDeviceValidationReport:
    """Load the configured model on a device and run one forward pass."""

    requested_device = device or config.device or "cpu"
    if require_weights and config.weights_path is None:
        return _error_report(
            requested_device=requested_device,
            weights_required=require_weights,
            weights_path=None,
            errors=("runtime.model.weights_path is unset",),
        )

    try:
        import torch
    except ModuleNotFoundError as error:
        return _error_report(
            requested_device=requested_device,
            weights_required=require_weights,
            weights_path=str(config.weights_path) if config.weights_path is not None else None,
            errors=(f"torch is required for model/device validation: {error}",),
        )

    if not _device_available(torch, requested_device):
        return _error_report(
            requested_device=requested_device,
            weights_required=require_weights,
            weights_path=str(config.weights_path) if config.weights_path is not None else None,
            errors=(f"requested device is not available: {requested_device}",),
        )

    try:
        torch_device = torch.device(requested_device)
        model = build_reconstruction_model(_model_build_config(config))
        model.to(torch_device)
        model.eval()
        batch = _make_probe_batch(
            torch,
            device=torch_device,
            batch_size=batch_size,
            camera_count=camera_count,
            image_size=image_size or config.image_size,
        )
        with torch.no_grad():
            prediction = model(batch)
        prediction_devices = _prediction_devices(prediction)
        parameter_devices = tuple(sorted({str(parameter.device) for parameter in model.parameters()}))
        total_parameter_tensors = sum(1 for _parameter in model.parameters())
        trainable_parameter_tensors = sum(1 for parameter in model.parameters() if parameter.requires_grad)
        return ModelDeviceValidationReport(
            ready=True,
            device=str(torch_device),
            requested_device=requested_device,
            weights_required=require_weights,
            weights_path=str(config.weights_path) if config.weights_path is not None else None,
            prediction_keys=tuple(sorted(str(key) for key in prediction.keys())),
            prediction_devices=prediction_devices,
            parameter_devices=parameter_devices,
            total_parameter_tensors=total_parameter_tensors,
            trainable_parameter_tensors=trainable_parameter_tensors,
            errors=(),
        )
    except Exception as error:
        return _error_report(
            requested_device=requested_device,
            weights_required=require_weights,
            weights_path=str(config.weights_path) if config.weights_path is not None else None,
            errors=(str(error),),
        )


def format_model_device_validation_report(report: ModelDeviceValidationReport) -> str:
    lines = [
        f"ready: {str(report.ready).lower()}",
        f"device: {report.device}",
        f"requested_device: {report.requested_device}",
        f"weights_required: {str(report.weights_required).lower()}",
        f"weights_path: {report.weights_path or '<unset>'}",
        "prediction_keys:",
    ]
    lines.extend(f"- {key}" for key in report.prediction_keys) if report.prediction_keys else lines.append("- none")
    lines.append("prediction_devices:")
    if report.prediction_devices:
        lines.extend(f"- {key}: {value}" for key, value in sorted(report.prediction_devices.items()))
    else:
        lines.append("- none")
    lines.append("parameter_devices:")
    lines.extend(f"- {device}" for device in report.parameter_devices) if report.parameter_devices else lines.append(
        "- none"
    )
    lines.extend(
        [
            f"total_parameter_tensors: {report.total_parameter_tensors}",
            f"trainable_parameter_tensors: {report.trainable_parameter_tensors}",
            "errors:",
        ]
    )
    lines.extend(f"- {error}" for error in report.errors) if report.errors else lines.append("- none")
    return "\n".join(lines)


def _model_build_config(config: ExperimentRunConfig) -> ModelBuildConfig:
    return ModelBuildConfig(
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
        point_count=config.point_count,
    )


def _device_available(torch, requested_device: str) -> bool:
    if requested_device == "cpu":
        return True
    if requested_device.startswith("cuda"):
        return bool(torch.cuda.is_available())
    if requested_device == "mps":
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    try:
        torch.device(requested_device)
    except Exception:
        return False
    return True


def _make_probe_batch(torch, *, device, batch_size: int, camera_count: int, image_size: int) -> dict:
    return {
        "bev_features": torch.zeros(batch_size, 8, image_size, image_size, device=device),
        "satellite_patch": torch.zeros(batch_size, 3, image_size, image_size, device=device),
        "camera_images": torch.zeros(batch_size, camera_count, 3, image_size, image_size, device=device),
    }


def _prediction_devices(prediction: dict) -> dict[str, str]:
    devices: dict[str, str] = {}
    for key, value in prediction.items():
        device = getattr(value, "device", None)
        if device is not None:
            devices[str(key)] = str(device)
    return devices


def _error_report(
    *,
    requested_device: str,
    weights_required: bool,
    weights_path: str | None,
    errors: tuple[str, ...],
) -> ModelDeviceValidationReport:
    return ModelDeviceValidationReport(
        ready=False,
        device=requested_device,
        requested_device=requested_device,
        weights_required=weights_required,
        weights_path=weights_path,
        prediction_keys=(),
        prediction_devices={},
        parameter_devices=(),
        total_parameter_tensors=0,
        trainable_parameter_tensors=0,
        errors=errors,
    )
