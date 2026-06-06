"""Runtime contract checks for scaffold and external reconstruction adapters."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from vggt_project.models.factory import (
    ModelBuildConfig,
    build_reconstruction_model,
    validate_reconstruction_prediction,
)


CAMERA_AWARE_KEYS = (
    "camera_depths",
    "camera_pointmaps",
    "camera_local_camera_to_gravity_poses",
)


@dataclass(frozen=True)
class ModelAdapterContractReport:
    model_family: str
    contract_ready: bool
    template_adapter: bool
    prediction_keys: tuple[str, ...]
    total_parameter_tensors: int
    trainable_parameter_tensors: int
    frozen_parameter_tensors: int
    trainable_parameter_names: tuple[str, ...]
    frozen_parameter_names: tuple[str, ...]
    adapter_status: dict[str, str]
    errors: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def probe_model_adapter_contract(
    config: ModelBuildConfig,
    *,
    batch_size: int = 2,
    camera_count: int = 3,
    image_size: int = 16,
) -> ModelAdapterContractReport:
    """Build a model, run one camera-aware forward pass, and validate outputs."""

    try:
        import torch

        model = build_reconstruction_model(config)
        model.eval()
        batch = {
            "bev_features": torch.zeros(batch_size, config.bev_channels, image_size, image_size),
            "satellite_patch": torch.zeros(batch_size, config.satellite_channels, image_size, image_size),
            "camera_images": torch.zeros(batch_size, camera_count, 3, image_size, image_size),
        }
        with torch.no_grad():
            prediction = model(batch)
        errors = list(_prediction_contract_errors(prediction, batch_size, camera_count, config.point_count))
        adapter_status = _adapter_status(model)
        template_adapter = adapter_status.get("status") == "template"
        parameter_summary = _parameter_summary(model)
        return ModelAdapterContractReport(
            model_family=config.family,
            contract_ready=not errors,
            template_adapter=template_adapter,
            prediction_keys=tuple(sorted(prediction.keys())),
            total_parameter_tensors=parameter_summary["total_parameter_tensors"],
            trainable_parameter_tensors=parameter_summary["trainable_parameter_tensors"],
            frozen_parameter_tensors=parameter_summary["frozen_parameter_tensors"],
            trainable_parameter_names=parameter_summary["trainable_parameter_names"],
            frozen_parameter_names=parameter_summary["frozen_parameter_names"],
            adapter_status=adapter_status,
            errors=tuple(errors),
        )
    except Exception as error:
        return ModelAdapterContractReport(
            model_family=config.family,
            contract_ready=False,
            template_adapter=False,
            prediction_keys=(),
            total_parameter_tensors=0,
            trainable_parameter_tensors=0,
            frozen_parameter_tensors=0,
            trainable_parameter_names=(),
            frozen_parameter_names=(),
            adapter_status={},
            errors=(str(error),),
        )


def format_model_adapter_contract_report(report: ModelAdapterContractReport) -> str:
    """Render a compact adapter contract report."""

    lines = [
        f"contract_ready: {str(report.contract_ready).lower()}",
        f"model_family: {report.model_family}",
        f"template_adapter: {str(report.template_adapter).lower()}",
        "prediction_keys:",
    ]
    lines.extend(f"- {key}" for key in report.prediction_keys)
    lines.extend(
        [
            "parameters:",
            f"- total_tensors: {report.total_parameter_tensors}",
            f"- trainable_tensors: {report.trainable_parameter_tensors}",
            f"- frozen_tensors: {report.frozen_parameter_tensors}",
            "trainable_parameter_names:",
        ]
    )
    if report.trainable_parameter_names:
        lines.extend(f"- {name}" for name in report.trainable_parameter_names)
    else:
        lines.append("- none")
    lines.append("frozen_parameter_names:")
    if report.frozen_parameter_names:
        lines.extend(f"- {name}" for name in report.frozen_parameter_names)
    else:
        lines.append("- none")
    lines.append("adapter_status:")
    if report.adapter_status:
        lines.extend(f"- {key}: {value}" for key, value in sorted(report.adapter_status.items()))
    else:
        lines.append("- none")
    lines.append("errors:")
    if report.errors:
        lines.extend(f"- {error}" for error in report.errors)
    else:
        lines.append("- none")
    return "\n".join(lines)


def _prediction_contract_errors(
    prediction: dict,
    batch_size: int,
    camera_count: int,
    point_count: int,
) -> tuple[str, ...]:
    errors: list[str] = []
    try:
        validate_reconstruction_prediction(prediction)
    except ValueError as error:
        errors.append(str(error))

    for key in CAMERA_AWARE_KEYS:
        if key not in prediction:
            errors.append(f"prediction missing camera-aware key: {key}")

    _append_shape_error(errors, prediction, "gravity_aligned_pointmap", (batch_size, point_count, 3))
    _append_shape_prefix_error(errors, prediction, "depth", (batch_size, 1))
    _append_shape_prefix_error(errors, prediction, "camera_depths", (batch_size, camera_count, 1))
    _append_shape_error(errors, prediction, "camera_pointmaps", (batch_size, camera_count, point_count, 3))
    _append_shape_error(errors, prediction, "local_camera_to_gravity_pose", (batch_size, 4))
    _append_shape_error(
        errors,
        prediction,
        "camera_local_camera_to_gravity_poses",
        (batch_size, camera_count, 4),
    )
    _append_shape_error(errors, prediction, "relative_yaw_translation", (batch_size, 4))
    return tuple(errors)


def _append_shape_error(errors: list[str], prediction: dict, key: str, expected: tuple[int, ...]) -> None:
    value = prediction.get(key)
    if value is None:
        return
    shape = tuple(getattr(value, "shape", ()))
    if shape != expected:
        errors.append(f"{key} expected shape {expected}, got {shape}")


def _append_shape_prefix_error(errors: list[str], prediction: dict, key: str, expected_prefix: tuple[int, ...]) -> None:
    value = prediction.get(key)
    if value is None:
        return
    shape = tuple(getattr(value, "shape", ()))
    if shape[: len(expected_prefix)] != expected_prefix:
        errors.append(f"{key} expected shape prefix {expected_prefix}, got {shape}")


def _adapter_status(model: Any) -> dict[str, str]:
    status = getattr(model, "adapter_status", None)
    if isinstance(status, dict):
        return {str(key): str(value) for key, value in status.items()}
    return {}


def _parameter_summary(model: Any) -> dict[str, Any]:
    trainable: list[str] = []
    frozen: list[str] = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            trainable.append(str(name))
        else:
            frozen.append(str(name))
    return {
        "total_parameter_tensors": len(trainable) + len(frozen),
        "trainable_parameter_tensors": len(trainable),
        "frozen_parameter_tensors": len(frozen),
        "trainable_parameter_names": tuple(trainable),
        "frozen_parameter_names": tuple(frozen),
    }
