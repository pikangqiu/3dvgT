"""Experiment configuration and dispatch helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from vggt_project.evaluation import evaluate_manifest_smoke, evaluate_synthetic
from vggt_project.training import train_manifest_smoke, train_synthetic


@dataclass(frozen=True)
class ExperimentRunConfig:
    """Runtime knobs for scaffold training and evaluation."""

    training_mode: str = "synthetic"
    manifest_path: Path | None = None
    train_manifest_path: Path | None = None
    eval_manifest_path: Path | None = None
    satellite_raster_config_path: Path | None = None
    model_family: str = "scaffold"
    adapter_module_path: Path | None = None
    weights_path: Path | None = None
    strict_weights: bool = True
    freeze_backbone: bool = False
    fine_tuning_policy: str = "full"
    use_reference_adapter: bool = False
    reference_root: Path | None = None
    reference_model: str = "g3t"
    reference_model_kwargs: dict[str, Any] = field(default_factory=dict)
    device: str | None = None
    seed: int | None = None
    output_dir: Path = Path("outputs/synthetic")
    checkpoint: Path = Path("outputs/synthetic/synthetic_scaffold.pt")
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 1e-3
    image_size: int = 32
    point_count: int = 128

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "ExperimentRunConfig":
        runtime = _mapping(mapping.get("runtime"))
        data = _mapping(runtime.get("data"))
        model = _mapping(runtime.get("model"))
        training = _mapping(runtime.get("training"))
        evaluation = _mapping(runtime.get("evaluation"))

        mode = str(training.get("mode", "synthetic"))
        manifest = _optional_path(training.get("manifest_path", data.get("manifest_path")))
        train_manifest = _optional_path(
            training.get("manifest_path", data.get("train_manifest_path", data.get("manifest_path")))
        )
        eval_manifest = _optional_path(
            evaluation.get("manifest_path", data.get("eval_manifest_path", data.get("manifest_path")))
        )
        output_dir = _path(training.get("output_dir", _default_output_dir(mode)))
        checkpoint = _path(evaluation.get("checkpoint", _default_checkpoint(mode, output_dir)))
        batch_size = int(training.get("batch_size", evaluation.get("batch_size", 4)))

        return cls(
            training_mode=mode,
            manifest_path=manifest,
            train_manifest_path=train_manifest,
            eval_manifest_path=eval_manifest,
            satellite_raster_config_path=_optional_path(data.get("satellite_raster_config_path")),
            model_family=str(model.get("family", "scaffold")),
            adapter_module_path=_optional_path(model.get("adapter_module_path")),
            weights_path=_optional_path(model.get("weights_path")),
            strict_weights=bool(model.get("strict_weights", True)),
            freeze_backbone=bool(model.get("freeze_backbone", False)),
            fine_tuning_policy=str(
                model.get(
                    "fine_tuning_policy",
                    "frozen_backbone" if bool(model.get("freeze_backbone", False)) else "full",
                )
            ),
            use_reference_adapter=bool(model.get("use_reference_adapter", False)),
            reference_root=_optional_path(model.get("reference_root")),
            reference_model=str(model.get("reference_model", "g3t")),
            reference_model_kwargs=_mapping(model.get("reference_model_kwargs")),
            device=runtime.get("device"),
            seed=_optional_int(runtime.get("seed")),
            output_dir=output_dir,
            checkpoint=checkpoint,
            epochs=int(training.get("epochs", 1)),
            batch_size=batch_size,
            learning_rate=float(training.get("learning_rate", 1e-3)),
            image_size=int(data.get("image_size", evaluation.get("image_size", 32))),
            point_count=int(data.get("point_count", evaluation.get("point_count", 128))),
        )


def load_experiment_config(path: Path) -> ExperimentRunConfig:
    """Load runtime experiment settings from a YAML or JSON file."""

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Experiment config must be a mapping: {path}")
        return ExperimentRunConfig.from_mapping(data)

    try:
        import yaml
    except ModuleNotFoundError as error:
        raise RuntimeError("PyYAML is required for --config; install requirements.txt") from error

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Experiment config must be a mapping: {path}")
    return ExperimentRunConfig.from_mapping(data)


def train_from_config(config: ExperimentRunConfig) -> dict[str, float]:
    """Dispatch scaffold training from an experiment config."""

    if config.training_mode == "synthetic":
        return train_synthetic(
            output_dir=config.output_dir,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            device=config.device,
            seed=config.seed,
            model_family=config.model_family,
            adapter_module_path=config.adapter_module_path,
            weights_path=config.weights_path,
            strict_weights=config.strict_weights,
            freeze_backbone=config.freeze_backbone,
            fine_tuning_policy=config.fine_tuning_policy,
            use_reference_adapter=config.use_reference_adapter,
            reference_root=config.reference_root,
            reference_model=config.reference_model,
            reference_model_kwargs=config.reference_model_kwargs,
        )
    if config.training_mode == "manifest-smoke":
        manifest_path = config.train_manifest_path or config.manifest_path
        if manifest_path is None:
            raise ValueError(
                "manifest-smoke training requires runtime.data.train_manifest_path or runtime.data.manifest_path"
            )
        return train_manifest_smoke(
            manifest_path=manifest_path,
            output_dir=config.output_dir,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            image_size=config.image_size,
            point_count=config.point_count,
            device=config.device,
            seed=config.seed,
            model_family=config.model_family,
            adapter_module_path=config.adapter_module_path,
            weights_path=config.weights_path,
            strict_weights=config.strict_weights,
            freeze_backbone=config.freeze_backbone,
            fine_tuning_policy=config.fine_tuning_policy,
            use_reference_adapter=config.use_reference_adapter,
            reference_root=config.reference_root,
            reference_model=config.reference_model,
            reference_model_kwargs=config.reference_model_kwargs,
        )
    raise ValueError(f"Unsupported training mode: {config.training_mode}")


def evaluate_from_config(config: ExperimentRunConfig) -> dict[str, float]:
    """Dispatch scaffold evaluation from an experiment config."""

    if config.training_mode == "synthetic":
        return evaluate_synthetic(
            checkpoint=config.checkpoint,
            batch_size=config.batch_size,
            device=config.device,
            model_family=config.model_family,
            adapter_module_path=config.adapter_module_path,
            weights_path=config.weights_path,
            strict_weights=config.strict_weights,
            fine_tuning_policy=config.fine_tuning_policy,
            use_reference_adapter=config.use_reference_adapter,
            reference_root=config.reference_root,
            reference_model=config.reference_model,
            reference_model_kwargs=config.reference_model_kwargs,
        )
    if config.training_mode == "manifest-smoke":
        manifest_path = config.eval_manifest_path or config.manifest_path
        if manifest_path is None:
            raise ValueError(
                "manifest-smoke evaluation requires runtime.data.eval_manifest_path or runtime.data.manifest_path"
            )
        return evaluate_manifest_smoke(
            checkpoint=config.checkpoint,
            manifest_path=manifest_path,
            batch_size=config.batch_size,
            image_size=config.image_size,
            point_count=config.point_count,
            device=config.device,
            model_family=config.model_family,
            adapter_module_path=config.adapter_module_path,
            weights_path=config.weights_path,
            strict_weights=config.strict_weights,
            fine_tuning_policy=config.fine_tuning_policy,
            use_reference_adapter=config.use_reference_adapter,
            reference_root=config.reference_root,
            reference_model=config.reference_model,
            reference_model_kwargs=config.reference_model_kwargs,
        )
    raise ValueError(f"Unsupported evaluation mode: {config.training_mode}")


def run_experiment_from_config(
    config: ExperimentRunConfig,
    *,
    report_path: Path,
    train_fn: Callable[[ExperimentRunConfig], dict[str, Any]] = train_from_config,
    evaluate_fn: Callable[[ExperimentRunConfig], dict[str, Any]] = evaluate_from_config,
) -> dict[str, Any]:
    """Run train+eval from one config and persist a JSON experiment report."""

    train_metrics = train_fn(config)
    eval_metrics = evaluate_fn(config)
    report = {
        "mode": config.training_mode,
        "config": _jsonable_config(config),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _optional_path(value: Any) -> Path | None:
    return Path(value) if value else None


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _path(value: Any) -> Path:
    return Path(value)


def _default_output_dir(mode: str) -> Path:
    if mode == "manifest-smoke":
        return Path("outputs/manifest-smoke")
    return Path("outputs/synthetic")


def _default_checkpoint(mode: str, output_dir: Path) -> Path:
    if mode == "manifest-smoke":
        return output_dir / "manifest_smoke_scaffold.pt"
    return output_dir / "synthetic_scaffold.pt"


def _jsonable_config(config: ExperimentRunConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
        elif value is None:
            data[key] = None
    return data
