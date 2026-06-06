#!/usr/bin/env python3
"""Write model weight settings into a JSON experiment config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vggt_project.checkpoint_inspection import CHECKPOINT_SUFFIXES
from vggt_project.experiments import DEFAULT_EXPERIMENT_CONFIG_PATH


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--weights-path", type=Path, required=True)
    parser.add_argument("--model-family", default=None)
    parser.add_argument("--adapter-module-path", type=Path, default=None)
    parser.add_argument("--fine-tuning-policy", default=None)
    parser.add_argument("--use-reference-adapter", action="store_true")
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--reference-model", default=None)
    args = parser.parse_args()

    try:
        output_path = _resolve_output_path(args.config, args.output, args.in_place)
        updated = configure_model_weights(
            config_path=args.config,
            weights_path=args.weights_path,
            model_family=args.model_family,
            adapter_module_path=args.adapter_module_path,
            fine_tuning_policy=args.fine_tuning_policy,
            use_reference_adapter=args.use_reference_adapter,
            reference_root=args.reference_root,
            reference_model=args.reference_model,
        )
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr, flush=True)
        return 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(updated, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"config_written: {output_path}")
    print(f"runtime.model.weights_path: {args.weights_path}")
    if args.model_family is not None:
        print(f"runtime.model.family: {args.model_family}")
    if args.use_reference_adapter:
        print("runtime.model.use_reference_adapter: true")
    print(f"next: PYTHONPATH=src python scripts/check_training_readiness.py --config {output_path}")
    print(f"next: PYTHONPATH=src python scripts/check_model_adapter.py --config {output_path}")
    return 0


def configure_model_weights(
    *,
    config_path: Path,
    weights_path: Path,
    model_family: str | None = None,
    adapter_module_path: Path | None = None,
    fine_tuning_policy: str | None = None,
    use_reference_adapter: bool = False,
    reference_root: Path | None = None,
    reference_model: str | None = None,
) -> dict:
    """Return a config mapping with runtime.model weight settings updated."""

    _validate_weights_path(weights_path)
    if config_path.suffix.lower() != ".json":
        raise ValueError("configure_model_weights currently writes JSON configs only")
    raw = json.loads(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a JSON object: {config_path}")

    runtime = raw.setdefault("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError("runtime must be a JSON object")
    model = runtime.setdefault("model", {})
    if not isinstance(model, dict):
        raise ValueError("runtime.model must be a JSON object")

    model["weights_path"] = str(weights_path)
    if model_family is not None:
        model["family"] = model_family
    if adapter_module_path is not None:
        model["adapter_module_path"] = str(adapter_module_path)
    if fine_tuning_policy is not None:
        model["fine_tuning_policy"] = fine_tuning_policy
    if use_reference_adapter:
        model["use_reference_adapter"] = True
    if reference_root is not None:
        model["reference_root"] = str(reference_root)
    if reference_model is not None:
        model["reference_model"] = reference_model
    return raw


def _validate_weights_path(weights_path: Path) -> None:
    if weights_path.exists() and weights_path.is_dir():
        raise ValueError("weights_path must be a concrete .pt, .pth, or .bin file, not a directory")
    if weights_path.suffix.lower() not in CHECKPOINT_SUFFIXES:
        raise ValueError("weights_path must be a concrete .pt, .pth, or .bin file")


def _resolve_output_path(config_path: Path, output_path: Path | None, in_place: bool) -> Path:
    if in_place and output_path is not None:
        raise ValueError("--output cannot be combined with --in-place")
    if in_place:
        return config_path
    if output_path is not None:
        return output_path
    return config_path.with_suffix(".weights.json")


if __name__ == "__main__":
    raise SystemExit(main())
