#!/usr/bin/env python3
"""Check whether the configured model adapter satisfies the training contract."""

from __future__ import annotations

import argparse
from pathlib import Path

from vggt_project.experiments import DEFAULT_EXPERIMENT_CONFIG_PATH, load_experiment_config
from vggt_project.models.adapter_contract import (
    format_model_adapter_contract_report,
    probe_model_adapter_contract,
)
from vggt_project.models.factory import ModelBuildConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG_PATH)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--require-non-template", action="store_true")
    args = parser.parse_args()

    try:
        config = load_experiment_config(args.config)
    except RuntimeError as error:
        print("contract_ready: false")
        print(f"config_error: {error}")
        return 1

    report = probe_model_adapter_contract(
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
            point_count=config.point_count,
        )
    )
    if args.json:
        print(report.to_json())
    else:
        print(format_model_adapter_contract_report(report))

    if args.require_non_template and report.template_adapter:
        return 1
    return 0 if report.contract_ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
