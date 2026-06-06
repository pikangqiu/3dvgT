import json
import unittest
from pathlib import Path

from vggt_project.experiments import ExperimentRunConfig


class TrainingPlanTest(unittest.TestCase):
    def test_plan_lists_preprocessing_commands_for_missing_manifest_split(self) -> None:
        from vggt_project.training_plan import build_training_run_plan, format_training_run_plan

        config = ExperimentRunConfig(
            training_mode="manifest-smoke",
            manifest_path=Path("data/manifests/nuscenes-mini.supervised.jsonl"),
            train_manifest_path=Path("data/manifests/nuscenes-mini.train.jsonl"),
            eval_manifest_path=Path("data/manifests/nuscenes-mini.val.jsonl"),
            satellite_raster_config_path=Path("data/satellite_rasters/config.json"),
            image_size=32,
            point_count=128,
        )

        plan = build_training_run_plan(config)
        rendered = format_training_run_plan(plan)

        self.assertFalse(plan.ready_to_train)
        self.assertIn("scripts/generate_manifest.py", rendered)
        self.assertIn("scripts/materialize_satellite_crops.py", rendered)
        self.assertIn("scripts/generate_camera_pose_targets.py", rendered)
        self.assertIn("--pointmap-target-frame camera", rendered)
        self.assertIn("scripts/generate_lidar_occupancy_targets.py", rendered)
        self.assertIn("scripts/generate_reference_supervision_targets.py", rendered)
        self.assertIn("scripts/inspect_manifest_sample.py", rendered)
        self.assertIn("scripts/check_model_adapter.py", rendered)
        self.assertIn("scripts/probe_manifest_forward.py", rendered)
        self.assertIn("scripts/split_manifest.py", rendered)
        self.assertIn("scripts/export_occupancy_predictions.py", rendered)
        self.assertIn("scripts/evaluate_occupancy_benchmark.py", rendered)
        self.assertIn("data/manifests/nuscenes-mini.train.jsonl", plan.missing_outputs)
        self.assertIn("data/manifests/nuscenes-mini.val.jsonl", plan.missing_outputs)
        self.assertIn("scripts/train.py --config configs/reconstruction_first.json", rendered)

    def test_plan_is_ready_when_required_outputs_exist(self) -> None:
        from vggt_project.training_plan import build_training_run_plan

        config = ExperimentRunConfig(
            training_mode="manifest-smoke",
            manifest_path=Path("ready.supervised.jsonl"),
            train_manifest_path=Path("ready.train.jsonl"),
            eval_manifest_path=Path("ready.val.jsonl"),
            satellite_raster_config_path=None,
        )

        plan = build_training_run_plan(
            config,
            path_exists=lambda path: path.name in {"ready.supervised.jsonl", "ready.train.jsonl", "ready.val.jsonl"},
        )

        self.assertTrue(plan.ready_to_train)
        self.assertEqual(plan.missing_outputs, ())

    def test_launch_steps_remain_pending_when_required_outputs_exist(self) -> None:
        from vggt_project.training_plan import build_training_run_plan

        config = ExperimentRunConfig(
            training_mode="manifest-smoke",
            manifest_path=Path("ready.supervised.jsonl"),
            train_manifest_path=Path("ready.train.jsonl"),
            eval_manifest_path=Path("ready.val.jsonl"),
            satellite_raster_config_path=None,
        )

        plan = build_training_run_plan(
            config,
            path_exists=lambda path: path.name in {"ready.supervised.jsonl", "ready.train.jsonl", "ready.val.jsonl"},
        )
        steps = {step.name: step for step in plan.steps}

        self.assertTrue(plan.ready_to_train)
        self.assertFalse(steps["check_training_readiness"].ready)
        self.assertFalse(steps["check_model_adapter"].ready)
        self.assertFalse(steps["probe_manifest_forward"].ready)
        self.assertFalse(steps["train"].ready)
        self.assertFalse(steps["evaluate"].ready)
        self.assertFalse(steps["export_occupancy_predictions"].ready)
        self.assertFalse(steps["evaluate_occupancy_benchmark"].ready)

    def test_plan_validates_split_manifests_before_readiness(self) -> None:
        from vggt_project.training_plan import build_training_run_plan, format_training_run_plan

        config = ExperimentRunConfig(
            training_mode="manifest-smoke",
            manifest_path=Path("ready.supervised.jsonl"),
            train_manifest_path=Path("ready.train.jsonl"),
            eval_manifest_path=Path("ready.val.jsonl"),
            satellite_raster_config_path=None,
        )

        plan = build_training_run_plan(
            config,
            path_exists=lambda path: path.name in {"ready.supervised.jsonl", "ready.train.jsonl", "ready.val.jsonl"},
        )
        rendered = format_training_run_plan(plan)
        step_names = [step.name for step in plan.steps]

        self.assertIn("validate_train_manifest", step_names)
        self.assertIn("validate_eval_manifest", step_names)
        self.assertLess(step_names.index("validate_eval_manifest"), step_names.index("check_training_readiness"))
        self.assertLess(step_names.index("check_model_adapter"), step_names.index("probe_manifest_forward"))
        self.assertLess(step_names.index("probe_manifest_forward"), step_names.index("train"))
        self.assertLess(step_names.index("evaluate"), step_names.index("export_occupancy_predictions"))
        self.assertLess(step_names.index("export_occupancy_predictions"), step_names.index("evaluate_occupancy_benchmark"))
        self.assertIn("scripts/validate_manifest.py ready.train.jsonl", rendered)
        self.assertIn("scripts/validate_manifest.py ready.val.jsonl", rendered)
        self.assertFalse(plan.steps[step_names.index("validate_train_manifest")].ready)
        self.assertFalse(plan.steps[step_names.index("validate_eval_manifest")].ready)

    def test_plan_includes_checkpoint_inspection_when_weights_path_is_configured(self) -> None:
        from vggt_project.training_plan import build_training_run_plan, format_training_run_plan

        config = ExperimentRunConfig(
            training_mode="manifest-smoke",
            manifest_path=Path("ready.supervised.jsonl"),
            train_manifest_path=Path("ready.train.jsonl"),
            eval_manifest_path=Path("ready.val.jsonl"),
            model_family="external",
            adapter_module_path=Path("adapters/g3t_vggt_adapter.py"),
            weights_path=Path("checkpoints/g3t/model.pt"),
            satellite_raster_config_path=None,
        )

        plan = build_training_run_plan(
            config,
            path_exists=lambda path: path.name in {
                "ready.supervised.jsonl",
                "ready.train.jsonl",
                "ready.val.jsonl",
                "model.pt",
            },
        )
        rendered = format_training_run_plan(plan)
        step_names = [step.name for step in plan.steps]

        self.assertIn("inspect_checkpoint", step_names)
        self.assertLess(step_names.index("inspect_checkpoint"), step_names.index("check_training_readiness"))
        self.assertIn("scripts/inspect_checkpoint.py checkpoints/g3t/model.pt", rendered)
        self.assertFalse(plan.steps[step_names.index("inspect_checkpoint")].ready)

    def test_reference_supervision_is_skipped_for_default_scaffold_plan(self) -> None:
        from vggt_project.training_plan import build_training_run_plan

        config = ExperimentRunConfig(
            training_mode="manifest-smoke",
            manifest_path=Path("ready.supervised.jsonl"),
            train_manifest_path=Path("ready.train.jsonl"),
            eval_manifest_path=Path("ready.val.jsonl"),
            model_family="scaffold",
            use_reference_adapter=False,
            satellite_raster_config_path=None,
        )

        plan = build_training_run_plan(config)
        steps = {step.name: step for step in plan.steps}

        self.assertTrue(steps["optional_generate_reference_supervision"].ready)
        self.assertIn("Skipped unless", steps["optional_generate_reference_supervision"].note)

    def test_reference_supervision_runs_when_reference_adapter_is_enabled(self) -> None:
        from vggt_project.training_plan import build_training_run_plan

        config = ExperimentRunConfig(
            training_mode="manifest-smoke",
            manifest_path=Path("ready.supervised.jsonl"),
            train_manifest_path=Path("ready.train.jsonl"),
            eval_manifest_path=Path("ready.val.jsonl"),
            model_family="g3t-vggt",
            adapter_module_path=Path("adapters/g3t_vggt_adapter.py"),
            use_reference_adapter=True,
            reference_root=Path("refs/g3t"),
            satellite_raster_config_path=None,
        )

        plan = build_training_run_plan(config)
        steps = {step.name: step for step in plan.steps}

        self.assertFalse(steps["optional_generate_reference_supervision"].ready)
        self.assertIn("dense configured G3T/VGGT", steps["optional_generate_reference_supervision"].note)

    def test_plan_json_serializes_steps_and_paths(self) -> None:
        from vggt_project.training_plan import build_training_run_plan

        config = ExperimentRunConfig(
            training_mode="manifest-smoke",
            manifest_path=Path("data/manifests/nuscenes-mini.supervised.jsonl"),
            train_manifest_path=Path("data/manifests/nuscenes-mini.train.jsonl"),
            eval_manifest_path=Path("data/manifests/nuscenes-mini.val.jsonl"),
            satellite_raster_config_path=Path("data/satellite_rasters/config.json"),
            image_size=32,
            point_count=128,
        )

        plan = build_training_run_plan(config)
        payload = json.loads(plan.to_json())

        self.assertFalse(payload["ready_to_train"])
        self.assertIn("data/manifests/nuscenes-mini.train.jsonl", payload["missing_outputs"])
        self.assertEqual(payload["steps"][0]["name"], "check_nuscenes_layout")
        self.assertEqual(payload["steps"][1]["output_path"], "data/manifests/nuscenes-mini.jsonl")
        self.assertIsInstance(payload["steps"][1]["ready"], bool)


if __name__ == "__main__":
    unittest.main()
